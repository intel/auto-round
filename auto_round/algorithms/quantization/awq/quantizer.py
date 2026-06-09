# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AWQ (Activation-Aware Weight Quantization) quantizer.

The algorithm:
1. Collects per-channel activation magnitudes during calibration.
2. For each smooth-balance mapping, performs a grid search over scaling ratios
   to find the one that minimizes quantization error (output-based loss).
3. Applies the best channel-wise scaling: balance_layer.weight *= scales,
   smooth_layer.weight /= scales (or smooth_layer.bias /= scales if 1-D).


Reference implementations:
  - AutoAWQ: https://github.com/casper-hansen/AutoAWQ
  - llm-compressor: https://github.com/vllm-project/llm-compressor
"""

from __future__ import annotations

import contextlib
import inspect
import traceback

import torch
from tqdm import tqdm

from auto_round.algorithms.quantization.awq.config import AWQConfig
from auto_round.algorithms.quantization.awq.mappings import (
    ResolvedMapping,
    _extract_block_prefix,
    resolve_mappings,
)
from auto_round.algorithms.quantization.base import BaseQuantizers
from auto_round.compressors.shard_writer import ShardWriter
from auto_round.compressors.utils import immediate_pack
from auto_round.data_type.utils import (
    get_quant_func,
    reshape_pad_tensor_by_group_size,
    revert_tensor_by_pad,
    update_block_global_scale_if_needed,
)
from auto_round.logger import logger
from auto_round.utils import (
    check_to_quantized,
    clear_memory,
    convert_module_to_hp_if_necessary,
    get_module,
    set_amax_for_all_moe_layers,
    set_module,
)
from auto_round.devices.device_manager_haha import device_manager
from auto_round.wrapper import WrapperLinear
from auto_round.wrapper import WrapperMultiblock as _WrapperMultiblock


class AWQQuantizer(BaseQuantizers):
    """AWQ quantizer: activation-aware channel scaling + delegated quantization.

    AWQ is a pre-quantization transform that applies channel-wise scaling to reduce
    quantization error. The actual quantization is delegated to an inner
    quantizer (currently RTN by default).
    """

    def __init__(self, config: AWQConfig):
        super().__init__(config)
        self.duo_scaling = config.duo_scaling
        self.n_grid = config.n_grid
        self.apply_smooth = config.apply_smooth

        # Populated during calibration
        self._user_mappings = config.mappings
        self._resolved_mappings: list[ResolvedMapping] = []
        self._block_groups: dict[str, list[ResolvedMapping]] = {}
        self._activation_stats: dict[str, list[torch.Tensor]] = {}
        # Parent module kwargs cache: parent_module → list of kwargs dicts
        self._parent_args_cache: dict[torch.nn.Module, list[dict]] = {}
        self._parent_signatures: dict[int, inspect.BoundArguments] = {}
        self._smoothing_applied: bool = False

        # Fail fast: validate scheme at construction time
        self._check_scheme_compatibility()

    def bind(self, compressor) -> None:
        """Wire shared state and validate compressor settings for AWQ compatibility."""
        super().bind(compressor)
        # Check for unsupported compressor-level args
        nblocks = getattr(compressor, "nblocks", 1)
        if nblocks > 1:
            logger.warning(
                f"AWQ does not support nblocks > 1 (got nblocks={nblocks}). "
                f"AWQ smoothing resolves activation hooks and mappings per single block prefix. "
                f"Falling back to nblocks=1."
            )
            compressor.nblocks = 1

    def __setattr__(self, name, value):
        """Trigger model-level AWQ setup when compress_context is assigned."""
        super().__setattr__(name, value)
        if name == "compress_context" and value is not None:
            self._prepare_model()

    # ── Public API ────────────────────────────────────────────────────────────

    def resolve_all_mappings(self, model: torch.nn.Module) -> dict[str, list[ResolvedMapping]]:
        """Resolve all AWQ mappings and group by block prefix.

        Call this once before the block-by-block loop.  The resolved mappings
        are stored internally and used by ``register_activation_hooks`` and
        ``smooth_block`` for per-block filtering.

        Returns:
            Dictionary mapping block prefix (e.g. "model.layers.0") to the
            list of resolved mappings for that block.
        """
        self._resolved_mappings = resolve_mappings(model, self._user_mappings)
        block_groups: dict[str, list[ResolvedMapping]] = {}
        for m in self._resolved_mappings:
            prefix = _extract_block_prefix(m.smooth_name)
            block_groups.setdefault(prefix, []).append(m)
        self._block_groups = block_groups
        return block_groups

    def apply_smoothing(
        self,
        model: torch.nn.Module,
        device: torch.device | str | None = None,
    ) -> None:
        """Apply AWQ smoothing to the model weights.

        This is the core AWQ step: for each mapping, find the best scaling
        factor and apply it to smooth/balance layers.

        When *device* is provided the model is assumed to reside on CPU and
        only the modules needed for each mapping are temporarily moved to
        *device* during the grid search.  This dramatically reduces peak VRAM
        (~14 GB savings for an 8B model).

        Should be called AFTER calibration data has been collected and BEFORE
        RTN quantization.
        """
        if self._smoothing_applied:
            return

        # Resolve mappings
        self._resolved_mappings = resolve_mappings(model, self._user_mappings)
        if not self._resolved_mappings:
            logger.warning("No AWQ mappings resolved; skipping smoothing.")
            self._smoothing_applied = True
            return

        if not self._activation_stats:
            logger.error("No activation statistics collected for AWQ smoothing.")

        logger.info(f"Applying AWQ smoothing to {len(self._resolved_mappings)} mappings...")

        for mapping in tqdm(self._resolved_mappings, desc="AWQ Smoothing"):
            if mapping.smooth_name not in self._activation_stats:
                logger.warning(f"No activation stats for '{mapping.smooth_name}', skipping this layer.")
                continue

            act_sum, act_count = self._activation_stats.pop(mapping.smooth_name)
            if act_count == 0:
                logger.warning(f"Zero activation count for '{mapping.smooth_name}', skipping this layer.")
                continue

            # Mean activation per channel (act_count is a plain int now,
            # avoiding a CPU tensor allocation per channel).
            x_mean = (act_sum / act_count).to(torch.float32)
            del act_sum

            with self._align_modules(mapping, device):
                best_scales = self._grid_search_scales(mapping, x_mean)

            if best_scales is not None:
                self._apply_scales(mapping, best_scales)

            # Eagerly free per-mapping parent cache to reduce RAM
            parent = mapping.parent
            self._parent_args_cache.pop(parent, None)

        self._activation_stats.clear()
        self._parent_args_cache.clear()
        self._parent_signatures.clear()
        self._smoothing_applied = True
        logger.info("AWQ smoothing complete.")

    def smooth_block(self, block_prefix: str) -> None:
        """Apply AWQ smoothing for one block's mappings.

        Assumes activation stats and parent kwargs have already been collected
        for this block via ``register_activation_hooks`` + block forward.
        The block's modules are expected to be on the compute device, so
        no device alignment is needed (unlike ``apply_smoothing`` which uses
        ``_align_modules`` for CPU-offloaded models).

        Args:
            block_prefix: Block prefix (e.g. "model.layers.0") identifying
                which mappings to smooth.
        """
        block_mappings = [m for m in self._resolved_mappings if _extract_block_prefix(m.smooth_name) == block_prefix]
        for mapping in block_mappings:
            if mapping.smooth_name not in self._activation_stats:
                logger.warning(f"No activation stats for '{mapping.smooth_name}', skipping.")
                continue

            act_sum, act_count = self._activation_stats.pop(mapping.smooth_name)
            if act_count == 0:
                logger.warning(f"Zero activation count for '{mapping.smooth_name}', skipping.")
                continue

            x_mean = (act_sum / act_count).to(torch.float32)
            del act_sum

            # Block is on the compute device — grid search runs in-place
            best_scales = self._grid_search_scales(mapping, x_mean)

            if best_scales is not None:
                self._apply_scales(mapping, best_scales)

        # Free parent kwargs cache after ALL mappings for this block are done.
        # (All mappings in a block typically share the same parent module;
        # popping inside the loop would break output-based loss for
        # subsequent mappings.)
        seen_parents = set()
        for mapping in block_mappings:
            pid = id(mapping.parent)
            if pid not in seen_parents:
                seen_parents.add(pid)
                self._parent_args_cache.pop(mapping.parent, None)

    # ── Model-level initialization ─────────────────────────────────────────────

    def _prepare_model(self) -> None:
        """One-time model-level AWQ setup: check compatibility and resolve mappings.

        Triggered automatically when model_context is assigned (via __setattr__).
        """
        from auto_round.algorithms.quantization.awq.mappings import check_model_compatibility

        model = self.model_context.model
        report = check_model_compatibility(model, self._user_mappings)
        for w in report["warnings"]:
            logger.warning(w)
        if not report["compatible"]:
            model_class = report.get("model_class", "unknown")
            raise ValueError(
                f"AWQ: no smooth-balance mappings could be resolved for "
                f"'{model_class}'. Either the model architecture is not "
                f"supported for automatic AWQ mapping detection, or the model "
                f"has no repeating transformer block structure. "
                f"You can provide explicit mappings via "
                f"AWQConfig(mappings=[{{'smooth_layer': '<regex>', "
                f"'balance_layers': ['<regex>', ...]}}])."
            )

        self.resolve_all_mappings(model)

        # AWQ caches block I/O on CPU — only one block lives on GPU at a time.
        self.compress_context.cache_device = torch.device("cpu")

    def _check_scheme_compatibility(self) -> None:
        """Validate the quantization scheme against AWQ inference backends."""
        bits = self.bits
        act_bits = self.act_bits or 16
        data_type = self.data_type or "int"

        if "int" not in data_type:
            raise ValueError(
                f"AWQ requires integer data_type, got '{data_type}'. "
                f"AWQ channel scaling is designed for integer quantization "
                f"grids. Use algorithm='autoround' for FP8/MXFP quantization."
            )

        if act_bits is not None and act_bits < 16 and act_bits != 8:
            raise ValueError(
                f"AWQ with act_bits={act_bits} is not supported. "
                f"No inference kernel exists for W{bits}A{act_bits} in vllm "
                f"or sglang. Supported schemes: W4A16 (canonical AWQ) or "
                f"W8A8 (compressed_tensors INT8 backend)."
            )

        if bits == 4 and act_bits >= 16:
            pass
        elif bits == 8 and act_bits == 8:
            logger.info(
                "AWQ with W8A8: AWQ smoothing will be applied, followed by "
                "INT8 quantization. This is served by vllm's "
                "compressed_tensors backend (cutlass INT8 GEMM), not AWQ "
                "kernels."
            )
        elif bits not in (4, 8):
            logger.warning(
                f"AWQ with bits={bits}: vllm AWQ kernels only support "
                f"bits=4 (AWQ/Marlin) natively. bits=8 is supported via "
                f"compressed_tensors. Other bit widths may not have "
                f"optimized inference kernels."
            )

    # ── Weight quantization (delegated RTN) ─────────────────────────────────────

    def quantize_layer(self, name: str, dtype: torch.dtype = None) -> None:
        """Quantize a single layer using RTN after AWQ smoothing has been applied.

        AWQ's quantize_layer is simpler than RTN's because:
        - AWQ always uses plain RTN (disable_opt_rtn=True, no imatrix)
        - No MoE-specific RTN disabling (AWQ handles MoE via mappings)
        - No GGUF special path (AWQ targets AWQ/Marlin inference kernels)

        Args:
            name: Fully-qualified module name (e.g. "model.layers.0.self_attn.q_proj").
            dtype: Optional dtype to cast the layer to before quantization.
        """
        m = get_module(self.model, name)
        if dtype is not None:
            m = m.to(dtype)

        m = convert_module_to_hp_if_necessary(m, self.model_context.amp_dtype, device_manager.device)
        set_module(self.model, name, m)
        tuning_device = m.tuning_device if hasattr(m, "tuning_device") else device_manager.device

        try:
            m = m.to(tuning_device)
            m = WrapperLinear(
                m,
                device=tuning_device,
                enable_minmax_tuning=False,
                enable_norm_bias_tuning=False,
                enable_round_tuning=False,
                enable_torch_compile=self.compress_context.enable_torch_compile,
                disable_opt_rtn=self.config.disable_opt_rtn,
                iters=0,
            )
            m = m.unwrapper({})
        except torch.OutOfMemoryError:
            cuda_error_msg = traceback.format_exc()
            m = m.orig_layer if hasattr(m, "orig_layer") else m
            try:
                logger.error(cuda_error_msg)
                logger.warning("AWQ quantize_layer: falling back to CPU.")
                m.to("cpu")
                m = WrapperLinear(
                    m,
                    enable_minmax_tuning=False,
                    enable_norm_bias_tuning=False,
                    enable_round_tuning=False,
                    enable_torch_compile=self.compress_context.enable_torch_compile,
                    disable_opt_rtn=self.config.disable_opt_rtn,
                    iters=0,
                )
                m = m.unwrapper({})
            except Exception:
                raise

        set_module(self.model, name, m)
        self._immediate_pack_and_save_module(name)

    def quantize_layer_outside_block(self, *args, **kwargs):
        """Quantize layers outside blocks (e.g. lm_head). Delegates to quantize_layer."""
        return self.quantize_layer(*args, **kwargs)

    def _immediate_pack_and_save_module(self, module_name: str) -> None:
        """Pack and/or save a quantized module if immediate mode is enabled."""
        shard_writer = ShardWriter.get_shard_writer()
        to_cpu = self.compress_context.low_gpu_mem_usage
        module = get_module(self.model, module_name)
        if self.compress_context.is_immediate_packing:
            immediate_pack(module_name, self.layer_config)
            if to_cpu:
                module = module.to("cpu")
                packed_module = get_module(self.model, module_name)
                set_module(self.model, module_name, packed_module.to("cpu"))
        else:
            if to_cpu:
                module = module.to("cpu")
            set_module(self.model, module_name, module)
        if self.compress_context.is_immediate_saving:
            module = get_module(self.model, module_name)
            module.to("cpu")
            shard_writer.write(module, module_name, False)
            module.to("meta")

    # ── Block-level quantization ──────────────────────────────────────────────

    def quantize_block(
        self, block: torch.nn.Module, input_ids=None, input_others=None, reference_output=None, **kwargs
    ) -> dict:
        """AWQ block quantization: collect stats → smooth → quantize.

        The full per-block AWQ pipeline:
          1. Register activation hooks for this block's mappings
          2. Run block forward to collect activation stats + parent kwargs
          3. Apply AWQ smoothing (grid search + scale application)
          4. Collect act_max AFTER smoothing (for activation quantization)
          5. Quantize the smoothed block via RTN

        Args:
            block: Module already on the compute device.
            input_ids: Calibration inputs for this block.
            input_others: Additional kwargs for block forward.
            reference_output: FP16 block output (kept for interface consistency).
            **kwargs: Must include 'block_name' for mapping lookup.
        """
        if isinstance(block, _WrapperMultiblock):
            raise ValueError(
                "AWQ does not support nblocks > 1 (multi-block quantization). "
                "Each block must be quantized individually. "
                "Please set nblocks=1 when using algorithm='awq'."
            )

        block_name = kwargs.get("block_name")
        if block_name is None:
            # Infer block_name from resolved mappings by matching the block's modules
            for prefix, mappings in self._block_groups.items():
                if any(m.smooth_layer is mod or m.parent is mod for m in mappings for mod in block.modules()):
                    block_name = prefix
                    break
            if block_name is None:
                raise ValueError(
                    "AWQQuantizer.quantize_block() could not determine block_name. "
                    "Pass block_name explicitly or ensure resolved mappings cover this block."
                )

        model = self.model
        bs = self.batch_size * self.infer_bs_coeff
        if self.compress_context.low_gpu_mem_usage:
            bs = 1
            logger.info("AWQ: low_gpu_mem_usage enabled, setting inference batch size to 1.")

        # Step 1 & 2: AWQ smoothing (optional)
        if self.apply_smooth:
            awq_hooks = self.register_activation_hooks(model, block_prefix=block_name)
            self._get_block_outputs(block, input_ids, input_others, bs, save_output=False)
            for h in awq_hooks:
                h.remove()

            self.smooth_block(block_name)

        # Step 3: Collect act_max AFTER smoothing
        # AWQ smoothing changes internal activations (LayerNorm output /= scales),
        # so act_max must be collected post-smoothing. Reset any pre-smoothing
        # act_max values first to avoid stale data persisting via max().
        act_max_hooks = self.register_calibration_hooks(block, imatrix=False)
        if act_max_hooks:
            for _name, m in block.named_modules():
                if hasattr(m, "act_max"):
                    del m.act_max
            self._get_block_outputs(block, input_ids, input_others, bs, save_output=False)
            for h in act_max_hooks:
                h.remove()

        # Step 4: Quantize the smoothed block (delegated RTN)
        # This is equivalent to RTNQuantizer.quantize_block but inlined here
        # to avoid inheritance coupling.
        update_block_global_scale_if_needed(block, self.data_type, self.group_size)
        if (
            self.config.is_act_nv_fp
            or self.config.is_static_afp8
            or (self.config.is_wfp8afp8 and not self.config.act_dynamic)
        ):
            set_amax_for_all_moe_layers(block, attr_name="act_max")

        for _name, m in block.named_modules():
            if hasattr(m, "global_name") and check_to_quantized(m):
                self.quantize_layer(m.global_name)

        return {}

    # ── Module alignment (device onload/offload) ─────────────────────────────

    @contextlib.contextmanager
    def _align_modules(
        self,
        mapping,
        device: torch.device | str | None,
    ):
        """Temporarily move mapping-related modules to *device* for grid search.

        If *device* is None (model already on the compute device), this is a
        no-op passthrough.
        """
        if device is None:
            yield
            return

        modules_to_move = [mapping.parent, mapping.smooth_layer] + list(mapping.balance_layers)
        # Deduplicate (parent may overlap with smooth/balance)
        seen = set()
        unique = []
        for m in modules_to_move:
            if id(m) not in seen:
                seen.add(id(m))
                unique.append(m)

        # Onload
        for m in unique:
            m.to(device)

        try:
            yield
        finally:
            # Offload back to CPU
            for m in unique:
                m.to("cpu")
            clear_memory()

    # ── Activation statistics collection ──────────────────────────────────────

    def register_activation_hooks(
        self,
        model: torch.nn.Module,
        block_prefix: str | None = None,
    ) -> list:
        """Register activation hooks for AWQ and return hook handles.

        Hooks are registered for:
        1. Smooth layers: accumulate abs activation sums/counts for x_mean.
        2. Parent modules: cache forward kwargs for output-based loss

        Args:
            model: The model (hooks are registered on the actual module
                objects, which are shared with blocks obtained via
                ``get_module``).
            block_prefix: When provided, only register hooks for mappings
                whose smooth layer belongs to this block (e.g.
                "model.layers.0").  Used in the block-by-block pipeline.

        Should be called by the compressor before calibration, and handles
        should be removed after calibration.
        """
        if not self._resolved_mappings:
            self._resolved_mappings = resolve_mappings(model, self._user_mappings)

        mappings = self._resolved_mappings
        if block_prefix is not None:
            mappings = [m for m in mappings if _extract_block_prefix(m.smooth_name) == block_prefix]

        hooks = []
        smooth_names = {m.smooth_name for m in mappings}

        # Hook smooth layers for activation statistics
        for name, module in model.named_modules():
            if name not in smooth_names:
                continue

            def _make_hook(layer_name):
                def hook_fn(mod, args, output):
                    if isinstance(output, tuple):
                        x = output[0]
                    else:
                        x = output
                    if x is None or x.numel() == 0:
                        return

                    # Compute abs-sum directly on GPU without materialising a
                    # full float32 intermediate. flatten→abs→sum produces a
                    # single [channels] tensor; .cpu() immediately frees the
                    # GPU allocation (aligned with AR's "immediate CPU move"
                    # pattern from CalibCompressor hooks).
                    channel_sum = x.detach().float().flatten(0, -2).abs().sum(dim=0).cpu()
                    count = x.shape[:-1].numel()

                    if layer_name not in self._activation_stats:
                        self._activation_stats[layer_name] = [
                            torch.zeros_like(channel_sum),
                            0,
                        ]
                    self._activation_stats[layer_name][0] += channel_sum
                    self._activation_stats[layer_name][1] += count

                return hook_fn

            h = module.register_forward_hook(_make_hook(name))
            hooks.append(h)

        # Hook parent modules to cache kwargs for output-based loss
        parent_modules_hooked = set()
        for mapping in mappings:
            parent = mapping.parent
            if id(parent) in parent_modules_hooked:
                continue
            parent_modules_hooked.add(id(parent))

            self._parent_args_cache[parent] = []

            def _make_parent_hook(parent_module):
                def hook_fn(mod, args, kwargs):
                    # Cache ALL calibration batches for output-based grid
                    # search loss, use all calibration data without subsampling.
                    mod_cls_id = id(type(mod))
                    if mod_cls_id not in self._parent_signatures:
                        self._parent_signatures[mod_cls_id] = inspect.signature(mod.forward)
                    sig = self._parent_signatures[mod_cls_id]
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()

                    # Infer the parent's compute dtype so that tensors
                    # upcasted by nn.LayerNorm (float32) are stored in the
                    # weight dtype (e.g. bfloat16).  This avoids repeated
                    # per-sample casting in the grid search stage.
                    param = next(mod.parameters(), None)
                    w_dtype = param.dtype if param is not None else None

                    stored = {}
                    for k, v in bound.arguments.items():
                        if isinstance(v, torch.Tensor):
                            v = v.detach()
                            if w_dtype is not None and v.is_floating_point() and v.dtype != w_dtype:
                                v = v.to(w_dtype)
                            stored[k] = v
                        elif isinstance(v, tuple) and any(isinstance(t, torch.Tensor) for t in v):
                            # Detach tensors in tuples (e.g. position_embeddings
                            # = (cos, sin)) to release computation graph refs.
                            stored[k] = tuple(
                                (
                                    (
                                        t.detach().to(w_dtype)
                                        if w_dtype and t.is_floating_point() and t.dtype != w_dtype
                                        else t.detach()
                                    )
                                    if isinstance(t, torch.Tensor)
                                    else t
                                )
                                for t in v
                            )
                        elif hasattr(v, "key_cache"):
                            # Null out KV cache objects (DynamicCache etc.)
                            stored[k] = None
                        else:
                            stored[k] = v

                    self._parent_args_cache[parent_module].append(stored)

                return hook_fn

            h = parent.register_forward_pre_hook(_make_parent_hook(parent), with_kwargs=True)
            hooks.append(h)

        return hooks

    # ── Grid search ───────────────────────────────────────────────────────────

    def _get_grid_search_params(self) -> list[tuple[float, bool]]:
        """Get grid search parameters (ratio, duo_scaling).

        Returns:
            List of (ratio, use_duo_scaling) tuples for the grid search.
        """
        match self.duo_scaling:
            # "both": half grid with duo off, half with duo on
            case "both":
                n_grid = max(int(self.n_grid / 2), 2)
                return [
                    (grid_idx / (n_grid - 1), duo_scaling)
                    for grid_idx in range(n_grid)
                    for duo_scaling in [False, True]
                ]
            case False:
                n_grid = max(self.n_grid, 2)
                return [(grid_idx / (n_grid - 1), False) for grid_idx in range(n_grid)]
            # True: include identity (0.0, False) as first, then duo points
            case True:
                n_grid = max(self.n_grid, 3)
                return [(0.0, False)] + [(grid_idx / (n_grid - 2), True) for grid_idx in range(n_grid - 1)]
            case _:
                raise ValueError(f"Found unexpected duo_scaling configuration {self.duo_scaling}")

    @staticmethod
    def _compute_layer_means(layers: list[torch.nn.Module], group_size: int) -> torch.Tensor:
        """Compute per-channel mean of normalised weights.

        Within each quantization group, weights are normalized by their group max
        (so values are on a 0-1 scale), then averaged across all groups to get
        per-channel importance.

        Args:
            layers: Balance layers whose weights are concatenated.
            group_size: Quantization group size. If -1, uses full channel width.

        Returns:
            Per-channel mean of normalised weights [in_features].
        """
        # Concatenate all balance layer weights [total_out, in_features]
        weight = torch.cat([m.weight.detach().float() for m in layers], dim=0)
        org_shape = weight.shape

        # Determine effective group size
        gs = group_size if group_size > 0 else org_shape[1]

        # Pad when needed so AWQ works with layers whose input width is not a
        # multiple of group_size, matching grouped RTN quantization behavior.
        weight, _, pad_len = reshape_pad_tensor_by_group_size(weight, gs)
        # Normalize within each group: abs / max (0-1 scale per group)
        w_scale = weight.abs() / (weight.abs().amax(dim=1, keepdim=True) + 1e-6)
        # Remove padding, then take mean across output channels.
        w_scale = revert_tensor_by_pad(w_scale, orig_shape=org_shape, pad_len=pad_len)
        w_mean = w_scale.mean(0)
        return w_mean

    @torch.no_grad()
    def _grid_search_scales(
        self,
        mapping: ResolvedMapping,
        x_mean: torch.Tensor,
    ) -> torch.Tensor | None:
        """Find the best scaling ratio via grid search.

        Uses output-based error:
        ``L(s) = || fp16_output - Q(W*s) @ (X/s) ||^2``

        For each candidate scaling ratio, applies scales to balance layer
        weights, quantize-dequantizes them, runs all cached calibration
        batches through the parent module, and computes the output MSE
        against the fp16 baseline.

        Returns:
            Best scales tensor, or None if no valid scale was found.
        """
        device = mapping.balance_layers[0].weight.device
        x_mean = x_mean.to(device)

        # Compute normalised weight means
        group_size = self.group_size if self.group_size > 0 else -1
        if self.duo_scaling is not False:
            w_mean = self._compute_layer_means(mapping.balance_layers, group_size).to(device)

        # Try to run parent module forward for output-based loss
        parent_kwargs_list = self._parent_args_cache.get(mapping.parent, [])
        use_parent_forward = len(parent_kwargs_list) > 0

        if use_parent_forward:
            # Compute fp16 baseline outputs for loss computation
            fp16_outputs = self._run_parent_samples(mapping.parent, parent_kwargs_list)
            if not fp16_outputs or all(f.numel() == 0 for f in fp16_outputs):
                use_parent_forward = False

        # Save original weights for restoration during grid search
        orig_state = {bl: bl.weight.data.clone() for bl in mapping.balance_layers}

        best_error = float("inf")
        best_scales = None
        best_ratio = -1

        # Pre-resolve quant function once.
        ref_layer = mapping.balance_layers[0]
        ref_name = getattr(ref_layer, "global_name", None) or ""
        ref_cfg = self.layer_config.get(ref_name, {})
        try:
            cached_quant_func, _ = get_quant_func(
                ref_cfg.get("data_type", self.data_type),
                ref_cfg.get("bits", self.bits),
                ref_cfg.get("sym", self.sym),
                disable_opt_rtn=True,
                group_size=ref_cfg.get("group_size", self.group_size),
                iters=0,
            )
        except Exception:
            cached_quant_func = None

        grid_params = self._get_grid_search_params()

        for ratio, use_duo in grid_params:
            # Compute scales
            if use_duo:
                scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
            else:
                scales = x_mean.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt()
            scales[torch.isinf(scales)] = 1
            scales[torch.isnan(scales)] = 1
            scales_view = scales.view(1, -1).to(device)

            if use_parent_forward:
                # Q(W * s) / s: mutate balance layer weights
                for bl in mapping.balance_layers:
                    bl.weight.data.copy_(orig_state[bl] * scales_view)
                    w_qdq = self._quantize_dequantize_weight(
                        bl,
                        bl.weight.data.float(),
                        quant_func=cached_quant_func,
                    )
                    if w_qdq is not None:
                        bl.weight.data = (w_qdq / scales_view).to(bl.weight.dtype)
                    else:
                        bl.weight.data.copy_(orig_state[bl])

                # Collect quantized outputs then compute loss
                int_w_outputs = self._run_parent_samples(mapping.parent, parent_kwargs_list)
                total_loss = self._compute_loss(fp16_outputs, int_w_outputs)
                del int_w_outputs

                # Restore original weights
                for bl in mapping.balance_layers:
                    bl.weight.data.copy_(orig_state[bl])
            else:
                # Weight-only fallback: || W - Q(W*s)/s ||^2
                total_loss = 0.0
                for bl in mapping.balance_layers:
                    w_orig = orig_state[bl].to(device)
                    w_scaled = w_orig * scales_view
                    w_qdq = self._quantize_dequantize_weight(
                        bl,
                        w_scaled,
                        quant_func=cached_quant_func,
                    )
                    if w_qdq is None:
                        total_loss = float("inf")
                        break
                    w_qdq_unscaled = w_qdq / scales_view
                    total_loss += (w_orig - w_qdq_unscaled).pow(2).sum().item()

            if total_loss < best_error:
                best_error = total_loss
                best_scales = scales.clone()
                best_ratio = ratio

        if best_ratio < 0:
            logger.warning(f"AWQ grid search failed for '{mapping.smooth_name}': " "no finite error found.")
            return None

        logger.debug(f"AWQ '{mapping.smooth_name}': best_ratio={best_ratio:.2f}, " f"best_error={best_error:.3e}")
        return best_scales

    # ── Parent module forward ─────────────────────────────

    @torch.no_grad()
    def _run_parent_samples(
        self,
        parent: torch.nn.Module,
        kwargs_list: list[dict],
    ) -> list[torch.Tensor]:
        """Run cached samples through the parent module.

        Feeds cached kwargs through the parent forward without any CUDA
        synchronisation between batches, so the GPU can pipeline all forwards.
        Outputs are kept on-device for loss computation.
        """
        outputs = []
        for stored_kwargs in kwargs_list:
            out = parent(**stored_kwargs)
            if isinstance(out, tuple):
                out = out[0]
            outputs.append(out)
        return outputs

    @staticmethod
    @torch.no_grad()
    def _compute_loss(
        fp16_outputs: list[torch.Tensor],
        int_w_outputs: list[torch.Tensor],
    ) -> float:
        """Compute normalised MSE between fp16 and quantized outputs."""
        device = fp16_outputs[0].device
        loss = torch.tensor(0.0, device=device)
        num_elements = torch.tensor(0, device=device, dtype=torch.long)

        for fp16_out, int_w_out in zip(fp16_outputs, int_w_outputs):
            loss += torch.nn.functional.mse_loss(
                fp16_out.float(),
                int_w_out.to(fp16_out.device).float(),
                reduction="sum",
            )
            num_elements += fp16_out.numel()

        if num_elements == 0:
            return float("inf")
        return (loss / num_elements).item()

    def _quantize_dequantize_weight(
        self,
        layer: torch.nn.Module,
        weight: torch.Tensor,
        quant_func=None,
    ) -> torch.Tensor | None:
        """Quantize-dequantize a weight tensor using the layer's config.

        Args:
            quant_func: Pre-resolved quantization function. When provided,
                skips the ``get_quant_func`` dispatch (avoids redundant
                lookups in the inner grid search loop).

        Returns the quantized-then-dequantized weight, or None on failure.
        """
        layer_name = getattr(layer, "global_name", None) or ""
        config = self.layer_config.get(layer_name, {})
        bits = config.get("bits", self.bits)
        group_size = config.get("group_size", self.group_size)
        sym = config.get("sym", self.sym)
        data_type = config.get("data_type", self.data_type)

        if quant_func is None:
            try:
                quant_func, _ = get_quant_func(
                    data_type,
                    bits,
                    sym,
                    disable_opt_rtn=True,  # AWQ always uses plain RTN
                    group_size=group_size,
                    iters=0,  # Route to rtn_int_sym, not int_sym
                )
            except Exception:
                return None

            if quant_func is None:
                return None

        try:
            qdq_weight, scale, zp = quant_func(
                weight,
                bits=bits,
                group_size=group_size,
            )
            return qdq_weight
        except Exception:
            return None

    # ── Apply scales ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _apply_scales(self, mapping: ResolvedMapping, scales: torch.Tensor) -> None:
        """Apply the computed AWQ scales to smooth and balance layers.

        - Balance layers (Linear): weight *= scales (along input channels)
        - Smooth layer (LayerNorm/RMSNorm): weight /= scales, bias /= scales
        """
        for bl in mapping.balance_layers:
            device = bl.weight.device
            s = scales.to(device).view(1, -1)
            bl.weight.data.mul_(s)

        smooth = mapping.smooth_layer
        device = smooth.weight.device
        s = scales.to(device)

        if smooth.weight.ndim == 1:
            # LayerNorm / RMSNorm: 1-D weight (per-channel)
            smooth.weight.data.div_(s)
        else:
            # Edge case: when smooth layer's out_features != balance layer's
            # in_features (e.g. fused qkv_proj smoothing o_proj). Scale the
            # last output features (aligned with AutoAWQ).
            # https://github.com/casper-hansen/AutoAWQ/blob/main/awq/quantize/scale.py#L123
            smooth.weight.data[-s.size(0) :].div_(s.view(-1, 1))

        if hasattr(smooth, "bias") and smooth.bias is not None:
            smooth.bias.data.div_(s)
