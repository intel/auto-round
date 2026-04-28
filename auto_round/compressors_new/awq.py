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

"""AWQ compressor: orchestrates AWQ smoothing + weight quantization.

AWQ (Activation-Aware Weight Quantization) is a standalone algorithm that
protects salient weight channels via activation-aware scaling before applying
standard RTN quantization.

The compressor inherits from CalibCompressor for stable infrastructure
(model loading, calibration data, device management, block input caching)
and implements its own quantization flow:

  1. Full-model forward pass with AWQ activation hooks to collect per-channel
     activation magnitudes for smooth layers.
  2. AWQ smoothing: grid-search for optimal channel-wise scales and apply them
     to smooth/balance layer pairs.
  3. Block-by-block weight quantization on the smoothed model using the
     quantizer's ``quantize_block`` / ``quantize_layer`` methods.
"""

from __future__ import annotations

import gc
import time

import accelerate
import torch
from tqdm import tqdm

from auto_round.algorithms.alg_config import AlgConfig
from auto_round.algorithms.quantization.awq.mappings import check_model_compatibility
from auto_round.compressors_new.calib import CalibCompressor
from auto_round.logger import logger
from auto_round.modeling.fused_moe.replace_modules import materialize_model_
from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
    clear_memory,
    convert_module_to_hp_if_necessary,
    get_block_names,
    get_module,
    is_auto_device_mapping,
    memory_monitor,
    mv_module_from_gpu,
    to_device,
    to_dtype,
)
from auto_round.utils.device import _force_trim_malloc


class AWQCompressor(CalibCompressor):
    """Compressor that applies AWQ smoothing before weight quantization.

    Inherits from ``CalibCompressor`` for calibration infrastructure (data
    loading, block input caching, multi-GPU dispatch) and implements its own
    AWQ-specific quantization pipeline.  This avoids coupling to Intel's
    internal ``CalibratedRTNCompressor`` which may evolve independently of
    the AWQ algorithm spec.
    """

    need_calib: bool = True

    def __init__(
        self,
        config: AlgConfig,
        model: torch.nn.Module,
        **kwargs,
    ):
        kwargs["iters"] = 0
        super().__init__(
            config,
            model,
            **kwargs,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def quantize(self):
        """Quantize with AWQ smoothing pre-processing.

        Returns:
            tuple: (quantized_model, layer_config)
        """
        # post_init must be called OUTSIDE @torch.no_grad() because
        # AutoScheme delta-loss selection requires autograd (backward pass).
        self.post_init()

        # Reclaim heap fragmentation from init/post_init before the memory-intensive quantize loop.
        gc.collect()
        _force_trim_malloc()

        return self._quantize_impl()

    # ── Quantization pipeline ─────────────────────────────────────────────────

    # Use no_grad instead of inference_mode
    # https://github.com/intel/auto-round/issues/1620
    @torch.no_grad()
    def _quantize_impl(self):
        """Block-by-block AWQ pipeline: calibrate → smooth → quantize per block.

        Merges the previous 3-phase approach (full-model calibration → global
        smoothing → block-wise RTN) into a single per-block loop, reusing
        CalibCompressor's ``_should_stop_cache_forward`` infrastructure for
        block-input caching.

        Per-block flow:
          1. Move block to GPU, register AWQ hooks, run block forward to
             collect activation statistics and parent module kwargs.
          2. Apply AWQ smoothing (grid search + scale application) for the
             block's mappings — the block is already on GPU, so no device
             alignment overhead.
          3. RTN-quantize the smoothed block weights.
          4. Move block to CPU, propagate block outputs as next block's input.

        Memory profile: only one block on GPU at a time (~0.5 GB for an 8B
        model), plus cached block inputs on ``cache_device``.  The full model
        never needs to reside on GPU simultaneously.
        """
        t_start = time.time()

        # ── Validation ────────────────────────────────────────────────────
        self._check_scheme_compatibility()

        report = check_model_compatibility(
            self.model_context.model,
            getattr(self.quantize_config, "mappings", None),
        )
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

        # ── Embedding quantization ────────────────────────────────────────
        formats = getattr(self, "formats", None) or []
        if not (any(fmt.is_gguf() for fmt in formats) or self.super_bits is not None):
            self._quantize_embedding_layer()
        clear_memory(device_list=self.compress_context.device_list)

        model = self.model_context.model

        # ── Resolve AWQ mappings once for all blocks ──────────────────────
        self.quantizer.resolve_all_mappings(model)

        all_blocks = self.quantizer.quant_block_list or get_block_names(model)
        if not all_blocks:
            raise ValueError("Could not find any blocks. Check the model or quant_block_list.")

        # ── Cache first block inputs ─────────────────────────────────────
        # AWQ only needs the first block's input (embedding output).
        # Use cache_inter_data directly on CPU — the model stays on CPU,
        # the patched forward runs embedding → first block → early stop via
        # _should_stop_cache_forward.  This avoids the GPU fallback in
        # try_cache_inter_data_gpucpu that loads the full model (~16 GB)
        # when has_qlayer_outside_block is True (irrelevant to AWQ).
        first_block_names = [block[0] for block in all_blocks]
        try:
            all_inputs = self.cache_inter_data(first_block_names, self.nsamples)
        except NotImplementedError:
            # Flash attention doesn't support CPU — fall back to GPU caching
            logger.info("CPU caching failed, falling back to GPU caching")
            all_inputs = self.try_cache_inter_data_gpucpu(first_block_names, self.nsamples)

        # Move model to CPU after caching
        if hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1:
            accelerate.hooks.remove_hook_from_submodules(model)
        model = mv_module_from_gpu(model)
        clear_memory(device_list=self.compress_context.device_list)
        _force_trim_malloc()
        t_cache_done = time.time()
        logger.info(f"AWQ: block input caching done ({t_cache_done - t_start:.1f}s)")

        # ── Low-CPU-memory offloading ─────────────────────────────────────
        if self.compress_context.low_cpu_mem_usage:
            if self.model_context.is_model_patched and not self.compress_context.is_immediate_saving:
                self._offloader(
                    model,
                    all_blocks,
                    clear_memory=True,
                    device_list=self.compress_context.device_list,
                )
                if not self._offloader.enabled:
                    self.compress_context.low_cpu_mem_usage = False
            else:
                self.compress_context.low_cpu_mem_usage = False

        # ── Block-by-block: calibrate → smooth → quantize ────────────────
        # AWQ operates one block at a time on GPU — block outputs must be
        # cached on CPU to avoid accumulating ~15 GB VRAM with large nsamples.
        self.compress_context.cache_device = torch.device("cpu")

        pbar = tqdm(range(sum(len(block) for block in all_blocks)))
        device = self.compress_context.device

        for block_names in all_blocks:
            first_block = block_names[0]
            inputs = all_inputs.pop(first_block)

            # Normalize hidden_state key → input_ids
            input_keys = [k for k in inputs if k.startswith("hidden_state")]
            if len(input_keys) != 1:
                raise RuntimeError(
                    "hidden_states arg mismatch. Please file an issue at " "https://github.com/intel/auto-round/issues"
                )
            inputs["input_ids"] = inputs.pop(input_keys[0])

            clear_memory(self.inputs, device_list=self.compress_context.device_list)

            total_samples = len(inputs["input_ids"])
            if total_samples < self.quantizer.batch_size:
                self.quantizer.batch_size = total_samples
                logger.warning(f"Forcing batch size to {total_samples}")

            # AWQ always caches block I/O on CPU — only one block lives
            # on GPU at a time, so there is no benefit to GPU caching, and
            # with nsamples=512 the tensors would consume ~15 GB VRAM.
            input_ids = to_device(inputs.pop("input_ids"), "cpu")
            input_others = to_device(inputs, "cpu")

            # Dtype handling
            tmp_dtype = self.model_context.amp_dtype if self.model_context.amp else torch.float32
            input_ids = [id_.to(tmp_dtype) for id_ in input_ids]
            for key, val in input_others.items():
                if isinstance(val, torch.Tensor) and val.dtype in (
                    torch.float16,
                    torch.bfloat16,
                ):
                    input_others[key] = val.to(tmp_dtype)
                elif isinstance(val, list):
                    input_others[key] = [to_dtype(v, tmp_dtype) for v in val]

            for block_name in block_names:
                pbar.set_description(f"AWQ {block_name}")
                block = get_module(model, block_name)

                if self.compress_context.low_cpu_mem_usage:
                    self._offloader.reload(model, block_name)

                # ── Infrastructure: materialize, dtype, device ────────────
                materialize_model_(block)
                block = convert_module_to_hp_if_necessary(
                    block,
                    dtype=self.model_context.amp_dtype,
                    device=device,
                )

                if (
                    is_auto_device_mapping(self.compress_context.device_map)
                    and len(self.compress_context.device_list) > 1
                    and not self.model_context.is_diffusion
                ):
                    from auto_round.utils.device import (
                        set_auto_device_map_for_block_with_tuning,
                    )

                    set_auto_device_map_for_block_with_tuning(
                        block,
                        self.compress_context.device_map,
                        input_ids,
                        self.compress_context.low_gpu_mem_usage,
                        self.quantizer.batch_size,
                        device,
                    )
                    if len(self.compress_context.device_list) > 1:
                        from accelerate.hooks import (
                            AlignDevicesHook,
                            add_hook_to_module,
                        )

                        for _, _mod in block.named_modules():
                            if len(list(_mod.children())) != 0 or not hasattr(_mod, "tuning_device"):
                                continue
                            add_hook_to_module(
                                _mod,
                                AlignDevicesHook(_mod.tuning_device, io_same_device=True),
                                True,
                            )
                else:
                    block = block.to(device)

                bs = self.quantizer.batch_size * self.quantizer.infer_bs_coeff

                # ── Step 1: Collect AWQ stats via block forward ───────────
                # Register AWQ hooks (activation stats + parent kwargs) for
                # this block's smooth/balance mappings only.  The hooks are
                # on the actual module objects shared with the block, so they
                # fire when _get_block_outputs runs the block forward.
                #
                # act_max hooks are NOT registered here — they must run
                # AFTER smoothing so that balance-layer input distributions
                # reflect the post-smoothing reality (smooth_output /= s).
                # For W4A16 (act_bits>=16) this is moot (zero hooks), but
                # for W8A8 it is a correctness requirement.
                awq_hooks = self.quantizer.register_activation_hooks(model, block_prefix=block_name)
                block_input_ids = input_ids  # keep ref for post-smooth forward
                input_ids = self.quantizer._get_block_outputs(block, block_input_ids, input_others, bs)
                for h in awq_hooks:
                    h.remove()

                # ── Step 2: Apply AWQ smoothing for this block ────────────
                # Block is on the compute device — grid search runs in-place
                # without _align_modules overhead.
                self.quantizer.smooth_block(block_name)

                # ── Step 2b: Collect act_max AFTER smoothing ──────────────
                # AWQ smoothing changes internal activations (LayerNorm
                # output /= scales), so act_max for W8A8 activation
                # quantization must be collected post-smoothing.  The block
                # output is mathematically invariant to smoothing
                # (W*s @ x/s = W@x), so pre-smoothing outputs are still
                # valid for the next block.
                act_max_hooks = self.quantizer._register_act_max_hook(block)
                if act_max_hooks:
                    self.quantizer._get_block_outputs(
                        block,
                        block_input_ids,
                        input_others,
                        bs,
                        save_output=False,
                    )
                    for h in act_max_hooks:
                        h.remove()
                del block_input_ids

                # ── Step 3: RTN quantize the smoothed block ───────────────
                if self.compress_context.low_gpu_mem_usage:
                    block.to("cpu")
                    self.compress_context.clear_memory()

                self.quantizer.quantize_block(block)

                # ── Cleanup ───────────────────────────────────────────────
                if len(self.compress_context.device_list) > 1:
                    accelerate.hooks.remove_hook_from_submodules(block)
                mv_module_from_gpu(block)

                if self.enable_torch_compile:
                    torch._dynamo.reset()
                    self.quantizer._invalidate_block_forward_cache()

                if self.compress_context.is_immediate_packing:
                    for _n, _mod in block.named_modules():
                        if hasattr(_mod, "bits") and check_to_quantized(_mod):
                            from auto_round.compressors_new.utils import (
                                immediate_pack,
                            )

                            immediate_pack(
                                _mod.global_name,
                                self.quantizer.layer_config,
                            )

                if self.compress_context.is_immediate_saving:
                    self._ensure_shard_writer()
                    self.shard_writer.write(block, is_finalize=False)

                if self.compress_context.low_cpu_mem_usage and not self.compress_context.is_immediate_saving:
                    self._offloader(model, block_name)

                if block_name == block_names[-1]:
                    clear_memory(
                        input_ids,
                        device_list=self.compress_context.device_list,
                    )
                else:
                    clear_memory(
                        device_list=self.compress_context.device_list,
                    )

                memory_monitor.log_summary()
                pbar.update(1)

        pbar.close()

        # ── Quantize remaining layers outside blocks ──────────────────────
        block_name_set = set(name for block in all_blocks for name in block)
        for n, m in model.named_modules():
            if not check_to_quantized(m):
                continue
            if any(n == bn or n.startswith(f"{bn}.") for bn in block_name_set):
                continue
            dtype = torch.float32 if self.super_group_size is not None else None
            self.quantizer.quantize_layer_outside_block(n, dtype=dtype)

        # ── Finalize ──────────────────────────────────────────────────────
        convert_module_to_hp_if_necessary(
            self.model_context.model,
            self.model_context.amp_dtype,
            self.compress_context.device,
        )
        if self.compress_context.low_cpu_mem_usage:
            self._offloader.reload(self.model_context.model)
        if self.compress_context.is_immediate_saving:
            self._ensure_shard_writer()
            self.shard_writer.write(is_finalize=True)

        t_total = time.time() - t_start
        logger.info(f"AWQ quantization time {t_total}")

        self.model_context.quantized = True
        return self.model_context.model, self.quantizer.layer_config

    # ── Scheme validation ─────────────────────────────────────────────────────

    def _check_scheme_compatibility(self) -> None:
        """Validate the quantization scheme against AWQ inference backends.

        Supported deployment paths for AWQ-smoothed models:
          - **W4A16** (bits=4, act_bits>=16, data_type=int): canonical AWQ.
            Served by vllm AWQ/Marlin CUDA kernels and sglang.
          - **W8A8** (bits=8, act_bits=8, data_type=int): INT8 weight+activation.
            Served by vllm's compressed_tensors backend (cutlass INT8 GEMM).
            Uses a different export path than AWQ format.

        Rejected schemes:
          - **W4A4**: no INT4×INT4 GEMM kernel exists in vllm or sglang.
          - Non-int data types (fp8, mxfp, etc.): AWQ channel scaling is
            designed for integer quantization grids.
        """
        bits = getattr(self, "bits", None)
        act_bits = getattr(self, "act_bits", None) or 16
        data_type = getattr(self, "data_type", None) or "int"

        # AWQ channel scaling only makes sense for integer quantization
        if "int" not in data_type:
            raise ValueError(
                f"AWQ requires integer data_type, got '{data_type}'. "
                f"AWQ channel scaling is designed for integer quantization "
                f"grids. Use algorithm='autoround' for FP8/MXFP quantization."
            )

        # W4A4: no inference kernel in vllm or sglang
        if act_bits is not None and act_bits < 16 and act_bits != 8:
            raise ValueError(
                f"AWQ with act_bits={act_bits} is not supported. "
                f"No inference kernel exists for W{bits}A{act_bits} in vllm "
                f"or sglang. Supported schemes: W4A16 (canonical AWQ) or "
                f"W8A8 (compressed_tensors INT8 backend)."
            )

        if bits == 4 and act_bits >= 16:
            # Canonical AWQ — W4A16, served by vllm AWQ/Marlin kernels
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
