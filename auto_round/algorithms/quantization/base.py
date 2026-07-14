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
import traceback
from typing import Any
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from auto_round.algorithms.pipeline import BlockContext

from auto_round.algorithms.base import BasePipelineMember
from auto_round.algorithms.quantization.config import QuantizationConfig
from auto_round.algorithms.transforms.base import BaseWeightTransformer
from auto_round.compressors.utils import (
    block_forward,
    check_need_act_calibration,
    immediate_pack,
)
from auto_round.data_type import QUANT_FUNC_WITH_DTYPE
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size
from auto_round.logger import logger
from auto_round.utils import (
    INNER_SUPPORTED_LAYER_TYPES,
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
    clear_memory,
    compile_func,
    convert_module_to_hp_if_necessary,
    get_module,
    set_module,
)
from auto_round.utils.device_manager import device_manager
from auto_round.wrapper import WrapperLinear


# TODO wenhuach annotate this class and functions clearly with details
class BaseQuantizer(BasePipelineMember):
    """Base class for terminal weight-compression algorithms in a QuantizationPipeline.

    Developers adding a new quantization algorithm should inherit from this
    class and override at minimum :meth:`quantize_block`.


    Lifecycle hooks to override as needed:
        - :meth:`prepare_run`                – model-level setup (once before all blocks)
        - :meth:`block_forward_hooks`        – register act-calib hooks (context manager)
        - :meth:`quantize_block`             – **must override**: quantize a single block
        - :meth:`quantize_layer_outside_block` – quantize layers outside blocks
        - :meth:`finalize_run`               – model-level teardown (once after all blocks)
    """

    # Class-level attribute declarations for convenient access in quantization methods.
    # Scheme-related attrs (layer_config, scale_dtype, has_qlayer_outside_block, etc.)
    # are resolved by SchemeMixin in BaseCompressor and synced here after post_init().

    supported_types = SUPPORTED_LAYER_TYPES
    inner_supported_types = INNER_SUPPORTED_LAYER_TYPES
    enable_alg_ext = False  # TODO delete wenhuach

    def __init__(self, config: QuantizationConfig) -> None:
        super().__init__(config)
        self.layer_config = None
        # Calibration-time state lives on a shared
        # :class:`~auto_round.calibration.state.CalibrationState` instance.
        # The compressor wires its own instance here in ``_resolve_scheme``;
        # until then we own a private placeholder so property-based reads /
        # writes during construction don't blow up.
        from auto_round.calibration.state import CalibrationState

        self._calibration_state = CalibrationState()
        self.infer_bs_coeff = getattr(config, "infer_bs_coeff", 1)  # TODO wenhuach move to block_forward or calib state
        # Whether to feed quantized-block outputs as inputs to the next block.
        # Subclasses that support cascaded quantized-input (e.g. SignRoundQuantizer)
        # override this from their config.  Defaults to False for zero-shot algorithms
        # (RTN) where activations are not used during weight optimization.
        self.enable_quanted_input = getattr(config, "enable_quanted_input", False)

    def is_support_compile_block(self):
        return True

    # ── Shared CalibrationState forwarders ───────────────────────────────────────
    @property
    def calibration_state(self) -> Any:  # TODO this one could be deleted?
        return self._calibration_state

    @calibration_state.setter
    def calibration_state(self, new_state: Any) -> None:
        # Compressor-supplied shared instance; just rebind.
        self._calibration_state = new_state

    @property
    def attention_mask(self) -> list:  # TODO better move to quantize_block
        return self._calibration_state.attention_mask

    @attention_mask.setter
    def attention_mask(self, value: list) -> None:
        self._calibration_state.attention_mask = value if value is not None else []

    def bind(self, compressor: Any) -> None:
        """Wire shared state from the owning compressor.

        The compressor owns the authoritative ``model_context`` /
        ``compress_context`` / ``CalibrationState`` and resolves
        ``scale_dtype`` (string → torch dtype).  All quantizer fields that
        merely mirror the compressor are pulled from here in one place.
        """
        self.compressor = compressor
        self.model_context = compressor.model_context
        self.compress_context = compressor.compress_context
        self.scheme = compressor.scheme_context
        self.scale_dtype = compressor.scale_dtype  # TODO better move to scheme? wenhuach
        # Share the compressor's CalibrationState instance.
        self._calibration_state = compressor._calibration_state

    @property
    def model(self) -> torch.nn.Module | None:
        return self.model_context.model if self.model_context is not None else None

    @property
    def amp(self) -> bool:
        return getattr(self.model_context, "amp", False)

    @property
    def amp_dtype(self) -> torch.dtype:
        return getattr(self.model_context, "amp_dtype", torch.float32)

    def register_fp_input_forward_hooks(self, block: "torch.nn.Module") -> list:
        """Register hooks that fire during the reference (FP-input) block forward.

        Includes act_max hooks (for static activation quantization) by default.
        Subclasses override to add additional statistics collection
        (e.g. imatrix, AWQ scales). Returns a list of hook handles
        that the caller must remove when done.
        """
        handles = self._register_act_max_hooks(block)
        return handles

    def register_qinput_forward_hooks(self, block: "torch.nn.Module") -> list:
        """Register hooks that fire during the quantized-input block forward.

        Used when act-calib policy requires collecting stats from quantized
        activations. Includes act_max hooks by default.
        Subclasses override to add custom hooks. Returns a list of hook handles
        that the caller must remove when done.
        """
        handles = self._register_act_max_hooks(block)
        return handles

    # ── Activation-calibration hook infrastructure ───────────────────────────────

    def _register_act_max_hooks(self, model):
        """Register per-module act_max tracking hooks for static activation quantization.

        Internal implementation called by :meth:`block_forward_hooks`.
        Returns a list of hook handles that the caller must remove when done.
        """

        def collect_act_max(module, input, output):
            input = input[0] if isinstance(input, (tuple, list)) else input
            if input.numel() == 0:
                return
            input, _, _ = reshape_pad_tensor_by_group_size(input, module.act_group_size)
            act_max = torch.max(torch.abs(input), dim=-1).values
            if not hasattr(module, "act_max") or module.act_max.numel() == 0:
                module.act_max = act_max
                if self.config.is_act_nv_fp:
                    max_val = act_max.max()
                    module.act_max = max_val.unsqueeze(0) if max_val.dim() == 0 else max_val
                return

            act_max = act_max.to(module.act_max.device)
            if self.config.is_act_nv_fp:
                max_val = torch.max(act_max.max(), module.act_max.max())
                module.act_max = max_val.unsqueeze(0) if max_val.dim() == 0 else max_val
            else:
                module.act_max = torch.max(act_max, module.act_max)

        def should_collect(name, module):
            if isinstance(module, SUPPORTED_LAYER_TYPES):
                return (
                    hasattr(module, "act_dynamic")
                    and check_need_act_calibration(module.act_dynamic, module.act_data_type, module.act_bits)
                    and check_to_quantized(module)
                )
            if name in self.layer_config:
                config = self.layer_config[name]
                act_dynamic = config.get("act_dynamic", True)
                act_data_type = config.get("act_data_type", None)
                act_bits = config.get("act_bits", 16)
                return (
                    config["bits"] <= 8
                    and check_need_act_calibration(act_dynamic, act_data_type, act_bits)
                    and check_to_quantized(config)
                )
            return False

        handles = []
        if should_collect("", model):
            handles.append(model.register_forward_hook(collect_act_max))
            return handles
        for name, module in model.named_modules():
            if name and should_collect(name, module):
                handles.append(module.register_forward_hook(collect_act_max))
        return handles



    # ── Embedding quantization ────────────────────────────────────────────────────
    @torch.inference_mode()
    def quantize_embedding_layer(self) -> bool:
        """Quantizes embedding layers in the model according to the configuration.

        This method iterates through all modules in the model, identifies embedding
        layers specified in `self.layer_config`, and applies the appropriate quantization
        function based on bit precision, grouping strategy, and dtype.

        Note:
            Most quantization schemes do **not** quantize embeddings. Currently only
            GGUF formats require embedding quantization. Subclasses almost never need
            to override or call this method directly.

        Returns:
            bool: True if any embedding layer was quantized.
        """
        is_quantized = False
        for name, module in self.model_context.model.named_modules():
            # Skip non-Embedding modules or layers not in config
            if not isinstance(module, torch.nn.Embedding) or name not in self.layer_config:
                continue

            config = self.layer_config[name]

            # Skip layers that are not marked for quantization
            if not check_to_quantized(config):
                continue
            is_quantized = True
            config["scale_dtype"] = self.scale_dtype
            dtype = config["data_type"]

            # Determine quantization function key with symmetry/asymmetry
            if dtype not in QUANT_FUNC_WITH_DTYPE:
                dtype = f"{dtype}_{'sym' if config['sym'] else 'asym'}"

            quant_func = QUANT_FUNC_WITH_DTYPE[dtype]
            dtype = module.weight.dtype
            # As typically float32 are used in RTN to search scale zp,
            # to avoid cache a bf16 copy we'd better use float32
            if config.get("super_group_size", None) is not None:
                dtype = torch.float32

            # Attempt quantization on GPU, fall back to CPU if OOM
            try:
                weight, scale, zp = quant_func(
                    module.weight.to(dtype=dtype, device=device_manager.device),
                    **{
                        k: config.get(k, None)
                        for k in ["bits", "group_size", "super_bits", "super_group_size", "scale_dtype"]
                    },
                )
            except torch.OutOfMemoryError:
                cuda_error_msg = traceback.format_exc()
                try:
                    logger.error(cuda_error_msg)
                    logger.warning("falling back to CPU")
                    weight, scale, zp = quant_func(
                        module.weight.to("cpu"),
                        **{
                            k: config.get(k, None)
                            for k in ["bits", "group_size", "super_bits", "super_group_size", "scale_dtype"]
                        },
                    )
                except Exception as e:
                    raise

            # Overwrite the module's weights with the quantized version
            module.weight.data.copy_(weight.cpu())

            # Attach scale and zero point (zp) to the module
            for param_name, value in zip(["scale", "zp"], [scale, zp]):
                if isinstance(value, dict):
                    for k, v in value.items():
                        setattr(module, k if k == "scale" else f"w_{k}", v.cpu())
                elif isinstance(value, torch.Tensor):
                    setattr(module, param_name, value.cpu())
                else:
                    setattr(module, param_name, value)

            # Update config
            self.layer_config.setdefault(name, {}).update(config)
            del weight
            del scale
            del zp
            clear_memory(device_list=device_manager.device_list)

        return is_quantized

    # ── Abstract quantization interface ──────────────────────────────────────────

    def quantize_block(
        self,
        block: "torch.nn.Module",
        fp_inputs: list[torch.Tensor] | dict,
        input_others: dict,
        fp_outputs: list[torch.Tensor],
        q_inputs: list[torch.Tensor] | None,
        block_ctx: "BlockContext",
        **kwargs,
    ) -> dict:
        """Apply the quantization algorithm to a prepared block.

        This is the **pure-algorithm** entry point called by the Compressor after
        all infrastructure work (device placement, data collection, act-max hook
        registration, DDP setup) has been completed.

        Args:
            block:        The transformer block module.
            fp_inputs:    FP calibration inputs for this block (list[Tensor] or dict for diffusion).
            input_others: Auxiliary kwargs (attention_mask, position_ids, etc.).
            fp_outputs:   FP reference outputs for this block (list[Tensor]).
            q_inputs:     Quantized inputs from previous block (list[Tensor] or None).
            block_ctx:    Per-block pipeline context (BlockContext).

        Returns:
            dict: Best quantization parameters found, or ``{}`` if not applicable.
        """
        raise NotImplementedError("quantize_block must be implemented in subclasses of BaseQuantizer")

    def quantize_layer_outside_block(
        self, layer_name: str, input_ids=None, disable_opt_rtn: bool | None = None, **kwargs
    ):
        """Quantizes a single layer outside of a block using RTN fallback.

        Args:
            layer_name (str): The name of the layer to quantize.
            input_ids: Optional calibration inputs (unused in RTN fallback).
        """
        dtype = kwargs.pop("dtype", None)
        if dtype is not None:
            layer = get_module(self.model, layer_name)
            set_module(self.model, layer_name, layer.to(dtype))
        self.quantize_layer_via_rtn(layer_name, disable_opt_rtn=disable_opt_rtn)

    @torch.no_grad()
    def quantize_layer_via_rtn(self, layer_name: str, disable_opt_rtn: bool | None = None) -> None:
        """Quantize one layer with RTN and handle optional immediate pack/save."""
        layer = get_module(self.model, layer_name)
        layer = convert_module_to_hp_if_necessary(layer, self.model_context.amp_dtype, device_manager.device)
        set_module(self.model, layer_name, layer)
        tuning_device = layer.tuning_device if hasattr(layer, "tuning_device") else device_manager.device
        try:
            if disable_opt_rtn is None:
                disable_opt_rtn = bool(getattr(self.config, "disable_opt_rtn", False))
            if (
                not disable_opt_rtn
                and getattr(self.config, "orig_disable_opt_rtn", None) is None
                and self.model_context.is_moe_model
                and "expert" in layer.global_name
                and "shared_expert" not in layer.global_name
                and self.config.super_bits is None
            ):
                disable_opt_rtn = True
                logger.warning_once(
                    "MoE layer detected: optimized RTN is disabled for efficiency. "
                    "Use `--enable_opt_rtn` to force-enable it for MoE layers."
                )
            layer = layer.to(tuning_device)
            layer = WrapperLinear(
                layer,
                device=tuning_device,
                enable_minmax_tuning=False,
                enable_norm_bias_tuning=False,
                enable_round_tuning=False,
                enable_torch_compile=self.compress_context.enable_torch_compile,
                disable_opt_rtn=disable_opt_rtn,
                iters=0,
            )
            layer = layer.unwrapper({})
        except torch.OutOfMemoryError:
            cuda_error_msg = traceback.format_exc()
            layer = layer.orig_layer if hasattr(layer, "orig_layer") else layer
            try:
                logger.error(cuda_error_msg)
                logger.warning("falling back to CPU.")
                layer.to("cpu")
                layer = WrapperLinear(
                    layer,
                    enable_minmax_tuning=False,
                    enable_norm_bias_tuning=False,
                    enable_round_tuning=False,
                    enable_torch_compile=self.compress_context.enable_torch_compile,
                    iters=0,
                )
                layer = layer.unwrapper({})
            except Exception:
                raise
        set_module(self.model, layer_name, layer)
        # self._immediate_pack_and_save_module(layer_name)  # TODO wenhuach should not handle it here



    def dispatch_block(self, block: "torch.nn.Module", input_ids, input_others: dict):
        """Place a block on the correct device(s) for quantization.

        Default: move to primary device. Returns (block, card_0_in_high_risk, loss_device).
        Subclasses override for multi-GPU tensor-parallel dispatch.
        """
        block = block.to(device_manager.device)
        return block, False, device_manager.device  # This should

    def _resolve_block_forward(self):
        """Resolve and cache the block forward function once.

        This avoids repeated attribute checks in the hot training loop
        (called thousands of times per block).

        Mirrors old-arch behaviour: act-quant hooks, alg-ext, and optimized RTN
        use the plain ``block_forward`` instead of ``torch.compile``.
        """
        cached = self.__dict__.get("_resolved_block_forward")
        if cached is not None:
            return cached
        if (
            (self.config.is_act_quantize and (not self.config.act_dynamic or self.config.is_act_nv_fp))
            or self.enable_alg_ext
            or not getattr(self.config, "disable_opt_rtn", True)
        ):
            self._resolved_block_forward = block_forward
        elif self.compress_context.enable_torch_compile:
            compiled = self.__dict__.get("_compiled_block_forward")
            if compiled is None:
                compiled = compile_func(block_forward, device_manager.device)
                self._compiled_block_forward = compiled
            self._resolved_block_forward = compiled
        else:
            self._resolved_block_forward = block_forward
        return self._resolved_block_forward

    def _invalidate_block_forward_cache(self):
        """Clear the cached block forward function (call when block changes)."""
        self.__dict__.pop("_resolved_block_forward", None)
        self.__dict__.pop("_compiled_block_forward", None)

    def prepare_run(self, compressor: Any) -> None:
        """Model-level preparation (called once before block iteration starts)."""
        return

    def finalize_run(self, compressor: Any) -> None:
        """Model-level teardown (called once after all blocks are processed).

        Must be idempotent – the Compressor calls this inside a ``try/finally``.
        """
        return
