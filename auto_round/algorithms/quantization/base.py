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
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from auto_round.algorithms.composer import AlgorithmComposer, BlockContext

from auto_round.algorithms.base import BaseAlgorithm
from auto_round.algorithms.quantization.config import QuantizationConfig
from auto_round.data_type import QUANT_FUNC_WITH_DTYPE
from auto_round.logger import logger
from auto_round.utils import (
    check_to_quantized,
    clear_memory,
    convert_module_to_hp_if_necessary,
    set_module,
)
from auto_round.utils.device_manager import device_manager
from auto_round.wrapper import WrapperLinear


# TODO wenhuach annotate this class and functions clearly with details
class BaseQuantizer(BaseAlgorithm):
    """Base class for terminal weight-compression algorithms in a QuantizationPipeline.
    Developers adding a new quantization algorithm should inherit from this
    class and override at minimum :meth:`quantize_block`.
    Lifecycle hooks to override as needed:
        - :meth:`prepare_run`                  – model-level setup (once before all blocks)
        - :meth:`register_fp_input_forward_hooks` – register act-calib hooks
        - :meth:`quantize_block`               – **must override**: quantize a single block
        - :meth:`quantize_layer_outside_block` – quantize layers outside blocks
        - :meth:`finalize_run`                 – model-level teardown (once after all blocks)
    """

    def __init__(self, config: QuantizationConfig) -> None:
        super().__init__(config)
        # Whether to feed quantized-block outputs as inputs to the next block.
        # Subclasses that support cascaded quantized-input (e.g. SignRoundQuantizer)
        # override this from their config.  Defaults to False for zero-shot algorithms
        # (RTN) where activations are not used during weight optimization.
        self.enable_quanted_input = getattr(config, "enable_quanted_input", False)

    def can_compile_block_forward(self):  # TODO support compile block
        return True

    # ── Calibration hook registration ─────────────────────────────────────────
    def register_fp_input_forward_hooks(self, block: torch.nn.Module) -> list:
        """Register hooks that fire during the reference (FP-input) block forward.
        Subclasses override to add statistics collection hooks (e.g. imatrix).
        Returns a list of hook handles that the caller must remove when done.
        Note: act_max hooks are registered by the Compressor, not by the quantizer.
        """
        return []

    def register_qinput_forward_hooks(self, block: torch.nn.Module) -> list:
        """Register hooks that fire during the quantized-input block forward.
        Used when act-calib policy requires collecting stats from quantized
        activations. Returns a list of hook handles that the caller must remove.
        Note: act_max hooks are registered by the Compressor, not by the quantizer.
        """
        return []

    # ── Embedding quantization ────────────────────────────────────────────────
    @torch.inference_mode()
    def quantize_embedding_layer(self) -> bool:
        """Quantize all embedding layers marked for quantization in the model.
        Iterates modules, applies the appropriate quantization function based on
        bits / group_size / dtype, and writes quantized weights + scale/zp back
        onto the module.
        Note:
            Most schemes do **not** quantize embeddings. Currently only GGUF
            formats require this. Subclasses rarely need to override.
        Returns:
            bool: ``True`` if at least one embedding layer was quantized.
        """
        is_quantized = False
        for name, module in self.model_context.model.named_modules():
            if not isinstance(module, torch.nn.Embedding):
                continue
            if not check_to_quantized(module):
                continue
            is_quantized = True
            bits = getattr(module, "bits", None)
            group_size = getattr(module, "group_size", None)
            sym = getattr(module, "sym", None)
            data_type = getattr(module, "data_type", None)
            super_bits = getattr(module, "super_bits", None)
            super_group_size = getattr(module, "super_group_size", None)
            scale_dtype = self.scale_dtype
            quant_dtype = data_type
            if quant_dtype not in QUANT_FUNC_WITH_DTYPE:
                quant_dtype = f"{quant_dtype}_{'sym' if sym else 'asym'}"
            if not hasattr(self, "iters") or self.iters <= 0:  # pylint: disable=E1101
                tmp_dtype = "rtn_" + quant_dtype
                if tmp_dtype in QUANT_FUNC_WITH_DTYPE:
                    quant_dtype = tmp_dtype
            quant_func = QUANT_FUNC_WITH_DTYPE[quant_dtype]
            # float32 is used in RTN scale search; avoids caching a bf16 copy.
            weight_dtype = torch.float32 if super_group_size is not None else module.weight.dtype
            quant_kwargs = {
                "bits": bits,
                "group_size": group_size,
                "super_bits": super_bits,
                "super_group_size": super_group_size,
                "scale_dtype": scale_dtype,
            }
            try:
                weight, scale, zp = quant_func(
                    module.weight.to(dtype=weight_dtype, device=device_manager.device),
                    **quant_kwargs,
                )
            except torch.OutOfMemoryError:
                cuda_error_msg = traceback.format_exc()
                try:
                    logger.error(cuda_error_msg)
                    logger.warning("falling back to CPU")
                    weight, scale, zp = quant_func(module.weight.to("cpu"), **quant_kwargs)
                except Exception:
                    raise
            module.weight.data.copy_(weight.cpu())
            for param_name, val in zip(["scale", "zp"], [scale, zp]):
                if isinstance(val, dict):
                    for k, v in val.items():
                        setattr(module, k if k == "scale" else f"w_{k}", v.cpu())
                elif isinstance(val, torch.Tensor):
                    setattr(module, param_name, val.cpu())
                else:
                    setattr(module, param_name, val)
            del weight, scale, zp
            clear_memory()
        return is_quantized

    # ── Abstract quantization interface ───────────────────────────────────────
    def quantize_block(
        self,
        block: "torch.nn.Module",
        fp_inputs: "list[torch.Tensor] | dict",
        input_others: dict,
        fp_outputs: "list[torch.Tensor]",
        q_inputs: "list[torch.Tensor] | None",
        block_ctx: "BlockContext",
        valid_token_mask: "list[torch.Tensor] | None" = None,
        **kwargs,
    ) -> dict:
        """Apply the quantization algorithm to a prepared block.
        This is the **pure-algorithm** entry point called by the Compressor after
        all infrastructure work (device placement, data collection, act-max hook
        registration, DDP setup) has been completed.
        Args:
            block:            The transformer block module to quantize.
            fp_inputs:        FP calibration inputs (list[Tensor] or dict for diffusion).
            input_others:     Auxiliary kwargs (attention_mask, position_ids, …).
            fp_outputs:       FP reference outputs used as quantization targets.
            q_inputs:         Quantized inputs from the previous block, or ``None``.
            block_ctx:        Per-block pipeline context (:class:`BlockContext`).
            valid_token_mask: Per-sample boolean masks ``[1, seq_len]``; ``None`` if
                              no padding masking is needed.
            **kwargs:         Reserved for future parameters.
        Returns:
            dict: Best quantization parameters found, or ``{}`` if not applicable.
        Raises:
            NotImplementedError: Subclasses must override this method.
        """
        raise NotImplementedError("quantize_block must be implemented in subclasses of BaseQuantizer")

    def quantize_layer_outside_block(
        self,
        layer: "torch.nn.Module",
        fp_input: "list[torch.Tensor] | None" = None,
        q_input: "list[torch.Tensor] | None" = None,
        disable_opt_rtn: "bool | None" = None,
        valid_token_mask: "list[torch.Tensor] | None" = None,
    ) -> None:
        """Quantize a single layer outside a transformer block using RTN fallback.
        Args:
            layer:            The layer module to quantize.  Must have a
                              ``global_name`` attribute for model re-insertion.
            fp_input:         Optional FP calibration inputs; unused in base RTN.
            q_input:          Optional quantized activations; unused in base RTN.
            disable_opt_rtn:  ``True`` skips optimized-RTN scale/zp search.
                              ``None`` defers to ``self.config.disable_opt_rtn``.
            valid_token_mask: Per-sample masks; unused in base RTN.
        """
        self._quantize_layer_via_rtn(layer, disable_opt_rtn=disable_opt_rtn)

    @torch.no_grad()
    def _quantize_layer_via_rtn(self, layer: "torch.nn.Module", disable_opt_rtn: "bool | None" = None) -> None:
        """Quantize one layer with RTN (with optional optimized scale/zp search)."""
        layer_name = layer.global_name
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

    def dispatch_block(self, block: "torch.nn.Module", input_ids, input_others: dict):
        """Place a block on the correct device(s) for quantization.
        Default: move to primary device.
        Returns ``(block, card_0_in_high_risk, loss_device)``.
        Subclasses override for multi-GPU tensor-parallel dispatch.
        """
        block = block.to(device_manager.device)
        return block, False, device_manager.device


    # ── Lifecycle hooks ───────────────────────────────────────────────────────
    def prepare_run(self, composer: "AlgorithmComposer" = None) -> None:
        """Model-level preparation (called once before block iteration starts)."""
        return

    def finalize_run(self) -> None:
        """Model-level teardown (called once after all blocks are processed).
        Must be idempotent — the Compressor calls this inside a ``try/finally``.
        """
        return
