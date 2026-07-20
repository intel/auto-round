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

import torch

from auto_round.algorithms.quantization.base import BaseQuantizer
from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig, RTNConfig
from auto_round.algorithms.registry import register_pipeline_member
from auto_round.logger import logger
from auto_round.utils import (
    check_to_quantized,
    get_module,
    set_module,
)


@register_pipeline_member(RTNConfig)
class RTNQuantizer(BaseQuantizer):

    def __init__(self, config: RTNConfig) -> None:
        BaseQuantizer.__init__(self, config)

    @torch.no_grad()
    def quantize_block(
        self,
        block,
        fp_inputs,
        input_others,
        fp_outputs,
        q_inputs,
        block_ctx,
        valid_token_mask=None,
        **kwargs,
    ) -> dict:
        """Apply zero-shot RTN quantization to a block.

        Args:
            block: The transformer block module to quantize.
            fp_inputs: FP calibration inputs for this block (list[Tensor] or dict
                for diffusion models).
            input_others: Auxiliary kwargs passed to the block forward
                (e.g. attention_mask, position_ids).
            fp_outputs: FP reference outputs of the block used as quantization
                targets (list[Tensor]).
            q_inputs: Quantized inputs from the previous block, or ``None`` when
                cascaded quantized-input is disabled.
            block_ctx: Per-block pipeline context (BlockContext).
            valid_token_mask: Per-sample boolean/int masks of shape
                ``[1, seq_len]`` indicating valid (non-padding) token positions.
                ``1`` means valid, ``0`` means padding. ``None`` if no masking
                is needed (e.g. standard string datasets without padding).
            **kwargs: Reserved for forward-compatibility with future parameters.

        Returns:
            dict: Empty dict — zero-shot RTN has no tunable parameters to track.
        """

        for _name, m in block.named_modules():
            if hasattr(m, "global_name") and check_to_quantized(m):
                self.quantize_layer(m.global_name)
        return {}

    @torch.no_grad()
    def quantize_layer(self, name: str, dtype: torch.dtype = None) -> None:
        if dtype is not None:
            layer = get_module(self.model, name)
            set_module(self.model, name, layer.to(dtype))
        self._quantize_layer_via_rtn(name, disable_opt_rtn=True)


@register_pipeline_member(OptimizedRTNConfig)
class OptimizedRTNQuantizer(RTNQuantizer):

    def __init__(self, config: RTNConfig) -> None:
        BaseQuantizer.__init__(self, config)
        if (
            self.scheme is not None
            and self.scheme.data_type
            and ("nv_fp" in self.scheme.data_type or "mx_fp" in self.scheme.data_type)
        ):
            logger.warning_once(
                "opt-rtn does not support NVFP or MXFP. It behaves the same as RTN but is much slower. "
                "Please use RTN instead."
            )

    def is_support_compile_block(self):
        return False

    def register_fp_input_forward_hooks(self, block):
        """Register FP-input hooks: imatrix."""
        handles = super().register_fp_input_forward_hooks(block)
        handles.extend(self._register_imatrix_hooks(block, with_count=True))
        return handles

    def _register_imatrix_hooks(self, model, *, with_count: bool = False):
        def collect_imatrix(module, input, output):
            input = input[0] if isinstance(input, (tuple, list)) else input
            flattened = input.reshape(-1, input.shape[-1]).to(torch.float32)
            squared = torch.sum(torch.pow(flattened, 2), dim=0).to(torch.float32)

            if not hasattr(module, "imatrix"):
                module.imatrix = squared
                if with_count:
                    module.imatrix_cnt = input.shape[0]
                return
            module.imatrix += squared.to(module.imatrix.device)
            if with_count:
                module.imatrix_cnt += input.shape[0]

        handles = []
        for _, module in model.named_modules():
            if check_to_quantized(module):
                handles.append(module.register_forward_hook(collect_imatrix))
        return handles

    @torch.no_grad()
    def quantize_block(
        self,
        block,
        fp_inputs,
        input_others,
        fp_outputs,
        q_inputs,
        block_ctx,
        valid_token_mask=None,
        **kwargs,
    ):
        """Apply imatrix-informed RTN quantization to a block.

        Args:
            block: The transformer block module to quantize.
            fp_inputs: FP calibration inputs for this block (list[Tensor] or dict
                for diffusion models).
            input_others: Auxiliary kwargs passed to the block forward
                (e.g. attention_mask, position_ids).
            fp_outputs: FP reference outputs of the block used as quantization
                targets (list[Tensor]).
            q_inputs: Quantized inputs from the previous block, or ``None`` when
                cascaded quantized-input is disabled.
            block_ctx: Per-block pipeline context (BlockContext).
            valid_token_mask: Per-sample boolean/int masks of shape
                ``[1, seq_len]`` indicating valid (non-padding) token positions.
                ``1`` means valid, ``0`` means padding. ``None`` if no masking
                is needed (e.g. standard string datasets without padding).
                Currently unused in imatrix-RTN; reserved for future use.
            **kwargs: Reserved for forward-compatibility with future parameters.
        """
        # Normalize imatrix and quantize layers
        for name, m in block.named_modules():
            if hasattr(m, "imatrix"):
                m.imatrix /= m.imatrix_cnt
            if hasattr(m, "global_name") and check_to_quantized(m):
                self.quantize_layer_outside_block(m)
