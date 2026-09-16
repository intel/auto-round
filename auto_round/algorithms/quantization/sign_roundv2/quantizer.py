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

from contextlib import nullcontext
from typing import TYPE_CHECKING, Callable, Union

import torch
from torch import autocast

from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig, SignRoundV2Config
from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer
from auto_round.algorithms.registry import register_pipeline_member
from auto_round.data_type.base import canonical_data_type
from auto_round.logger import logger
from auto_round.utils import SUPPORTED_LAYER_TYPES, check_to_quantized

if TYPE_CHECKING:
    from auto_round.algorithms.composer import AlgorithmComposer


@register_pipeline_member(SignRoundV2Config)
class SignRoundV2Quantizer(SignRoundQuantizer):
    """SignRound variant using the open algorithm-extension path in the new architecture."""

    def __init__(self, config: SignRoundConfig) -> None:
        super().__init__(config)
        self._use_outlier_suppressed_loss = False
        logger.info("using algorithm extension for quantization.")

    def prepare_run(self, composer: "AlgorithmComposer" = None) -> None:
        """Model-level setup: initialise scheme-dependent state once before block iteration."""
        super().prepare_run(composer=composer)

        canonical_dtype = canonical_data_type(self.scheme.data_type)
        optimized_families = {"int", "mx_fp4", "mx_int8", "nv_fp4", "nvfp4_v2"}
        is_optimized_family = canonical_dtype in optimized_families
        if self.scheme.sym and self.scheme.super_group_size is None and is_optimized_family:
            if self.scheme.bits > 2 and canonical_dtype == "int":
                logger.warning_once(
                    "algorithm extension has only undergone limited validation on "
                    "W2A16,INT4, MXFP4 and NVFP4; use with caution."
                )
            if self.scheme.act_bits <= 4 or self.scheme.bits < 4:
                self._use_outlier_suppressed_loss = True
            else:
                self._use_outlier_suppressed_loss = False
        # QDQ ownership stays with the generic wrapper.  Datatype adapters
        # select optimized and double-quant behavior from the layer request;
        # SignRoundV2 only owns its loss policy.

    def can_compile_block_forward(self):
        return False

    def _get_loss(
        self,
        pred_output: torch.Tensor,
        ref_output: torch.Tensor,
        indices: torch.Tensor,
        mse_loss: Callable,
        device: Union[str, torch.device] = "cpu",
        valid_token_mask: list[torch.Tensor] | None = None,
    ):
        if self._use_outlier_suppressed_loss:
            loss_diff = torch.abs(pred_output - ref_output)
            flat_diff = loss_diff.view(-1)
            topk = max(1, int(flat_diff.numel() / 1000))
            _, top_indices = torch.topk(torch.abs(flat_diff), topk)
            mask = torch.zeros_like(flat_diff, dtype=torch.bool)
            mask[top_indices] = True
            mask = (~mask).view_as(loss_diff)

            autocast_ctx = (
                autocast(device_type=str(device).split(":")[0], dtype=self.amp_dtype) if self.amp else nullcontext()
            )
            if valid_token_mask:
                tmp_attention_mask = [valid_token_mask[i] for i in indices]
                tmp_attention_mask = torch.cat(tmp_attention_mask, dim=0).to(device)
                tmp_attention_mask.unsqueeze_(-1)
                with autocast_ctx:
                    return torch.mean(
                        (
                            torch.abs(pred_output.to(torch.float32) - ref_output.to(torch.float32))
                            * tmp_attention_mask
                            * mask
                        )
                        ** 2
                    )

            with autocast_ctx:
                return torch.mean((torch.abs(pred_output.to(torch.float32) - ref_output.to(torch.float32)) * mask) ** 2)
        return super()._get_loss(pred_output, ref_output, indices, mse_loss, device)

    def register_fp_input_forward_hooks(self, block):
        """Register FP-input hooks: imatrix."""
        handles = super().register_fp_input_forward_hooks(block)
        if not self._is_wint4aint4():
            handles.extend(self._register_imatrix_hooks(block))
        return handles

    def _is_wint4aint4(self):
        return (
            "int4" in self.scheme.act_data_type or ("int" in self.scheme.act_data_type and self.scheme.act_bits == 4)
        ) and ("int4" in self.scheme.data_type or ("int" in self.scheme.data_type and self.scheme.bits == 4))

    def _register_imatrix_hooks(self, model):
        def collect_imatrix(module, input, output):
            input = input[0] if isinstance(input, (tuple, list)) else input
            flattened = input.reshape(-1, input.shape[-1]).to(torch.float32)
            squared = torch.sum(torch.pow(flattened, 2), dim=0).to(torch.float32)

            if not hasattr(module, "imatrix"):
                module.imatrix = squared
                return
            module.imatrix += squared.to(module.imatrix.device)

        handles = []
        for _, module in model.named_modules():
            if isinstance(module, SUPPORTED_LAYER_TYPES) and check_to_quantized(module):
                handles.append(module.register_forward_hook(collect_imatrix))
        return handles
