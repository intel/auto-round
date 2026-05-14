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
from functools import partial
from typing import Callable, Union

import torch
import transformers
from torch import autocast

from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer
from auto_round.data_type.gguf import quant_tensor_gguf_asym_dq, quant_tensor_gguf_sym_dq
from auto_round.data_type.int import quant_tensor_asym, quant_tensor_sym, search_scales
from auto_round.data_type.mxfp import quant_mx, search_mx_scale
from auto_round.data_type.nvfp import nv_fp4, search_nvfp4_scale
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size
from auto_round.logger import logger
from auto_round.utils import check_to_quantized, compile_func
from auto_round.wrapper import WrapperLinear, wrapper_block


def _named_wrapper_block(wrapper_cls, name: str):
    wrapped = partial(wrapper_block, wrapper_cls=wrapper_cls)
    wrapped.__name__ = name
    return wrapped


class SignRoundOptimizedWrapperLinear(WrapperLinear):
    minmax_scale_bound = (0.0, 2.0)

    def _init_tuning_params_and_quant_func(self):
        super()._init_tuning_params_and_quant_func()

        orig_weight = getattr(self.orig_layer, "get_weight", lambda: self.orig_layer.weight)()
        if type(self.orig_layer) == transformers.pytorch_utils.Conv1D:
            orig_weight = orig_weight.t()
        weight_reshape, _, _ = reshape_pad_tensor_by_group_size(orig_weight.data, self.orig_layer.group_size)
        if hasattr(self.orig_layer, "imatrix"):
            imatrix = self.orig_layer.imatrix.reshape(1, -1)
            imatrix = reshape_pad_tensor_by_group_size(imatrix, self.orig_layer.group_size, val=1e-5)[0].view(1, -1)
            imatrix = imatrix.expand(weight_reshape.numel() // imatrix.numel(), -1)
            imatrix = imatrix.reshape(weight_reshape.shape).to(orig_weight.device)
        else:
            imatrix = 1.0

        if self.orig_layer.data_type.startswith("int"):
            self.init_scale = search_scales(weight_reshape, self.orig_layer.bits, imatrix)
            self.init_scale = torch.where(
                self.init_scale < 0,
                torch.clamp(self.init_scale, max=-self.q_scale_thresh),
                torch.clamp(self.init_scale, min=self.q_scale_thresh),
            )
            self.weight_quant_func = quant_tensor_sym
        elif self.orig_layer.data_type.startswith("mx"):
            self.init_scale = search_mx_scale(weight_reshape, self.orig_layer.bits, imatrix)
            self.weight_quant_func = quant_mx
        elif self.orig_layer.data_type.startswith("nv"):
            self.init_scale = search_nvfp4_scale(weight_reshape, self.orig_layer.bits, imatrix)
            self.weight_quant_func = nv_fp4
        else:
            raise ValueError(f"unsupported SignRound optimized data type: {self.orig_layer.data_type}")

        self.data_type = self.orig_layer.data_type
        if hasattr(self.orig_layer, "imatrix"):
            delattr(self.orig_layer, "imatrix")
        if self.enable_torch_compile:
            self.weight_quant_func = compile_func(self.weight_quant_func, self.device)


class SignRoundDQWrapperLinear(WrapperLinear):
    minmax_scale_bound = (0.5, 1.5)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_scale = None
        self.prev_wmin = None
        self.prev_d_scale = None
        self.prev_d_wmin = None

    def _init_tuning_params_and_quant_func(self):
        super()._init_tuning_params_and_quant_func()
        if hasattr(self.orig_layer, "super_group_size") and self.orig_layer.super_group_size is not None:
            self.weight_quant_func = (
                quant_tensor_gguf_asym_dq if self.orig_layer.data_type == "int_asym_dq" else quant_tensor_gguf_sym_dq
            )
        elif self.orig_layer.sym:
            self.weight_quant_func = quant_tensor_sym
        else:
            self.weight_quant_func = quant_tensor_asym
        self.data_type = self.orig_layer.data_type
        if self.enable_act_quant:
            self.act_quant_func = (
                quant_tensor_gguf_asym_dq
                if self.orig_layer.act_data_type == "int_asym_dq"
                else quant_tensor_gguf_sym_dq
            )
            if self.enable_torch_compile:
                self.act_quant_func = compile_func(self.act_quant_func, self.device)
            self._init_params("act_max_scale", torch.float32, (1), 1.0, not self.orig_layer.act_dynamic)
        if self.enable_torch_compile:
            self.weight_quant_func = compile_func(self.weight_quant_func, self.device)

    def _qdq_weight(self, value, min_scale, max_scale):
        weight_q, scale, zp = super()._qdq_weight(value, min_scale, max_scale)
        if isinstance(scale, dict) and "d_scale" in scale and self.prev_scale is None:
            self.prev_scale = scale["scale"]
            self.prev_d_scale = scale["d_scale"]
            if isinstance(zp, dict):
                self.prev_wmin = zp["wmin"]
                self.prev_d_wmin = zp["d_wmin"]
        elif self.prev_scale is None:
            self.prev_scale = scale
        return weight_q, scale, zp

    def _extra_quant_kwargs(self):
        return {
            "prev_scale": self.prev_scale,
            "prev_wmin": self.prev_wmin,
            "prev_d_scale": self.prev_d_scale,
            "prev_d_wmin": self.prev_d_wmin,
            "iter": getattr(self, "cur_iter", None),
        }


class SignRoundV2Quantizer(SignRoundQuantizer):
    """SignRound variant using the open algorithm-extension path in the new architecture."""

    def __init__(self, config: SignRoundConfig):
        super().__init__(config)
        self._use_outlier_suppressed_loss = False
        logger.info("using algorithm extension for quantization.")

        if (
            self.sym
            and self.super_group_size is None
            and (self.data_type.startswith("int") or self.data_type.startswith("mx") or self.data_type.startswith("nv"))
        ):
            if self.bits > 2 and not (self.data_type.startswith("mx") or self.data_type.startswith("nv")):
                logger.warning_once(
                    "algorithm extension has only undergone limited validation on "
                    "W2A16,INT4, MXFP4 and NVFP4; use with caution."
                )
            self._use_outlier_suppressed_loss = True
            self.wrapper_block = _named_wrapper_block(SignRoundOptimizedWrapperLinear, "wrapper_block")

        if self.data_type.endswith("dq"):
            self.wrapper_block = _named_wrapper_block(SignRoundDQWrapperLinear, "dq_wrapper_block")

    def _get_loss(
        self,
        output_q: torch.Tensor,
        current_output: torch.Tensor,
        indices: torch.Tensor,
        mse_loss: Callable,
        device: Union[str, torch.device] = "cpu",
    ):
        if self._use_outlier_suppressed_loss:
            loss_diff = torch.abs(output_q - current_output)
            flat_diff = loss_diff.view(-1)
            topk = max(1, int(flat_diff.numel() / 1000))
            _, top_indices = torch.topk(torch.abs(flat_diff), topk)
            mask = torch.zeros_like(flat_diff, dtype=torch.bool)
            mask[top_indices] = True
            mask = (~mask).view_as(loss_diff)

            autocast_ctx = (
                autocast(device_type=str(device).split(":")[0], dtype=self.amp_dtype) if self.amp else nullcontext()
            )
            if self.attention_mask:
                tmp_attention_mask = [self.attention_mask[i] for i in indices]
                tmp_attention_mask = torch.cat(tmp_attention_mask, dim=0).to(device)
                tmp_attention_mask.unsqueeze_(-1)
                with autocast_ctx:
                    return torch.mean(
                        (
                            torch.abs(output_q.to(torch.float32) - current_output.to(torch.float32))
                            * tmp_attention_mask
                            * mask
                        )
                        ** 2
                    )

            with autocast_ctx:
                return torch.mean(
                    (torch.abs(output_q.to(torch.float32) - current_output.to(torch.float32)) * mask) ** 2
                )
        return super()._get_loss(output_q, current_output, indices, mse_loss, device)

    def _register_act_max_hook(self, model):
        hook_handles = super()._register_act_max_hook(model)

        is_wint4aint4 = ("int4" in self.act_data_type or ("int" in self.act_data_type and self.act_bits == 4)) and (
            "int4" in self.data_type or ("int" in self.data_type and self.bits == 4)
        )
        if is_wint4aint4:
            return hook_handles

        def get_imatrix_hook(module, input, output):
            input = input[0] if isinstance(input, (tuple, list)) else input
            flattened = input.reshape(-1, input.shape[-1]).to(torch.float32)
            squared = torch.sum(torch.pow(flattened, 2), dim=0).to(torch.float32)

            if not hasattr(module, "imatrix"):
                module.imatrix = squared
            else:
                module.imatrix += squared.to(module.imatrix.device)

        for _, module in model.named_modules():
            if isinstance(module, self.supported_types) and check_to_quantized(module):
                hook_handles.append(module.register_forward_hook(get_imatrix_hook))
        return hook_handles
