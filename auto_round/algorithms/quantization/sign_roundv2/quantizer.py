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

from contextlib import contextmanager, nullcontext
from functools import partial
from typing import Callable, Union

import torch
import transformers
from torch import autocast

from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig, SignRoundV2Config
from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer
from auto_round.algorithms.registry import register_pipeline_member
from auto_round.data_type.gguf import (
    double_quant_tensor_sym_rtn,
    quant_tensor_gguf_asym_dq,
    quant_tensor_gguf_sym_dq,
    search_gguf_scale_min_asym,
    search_gguf_scale_min_sym,
)
from auto_round.data_type.int import quant_tensor_asym, quant_tensor_sym
from auto_round.data_type.utils import (
    get_optimized_quant_func,
    reshape_imatrix_for_weight,
    reshape_pad_tensor_by_group_size,
    revert_tensor_by_pad,
    round_ste,
    search_optimized_init_scale,
)
from auto_round.logger import logger
from auto_round.utils import check_to_quantized, compile_func, get_reciprocal
from auto_round.wrapper import WrapperLinear, wrapper_block


def _dq_asym_qdq(tensor, scale, wmin, bits, group_size, v=0):
    """Pure asym double-quant qdq math given precomputed scale/wmin (compilable)."""
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    orig_dtype = tensor.dtype
    tensor = tensor.to(torch.float32)
    maxq = int(2.0**bits) - 1
    inverse_scale = get_reciprocal(scale)
    int_w = torch.clamp(round_ste((tensor + wmin) * inverse_scale + v), 0, maxq)
    qdq = (scale * int_w - wmin).to(orig_dtype)
    qdq = revert_tensor_by_pad(qdq, orig_shape=orig_shape, pad_len=pad_len)
    return qdq


def _dq_sym_qdq(tensor, scale, bits, v=0):
    """Pure sym double-quant qdq math given precomputed scale (compilable)."""
    from auto_round.export.export_to_gguf.config import GGML_QUANT_SIZES, QK_K

    group_size = 16
    super_group_size = 16
    maxq = int(2.0 ** (bits - 1))
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    orig_dtype = tensor.dtype
    tensor = tensor.to(torch.float32)
    ggml_type = f"q{bits}_k"
    block_size, _ = GGML_QUANT_SIZES[ggml_type]
    n_blocks = tensor.nelement() // block_size
    tensor = tensor.reshape(n_blocks, super_group_size, QK_K // super_group_size)
    if isinstance(v, torch.Tensor):
        v_r, _, _ = reshape_pad_tensor_by_group_size(v, group_size)
        v_r = v_r.reshape(n_blocks, super_group_size, QK_K // super_group_size)
    else:
        v_r = v
    zp = torch.full_like(scale, maxq)
    inverse_scale = get_reciprocal(scale)
    int_w = round_ste(tensor * inverse_scale + v_r).clip(-maxq, maxq - 1) + maxq
    qdq = (scale * (int_w - zp)).to(orig_dtype)
    qdq = revert_tensor_by_pad(qdq, orig_shape=orig_shape, pad_len=pad_len)
    return qdq


def _named_wrapper_block(wrapper_cls, name: str):
    wrapped = partial(wrapper_block, wrapper_cls=wrapper_cls)
    wrapped.__name__ = name
    return wrapped


def _dq_asym_group_size(bits: int) -> int:
    """Group size used by the GGUF asym double-quant path (16 for 2-bit, else 32)."""
    return 16 if bits == 2 else 32


class SignRoundOptimizedWrapperLinear(WrapperLinear):
    minmax_scale_bound = (0.0, 2.0)

    def _init_tuning_params_and_quant_func(self):
        super()._init_tuning_params_and_quant_func()

        layer = self.orig_layer
        data_type = layer.data_type
        weight_reshape = self._prepare_init_scale_weight()
        imatrix = reshape_imatrix_for_weight(getattr(layer, "imatrix", None), weight_reshape, layer.group_size)

        self.init_scale = search_optimized_init_scale(
            weight_reshape, data_type, layer.bits, imatrix, self.q_scale_thresh
        )
        self.weight_quant_func = get_optimized_quant_func(data_type)
        if self.init_scale is None or self.weight_quant_func is None:
            raise ValueError(
                f"SignRound optimized path does not support data_type={data_type!r}; "
                "expected a symmetric int / mx / nv type."
            )

        self.data_type = data_type
        if hasattr(layer, "imatrix"):
            del layer.imatrix
        if self.enable_torch_compile:
            self.weight_quant_func = compile_func(self.weight_quant_func, self.device)

    def _prepare_init_scale_weight(self) -> torch.Tensor:
        """Return the group-reshaped weight that seeds the init-scale search.

        Reproduces the layout the quant func sees at tuning time: the (optionally
        transposed) FP weight grouped to ``[-1, group_size]``. When AWQ ran in
        ``clip_as_init`` mode and stored a per-group ``awq_clip_max`` on the layer,
        the weight is clamped to that range first so the searched ``init_scale``
        already reflects the AWQ clip (``max_scale`` then tunes a coefficient on
        top of it).
        """
        layer = self.orig_layer
        weight = layer.get_weight() if hasattr(layer, "get_weight") else layer.weight
        if isinstance(layer, transformers.pytorch_utils.Conv1D):
            weight = weight.t()
        weight_reshape, _, _ = reshape_pad_tensor_by_group_size(weight.data, layer.group_size)

        clip_max = getattr(layer, "awq_clip_max", None)
        if clip_max is not None:
            clip_max = clip_max.reshape(-1, 1).to(weight_reshape.device, weight_reshape.dtype)
            if clip_max.shape[0] == weight_reshape.shape[0]:
                weight_reshape = torch.clamp(weight_reshape, -clip_max, clip_max)
            else:
                logger.warning_once(
                    "SignRoundV2: ignoring awq_clip_max with shape %s incompatible with "
                    "grouped weight shape %s." % (tuple(clip_max.shape), tuple(weight_reshape.shape))
                )
        return weight_reshape


class SignRoundDQWrapperLinear(WrapperLinear):
    minmax_scale_bound = (0.5, 1.5)

    def __init__(self, *args, **kwargs) -> None:
        if "enable_minmax_tuning" in kwargs:
            logger.warning_once("disable minmax tuning for a little better accuracy and lower cost")
            kwargs["enable_minmax_tuning"] = False
        super().__init__(*args, **kwargs)
        self.prev_scale = None
        self.prev_wmin = None
        self.prev_d_scale = None
        self.prev_d_wmin = None

    def _init_tuning_params_and_quant_func(self):
        super()._init_tuning_params_and_quant_func()
        # The double-quant search path is data-dependent and kept un-compiled,
        # while ``weight_quant_func`` is the compilable pure-math half.
        self._is_dq_path = False
        self._dq_kind = None
        self.search_func = None
        if hasattr(self.orig_layer, "super_group_size") and self.orig_layer.super_group_size is not None:
            self._is_dq_path = True
            if self.orig_layer.data_type == "int_asym_dq":
                self.search_func = search_gguf_scale_min_asym
                self.weight_quant_func = _dq_asym_qdq
                self._dq_kind = "asym"
            else:
                self.search_func = search_gguf_scale_min_sym
                self.weight_quant_func = _dq_sym_qdq
                self._dq_kind = "sym"
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

    @torch.no_grad()
    def _run_search(self, weight, v):
        from auto_round.export.export_to_gguf.config import GGML_QUANT_SIZES, QK_K

        bits = self.orig_layer.bits
        scale_dtype = self.orig_layer.scale_dtype
        imatrix = getattr(self.orig_layer, "imatrix", None)

        if self._dq_kind == "asym":
            group_size = _dq_asym_group_size(bits)
            t, _, _ = reshape_pad_tensor_by_group_size(weight.to(torch.float32), group_size)
            v_r = v
            if isinstance(v, torch.Tensor):
                v_r, _, _ = reshape_pad_tensor_by_group_size(v, group_size)
            scale, wmin, d_scale, d_wmin = self.search_func(
                t,
                bits=bits,
                scale_dtype=scale_dtype,
                imatrix=imatrix,
                split_num=1,
                v=v_r,
            )
            return {
                "scale": scale.clone(),
                "wmin": wmin.clone(),
                "d_scale": d_scale.clone(),
                "d_wmin": d_wmin.clone(),
            }

        group_size = 16
        super_group_size = 16
        t, _, _ = reshape_pad_tensor_by_group_size(weight.to(torch.float32), group_size)
        ggml_type = f"q{bits}_k"
        block_size, _ = GGML_QUANT_SIZES[ggml_type]
        n_blocks = t.nelement() // block_size
        t = t.reshape(n_blocks, super_group_size, QK_K // super_group_size)
        v_r = v
        if isinstance(v, torch.Tensor):
            v_r, _, _ = reshape_pad_tensor_by_group_size(v, group_size)
            v_r = v_r.reshape(n_blocks, super_group_size, QK_K // super_group_size)
        super_bits = 6 if bits == 3 else 8
        scale = self.search_func(t, bits, imatrix, scale_dtype, split_num=1, v=v_r)
        scale = scale.to(scale_dtype)
        scale = torch.where(torch.abs(scale) < 1e-30, torch.zeros_like(scale), scale)
        scale, d_scale = double_quant_tensor_sym_rtn(scale, super_bits)
        scale = scale.unsqueeze(-1)
        return {"scale": scale.clone(), "d_scale": d_scale.clone()}

    def _qdq_weight(self, value, min_scale, max_scale):
        if not self._is_dq_path:
            return super()._qdq_weight(value, min_scale, max_scale)

        if self.orig_layer.bits >= 16:
            return self.orig_layer.weight, None, None
        min_bound, max_bound = self.minmax_scale_bound
        min_scale.data.clamp_(min_bound, max_bound)
        max_scale.data.clamp_(min_bound, max_bound)
        weight = self.orig_layer.weight
        if weight.device.type == "meta":
            weight = self.orig_layer.get_weight().to(self.device)
        if isinstance(self.orig_layer, transformers.pytorch_utils.Conv1D):
            weight = weight.t()

        iter_v = getattr(self, "cur_iter", 0)
        need_search = (iter_v == 0) or (iter_v == -1) or (self.prev_scale is None)
        if need_search:
            params = self._run_search(weight, value)
            self.prev_scale = params["scale"]
            self.prev_d_scale = params["d_scale"]
            if self._dq_kind == "asym":
                self.prev_wmin = params["wmin"]
                self.prev_d_wmin = params["d_wmin"]
        else:
            params = {
                "scale": self.prev_scale.detach(),
                "d_scale": self.prev_d_scale.detach(),
            }
            if self._dq_kind == "asym":
                params["wmin"] = self.prev_wmin.detach()
                params["d_wmin"] = self.prev_d_wmin.detach()

        bits = self.orig_layer.bits
        if self._dq_kind == "asym":
            group_size = _dq_asym_group_size(bits)
            weight_q = self.weight_quant_func(
                weight,
                params["scale"],
                params["wmin"],
                bits,
                group_size,
                v=value,
            )
            scale_out = {"scale": params["scale"], "d_scale": params["d_scale"]}
            zp_out = {"wmin": params["wmin"], "d_wmin": params["d_wmin"]}
        else:
            weight_q = self.weight_quant_func(
                weight,
                params["scale"],
                bits,
                v=value,
            )
            scale_out = {"scale": params["scale"], "d_scale": params["d_scale"]}
            zp_out = torch.full_like(params["scale"], int(2.0 ** (bits - 1)))

        weight_q = weight_q.to(weight.dtype)
        if isinstance(self.orig_layer, transformers.pytorch_utils.Conv1D):
            weight_q = weight_q.t()
        return weight_q, scale_out, zp_out


@register_pipeline_member(SignRoundV2Config)
class SignRoundV2Quantizer(SignRoundQuantizer):
    """SignRound variant using the open algorithm-extension path in the new architecture."""

    def __init__(self, config: SignRoundConfig) -> None:
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
            if self.act_bits <= 4 or self.bits < 4:
                self._use_outlier_suppressed_loss = True
            else:
                self._use_outlier_suppressed_loss = False
            self.wrapper_block = _named_wrapper_block(SignRoundOptimizedWrapperLinear, "wrapper_block")

        if self.data_type.endswith("dq"):
            self.wrapper_block = _named_wrapper_block(SignRoundDQWrapperLinear, "dq_wrapper_block")

    def _get_loss(
        self,
        pred_output: torch.Tensor,
        ref_output: torch.Tensor,
        indices: torch.Tensor,
        mse_loss: Callable,
        device: Union[str, torch.device] = "cpu",
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
            if self.attention_mask:
                tmp_attention_mask = [self.attention_mask[i] for i in indices]
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

    @contextmanager
    def block_forward_hooks(self, ctx):
        with super().block_forward_hooks(ctx) as hook_handles:
            if not self._is_wint4aint4():
                hook_handles.extend(self._register_imatrix_hooks(ctx.block))
            yield hook_handles

    def _is_wint4aint4(self):
        return ("int4" in self.act_data_type or ("int" in self.act_data_type and self.act_bits == 4)) and (
            "int4" in self.data_type or ("int" in self.data_type and self.bits == 4)
        )

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
            if isinstance(module, self.supported_types) and check_to_quantized(module):
                handles.append(module.register_forward_hook(collect_imatrix))
        return handles
