# Copyright (c) 2025 Intel Corporation
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

import logging
import types
from contextlib import nullcontext
from functools import lru_cache, partial
from typing import Any, Callable, Union

import torch
import transformers
from torch import autocast
from torch.functional import F

from auto_round import AutoRound
from auto_round.compressors.utils import check_need_act_calibration, is_nv_fp, is_wint4aint4
from auto_round.data_type.int import search_scales
from auto_round.data_type.mxfp import MXFP_FORMAT_CACHE, quant_element
from auto_round.data_type.nvfp import FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX, ref_nvfp4_quant
from auto_round.data_type.utils import floor_ste, reshape_pad_tensor_by_group_size, revert_tensor_by_pad, round_ste
from auto_round.logger import logger
from auto_round.utils import SUPPORTED_LAYER_TYPES, check_to_quantized, compile_func, get_reciprocal, set_module
from auto_round.wrapper import NORM_MAPPING, WrapperLinear

__all__ = ["wrapper_autoround"]


def wrapper_autoround(cls: AutoRound):
    cls._register_act_max_hook = types.MethodType(_register_act_max_hook_ext, cls)
    if (
        cls.sym
        and cls.enable_alg_ext
        and cls.super_group_size is None
        and ((cls.data_type.startswith("int")) or cls.data_type.startswith("mx") or cls.data_type.startswith("nv"))
    ):
        if cls.bits > 2 and (not cls.data_type.startswith("mx") or not cls.data_type.startswith("nv")):
            logger.warning_once(
                "algorithm extension has only undergone limited validation on "
                "W2A16,INT4, MXFP4 and NVFP4; use with caution."
            )
        cls._get_loss = types.MethodType(_get_loss_ext, cls)
        setattr(cls, "wrapper_block", wrapper_block_v2)
    if cls.data_type.endswith("dq"):
        setattr(cls, "wrapper_block", dq_wrapper_block)


def get_abs_top_percent_mask(x: torch.Tensor, percent: float = 1.0):
    """
    Return a mask for the top `percent` absolute values in x and its inverse.

    Args:
        x (torch.Tensor): Input tensor.
        percent (float): Percentage of elements to select (0~100).

    Returns:
        mask (torch.BoolTensor): True for top `percent` abs elements.
        inv_mask (torch.BoolTensor): Inverse of mask.
    """
    flat = x.view(-1)
    k = max(1, int(flat.numel() * percent / 1000))
    _, idx = torch.topk(torch.abs(flat), k)

    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[idx] = True
    mask = mask.view_as(x)
    return mask, ~mask


def _get_loss_ext(
    self: AutoRound,
    output_q: torch.Tensor,
    current_output: torch.Tensor,
    indices: torch.Tensor,
    mse_loss: Callable,
    device: Union[str, torch.device] = "cpu",
):
    _, mask = get_abs_top_percent_mask(torch.abs(output_q - current_output))
    autocast_ctx = nullcontext() if self.amp else autocast(device_type=str(device).split(":")[0], dtype=self.amp_dtype)
    if self.attention_mask:
        tmp_attention_mask = [self.attention_mask[i] for i in indices]
        tmp_attention_mask = torch.cat(tmp_attention_mask, dim=0).to(device)
        tmp_attention_mask.unsqueeze_(-1)

        with autocast_ctx:
            loss = torch.mean(
                (torch.abs(output_q.to(torch.float32) - current_output.to(torch.float32)) * tmp_attention_mask * mask)
                ** 2
            )
    else:
        with autocast_ctx:
            loss = torch.mean((torch.abs(output_q.to(torch.float32) - current_output.to(torch.float32)) * mask) ** 2)

    return loss


def quant_tensor_sym(
    tensor,
    bits=4,
    group_size=-1,
    v=0,
    min_scale=1.0,
    max_scale=1.0,
    scale_dtype=torch.float16,
    tensor_min=None,
    tensor_max=None,
    q_scale_thresh=1e-5,
    init_scale=None,
    **kwargs,
):
    """Quantize and de-quantize tensor symmetrically (full-range, llama.cpp style).

    ``maxq`` is computed via ``int(2.0 ** (bits - 1))`` so it stays a plain
    Python int constant inside the inductor graph and Triton never tries to
    lower a ``2 ** SymInt`` through ``libdevice.pow(fp32, i64)``.
    """
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = int(2.0 ** (bits - 1))
    scale = init_scale * max_scale.unsqueeze(dim=-1)
    int_w = round_ste(tensor / scale + v)
    q = torch.clamp(int_w, -maxq, maxq - 1)
    qdq_result = (scale * q).to(tensor.dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, scale, maxq


@torch.inference_mode()
def qdq_mxfp(tensor, max_val, max_norm, emax, ebits, mbits):
    shared_exp = torch.where(max_val == 0, torch.ones_like(max_val), torch.log2(max_val))
    shared_exp = torch.floor(shared_exp)
    scale_emax = (1 << (8 - 1)) - 1
    shared_exp = (shared_exp - emax).clamp(min=-scale_emax, max=scale_emax)

    scale = torch.pow(2.0, shared_exp)
    tensor = tensor / scale
    tensor = torch.clamp(tensor, min=-max_norm, max=max_norm)
    tensor = quant_element(tensor, ebits, mbits, max_norm)

    tensor = tensor * scale
    return tensor


def mx_init(tensor, bits, qw=None):
    data_type = "mx_fp" + str(bits)
    ebits, mbits, emax, max_norm, min_norm = MXFP_FORMAT_CACHE[data_type]
    tensor = tensor.to(torch.float32)
    max_val, _ = torch.max(torch.abs(tensor), dim=-1, keepdim=True)
    qdq_t = qdq_mxfp(tensor, max_val, max_norm, emax, ebits, mbits)
    best_loss = torch.sum((qdq_t - tensor) ** 2 * qw, dim=-1)
    scales = torch.ones_like(max_val)
    tmp_scale = 0.5
    while tmp_scale < 1.51:
        if tmp_scale == 1.0:
            continue
        max_val_new = max_val * tmp_scale
        tmp_qdq_t = qdq_mxfp(tensor, max_val_new, max_norm, emax, ebits, mbits)
        loss = torch.sum((tmp_qdq_t - tensor) ** 2 * qw, dim=-1)
        replace_id = loss < best_loss
        scales[replace_id] = (torch.ones_like(scales) * tmp_scale)[replace_id]
        best_loss[replace_id] = loss[replace_id]
        tmp_scale += 0.01
    return scales


def nv_fp4(tensor, bits=4, group_size=16, v=0, global_scale=None, max_scale=1.0, init_scale=1.0, **kwargs):
    orig_dtype = tensor.dtype
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    if global_scale is None:
        tensor_max = tensor.abs().max().to(torch.float32)
        global_scale = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX * get_reciprocal(tensor_max)
    global_scale = global_scale.to(device=tensor.device, dtype=torch.float32)
    if isinstance(init_scale, torch.Tensor):
        init_scale = init_scale.view(-1)
    max_scale = max_scale * init_scale
    qdq_res, scale = ref_nvfp4_quant(tensor, global_scale, group_size, v, scale_coeff=max_scale)
    qdq_res = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_res.to(orig_dtype), scale, None


def nv_init(tensor, bits, qw=None):
    tensor = tensor.to(torch.float32)
    qdq_t, dummy_scale, _ = nv_fp4(tensor, bits=4, group_size=16, v=0, max_scale=1.0)
    best_loss = torch.sum((qdq_t - tensor) ** 2 * qw, dim=-1)
    scales = torch.ones_like(dummy_scale)
    tmp_scale = 0.5
    while tmp_scale < 1.51:
        if tmp_scale == 1.0:
            continue
        scales_new = torch.ones_like(dummy_scale) * tmp_scale
        tmp_qdq_t, _, _ = nv_fp4(tensor, bits=4, group_size=16, v=0, max_scale=scales_new)
        loss = torch.sum((tmp_qdq_t - tensor) ** 2 * qw, dim=-1)
        replace_id = loss < best_loss
        scales[replace_id] = scales_new[replace_id]
        best_loss[replace_id] = loss[replace_id]
        tmp_scale += 0.01
    return scales


def quant_mx(
    tensor,
    bits=4,
    group_size=-1,
    v=0,
    max_scale=1.0,
    init_scale=1.0,
    mantissa_rounding="even",
    data_type="mx_fp",
    **kwargs,
):
    """Quantize the given tensor using the specified parameters.

    This function performs quantization on the `tensor` tensor according to the
    given bit width (`bits`), data type (`data_type`), and additional parameters.
    The quantization process involves scaling the tensor values and adjusting
    the exponent and mantissa to fit within the specified format.

    Args:
        tensor (torch.Tensor): The tensor containing the tensors to be quantized.
        bits (int): The bit width to be used for quantization.
        group_size (int): The group size of sharing scale and exponent.
        data_type (str): The data type for quantization (e.g., 'mx_fp4').
        v (float): A value used for adjusting the tensors.
        max_scale (float or torch.Tensor): The maximum scale to be applied to the tensors.
        mantissa_rounding (str): rounding method for mantissa,currently support even,nearest,floor

    Returns:
        tuple: A tuple containing the quantized tensors, shared exponent, and None (reserved for future use).

    Raises:
        KeyError: If `data_type` is not found in `MXFP_FORMAT_CACHE`.
    """
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    data_type = data_type if data_type in MXFP_FORMAT_CACHE else "mx_fp" + str(bits)
    ebits, mbits, emax, max_norm, min_norm = MXFP_FORMAT_CACHE[data_type]
    orig_dtype = tensor.dtype
    tensor = tensor.to(torch.float32)
    max_val, _ = torch.max(torch.abs(tensor), dim=-1, keepdim=True)
    if isinstance(max_scale, torch.Tensor):
        max_val *= init_scale * (max_scale.unsqueeze(dim=-1)).to(tensor.device)
    else:
        max_val *= init_scale * max_scale

    # shared_exp = torch.log2(shared_exp + FP32_MIN_NORMAL * (shared_exp == 0).type(shared_exp.dtype))
    shared_exp = torch.where(max_val == 0, torch.ones_like(max_val), torch.log2(max_val))
    shared_exp = floor_ste(shared_exp)
    scale_emax = (1 << (8 - 1)) - 1
    shared_exp = (shared_exp - emax).clamp(min=-scale_emax, max=scale_emax)

    scale = torch.pow(2.0, shared_exp)
    tensor = tensor / scale + v
    tensor = torch.clamp(tensor, min=-max_norm, max=max_norm)
    tensor = quant_element(tensor, ebits, mbits, max_norm, mantissa_rounding)

    tensor = tensor * scale
    tensor = revert_tensor_by_pad(tensor, orig_shape=orig_shape, pad_len=pad_len)
    return tensor.to(orig_dtype), shared_exp.to(orig_dtype), None


class WrapperLinearV2(WrapperLinear):
    """A wrapper for linear/conv1d layers to enable quantization and tuning.

    This module wraps an existing linear or conv1d layer and provides additional functionality
    for quantization, parameter tuning, and activation/bias normalization.

    Args:
        orig_layer (torch.nn.Module): The original layer to be wrapped (linear or conv1d).
        enable_minmax_tuning (bool): Whether to enable min-max scale tuning.
        enable_norm_bias_tuning (bool): Whether to enable normalization and tuning of the bias term.
        device (str): Device on which to run computations (e.g., 'cpu' or 'cuda').
    """

    def __init__(
        self,
        orig_layer,
        enable_minmax_tuning=True,
        enable_norm_bias_tuning=False,
        device="cpu",
        enable_round_tuning=True,
        enable_torch_compile=False,
        disable_opt_rtn=True,  # TODO does not support it
        **kwargs,
    ):
        """Initializes the WrapperLinear module.

        Args:
            orig_layer (torch.nn.Module): The original layer to wrap.
            enable_minmax_tuning (bool): Whether to enable min-max scale tuning.
            enable_norm_bias_tuning (bool): Whether to enable normalization and tuning for the bias term.
            device (str): The computation device, such as 'cpu' or 'cuda'.
        """
        super(WrapperLinearV2, self).__init__(
            orig_layer=orig_layer,
            enable_minmax_tuning=enable_minmax_tuning,
            enable_norm_bias_tuning=enable_norm_bias_tuning,
            device=device,
            enable_round_tuning=enable_round_tuning,
            enable_torch_compile=enable_torch_compile,
            disable_opt_rtn=disable_opt_rtn,
            **kwargs,
        )

    def _init_tuning_params_and_quant_func(self):
        """Initializes tuning parameters and quantization functions.

        This method sets up required parameters and functions for weight quantization,
        activation quantization, and bias/normalization.
        """
        super()._init_tuning_params_and_quant_func()

        orig_weight = getattr(self.orig_layer, "get_weight", lambda: self.orig_layer.weight)()
        weight_reshape, _, _ = reshape_pad_tensor_by_group_size(orig_weight.data, self.orig_layer.group_size)
        if hasattr(self.orig_layer, "imatrix"):  # MOE model may have no imatrix
            imatrix = self.orig_layer.imatrix.reshape(1, -1)
            imatrix = reshape_pad_tensor_by_group_size(imatrix, self.orig_layer.group_size, val=1e-5)[0].view(1, -1)
            imatrix = imatrix.expand(weight_reshape.numel() // imatrix.numel(), -1)
            imatrix = imatrix.reshape(weight_reshape.shape)
            imatrix = imatrix.to(orig_weight.device)
        else:
            imatrix = 1.0
        if self.orig_layer.data_type.startswith("int"):
            self.init_scale = search_scales(weight_reshape, self.orig_layer.bits, imatrix)
            self.init_scale = torch.where(
                self.init_scale < 0,
                torch.clamp(self.init_scale, max=-self.q_scale_thresh),
                torch.clamp(self.init_scale, min=self.q_scale_thresh),
            )
        elif self.orig_layer.data_type.startswith("mx"):
            self.init_scale = mx_init(weight_reshape, self.orig_layer.bits, imatrix)
        elif self.orig_layer.data_type.startswith("nv"):
            self.init_scale = nv_init(weight_reshape, self.orig_layer.bits, imatrix)
        else:
            self.init_scale = 1.0
        self.orig_layer.imatrix = None
        delattr(self.orig_layer, "imatrix")

        # self.weight_quant_func, self.data_type = get_quant_func(orig_layer.data_type, orig_layer.bits, orig_layer.sym)
        if self.orig_layer.data_type.startswith("int"):
            self.weight_quant_func = quant_tensor_sym
        elif self.orig_layer.data_type.startswith("mx"):
            self.weight_quant_func = quant_mx
        elif self.orig_layer.data_type.startswith("nv"):
            self.weight_quant_func = nv_fp4
        else:
            logger.error("unsupported dtype")
            exit(-1)
        self.data_type = self.orig_layer.data_type
        if self.enable_torch_compile:
            self.weight_quant_func = compile_func(self.weight_quant_func, self.device)

    def _qdq_weight(self, value, min_scale, max_scale):
        """Quantizes and dequantizes weights with tuning parameters.

        Args:
            value (torch.Tensor): Value added for rounding for tuning.
            min_scale (torch.Tensor): Minimum scale for the min value of quantization.
            max_scale (torch.Tensor): Maximum scale for the max value of quantization.

        Returns:
            tuple: Quantized weight, scale, and zero point.
        """
        if self.orig_layer.bits >= 16:
            return self.orig_layer.weight, None, None
        min_scale.data.clamp_(0, 2.0)  # TODO changed
        max_scale.data.clamp_(0, 2.0)
        weight = self.orig_layer.weight
        if weight.device.type == "meta":
            weight = self.orig_layer.get_weight().to(self.device)
        if isinstance(self.orig_layer, transformers.pytorch_utils.Conv1D):
            weight = weight.t()

        quant_kwargs = {}
        if hasattr(self.orig_layer, "super_bits"):
            quant_kwargs["super_bits"] = self.orig_layer.super_bits
            quant_kwargs["super_group_size"] = self.orig_layer.super_group_size

        weight_q, scale, zp = self.weight_quant_func(
            weight,
            bits=self.orig_layer.bits,
            group_size=self.orig_layer.group_size,
            v=value,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_dtype=self.orig_layer.scale_dtype,
            tensor_min=self.weight_min,
            tensor_max=self.weight_max,
            data_type=self.data_type,
            q_scale_thresh=self.q_scale_thresh,
            global_scale=getattr(self, "weight_global_scale", None),
            init_scale=self.init_scale,
            **quant_kwargs,
        )
        weight_q = weight_q.to(weight.dtype)
        if isinstance(self.orig_layer, transformers.pytorch_utils.Conv1D):
            weight_q = weight_q.t()
        return weight_q, scale, zp


def wrapper_block_v2(block, enable_minmax_tuning, enable_norm_bias_tuning, device="cpu", **kwargs):
    """Wraps the layers in the given block with a custom Wrapper module.

    Args:
        block: The input block containing linear and conv1d layers to be wrapped.
        enable_minmax_tuning: A boolean indicating whether min-max tuning is enabled.

    Returns:
        list: A list of names of the wrapped layers and unwrapped layers.
    """
    quantized_layers = []
    unquantized_layers = []
    for n, m in block.named_modules():
        if isinstance(m, SUPPORTED_LAYER_TYPES):
            if not check_to_quantized(m):
                unquantized_layers.append(n)
                continue
            new_m = WrapperLinearV2(
                m,
                enable_minmax_tuning=enable_minmax_tuning,
                enable_norm_bias_tuning=enable_norm_bias_tuning,
                device=device,
                **kwargs,
            )
            set_module(block, n, new_m)
            quantized_layers.append(n)

        elif enable_norm_bias_tuning:
            if "norm" in m.__class__.__name__.lower():
                if m.__class__.__name__ in NORM_MAPPING.keys():
                    wrapper_layer_class = NORM_MAPPING[m.__class__.__name__]
                    new_m = wrapper_layer_class(m, device=device)
                    set_module(block, n, new_m)
                elif "RMSNorm" in m.__class__.__name__:
                    logger.warning_once(
                        f"use LlamaRMSNorm to wrap {m.__class__.__name__}, please check the correctness yourself"
                    )
                    wrapper_layer_class = NORM_MAPPING["LlamaRMSNorm"]
                    new_m = wrapper_layer_class(m, device=device)
                    set_module(block, n, new_m)
                else:
                    logger.warning_once(f"{m.__class__.__name__} is not supported")
    return quantized_layers, unquantized_layers


def _register_act_max_hook_ext(self, model):
    def get_act_max_hook(module, input, output):
        if isinstance(input, (tuple, list)):
            input = input[0]
        if input.numel() == 0:
            return  # as no needs for act_max update
        input, _, _ = reshape_pad_tensor_by_group_size(input, self.act_group_size)
        act_max = torch.max(torch.abs(input), dim=-1).values
        if not hasattr(module, "act_max") or module.act_max.numel() == 0:
            module.act_max = act_max
        else:
            act_max = act_max.to(module.act_max.device)
            if is_nv_fp(self.act_data_type):  ## for nvfp per-tensor input_global_scale calculation usage
                module.act_max = torch.max(torch.tensor([act_max.max(), module.act_max.max()], device=act_max.device))
            else:
                module.act_max = torch.max(act_max, module.act_max)

    def get_imatrix_hook(module, input, output):
        input = input[0] if isinstance(input, (tuple, list)) else input
        flattened = input.reshape(-1, input.shape[-1]).to(torch.float32)
        squared = torch.sum(torch.pow(flattened, 2), dim=0).to(torch.float32)

        if not hasattr(module, "imatrix"):
            module.imatrix = squared
        else:
            module.imatrix += squared.to(module.imatrix.device)

    hook_handles = []

    for n, m in model.named_modules():
        if isinstance(m, self.supported_types) and check_to_quantized(m):
            if not is_wint4aint4(self):  # INT4 no imatrix is much better
                hook = m.register_forward_hook(get_imatrix_hook)
                hook_handles.append(hook)

        if (
            hasattr(m, "act_dynamic")
            and check_need_act_calibration(m.act_dynamic, m.act_data_type, m.act_bits)
            and check_to_quantized(m)
        ):
            hook = m.register_forward_hook(get_act_max_hook)
            hook_handles.append(hook)
            continue

        # for whole model, RTN
        if n in self.layer_config:
            config = self.layer_config[n]
            act_dynamic = config.get("act_dynamic", True)
            act_data_type = config.get("act_data_type", None)
            act_bits = config.get("act_bits", 16)
            if (
                config["bits"] <= 8
                and check_need_act_calibration(act_dynamic, act_data_type, act_bits)
                and check_to_quantized(config)
            ):
                hook = m.register_forward_hook(get_act_max_hook)
                hook_handles.append(hook)
                continue
    return hook_handles


def _dq_asym_qdq(tensor, scale, wmin, bits, group_size, v=0):
    """Pure asym double-quant qdq math given precomputed scale/wmin.

    ``maxq`` is computed via ``int(2.0 ** bits) - 1`` so that any SymInt
    handling for ``bits`` does not produce a ``libdevice.pow(fp32, i64)`` call
    in Triton (which lacks that overload). The final ``int(...)`` cast keeps
    ``maxq`` as a Python int constant inside the compiled graph.
    """
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
    """Pure sym double-quant qdq math given precomputed scale.

    ``maxq`` is computed via float ``2.0 ** (bits - 1)`` then cast to
    ``int`` to avoid SymInt-driven shifts being lowered through
    ``libdevice.pow``.
    """
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


class DQWrapperLinear(WrapperLinear):
    """A wrapper for linear/conv1d layers to enable quantization and tuning.

    This module wraps an existing linear or conv1d layer and provides additional functionality
    for quantization, parameter tuning, and activation/bias normalization.

    Args:
        orig_layer (torch.nn.Module): The original layer to be wrapped (linear or conv1d).
        enable_minmax_tuning (bool): Whether to enable min-max scale tuning.
        enable_norm_bias_tuning (bool): Whether to enable normalization and tuning of the bias term.
        device (str): Device on which to run computations (e.g., 'cpu' or 'cuda').
    """

    def __init__(
        self,
        orig_layer,
        enable_minmax_tuning=True,
        enable_norm_bias_tuning=False,
        device="cpu",
        enable_round_tuning=True,
        enable_torch_compile=False,
        disable_opt_rtn=True,
        **kwargs,
    ):
        """Initializes the WrapperLinear module.

        Args:
            orig_layer (torch.nn.Module): The original layer to wrap.
            enable_minmax_tuning (bool): Whether to enable min-max scale tuning.
            enable_norm_bias_tuning (bool): Whether to enable normalization and tuning for the bias term.
            device (str): The computation device, such as 'cpu' or 'cuda'.
        """
        super(DQWrapperLinear, self).__init__(
            orig_layer=orig_layer,
            enable_minmax_tuning=enable_minmax_tuning,
            enable_norm_bias_tuning=enable_norm_bias_tuning,
            device=device,
            enable_round_tuning=enable_round_tuning,
            enable_torch_compile=enable_torch_compile,
            disable_opt_rtn=disable_opt_rtn,
            **kwargs,
        )
        self.prev_scale = None
        self.prev_wmin = None
        self.prev_d_scale = None
        self.prev_d_wmin = None

    def _init_tuning_params_and_quant_func(self):
        """Initializes tuning parameters and quantization functions.

        This method sets up required parameters and functions for weight quantization,
        activation quantization, and bias/normalization.
        """
        super()._init_tuning_params_and_quant_func()
        p_dtype = torch.float32
        # ``search_func`` is kept un-compiled because it contains data-dependent
        # control flow (imatrix branches, iterative searches), while
        # ``weight_quant_func`` is the compilable pure-math part.
        self.search_func = None
        self._dq_kind = None
        self._is_dq_path = False
        if hasattr(self.orig_layer, "super_group_size") and self.orig_layer.super_group_size is not None:
            self._is_dq_path = True
            from auto_round.data_type.gguf import search_gguf_scale_min_asym, search_gguf_scale_min_sym

            if self.orig_layer.data_type == "int_asym_dq":
                self.search_func = search_gguf_scale_min_asym
                self.weight_quant_func = _dq_asym_qdq
                self._dq_kind = "asym"
            else:
                self.search_func = search_gguf_scale_min_sym
                self.weight_quant_func = _dq_sym_qdq
                self._dq_kind = "sym"
        elif self.orig_layer.sym:
            from auto_round.data_type.int import quant_tensor_sym

            self.weight_quant_func = quant_tensor_sym
        else:
            from auto_round.data_type.int import quant_tensor_asym

            self.weight_quant_func = quant_tensor_asym
        self.data_type = self.orig_layer.data_type
        if self.enable_act_quant:
            from auto_round.data_type.gguf import quant_tensor_gguf_asym_dq as _gguf_asym_dq
            from auto_round.data_type.gguf import quant_tensor_gguf_sym_dq as _gguf_sym_dq

            self.act_quant_func = _gguf_asym_dq if self.orig_layer.act_data_type == "int_asym_dq" else _gguf_sym_dq
            if self.enable_torch_compile:
                self.act_quant_func = compile_func(self.act_quant_func, self.device)
            self._init_params("act_max_scale", p_dtype, (1), 1.0, not self.orig_layer.act_dynamic)

        if self.enable_torch_compile:
            self.weight_quant_func = compile_func(self.weight_quant_func, self.device)

    @torch.no_grad()
    def _run_search(self, weight, v):
        """Run the per-format scale/wmin search separately from the quant func.

        Uses the search routines from ``auto_round.data_type.gguf`` and forwards
        the tuning perturbation ``v``. Returns the parameters to feed into the
        (compilable) ``weight_quant_func``.
        """
        from auto_round.data_type.gguf import double_quant_tensor_sym_rtn
        from auto_round.export.export_to_gguf.config import GGML_QUANT_SIZES, QK_K

        bits = self.orig_layer.bits
        scale_dtype = self.orig_layer.scale_dtype
        imatrix = getattr(self.orig_layer, "imatrix", None)

        if self._dq_kind == "asym":
            group_size = 16 if bits == 2 else 32
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
            return {"scale": scale, "wmin": wmin, "d_scale": d_scale, "d_wmin": d_wmin}

        # sym path
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
        return {"scale": scale, "d_scale": d_scale}

    def _qdq_weight(self, value, min_scale, max_scale, scale_v=None, iter=None):
        """Quantizes and dequantizes weights with tuning parameters.

        Args:
            value (torch.Tensor): Value added for rounding for tuning.
            min_scale (torch.Tensor): Minimum scale for the min value of quantization.
            max_scale (torch.Tensor): Maximum scale for the max value of quantization.

        Returns:
            tuple: Quantized weight, scale, and zero point.
        """
        if self.orig_layer.bits >= 16:
            return self.orig_layer.weight, None, None
        min_scale.data.clamp_(0.5, 1.5)
        max_scale.data.clamp_(0.5, 1.5)
        weight = self.orig_layer.weight
        if weight.device.type == "meta":
            weight = self.orig_layer.get_weight().to(self.device)
        if isinstance(self.orig_layer, transformers.pytorch_utils.Conv1D):
            weight = weight.t()

        if self._is_dq_path:
            # Split search (data-dependent, un-compiled) from quant math (compilable).
            iter_v = self.cur_iter if (iter is None and hasattr(self, "cur_iter")) else iter
            if iter_v is None:
                iter_v = 0
            need_search = (iter_v % 10 == 0) or (iter_v == -1) or (self.prev_scale is None)
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
                group_size = 16 if bits == 2 else 32
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

        # Non-dq path: preserve original behavior.
        quant_kwargs = {}
        if hasattr(self.orig_layer, "super_bits"):
            quant_kwargs["super_bits"] = self.orig_layer.super_bits
            quant_kwargs["super_group_size"] = self.orig_layer.super_group_size
        weight_q, scale, zp = self.weight_quant_func(
            weight,
            bits=self.orig_layer.bits,
            group_size=self.orig_layer.group_size,
            v=value,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_dtype=self.orig_layer.scale_dtype,
            tensor_min=self.weight_min,
            tensor_max=self.weight_max,
            data_type=self.data_type,
            q_scale_thresh=self.q_scale_thresh,
            prev_scale=self.prev_scale,
            prev_wmin=self.prev_wmin,
            prev_d_scale=self.prev_d_scale,
            prev_d_wmin=self.prev_d_wmin,
            **quant_kwargs,
        )
        weight_q = weight_q.to(weight.dtype)
        if isinstance(self.orig_layer, transformers.pytorch_utils.Conv1D):
            weight_q = weight_q.t()
        if isinstance(scale, dict) and "d_scale" in scale and self.prev_scale is None:
            self.prev_scale = scale["scale"]
            self.prev_d_scale = scale["d_scale"]
            if isinstance(zp, dict):
                self.prev_wmin = zp["wmin"]
                self.prev_d_wmin = zp["d_wmin"]
        elif self.prev_scale is None:
            self.prev_scale = scale
        # self.orig_layer.imatrix = None
        return weight_q, scale, zp


def dq_wrapper_block(block, enable_minmax_tuning, enable_norm_bias_tuning, device="cpu", **kwargs):
    """Wraps the layers in the given block with a custom Wrapper module.

    Args:
        block: The input block containing linear and conv1d layers to be wrapped.
        enable_minmax_tuning: A boolean indicating whether min-max tuning is enabled.

    Returns:
        list: A list of names of the wrapped layers and unwrapped layers.
    """
    quantized_layers = []
    unquantized_layers = []
    for n, m in block.named_modules():
        if isinstance(m, SUPPORTED_LAYER_TYPES):
            if not check_to_quantized(m):
                unquantized_layers.append(n)
                continue
            new_m = DQWrapperLinear(
                m,
                enable_minmax_tuning=enable_minmax_tuning,
                enable_norm_bias_tuning=enable_norm_bias_tuning,
                device=device,
                **kwargs,
            )
            set_module(block, n, new_m)
            quantized_layers.append(n)

        elif enable_norm_bias_tuning:
            if "norm" in m.__class__.__name__.lower():
                if m.__class__.__name__ in NORM_MAPPING.keys():
                    wrapper_layer_class = NORM_MAPPING[m.__class__.__name__]
                    new_m = wrapper_layer_class(m, device=device)
                    set_module(block, n, new_m)
                elif "RMSNorm" in m.__class__.__name__:
                    logger.warning_once(
                        f"use LlamaRMSNorm to wrap {m.__class__.__name__}, please check the correctness yourself"
                    )
                    wrapper_layer_class = NORM_MAPPING["LlamaRMSNorm"]
                    new_m = wrapper_layer_class(m, device=device)
                    set_module(block, n, new_m)
                else:
                    logger.warning_once(f"{m.__class__.__name__} is not supported")

    return quantized_layers, unquantized_layers
