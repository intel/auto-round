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
from auto_round.compressors.utils import check_need_act_calibration, is_nv_fp
from auto_round.data_type.int import search_scales
from auto_round.data_type.mxfp import MXFP_FORMAT_CACHE, quant_element
from auto_round.data_type.nvfp import FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX, ref_nvfp4_quant
from auto_round.data_type.utils import floor_ste, reshape_pad_tensor_by_group_size, revert_tensor_by_pad, round_ste
from auto_round.logger import logger
from auto_round.utils import SUPPORTED_LAYER_TYPES, check_to_quantized, compile_func, get_reciprocal, set_module
from auto_round.wrapper import NORM_MAPPING, WrapperLinear, reshape_and_pad_tensor

__all__ = ["wrapper_autoround"]

FUNC_LIST = {}


def wrapper_func(cls, func_name, *args, **kwargs):
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
        k = max(1, int(flat.numel() * percent / 1000))  # 至少选1个
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
        autocast_ctx = (
            nullcontext() if self.amp else autocast(device_type=str(device).split(":")[0], dtype=self.amp_dtype)
        )
        if self.attention_mask:
            tmp_attention_mask = [self.attention_mask[i] for i in indices]
            tmp_attention_mask = torch.cat(tmp_attention_mask, dim=0).to(device)
            tmp_attention_mask.unsqueeze_(-1)

            with autocast_ctx:
                loss = torch.mean(
                    (
                        torch.abs(output_q.to(torch.float32) - current_output.to(torch.float32))
                        * tmp_attention_mask
                        * mask
                    )
                    ** 2
                )  # pylint: disable=not-callable
                loss = mse_loss(  # pylint: disable=not-callable
                    (output_q * tmp_attention_mask).to(torch.float32),
                    (current_output * tmp_attention_mask).to(torch.float32),
                )
        else:
            with autocast_ctx:
                loss = torch.mean(
                    (torch.abs(output_q.to(torch.float32) - current_output.to(torch.float32)) * mask) ** 2
                )

        return loss

    if func_name == "_register_act_max_hook":
        return _register_act_max_hook_ext(cls, *args, **kwargs)
    if (
        cls.sym
        and cls.enable_alg_ext
        and cls.super_group_size is None
        and (
            (cls.data_type.startswith("int") and cls.act_bits >= 8)
            or cls.data_type.startswith("mx")
            or cls.data_type.startswith("nv")
        )
    ):
        if cls.bits > 2 and (not cls.data_type.startswith("mx") or not cls.data_type.startswith("nv")):
            logger.warning_once(
                "algorithm extension has only undergone limited validation on "
                "INT2,mxfp4 and nvfp4; use with caution."
            )
        if func_name == "_get_loss":
            return _get_loss_ext(cls, *args, **kwargs)

        if func_name == "wrapper_block":
            return wrapper_block_v2(*args, **kwargs)
    if cls.data_type.endswith("dq"):
        if func_name == "wrapper_block":
            return dq_wrapper_block(*args, **kwargs)
    return FUNC_LIST[func_name](*args, **kwargs)


def wrapper_autoround(cls: AutoRound):
    for name in dir(cls):
        if name.startswith("__"):
            continue
        attr = getattr(cls, name)
        if callable(attr) and isinstance(attr, (types.MethodType, types.FunctionType)):
            FUNC_LIST[name] = attr
            setattr(cls, name, partial(wrapper_func, cls, name))


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
    """Quantize and de-quantize tensor asymmetrically. full range, credict goes to llamacpp community

    Args:
        tensor: Tensor containing the tensor to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: Number of elements to share scale for quantization
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for tensor
        max_scale: Maximum scale coefficient for tensor
        tensor_min (Tensor, optional): Minimum tensor value for quantization. Defaults to None.
        tensor_max (Tensor, optional): Maximum tensor value for quantization. Defaults to None.
        scale_dtype: dtype of the quantized scale,as most kernels only support FP16 or FP32, while this value is import
        q_scale_thresh: clip the quantized scale's magnitude to this value to improve the numerical stability

    Returns:
        Quantized and de-quantized tensor, scale, zero-point
    """
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    maxq = 2 ** (bits - 1)
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
    scale_emax = 2 ** (8 - 1) - 1
    shared_exp = (shared_exp - emax).clamp(min=-scale_emax, max=scale_emax)

    scale = torch.pow(2, shared_exp)
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
    scale_emax = 2 ** (8 - 1) - 1
    shared_exp = (shared_exp - emax).clamp(min=-scale_emax, max=scale_emax)

    scale = torch.pow(2, shared_exp)
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
        if self.enable_act_quant and self.orig_layer.data_type.startswith("int"):
            raise ValueError("does not support act quantization in WrapperLinearV2, please use WrapperLinear instead")

    def _init_tuning_params_and_quant_func(self):
        """Initializes tuning parameters and quantization functions.

        This method sets up required parameters and functions for weight quantization,
        activation quantization, and bias/normalization.
        """
        super()._init_tuning_params_and_quant_func()

        orig_weight = getattr(self.orig_layer, "get_weight", lambda: self.orig_layer.weight)()
        weight_reshape = reshape_and_pad_tensor(orig_weight.data, self.orig_layer.group_size)
        if hasattr(self.orig_layer, "imatrix"):  # MOE model may have no imatrix
            imatrix = self.orig_layer.imatrix.reshape(1, -1)
            imatrix = reshape_pad_tensor_by_group_size(imatrix, self.orig_layer.group_size, val=1e-5)[0].view(1, -1)
            imatrix = imatrix.expand(weight_reshape.numel() // imatrix.numel(), -1)
            imatrix = imatrix.reshape(weight_reshape.shape)
            imatrix = imatrix.to(orig_weight.device)
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
            self.orig_layer.imatrix = None
            delattr(self.orig_layer, "imatrix")
        else:
            self.init_scale = 1.0

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


# ---------------------------- gguf alg ----------------------------
from auto_round.data_type.gguf import double_quant_tensor
from auto_round.export.export_to_gguf.packing import make_qx_quants


def make_qp_quants(nmax, data, quant_weights, v=0):
    data = data.to(torch.float32)
    quant_weights = quant_weights.to(torch.float32)
    group_max = torch.max(data, dim=-1, keepdim=True)[0]
    scale = group_max / nmax
    iscale = get_reciprocal(scale)
    if isinstance(v, torch.Tensor) and v.numel() != 1:
        v = v.view(data.shape)
        v = v.to(data.device)

    L = torch.round(iscale * data + v)
    diffs = data - scale * L
    best_mse = torch.sum(quant_weights * diffs * diffs, dim=-1)

    for _is in range(-9, 10):
        if _is == 0:
            continue
        scale_is = group_max / (0.1 * _is + nmax)
        iscale_is = get_reciprocal(scale_is)

        tmp_L = torch.round(iscale_is * data + v).clip(max=nmax)
        diffs = data - scale_is * tmp_L
        mse = torch.sum(quant_weights * diffs * diffs, dim=-1)

        replace_idx = mse < best_mse
        best_mse[replace_idx] = mse[replace_idx]
        iscale[replace_idx] = iscale_is[replace_idx]

    L = torch.round(iscale * data + v).clip(max=nmax)
    sumlx = torch.sum(quant_weights * data * L, dim=-1)
    suml2 = torch.sum(quant_weights * L * L, dim=-1)
    return sumlx / suml2, L


def iterative_wls_quant_search(data, bits=4, rrmin=-1.0, rdelta=0.1, nstep=20, use_mad=False, weights=None, v=0):
    """Adapted from Llamacpp. Performs iterative weighted least squares quantization search.

    Args:
        data (torch.Tensor): Input tensor to quantize.
        bits (int): Number of quantization bits.
        rrmin (float): Initial range scaling factor.
        rdelta (float): Step size for range scaling.
        nstep (int): Number of search steps.
        use_mad (bool): Whether to use mean absolute deviation instead of squared error.
        weights (torch.Tensor): Weight matrix for each element.

    Returns:
        Tuple: (Optimal scale tensor, optimal minimum value tensor)
    """
    dtype = torch.float32
    data = data.to(dtype)
    maxq = 2**bits - 1
    minq = 0
    weights = 1.0 if weights is None else weights.to(dtype)

    rmin = torch.min(data, dim=1, keepdim=True)[0]
    rmax = torch.max(data, dim=1, keepdim=True)[0]

    sum_w = torch.sum(weights, dim=1, keepdim=True)
    sum_x = torch.sum(weights * data, dim=1, keepdim=True)

    # scale = 1 / ((maxq - minq) / (rmax - rmin + 1e-8))
    scale = (rmax - rmin) / (maxq - minq)
    if isinstance(v, torch.Tensor) and v.numel() > 1:
        v = v.reshape(data.shape)

    iscale = get_reciprocal(scale)
    # quant_data = torch.clamp(torch.round((maxq - minq) / (rmax - rmin + 1e-8) * (data - rmin)), minq, maxq)
    quant_data = torch.clamp(torch.round(iscale * (data - rmin) + v), minq, maxq)
    diff = scale * quant_data + rmin - data

    best_mad = torch.sum((weights * torch.abs(diff)) if use_mad else weights * diff**2, dim=1, keepdim=True)

    for is_ in range(nstep):
        factor = rrmin + rdelta * is_ + maxq - minq
        # iscale_new = factor / (rmax - rmin + 1e-8)
        scale_new = (rmax - rmin) / factor
        iscale_new = get_reciprocal(scale_new)
        quant_data_new = torch.clamp(torch.round(iscale_new * (data - rmin)), minq, maxq)

        mul_weights_quant_data = weights * quant_data_new
        sum_l = torch.sum(mul_weights_quant_data, dim=-1, keepdim=True)
        sum_l2 = torch.sum(mul_weights_quant_data * quant_data_new, dim=-1, keepdim=True)
        sum_xl = torch.sum(mul_weights_quant_data * data, dim=-1, keepdim=True)

        D = sum_w * sum_l2 - sum_l**2
        this_scale = (sum_w * sum_xl - sum_x * sum_l) / D
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D
        this_min[this_min > 0] = 0
        this_scale[this_min > 0] = (sum_xl / sum_l2)[this_min > 0]
        reverse_this_scale = get_reciprocal(this_scale)

        quant_data = torch.clamp(torch.round(reverse_this_scale * (data - this_min) + v), minq, maxq)
        diff = this_scale * quant_data + this_min - data
        # diff = this_scale * quant_data_new + this_min - data
        mad = torch.sum((weights * torch.abs(diff)) if use_mad else weights * diff**2, dim=-1, keepdim=True)

        idx_to_replace = torch.where((mad < best_mad) & (D > 0))[0]
        best_mad[idx_to_replace] = mad[idx_to_replace]
        scale[idx_to_replace] = this_scale[idx_to_replace]
        rmin[idx_to_replace] = this_min[idx_to_replace]

    return scale.to(torch.float32), -rmin.to(torch.float32)


def make_qp_new_quants(data, orig_scale, orig_mins, quant_weights, bits=4, super_bits=6, data_v=0, scale_v=0, min_v=0):
    nmax = 2**super_bits - 1
    maxq = 2**bits - 1
    minq = 0
    orig_scale = orig_scale.to(torch.float32)
    quant_weights = quant_weights.to(torch.float32)
    group_max = torch.max(orig_scale, dim=-1, keepdim=True)[0]
    s_scale = group_max / nmax
    i_sscale = get_reciprocal(s_scale)
    if isinstance(scale_v, torch.Tensor):
        if scale_v.numel() != 1:
            scale_v = scale_v.view(orig_scale.shape)
        scale_v = scale_v.to(data.device)
    data_v = data_v.to(data.device)

    L_scale = torch.round(i_sscale * orig_scale + scale_v)
    qdq_scale = L_scale * s_scale
    id_scale = get_reciprocal(qdq_scale)
    id_scale = id_scale.view(-1, 1)
    orig_mins = orig_mins.view(-1, 1) if orig_mins is not None and isinstance(orig_mins, torch.Tensor) else orig_mins
    quant_data = torch.clamp(torch.round(id_scale * (data - orig_mins) + data_v.to(data.device)), minq, maxq)
    qdq_scale = qdq_scale.view(-1, 1)
    diff = qdq_scale * quant_data + orig_mins - data
    best_mse = torch.sum(quant_weights * diff * diff, dim=-1)
    best_mse = best_mse.view(orig_scale.shape)
    best_mse = torch.sum(best_mse, dim=-1)
    for _is in range(-9, 10):
        if _is == 0:
            continue
        scale_s_is = group_max / (0.1 * _is + nmax)
        iscale_s_is = get_reciprocal(scale_s_is)

        tmp_L_scale = torch.round(iscale_s_is * orig_scale + scale_v).clip(min=0, max=nmax)
        qdq_scale = scale_s_is * tmp_L_scale
        reverse_this_scale = get_reciprocal(qdq_scale)
        reverse_this_scale = reverse_this_scale.view(-1, 1)
        quant_data = torch.clamp(torch.round(reverse_this_scale * (data - orig_mins) + data_v), minq, maxq)
        diffs = qdq_scale.view(-1, 1) * quant_data + orig_mins - data
        mse = torch.sum(quant_weights * diffs * diffs, dim=-1)
        mse = mse.view(orig_scale.shape)
        mse = torch.sum(mse, dim=-1)
        replace_idx = mse < best_mse
        best_mse[replace_idx] = mse[replace_idx]
        i_sscale[replace_idx] = iscale_s_is[replace_idx]

    L = torch.round(i_sscale * orig_scale + scale_v).clip(max=nmax)
    quant_weights = torch.sum(quant_weights, dim=-1)
    quant_weights = quant_weights.view(orig_scale.shape)
    sumlx = torch.sum(quant_weights * orig_scale * L, dim=-1)
    suml2 = torch.sum(quant_weights * L * L, dim=-1)
    return sumlx / suml2, L


def quant_tensor_gguf_asym_dq(
    tensor,
    bits=4,
    v=0,
    min_scale=1.0,
    max_scale=1.0,
    scale_dtype=torch.float16,
    tensor_min=None,
    tensor_max=None,
    q_scale_thresh=1e-5,
    imatrix=None,
    prev_scale=None,
    prev_wmin=None,
    prev_d_scale=None,
    prev_d_wmin=None,
    iter=0,
    scale_v=0,
    wmin_v=0,
    **kwargs,
):
    """Quantizes and dequantizes a tensor using asymmetric integer quantization for formats like Q2_K, Q4_K, and Q5_K.
    Only fit for iters 0

    Args:
        tensor (torch.Tensor): Input tensor to quantize.
        bits (int): Number of bits for quantization.
        group_size (int): Group size for per-group quantization.
        v (float): Perturbation added before rounding.
        min_scale (float): Minimum allowed scale value.
        max_scale (float): Maximum allowed scale value.
        scale_dtype (torch.dtype): Data type for quantized scale.
        tensor_min (torch.Tensor, optional): Minimum values for the tensor groups.
        tensor_max (torch.Tensor, optional): Maximum values for the tensor groups.
        q_scale_thresh (float): Threshold to clamp the quantized scale.
        super_group_size (int): Number of groups to bundle for secondary quantization.
        super_bits (int): Number of bits used in secondary quantization.
        imatrix (torch.Tensor, optional): Importance matrix for weighted quantization.

    Returns:
        Tuple: (Quantized-dequantized tensor, scale dictionary, zero-point dictionary)
    """

    orig_dtype = tensor.dtype
    maxq = 2**bits - 1
    group_size = 16 if bits == 2 else 32
    super_bits = 4 if bits == 2 else 6
    super_group_size = 16 if bits == 2 else 8
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    tensor = tensor.to(torch.float32)
    if iter is None:
        iter = 0
    if iter % 10 == 0 or iter == -1 or prev_scale is None:
        if bits not in [2, 4, 5]:
            raise ValueError(f"bits={bits} not supported by rtn_int_asym_dq")
        quant_weights = None
        if imatrix is None or (imatrix is not None and torch.sum(imatrix) == 0):
            search_kwargs = {
                2: {"rmin": -0.5, "rdelta": 0.1, "nstep": 15, "use_mad": True},
                4: {"rmin": -1, "rdelta": 0.1, "nstep": 20, "use_mad": False},
                5: {"rmin": -0.5, "rdelta": 0.1, "nstep": 15, "use_mad": False},
            }
            if bits == 2:
                quant_weights = torch.abs(tensor)
            elif bits == 4 or bits == 5:
                sigma2 = torch.sum(tensor**2, dim=-1, keepdim=True) / 32  ##Note 32 is different from QK_K
                av_x = torch.sqrt(sigma2)
                quant_weights = torch.abs(tensor) + av_x
            params = search_kwargs[bits]
            scale, wmin = iterative_wls_quant_search(
                tensor,
                bits=bits,
                rrmin=params["rmin"],
                rdelta=params["rdelta"],
                nstep=params["nstep"],
                use_mad=params["use_mad"],
                weights=quant_weights,
                v=v,
            )
            scale = scale.to(scale_dtype)
            scale = torch.where(torch.abs(scale) < 1e-30, torch.zeros_like(scale), scale)
            scale = scale.reshape(-1, super_group_size)
            wmin = wmin.reshape(-1, super_group_size)
            scale, d_scale = double_quant_tensor(scale, super_bits)
            wmin = torch.where(torch.abs(wmin) < 1e-30, torch.zeros_like(wmin), wmin)
            wmin, d_wmin = double_quant_tensor(wmin, super_bits)
            wmin = wmin.view(-1, 1)
            scale = scale.view(-1, 1)
        else:
            imatrix = imatrix.to(tensor.device)
            search_kwargs = {
                2: {"rmin": -0.9, "rdelta": 0.05, "nstep": 36, "use_mad": False},
                4: {"rmin": -0.9, "rdelta": 0.05, "nstep": 36, "use_mad": False},
                5: {"rmin": -0.9, "rdelta": 0.05, "nstep": 36, "use_mad": False},
            }

            weights = imatrix.reshape(1, -1)

            weights = weights.expand(tensor.numel() // weights.numel(), -1)
            quant_weights = weights.reshape(tensor.shape)

            if torch.min(quant_weights) == 0:
                logger.warning_once(
                    "please use more data via setting `nsamples` "
                    "to improve accuracy as calibration activations contain 0"
                )

                zero_cnt = torch.sum(quant_weights == 0, dim=-1)
                replace_index = zero_cnt > group_size // 2
                if torch.sum(replace_index) > 0:
                    # Fallback to no imatrix
                    if bits == 2:
                        tmp_quant_weights = torch.abs(tensor)
                    elif bits == 4 or bits == 5:
                        sigma2 = torch.sum(tensor**2, dim=-1, keepdim=True) / 32  ## Note 32 is different from QK_K
                        av_x = torch.sqrt(sigma2)
                        tmp_quant_weights = torch.abs(tensor) + av_x
                    quant_weights[replace_index, :] = tmp_quant_weights[replace_index, :]
                mean_replace_index = (zero_cnt > 0) & (zero_cnt <= group_size // 2)
                if torch.sum(mean_replace_index) > 0:
                    ## use mean values to fill zero values
                    tmp_quant_weights = torch.sum(quant_weights, dim=-1) / (quant_weights.shape[1] - zero_cnt)
                    tmp_quant_weights = tmp_quant_weights.view(-1, 1).expand(-1, quant_weights.shape[1])
                    quant_weights[mean_replace_index, :] = tmp_quant_weights[mean_replace_index, :]

            params = search_kwargs[bits]

            scale, wmin_0 = iterative_wls_quant_search(
                tensor,
                bits=bits,
                rrmin=params["rmin"],
                rdelta=params["rdelta"],
                nstep=params["nstep"],
                use_mad=params["use_mad"],
                weights=quant_weights,
                v=v,
            )
            scale = scale.to(scale_dtype)
            scale = torch.where(torch.abs(scale) < 1e-30, torch.zeros_like(scale), scale)
            nmax = 2**super_bits - 1
            scale = scale.reshape(-1, super_group_size)
            wmin = wmin_0.reshape(-1, super_group_size)
            sum_quant_weights = quant_weights.sum(-1, keepdim=True).reshape(-1, super_group_size)

            d_scale, q_scale = make_qp_new_quants(tensor, scale, wmin, quant_weights, bits, super_bits, data_v=v)
            d_scale = d_scale.unsqueeze(-1)

            d_wmin, q_wmin = make_qp_quants(nmax, wmin, sum_quant_weights, v=wmin_v)

            d_wmin = d_wmin.unsqueeze(-1)
            scale = (d_scale * q_scale).view(-1, 1)
            wmin = (d_wmin * q_wmin).view(-1, 1)
    else:
        scale = prev_scale.detach()
        d_scale = prev_d_scale.detach()
        wmin = prev_wmin.detach()
        d_wmin = prev_d_wmin.detach()
    inverse_scale = get_reciprocal(scale)

    int_w = torch.clamp(round_ste((tensor + wmin) * inverse_scale + v), 0, maxq)
    qdq_result = (scale * int_w - wmin).to(orig_dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)
    return qdq_result, {"scale": scale, "d_scale": d_scale}, {"wmin": wmin, "d_wmin": d_wmin}


def quant_tensor_gguf_sym_dq(
    tensor,
    bits=3,
    v=0,
    imatrix=None,
    prev_scale=None,
    prev_d_scale=None,
    iter=0,
    **kwargs,
):
    """Quantize and de-quantize tensor asymmetrically. For Q3_K, Q6_K.

    Args:
        tensor: Tensor containing the tensor to be quantized
        bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: Number of elements to share scale for quantization
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for tensor
        max_scale: Maximum scale coefficient for tensor
        tensor_min (Tensor, optional): Minimum tensor value for quantization. Defaults to None.
        tensor_max (Tensor, optional): Maximum tensor value for quantization. Defaults to None.
        scale_dtype: dtype of the quantized scale,as most kernels only support FP16 or FP32, while this value is import
        q_scale_thresh: clip the quantized scale's magnitude to this value to improve the numerical stability

    Returns:
        Quantized and de-quantized tensor, scale, zero-point
    """
    from auto_round.export.export_to_gguf.config import GGML_QUANT_SIZES, K_SCALE_SIZE, QK_K

    if bits not in [3, 6]:
        raise KeyError(f"bits={bits} is not supported by gguf_int_sym_dq, please check.")

    maxq = 2 ** (bits - 1)
    group_size = 16
    super_bits = 6 if bits == 3 else 8
    super_group_size = 16

    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    ggml_type = f"q{bits}_k"
    block_size, type_size = GGML_QUANT_SIZES[ggml_type]
    orig_dtype = tensor.dtype

    tensor = tensor.to(torch.float32)
    n_blocks = tensor.nelement() // block_size
    # (nb, 16, 16)
    # tensor = tensor.reshape(n_blocks, super_group_size, QK_K // super_group_size)

    if iter is None:
        iter = 0
    if iter % 10 == 0 or iter == -1 or prev_scale is None:
        if imatrix is None or (imatrix is not None and torch.sum(imatrix) == 0):
            if bits == 3:
                from auto_round.export.export_to_gguf.packing import make_q3_quants

                scale, int_w = make_q3_quants(tensor, bits=bits, do_rmse=True)
            elif bits == 6:
                scale, int_w = make_qx_quants(tensor, bits=bits, rmse_type=1, qw=None)
        else:
            imatrix = imatrix.to(tensor.device)

            weights = imatrix.reshape(1, -1)
            weights = weights.expand(tensor.numel() // weights.numel(), -1)
            quant_weights = weights.reshape(tensor.shape)
            if torch.min(quant_weights) == 0:
                logger.warning_once(
                    "please use more data via setting `nsamples` "
                    "to improve accuracy as calibration activations contain 0"
                )
                zero_cnt = torch.sum(quant_weights == 0, dim=-1)
                replace_index = zero_cnt > group_size // 2
                if torch.sum(replace_index) > 0:
                    if bits == 6:
                        quant_weights[replace_index] = tensor[replace_index] * tensor[replace_index]
                    else:
                        sigma2 = 2 * torch.sum(torch.pow(tensor, 2), dim=-1, keepdim=True) / QK_K
                        tmp_quant_weights = torch.sqrt(sigma2 + tensor * tensor)
                        quant_weights[replace_index] = tmp_quant_weights[replace_index]
                mean_replace_index = (zero_cnt > 0) & (zero_cnt <= group_size // 2)
                if torch.sum(mean_replace_index) > 0:
                    ## use mean values to fill zero values
                    tmp_quant_weights = torch.sum(quant_weights, dim=-1) / (quant_weights.shape[-1] - zero_cnt)
                    tmp_quant_weights = (
                        tmp_quant_weights.view(-1, 1).expand(-1, quant_weights.shape[1]).reshape(tensor.shape)
                    )
                    quant_weights[mean_replace_index] = tmp_quant_weights[mean_replace_index]

            # scale, int_w = make_qx_quants(tensor, bits=bits, rmse_type=1, qw=quant_weights, v=v)
            scale, int_w = make_qx_quants(tensor, bits=bits, rmse_type=1, qw=quant_weights)
        scale = torch.where(torch.abs(scale) < 1e-30, torch.zeros_like(scale), scale)
        # conduct double quant
        d_scale, q_scale = make_qp_new_quants(tensor, scale, 0, quant_weights, bits, super_bits, data_v=v)
        d_scale = d_scale.unsqueeze(-1)
        scale = (d_scale * q_scale).unsqueeze(-1)
    else:
        scale = prev_scale.detach()
        d_scale = prev_d_scale.detach()
    zp = torch.full_like(scale, maxq)  # pylint: disable=E1130
    inverse_scale = get_reciprocal(scale)
    int_w = round_ste(tensor * inverse_scale + v).clip(-maxq, maxq - 1) + maxq
    qdq_result = (scale * (int_w - zp)).to(orig_dtype)
    qdq_result = revert_tensor_by_pad(qdq_result, orig_shape=orig_shape, pad_len=pad_len)

    return qdq_result, {"scale": scale, "d_scale": d_scale}, zp


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
        if hasattr(self.orig_layer, "super_group_size") and self.orig_layer.super_group_size is not None:
            self.weight_quant_func = (
                quant_tensor_gguf_asym_dq if self.orig_layer.data_type == "int_asym_dq" else quant_tensor_gguf_sym_dq
            )
        elif self.orig_layer.sym:
            from auto_round.data_type.int import quant_tensor_sym

            self.weight_quant_func = quant_tensor_sym
        else:
            from auto_round.data_type.int import quant_tensor_asym

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
            self._init_params("act_max_scale", p_dtype, (1), 1.0, not self.orig_layer.act_dynamic)

        if self.enable_torch_compile:
            self.weight_quant_func = compile_func(self.weight_quant_func, self.device)

    def _qdq_weight(self, value, min_scale, max_scale, scale_v=None, wmin_v=None, iter=None):
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
            imatrix=self.orig_layer.imatrix if hasattr(self.orig_layer, "imatrix") else None,
            iter=self.cur_iter if (iter is None and hasattr(self, "cur_iter")) else iter,
            # scale_v=self.scale_v if scale_v is None else scale_v,
            # wmin_v=self.wmin_v if wmin_v is None else wmin_v,
            # xtx=self.orig_layer.xtx if hasattr(self.orig_layer, "xtx") else None,
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
