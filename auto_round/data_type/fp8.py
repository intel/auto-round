# Copyright (c) 2024 Intel Corporation
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

from auto_round.data_type.base import register_dtype, register_quantizer
from auto_round.data_type.utils import (
    float8_e4m3fn_ste,
    float8_e5m2_ste,
    get_gaudi_fp8_ste_func,
    reshape_pad_tensor_by_group_size,
    revert_tensor_by_pad,
)
from auto_round.utils import is_gaudi2, logger


@register_dtype(("block_fp8_sym", "block_fp8", "block_fp8_e4m3"))
def quant_block_fp_sym(tensor, max_scale=1.0, tensor_max=None, group_size=(128, 128), v=0, tensor_min=None, **kwargs):
    """Symmetric quantization using block float8 format.

    Args:
        tensor (torch.Tensor): Input tensor to quantize.
        max_scale (float, optional): Maximum scaling factor. Defaults to 1.0.
        tensor_max (float, optional): Maximum tensor value for precomputed scale. Defaults to None.
        **kwargs: Additional arguments for compatibility.

    Returns:
        tuple:
            - Quantized and dequantized tensor (torch.Tensor).
            - Scale tensor used for quantization (torch.Tensor).
            - Placeholder for zp (None).
    """
    info = torch.finfo(torch.float8_e4m3fn)
    orig_dtype = tensor.dtype
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    if isinstance(max_scale, torch.Tensor):
        max_scale = max_scale.to(tensor.device)
    if isinstance(v, torch.Tensor):
        v = v.to(tensor.device)
    if tensor_max is None:
        max_tensor = tensor.abs().amax(dim=(-2, -1)) * max_scale
    elif isinstance(tensor_max, torch.Tensor):
        max_tensor = (
            tensor_max.to(tensor.device) * max_scale
            if tensor_min is None
            else torch.maximum(tensor_max.abs(), tensor_min.abs()).to(tensor.device) * max_scale
        )
    else:
        max_tensor = (
            torch.tensor(tensor_max).to(tensor.device) * max_scale
            if tensor_min is None
            else torch.maximum(torch.tensor(tensor_max).abs(), torch.tensor(tensor_min).abs()).to(tensor.device)
            * max_scale
        )
    scale = max_tensor / info.max
    assert len(scale.shape) == 2, f"Only support 2D group_size, but get {len(scale.shape)}"
    min_scaling_factor = float(1.0 / (info.max * 512.0))  ##copy from vllm
    scale = torch.clip(scale, min=min_scaling_factor)
    if tensor.dtype == torch.float16:  ## Avoid NaN gradients with float16
        tensor = tensor.to(torch.bfloat16)

    fp8_res = tensor / scale.unsqueeze(-1).unsqueeze(-1) + v
    fp8_res = torch.clip(fp8_res, info.min, info.max)
    fp8_res = float8_e4m3fn_ste(fp8_res)
    qdq_res = fp8_res * scale.unsqueeze(-1).unsqueeze(-1)
    qdq_res = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)
    qdq_res = qdq_res.to(orig_dtype)
    return qdq_res, scale, None


@register_dtype(("fp8_sym", "fp8", "fp8_e4m3"))
def quant_fp8_sym(tensor, max_scale=1.0, tensor_max=None, group_size=-1, v=0, tensor_min=None, **kwargs):
    """Symmetric quantization using float8 format.

    Allows both dynamic per-token scaling and tensor-wide quantization depending on input.

    Args:
        tensor (torch.Tensor): Input tensor to quantize.
        max_scale (float, optional): Maximum scaling factor. Defaults to 1.0.
        tensor_max (float, optional): Maximum tensor value for precomputed scale. Defaults to None.
        **kwargs: Additional arguments for compatibility.

    Returns:
        tuple:
            - Quantized and dequantized tensor (torch.Tensor).
            - Scale tensor used for quantization (torch.Tensor).
            - Placeholder for zp (None).
    """
    info = torch.finfo(torch.float8_e4m3fn)
    orig_dtype = tensor.dtype
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    if isinstance(max_scale, torch.Tensor):
        max_scale = max_scale.to(tensor.device)
    if isinstance(v, torch.Tensor):
        v = v.to(tensor.device)
    if tensor_max is None:  ##dynamic per-token
        max_tensor = torch.max(torch.abs(tensor), dim=-1)[0] * max_scale
    elif isinstance(tensor_max, torch.Tensor):
        max_tensor = (
            tensor_max.to(tensor.device) * max_scale
            if tensor_min is None
            else torch.maximum(tensor_max.abs(), tensor_min.abs()).to(tensor.device) * max_scale
        )
    else:
        max_tensor = (
            torch.tensor(tensor_max).to(tensor.device) * max_scale
            if tensor_min is None
            else torch.maximum(torch.tensor(tensor_max).abs(), torch.tensor(tensor_min).abs()).to(tensor.device)
            * max_scale
        )
    scale = max_tensor.to(torch.float32) / info.max
    min_scaling_factor = float(1.0 / (info.max * 512.0))  ##copy from vllm
    scale = torch.clip(scale, min=min_scaling_factor)
    if tensor.dtype == torch.float16:  ## Avoid NaN gradients with float16
        tensor = tensor.to(torch.bfloat16)
    scale = scale.unsqueeze(dim=-1)
    fp8_res = tensor / scale + v
    fp8_res = torch.clip(fp8_res, info.min, info.max)
    fp8_res = float8_e4m3fn_ste(fp8_res)
    qdq_res = fp8_res * scale
    qdq_res = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)
    qdq_res = qdq_res.to(orig_dtype)
    return qdq_res, scale, None


@register_dtype("fp8_e5m2")
def quant_fp8_e5m2(tensor, max_scale=1.0, tensor_max=None, group_size=-1, v=0, tensor_min=None, **kwargs):
    """Symmetric quantization using float8 format.

    Allows both dynamic per-token scaling and tensor-wide quantization depending on input.

    Args:
        tensor (torch.Tensor): Input tensor to quantize.
        max_scale (float, optional): Maximum scaling factor. Defaults to 1.0.
        tensor_max (float, optional): Maximum tensor value for precomputed scale. Defaults to None.
        **kwargs: Additional arguments for compatibility.

    Returns:
        tuple:
            - Quantized and dequantized tensor (torch.Tensor).
            - Scale tensor used for quantization (torch.Tensor).
            - Placeholder for zp (None).
    """
    info = torch.finfo(torch.float8_e5m2)
    orig_dtype = tensor.dtype
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    if tensor_max is None:  ##dynamic per-token
        max_tensor = torch.max(torch.abs(tensor), dim=-1)[0] * max_scale
    elif isinstance(tensor_max, torch.Tensor):
        max_tensor = (
            tensor_max.to(tensor.device) * max_scale
            if tensor_min is None
            else torch.maximum(tensor_max.abs(), tensor_min.abs()).to(tensor.device) * max_scale
        )
    else:
        max_tensor = (
            torch.tensor(tensor_max).to(tensor.device) * max_scale
            if tensor_min is None
            else torch.maximum(torch.tensor(tensor_max).abs(), torch.tensor(tensor_min).abs()).to(tensor.device)
            * max_scale
        )
    scale = max_tensor.to(torch.float32) / info.max
    min_scaling_factor = float(1.0 / (info.max * 512.0))  ##copy from vllm
    scale = torch.clip(scale, min=min_scaling_factor)
    if tensor.dtype == torch.float16:  ## Avoid NaN gradients with float16
        tensor = tensor.to(torch.bfloat16)
    scale = scale.unsqueeze(dim=-1)
    fp8_res = tensor / scale + v
    fp8_res = torch.clip(fp8_res, info.min, info.max)
    fp8_res = float8_e5m2_ste(fp8_res)
    qdq_res = fp8_res * scale
    qdq_res = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)
    qdq_res = qdq_res.to(orig_dtype)
    return qdq_res, scale, None


@register_dtype("fp8_unit_scale")
def quant_fp8_unit_scale(tensor, max_scale=1.0, tensor_max=None, group_size=-1, v=0, **kwargs):
    """Symmetric quantization using float8 format.

    Allows both dynamic per-token scaling and tensor-wide quantization depending on input.

    Args:
        tensor (torch.Tensor): Input tensor to quantize.
        max_scale (float, optional): Maximum scaling factor. Defaults to 1.0.
        tensor_max (float, optional): Maximum tensor value for precomputed scale. Defaults to None.
        **kwargs: Additional arguments for compatibility.

    Returns:
        tuple:
            - Quantized and dequantized tensor (torch.Tensor).
            - Scale tensor used for quantization (torch.Tensor).
            - Placeholder for zp (None).
    """
    info = torch.finfo(torch.float8_e4m3fn)
    orig_dtype = tensor.dtype
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    if tensor.dtype == torch.float16:  ## Avoid NaN gradients with float16
        tensor = tensor.to(torch.bfloat16)
    scale = torch.ones((1), device=tensor.device)
    fp8_res = tensor / scale + v
    fp8_res = torch.clip(fp8_res, info.min, info.max)
    fp8_res = float8_e4m3fn_ste(fp8_res)
    qdq_res = fp8_res * scale
    qdq_res = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)
    qdq_res = qdq_res.to(orig_dtype)
    return qdq_res, scale, None


@register_dtype("fp8_e5m2_unit_scale")
def quant_fp8_e5m2_unit_scale(tensor, max_scale=1.0, tensor_max=None, group_size=-1, v=0, **kwargs):
    """Symmetric quantization using float8 format.

    Allows both dynamic per-token scaling and tensor-wide quantization depending on input.

    Args:
        tensor (torch.Tensor): Input tensor to quantize.
        max_scale (float, optional): Maximum scaling factor. Defaults to 1.0.
        tensor_max (float, optional): Maximum tensor value for precomputed scale. Defaults to None.
        **kwargs: Additional arguments for compatibility.

    Returns:
        tuple:
            - Quantized and dequantized tensor (torch.Tensor).
            - Scale tensor used for quantization (torch.Tensor).
            - Placeholder for zp (None).
    """
    info = torch.finfo(torch.float8_e5m2)
    orig_dtype = tensor.dtype
    tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
    if tensor.dtype == torch.float16:  ## Avoid NaN gradients with float16
        tensor = tensor.to(torch.bfloat16)
    scale = torch.ones((1), device=tensor.device)
    fp8_res = tensor / scale + v
    fp8_res = torch.clip(fp8_res, info.min, info.max)
    fp8_res = float8_e5m2_ste(fp8_res)
    qdq_res = fp8_res * scale
    qdq_res = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)
    qdq_res = qdq_res.to(orig_dtype)
    return qdq_res, scale, None


@register_dtype("fp8_gaudi3_sym")
def quant_fp8_sym_gaudi3(tensor, max_scale=1.0, tensor_max=None, tensor_min=None, **kwargs):
    """Symmetric quantization using float8 format.

    Allows both dynamic per-token scaling and tensor-wide quantization depending on input.

    Args:
        tensor (torch.Tensor): Input tensor to quantize.
        max_scale (float, optional): Maximum scaling factor. Defaults to 1.0.
        tensor_max (float, optional): Maximum tensor value for precomputed scale. Defaults to None.
        **kwargs: Additional arguments for compatibility.

    Returns:
        tuple:
            - Quantized and dequantized tensor (torch.Tensor).
            - Scale tensor used for quantization (torch.Tensor).
            - Placeholder for zp (None).
    """
    orig_shape = tensor.shape
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    orig_dtype = tensor.dtype

    if tensor_max is None:  ##dynamic per-te
        tensor = tensor.reshape(-1, orig_shape[-1])
        max_tensor = torch.max(torch.abs(tensor), dim=-1)[0] * max_scale
    elif isinstance(tensor_max, torch.Tensor):
        max_tensor = (
            tensor_max.to(tensor.device) * max_scale
            if tensor_min is None
            else torch.maximum(tensor_max.clone().detach().abs(), tensor_min.clone().detach().abs()).to(tensor.device)
            * max_scale
        )
    else:
        max_tensor = (
            torch.tensor(tensor_max).to(tensor.device) * max_scale
            if tensor_min is None
            else torch.maximum(torch.tensor(tensor_max).abs(), torch.tensor(tensor_min).abs()).to(tensor.device)
            * max_scale
        )
    scale = max_tensor.to(torch.float32) / fp8_max
    min_scaling_factor = float(1.0 / (fp8_max * 512.0))  ##copy from vllm
    scale = torch.clip(scale, min=min_scaling_factor)
    if tensor.dtype == torch.float16:  ## Avoid NaN gradients with float16
        tensor = tensor.to(torch.bfloat16)
    scale = scale.unsqueeze(dim=-1)
    fp8_res = tensor / scale
    fp8_res = torch.clip(fp8_res, -fp8_max, fp8_max)
    float8_e4m3fn_ste_gaudi = get_gaudi_fp8_ste_func()
    fp8_res = float8_e4m3fn_ste_gaudi(fp8_res)
    qdq_res = fp8_res * scale
    qdq_res = qdq_res.to(orig_dtype).reshape(orig_shape)
    return qdq_res, scale, None


if is_gaudi2():

    @register_dtype(("fp8_sym", "fp8", "fp8_e4m3"))
    def quant_fp8_sym(
        tensor, max_scale=1.0, tensor_max=None, group_size=-1, v=0, tensor_min=None, **kwargs
    ):  # pylint: disable=E0102
        """Symmetric quantization using float8 format.

        Allows both dynamic per-token scaling and tensor-wide quantization depending on input.

        Args:
            tensor (torch.Tensor): Input tensor to quantize.
            max_scale (float, optional): Maximum scaling factor. Defaults to 1.0.
            tensor_max (float, optional): Maximum tensor value for precomputed scale. Defaults to None.
            **kwargs: Additional arguments for compatibility.

        Returns:
            tuple:
                - Quantized and dequantized tensor (torch.Tensor).
                - Scale tensor used for quantization (torch.Tensor).
                - Placeholder for zp (None).
        """
        logger.warning_once("Using float8_e4m3fnuz G2.")
        # !!! USE float8_e4m3fnuz for Gaudi2
        info = torch.finfo(torch.float8_e4m3fnuz)
        orig_dtype = tensor.dtype
        tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(tensor, group_size)
        if isinstance(max_scale, torch.Tensor):
            max_scale = max_scale.to(tensor.device)
        if isinstance(v, torch.Tensor):
            v = v.to(tensor.device)
        if tensor_max is None:  ##dynamic per-token
            max_tensor = torch.max(torch.abs(tensor), dim=-1)[0] * max_scale
        elif isinstance(tensor_max, torch.Tensor):
            max_tensor = (
                tensor_max.to(tensor.device) * max_scale
                if tensor_min is None
                else torch.maximum(tensor_max.abs(), tensor_min.abs()).to(tensor.device) * max_scale
            )
        else:
            max_tensor = (
                torch.tensor(tensor_max).to(tensor.device) * max_scale
                if tensor_min is None
                else torch.maximum(torch.tensor(tensor_max).abs(), torch.tensor(tensor_min).abs()).to(tensor.device)
                * max_scale
            )
        scale = max_tensor.to(torch.float32) / info.max
        min_scaling_factor = float(1.0 / (info.max * 512.0))  ##copy from vllm
        scale = torch.clip(scale, min=min_scaling_factor)
        if tensor.dtype == torch.float16:  ## Avoid NaN gradients with float16
            tensor = tensor.to(torch.bfloat16)
        scale = scale.unsqueeze(dim=-1)
        fp8_res = tensor / scale + v
        fp8_res = torch.clip(fp8_res, info.min, info.max)
        from auto_round.data_type.utils import float8_e4m3fnuz_hpu_ste as ste_fn

        fp8_res2 = ste_fn(fp8_res)
        qdq_res = fp8_res2 * scale
        qdq_res = revert_tensor_by_pad(qdq_res, orig_shape=orig_shape, pad_len=pad_len)
        qdq_res = qdq_res.to(orig_dtype)
        return qdq_res, scale, None


class _FP8WeightQuantizer:
    """Select the requested FP8 format and materialize its scale layout."""

    def __init__(self, spec, primitive=None, scale_layout=None):
        self.spec = spec
        data_type = spec.data_type.lower()
        if primitive is None:
            if data_type in ("fp8_e5m2",):
                primitive, scale_layout = quant_fp8_e5m2, "row"
            elif data_type in ("fp8_unit_scale",):
                primitive, scale_layout = quant_fp8_unit_scale, "scalar"
            elif data_type in ("fp8_e5m2_unit_scale",):
                primitive, scale_layout = quant_fp8_e5m2_unit_scale, "scalar"
            elif data_type == "fp8_gaudi3_sym":
                primitive, scale_layout = quant_fp8_sym_gaudi3, "row"
            elif isinstance(spec.group_size, tuple):
                primitive, scale_layout = quant_block_fp_sym, "block"
            else:
                primitive, scale_layout = quant_fp8_sym, "row"
        self.primitive = primitive
        self.scale_layout = scale_layout

    @classmethod
    def from_spec(cls, spec, canonical=None):
        """Create the FP8 weight quantizer selected by the requested format."""
        return cls(spec)

    @staticmethod
    def create_activation(spec):
        """Create the FP8 activation quantizer for the same format."""
        return _create_fp8_activation(spec)

    def create_state(self, weight, *, imatrix=None, mode, tune_rounding, tune_minmax):
        grouped, _, _ = reshape_pad_tensor_by_group_size(weight, self.spec.group_size)
        reduction_dims = 2 if self.scale_layout == "block" else 1
        tunable_shape = grouped.shape[:-reduction_dims]
        tunables = {}
        if tune_rounding:
            tunables["value"] = torch.nn.Parameter(torch.zeros_like(grouped, dtype=torch.float32))
        if tune_minmax:
            tunables["max_scale"] = torch.nn.Parameter(
                torch.ones(tunable_shape, device=weight.device, dtype=torch.float32)
            )
        return tunables

    def qdq(self, weight, state, *, tunables, materialize=False):
        quantized, scale, zero_point = self.primitive(
            weight,
            bits=self.spec.bits,
            group_size=self.spec.group_size,
            v=tunables.get("value", 0),
            max_scale=tunables.get("max_scale", 1.0),
        )
        from auto_round.data_type.base import WeightQuantizationResult

        return WeightQuantizationResult(
            quantized,
            scale if materialize else None,
            zero_point if materialize else None,
            self.scale_layout if materialize else None,
        )

    @staticmethod
    def apply_result(module, result):
        if result.scale is None:
            raise ValueError("FP8 weight result was not materialized")
        scale = result.scale.reshape(result.weight.shape[0], -1) if result.metadata == "row" else result.scale
        module.weight.data.copy_(result.weight)
        module.scale = scale.cpu()
        module.zp = result.zero_point


class _FP8ActivationQuantizer:
    """Apply the matching FP8 primitive to dynamic or calibrated activations."""

    def __init__(self, spec, data_type, primitive, scale_layout):
        self.spec = spec
        self.data_type = data_type
        self.primitive = primitive
        self.scale_layout = scale_layout
        self.bits = spec.bits
        self.group_size = spec.group_size
        self.requires_calibration = not spec.dynamic

    def observe(self, activation, current):
        if self.scale_layout == "tensor":
            maximum = activation.detach().abs().max()
        else:
            grouped, _, _ = reshape_pad_tensor_by_group_size(activation, self.spec.group_size)
            dims = (-2, -1) if self.scale_layout == "block" else -1
            maximum = grouped.detach().abs().amax(dim=dims)
        return maximum if current is None else torch.maximum(maximum.to(current), current)

    def qdq_tensor(self, activation, observed_max, min_scale, max_scale):
        quantized, _, _ = self.primitive(
            activation,
            bits=self.bits,
            group_size=self.group_size,
            tensor_max=observed_max if self.requires_calibration else None,
            max_scale=max_scale,
        )
        return quantized

    def qdq(self, activation, *, observed_max=None, min_scale=1.0, max_scale=1.0):
        if self.requires_calibration and observed_max is None:
            raise ValueError(f"{self.data_type} activation requires observed_max")
        quantized = self.qdq_tensor(activation, observed_max, min_scale, max_scale)
        return quantized


def _create_fp8_activation(spec):
    """Build an activation quantizer using the same format choice as weights."""
    if spec.data_type == "fp8_gaudi3_sym":
        return _FP8ActivationQuantizer(spec, spec.data_type, quant_fp8_sym_gaudi3, "tensor")
    weight_quantizer = _FP8WeightQuantizer(spec)
    return _FP8ActivationQuantizer(spec, spec.data_type, weight_quantizer.primitive, weight_quantizer.scale_layout)


register_quantizer(
    "fp8_sym",
    aliases=("fp", "float", "fp8", "fp8_e4m3", "block_fp8_sym", "block_fp8", "block_fp8_e4m3"),
)(_FP8WeightQuantizer)
register_quantizer("fp8_e5m2")(_FP8WeightQuantizer)
register_quantizer("fp8_unit_scale")(_FP8WeightQuantizer)
register_quantizer("fp8_e5m2_unit_scale")(_FP8WeightQuantizer)
if is_gaudi2():
    register_quantizer("fp8_gaudi3_sym")(_FP8WeightQuantizer)
