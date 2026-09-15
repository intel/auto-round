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

import torch

from auto_round.data_type.base import register_dtype, register_quantizer
from auto_round.data_type.utils import float8_e4m3fn_ste, get_gaudi_fp8_ste_func


@register_dtype("fp8_to_int_sym")
def progressive_quant_fp8_int4(
    tensor, bits=4, group_size=-1, v=0, min_scale=1.0, max_scale=1.0, q_scale_thresh=1e-5, **kwargs
):
    """Two-stage quantization: quantize tensor to fp8 by per tensor, then quantize fp8 to w4g128

    This method first quantizes the input tensor into float8 format and then performs
    a secondary quantization to int4 with grouping.

    Args:
        tensor (torch.Tensor): Input tensor to quantize.
        bits (int, optional): Bit precision for secondary quantization. Defaults to 4.
        group_size (int, optional): Group size for int4 quantization. Defaults to -1 (no grouping).
        v (float, optional): Optional parameter for variance tuning. Defaults to 0.
        min_scale (float, optional): Minimum scaling factor for int4 quantization. Defaults to 1.0.
        max_scale (float, optional): Maximum scaling factor for int4 quantization. Defaults to 1.0.
        q_scale_thresh (float, optional): Threshold for scaling. Defaults to 1e-5.
        **kwargs: Additional arguments for compatibility.

    Returns:
        tuple:
            - Quantized and dequantized tensor (torch.Tensor).
            - Combined scaling factor (torch.Tensor).
            - Placeholder for zp (None).
    """

    info = torch.finfo(torch.float8_e4m3fn)
    tensor_max = torch.max(torch.abs(tensor)).to(torch.float32)
    scale = tensor_max.to(torch.float32) / info.max
    min_scaling_factor = 1.0 / (info.max * 512.0)  ##copy from vllm
    bf16_to_fp8_scale = torch.clip(scale, min=min_scaling_factor)
    fp8_res = tensor / bf16_to_fp8_scale
    fp8_res = torch.clip(fp8_res, info.min, info.max)
    fp8_res = float8_e4m3fn_ste(fp8_res)

    ##convert to bf16
    fp8_res_using_16bit = fp8_res.to(tensor.dtype)
    ##convert to int4
    from auto_round.data_type.int import quant_tensor_sym

    qdq_int4_tensor, scale_fp8_to_int4, zp_fp8_to_int4 = quant_tensor_sym(
        fp8_res_using_16bit,
        bits=bits,
        group_size=group_size,
        v=v,
        min_scale=min_scale,
        max_scale=max_scale,
        scale_dtype=torch.bfloat16,
        q_scale_thresh=q_scale_thresh,
    )
    qdq_tensor = qdq_int4_tensor * bf16_to_fp8_scale

    bf16_to_int4_scale = scale_fp8_to_int4 * bf16_to_fp8_scale
    return qdq_tensor, {"scale": bf16_to_int4_scale, "bf16_to_fp8_scale": bf16_to_fp8_scale}, zp_fp8_to_int4


@register_quantizer("fp8_to_int_sym")
class _ProgressiveFP8WeightQuantizer:
    """Own the progressive BF16-to-FP8-to-INT4 weight quantization path."""

    def __init__(self, spec):
        self.spec = spec

    @classmethod
    def from_spec(cls, spec, tuning_options, canonical=None):
        """Create the progressive FP8-to-INT4 weight quantizer."""
        return cls(spec)

    def create_state(self, weight, *, imatrix=None, tuning_options):
        from auto_round.data_type.utils import reshape_pad_tensor_by_group_size

        grouped, _, _ = reshape_pad_tensor_by_group_size(weight, self.spec.group_size)
        tunables = {}
        if tuning_options.enable_round_tuning:
            tunables["value"] = torch.nn.Parameter(torch.zeros_like(grouped, dtype=torch.float32))
        if tuning_options.enable_minmax_tuning:
            shape = grouped.shape[:-1]
            tunables["min_scale"] = torch.nn.Parameter(torch.ones(shape, device=weight.device, dtype=torch.float32))
            tunables["max_scale"] = torch.nn.Parameter(torch.ones(shape, device=weight.device, dtype=torch.float32))
        return tunables

    def qdq(self, weight, state, *, tunables, materialize=False):
        quantized, scales, zero_point = progressive_quant_fp8_int4(
            weight,
            bits=self.spec.bits,
            group_size=self.spec.group_size,
            v=tunables.get("value", 0),
            min_scale=tunables.get("min_scale", 1.0),
            max_scale=tunables.get("max_scale", 1.0),
            q_scale_thresh=self.spec.q_scale_thresh,
        )
        from auto_round.data_type.base import WeightQuantizationResult

        return WeightQuantizationResult(
            quantized,
            scales["scale"] if materialize else None,
            zero_point if materialize else None,
            scales["bf16_to_fp8_scale"] if materialize else None,
        )

    @staticmethod
    def apply_result(module, result):
        if result.scale is None:
            raise ValueError("Progressive FP8 weight result was not materialized")
        module.weight.data.copy_(result.weight)
        module.scale = result.scale.reshape(result.weight.shape[0], -1).cpu()
        module.zp = result.zero_point
        module.w_bf16_to_fp8_scale = result.metadata.cpu()
