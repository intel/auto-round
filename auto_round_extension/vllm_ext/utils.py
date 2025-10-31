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

from auto_round.schemes import QuantizationScheme

E8M0_EXPONENT_BIAS = 127
E8M0_EXPONENT_NAN_VAL = 255


def get_scheme(quant_config, prefix: str):
    # Check extra_config first
    layer_schemes = quant_config.layer_schemes
    # FIXME: make more robust
    for name, scheme in layer_schemes.items():
        if prefix.startswith(name):
            return scheme
    # If not found, use default
    return quant_config.quant_scheme


def check_quantized(weight_bits: int) -> bool:
    return weight_bits < 16


def _is_mxfp4_w4a4(scheme: QuantizationScheme):
    # FIXME: below impl is incomplete
    return scheme.bits == 4 and scheme.group_size == 32


def _is_mxfp8_w8a8(scheme: QuantizationScheme):
    # FIXME: below impl is incomplete
    return scheme.bits == 8 and scheme.group_size == 32


def get_fp_scale(scale_e8m0):
    scale_e8m0 = scale_e8m0.view(torch.uint8)
    s_offset = scale_e8m0.to(torch.int16) - E8M0_EXPONENT_BIAS
    # TODO(later): it would be nice if there was a way to do the 2^x operation
    # in PyTorch without creating a tensor of twos
    two = torch.full(s_offset.size(), 2.0, device=scale_e8m0.device)
    # pow(two, s_offset) can be out of range of floating point formats.
    # TODO(later): handle this for float16 if we decide to support float16
    # scales.
    s_fp = torch.pow(two, s_offset)

    # If a block exponent was 255, set values of that block to NaN
    s_fp = torch.where(scale_e8m0 != E8M0_EXPONENT_NAN_VAL, s_fp, float("nan"))

    return s_fp


def _to_mx_rceil(
    data_hp: torch.Tensor,
    max_abs: torch.Tensor,
    max_pos: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    A prototype implementation of MXFP scale factor derivation method described in
    https://docs.nvidia.com/cuda/cublas/#d-block-quantization

    For Nvidia GPU with Blackwell+ architecture, the scale factor derivation method
    could be accelerated by the `cvt.rp.satfinite.ue8m0x2.f32` instruction.

    Args:
        data_hp: High precision data.
        max_abs: Maximum absolute value for data_hp along specified dimension/block_size.
        max_pos: The maximum value of the low precision data type.

    Returns:
        exponent: The biased exponent with dtype E8M0 in uint8 container.
        data_lp: The targeted low precision data, in high precision container
            (requires cast to low precision data type).
    """
    descale = max_abs / max_pos
    # TODO: nan/inf needs to be set for any value
    # of nan/inf in input not just amax.
    exponent = torch.where(
        torch.isnan(descale),
        0xFF,  # Handle biased exponent for nan
        # NOTE: descale < (torch.finfo(torch.float32).smallest_normal / 2) is handled through clamping
        (
            torch.clamp(
                torch.ceil(torch.log2(descale)),
                min=-E8M0_EXPONENT_BIAS,
                max=E8M0_EXPONENT_BIAS,
            )
            + E8M0_EXPONENT_BIAS
        ).to(torch.uint8),
    )

    descale_fp = torch.where(exponent == 0, 1.0, torch.exp2(E8M0_EXPONENT_BIAS - exponent.to(torch.float32)))

    # scale and saturated cast the data elements to max of target dtype
    data_lp = torch.clamp(data_hp * descale_fp.unsqueeze(1), min=-1 * max_pos, max=max_pos)
    return exponent, data_lp
