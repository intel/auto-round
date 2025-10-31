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

from enum import Enum, auto
from typing import Union

import torch

from .utils import _to_mx_rceil, get_fp_scale


class ScaleCalculationMode(Enum):
    """
    Enum representing the different methods for calculating MX block scaling.
    There are three methods available:
    FLOOR: This method is recommended by the OCP MX Spec 1.0 and uses X = 2^floor(log2(max_abs(v))-max_exp).
           It result in overflow issues for large values and bad for gradient quantization.
    CEIL: This method avoids overflow issues, but small values may shift to 0 due to a large scaling factor.
           It uses X = 2^ceil(log2(max_abs(v))-max_exp).
    EVEN: This method is a trade-off between Option 1 and Option 2. It uses X = 2^(floor(log2(rounding(max_abs(v)))-max_exp)).
           It provides better accuracy for MX4 training compared to FLOOR and CEIL.
    RCEIL: The method is to apply ceil to the ratio of max_abs(v) and max_pos.
           This method's detail is described in https://docs.nvidia.com/cuda/cublas/index.html#d-block-quantization
           Section "Computing scaling and conversion factors for FP8 with UE8M0 scales"

    By default, we use the EVEN method for better accuracy.
    """

    FLOOR = auto()
    CEIL = auto()
    EVEN = auto()
    RCEIL = auto()


# This is conceptually an enum of non-core dtypes
# TODO(future PR): change to a cleaner way to represent this without
# regressing torch.compile and while keeping things readable.
DTYPE_FP6_E3M2 = "fp6_e3m2"
DTYPE_FP6_E2M3 = "fp6_e2m3"

# Supported element dtypes
# TODO(future PR): add support for MX int8
SUPPORTED_ELEM_DTYPES = [
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    DTYPE_FP6_E2M3,
    DTYPE_FP6_E3M2,
]


F8E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0
F8E5M2_MAX = torch.finfo(torch.float8_e5m2).max  # 57344.0

F8E4M3_MAX_POW2 = 8  # 256
F8E5M2_MAX_POW2 = 15  # 32768
F6_E2M3_MAX_POW2 = 2  # 4
F6_E3M2_MAX_POW2 = 4  # 16
F4_E2M1_MAX_POW2 = 2  # 4

E8M0_EXPONENT_BIAS = 127
E8M0_EXPONENT_NAN_VAL = 255

F32_EXP_BIAS = 127
BF16_EXP_BIAS = 127
F6_E2M3_EXP_BIAS = 1
F6_E3M2_EXP_BIAS = 3
F4_E2M1_EXP_BIAS = 1

F32_MIN_NORMAL = 2 ** (-F32_EXP_BIAS + 1)

F6_E2M3_MAX = 7.5
F6_E2M3_MIN_NORMAL = 1.0
F6_E2M3_MAX_INT = 31  # integer corresponding to 0b00011111

F6_E3M2_MAX = 28.0
F6_E3M2_MIN_NORMAL = 0.25
F6_E3M2_MAX_INT = 31  # integer corresponding to 0b00011111

F4_E2M1_MAX = 6.0
F4_E2M1_MIN_NORMAL = 1.0
F4_E2M1_MAX_INT = 7

BLOCK_SIZE_DEFAULT = 32


# TODO(later): read from somewhere else?
SBITS, EBITS_F32, MBITS_F32 = 1, 8, 23
EBITS_BF16, MBITS_BF16 = 8, 7
EBITS_F4_E2M1, MBITS_F4_E2M1 = 2, 1
EBITS_F6_E2M3, MBITS_F6_E2M3 = 2, 3
EBITS_F6_E3M2, MBITS_F6_E3M2 = 3, 2
EBITS_F8_E4M3, MBITS_F8_E4M3 = 4, 3
EBITS_F8_E5M2, MBITS_F8_E5M2 = 5, 2


def to_mx(
    data_hp: torch.Tensor,
    elem_dtype: Union[torch.dtype, str],
    block_size: int,
    scaling_mode: ScaleCalculationMode = ScaleCalculationMode.FLOOR,
    pack_fp6: bool = False,
):
    """
    Takes a high precision tensor and converts to MX scale and raw data, in
    naive layout (scale and raw data are separate tensors).
    """

    assert data_hp.dtype in (
        torch.bfloat16,
        torch.float,
    ), f"{data_hp.dtype} is not supported yet"
    # TODO(future PR): consider supporting padding
    assert data_hp.numel() % block_size == 0, "unsupported"
    assert data_hp.is_contiguous(), "unsupported"
    assert elem_dtype in SUPPORTED_ELEM_DTYPES, "unsupported"

    # calculate the scale in e8m0 format

    orig_shape = data_hp.shape
    data_hp = data_hp.reshape(-1, block_size)

    # find max value of the data
    # Note: this only implements the `minimally supported` version of
    # https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    # section 6.3.
    max_abs = torch.amax(torch.abs(data_hp), 1)

    # Add an epsilon to prevent the log2 function call for returning -inf
    # where the values are zero.
    eps = F32_MIN_NORMAL * (max_abs == 0).type(max_abs.dtype)

    # Set X to be the largest power-of-two less than or equal to
    # max_abs(v), divided by the largest power of two representable
    # in the element data type, and get the mbits at the same time
    if elem_dtype == torch.float8_e4m3fn:
        target_max_pow2 = F8E4M3_MAX_POW2
        mbits = MBITS_F8_E4M3
        max_pos = F8E4M3_MAX
    else:
        raise AssertionError("unsupported element dtype")

    if scaling_mode == ScaleCalculationMode.RCEIL:
        scale_e8m0_biased, data_lp = _to_mx_rceil(data_hp, max_abs, max_pos)
    else:
        if data_hp.dtype is torch.float32:
            hp_int_dtype = torch.int32
            hp_mbits = MBITS_F32
            hp_ebits = EBITS_F32
            hp_exp_bias = F32_EXP_BIAS
        else:
            assert data_hp.dtype is torch.bfloat16
            hp_int_dtype = torch.int16
            hp_mbits = MBITS_BF16
            hp_ebits = EBITS_BF16
            hp_exp_bias = BF16_EXP_BIAS

        # rounding before calculating the largest power of 2
        # X = 2^(floor(log2(rounding(max_abs(v)))-max_exp))
        if scaling_mode == ScaleCalculationMode.EVEN:
            nan_mask = torch.isnan(max_abs)
            max_abs = max_abs.view(hp_int_dtype)
            val_to_add = 1 << (hp_mbits - mbits - 1)
            mask = ((1 << (hp_ebits + SBITS)) - 1) << hp_mbits
            max_abs = (max_abs + val_to_add) & mask
            max_abs = max_abs.view(data_hp.dtype)
            max_abs[nan_mask] = torch.tensor(float("nan"), device=max_abs.device, dtype=max_abs.dtype)

        # Calculate the scale for different modes
        max_abs_int32 = (max_abs + eps).view(hp_int_dtype)
        extracted_pow2 = ((max_abs_int32 >> hp_mbits) & 0b11111111) - hp_exp_bias

        if scaling_mode in (ScaleCalculationMode.FLOOR, ScaleCalculationMode.EVEN):
            scale_e8m0_unbiased = extracted_pow2 - target_max_pow2
        elif scaling_mode == ScaleCalculationMode.CEIL:
            # round up: add one to scale if the mantissa is larger than 0
            # 0x7FFFFF is equal to 23 ones
            mantissa_gt_one = (max_abs_int32 & 0x7FFFFF) > 0
            extracted_pow2 += mantissa_gt_one
            scale_e8m0_unbiased = extracted_pow2 - target_max_pow2
        else:
            raise AssertionError("unsupported scaling calculation mode")

        # Clamp to exponents that can be represented in e8m0
        # add one to positive range to capture NaNs
        scale_e8m0_unbiased = torch.clamp(scale_e8m0_unbiased, min=-E8M0_EXPONENT_BIAS, max=E8M0_EXPONENT_BIAS + 1)

        # Create the biased e8m0 representation and cast it to 8 bits
        scale_e8m0_biased = scale_e8m0_unbiased + E8M0_EXPONENT_BIAS
        scale_e8m0_biased = scale_e8m0_biased.to(torch.uint8)

        # Conversion to torch.uint8 sets NaN values to 0, fix this by
        # explicitly setting known NaN values to 255
        scale_e8m0_biased = torch.where(
            torch.isnan(max_abs),
            E8M0_EXPONENT_NAN_VAL,
            scale_e8m0_biased,
        )

        # For now, calculate the scale in floating point.
        scale_fp32 = (scale_e8m0_biased.to(torch.int32) << MBITS_F32).view(torch.float32)

        # Today, 2**-127 returns 0 in compile+inductor+triton because it is in the
        # float32 denormal range. For now, manually adjust the fp scale. This is
        # relevant if all of the incoming block values are zeroes.
        # See https://github.com/pytorch/pytorch/issues/125557 for details.
        # Note: it would be more correct to set the minimum to 2**-127, but this
        # does not work in triton either as it looks like subnormal value handling
        # has some gaps.  So, for now just set to the minimum normal value.
        scale_fp32 = torch.clamp(scale_fp32, min=F32_MIN_NORMAL)

        # scale and saturated cast the data elements to max of target dtype
        data_lp = data_hp / scale_fp32.unsqueeze(1)

        if elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and not torch._dynamo.is_compiling():
            # As of 20250317, the Pytorch eager mode cast to `torch.float8_e4m3fn`
            # is unsaturated. This cast is saturated in triton. If we are compute bound,
            # we see a speedup if we remove this redundant clamp if we are compiling
            # to triton.
            # TODO(#1912): make the saturated cast work in eager mode and remove this
            # workaround.
            data_lp = torch.clamp(data_lp, min=-1 * max_pos, max=max_pos)

    # cast to target dtype
    if elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        data_lp = data_lp.to(elem_dtype)
        # need to reshape at the end to help inductor fuse things
        data_lp = data_lp.reshape(orig_shape)
    else:
        raise AssertionError("unsupported")

    # scale_e8m0_biased = scale_e8m0_biased.view(torch.float8_e8m0fnu)
    return scale_e8m0_biased, data_lp


def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


def pack_uint4(uint8_data: torch.Tensor) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[::2] << 4 | uint8_data[1::2]).view(down_size(shape))
