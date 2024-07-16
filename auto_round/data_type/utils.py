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
from auto_round.data_type.register import QUANT_FUNC_WITH_DTYPE


def get_quant_func(dtype, bits, sym):
    """Retrieve the quantization function based on data type, bit width, and symmetry.

       This function returns the appropriate quantization function from the QUANT_FUNC_WITH_DTYPE
       dictionary based on the provided data type (`dtype`), bit width (`bits`), and whether
       the quantization is symmetric (`sym`). If the function does not exist, it asserts False.

       Args:
           dtype (str): The data type for the quantization (e.g., 'int', 'mxfp4').
           bits (int): The bit width for the quantization (e.g., 2,4,8).
           sym (bool): A flag indicating whether the quantization is symmetric (True) or asymmetric (False).

       Returns:
           function: The quantization function corresponding to the specified parameters.
    """
    key = dtype
    if key in QUANT_FUNC_WITH_DTYPE.keys():
        return QUANT_FUNC_WITH_DTYPE[key], key

    if sym:
        key = dtype + "_sym"
    else:
        key = dtype + "_asym"

    if key in QUANT_FUNC_WITH_DTYPE.keys():
        return QUANT_FUNC_WITH_DTYPE[key], key

    ##need to add bits
    if sym:
        key = dtype + str(bits) + "_sym"
    else:
        key = dtype + str(bits) + "_asym"

    if key in QUANT_FUNC_WITH_DTYPE.keys():
        return QUANT_FUNC_WITH_DTYPE[key], key

    if sym:
        key = dtype + str(bits)
    else:
        key = dtype + str(bits)

    if key in QUANT_FUNC_WITH_DTYPE.keys():
        return QUANT_FUNC_WITH_DTYPE[key], key

    assert False, f"{dtype} is not supported"


def round_ste(x: torch.Tensor):
    """Straight-Through Estimator for rounding.

    Args:
        x: torch.Tensor

    Returns:
        torch.Tensor
    """
    return (x.round() - x).detach() + x


def floor_ste(x: torch.Tensor):
    """Straight-Through Estimator for floor.

    Args:
        x: torch.Tensor

    Returns:
        torch.Tensor
    """
    return (x.floor() - x).detach() + x
