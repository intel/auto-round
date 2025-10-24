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
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple, Union

import torch


def convert_dtype_str2torch(str_dtype):
    """Converts a string dtype to its corresponding PyTorch dtype.

    Args:
        str_dtype (str): The string representation of the dtype.

    Returns:
        torch.dtype: The PyTorch dtype.

    Raises:
        ValueError: If the input str_dtype is unsupported.
    """
    if isinstance(str_dtype, torch.dtype) or str_dtype is None:
        return str_dtype
    if str_dtype == "int8":
        return torch.int8
    elif str_dtype == "fp32" or str_dtype == "float32" or str_dtype == "auto":
        return torch.float
    elif str_dtype == "fp16" or str_dtype == "float16":
        return torch.float16
    elif str_dtype == "bf16" or str_dtype == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported string dtype '{str_dtype}' for conversion to torch dtype.")


def convert_dtype_torch2str(dtype):
    """Converts a PyTorch dtype to its corresponding string representation.

    Args:
        dtype: PyTorch dtype or str. The dtype to convert.

    Returns:
        str: The string representation of the dtype.

    Raises:
        ValueError: If the input dtype is unsupported.
    """
    if isinstance(dtype, str) or dtype is None:
        return dtype
    if dtype == torch.int8:
        return "int8"
    elif dtype == torch.float:
        return "fp32"
    elif dtype == torch.float16:
        return "fp16"
    elif dtype == torch.bfloat16:
        return "bf16"
    elif isinstance(dtype, str) and dtype in ["int8", "fp32", "fp16", "bf16"]:
        return dtype
    else:
        raise ValueError(f"Unsupported PyTorch dtype '{dtype}' for conversion to string dtype.")


def convert_dtype_torch2str_hf(dtype):
    """Converts a PyTorch dtype to its corresponding huggingface string dtype, e.g. torch.float32 -> 'float32'.

    Args:
        dtype: PyTorch dtype or str. The dtype to convert.

    Returns:
         str: The string representation of the dtype.

    Raises:
        ValueError: If the input str_dtype is unsupported.
    """
    if dtype is None:
        return dtype
    if isinstance(dtype, str):
        if "float" not in dtype and "int" not in dtype:
            dtype = convert_dtype_str2torch(dtype)
        else:
            return dtype
    str_dtype = str(dtype)
    if "." not in str_dtype:
        raise ValueError(f"Unsupported pytorch dtype '{dtype}' for conversion to huggingface str dtype")
    str_dtype = str_dtype.split(".")[1]
    return str_dtype


class BackendDataType(str, Enum):
    STANDARD_FP = "fp"
    MX_FP = "mx_fp"
    NV_FP = "nv_fp"


def is_standard_fp(backend):
    backend = backend.lower()
    return BackendDataType.STANDARD_FP in backend and not is_mx_fp(backend) and not is_nv_fp(backend)


def is_mx_fp(backend):
    backend = backend.lower()
    return BackendDataType.MX_FP in backend


def is_nv_fp(backend):
    backend = backend.lower()
    return BackendDataType.NV_FP in backend


def _is_weight_fp8_activation_static_fp8(
    bit: int, group_size: int, sym: bool, data_type: str, act_dynamic: bool
) -> bool:
    return bit == 8 and group_size == -1 and sym and data_type == "fp" and not act_dynamic


def is_wfp8afp8(ar):
    if (
        ("fp8" in ar.act_data_type or ("fp" in ar.act_data_type and ar.act_bits == 8))
        and ("fp8" in ar.data_type or ("fp" in ar.data_type and ar.bits == 8))
        and is_standard_fp(ar.act_data_type)
        and is_standard_fp(ar.data_type)
    ):
        return True
    else:
        return False


def is_static_wfp8afp8(ar_or_format: Union[str, Callable]) -> bool:
    if isinstance(ar_or_format, str):
        return "fp8_static" in ar_or_format
    if ar_or_format.act_dynamic:
        return False
    if is_wfp8afp8(ar_or_format):
        return True
    return False
