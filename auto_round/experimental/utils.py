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

from typing import Any

import torch

from auto_round.compressors.utils import is_nv_fp, is_mx_fp
from auto_round.experimental.transform.hadamard_config import HadamardConfig
from auto_round.experimental.transform.hadamards import HADAMARDS
from auto_round.utils import logger

SUPPORTED_QUANTIZATION_SCHEMES = ["MXFP8", "MXFP4", "NVFP4"]


def per_tensor_fp8_qdq(
    tensor: torch.Tensor, tensor_max: None | torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor]:
    from auto_round.data_type.fp8 import quant_fp8_sym

    qdq_tensor, scale, _ = quant_fp8_sym(tensor, max_scale=1.0, tensor_max=tensor_max, group_size=0, v=0)
    return qdq_tensor, scale


@torch.compiler.disable
def update_parameter_data(module: torch.nn.Module, new_val: torch.Tensor, name: str):
    """
    Update the data of a parameter in a module.
    If the parameter does not exist, it will be created.
    """
    if hasattr(module, name):
        param = getattr(module, name)
        if isinstance(param, torch.nn.Parameter):
            param.data.copy_(new_val)
        else:
            module.register_parameter(name, torch.nn.Parameter(new_val))
    else:
        logger.warning_once(
            "Parameter %s not found in module %s, creating new parameter."
            % (name, module.__class__.__name__ + str(getattr(module, "layer_idx", "")))
        )
        module.register_parameter(name, torch.nn.Parameter(new_val))


def normalize_static_kv_dtype(static_kv_dtype: str | torch.dtype) -> torch.dtype:
    valid_dtype_name_lst = ["float16", "bfloat16", "fp8", "float32", "float"]
    valid_torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "fp8": torch.float8_e4m3fn,
        "float8_e4m3fn": torch.float8_e4m3fn,
        "float32": torch.float32,
        "float": torch.float32,  # Alias for float32
    }
    if static_kv_dtype in valid_dtype_name_lst:
        new_dtype = valid_torch_dtype[static_kv_dtype]
    elif static_kv_dtype in valid_torch_dtype.values():
        new_dtype = static_kv_dtype
    else:
        raise ValueError(
            f"Invalid static kv dtype: {static_kv_dtype}. "
            # f"Valid options are: {', '.join(valid_dtype_name_lst  + list(valid_torch_dtype.values()))}."
        )
    return new_dtype


def is_attention_module(module: torch.nn.Module):
    # FIXME: Handle this better.
    return (
        "attention" in module.__class__.__name__.lower()
        and module.__class__.__name__ != "Llama4VisionAttention"  # llama4 vision attention doesn't have cache
        and (
            hasattr(module, "k_proj")
            or hasattr(module, "v_proj")
            or hasattr(module, "qkv_proj")
            or hasattr(module, "kv_b_proj")  # for DeepSpeed
        )
    )


def _clean_param_or_buff_if_exists(module: torch.nn.Module, name_tuple: tuple[str, ...]):
    """
    Deletes parameters or buffers from a module if they exist.

    :param module: module to delete parameters/buffers from
    :param name_tuple: tuple of parameter/buffer names to delete
    """
    for name in name_tuple:
        if hasattr(module, name):
            try:
                delattr(module, name)
            except Exception as e:
                logger.warning(f"Could not delete {name} from module {module}: {e}")


def clean_model_parameters_and_buffers_(model: torch.nn.Module, name_tuple: tuple[str, ...]):
    """
    Cleans parameters and buffers from all modules in the model.

    :param model: model to clean parameters/buffers from
    :param name_tuple: tuple of parameter/buffer names to delete
    """
    for module in model.modules():
        _clean_param_or_buff_if_exists(module, name_tuple)


def is_triton_kernel_available(data_type: str) -> bool:
    """
    Best-effort check for whether Triton kernel path can be used.
    """
    if is_nv_fp(data_type):
        return False
    try:
        import triton  # pylint: disable=E0401
    except Exception:
        return False

    if not torch.cuda.is_available():
        return False

    try:
        from auto_round.experimental.transform.triton.mxfp4 import mxfp4_forward_kernel_wrapper  # pylint: disable=E0401
    except Exception:
        return False

    return True


def normalize_hadamard_config(hadamard_config: str | dict | HadamardConfig | None, data_type: str) -> dict[str, Any]:
    """
    Normalize and validate `hadamard_config`.

    Supported input types:
        - None           -> {}
        - dict           -> validated via HadamardConfig
        - HadamardConfig -> validated & converted to dict
        - str            -> shorthand for `hadamard_type` in HADAMARDS keys

    Additional behavior:
        - If block_size is not set by user:
            - mx_fp -> default block_size to 32
            - nv_fp -> default block_size to 16
            - other data types -> emit a warning
        - If block_size is set but does not match the recommended value:
            - mx_fp expects 32
            - nv_fp expects 16
            - emit a warning
    """


    def _apply_data_type_block_size(cfg_dict: dict[str, Any], block_size_explicitly_set: bool) -> dict[str, Any]:
        block_size = cfg_dict.get("block_size")

        if not block_size_explicitly_set or block_size is None:
            if is_mx_fp(data_type):
                cfg_dict["block_size"] = 32
                logger.warning("block_size is not set for data_type 'mx_fp'; defaulting to 32.")
            elif is_nv_fp(data_type):
                cfg_dict["block_size"] = 16
                logger.warning("block_size is not set for data_type 'nv_fp'; defaulting to 16.")
            else:
                logger.warning(
                    f"block_size is not set and cannot be inferred for data_type {data_type!r}; "
                    "please set block_size explicitly in hadamard_config if needed."
                )
        else:
            if is_mx_fp(data_type) and block_size != 32:
                logger.warning(f"data_type is 'mx_fp' but block_size={block_size}; recommended value is 32.")
            elif is_nv_fp(data_type) and block_size != 16:
                logger.warning(f"data_type is 'nv_fp' but block_size={block_size}; recommended value is 16.")

        return cfg_dict

    # 1) None -> {}
    if hadamard_config is None:
        return {}

    # 2) HadamardConfig instance
    if isinstance(hadamard_config, HadamardConfig):
        raw_cfg_dict = hadamard_config.model_dump(exclude_unset=True)
        block_size_explicitly_set = "block_size" in raw_cfg_dict

        cfg_dict = dict(raw_cfg_dict)
        cfg_dict = _apply_data_type_block_size(cfg_dict, block_size_explicitly_set)

        try:
            return HadamardConfig.model_validate(cfg_dict).model_dump()
        except Exception as e:
            raise ValueError(f"Invalid HadamardConfig: {e}") from e

    # 3) dict
    if isinstance(hadamard_config, dict):
        block_size_explicitly_set = "block_size" in hadamard_config

        cfg_dict = dict(hadamard_config)
        cfg_dict = _apply_data_type_block_size(cfg_dict, block_size_explicitly_set)

        try:
            return HadamardConfig.model_validate(cfg_dict).model_dump()
        except Exception as e:
            raise ValueError(f"Invalid hadamard_config dict: {e}") from e

    # 4) str -> shorthand for hadamard_type
    if isinstance(hadamard_config, str):
        key = hadamard_config.strip()
        if not key:
            return {}

        if key == "default":
            cfg_dict = {}
            cfg_dict = _apply_data_type_block_size(cfg_dict, block_size_explicitly_set=False)
            try:
                return HadamardConfig.model_validate(cfg_dict).model_dump()
            except Exception as e:
                raise ValueError(f"Invalid default hadamard_config after data_type adjustment: {e}") from e

        if key not in HADAMARDS:
            raise ValueError(f"Invalid hadamard_config string: {key!r}. Expected one of {sorted(HADAMARDS.keys())}.")

        cfg_dict = {"hadamard_type": key}
        cfg_dict = _apply_data_type_block_size(cfg_dict, block_size_explicitly_set=False)

        try:
            return HadamardConfig.model_validate(cfg_dict).model_dump()
        except Exception as e:
            raise ValueError(f"hadamard_config built from string {key!r} is invalid for HadamardConfig: {e}") from e

    raise TypeError(
        "hadamard_config must be one of: None, dict, HadamardConfig, or str "
        f"(got {type(hadamard_config).__name__})"
    )

def check_supported_schemes(scheme: str):
    if scheme not in SUPPORTED_QUANTIZATION_SCHEMES:
        raise ValueError(
            f"Unsupported quantization scheme: {scheme}. Currently {SUPPORTED_QUANTIZATION_SCHEMES} are supported."
        )
