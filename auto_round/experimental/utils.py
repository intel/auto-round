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

from auto_round.utils import logger


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
    return "attention" in module.__class__.__name__.lower() and (
        hasattr(module, "k_proj")
        or hasattr(module, "v_proj")
        or hasattr(module, "qkv_proj")
        or hasattr(module, "kv_b_proj")  # for DeepSpeed
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
