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
import traceback
from typing import Any, Callable, Optional, Union

import torch

from auto_round.compressors.utils import (
    check_need_act_calibration,
    is_nv_fp,
)
from auto_round.data_type import QUANT_FUNC_WITH_DTYPE
from auto_round.data_type.utils import reshape_pad_tensor_by_group_size
from auto_round.logger import logger
from auto_round.utils import (
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
    clear_memory,
    get_layer_names_in_block,
    get_module,
    to_device,
    to_dtype,
)


def register_act_max_hook(model: torch.nn.Module, layer_config: dict, act_group_size: int, act_data_type: str):
    def get_act_max_hook(module, input, output):
        if isinstance(input, (tuple, list)):
            input = input[0]
        if input.numel() == 0:
            return  # as no needs for act_max update
        input, _, _ = reshape_pad_tensor_by_group_size(input, act_group_size)
        act_max = torch.max(torch.abs(input), dim=-1).values
        if not hasattr(module, "act_max") or module.act_max.numel() == 0:
            module.act_max = act_max
        else:
            act_max = act_max.to(module.act_max.device)
            if is_nv_fp(act_data_type):  ## for nvfp per-tensor input_global_scale calculation usage
                module.act_max = torch.max(torch.tensor([act_max.max(), module.act_max.max()], device=act_max.device))
            else:
                module.act_max = torch.max(act_max, module.act_max)

    hook_handles = []
    # for single layers out of blocks, like lm_head
    if isinstance(model, SUPPORTED_LAYER_TYPES):
        m = model
        if (
            hasattr(m, "act_dynamic")
            and check_need_act_calibration(m.act_dynamic, m.act_data_type, m.act_bits)
            and check_to_quantized(m)
        ):
            hook = m.register_forward_hook(get_act_max_hook)
            hook_handles.append(hook)
        return hook_handles

    for n, m in model.named_modules():
        if (
            hasattr(m, "act_dynamic")
            and check_need_act_calibration(m.act_dynamic, m.act_data_type, m.act_bits)
            and check_to_quantized(m)
        ):
            hook = m.register_forward_hook(get_act_max_hook)
            hook_handles.append(hook)
            continue

        # for whole model, RTN
        if n in layer_config:
            config = layer_config[n]
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


@torch.inference_mode()
def quantize_embedding_layer(
    model: torch.nn.Module,
    layer_config: dict,
    scale_dtype: str,
    disable_opt_rtn: bool,
    device: Union[str, torch.device],
    device_list: list,
) -> bool:
    """Quantizes embedding layers in the model according to the configuration.

    This method iterates through all modules in the model, identifies embedding
    layers specified in `layer_config`, and applies the appropriate quantization
    function based on bit precision, grouping strategy, and dtype.

    Returns:
        bool: True if the quantization process completes without critical errors.
    """
    is_quantized = False
    for name, module in model.named_modules():
        # Skip non-Embedding modules or layers not in config
        if not isinstance(module, torch.nn.Embedding) or name not in layer_config:
            continue

        config = layer_config[name]

        # Skip layers that are not marked for quantization
        if not check_to_quantized(config):
            continue
        is_quantized = True
        config["scale_dtype"] = scale_dtype
        dtype = config["data_type"]

        # Determine quantization function key with symmetry/asymmetry
        if dtype not in QUANT_FUNC_WITH_DTYPE:
            dtype = f"{dtype}_{'sym' if config['sym'] else 'asym'}"

        # Optionally use optimized rounding (RTN) variant
        if not disable_opt_rtn and f"rtn_{dtype}" in QUANT_FUNC_WITH_DTYPE:
            dtype = f"rtn_{dtype}"

        quant_func = QUANT_FUNC_WITH_DTYPE[dtype]
        dtype = module.weight.dtype
        # As typically float32 are used in RTN to search scale zp,
        # to avoid cache a bf16 copy we'd better use float32
        if config.get("super_group_size", None) is not None:
            dtype = torch.float32

        # Attempt quantization on GPU, fall back to CPU if OOM
        try:
            weight, scale, zp = quant_func(
                module.weight.to(dtype=dtype, device=device),
                **{
                    k: config.get(k, None)
                    for k in ["bits", "group_size", "super_bits", "super_group_size", "scale_dtype"]
                },
            )
        except torch.OutOfMemoryError:
            cuda_error_msg = traceback.format_exc()
            try:
                logger.error(cuda_error_msg)
                logger.warning("falling back to CPU")
                weight, scale, zp = quant_func(
                    module.weight.to("cpu"),
                    **{
                        k: config.get(k, None)
                        for k in ["bits", "group_size", "super_bits", "super_group_size", "scale_dtype"]
                    },
                )
            except Exception as e:
                raise

        # Overwrite the module's weights with the quantized version
        module.weight.data.copy_(weight.cpu())

        # Attach scale and zero point (zp) to the module
        for param_name, value in zip(["scale", "zp"], [scale, zp]):
            if isinstance(value, dict):
                for k, v in value.items():
                    setattr(module, k if k == "scale" else f"w_{k}", v.cpu())
            elif isinstance(value, torch.Tensor):
                setattr(module, param_name, value.cpu())
            else:
                setattr(module, param_name, value)

        # Update config
        layer_config.setdefault(name, {}).update(config)
        del weight
        del scale
        del zp
        clear_memory(device_list=device_list)
    return is_quantized


def get_quantized_layer_names_outside_blocks(
    model: torch.nn.Module, layer_config: dict, supported_types: list, quant_block_list: list
) -> list:
    """Gets the names of quantized layers outside blocks in the model.

    Returns:
        list: List of layer names outside blocks.
    """
    if layer_config is None or len(layer_config) == 0:
        return []

    layer_names = []
    all_layers_in_block = get_layer_names_in_block(model, supported_types, quant_block_list)

    for key in layer_config.keys():
        if key in all_layers_in_block:
            continue
        layer = get_module(model, key)
        if layer is None:
            logger.error(f"could not find layer {key} in the model, exit...")
            exit(-1)
        if type(layer) in supported_types and check_to_quantized(layer_config[key]):
            layer_names.append(key)

    return layer_names


def get_non_zero_cnt(tensor: list[torch.Tensor], indices: list[int]) -> int:
    current_tensors = [tensor[i] for i in indices]
    non_zero_cnt = 0
    for t in current_tensors:
        non_zero_cnt += torch.count_nonzero(t).item()
    return non_zero_cnt


def split_inputs(inputs: dict, first_input_name: str, diffusion: bool = False) -> tuple[torch.Tensor, dict]:
    if diffusion:
        input_id_str = [key for key in inputs.keys() if "hidden_state" in key]
        input_ids = {k: inputs.pop(k, None) for k in input_id_str}
        input_others = inputs
        return input_ids, input_others
    else:
        input_ids = inputs[first_input_name]
        inputs.pop(first_input_name, None)
        input_others = inputs
        return input_ids, input_others


def preprocess_block_inputs(
    inputs,
    device_list: list,
    first_input_name="input_ids",
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
    cache_device: Union[str, torch.device] = "cpu",
    diffusion: bool = False,
):
    input_ids, input_others = split_inputs(inputs, first_input_name, diffusion=diffusion)
    clear_memory(device_list=device_list)
    input_ids = to_device(input_ids, cache_device)
    input_others = to_device(input_others, cache_device)
    # As in calibration phase, we may use bf16 for calibration due to low_gpu_memory usage

    tmp_dtype = amp_dtype if amp else torch.float32
    input_ids = to_dtype(input_ids, tmp_dtype)

    for key in input_others.keys():
        if isinstance(input_others[key], torch.Tensor) and (
            input_others[key].dtype == torch.float16 or input_others[key].dtype == torch.bfloat16
        ):
            input_others[key] = input_others[key].to(tmp_dtype)
        elif isinstance(input_others[key], list):
            for i in range(len(input_others[key])):
                to_dtype(input_others[key][i], tmp_dtype)
    return input_ids, input_others
