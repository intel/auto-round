# Copyright (c) 2023 Intel Corporation
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
from collections import UserDict
special_states_dim_tuple = ("chatglm",) # input_dim is not the default dimension 0
shareable_keywords = ("position_ids", "cache_position", "position_embeddings")
mllms_with_limited_bs = ("llava", "qwen2-vl", "phi3_v", "mllama") # Limitations on batch_size
skippable_cache_keys = ("past_key_value",)

def to_device(input, device=torch.device("cpu")):
    """Moves input data to the specified device.

    Args:
    input: The input data to be moved.
    device: The target device.

    Returns:
    The input data on the specified device.
    """
    if input is None:
        return None
    if isinstance(input, torch.Tensor):
        return input.to(device)
    if isinstance(input, dict) or isinstance(input, UserDict):
        for inp in input.keys():
            input[inp] = to_device(input[inp], device)

    elif isinstance(input, list) or isinstance(input, tuple):
        if len(input) == 0:
            return input
        input_res = []
        for inp in input:
            input_res.append(to_device(inp, device))
        if isinstance(input, tuple):
            input_res = tuple(input_res)
        input = input_res

    return input


def check_hidden_state_dim(model, positional_inputs):
    """Check the concatenable dimension of hidden states.

    Args:
        positional_inputs: The positional arguments.

    Returns:
        int: 1 if the model type is 'chatglm' and positional arguments are not None, 0 otherwise.
    """
    is_special = False
    for key in special_states_dim_tuple:
        if hasattr(model, "config") and key in model.config.model_type:
            is_special = True
            break
    return int(is_special and positional_inputs is not None)


def special_model_init(model, positional_inputs, inputs):
    """
    Initializes special model inputs by adding positional inputs if missing.

    Args:
        model: The model instance being initialized.
        positional_inputs (list): List of positional inputs to add to inputs.
        inputs (dict): Dictionary of model inputs.
    
    Modifies:
        inputs (dict): Adds "positional_inputs" key if not present.
    """
    if "positional_inputs" not in inputs: # for chatglm Series
        inputs["positional_inputs"] = []
    for idx, item in enumerate(positional_inputs):
        inputs["positional_inputs"] = to_device(positional_inputs)


def reset_params(inputs):
    """
    Resets specific input parameters to avoid saving the key-value cache during fine-tuning.
    
    Args:
        inputs (dict): Dictionary of model inputs.
    
    Modifies:
        inputs (dict): Sets "use_cache" to False if the key is present.
    """
    if "use_cache" in inputs.keys(): # Not storing kv cache
        inputs['use_cache'] = False
        

def skip_keywards_hint(key):
    """
    Prints a reminder if a key is not stored during quantization fine-tuning.
    """
    for cache_key in skippable_cache_keys:
        if cache_key not in key:
            return True
    return False
            

def check_model_batch(model, batch_size, gradient_accumulate_steps):
    """
    Checks model configuration to determine if it's necessary to limit bs to avoid potential input shape mismatches.
    """
    for key in mllms_with_limited_bs:
        if hasattr(model, "config") and key in model.config.model_type and batch_size != 1:
            accumulate_steps = batch_size * gradient_accumulate_steps
            raise RuntimeError("To avoid the tensor concat mismatch problem, please modify parameters to " \
                    f"batch_size=1. As an alternative, you can set the gradient_accumulate_steps={accumulate_steps}")
                
