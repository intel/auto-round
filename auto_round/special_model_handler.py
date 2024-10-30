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
special_states_dim_tuple = ("chatglm",) # input_dim is not the default dimension 0
shareable_keywords = ("position_ids", "cache_position", "position_embeddings")
mllm_special_model = ("llava", "qwen2-vl", "phi3_v", "mllama") # Limitations on batch_size


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
    if 'past_key_value' not in key:
        print(f"Please note that this '{key}' key is not currently used in quantization fine-tuning.")
        

def check_model_batch(model, batch):
    """
    Checks model configuration to determine if it's necessary to limit bs to 1 to avoid potential input shape mismatches.

    Args:
        model: The model instance to check.
        batch: Batch data to return or modify.

    Returns:
        int or original batch
    """
    for key in mllm_special_model:
        if hasattr(model, "config") and key in model.config.model_type:
            return 1
    return batch


def get_cache_data(batch_size, data, data_name):
    """
    Processes store data for batch handling, reshaping if necessary.

    Args:
        batch_size (int): The size of the batch.
        data: The data value to store, potentially for caching.
        data_name (str): Name of the data.

    Returns:
        Processed data or None
    """
    new_data = data
    if batch_size <= 1:
        return new_data
    if data_name in shareable_keywords:
        return None
    if "alibi" in data_name:
        if isinstance(data, torch.Tensor):
            alibi = data
            alibi = alibi.reshape(batch_size, -1, alibi.shape[1], alibi.shape[2])
            new_data = alibi
    return new_data
                
