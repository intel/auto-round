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
import json
special_states_dim_tuple = ("chatglm",)
shareable_keywords = ("position_ids", "cache_position", "position_embeddings")

def check_hidden_state_dim(model, positional_args):
    """Checks the dimensionality of the hidden states.

    Args:
        positional_args: The positional arguments.

    Returns:
        int: 1 if the model type is 'chatglm' and positional arguments are not None, 0 otherwise.
    """
    is_special = False
    for key in special_states_dim_tuple:
        if hasattr(model, "config") and key in model.config.model_type:
            is_special = True
            break
    return int(is_special and positional_args is not None)


def get_cache_data(batch_size, data, data_name):
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
                
