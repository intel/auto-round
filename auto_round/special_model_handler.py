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

from .utils import logger, torch

share_attention_mask_tuple = ("baichuan",)
special_states_dim_tuple = ("chatglm",)


def check_share_attention_mask(model, hidden_states, attention_mask=None, **kwargs):
    """Checks if the attention mask states of the hidden states are shared in the model.

    Args:
        hidden_states (torch.Tensor): The hidden states of the model.
        attention_mask (torch.Tensor, optional): The attention mask tensor. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        bool: True if attention mask is shared in the model, False otherwise.
    """
    if attention_mask is None or not isinstance(hidden_states, torch.Tensor):
        return False
    is_special = False
    for key in share_attention_mask_tuple:
        if hasattr(model, "config") and key in model.config.model_type:
            is_special = True
            break
    return bool(is_special and attention_mask.shape[0] != hidden_states.shape[0])


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
