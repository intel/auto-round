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


def check_share_attention_mask(model, hidden_states, attention_mask=None, **kwargs):
    if attention_mask is None or not isinstance(hidden_states, torch.Tensor):
        return False
    is_baichuan = bool(hasattr(model, 'config') and 'baichuan' in model.config.model_type)
    return bool(attention_mask.shape[0] != hidden_states.shape[0] and is_baichuan)
    
    
def check_hidden_state_dim(model, positional_args):
    is_chatglm = hasattr(model, "config") and "chatglm" in model.config.model_type
    return int(is_chatglm and positional_args is not None)
    

