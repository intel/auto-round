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


class _FakeDeocodingLayer(torch.nn.Module):
    def __init__(self, norm_layer, lm_head):
        super().__init__()
        self.norm = norm_layer
        self.lm_head = lm_head

    def forward(self, hidden_states, **kwargs):
        # hidden_states = self.norm(hidden_states)
        # breakpoint()
        hidden_states = self.norm(hidden_states)
        # return BaseModelOutputWithPast(
        #     last_hidden_state=hidden_states,
        #     past_key_values=past_key_values,
        # )
        # outputs = BaseModelOutputWithPast(
        #     last_hidden_state=hidden_states,
        #     past_key_values=past_key_values,
        # )
        logits_to_keep = 1
        # hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        return logits


from auto_round.utils import logger


def wrap_lm_head(original_model):
    for name, module in original_model.named_modules():
        if hasattr(module, "lm_head"):
            lm_head = module.lm_head
            model = module.model
            norm_layer = model.norm if hasattr(model, "norm") else model.final_layer_norm
            fake_decoding_layer = _FakeDeocodingLayer(norm_layer, lm_head)
            model.layers.append(fake_decoding_layer)
            model.config.num_hidden_layers += 1
            logger.info(f"Replaced lm_head in module {name} with _FakeDecodingLayer.")
            break


def clean_norm_in_fake_decoding_layer(model):
    for name, module in model.named_modules():
        if isinstance(module, _FakeDeocodingLayer):
            del module.norm
            logger.info(f"Cleaned norm layer in _FakeDecodingLayer in module {name}.")
            break
