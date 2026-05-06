# Copyright (c) 2026 Intel Corporation
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

from auto_round.modeling.fused_moe.replace_modules import materialize_model_, safe_to_cpu_
from auto_round.utils import (
    is_quantized_input_module,
)


def _infer_last_cache_name(block_names, layer_names=None, requested_last_cache_name=None):
    """The latest required cache layer for early-stop forward.

    If there are multiple cache targets, return ``None`` and let runtime
    hooks stop only after all targets are observed in real forward execution.
    """
    if layer_names is None:
        layer_names = []

    if requested_last_cache_name is not None:
        return requested_last_cache_name

    cache_targets = list(block_names) + list(layer_names)
    if len(cache_targets) == 1:
        return cache_targets[0]

    # return None here to enable the logic in _should_stop_cache_forward
    return None


def _update_inputs(inputs: dict, q_inputs: dict) -> tuple[dict, torch.Tensor]:
    from auto_round.context.model import ModelContext

    model_context = ModelContext()
    if model_context.is_diffusion:
        # flux transformer model's blocks will update hidden_states and encoder_hidden_states
        input_id_str = [key for key in inputs.keys() if "hidden_state" in key]
        if q_inputs is not None:
            q_inputs = {k: q_inputs.pop(k, None) for k in input_id_str}
        return inputs, q_inputs

    keys = inputs.keys()
    input_id_str = [key for key in keys if key.startswith("hidden_state")]
    if len(input_id_str) != 1:
        raise RuntimeError(
            "hidden_states arg mismatch error," "please raise an issue in https://github.com/intel/auto-round/issues"
        )
    inputs["input_ids"] = inputs.pop(input_id_str[0], None)
    if q_inputs is not None:
        q_inputs = q_inputs.pop(input_id_str[0], None)
    return inputs, q_inputs
