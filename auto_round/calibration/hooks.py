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
"""Hook factories for collecting block-level inputs during calibration.

These module-level factories can be reused by any ``Calibrator`` subclass
without inheriting from a particular Compressor.

A ``state`` object (typically the Compressor / Calibrator ``self``) is
passed in as the mutable holder of the following attributes:

- ``inputs``                       : dict[str, dict | list]
- ``quantizer``                    : has ``batch_size`` / ``batch_dim`` / ``attention_mask``
- ``has_variable_block_shape``     : bool
- ``blocks_requiring_input_ids``   : list[str]
- ``model_context``                : ``ModelContext`` (uses ``shared_cache_keys`` / ``replace_forward``)
- ``_should_stop_cache_forward``   : callable(name) -> bool   (kept on ``state`` so subclasses can override it; e.g. DiffusionMixin always returns False)
- ``to_cached_layers``             : list[str]   (only required by ``replace_forward``)
"""

from functools import partial
from typing import Callable

import torch

from auto_round.compressors_new.utils import check_skippable_keywords, init_cache, reset_params
from auto_round.logger import logger
from auto_round.utils import SUPPORTED_LAYER_TYPES, to_device


def make_block_forward_func(state, name: str) -> Callable:
    """Build a ``forward`` replacement that captures inputs for *block* ``name``.

    Mirrors the legacy ``DataDrivenCompressor._get_block_forward_func`` exactly.
    The returned function expects to be bound as ``module.forward = partial(fn, module)``.
    """

    def post_process_cache_data(batch_size, data, data_name):
        new_data = data
        if data_name in state.model_context.shared_cache_keys:
            return None
        if batch_size <= 1:
            return new_data
        if "alibi" in data_name:
            if isinstance(data, torch.Tensor):
                alibi = data
                alibi = alibi.reshape(batch_size, -1, alibi.shape[1], alibi.shape[2])
                new_data = alibi
        return new_data

    def forward(m, hidden_states=None, *positional_inputs, **kwargs):
        if name not in state.inputs:
            state.inputs[name] = {}
            init_cache(positional_inputs, state.inputs[name])

        if state.quantizer.batch_dim is None:
            state.quantizer.batch_dim = 0
            if hidden_states is not None and state.quantizer.batch_size > 1:
                if hidden_states.shape[0] > state.quantizer.batch_size:
                    state.quantizer.batch_dim = 1
                    if len(hidden_states.shape) > 1 and hidden_states.shape[1] > state.quantizer.batch_size:
                        logger.error(
                            "this model has not been supported, "
                            "please raise an issue in https://github.com/intel/auto-round/issues"
                            " or try to set the `batch_size` to 1 and "
                            "`gradient_accumulate_steps` to your current batch size."
                        )
                        exit(-1)

        if hidden_states is not None:
            kwargs["hidden_states"] = hidden_states

        for key in kwargs.keys():
            if isinstance(kwargs[key], torch.Tensor) or isinstance(kwargs[key], list) or isinstance(kwargs[key], tuple):
                if (
                    state.has_variable_block_shape
                    and name not in state.blocks_requiring_input_ids
                    and key == "hidden_states"
                ):
                    continue
                if key not in state.inputs[name].keys():  # initialization
                    data = to_device(kwargs[key], device=torch.device("cpu"))
                    if data is None or key in state.model_context.shared_cache_keys:
                        state.inputs[name][key] = data
                        continue
                    if state.quantizer.batch_size <= 1:
                        state.inputs[name][key] = [data]
                    else:
                        data = post_process_cache_data(state.quantizer.batch_size, data, key)
                        if isinstance(data, torch.Tensor):
                            state.inputs[name][key] = list(torch.split(data, 1, dim=state.quantizer.batch_dim))
                        else:
                            state.inputs[name][key] = [data]
                else:  # append cache inputs
                    new_data = post_process_cache_data(state.quantizer.batch_size, kwargs[key], key)
                    if new_data is None:  # shareable args or NoneType
                        continue
                    new_data = to_device(new_data, device=torch.device("cpu"))
                    if state.quantizer.batch_size <= 1:
                        state.inputs[name][key].append(new_data)
                    else:
                        if isinstance(new_data, torch.Tensor):
                            state.inputs[name][key].extend(
                                list(torch.split(new_data, 1, dim=state.quantizer.batch_dim))
                            )
                        else:
                            state.inputs[name][key].append(new_data)
            elif isinstance(kwargs[key], (str, bool, type(None))):
                if key not in state.inputs[name].keys():
                    state.inputs[name][key] = kwargs[key]
            else:
                # Parameters not to be cached
                if check_skippable_keywords(key):
                    logger.warning_once(
                        f"Please note that '{key}' key" " is not currently used in quantization fine-tuning."
                    )
        reset_params(state.inputs[name])

        if state._should_stop_cache_forward(name):
            raise NotImplementedError
        else:
            if hidden_states is not None:
                kwargs.pop("hidden_states")
                return m.orig_forward(hidden_states, *positional_inputs, **kwargs)
            else:
                # Currently only for Llama-3.2-Vision-Instruct Series
                return m.orig_forward(*positional_inputs, **kwargs)

    return forward


def make_layer_cache_hook(state, name: str) -> Callable:
    """Build a forward-hook that captures inputs for *layer* ``name``.

    Mirrors the legacy ``DataDrivenCompressor._get_cache_data_hook_for_layer`` exactly.
    """

    def cache_input_hook(module, inputs, outputs):
        input = inputs
        if isinstance(inputs, tuple) or isinstance(input, list):
            input = inputs[0]
        if name in state.inputs:
            state.inputs[name].extend(list(torch.split(input.to("cpu"), 1, dim=0)))
        else:
            state.inputs[name] = list(torch.split(input.to("cpu"), 1, dim=0))

        if state._should_stop_cache_forward(name):
            raise NotImplementedError

    return cache_input_hook


def replace_forward_with_hooks(state) -> None:
    """Install block-forward replacements and layer hooks via ``model_context.replace_forward``.

    Mirrors the legacy ``DataDrivenCompressor._replace_forward`` exactly. The
    ``state`` is expected to expose ``to_cached_layers`` / ``hook_handles`` /
    ``model_context`` and the two factory methods on its class
    (``_get_block_forward_func`` / ``_get_cache_data_hook_for_layer``) so
    that subclass overrides (e.g. ``DiffusionMixin``) still take effect.
    """

    def register_hook(n, m, hook_handles):
        if n in state.to_cached_layers and type(m) not in SUPPORTED_LAYER_TYPES:  # block
            m.orig_forward = m.forward
            m.forward = partial(state._get_block_forward_func(n), m)
        elif n in state.to_cached_layers:  # linear / conv1d layer
            hook_func = state._get_cache_data_hook_for_layer(n)
            hook_handle = m.register_forward_hook(hook_func)
            hook_handles.append(hook_handle)

    state.model_context.replace_forward(register_hook)


def should_stop_cache_forward(state, name: str) -> bool:
    """Default early-stop policy for block input collection.

    Mirrors the legacy ``DataDrivenCompressor._should_stop_cache_forward`` exactly.
    Subclasses (e.g. ``DiffusionMixin``) override the method on the Compressor
    class to always return ``False``; this helper is only used by the default
    LLM path.
    """
    if name == state.last_cache_name:
        return True

    if state.last_cache_name is not None:
        return False

    if not hasattr(state, "_cache_target_set") or not hasattr(state, "_cache_seen_targets"):
        return False

    if name in state._cache_target_set:
        state._cache_seen_targets.add(name)

    if not state._cache_target_set.issubset(state._cache_seen_targets):
        return False

    # Lock the last cache name after the first full forward pass.
    state.last_cache_name = name
    return True
