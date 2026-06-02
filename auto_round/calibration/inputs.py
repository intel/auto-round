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
"""Pure helpers for shaping cached block inputs."""

from typing import Tuple

import torch

from auto_round.utils import clear_memory, to_device, to_dtype
from auto_round.utils.device_manager import device_manager

__all__ = ["split_inputs", "preprocess_block_inputs"]


def split_inputs(
    inputs: dict,
    first_input_name: str,
    *,
    is_diffusion: bool,
    shared_cache_keys: tuple = (),
) -> Tuple[object, dict]:
    """Split a captured ``inputs`` dict into ``(input_ids, input_others)``.

    Mirrors the original ``DataDrivenCompressor._split_inputs`` exactly:

    - For diffusion models, every key containing ``"hidden_state"`` is pulled
      out into a dict and returned as ``input_ids``; the remaining kwargs are
      returned as ``input_others``.  Keys in *shared_cache_keys* are kept in
      ``input_others`` even if they match ``"hidden_state"``.
    - Otherwise, ``inputs[first_input_name]`` is popped and returned as
      ``input_ids`` (may be ``None``); the remainder is ``input_others``.

    The function mutates ``inputs`` (pops keys) — this matches the legacy
    behaviour that downstream code relies on.
    """
    if is_diffusion:
        input_id_str = [key for key in inputs.keys() if "hidden_state" in key and key not in shared_cache_keys]
        input_ids = {k: inputs.pop(k, None) for k in input_id_str}
        input_others = inputs
        return input_ids, input_others
    input_ids = inputs.get(first_input_name, None)
    inputs.pop(first_input_name, None)
    input_others = inputs
    return input_ids, input_others


def _unwrap_single_element(input_others):
    """Unwrap single-element list/tuple values from kwargs dict."""
    for key in list(input_others.keys()):
        if key == "positional_inputs":
            continue
        val = input_others[key]
        if isinstance(val, (list, tuple)) and len(val) == 1:
            input_others[key] = val[0]


def preprocess_block_inputs(
    inputs: dict,
    *,
    model_context,
    compress_context,
    first_input_name: str = "input_ids",
) -> Tuple[object, dict]:
    """Move/cast cached block inputs onto the calibration cache device.

    Mirrors the original ``DataDrivenCompressor._preprocess_block_inputs`` exactly.
    Parameterized on ``model_context`` (for ``amp`` / ``amp_dtype`` /
    ``is_diffusion``) and ``compress_context`` (for ``cache_device`` /
    ``device_list``) so it does not require a Compressor ``self``.
    """
    input_ids, input_others = split_inputs(
        inputs,
        first_input_name,
        is_diffusion=model_context.is_diffusion,
        shared_cache_keys=getattr(model_context, "shared_cache_keys", ()),
    )
    clear_memory(device_list=device_manager.device_list)
    tmp_dtype = model_context.amp_dtype if model_context.amp else torch.float32
    if input_ids is not None:
        input_ids = to_device(input_ids, compress_context.cache_device)
        input_ids = to_dtype(input_ids, tmp_dtype)
    input_others = to_device(input_others, compress_context.cache_device)

    _unwrap_single_element(input_others)

    for key in input_others.keys():
        if isinstance(input_others[key], torch.Tensor) and (
            input_others[key].dtype == torch.float16 or input_others[key].dtype == torch.bfloat16
        ):
            input_others[key] = input_others[key].to(tmp_dtype)
        elif isinstance(input_others[key], list):
            for i in range(len(input_others[key])):
                v = input_others[key][i]
                if isinstance(v, torch.Tensor) and v.dtype in (torch.int32, torch.int64):
                    continue
                input_others[key][i] = to_dtype(v, tmp_dtype)
    return input_ids, input_others
