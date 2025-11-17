# Copyright (c) 2025 Red Hat AI, vLLM Project and Intel Corporation
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

# NOTICE: The design adapted from:
# https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modifiers/quantization/cache.py


import contextlib
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers.cache_utils import DynamicCache

from auto_round.experimental.utils import (
    fp8_per_tensor_qdq,
    is_attention_module,
    normalize_static_kv_dtype,
    update_parameter_data,
)
from auto_round.utils import logger

__all__ = [
    "initialize_quantized_kv_cache",
    "prep_attention_module_for_calibration",
    "freeze_module_quantization_",
    "kvcache_quant_context",
]


def freeze_module_quantization_(module: torch.nn.Module):
    """
    deletes observers when calibration is complete.

    apply to full model with `model.apply(freeze_module_quantization_)`

    :param module: module to freeze quantization for
    """

    # remove observers if needed
    for name in ("input", "weight", "output"):
        obs_name = f"{name}_observer"
        if hasattr(module, obs_name):
            delattr(module, obs_name)

    # remove quantized kv_cache
    kv_cache = getattr(module, "kv_cache", None)
    if isinstance(kv_cache, QuantizedKVParameterCache):
        delattr(module, "kv_cache")


class KVCacheScaleType(Enum):
    KEY = "k_scale"
    VALUE = "v_scale"


# NOTE: Using _ suffix to denote l is modified in place
def _pad_and_append_at_idx_(lst: List, idx: int, val: Any) -> list:
    """
    Append value val to list lst at index idx, right padding if necessary
    Needed because user may ignore some layers in configuration, meaning
    len(lst) <= idx-1

    >>> _pad_and_append_at_idx_([0,1,2], 5, 5)
    [0, 1, 2, None, None, 5]
    >>> _pad_and_append_at_idx_([0,1,2], 3, 8)
    [0, 1, 2, 8]
    >>> _pad_and_append_at_idx_([0,1,2], 1, 5)
    [0, 5, 2]
    """
    num_to_pad = idx - len(lst) + 1
    if num_to_pad > 0:
        lst += [None] * num_to_pad
    lst[idx] = val
    return lst


class QuantizedKVParameterCache(DynamicCache):
    """
    Quantized KV cache used in the forward call based on HF's dynamic cache.
    Singleton, so that the same cache gets reused in all forward call of self_attn.
    Each time forward is called, .update() is called, and ._quant_dequant() gets called appropriately.
    The size of tensor is
     `[batch_size, num_heads, seq_len - residual_length, head_dim]`.

    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """Singleton"""
        if cls._instance is None:
            cls._instance = super(QuantizedKVParameterCache, cls).__new__(cls)
        return cls._instance

    def __init__(self, dtype: torch.dtype = torch.float8_e4m3fn):

        assert dtype == torch.float8_e4m3fn, "Only fp8_e4m3fn is supported for now."
        if not self._initialized:
            super().__init__()

            # each index corresponds to layer_idx of the attention layer
            self.k_scales: List[torch.Tensor] = []
            self.v_scales: List[torch.Tensor] = []
            self._initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the k_scale and v_scale and output the quant-dequant key_states and value_states
        """
        qdq_key_states = self._quant_dequant(key_states.contiguous(), KVCacheScaleType.KEY, layer_idx)
        qdq_value_states = self._quant_dequant(value_states.contiguous(), KVCacheScaleType.VALUE, layer_idx)

        keys_to_return, values_to_return = qdq_key_states, qdq_value_states

        return keys_to_return, values_to_return

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """
        Returns the sequence length of the cached states.
        A layer index can be optionally passed.
        """
        if len(self.key_cache) <= layer_idx:
            return 0
        # since we cannot get the seq_length of each layer directly and
        # rely on `_seen_tokens` which is updated every "layer_idx" == 0,
        # this is a hack to get the actual seq_length for the given layer_idx
        # this part of code otherwise fails when used to
        # verify attn_weight shape in some models
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1

    def reset_states(self):
        """reset the kv states (used in calibration)"""
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        # Used in `generate` to keep tally of how many tokens the cache has seen
        self._seen_tokens = 0
        self._quantized_key_cache: List[torch.Tensor] = []
        self._quantized_value_cache: List[torch.Tensor] = []

    def reset(self):
        """
        Reset the instantiation, create new instance on init
        """
        QuantizedKVParameterCache._instance = None
        QuantizedKVParameterCache._initialized = False

    def _quant_dequant(self, tensor: torch.Tensor, kv_type: KVCacheScaleType, layer_idx: int):
        """Quantizes a key/value using a defined quantization method."""
        if kv_type == KVCacheScaleType.KEY:  # key type
            scales = self.k_scales
        else:
            assert kv_type == KVCacheScaleType.VALUE
            scales = self.v_scales

        qdq_tensor, scale = fp8_per_tensor_qdq(tensor)
        _pad_and_append_at_idx_(scales, layer_idx, scale.squeeze(0))
        return qdq_tensor


def initialize_quantized_kv_cache(module: torch.nn.Module, dtype=torch.float8_e4m3fn):
    """
    Initialize a quantized kv_cache on a module (analogous to initializing an observer)
    """
    if not is_attention_module(module):
        return
    existing_kv_cache = getattr(module, "kv_cache", None)

    if isinstance(existing_kv_cache, QuantizedKVParameterCache):
        return

    quantized_kv_cache = QuantizedKVParameterCache(dtype=dtype)
    setattr(module, "kv_cache", quantized_kv_cache)
    logger.debug(f"Initialized quantized kv_cache for {module.__class__.__name__} {getattr(module, 'layer_idx', None)}")


def calibrate_kv_cache_input_hook(
    module: torch.nn.Module, args: Any, kwargs: Dict[str, Any]
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Hook to update inputs to attention layers when running
    kv_cache quantization. Will update the passed in
    kv_cache to singleton QuantizedKVParameterCache.
    """
    logger.debug(f"calibrate kv_cache input hook for {module.__class__.__name__} {getattr(module, 'layer_idx', None)}")
    kv_cache = getattr(module, "kv_cache")
    #  Start from transformers 4.55.2, the `past_key_value` was renamed to `past_key_values`.
    # https://github.com/huggingface/transformers/blob/52c6c1bb6e27ca87c4faede34a4c2a7404c17c4d/src/transformers/models/llama/modeling_llama.py#L279-L280
    if "past_key_values" in kwargs:
        kwargs["past_key_values"] = kv_cache
    else:
        kwargs["past_key_value"] = kv_cache
    kwargs["use_cache"] = False
    return args, kwargs


def calibrate_kv_cache_output_hook(module: torch.nn.Module, _args: Any, _output: torch.Tensor):
    """
    Hook to update k_scale and v_scale parameters when running kv_cache quantization.
    """
    logger.debug(
        "Calibrate kv_cache output hook for %s %s"
        % (module.__class__.__name__, str(getattr(module, "layer_idx", None)))
    )
    kv_cache = getattr(module, "kv_cache")
    k_scale = kv_cache.k_scales[module.layer_idx]
    v_scale = kv_cache.v_scales[module.layer_idx]
    update_parameter_data(module, k_scale, KVCacheScaleType.KEY.value)
    update_parameter_data(module, v_scale, KVCacheScaleType.VALUE.value)


def prep_attention_module_for_calibration(module: torch.nn.Module):
    if is_attention_module(module):
        module.register_forward_pre_hook(calibrate_kv_cache_input_hook, with_kwargs=True)
        module.register_forward_hook(calibrate_kv_cache_output_hook)


@contextlib.contextmanager
def kvcache_quant_context(model: torch.nn.Module, static_kv_dtype=torch.float8_e4m3fn):
    """Context manager for FP8 KV cache quantization operations."""
    try:
        # Setup phase: Initialize KV cache for quantization
        static_kv_dtype = normalize_static_kv_dtype(static_kv_dtype)
        if static_kv_dtype != torch.float8_e4m3fn:
            logger.warning(f"Ignoring static kv dtype {static_kv_dtype}, only fp8_e4m3fn is supported.")
        else:
            initialize_fn = partial(initialize_quantized_kv_cache, dtype=static_kv_dtype)
            model.apply(initialize_fn)
            model.apply(prep_attention_module_for_calibration)

        # Provide the model to the with block
        yield model

    finally:
        # Cleanup phase: Freeze quantization parameters
        model.apply(freeze_module_quantization_)
