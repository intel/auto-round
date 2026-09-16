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
    FP8_GRANULARITY_TENSOR,
    NVFP4_KV_BLOCK_SIZE,
    NVFP4_KV_DTYPE,
    fp8_qdq,
    is_attention_module,
    normalize_fp8_granularity,
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

    def __init__(self, dtype: torch.dtype | str = torch.float8_e4m3fn, granularity: str = "tensor"):
        dtype = normalize_static_kv_dtype(dtype)
        self.is_nvfp4 = dtype == NVFP4_KV_DTYPE
        if self.is_nvfp4:
            # NVFP4 block scales are computed per 16-element group at runtime;
            # only a static per-tensor global scale is calibrated, so "head"
            # granularity does not apply.
            self.granularity = FP8_GRANULARITY_TENSOR
        else:
            assert dtype == torch.float8_e4m3fn, "Only fp8_e4m3fn is supported for now."
            self.granularity = normalize_fp8_granularity(granularity)
        # Set when a layer turns out incompatible with NVFP4 KV (e.g. head_dim
        # not divisible by 16); all subsequent updates become identity passes.
        self.disabled = False
        if not self._initialized:
            super().__init__()

            # each index corresponds to layer_idx of the attention layer
            self.k_scales: List[torch.Tensor] = []
            self.v_scales: List[torch.Tensor] = []
            self.k_amax: List[float] = []
            self.v_amax: List[float] = []
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
        if self.is_nvfp4:
            if self.disabled:
                return key_states, value_states
            qdq_key_states = self._nvfp4_quant_dequant(key_states.contiguous(), KVCacheScaleType.KEY, layer_idx)
            qdq_value_states = self._nvfp4_quant_dequant(value_states.contiguous(), KVCacheScaleType.VALUE, layer_idx)
            return qdq_key_states, qdq_value_states

        qdq_key_states = self._quant_dequant(key_states.contiguous(), KVCacheScaleType.KEY, layer_idx)
        qdq_value_states = self._quant_dequant(value_states.contiguous(), KVCacheScaleType.VALUE, layer_idx)

        keys_to_return, values_to_return = qdq_key_states, qdq_value_states

        return keys_to_return, values_to_return

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """
        Returns the sequence length of the cached states.
        A layer index can be optionally passed.
        """
        # Transformers' newer DynamicCache stores cache state in `layers`, not `key_cache`.
        layer_idx = 0 if layer_idx is None else layer_idx
        if len(getattr(self, "key_cache", ())) <= layer_idx:
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

        qdq_tensor, scale = fp8_qdq(tensor, granularity=self.granularity)
        # Detach scale to prevent holding computation graph references
        _pad_and_append_at_idx_(scales, layer_idx, scale.reshape(-1).detach())
        return qdq_tensor

    def _nvfp4_quant_dequant(self, tensor: torch.Tensor, kv_type: KVCacheScaleType, layer_idx: int):
        """NVFP4 quant-dequant for a K/V tensor.

        Keeps a running max amax per layer and per side; the QDQ uses the
        corresponding running global scale ``gs = 2688 / amax`` (fp4 max 6 x
        fp8 max 448), mirroring the runtime semantics where the fp8 block
        scale is computed on the fly and only the global scale is static. The
        final amax is converted to the stored ``k_global_scale``/
        ``v_global_scale`` parameters by the output hook, which writes the
        vLLM dequant-multiplier convention (``amax / 2688``) -- see
        ``_nvfp4_global_scale``.
        """
        if tensor.shape[-1] % NVFP4_KV_BLOCK_SIZE != 0:
            logger.warning(
                "NVFP4 KV cache requires the last dim (head_dim) to be divisible by %d "
                "(got %d); disabling NVFP4 KV cache quantization.",
                NVFP4_KV_BLOCK_SIZE,
                tensor.shape[-1],
            )
            self.disabled = True
            return tensor

        amax = tensor.abs().max().item()
        amax_list = self.k_amax if kv_type == KVCacheScaleType.KEY else self.v_amax
        if layer_idx >= len(amax_list):
            _pad_and_append_at_idx_(amax_list, layer_idx, amax)
        else:
            amax_list[layer_idx] = max(amax_list[layer_idx], amax)
        running_amax = amax_list[layer_idx]
        if running_amax <= 0:
            return tensor

        from auto_round.data_type.nvfp import nv_fp4_with_static_gs

        qdq_tensor, _, _ = nv_fp4_with_static_gs(tensor, tensor_max=running_amax)
        return qdq_tensor


def initialize_quantized_kv_cache(module: torch.nn.Module, dtype=torch.float8_e4m3fn, granularity: str = "tensor"):
    """
    Initialize a quantized kv_cache on a module (analogous to initializing an observer)
    """
    if not is_attention_module(module):
        return
    existing_kv_cache = getattr(module, "kv_cache", None)

    if isinstance(existing_kv_cache, QuantizedKVParameterCache):
        return

    quantized_kv_cache = QuantizedKVParameterCache(dtype=dtype, granularity=granularity)
    setattr(module, "kv_cache", quantized_kv_cache)
    logger.debug(f"Initialized quantized kv_cache for {module.__class__.__name__} {getattr(module, 'layer_idx', None)}")
    if quantized_kv_cache.is_nvfp4:
        # Global scales are only registered once calibration has observed KV
        # magnitudes; creating zero/placeholder scales up front would export a
        # broken scheme when no calibration forward happens.
        return
    init_scale = torch.tensor([0.0], device=next(module.parameters()).device)
    update_parameter_data(module, init_scale.clone(), KVCacheScaleType.KEY.value)
    update_parameter_data(module, init_scale.clone(), KVCacheScaleType.VALUE.value)


def calibrate_kv_cache_input_hook(
    module: torch.nn.Module, args: Any, kwargs: Dict[str, Any]
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Hook to update inputs to attention layers when running
    kv_cache quantization. Will update the passed in
    kv_cache to singleton QuantizedKVParameterCache.
    """
    kv_cache = getattr(module, "kv_cache")
    #  Start from transformers 4.55.2, the `past_key_value` was renamed to `past_key_values`.
    # https://github.com/huggingface/transformers/blob/52c6c1bb6e27ca87c4faede34a4c2a7404c17c4d/src/transformers/models/llama/modeling_llama.py#L279-L280
    if "past_key_values" in kwargs:
        kwargs["past_key_values"] = kv_cache
    else:
        kwargs["past_key_value"] = kv_cache
    kwargs["use_cache"] = False
    return args, kwargs


def _nvfp4_global_scale(amax: float, device: torch.device) -> torch.Tensor:
    """Static NVFP4 KV scale for the checkpoint, in the vLLM serving convention.

    vLLM's NVFP4 KV cache store kernel (``reshape_and_cache_nvfp4``) treats the
    checkpoint value as the *dequantization multiplier*: it computes
    ``global_scale = 1 / k_scale`` and the per-16-element fp8 block scale as
    ``sf = global_scale * block_max / 6``, then reads back
    ``x = fp4 * sf * k_scale``.  Storing the weight-style ``2688 / amax``
    instead would make the runtime build fp8 block scales from
    ``block_max * amax / (6 * 2688)``, which underflows e4m3 (min subnormal
    ~2**-9) for typical activation magnitudes and zeroes the KV cache.

    The reciprocal ``amax / 2688 = 1 / calculate_gparam(amax)`` makes the
    runtime's block scale ``(2688 / amax) * block_max / 6`` -- exactly the
    value auto-round's QDQ simulation (``nv_fp4_with_static_gs``) uses during
    calibration, so served outputs match calibration up to fp8 rounding.

    Note: the pure-Python compressed-tensors runtime (``forward_quantize``)
    interprets the stored value the other way around (``sf = gs * block_max /
    6``).  The two runtimes use opposite conventions; vLLM is the target
    serving engine, hence this convention.
    """
    from auto_round.data_type.nvfp import calculate_gparam

    global_scale = calculate_gparam(amax, device=device)
    return (1.0 / global_scale).reshape(1).detach()


def calibrate_kv_cache_output_hook(module: torch.nn.Module, _args: Any, _output: torch.Tensor):
    """
    Hook to update k_scale and v_scale parameters when running kv_cache quantization.
    """
    kv_cache = getattr(module, "kv_cache")
    if kv_cache.is_nvfp4:
        if kv_cache.disabled:
            return
        layer_idx = module.layer_idx
        k_amax = kv_cache.k_amax[layer_idx] if layer_idx < len(kv_cache.k_amax) else 0.0
        v_amax = kv_cache.v_amax[layer_idx] if layer_idx < len(kv_cache.v_amax) else 0.0
        if k_amax > 0:
            update_parameter_data(
                module, _nvfp4_global_scale(k_amax, next(module.parameters()).device), "k_global_scale"
            )
        if v_amax > 0:
            update_parameter_data(
                module, _nvfp4_global_scale(v_amax, next(module.parameters()).device), "v_global_scale"
            )
        return
    k_scale = kv_cache.k_scales[module.layer_idx]
    v_scale = kv_cache.v_scales[module.layer_idx]
    update_parameter_data(module, k_scale, KVCacheScaleType.KEY.value)
    update_parameter_data(module, v_scale, KVCacheScaleType.VALUE.value)


def prep_attention_module_for_calibration(module: torch.nn.Module):
    if is_attention_module(module):
        module.register_forward_pre_hook(calibrate_kv_cache_input_hook, with_kwargs=True)
        module.register_forward_hook(calibrate_kv_cache_output_hook)


@contextlib.contextmanager
def kvcache_quant_context(
    model: torch.nn.Module, static_kv_dtype=torch.float8_e4m3fn, static_kv_granularity: str = "tensor"
):
    """Context manager for static KV cache quantization (FP8 or NVFP4) operations."""
    try:
        # Setup phase: Initialize KV cache for quantization
        static_kv_dtype = normalize_static_kv_dtype(static_kv_dtype)
        static_kv_granularity = normalize_fp8_granularity(static_kv_granularity)
        if static_kv_dtype == NVFP4_KV_DTYPE:
            if static_kv_granularity != FP8_GRANULARITY_TENSOR:
                logger.warning(
                    "NVFP4 KV cache only supports 'tensor' granularity; ignoring granularity %r.",
                    static_kv_granularity,
                )
            initialize_fn = partial(
                initialize_quantized_kv_cache, dtype=NVFP4_KV_DTYPE, granularity=FP8_GRANULARITY_TENSOR
            )
        elif static_kv_dtype != torch.float8_e4m3fn:
            logger.warning(f"Ignoring static kv dtype {static_kv_dtype}, only fp8_e4m3fn and nvfp4 are supported.")
        else:
            initialize_fn = partial(
                initialize_quantized_kv_cache, dtype=static_kv_dtype, granularity=static_kv_granularity
            )
        if static_kv_dtype in (torch.float8_e4m3fn, NVFP4_KV_DTYPE):
            model.apply(initialize_fn)
            model.apply(prep_attention_module_for_calibration)

        # Provide the model to the with block
        yield model

    finally:
        # Cleanup phase: Freeze quantization parameters
        model.apply(freeze_module_quantization_)
