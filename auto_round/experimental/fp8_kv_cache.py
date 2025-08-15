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


import os
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from loguru import logger
from torch import FloatTensor, IntTensor, Tensor
from torch.nn import Module
from transformers.cache_utils import DynamicCache

__all__ = [
    "initialize_quantized_kv_cache",
    "prep_attention_module_for_calibration",
    "freeze_module_quantization_",
]

import functools
import sys

logger.add(sys.stderr, level="TRACE")

import packaging


def is_greater_or_equal_version(cur_version, deprecated_version_str):
    deprecated_version = packaging.version.parse(deprecated_version_str)
    current_version = packaging.version.parse(cur_version)
    return current_version >= deprecated_version


def freeze_module_quantization_(module: Module):
    """
    deletes observers when calibration is complete.

    apply to full model with `model.apply(freeze_module_quantization_)`

    :param module: module to freeze quantization for
    """

    # remove observers
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


def calculate_qparams(
    min_vals: Tensor,
    max_vals: Tensor,
) -> Tuple[FloatTensor, IntTensor]:
    """
    :param min_vals: tensor of min value(s) to calculate scale(s) and zero point(s)
        from
    :param max_vals: tensor of max value(s) to calculate scale(s) and zero point(s)
        from
    :param quantization_args: settings to quantization
    :param global_scale: additional global scale to scale the locally generated scale
        currently only applied/supported for Fp4

    :return: tuple of the calculated scale(s) and zero point(s). For FP4, the calculated
        scale is of dtype FP8
    """
    # based on the implementations for consuming quantized values,
    # 0.0 must always be representable within the quantized range
    min_vals = torch.min(min_vals, torch.zeros_like(min_vals))
    max_vals = torch.max(max_vals, torch.zeros_like(max_vals))

    device = min_vals.device

    # bit_min, bit_max = calculate_range(quantization_args, device)
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    bit_min, bit_max = fp8_info.min, fp8_info.max

    bit_range = bit_max - bit_min
    zp_dtype = min_vals.dtype

    max_val_pos = torch.max(torch.abs(min_vals), torch.abs(max_vals))
    scales = max_val_pos / (float(bit_range) / 2)
    scales = torch.clamp(scales, min=torch.finfo(torch.float32).eps)
    # TODO: in the case of MoEs, the global_scale may also be 0/need to be clamped

    zero_points = torch.zeros(scales.shape, device=device, dtype=min_vals.dtype)

    scales = (max_vals - min_vals) / float(bit_range)
    scales = torch.clamp(scales, min=torch.finfo(torch.float32).eps)
    zero_points = bit_min - (min_vals / scales)
    zero_points = torch.clamp(zero_points, bit_min, bit_max)

    return scales, zero_points


class MinMaxObserver(Module):
    def __init__(
        self,
        name: str = "_observer",
        averaging_constant: float = 0.01,
    ):
        super().__init__()
        self.name = name
        self._scale = None
        self._zero_point = None
        self.min_val = {}
        self.max_val = {}
        self.averaging_constant = averaging_constant

    @torch.no_grad()
    def forward(
        self,
        observed: Tensor,
        global_scale: Optional[Tensor] = None,
    ) -> Tuple[FloatTensor, IntTensor]:
        """
        maps directly to get_qparams
        :param observed: optional observed tensor from which to calculate
            quantization parameters
        :param global_scale: optional scale to further scale local quantization scales
        :return: tuple of scale and zero point based on last observed value
        """
        return self.get_qparams(
            observed=observed,
            global_scale=global_scale,
        )

    def calculate_updated_min_max(
        self,
        observed: torch.Tensor,
        reduce_dims: Optional[Tuple[int]] = None,
        tensor_id: Optional[Any] = None,
    ):
        """
        Updates the observed min and max using a moving average smoothed by the
        averaging_constant. Set the averaging_constant to 1.0 to disable averaging.

        :param observed: observed tensor to calculate quantization parameters for
        :param reduce_dims: optional tuple of dimensions to reduce along,
            returned scale and zero point will be shaped (1,) along the
            reduced dimensions
        :param tensor_id: Optional id if different ranges of observed tensors are
            passed, useful for sharding tensors by group_size
        :return: updated min and max values
        """
        tensor_id = tensor_id or "default"

        if not reduce_dims:
            min_val, max_val = torch.aminmax(observed)
        else:
            min_val = torch.amin(observed, dim=reduce_dims, keepdims=True)
            max_val = torch.amax(observed, dim=reduce_dims, keepdims=True)

        # early stopping, save some computation and memory
        # if self.averaging_constant == 1.0:
        #     return min_val, max_val

        running_min_val = self.min_val.get(tensor_id, None)
        running_max_val = self.max_val.get(tensor_id, None)

        if running_min_val is None or running_max_val is None:
            updated_min_val = min_val
            updated_max_val = max_val
        else:
            updated_min_val = running_min_val + self.averaging_constant * (min_val - running_min_val)
            updated_max_val = running_max_val + self.averaging_constant * (max_val - running_max_val)

        self.min_val[tensor_id] = updated_min_val
        self.max_val[tensor_id] = updated_max_val
        return updated_min_val, updated_max_val

    def calculate_qparams(
        self,
        observed: torch.Tensor,
        reduce_dims: Optional[Tuple[int]] = None,
        tensor_id: Optional[Any] = None,
        global_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.IntTensor]:
        """
        Generate a scale and zero-point using the observed min and max.

        :param observed: observed tensor to calculate quantization parameters for
        :param reduce_dims: optional tuple of dimensions to reduce along,
            returned scale and zero point will be shaped (1,) along the
            reduced dimensions
        :param tensor_id: Optional id if different ranges of observed tensors are
            passed, useful for sharding tensors by group_size
        :param global_scale: optional scale to further scale local quantization scales
        :return: tuple of scale and zero point derived from the observed tensor
        """

        updated_min_val, updated_max_val = self.calculate_updated_min_max(
            observed=observed, tensor_id=tensor_id, reduce_dims=reduce_dims
        )
        return calculate_qparams(
            min_vals=updated_min_val,
            max_vals=updated_max_val,
        )

    def post_calculate_qparams(self) -> None:
        """
        Run any logic specific to its observers after running calculate_qparams
        """

    def get_qparams(
        self,
        observed: Optional[Tensor] = None,
        g_idx: Optional[Tensor] = None,
        global_scale: Optional[Tensor] = None,
    ) -> Tuple[FloatTensor, IntTensor]:
        """
        Convenience function to wrap overwritten calculate_qparams
        adds support to make observed tensor optional and support for tracking latest
        calculated scale and zero point

        :param observed: optional observed tensor to calculate quantization parameters
            from
        :param g_idx: optional mapping from column index to group index
        :param global_scale: optional scale to further scale local quantization scales
        :return: tuple of scale and zero point based on last observed value
        """
        self._scale, self._zero_point = self.calculate_qparams(observed)
        return self._scale, self._zero_point

    def get_qparams_along_dim(
        self,
        observed,
        dim: Union[int, Iterable[int]],
        tensor_id: Optional[Any] = None,
        global_scale: Optional[Tensor] = None,
    ):
        if isinstance(dim, int):
            dim = [dim]
        dim = set(dim)

        reduce_dims = tuple(idx for idx in range(observed.ndim) if idx not in dim)
        return self.calculate_qparams(
            observed,
            reduce_dims=reduce_dims,
            tensor_id=tensor_id,
            global_scale=global_scale,
        )

    def reset(self):
        """
        Reset the state of the observer
        """
        self._scale = None
        self._zero_point = None


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
    Quantization strategy (tensor, group, channel) set from Quantization arg's strategy
    Singleton, so that the same cache gets reused in all forward call of self_attn.
    Each time forward is called, .update() is called, and ._quantize(), ._dequantize()
     gets called appropriately.
    The size of tensor is
     `[batch_size, num_heads, seq_len - residual_length, head_dim]`.


    # TODO: Triggered by adding kv_cache_scheme in ...

    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """Singleton"""
        if cls._instance is None:
            cls._instance = super(QuantizedKVParameterCache, cls).__new__(cls)
        return cls._instance

    def __init__(self, quantization_args=None):
        if not self._initialized:
            super().__init__()

            self.quantization_args = quantization_args

            self.k_observers: List[MinMaxObserver] = []
            self.v_observers: List[MinMaxObserver] = []

            # each index corresponds to layer_idx of the attention layer
            self.k_scales: List[Tensor] = []
            self.v_scales: List[Tensor] = []

            self.k_zps: List[Tensor] = []
            self.v_zps: List[Tensor] = []

            self._initialized = True

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Get the k_scale and v_scale and output the
         fakequant-ed key_states and value_states
        """

        if len(self.k_observers) <= layer_idx:
            k_observer = MinMaxObserver(f"k_observer_{layer_idx}")
            v_observer = MinMaxObserver(f"v_observer_{layer_idx}")

            # NOTE: User may ignore some layers in configuration,
            # meaning len(self.k_observers) <= layer_idx-1
            # Must account for that case by padding list so that
            # index of lists corresponds to layer_idx
            _pad_and_append_at_idx_(self.k_observers, layer_idx, k_observer)
            _pad_and_append_at_idx_(self.v_observers, layer_idx, v_observer)
        # FIXME: Should we append the key_states/value_states to the cache?
        q_key_states = self._quantize(key_states.contiguous(), KVCacheScaleType.KEY, layer_idx)
        q_value_states = self._quantize(value_states.contiguous(), KVCacheScaleType.VALUE, layer_idx)

        qdq_key_states = self._dequantize(q_key_states, KVCacheScaleType.KEY, layer_idx)
        qdq_value_states = self._dequantize(q_value_states, KVCacheScaleType.VALUE, layer_idx)

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
        self.key_cache: List[Tensor] = []
        self.value_cache: List[Tensor] = []
        # Used in `generate` to keep tally of how many tokens the cache has seen
        self._seen_tokens = 0
        self._quantized_key_cache: List[Tensor] = []
        self._quantized_value_cache: List[Tensor] = []

    def reset(self):
        """
        Reset the instantiation, create new instance on init
        """
        QuantizedKVParameterCache._instance = None
        QuantizedKVParameterCache._initialized = False

    def _quantize(self, tensor, kv_type, layer_idx):
        """Quantizes a key/value using a defined quantization method."""
        # from compressed_tensors.quantization.lifecycle.forward import quantize
        if kv_type == KVCacheScaleType.KEY:  # key type
            observer = self.k_observers[layer_idx]
            scales = self.k_scales
            zps = self.k_zps
        else:
            assert kv_type == KVCacheScaleType.VALUE
            observer = self.v_observers[layer_idx]
            scales = self.v_scales
            zps = self.v_zps

        scale, zp = observer(tensor)
        _pad_and_append_at_idx_(scales, layer_idx, scale)
        _pad_and_append_at_idx_(zps, layer_idx, zp)

        def quantize_fp8_tensor(x, scale, zero_point):
            fp8_info = torch.finfo(torch.float8_e4m3fn)
            q_min, q_max = fp8_info.min, fp8_info.max

            scaled = x / scale
            # clamp first because cast isn't guaranteed to be saturated (ie for fp8)
            clamped_value = torch.clamp(
                scaled,
                q_min,
                q_max,
            )

            # round
            quantized_value = clamped_value.to(torch.float8_e4m3fn)
            return quantized_value

        q_tensor = quantize_fp8_tensor(x=tensor, scale=scale, zero_point=zp)
        return q_tensor

    def _dequantize(self, qtensor, kv_type, layer_idx):
        """Dequantizes back the tensor that was quantized by `self._quantize()`"""
        from compressed_tensors.quantization.lifecycle.forward import dequantize

        if kv_type == KVCacheScaleType.KEY:
            scale = self.k_scales[layer_idx]
            zp = self.k_zps[layer_idx]
        else:
            assert kv_type == KVCacheScaleType.VALUE
            scale = self.v_scales[layer_idx]
            zp = self.v_zps[layer_idx]

        qdq_tensor = dequantize(
            x_q=qtensor,
            scale=scale,
            zero_point=zp,
            args=self.quantization_args,
        )
        return qdq_tensor


def initialize_quantized_kv_cache(module: Module):
    """
    Initialize a quantized kv_cache on a module (analogous to initializing an observer)
    """
    if not is_attention_module(module):
        return
    existing_kv_cache = getattr(module, "kv_cache", None)

    if isinstance(existing_kv_cache, QuantizedKVParameterCache):
        return

    quantized_kv_cache = QuantizedKVParameterCache()
    setattr(module, "kv_cache", quantized_kv_cache)
    logger.trace(f"Initialized quantized kv_cache for {module.__class__.__name__} {getattr(module, 'layer_idx', None)}")


def is_attention_module(module: Module):
    # FIXME: Handle this better.
    return "attention" in module.__class__.__name__.lower() and (
        hasattr(module, "k_proj") or hasattr(module, "v_proj") or hasattr(module, "qkv_proj")
    )


def calibrate_kv_cache_input_hook(
    module: Module, args: Any, kwargs: Dict[str, Any]
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Hook to update inputs to attention layers when running
    kv_cache quantization. Will update the passed in
    kv_cache to singleton QuantizedKVParameterCache.
    """
    logger.trace(f"calibrate kv_cache input hook for {module.__class__.__name__} {getattr(module, 'layer_idx', None)}")
    # breakpoint()
    kv_cache = getattr(module, "kv_cache")
    #  Start from transformers 4.55.2, the `past_key_value` was renamed to `past_key_values`.
    # https://github.com/huggingface/transformers/blob/52c6c1bb6e27ca87c4faede34a4c2a7404c17c4d/src/transformers/models/llama/modeling_llama.py#L279-L280
    if "past_key_values" in kwargs:
        kwargs["past_key_values"] = kv_cache
    else:
        kwargs["past_key_value"] = kv_cache
    kwargs["use_cache"] = False
    return args, kwargs


def update_parameter_data(module, new_val, name: str):
    """
    Update the data of a parameter in a module.
    If the parameter does not exist, it will be created.
    """
    if hasattr(module, name):
        param = getattr(module, name)
        if isinstance(param, torch.nn.Parameter):
            param.data = new_val
        else:
            module.register_parameter(name, torch.nn.Parameter(new_val))
    else:
        logger.warning(
            "Parameter %s not found in module %s, creating new parameter."
            % (name, module.__class__.__name__ + str(getattr(module, "layer_idx", "")))
        )
        module.register_parameter(name, torch.nn.Parameter(new_val))


def calibrate_kv_cache_output_hook(module: Module, _args: Any, _output: torch.Tensor):
    """
    Hook to update k_scale and v_scale parameters when running kv_cache quantization.
    """
    logger.trace(
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
