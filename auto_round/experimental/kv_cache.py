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
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache, CacheLayerMixin, DynamicCache

from auto_round.experimental.turboquant import (
    QJLResidualConfig,
    TurboQuantConfig,
    TurboQuantPackedTensor,
    build_turboquant_state,
    turboquant_pack,
    turboquant_qdq,
    turboquant_unpack,
)
from auto_round.experimental.utils import (
    is_attention_module,
    normalize_static_kv_dtype,
    per_tensor_fp8_qdq,
    update_parameter_data,
)
from auto_round.utils import logger

__all__ = [
    "initialize_quantized_kv_cache",
    "normalize_kv_cache_backend_config",
    "prep_attention_module_for_calibration",
    "freeze_module_quantization_",
    "kvcache_quant_context",
    "TurboQuantPackedKVCache",
    "build_turboquant_runtime_cache",
]


class KVCacheScaleType(Enum):
    KEY = "k_scale"
    VALUE = "v_scale"


@dataclass(frozen=True)
class KVCacheBackendConfig:
    backend: str
    dtype: torch.dtype | None = None
    bits: int | None = None
    seed: int = 42
    packed: bool = False
    residual_length: int = 128
    qjl_residual: bool = False


def _normalize_backend_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def normalize_kv_cache_backend_config(spec: Union[str, torch.dtype, Dict[str, Any], KVCacheBackendConfig]):
    if isinstance(spec, KVCacheBackendConfig):
        return spec

    if isinstance(spec, dict):
        backend = _normalize_backend_name(spec.get("backend") or spec.get("name") or spec.get("dtype") or "")
        if backend in ("fp8", "float8-e4m3fn", "float8_e4m3fn"):
            dtype = normalize_static_kv_dtype(spec.get("dtype", "fp8"))
            return KVCacheBackendConfig(backend="fp8", dtype=dtype, seed=int(spec.get("seed", 42)))
        if backend == "turboquant":
            return KVCacheBackendConfig(
                backend="turboquant",
                bits=int(spec.get("bits", 4)),
                seed=int(spec.get("seed", 42)),
                packed=bool(spec.get("packed", False)),
                residual_length=int(spec.get("residual_length", 128)),
                qjl_residual=bool(spec.get("qjl_residual", False)),
            )
        raise ValueError(f"Unsupported kv cache backend config: {spec}")

    if isinstance(spec, torch.dtype):
        return KVCacheBackendConfig(backend="fp8", dtype=normalize_static_kv_dtype(spec))

    if not isinstance(spec, str):
        raise TypeError(f"Unsupported kv cache backend spec type: {type(spec)}")

    normalized = spec.strip().lower()
    if normalized in ("fp8", "float8_e4m3fn"):
        return KVCacheBackendConfig(backend="fp8", dtype=normalize_static_kv_dtype(normalized))

    if normalized.startswith("turboquant"):
        bits = 4
        packed = False
        residual_length = 128
        qjl_residual = False
        tokens = normalized.split(":")
        for token in tokens[1:]:
            if token.isdigit():
                bits = int(token)
            elif token == "packed":
                packed = True
            elif token.startswith("residual="):
                residual_length = int(token.split("=", 1)[1])
            elif token in ("qjl",):
                qjl_residual = True
        return KVCacheBackendConfig(
            backend="turboquant",
            bits=bits,
            seed=42,
            packed=packed,
            residual_length=residual_length,
            qjl_residual=qjl_residual,
        )

    raise ValueError(
        "Invalid static kv dtype/backend: %s. Supported values include 'fp8', 'float8_e4m3fn', "
        "'turboquant', and 'turboquant:2|3|4'." % spec
    )


class KVCacheBackend:
    name = "base"

    def __init__(self, config: KVCacheBackendConfig):
        self.config = config

    def init_module_parameters(self, module: torch.nn.Module):
        init_scale = torch.tensor([0.0], device=next(module.parameters()).device)
        update_parameter_data(module, init_scale.clone(), KVCacheScaleType.KEY.value)
        update_parameter_data(module, init_scale.clone(), KVCacheScaleType.VALUE.value)

    def reset(self):
        return None

    def quant_dequant(self, tensor: torch.Tensor, kv_type: KVCacheScaleType, layer_idx: int):
        raise NotImplementedError


class FP8KVCacheBackend(KVCacheBackend):
    name = "fp8"

    def __init__(self, config: KVCacheBackendConfig):
        super().__init__(config)
        if config.dtype != torch.float8_e4m3fn:
            raise ValueError(f"Only fp8_e4m3fn KV cache is supported, but got {config.dtype}.")

    def quant_dequant(self, tensor: torch.Tensor, kv_type: KVCacheScaleType, layer_idx: int):
        del kv_type, layer_idx
        return per_tensor_fp8_qdq(tensor)


class TurboQuantKVCacheBackend(KVCacheBackend):
    name = "turboquant"

    def __init__(self, config: KVCacheBackendConfig):
        super().__init__(config)
        self.turboquant_config = TurboQuantConfig(bits=config.bits or 4, seed=config.seed)
        self._state_cache: Dict[tuple[int, str, int, str], Any] = {}

    def reset(self):
        self._state_cache = {}

    def _get_state(self, tensor: torch.Tensor, kv_type: KVCacheScaleType, layer_idx: int):
        head_dim = tensor.shape[-1]
        state_key = (layer_idx, kv_type.value, head_dim, str(tensor.device))
        if state_key not in self._state_cache:
            state_seed = self.turboquant_config.seed + layer_idx * 17 + (0 if kv_type == KVCacheScaleType.KEY else 1)
            self._state_cache[state_key] = build_turboquant_state(
                head_dim=head_dim,
                bits=self.turboquant_config.bits,
                seed=state_seed,
                device=tensor.device,
            )
        return self._state_cache[state_key]

    def quant_dequant(self, tensor: torch.Tensor, kv_type: KVCacheScaleType, layer_idx: int):
        state = self._get_state(tensor, kv_type, layer_idx)
        return turboquant_qdq(tensor, state, eps=self.turboquant_config.eps)


def build_kv_cache_backend(config: KVCacheBackendConfig) -> KVCacheBackend:
    if config.backend == "fp8":
        return FP8KVCacheBackend(config)
    if config.backend == "turboquant":
        return TurboQuantKVCacheBackend(config)
    raise ValueError(f"Unsupported kv cache backend {config.backend}")


def _cleanup_kv_cache_hooks(module: torch.nn.Module):
    hook_handles = getattr(module, "_kv_cache_hook_handles", None)
    if hook_handles is None:
        return
    for handle in hook_handles:
        handle.remove()
    delattr(module, "_kv_cache_hook_handles")


def freeze_module_quantization_(module: torch.nn.Module):
    """
    deletes observers when calibration is complete.

    apply to full model with `model.apply(freeze_module_quantization_)`

    :param module: module to freeze quantization for
    """

    for name in ("input", "weight", "output"):
        obs_name = f"{name}_observer"
        if hasattr(module, obs_name):
            delattr(module, obs_name)

    kv_cache = getattr(module, "kv_cache", None)
    if isinstance(kv_cache, QuantizedKVParameterCache):
        delattr(module, "kv_cache")

    _cleanup_kv_cache_hooks(module)


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
        if cls._instance is None:
            cls._instance = super(QuantizedKVParameterCache, cls).__new__(cls)
        return cls._instance

    def __init__(self, config: KVCacheBackendConfig):
        if not self._initialized:
            super().__init__()
            self._initialized = True
        self.backend_config = config
        self.backend = build_kv_cache_backend(config)
        self.reset_states()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        qdq_key_states = self._quant_dequant(key_states.contiguous(), KVCacheScaleType.KEY, layer_idx)
        qdq_value_states = self._quant_dequant(value_states.contiguous(), KVCacheScaleType.VALUE, layer_idx)
        return qdq_key_states, qdq_value_states

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return self._seen_tokens if layer_idx == 0 else self._seen_tokens - 1

    def reset_states(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0
        self._quantized_key_cache: List[torch.Tensor] = []
        self._quantized_value_cache: List[torch.Tensor] = []
        self.k_scales: List[torch.Tensor] = []
        self.v_scales: List[torch.Tensor] = []
        if hasattr(self, "backend"):
            self.backend.reset()

    def reset(self):
        QuantizedKVParameterCache._instance = None
        QuantizedKVParameterCache._initialized = False

    def _quant_dequant(self, tensor: torch.Tensor, kv_type: KVCacheScaleType, layer_idx: int):
        scales = self.k_scales if kv_type == KVCacheScaleType.KEY else self.v_scales
        qdq_tensor, scale = self.backend.quant_dequant(tensor, kv_type, layer_idx)
        _pad_and_append_at_idx_(scales, layer_idx, scale.squeeze(0).detach())
        return qdq_tensor


class TurboQuantPackedKVCacheLayer(CacheLayerMixin):
    is_compilable = False

    def __init__(
        self,
        bits: int = 4,
        residual_length: int = 128,
        seed: int = 42,
        qjl_residual: bool = False,
    ):
        super().__init__()
        self.bits = bits
        self.residual_length = residual_length
        self.seed = seed
        self.qjl_residual = qjl_residual
        self.cumulative_length = 0
        self._packed_key_segments: list[TurboQuantPackedTensor] = []
        self._packed_value_segments: list[TurboQuantPackedTensor] = []
        self._key_state = None
        self._value_state = None
        self._residual_config = None
        self.dtype = None
        self.device = None
        self._batch_size = None
        self._num_heads = None
        self._k_head_dim = None
        self._v_head_dim = None

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self._batch_size, self._num_heads = key_states.shape[:2]
        self._k_head_dim = key_states.shape[-1]
        self._v_head_dim = value_states.shape[-1]
        self.keys = torch.empty(
            (self._batch_size, self._num_heads, 0, self._k_head_dim), dtype=self.dtype, device=self.device
        )
        self.values = torch.empty(
            (self._batch_size, self._num_heads, 0, self._v_head_dim), dtype=self.dtype, device=self.device
        )
        self._residual_config = QJLResidualConfig(
            enabled=self.qjl_residual,
            seed=self.seed + 7919,
        )
        self._key_state = build_turboquant_state(
            self._k_head_dim,
            self.bits,
            self.seed,
            self.device,
            qjl_config=self._residual_config,
        )
        self._value_state = build_turboquant_state(
            self._v_head_dim,
            self.bits,
            self.seed + 1,
            self.device,
            qjl_config=self._residual_config,
        )
        self.is_initialized = True

    def _empty_like_keys(self):
        return torch.empty(
            (self._batch_size, self._num_heads, 0, self._k_head_dim), dtype=self.dtype, device=self.device
        )

    def _empty_like_values(self):
        return torch.empty(
            (self._batch_size, self._num_heads, 0, self._v_head_dim), dtype=self.dtype, device=self.device
        )

    def _spill_residual_to_packed(self):
        if self.residual_length < 0:
            raise ValueError(f"residual_length must be >= 0, but got {self.residual_length}.")

        buf_len = self.keys.shape[-2]
        # Only spill when buffer reaches 2x residual_length (or residual_length
        # if it's 0).  This ensures each packed segment contains at least
        # residual_length tokens, avoiding 1-token segments during token-by-token
        # decode that would cause O(n²) unpack overhead.
        spill_threshold = max(2 * self.residual_length, 1)
        if buf_len < spill_threshold:
            return

        spill = buf_len - self.residual_length

        prefix_keys = self.keys[..., :spill, :].contiguous()
        prefix_values = self.values[..., :spill, :].contiguous()
        if prefix_keys.numel() > 0:
            self._packed_key_segments.append(
                turboquant_pack(prefix_keys, self._key_state, residual_config=self._residual_config)
            )
            self._packed_value_segments.append(
                turboquant_pack(prefix_values, self._value_state, residual_config=self._residual_config)
            )

        self.keys = self.keys[..., spill:, :].contiguous() if self.residual_length > 0 else self._empty_like_keys()
        self.values = (
            self.values[..., spill:, :].contiguous() if self.residual_length > 0 else self._empty_like_values()
        )
        # NOTE: we intentionally do NOT merge packed segments by dequantize→requantize,
        # because each round of requantization compounds quantization noise.
        # With token-by-token decode and small residual_length, the oldest tokens
        # could be requantized 100+ times, destroying quality completely.
        # Instead, we keep segments as-is and pay O(n_segments) unpack cost per step.

    def _dequantize_segments(self, packed_segments, state, empty_tensor):
        if len(packed_segments) == 0:
            return empty_tensor
        if len(packed_segments) == 1:
            return turboquant_unpack(packed_segments[0], state, dtype=self.dtype, residual_config=self._residual_config)

        # Merge all packed segments into one and do a single unpack call.
        # This turns O(n_segments) kernel launches into O(1).
        # Safe because each segment's packed bytes are byte-aligned
        # (n_values_per_token * bits is always divisible by 8).
        base = packed_segments[0]
        total_seq = sum(s.original_shape[-2] for s in packed_segments)
        merged_shape = base.original_shape[:-2] + (total_seq,) + base.original_shape[-1:]

        merged_codes = torch.cat([s.packed_codes for s in packed_segments])
        merged_norms = torch.cat([s.norms for s in packed_segments], dim=-2)

        qjl_signs = None
        qjl_norms = None
        if base.qjl_packed_signs is not None:
            qjl_signs = torch.cat([s.qjl_packed_signs for s in packed_segments])
            qjl_norms = torch.cat([s.qjl_norms for s in packed_segments], dim=-1)

        merged = TurboQuantPackedTensor(
            packed_codes=merged_codes,
            norms=merged_norms,
            original_shape=merged_shape,
            bits=base.bits,
            qjl_packed_signs=qjl_signs,
            qjl_norms=qjl_norms,
        )
        return turboquant_unpack(merged, state, dtype=self.dtype, residual_config=self._residual_config)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self.cumulative_length += key_states.shape[-2]
        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        self._spill_residual_to_packed()

        dequantized_keys = self._dequantize_segments(
            self._packed_key_segments, self._key_state, self._empty_like_keys()
        )
        dequantized_values = self._dequantize_segments(
            self._packed_value_segments, self._value_state, self._empty_like_values()
        )
        keys_to_return = torch.cat([dequantized_keys, self.keys], dim=-2)
        values_to_return = torch.cat([dequantized_values, self.values], dim=-2)
        return keys_to_return, values_to_return

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        kv_length = self.get_seq_length() + cache_position.shape[0]
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return -1

    def _move_packed_segments(self, segments, device: str):
        moved = []
        for segment in segments:
            moved.append(
                TurboQuantPackedTensor(
                    packed_codes=segment.packed_codes.to(device, non_blocking=True),
                    norms=segment.norms.to(device, non_blocking=True),
                    original_shape=segment.original_shape,
                    bits=segment.bits,
                    qjl_packed_signs=(
                        None
                        if segment.qjl_packed_signs is None
                        else segment.qjl_packed_signs.to(device, non_blocking=True)
                    ),
                    qjl_norms=(None if segment.qjl_norms is None else segment.qjl_norms.to(device, non_blocking=True)),
                )
            )
        return moved

    def offload(self):
        super().offload()
        self._packed_key_segments = self._move_packed_segments(self._packed_key_segments, "cpu")
        self._packed_value_segments = self._move_packed_segments(self._packed_value_segments, "cpu")

    def prefetch(self):
        super().prefetch()
        if (
            self.is_initialized
            and len(self._packed_key_segments) > 0
            and self._packed_key_segments[0].packed_codes.device != self.device
        ):
            self._packed_key_segments = self._move_packed_segments(self._packed_key_segments, self.device)
            self._packed_value_segments = self._move_packed_segments(self._packed_value_segments, self.device)

    def reset(self) -> None:
        if self.is_initialized:
            self.keys = self._empty_like_keys()
            self.values = self._empty_like_values()
        self._packed_key_segments = []
        self._packed_value_segments = []
        self.cumulative_length = 0

    def _repack_segments_with_batch_indices(self, segments, state, indices: torch.Tensor):
        repacked = []
        for segment in segments:
            unpacked = turboquant_unpack(segment, state, dtype=self.dtype, residual_config=self._residual_config)
            repacked.append(
                turboquant_pack(
                    unpacked.index_select(0, indices.to(unpacked.device)), state, residual_config=self._residual_config
                )
            )
        return repacked

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        if self.keys is not None and self.keys.numel() > 0:
            self.keys = self.keys.index_select(0, beam_idx.to(self.keys.device))
            self.values = self.values.index_select(0, beam_idx.to(self.values.device))
        self._packed_key_segments = self._repack_segments_with_batch_indices(
            self._packed_key_segments, self._key_state, beam_idx
        )
        self._packed_value_segments = self._repack_segments_with_batch_indices(
            self._packed_value_segments, self._value_state, beam_idx
        )

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.keys is not None and self.keys.numel() > 0:
            self.keys = self.keys.repeat_interleave(repeats, dim=0)
            self.values = self.values.repeat_interleave(repeats, dim=0)
        indices = torch.arange(self._batch_size, device=self.device).repeat_interleave(repeats)
        self._packed_key_segments = self._repack_segments_with_batch_indices(
            self._packed_key_segments, self._key_state, indices
        )
        self._packed_value_segments = self._repack_segments_with_batch_indices(
            self._packed_value_segments, self._value_state, indices
        )
        self._batch_size *= repeats

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.keys is not None and self.keys.numel() > 0:
            self.keys = self.keys[indices, ...]
            self.values = self.values[indices, ...]
        self._packed_key_segments = self._repack_segments_with_batch_indices(
            self._packed_key_segments, self._key_state, indices
        )
        self._packed_value_segments = self._repack_segments_with_batch_indices(
            self._packed_value_segments, self._value_state, indices
        )
        self._batch_size = indices.numel()

    def packed_memory_bytes(self) -> int:
        return sum(segment.memory_bytes() for segment in self._packed_key_segments + self._packed_value_segments)

    def residual_memory_bytes(self) -> int:
        return self.keys.numel() * self.keys.element_size() + self.values.numel() * self.values.element_size()

    def raw_memory_bytes(self) -> int:
        return (
            self.cumulative_length
            * self._batch_size
            * self._num_heads
            * (self._k_head_dim + self._v_head_dim)
            * torch.tensor([], dtype=self.dtype).element_size()
        )


class TurboQuantPackedKVCache(Cache):
    def __init__(
        self,
        bits: int = 4,
        residual_length: int = 128,
        seed: int = 42,
        qjl_residual: bool = False,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        super().__init__(layers=[], offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)
        self.bits = bits
        self.residual_length = residual_length
        self.seed = seed
        self.qjl_residual = qjl_residual

    def _new_layer(self, layer_idx: int) -> TurboQuantPackedKVCacheLayer:
        return TurboQuantPackedKVCacheLayer(
            bits=self.bits,
            residual_length=self.residual_length,
            seed=self.seed + layer_idx * 17,
            qjl_residual=self.qjl_residual,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        while len(self.layers) <= layer_idx:
            self.layers.append(self._new_layer(len(self.layers)))
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    def packed_memory_bytes(self) -> int:
        return sum(layer.packed_memory_bytes() for layer in self.layers)

    def residual_memory_bytes(self) -> int:
        return sum(layer.residual_memory_bytes() for layer in self.layers)

    def total_memory_bytes(self) -> int:
        return self.packed_memory_bytes() + self.residual_memory_bytes()

    def raw_memory_bytes(self) -> int:
        return sum(layer.raw_memory_bytes() for layer in self.layers)

    def compression_ratio(self) -> float:
        packed_bytes = max(self.total_memory_bytes(), 1)
        return self.raw_memory_bytes() / packed_bytes


class TurboQuantPreDequantCache(DynamicCache):
    """quantize→dequantize K/V, store bf16 in standard cache.

    At write time: K,V → encode → decode → store dequantized bf16
    At read time:  read bf16, zero overhead (standard DynamicCache)

    This correctly simulates the quantization error without the decode overhead
    at attention time. No actual memory compression.
    """

    def __init__(self, bits: int = 4, seed: int = 42, qjl_residual: bool = False):
        super().__init__()
        self.bits = bits
        self.seed = seed
        self.qjl_residual = qjl_residual
        self._states: dict[int, object] = {}

    def _get_state(self, layer_idx: int, head_dim: int, device: torch.device):
        if layer_idx not in self._states:
            qjl_config = QJLResidualConfig(enabled=self.qjl_residual, seed=1729) if self.qjl_residual else None
            self._states[layer_idx] = build_turboquant_state(
                head_dim=head_dim,
                bits=self.bits,
                seed=self.seed + layer_idx * 17,
                device=device,
                qjl_config=qjl_config,
            )
        return self._states[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head_dim = key_states.shape[-1]
        state = self._get_state(layer_idx, head_dim, key_states.device)
        residual_config = QJLResidualConfig(enabled=self.qjl_residual, seed=1729) if self.qjl_residual else None
        # Pre-dequant: quant→dequant before storing
        key_dq, _ = turboquant_qdq(key_states, state, residual_config=residual_config)
        value_dq, _ = turboquant_qdq(value_states, state, residual_config=residual_config)
        return super().update(key_dq, value_dq, layer_idx, cache_kwargs)


def build_turboquant_runtime_cache(
    bits: int = 4,
    residual_length: int = 128,
    seed: int = 42,
    qjl_residual: bool = False,
    offloading: bool = False,
    offload_only_non_sliding: bool = False,
    mode: str = "packed",
) -> Cache:
    """Build a TurboQuant KV cache.

    Args:
        mode: "packed" stores bit-packed codes (real compression, higher latency),
              "pre_dequant" stores bf16 after quant→dequant (zero read overhead,
              no compression — matches vLLM Phase 1 approach).
    """
    if mode == "pre_dequant":
        return TurboQuantPreDequantCache(
            bits=bits,
            seed=seed,
            qjl_residual=qjl_residual,
        )
    return TurboQuantPackedKVCache(
        bits=bits,
        residual_length=residual_length,
        seed=seed,
        qjl_residual=qjl_residual,
        offloading=offloading,
        offload_only_non_sliding=offload_only_non_sliding,
    )


def initialize_quantized_kv_cache(module: torch.nn.Module, config: KVCacheBackendConfig):
    """
    Initialize a quantized kv_cache on a module (analogous to initializing an observer)
    """
    if not is_attention_module(module):
        return
    existing_kv_cache = getattr(module, "kv_cache", None)

    if isinstance(existing_kv_cache, QuantizedKVParameterCache) and existing_kv_cache.backend_config == config:
        return

    quantized_kv_cache = QuantizedKVParameterCache(config=config)
    setattr(module, "kv_cache", quantized_kv_cache)
    logger.debug(f"Initialized quantized kv_cache for {module.__class__.__name__} {getattr(module, 'layer_idx', None)}")
    quantized_kv_cache.backend.init_module_parameters(module)


def calibrate_kv_cache_input_hook(
    module: torch.nn.Module, args: Any, kwargs: Dict[str, Any]
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """
    Hook to update inputs to attention layers when running kv_cache quantization.
    """
    kv_cache = getattr(module, "kv_cache")
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
    kv_cache = getattr(module, "kv_cache")
    k_scale = kv_cache.k_scales[module.layer_idx]
    v_scale = kv_cache.v_scales[module.layer_idx]
    update_parameter_data(module, k_scale, KVCacheScaleType.KEY.value)
    update_parameter_data(module, v_scale, KVCacheScaleType.VALUE.value)


def prep_attention_module_for_calibration(module: torch.nn.Module):
    if is_attention_module(module):
        if hasattr(module, "_kv_cache_hook_handles"):
            return
        pre_handle = module.register_forward_pre_hook(calibrate_kv_cache_input_hook, with_kwargs=True)
        post_handle = module.register_forward_hook(calibrate_kv_cache_output_hook)
        module._kv_cache_hook_handles = (pre_handle, post_handle)


@contextlib.contextmanager
def kvcache_quant_context(model: torch.nn.Module, static_kv_dtype=torch.float8_e4m3fn):
    """Context manager for KV cache quantization operations."""
    try:
        backend_config = normalize_kv_cache_backend_config(static_kv_dtype)
        attention_module_count = sum(1 for module in model.modules() if is_attention_module(module))
        if backend_config.backend == "turboquant":
            logger.info(
                "Enable KV cache backend turboquant (bits=%s, seed=%s) for %s attention modules.",
                backend_config.bits,
                backend_config.seed,
                attention_module_count,
            )
        else:
            logger.info(
                "Enable KV cache backend %s (dtype=%s) for %s attention modules.",
                backend_config.backend,
                backend_config.dtype,
                attention_module_count,
            )
        initialize_fn = partial(initialize_quantized_kv_cache, config=backend_config)
        model.apply(initialize_fn)
        model.apply(prep_attention_module_for_calibration)
        yield model

    finally:
        model.apply(_cleanup_kv_cache_hooks)
        model.apply(freeze_module_quantization_)
        QuantizedKVParameterCache._instance = None
        QuantizedKVParameterCache._initialized = False
