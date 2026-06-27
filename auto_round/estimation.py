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

"""Dry-run estimation utilities for AutoRound.

The estimator intentionally loads only model configuration metadata.  It builds a
representative decoder block on the meta device and routes that block through
AutoRound's real block-wise tuning-memory helper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from auto_round.logger import logger

DTYPE_BYTES = {
    "float32": 4,
    "fp32": 4,
    "float16": 2,
    "fp16": 2,
    "bfloat16": 2,
    "bf16": 2,
    "float8_e4m3fn": 1,
    "fp8": 1,
    "auto": 2,
}

_TORCH_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "auto": torch.float16,
}

_HIDDEN_SIZE_ATTRS = (
    "hidden_size",
    "n_embd",
    "d_model",
    "embed_dim",
    "decoder_hidden_size",
    "text_hidden_size",
)
_INTERMEDIATE_SIZE_ATTRS = (
    "intermediate_size",
    "n_inner",
    "ffn_dim",
    "ffn_hidden_size",
    "hidden_dim",
    "d_ff",
    "decoder_ffn_dim",
    "filter_size",
)
_MOE_INTERMEDIATE_SIZE_ATTRS = (
    "moe_intermediate_size",
    "moe_ffn_hidden_size",
    "expert_intermediate_size",
    "expert_ffn_hidden_size",
)
_NUM_HEADS_ATTRS = ("num_attention_heads", "n_head", "num_heads", "decoder_attention_heads")
_NUM_KV_HEADS_ATTRS = ("num_key_value_heads", "num_kv_heads", "n_kv_heads", "multi_query_group_num")
_VOCAB_SIZE_ATTRS = ("vocab_size", "padded_vocab_size", "n_vocab")
_LAYER_COUNT_ATTRS = (
    "num_hidden_layers",
    "n_layer",
    "n_layers",
    "num_layers",
    "num_decoder_layers",
    "decoder_layers",
    "encoder_layers",
    "depth",
    "layers",
)
_EXTRA_LAYER_COUNT_ATTRS = ("mtp_num_hidden_layers", "num_nextn_predict_layers")
_TEXT_CONFIG_ATTRS = (
    "text_config",
    "llm_config",
    "language_config",
    "language_model_config",
    "decoder_config",
    "thinker_config",
    "talker_config",
)
_EXPERT_COUNT_ATTRS = ("num_local_experts", "num_experts", "moe_num_primary_experts", "n_routed_experts")

# Rough seconds per layer per iteration, measured on A100 for a 7B-class model.
# Actual speed varies widely by hardware and model architecture.
_SECS_PER_LAYER_PER_ITER = 0.12


@dataclass(frozen=True)
class _BlockMemoryEstimate:
    layer_activation_gb: float
    block_input_output_gb: float
    effective_block_input_output_gb: float
    additional_gb: float
    block_param_gb: float
    card_0_used_gb: float
    has_moe: bool
    moe_memory_ratio: float


class _ConfigView:
    """Attribute fallback over nested HF config objects."""

    def __init__(self, *configs):
        self.configs = [config for config in configs if config is not None]

    def __getattr__(self, name: str):
        for config in self.configs:
            value = _get_attr(config, name)
            if value is not None:
                return value
        raise AttributeError(name)


def _get_attr(obj, name: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _get_first_attr(obj, names: Iterable[str], default=None):
    for name in names:
        value = _get_attr(obj, name)
        if value is not None:
            return value
    return default


def _positive_int(value) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer():
        int_value = int(value)
        return int_value if int_value > 0 else None
    if isinstance(value, str) and value.isdigit():
        int_value = int(value)
        return int_value if int_value > 0 else None
    if isinstance(value, (list, tuple)):
        return len(value) if len(value) > 0 else None
    return None


def _iter_config_candidates(config) -> list[Any]:
    candidates = []
    seen = set()

    def add(candidate):
        if candidate is None or id(candidate) in seen:
            return
        seen.add(id(candidate))
        candidates.append(candidate)

    def add_text_children(candidate):
        for attr in _TEXT_CONFIG_ATTRS:
            child = _get_attr(candidate, attr)
            if child is not None:
                add(child)
                add_text_children(child)

    add_text_children(config)
    add(config)
    for attr in _TEXT_CONFIG_ATTRS:
        child = _get_attr(config, attr)
        if child is not None:
            add(child)
    return candidates


def _config_view(config) -> _ConfigView:
    return _ConfigView(*_iter_config_candidates(config))


def discover_num_layers(config) -> int | None:
    """Discover decoder block count across common HF config field names."""
    for candidate in _iter_config_candidates(config):
        layer_types = _get_attr(candidate, "layer_types")
        layer_count = _positive_int(layer_types)
        if layer_count is not None:
            return layer_count + _extra_layer_count(candidate)

        for attr in _LAYER_COUNT_ATTRS:
            layer_count = _positive_int(_get_attr(candidate, attr))
            if layer_count is not None:
                return layer_count + _extra_layer_count(candidate)
    return None


def _extra_layer_count(config) -> int:
    total = 0
    for attr in _EXTRA_LAYER_COUNT_ATTRS:
        total += _positive_int(_get_attr(config, attr)) or 0
    return total


def _discover_hidden_size(config) -> int | None:
    for candidate in _iter_config_candidates(config):
        hidden_size = _positive_int(_get_first_attr(candidate, _HIDDEN_SIZE_ATTRS))
        if hidden_size is not None:
            return hidden_size
    return None


def _discover_intermediate_size(config, hidden_size: int) -> int:
    for attrs in (_MOE_INTERMEDIATE_SIZE_ATTRS, _INTERMEDIATE_SIZE_ATTRS):
        for candidate in _iter_config_candidates(config):
            intermediate_size = _positive_int(_get_first_attr(candidate, attrs))
            if intermediate_size is not None:
                return intermediate_size
    return hidden_size * 4


def _discover_num_heads(config) -> int | None:
    for candidate in _iter_config_candidates(config):
        num_heads = _positive_int(_get_first_attr(candidate, _NUM_HEADS_ATTRS))
        if num_heads is not None:
            return num_heads
    return None


def _discover_num_kv_heads(config, num_heads: int | None) -> int | None:
    for candidate in _iter_config_candidates(config):
        num_kv_heads = _positive_int(_get_first_attr(candidate, _NUM_KV_HEADS_ATTRS))
        if num_kv_heads is not None:
            return num_kv_heads
    return num_heads


def _discover_vocab_size(config) -> int | None:
    for candidate in _iter_config_candidates(config):
        vocab_size = _positive_int(_get_first_attr(candidate, _VOCAB_SIZE_ATTRS))
        if vocab_size is not None:
            return vocab_size
    return None


def _discover_num_experts(config) -> int | None:
    for candidate in _iter_config_candidates(config):
        num_experts = _positive_int(_get_first_attr(candidate, _EXPERT_COUNT_ATTRS))
        if num_experts is not None:
            return num_experts
    return None


def _torch_dtype(dtype_name: str | None):
    return _TORCH_DTYPES.get((dtype_name or "float16").lower(), torch.float16)


def _dtype_bytes(dtype_name: str | None) -> int:
    return DTYPE_BYTES.get((dtype_name or "float16").lower(), 2)


def _linear(in_features, out_features, *, bits, act_bits, dtype):
    layer = torch.nn.Linear(
        max(1, int(in_features)),
        max(1, int(out_features)),
        bias=False,
        device="meta",
        dtype=dtype,
    )
    layer.bits = 16 if bits is None else bits
    layer.act_bits = 16 if act_bits is None else act_bits
    return layer


class _SyntheticAttention(torch.nn.Module):
    def __init__(self, hidden_size, kv_size, *, bits, act_bits, dtype):
        super().__init__()
        self.q_proj = _linear(hidden_size, hidden_size, bits=bits, act_bits=act_bits, dtype=dtype)
        self.k_proj = _linear(hidden_size, kv_size, bits=bits, act_bits=act_bits, dtype=dtype)
        self.v_proj = _linear(hidden_size, kv_size, bits=bits, act_bits=act_bits, dtype=dtype)
        self.o_proj = _linear(hidden_size, hidden_size, bits=bits, act_bits=act_bits, dtype=dtype)


class _SyntheticDenseMLP(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, *, bits, act_bits, dtype):
        super().__init__()
        self.gate_proj = _linear(hidden_size, intermediate_size, bits=bits, act_bits=act_bits, dtype=dtype)
        self.up_proj = _linear(hidden_size, intermediate_size, bits=bits, act_bits=act_bits, dtype=dtype)
        self.down_proj = _linear(intermediate_size, hidden_size, bits=bits, act_bits=act_bits, dtype=dtype)


class _SyntheticExpert(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, *, bits, act_bits, dtype):
        super().__init__()
        self.gate_proj = _linear(hidden_size, intermediate_size, bits=bits, act_bits=act_bits, dtype=dtype)
        self.up_proj = _linear(hidden_size, intermediate_size, bits=bits, act_bits=act_bits, dtype=dtype)
        self.down_proj = _linear(intermediate_size, hidden_size, bits=bits, act_bits=act_bits, dtype=dtype)


class _SyntheticMoeLayer(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts, *, bits, act_bits, dtype):
        super().__init__()
        self.gate = _linear(hidden_size, num_experts, bits=bits, act_bits=act_bits, dtype=dtype)
        self.experts = torch.nn.ModuleList(
            [
                _SyntheticExpert(hidden_size, intermediate_size, bits=bits, act_bits=act_bits, dtype=dtype)
                for _ in range(num_experts)
            ]
        )


class _SyntheticDecoderBlock(torch.nn.Module):
    def __init__(
        self,
        config,
        *,
        hidden_size,
        intermediate_size,
        kv_size,
        num_experts,
        bits,
        act_bits,
        dtype,
    ):
        super().__init__()
        self.config = config
        self.self_attn = _SyntheticAttention(hidden_size, kv_size, bits=bits, act_bits=act_bits, dtype=dtype)
        if num_experts:
            self.mlp = _SyntheticMoeLayer(
                hidden_size,
                intermediate_size,
                num_experts,
                bits=bits,
                act_bits=act_bits,
                dtype=dtype,
            )
        else:
            self.mlp = _SyntheticDenseMLP(hidden_size, intermediate_size, bits=bits, act_bits=act_bits, dtype=dtype)


def _build_synthetic_block(config, scheme_bits, act_bits, model_dtype):
    hidden_size = _discover_hidden_size(config)
    if hidden_size is None:
        raise ValueError("Could not infer hidden size from model config.")

    num_heads = _discover_num_heads(config)
    head_dim = hidden_size // num_heads if num_heads else hidden_size
    num_kv_heads = _discover_num_kv_heads(config, num_heads)
    kv_size = hidden_size if num_kv_heads is None else num_kv_heads * head_dim
    intermediate_size = _discover_intermediate_size(config, hidden_size)
    num_experts = _discover_num_experts(config)

    return _SyntheticDecoderBlock(
        _config_view(config),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        kv_size=kv_size,
        num_experts=num_experts,
        bits=scheme_bits,
        act_bits=act_bits,
        dtype=_torch_dtype(model_dtype),
    )


def _synthetic_block_inputs(hidden_size, nsamples, seqlen, model_dtype):
    return [torch.empty(max(1, nsamples), max(1, seqlen), hidden_size, device="meta", dtype=_torch_dtype(model_dtype))]


def estimate_block_vram(
    config,
    scheme_bits,
    *,
    act_bits=None,
    model_dtype="float16",
    batch_size=8,
    seqlen=2048,
    nsamples=128,
    low_gpu_mem_usage=False,
) -> _BlockMemoryEstimate:
    """Estimate peak block tuning memory using ``auto_round.utils.device`` helpers."""
    from auto_round.utils.device import estimate_tuning_block_mem, get_moe_memory_ratio

    hidden_size = _discover_hidden_size(config)
    if hidden_size is None:
        raise ValueError("Could not infer hidden size from model config.")

    block = _build_synthetic_block(config, scheme_bits, act_bits, model_dtype)
    input_ids = _synthetic_block_inputs(hidden_size, nsamples, seqlen, model_dtype)
    effective_batch_size = min(max(1, batch_size), max(1, nsamples))
    layer_memory_dict, layer_activation_gb, block_input_output_gb, additional_gb = estimate_tuning_block_mem(
        block,
        input_ids,
        effective_batch_size,
    )
    moe_memory_ratio, has_moe = get_moe_memory_ratio(block)
    effective_block_input_output_gb = 0 if low_gpu_mem_usage else block_input_output_gb
    card_0_used_gb = effective_block_input_output_gb + layer_activation_gb + additional_gb
    block_param_gb = sum(info["param_memory"] for info in layer_memory_dict.values())

    return _BlockMemoryEstimate(
        layer_activation_gb=layer_activation_gb,
        block_input_output_gb=block_input_output_gb,
        effective_block_input_output_gb=effective_block_input_output_gb,
        additional_gb=additional_gb,
        block_param_gb=block_param_gb,
        card_0_used_gb=card_0_used_gb,
        has_moe=has_moe,
        moe_memory_ratio=moe_memory_ratio,
    )


def _count_block_parameters(block) -> int:
    return sum(module.weight.numel() for module in block.modules() if hasattr(module, "weight"))


def estimate_parameter_count(config, scheme_bits=4, act_bits=None, model_dtype="float16") -> int | None:
    """Estimate total model parameters from a synthetic block plus embeddings."""
    num_layers = discover_num_layers(config)
    if num_layers is None:
        return None

    try:
        block = _build_synthetic_block(config, scheme_bits, act_bits, model_dtype)
    except ValueError:
        return None

    hidden_size = _discover_hidden_size(config)
    total = _count_block_parameters(block) * num_layers
    vocab_size = _discover_vocab_size(config)
    if hidden_size is not None and vocab_size is not None:
        embedding_params = vocab_size * hidden_size
        total += embedding_params
        if not bool(_get_first_attr(_config_view(config), ("tie_word_embeddings",), True)):
            total += embedding_params
    return total


def _format_bytes(num_bytes):
    """Format byte count as a human-readable string."""
    if num_bytes >= 1e12:
        return f"{num_bytes / 1e12:.2f} TB"
    if num_bytes >= 1e9:
        return f"{num_bytes / 1e9:.2f} GB"
    if num_bytes >= 1e6:
        return f"{num_bytes / 1e6:.2f} MB"
    return f"{num_bytes / 1e3:.2f} KB"


def _format_time(seconds):
    """Format seconds as a human-readable time string."""
    if seconds >= 3600:
        hours = seconds / 3600
        return f"{hours:.1f} hours"
    if seconds >= 60:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    return f"{seconds:.0f} seconds"


def estimate_output_size(param_count, target_bits, group_size):
    """Estimate output file size in bytes for the quantized model."""
    weight_bits = param_count * target_bits
    normalized_group_size = _normalize_group_size(group_size)
    if normalized_group_size and normalized_group_size > 0:
        num_groups = math.ceil(param_count / normalized_group_size)
        overhead_bits = num_groups * (16 + target_bits)
    else:
        overhead_bits = 0
    return int(math.ceil((weight_bits + overhead_bits) / 8))


def _normalize_group_size(group_size):
    if isinstance(group_size, (tuple, list)):
        positive_sizes = [int(item) for item in group_size if int(item) > 0]
        return positive_sizes[0] if positive_sizes else 0
    return group_size


def estimate_time(num_layers, iters, nsamples, batch_size):
    """Estimate approximate quantization time in seconds."""
    batches_per_iter = math.ceil(nsamples / batch_size)
    return num_layers * iters * batches_per_iter * _SECS_PER_LAYER_PER_ITER


_DRY_RUN_DEFAULTS = {
    "model_dtype": "float16",
    "batch_size": 8,
    "seqlen": 2048,
    "nsamples": 128,
    "iters": 200,
    "trust_remote_code": True,
    "platform": "hf",
    "low_gpu_mem_usage": False,
    "act_bits": None,
}


def dry_run_estimate(model_name, scheme_bits, group_size, **kwargs):
    """Run a dry-run estimation and return a dict of estimates."""
    opts = {**_DRY_RUN_DEFAULTS, **{key: value for key, value in kwargs.items() if value is not None}}
    config = _load_model_config(model_name, opts)

    num_layers = discover_num_layers(config)
    param_count = estimate_parameter_count(config, scheme_bits, opts["act_bits"], opts["model_dtype"])
    if num_layers is None or param_count is None:
        logger.warning("Could not estimate model structure from model config.")
        return None

    return _build_estimate_result(model_name, scheme_bits, group_size, param_count, num_layers, config, opts)


def _load_model_config(model_name, opts):
    """Load model config from the specified platform."""
    auto_config = _load_auto_config(opts["platform"])
    return auto_config.from_pretrained(model_name, trust_remote_code=opts["trust_remote_code"])


def _build_estimate_result(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    model_name,
    scheme_bits,
    group_size,
    param_count,
    num_layers,
    config,
    opts,
):
    """Build the estimation result dictionary."""
    block_memory = estimate_block_vram(
        config,
        scheme_bits,
        act_bits=opts["act_bits"],
        model_dtype=opts["model_dtype"],
        batch_size=opts["batch_size"],
        seqlen=opts["seqlen"],
        nsamples=opts["nsamples"],
        low_gpu_mem_usage=opts["low_gpu_mem_usage"],
    )
    peak_vram = int(block_memory.card_0_used_gb * 1024**3)
    output_size = estimate_output_size(param_count, scheme_bits, group_size)
    est_time = estimate_time(num_layers, opts["iters"], opts["nsamples"], opts["batch_size"])
    param_str = f"{param_count / 1e9:.2f}B" if param_count >= 1e9 else f"{param_count / 1e6:.1f}M"

    return {
        "model_name": model_name,
        "param_count": param_count,
        "param_count_str": param_str,
        "peak_vram_bytes": peak_vram,
        "peak_vram_str": _format_bytes(peak_vram),
        "output_size_bytes": output_size,
        "output_size_str": _format_bytes(output_size),
        "estimated_time_secs": est_time,
        "estimated_time_str": _format_time(est_time),
        "scheme_bits": scheme_bits,
        "group_size": group_size,
        "num_layers": num_layers,
        "block_input_output_cache_bytes": int(block_memory.block_input_output_gb * 1024**3),
        "block_input_output_cache_str": _format_bytes(int(block_memory.block_input_output_gb * 1024**3)),
        "effective_block_input_output_cache_bytes": int(block_memory.effective_block_input_output_gb * 1024**3),
        "layer_activation_bytes": int(block_memory.layer_activation_gb * 1024**3),
        "additional_memory_bytes": int(block_memory.additional_gb * 1024**3),
        "block_param_memory_bytes": int(block_memory.block_param_gb * 1024**3),
        "has_moe": block_memory.has_moe,
        "moe_memory_ratio": block_memory.moe_memory_ratio,
        **{k: opts[k] for k in ("model_dtype", "batch_size", "seqlen", "nsamples", "iters", "low_gpu_mem_usage")},
    }


def _load_auto_config(platform):
    """Load the appropriate AutoConfig class for the platform."""
    if platform == "model_scope":
        from modelscope import AutoConfig  # pylint: disable=E0401,import-outside-toplevel
    else:
        from transformers import AutoConfig  # pylint: disable=import-outside-toplevel
    return AutoConfig


def print_dry_run_report(estimates):
    """Print a formatted dry-run estimation report to stdout."""
    if estimates is None:
        logger.error("Dry-run estimation failed: could not determine model structure.")
        return

    border = "=" * 60
    print(f"\n{border}")
    print("  AutoRound Dry-Run Estimation")
    print(border)
    print(f"  Model:              {estimates['model_name']}")
    print(f"  Parameters:         {estimates['param_count_str']}")
    print(f"  Layers:             {estimates['num_layers']}")
    print(f"  Target bits:        {estimates['scheme_bits']}")
    print(f"  Group size:         {estimates['group_size']}")
    print(f"  Model dtype:        {estimates['model_dtype']}")
    print(f"  low_gpu_mem_usage:  {estimates['low_gpu_mem_usage']}")
    if estimates["has_moe"]:
        print(f"  MoE active ratio:   {estimates['moe_memory_ratio']:.4f}")
    print(border)
    print(f"  Estimated peak VRAM:       {estimates['peak_vram_str']}")
    print(f"  Block I/O cache overhead:  {estimates['block_input_output_cache_str']}")
    print(f"  Estimated output size:     {estimates['output_size_str']}")
    print(f"  Estimated time:            {estimates['estimated_time_str']}")
    print(
        f"    (batch_size={estimates['batch_size']}, seqlen={estimates['seqlen']}, "
        f"nsamples={estimates['nsamples']}, iters={estimates['iters']})"
    )
    print(border)
    print("  NOTE: VRAM is estimated per tuned block from AutoRound's")
    print("  block-wise device memory model. Actual values depend on")
    print("  hardware, model architecture, and runtime conditions.")
    print(f"{border}\n")
