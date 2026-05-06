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

"""Dry-run estimation utilities for AutoRound.

Estimates VRAM usage, output file size, and approximate quantization time
from model configuration metadata without loading model weights.
"""

import math

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
}

# Rough seconds per layer per iteration, measured on A100 for a 7B-class model.
# Actual speed varies widely by hardware and model architecture.
_SECS_PER_LAYER_PER_ITER = 0.12


def _count_parameters(config):  # pylint: disable=too-many-locals
    """Estimate total parameter count from a transformers model config.

    Uses hidden_size, intermediate_size, num_hidden_layers, and vocab_size
    to compute a rough parameter count.  Falls back to a simple
    hidden_size^2 * num_layers heuristic when fields are missing.
    """
    hidden = getattr(config, "hidden_size", None)
    num_layers = getattr(config, "num_hidden_layers", None)
    if hidden is None or num_layers is None:
        return None

    intermediate = getattr(config, "intermediate_size", None)
    vocab_size = getattr(config, "vocab_size", None)
    num_heads = getattr(config, "num_attention_heads", None)
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)

    attn_params = _count_attention_params(hidden, num_heads, num_kv_heads)
    ffn_params = _count_ffn_params(hidden, intermediate)
    layer_params = attn_params + ffn_params + 2 * hidden  # 2 layer norms

    total = num_layers * layer_params
    total += _count_embedding_params(config, hidden, vocab_size)
    return total


def _count_attention_params(hidden, num_heads, num_kv_heads):
    """Count attention layer parameters (Q, K, V, O projections)."""
    head_dim = hidden // num_heads if num_heads else hidden
    kv_dim = num_kv_heads * head_dim if num_kv_heads else hidden
    return hidden * hidden + 2 * hidden * kv_dim + hidden * hidden


def _count_ffn_params(hidden, intermediate):
    """Count FFN layer parameters."""
    if intermediate is not None:
        return 3 * hidden * intermediate  # gate + up + down
    return 4 * hidden * hidden  # classic 4x expansion


def _count_embedding_params(config, hidden, vocab_size):
    """Count embedding and LM head parameters."""
    if vocab_size is None:
        return 0
    embedding_params = vocab_size * hidden
    if getattr(config, "tie_word_embeddings", True):
        return embedding_params
    return 2 * embedding_params


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


def estimate_vram(param_count, model_dtype_bytes, batch_size, seqlen, hidden_size):
    """Estimate peak VRAM usage in bytes during quantization.

    This accounts for:
    - Model weights in the original dtype
    - Optimizer state and gradients for one block
    - Calibration activations (batch_size * seqlen * hidden_size)
    - CUDA overhead and fragmentation (~20% buffer)
    """
    # Model weights
    model_bytes = param_count * model_dtype_bytes

    # Activation memory for calibration (rough upper bound for one block)
    activation_bytes = batch_size * seqlen * hidden_size * model_dtype_bytes

    # Optimizer state: roughly 2x one block's parameters (momentum + variance for Adam)
    # Approximate one block as total_params / num_layers
    block_overhead = model_bytes * 0.05  # ~5% of model for one block's optimizer state

    # CUDA overhead and fragmentation buffer (~20%)
    subtotal = model_bytes + activation_bytes + block_overhead
    total = subtotal * 1.2

    return int(total)


def estimate_output_size(param_count, target_bits, group_size):
    """Estimate output file size in bytes for the quantized model.

    Accounts for quantized weights plus scale/zero-point overhead.
    """
    # Quantized weight bits
    weight_bits = param_count * target_bits

    # Scale and zero-point overhead (one fp16 scale per group, one zp per group)
    if group_size > 0:
        num_groups = math.ceil(param_count / group_size)
        # fp16 scale (2 bytes) + zero-point packed into target_bits
        overhead_bits = num_groups * (16 + target_bits)
    else:
        overhead_bits = 0

    total_bits = weight_bits + overhead_bits
    return int(math.ceil(total_bits / 8))


def estimate_time(num_layers, iters, nsamples, batch_size):
    """Estimate approximate quantization time in seconds.

    Based on empirical measurements - actual time varies significantly
    by hardware, model architecture, and sequence length.
    """
    batches_per_iter = math.ceil(nsamples / batch_size)
    total_seconds = num_layers * iters * batches_per_iter * _SECS_PER_LAYER_PER_ITER
    return total_seconds


_DRY_RUN_DEFAULTS = {
    "model_dtype": "float16",
    "batch_size": 8,
    "seqlen": 2048,
    "nsamples": 128,
    "iters": 200,
    "trust_remote_code": True,
    "platform": "hf",
}


def dry_run_estimate(model_name, scheme_bits, group_size, **kwargs):
    """Run a dry-run estimation and return a dict of estimates.

    Args:
        model_name: HuggingFace model name or local path.
        scheme_bits: Target quantization bit width (e.g. 4 for W4A16).
        group_size: Quantization group size.
        **kwargs: Optional overrides - model_dtype, batch_size, seqlen,
            nsamples, iters, trust_remote_code, platform.

    Returns:
        dict with keys: param_count, peak_vram_bytes, output_size_bytes,
        estimated_time_secs, and their formatted string versions.
    """
    opts = {**_DRY_RUN_DEFAULTS, **kwargs}
    config = _load_model_config(model_name, opts)

    param_count = _count_parameters(config)
    if param_count is None:
        logger.warning("Could not estimate parameter count from model config.")
        return None

    return _build_estimate_result(model_name, scheme_bits, group_size, param_count, config, opts)


def _load_model_config(model_name, opts):
    """Load model config from the specified platform."""
    auto_config = _load_auto_config(opts["platform"])
    return auto_config.from_pretrained(model_name, trust_remote_code=opts["trust_remote_code"])


def _build_estimate_result(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    model_name, scheme_bits, group_size, param_count, config, opts
):
    """Build the estimation result dictionary."""
    hidden_size = getattr(config, "hidden_size", 4096)
    num_layers = getattr(config, "num_hidden_layers", 32)
    dtype_bytes = DTYPE_BYTES.get(opts["model_dtype"], 2)

    peak_vram = estimate_vram(param_count, dtype_bytes, opts["batch_size"], opts["seqlen"], hidden_size)
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
        **{k: opts[k] for k in ("model_dtype", "batch_size", "seqlen", "nsamples", "iters")},
    }


def _load_auto_config(platform):
    """Load the appropriate AutoConfig class for the platform."""
    if platform == "model_scope":
        from modelscope import AutoConfig  # pylint: disable=import-outside-toplevel
    else:
        from transformers import AutoConfig  # pylint: disable=import-outside-toplevel
    return AutoConfig


def print_dry_run_report(estimates):
    """Print a formatted dry-run estimation report to stdout."""
    if estimates is None:
        logger.error("Dry-run estimation failed: could not determine model parameters.")
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
    print(border)
    print(f"  Estimated peak VRAM:    {estimates['peak_vram_str']}")
    print(f"  Estimated output size:  {estimates['output_size_str']}")
    print(f"  Estimated time:         {estimates['estimated_time_str']}")
    print(
        f"    (batch_size={estimates['batch_size']}, seqlen={estimates['seqlen']}, "
        f"nsamples={estimates['nsamples']}, iters={estimates['iters']})"
    )
    print(border)
    print("  NOTE: These are rough estimates. Actual values depend on")
    print("  hardware, model architecture, and runtime conditions.")
    print(f"{border}\n")
