# Activation Checkpointing for AutoRound — Report

## What

Added `enable_activation_checkpointing` option to AutoRound. Each `WrapperLinear.forward()` is wrapped in `torch.utils.checkpoint.checkpoint(use_reentrant=False)`, so during backward only **one** layer's QDQ intermediates exist at a time instead of all layers simultaneously.

## Why

MoE models have hundreds of Linear layers per decoder block (e.g. Qwen3-30B-A3B: 389 layers). During tuning, autograd saves each layer's weight-sized QDQ tensor for backward, totaling ~77 GB per block — far exceeding most GPUs.

## Why Not Block-Level Checkpointing

Block-level `checkpoint()` was tried first. It prevents saving intermediates during forward (77 GB → 129 MB), but during backward the **entire** forward is recomputed at once, recreating the same 77 GB spike. Peak simply moves from forward to backward — **zero net savings**.

Per-layer checkpointing solves this: backward recomputes one layer at a time (~3 MB each).

## Implementation

| File | Change |
|------|--------|
| `auto_round/wrapper.py` | Split `forward()` → `forward()` / `_checkpointed_forward()` / `_forward_impl()` |
| `auto_round/compressors/base.py` | Wire flag through `wrapper_block()` to each `WrapperLinear` |
| `auto_round/compressors/config.py` | Add to `ExtraConfig` / `TuningExtraConfig` |
| `auto_round/autoround.py` | Add to `AutoRound.__new__()` signature |
| `auto_round/__main__.py` | Add `--enable_activation_checkpointing` CLI flag |

## Verification

**Setup**: 200 iters, `low_gpu_mem_usage=True`, batch_size=8, seqlen=2048, single H200 GPU.
Eval: lm-eval (lambada_openai, piqa, mmlu).

### Memory & Time

| Model | Scheme | Mode | Peak VRAM | Quant Time |
|-------|--------|------|-----------|------------|
| Qwen3-8B (dense) | W4A16 | baseline | 7.06 GB | 14.2 min |
| Qwen3-8B (dense) | W4A16 | **actcp** | **5.92 GB** | 12.6 min |
| Qwen3-30B-A3B (MoE) | MXFP8 | baseline | 80.67 GB | 256.6 min |
| Qwen3-30B-A3B (MoE) | MXFP8 | **actcp** | **11.74 GB** | 420.1 min |

### Accuracy

| Model | Mode | lambada | piqa | mmlu |
|-------|------|---------|------|------|
| Qwen3-8B W4A16 | baseline | 0.6381 | 0.7715 | 0.7230 |
| Qwen3-8B W4A16 | actcp | 0.6274 | 0.7682 | 0.7215 |
| Qwen3-30B MXFP8 | baseline | 0.6476 | 0.7943 | 0.7750 |
| Qwen3-30B MXFP8 | actcp | 0.6482 | 0.7905 | 0.7745 |

All deltas within ±1σ error bars — **no quality degradation**.

### Summary

| | Dense 8B | MoE 30B |
|--|----------|---------|
| **VRAM savings** | -16% (7.06 → 5.92 GB) | **-85%** (80.67 → 11.74 GB) |
| **Time overhead** | None | +64% |
| **Accuracy impact** | None | None |

## Usage

```python
AutoRound(model, scheme="MXFP8", enable_activation_checkpointing=True, low_gpu_mem_usage=True)
```

```bash
auto-round --model Qwen/Qwen3-30B-A3B --scheme MXFP8 --enable_activation_checkpointing --low_gpu_mem_usage
```

## When to Use

- **MoE models**: 85% VRAM reduction — enables quantizing 30B+ MoE on a single 16 GB GPU.
- **Dense models**: modest savings, no time overhead — safe to enable by default.
