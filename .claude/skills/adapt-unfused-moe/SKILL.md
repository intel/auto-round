---
name: adapt-unfused-moe
description: "Register a new MoE or hybrid SSM+MoE LLM via AutoRound's transformers-5.0+ unfused_moe pipeline. Use when the model has fused 3D expert Parameters, resolves its mixer via a dispatch dict, needs post-load tensor fixups, or requires per-model default layer_config overrides. Reference implementation: Nemotron-H."
---

# `unfused_moe/` registration

Register in `auto_round/modeling/unfused_moe/__init__.py::MODEL_CONFIG`.

| Field | Purpose |
|---|---|
| `block_patch` | `(orig, replacement)` class swap via `setattr`. |
| `dispatch_dict_patch` | For mixers resolved via a dict (e.g. NH `MIXER_TYPES["moe"]`); `setattr` alone misses it. |
| `preserve_upstream_conversion_mapping` + `drop_conversion_target_patterns` | Keep upstream `backbone.*→model.*` renames; drop only expert-bundling converters. Empty `checkpoint_mapping=[]` silently loads backbone at random init. |
| `post_load_fn` | Dotted path, run after `from_pretrained` before calibration. |
| `default_layer_config_patterns_fn` | Returns `{regex: overlay}`; merged into user `layer_config` (user wins per-pattern). |

Replacement MoE block exposes routed experts as `nn.ModuleList[MLP]` so
AutoRound's `nn.Linear` walker sees them. Forward = verbatim port of upstream.

## Hybrid SSM+MoE (Nemotron-H reference)

Reference: `nemotron_h.py` + `nemotron_h_setup.py`.

- **`low_cpu_mem_usage` bypasses `__init__`** — set missing attrs on both
  instance (`object.__setattr__`) and class. Example: `Zamba2RMSNormGated.group_size`.
- **BF16 kills SSM recurrence + router bias** — `A_log`/`D`/`dt_bias` accumulate
  through `exp()`, `e_score_correction_bias` flips top-k. Reload from source at
  FP32 via `_restore_tensors_from_source` (<1 MB for 30B).
- **Small-magnitude out_proj** — Mamba2 `mixer.out_proj` collapses FP16 scales to
  sub-normals → default `scale_dtype=bfloat16` via pattern.
- **Pattern syntax** uses `to_standard_regex` input: plain dots, no backslash.
- **Idempotency** — set `model._autoround_<name>_post_load_applied = True`.

## User API

```python
AutoRound(
    "nvidia/Nemotron-Cascade-2-30B-A3B", bits=4, group_size=64, sym=True, low_cpu_mem_usage=True
).quantize_and_save(...)
# group_size=64 because intermediate_size=1856 is not divisible by 128 (1856/128=14.5); 1856/64=29 ✓
# offline: post_load_overrides={"enable_high_precision_overrides": False}
```

## Tests

Per registration: (a) conversion mapping has expected drops, (b) `post_load_fn`
no-ops on other `model_type`, (c) default patterns merge without overwriting
user entries. See `test/test_cpu/models/test_nemotron_h.py`.
