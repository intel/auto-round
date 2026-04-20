---
name: adapt-unfused-moe
description: "Register a new MoE or hybrid SSM+MoE LLM via AutoRound's transformers-5.0+ unfused_moe pipeline. Use when the model has fused 3D expert Parameters, resolves its mixer via a dispatch dict, needs post-load tensor fixups, or requires per-model default layer_config overrides. Reference implementation: Nemotron-H."
---

# Adapting MoE / Hybrid Models via `unfused_moe/`

Complement to [adapt-new-llm](../adapt-new-llm/SKILL.md). Covers registration
in `auto_round/modeling/unfused_moe/` for transformers >= 5.0 models.

Existing entries: `deepseek_v3`, `qwen3_moe`, `qwen3_next`, `glm4_moe`,
`glm4_moe_lite`, `glm_moe_dsa`, `ernie4_5_moe`, `nemotron_h`.

## `MODEL_CONFIG` fields

Register in `auto_round/modeling/unfused_moe/__init__.py`.

| Field | Purpose |
|---|---|
| `block_patch` | `(orig_class_path, replacement_class_path)` — `setattr` swap of the MoE block class on the transformers module. |
| `dispatch_dict_patch` | For hybrid mixers resolved via a module-level dict (e.g. NH `MIXER_TYPES["moe"]`). `setattr` alone is insufficient. |
| `preserve_upstream_conversion_mapping` | Keep transformers' upstream `backbone.*`→`model.*` renames; drop only the expert-bundling entries via `drop_conversion_target_patterns`. Without this, returning `[]` silently loads the backbone at random init. |
| `drop_conversion_target_patterns` | Regex targets removed from the upstream rename rules. |
| `post_load_fn` | Dotted path to a callable run after `from_pretrained`, before calibration. See below. |
| `default_layer_config_patterns_fn` | Dotted path to a callable returning `{regex: overlay_dict}` merged into the user's `layer_config` (user wins per-pattern). |

The MoE block class (e.g. `LinearNemotronHMoE`) must expose routed experts as
`nn.ModuleList[MLP]` so AutoRound's `nn.Linear` walker finds them. The forward
logic is typically a verbatim port of the upstream MoE block.

## Hybrid SSM + MoE (Nemotron-H pattern)

Mamba2/Attention/MoE hybrids have three correctness concerns on top of MoE
unfusing. All gated on `config.model_type`, all invoked automatically by
`BaseCompressor.__init__` — never called by user code.

Reference: `auto_round/modeling/unfused_moe/nemotron_h.py` (block) +
`nemotron_h_setup.py` (fixups).

### 1. Post-load fixups (`post_load_fn`)

- **Missing attributes after `low_cpu_mem_usage=True`** — HF uses
  `cls.__new__(cls)` to materialise modules, skipping `__init__`. Any
  attribute set there is absent at forward time. Set it on both the
  instance (via `object.__setattr__`) and the class (defence against
  re-wraps). Example: `Zamba2RMSNormGated.group_size`.
- **High-precision tensor restore** — `torch_dtype=bf16` downcasts every
  float tensor. BF16 compounds catastrophically through `exp()` +
  sequence-length accumulation (SSM `A_log`, `D`, `dt_bias`) and flips
  top-k expert choice (`e_score_correction_bias`). Reload these few small
  tensors from the source checkpoint at FP32 via
  `auto_round.utils.source_tensor_overrides.restore_tensors_from_source`.
  Footprint < 1 MB for a 30B MoE.

Set `model._autoround_<name>_post_load_applied = True` at the end for
idempotency against legacy launcher pre-patching.

### 2. Default `layer_config` patterns

Force non-default quant settings for specific layers. Example: Mamba2
`mixer.out_proj` projects very small-magnitude SSM state, causing FP16
per-group scales to collapse to sub-normals → promote to BF16:

```python
def nemotron_h_default_layer_config_patterns():
    return {".*mixer.out_proj$": {"scale_dtype": torch.bfloat16}}
```

Pattern uses `to_standard_regex` input form — plain dots, no
backslash-escape (the helper escapes bare dots itself).

### 3. Conversion-mapping pitfall

On-disk `backbone.*` vs in-memory `model.*` is handled by transformers'
upstream rename. Setting `checkpoint_mapping = []` drops it and the
backbone silently stays at random init — quant produces garbage. Use
`preserve_upstream_conversion_mapping: True` + `drop_conversion_target_patterns`
listing only the expert-bundling converter targets.

## User API

Zero extra kwargs — the registry drives everything:

```python
AutoRound(
    "nvidia/Nemotron-Cascade-2-30B-A3B", bits=4, group_size=128, sym=True, low_cpu_mem_usage=True
).quantize_and_save(output_dir=...)
```

Opt-out for offline environments where the source checkpoint isn't reachable:

```python
AutoRound(..., post_load_overrides={"enable_high_precision_overrides": False})
```

## Tests

Each new registration should add tests under `test/test_cpu/models/`
covering: (a) conversion mapping preserved with expected drops, (b)
`post_load_fn` dispatches on matching `model_type` and no-ops otherwise,
(c) `default_layer_config_patterns_fn` merges without overriding user
entries. See `test_nemotron_h_registration.py` + `test_nemotron_h_post_load.py`.
