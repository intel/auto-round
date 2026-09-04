## feat(rrq): Phase 1 - Recurrent Residual Quantization (packed INT2 2+2+2+2)

### Summary

Implements the RRQ (Recurrent Residual Quantization) algorithm for LLM quantization. Each layer is recursively quantized into 4 planes of INT2 (2+2+2+2), enabling 2/4/6/8-bit dynamic precision switching at inference time. The base plane uses the standard AutoRound INT2 export; the 3 residual planes are packed together into a single `auto_round:rrq` artifact.

### Design

- **Quantization**: Pure RTN (zero-shot, no calibration). 4 recursive rounds per layer, each quantizing the residual of the previous round.
- **Storage**: Packed INT2 (reuses the W2A16 `QuantLinear` pack/forward code path).
  - Base: standard `auto_round` INT2 export (`qweight/scales/qzeros`)
  - Residual: 3 planes packed into a single `auto_round:rrq` artifact (`qweight_1/2/3`, `scales_1/2/3`, `qzeros_1/2/3`)
- **Inference**: `RRQLinear` = base QuantLinear + N residual QuantLinear. `set_rrq_bits(model, bits)` switches precision dynamically (2/4/6/8-bit).

### Changed Files

| Category | Files |
| --- | --- |
| Algorithm | `auto_round/algorithms/quantization/rrq/` (config + quantizer) |
| Export | `export_to_rrq.py`, `formats/backends/rrq.py` |
| Inference | `inference/rrq_linear.py`, `inference/rrq_model.py` |
| Integration | `__init__.py`, `registry.py`, `common.py`, `model_free.py`, `backend.py` |
| Bug fixes | `qlinear_torch.py` (asym pack `self.device` → `device`), GGUF/MLX export guards |
| Tests | `test/unit/test_cpu/algorithms/test_rrq.py` (23 tests) |
| Tooling | `test_rrq_qwen3_06b.py` (quantize + verify), `test_rrq_lm_eval.py` (lm-eval benchmark) |
| Docs | `docs/rrq_rfc.md`, `rrq_rfc_CN.md`, `rrq_progress_CN.md` |

### Validation (Qwen3-0.6B, group_size=128, asymmetric, XPU)

#### File Size

| Artifact | Size | vs. fp32 (1.40 GB) |
| --- | --- | --- |
| Base model | 420 MB | 29.3% |
| Residual model | 337 MB | 23.5% |
| Combined | 757 MB | 53.8% |

The residual quantized-weight portion is **3×** the base quantized-weight portion (as designed). The base file is larger due to float16 embeddings + layernorms that are not included in the residual.

#### Accuracy (HellaSwag, limit=200, XPU)

| Bit-width | Planes | Accuracy | Δ vs fp32 |
| --- | --- | --- | --- |
| fp32 | — | 43.5% | — |
| **6-bit** | **3** | **43.5%** | **0** ✅ |
| 8-bit | 4 | 42.5% | -1.0pp |
| 4-bit | 2 | 35.5% | -8.0pp |
| 2-bit | 1 | 26.5% | -17.0pp |

> 6-bit accuracy matches fp32 (the 1pp difference at 8-bit is within statistical noise for 200 samples).

#### Unit Tests

```
23/23 passed in 0.15s
```

Coverage: config validation, packed INT2 storage format, residual convergence, symmetric/asymmetric, RRQLinear forward & precision switching, export buffer rename, save config attach, load validation.

### Known Limitations (Phase 1)

- Pure RTN, no sign-SGD tuning (planned for Phase 2)
- Weight-only (act_bits=16), no activation quantization
- Fixed 2-bit per plane (no 3/4-bit per-plane support)
- GGUF/MLX export explicitly rejects RRQ residual models (fail-fast, no silent dropping)
- `save_quantized` requires manual `ar.formats = fmt` reset to prevent format latching (API-level cleanup pending)
