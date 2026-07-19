# Generic SVDQuant Nunchaku Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a generic SVDQuant export surface that can describe Nunchaku-style MXFP4 artifacts without depending on DeepCompressor, Nunchaku, or a FLUX-specific module layout.

**Architecture:** Keep SVDQuant decomposition model-agnostic. Add a standalone safetensors exporter that scans `SVDQuantLinear` modules, writes smooth and low-rank tensors in a stable schema, and leaves residual MXFP4 packing behind an explicit provider interface. Document the current branch, gaps to DeepCompressor parity, and the follow-up work needed for complete Nunchaku runtime loading.

**Tech Stack:** Python, PyTorch, safetensors, pytest.

---

### Task 1: Document the Current SVDQuant Design

**Files:**
- Create: `docs/svdquant_nunchaku_mxfp4_review.md`

**Step 1: Write the review document**

Cover:
- Current `SVDQuantTransform` behavior.
- What makes it generic.
- What is missing for DeepCompressor-level SVDQuant.
- The Nunchaku-style tensor schema.
- Why exporter code must not import DeepCompressor or Nunchaku.
- Follow-up tests for MXFP4 packing.

**Step 2: Review for ambiguity**

Run:
```bash
sed -n '1,240p' docs/svdquant_nunchaku_mxfp4_review.md
```

Expected: The doc clearly distinguishes implemented generic export from future full Nunchaku runtime compatibility.

### Task 2: Add a Generic SVDQuant Exporter

**Files:**
- Create: `auto_round/export/svdquant_nunchaku.py`

**Step 1: Write failing tests**

Create tests that build a toy `SVDQuantLinear`, export it, and verify:
- `smooth` and `smooth_orig` exist.
- `lora_down` is saved as `(in_features, rank)`.
- `lora_up` is saved as `(out_features, rank)`.
- metadata contains a JSON `quantization_config`.
- exporter source does not import `deepcompressor` or `nunchaku`.

**Step 2: Implement exporter**

Implement:
- `SVDQuantExportConfig`
- `ResidualTensorProvider`
- `collect_svdquant_tensors`
- `save_svdquant_nunchaku_safetensors`

The default exporter should save unpacked residual weights as an explicit fallback. A future MXFP4 packer can implement `ResidualTensorProvider`.

**Step 3: Run tests**

Run:
```bash
pytest test/test_cpu/export/test_svdquant_nunchaku_export.py -q
```

Expected: PASS.

### Task 3: Verify No External Runtime Dependency

**Files:**
- Test: `test/test_cpu/export/test_svdquant_nunchaku_export.py`

**Step 1: Add source inspection test**

Assert `auto_round.export.svdquant_nunchaku` source contains no `deepcompressor` or `nunchaku` import.

**Step 2: Run focused tests**

Run:
```bash
pytest test/test_cpu/export/test_svdquant_nunchaku_export.py test/test_cpu/algorithms/test_svdquant.py -q
```

Expected: PASS.
