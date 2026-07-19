# Fixed RTN Residual Outer Iteration Implementation Plan

**Goal:** Define SVDQuant residual outer iteration as permanently RTN QDQ while allowing the downstream residual quantizer to remain RTN or SignRound.

**Architecture:** Keep `residual_quant_method="rtn"` as a compatibility-only public argument. Reject every other value, use RTN QDQ inside SVDQuant, and keep `--algorithm` responsible for selecting the final downstream quantizer.

**Tech Stack:** Python, PyTorch, AutoRound quantization pipeline, pytest.

---

### Task 1: Lock the public contract

**Files:**
- Modify: `test/test_cpu/algorithms/test_svdquant.py`
- Modify: `auto_round/algorithms/transforms/svdquant/config.py`
- Modify: `auto_round/cli/parser.py`

1. Change the rejection test to require wording that RTN is fixed by design.
2. Run the test and verify it fails against the old "not implemented" wording.
3. Update configuration validation and CLI help without changing accepted values.
4. Run the focused configuration and CLI tests.

### Task 2: Correct the design documentation

**Files:**
- Modify: `docs/superpowers/specs/2026-07-13-svdquant-rtn-residual-iteration-design.md`
- Modify: `docs/svdquant_nunchaku_mxfp4_review.md`

1. Remove the future SignRound outer-iteration proposal.
2. State that RTN and SignRound pipelines both use RTN QDQ for residual outer iteration.
3. Keep final downstream quantization ownership separate from the outer loop.

### Task 3: Verify behavior

1. Run SVDQuant algorithm, residual, CLI, and exporter tests.
2. Verify that production SVDQuant code still has no DeepCompressor or Nunchaku imports.
3. Run the Nunchaku MXFP4 CUDA smoke test in the shared uv environment.
