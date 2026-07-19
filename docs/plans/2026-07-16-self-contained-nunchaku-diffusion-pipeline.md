# Self-Contained Nunchaku Diffusion Pipeline Implementation Plan

**Goal:** Export and directly load a complete Diffusers pipeline with BF16 auxiliary components and an MXFP4 Nunchaku transformer.

**Architecture:** Reuse AutoRound's existing diffusion backend for full-pipeline loading and component saving. Rewrite the saved transformer component descriptor from onefile metadata, and teach Nunchaku to resolve a onefile inside a component directory.

**Tech Stack:** Python, PyTorch, Diffusers, safetensors, AutoRound, Nunchaku, pytest.

---

### Task 1: Lock the pipeline metadata contract

**Files:**
- Modify: `test/test_cpu/export/test_svdquant_nunchaku_format.py`
- Modify: `auto_round/compressors/diffusion_mixin.py`

1. Add failing tests for identifying an SVDQuant Nunchaku pipeline export.
2. Add a failing test that rewrites a transformer entry from safetensors metadata.
3. Implement metadata validation and atomic `model_index.json` rewriting.

### Task 2: Reuse the existing diffusion loader

**Files:**
- Delete: `auto_round/cli/model_loader.py`
- Modify: `auto_round/cli/parser.py`
- Modify: `auto_round/cli/main.py`
- Modify: `test/test_cpu/utils/test_cli_usage.py`

1. Remove `model_loader` from the public CLI and tests.
2. Pass the complete model path into `PipelineAutoRound` unchanged.
3. Keep Flux adapter auto-detection based on `model_index.json`.

### Task 3: Save a hybrid BF16/MXFP4 pipeline

**Files:**
- Modify: `auto_round/compressors/diffusion_mixin.py`
- Modify: `test/test_cpu/export/test_svdquant_nunchaku_format.py`

1. Save all non-transformer components through the existing diffusion path.
2. Save the packed transformer under `transformer/diffusion_pytorch_model.safetensors`.
3. Save `transformer/config.json` and patch the top-level component descriptor.
4. Verify no BF16 transformer shard is emitted.

### Task 4: Load a onefile component directory in Nunchaku

**Files:**
- Modify: `nunchaku/models/transformers/utils.py`
- Modify: `nunchaku/models/transformers/transformer_flux.py`
- Test: `tests/flux/test_directory_onefile.py`

1. Add a failing resolver test for `<component>/diffusion_pytorch_model.safetensors`.
2. Implement local-directory onefile discovery before precision detection.
3. Preserve legacy split-directory fallback.

### Task 5: Verify and run FLUX

1. Run focused AutoRound SVDQuant, exporter, diffusion, and CLI tests.
2. Run focused Nunchaku loader and MXFP4 tests.
3. Run a direct Diffusers component-loading fixture.
4. Update the runbook and residual-iters=2 CLI script.
5. Run the full FLUX export and direct pipeline image smoke test.
