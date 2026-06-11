# Sparge Preprocess Port Plan

## Summary

Port the Sparge preprocess side as a new ARK/XPU prefill path that stays structurally aligned with the original `get_block_map_meansim_fuse_quant` implementation.

The first slice targets:

- meansim + top-k routing
- fused Q/K quantization in the preprocess path
- Triton XPU for steps that are Triton in the original implementation
- pure PyTorch/XPU fallback only when a Triton XPU step fails or is blocked
- existing ARK sparse kernel for the final attention execution

This slice remains:

- prefill only
- `seq_len_q == seq_len_kv`
- fp16/bf16 V path only
- `head_dim = 64` and `128`

## Current Status (2026-06-10)

### What is already implemented

- Public preprocess entrypoints exist on the ARK/XPU side:
  - `sparge_preprocess_topk(...)`
  - `sparge_sage2_attn_meansim_topk_xpu(...)`
- The current preprocess path is implemented in pure PyTorch/XPU in
  `auto_round_extension/ark/auto_round_kernel/__init__.py`.
- The preprocess path already produces kernel-ready outputs:
  - `lut`
  - `valid_block_num`
  - quantized int8 `Q` / `K`
  - `qscale` / `kscale`
  - debug metadata such as `block_map`, `raw_block_map`, `tile_block_map`, and sparsity stats
- The backend-dispatch scaffold exists in
  `auto_round_extension/ark/auto_round_kernel/sparge_preprocess_triton.py`.
- The public path is already wired into model-level smoke coverage so preprocess-generated sparse metadata is exercised end to end.

### What is verified

- The ARK sparse kernel already accepts preprocess-generated metadata through `sage_sparse(...)`.
- Wrapper-level preprocess tests exist for metadata generation and sparse replay.
- Model-level Qwen no-cache smoke coverage already includes a `sparse_preprocess` path.
- The sparse kernel side is sufficiently mature for preprocess integration; current work is focused on preprocess correctness rather than kernel bring-up.

### What is not complete yet

- Triton-XPU preprocess kernels are not implemented yet.
  - `auto` mode currently falls back internally to the torch/XPU implementation.
- This slice is still top-k only.
  - `cdfthreshd` routing is not implemented.
- Scope is still limited to:
  - prefill only
  - `seq_len_q == seq_len_kv`
  - `head_dim = 64` and `128`
  - fp16/bf16 `V`
- Some sparse patterns still require the existing kernel-compatibility expansion pass after preprocess routing.

### Current known blocker

The main open correctness issue is no longer just test coverage.

For the preprocess-generated path, the dense-equivalent configuration
`topk=1, smooth_k=False` currently produces incorrect model generation output, even though it should behave similarly to the all-selected dense reference.

The strongest current hypothesis is in the torch preprocess routing fill logic:

- forced-selected entries are added to the routing map before ranked top-k fill
- `_fill_block_map_torch(...)` then iterates over the global sorted ranks without accounting for entries that were already selected
- this can waste the effective top-k budget on already-selected blocks
- as a result, `topk=1` may fail to become fully selected

### Immediate next steps

1. Fix the routing fill logic so the top-k budget counts newly added coherent selections rather than raw rank position.
2. Update wrapper preprocess regression coverage for the exact failing regime:
   - `topk=1.0`
   - `smooth_k=False`
   - non-causal first, then broader layout/head-dim coverage
3. Replace the over-strong compatibility assertion with a row-local check that matches `_apply_sparse_kernel_compatibility(...)`.
4. Strengthen the Qwen smoke to compare the dense-equivalent preprocess path against the all-selected dense reference.
5. Resume Triton-XPU preprocess work only after the shared torch/XPU oracle path is correct again.

## Implementation Changes

1. Add a new ARK-side preprocess helper that reproduces the Sparge process outputs.
- Inputs: fp16/bf16 `q, k, v`, routing hyperparameters, causal flag, layout
- Outputs:
  - `lut`
  - `valid_block_num`
  - `q_int8`, `q_scale`
  - `k_int8`, `k_scale`
- Keep the existing `sage_sparse(...)` kernel contract unchanged.

2. Match the original Sparge preprocessing stages closely.
- pooled block mean computation for Q and K
- block coherence / mean-sim classification
- fused block quantization for Q and K
- pooled proxy attention score computation
- top-k row selection
- force-include rules:
  - incoherent K blocks included for all rows
  - incoherent Q blocks include all K blocks
  - optional attention sink block 0
  - causal block filtering after selection
- LUT delta encoding

3. Use Triton XPU first where the original uses Triton.
- target these steps for Triton XPU first:
  - block-map-to-LUT
  - pool + sim + fused quant
  - block-map fill
  - block-level causal mask helper
- if a Triton XPU step is blocked or incorrect, replace only that step with pure PyTorch/XPU
- keep the fallback internal; do not expose separate public modes

4. Keep the first routing slice top-k only.
- implement `topk`
- do not implement CDF routing in this slice
- do not add fp8-V support in this slice

5. Add a Python entrypoint that mirrors the Sparge preprocess+kernel behavior.
- input: fp16/bf16 `q, k, v`
- preprocess:
  - build block routing
  - quantize Q/K
- execute:
  - call existing `sage_sparse(...)`
- optional debug output may include:
  - `lut`
  - `valid_block_num`
  - sparsity statistics

## Test Plan

1. Preprocess correctness checks.
- validate pooled block tensor shapes
- validate coherence mask shape and semantics
- validate top-k block selection behavior
- validate `lut` delta encoding
- validate `valid_block_num`

2. Integrated sparse attention checks.
- feed preprocess-generated metadata into `sage_sparse(...)`
- compare against dense masked ARK reference for:
  - all-selected equivalent rows
  - partial sparse rows
  - causal sparse rows
- cover `D=64` and `D=128`

3. Python smoke coverage.
- add a prefill-only Python smoke test for the new preprocess+kernel path
- cover:
  - non-causal top-k sparse prefill
  - causal top-k sparse prefill

4. Model-level smoke coverage.
- reuse the current Qwen no-cache prefill-style test shape
- add one path that uses preprocess-generated sparse metadata instead of handcrafted block patterns
- acceptance:
  - all-selected mode still gives exact parity
  - preprocess-generated partial-sparse mode runs end-to-end without non-finite outputs

## Assumptions

- first slice is top-k only
- the port should remain structurally close to the original Sparge implementation
- Triton XPU is the first-choice backend where the original uses Triton
- pure PyTorch/XPU fallback is allowed per-step if needed
- current node does not support fp8-V, so V stays on the existing ARK fp16/bf16 sparse kernel path
- decode, KV-cache, `seq_len_q != seq_len_kv`, CDF routing, and fp8-V are deferred
