# Sparse Sage Kernel Fix Plan

## Summary

Fix the current sparse-kernel limitation for rows like `block 0 + later non-contiguous blocks` by introducing a dedicated sparse forward mainloop instead of continuing to patch sparse behavior inside the dense Sage mainloop.

The public sparse interfaces stay unchanged:
- `sage_sparse(...)`
- `sage_sparse_decode(...)`
- `lut + valid_block_num`

The change is internal:
- sparse execution moves to a new sparse-specific mainloop
- dense Sage remains on the existing mainloop
- Python-side compatibility densification is removed after kernel parity is restored

## Key Changes

### 1. Add a sparse-dedicated forward mainloop

- Create a new header, e.g. `xe_sparse_sagev1_fwd_mainloop.hpp`.
- Make sparse traversal first-class in that file rather than an optional branch inside the dense mainloop.
- Keep dense/cached baseline behavior in `xe_sagev1_fwd_mainloop.hpp`.

### 2. Make LUT traversal the source of truth

- Iterate sparse rows from `lut + valid_block_num` using explicit selected-block descriptors, not only `cur_block += lut_row[i]`.
- Carry per selected block:
  - logical KV block id
  - block ordinal in the sparse row
  - global column start/end
  - whether it overlaps the sequence tail
  - whether it belongs to cache or current KV in decode mode

### 3. Rewrite sparse masking logic

- Replace dense assumptions like `K == total_blk - 1` with sparse-aware conditions.
- Causal masking must use the selected block's real global column range.
- Remainder-K masking must use whether the selected block overlaps the true tail of the KV sequence.
- This must work for:
  - contiguous sparse prefixes
  - non-contiguous rows such as `[0, 3, 5]`
  - rows that do not include the final logical KV block
  - cached decode logical block numbering over `cache + current`

### 4. Add sparse-aware prefetch

- Implement K/V prefetch over the selected sparse block sequence.
- Do not reuse the dense contiguous `K + Stages` prefetch logic unchanged.
- Keep dense prefetch behavior isolated in the existing mainloop.

### 5. Rewire sparse launches and remove compatibility expansion

- Route sparse prefill and sparse decode launchers to the new sparse mainloop.
- Keep wrapper and API contracts unchanged.
- Remove `_apply_sparse_kernel_compatibility(...)` after sparse-kernel parity is restored.
- Update preprocess and Wan sparse paths to pass raw routing output through unchanged.
- Update status docs to remove the `force-included gap blocks` limitation once fixed.

## Test Plan

- Keep existing sparse tests green:
  - all-selected prefill
  - partial sparse prefill
  - causal sparse prefill
  - cached sparse decode
- Add explicit regression cases for the previously broken pattern family:
  - `[0, 3, 5]`
  - `[0, 2]`
  - `[1, 3, 5]`
  - causal and non-causal variants where applicable
- Compare sparse output against dense masked reference built from the exact selected block set.
- Re-run preprocess-integrated tests and verify no compatibility-added blocks are needed.
- Re-run model-level smoke tests:
  - Qwen sparse prefill
  - Qwen sparse decode
  - Wan sparse run

## Acceptance Criteria

- Raw preprocess routing works without compatibility densification.
- Sparse kernel matches dense masked reference for non-contiguous sparse rows.
- Existing all-selected parity remains exact.
- Wan sparse execution no longer emits the current compatibility warning for gap rows after block `0`.

## Assumptions

- This is an internal kernel refactor and correctness fix; public sparse APIs do not change.
- Dense Sage stays on the existing mainloop; sparse behavior moves to a new mainloop file.
- Prefill and decode sparse paths should share the same sparse mainloop design.
- The Python compatibility workaround is temporary and should be removed after the kernel fix lands.
