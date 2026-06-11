# Sparse Sage Causal Prefill Plan

## Summary

The next implementation slice adds **causal sparse attention support first** for the existing **prefill-only** sparse Sage path, where `seq_len_q == seq_len_kv`.

This slice keeps the current sparse metadata contract unchanged:

- `sage_sparse(...)` still consumes `lut + valid_block_num`
- callers are not required to causally prune sparse rows
- the kernel remains responsible for masking future-invalid sparse selections when `is_causal=True`

The milestone is complete when:

- sparse causal prefill matches a dense causal+masked reference in XPU unit tests
- the public Python `sage_sparse(...)` path has a causal prefill smoke test
- the docs explicitly describe the causal contract and the current scope

## Scope For This Slice

- prefill only
- `seq_len_q == seq_len_kv`
- causal support added first
- block-level sparse routing using `lut + valid_block_num`
- XPU unit coverage
- Python `sage_sparse(...)` smoke coverage

Deferred:

- decode / KV-cache sparse causal
- `seq_len_q != seq_len_kv`
- GQA sparse causal expansion
- bf16 sparse causal e2e
- malformed-metadata and zero-row stress coverage

## Implementation Focus

1. Keep the public sparse API unchanged and use `is_causal=True`.
2. Make the dense reference causal by combining sparse block visibility with token-level causal masking.
3. Add causal sparse-prefill unit tests for `head_dim=64` and `128`.
4. Add at least one causal test where `lut` includes future KV blocks, proving the kernel mask is authoritative.
5. Add Python smoke coverage for causal prefill under `source /opt/intel/oneapi/setvars.sh`.

## Correctness Model

- Sparse routing remains block-level.
- Selected KV blocks may still be future-invalid for some query rows.
- When `is_causal=True`, the kernel must suppress those future positions even if the sparse metadata includes them.
- The dense reference is built by:
  - enabling visibility only for selected sparse KV blocks
  - then applying token-level causal masking inside those visible blocks

## Acceptance Criteria

- Existing non-causal sparse tests continue to pass.
- New causal sparse-prefill tests pass for `D=64` and `D=128`.
- New Python `sage_sparse(...)` causal prefill smoke test passes.
- At least one causal case includes future KV blocks in `lut` and still matches the dense causal reference.
