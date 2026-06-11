# Sparse Sage Prefill Partial-Sparsity E2E Plan

## Summary

The next implementation slice is limited to the **prefill** case where `seq_len_q == seq_len_kv`. The milestone is to prove that:

- the sparse Sage kernel handles **real partial sparsity**, not only the dense-equivalent all-blocks-selected case
- the public Python API `sage_sparse(...)` works end-to-end for prefill
- correctness is defined against a **dense masked attention** reference built from the selected sparse blocks

Corner-case expansion is intentionally deferred until this narrower path is green.

## Scope For This Slice

- prefill only
- `seq_len_q == seq_len_kv`
- non-causal only
- `Hq == Hkv`
- fp16 `V/O`
- `head_dim = 64` and `128`
- block-level partial sparsity using `lut + valid_block_num`

Deferred:

- decode
- `seq_len_q != seq_len_kv`
- causal sparse coverage
- GQA sparse coverage
- bf16 sparse e2e
- zero-row / malformed-metadata stress coverage

## Implementation Focus

1. Extend the XPU unit tests so sparse rows can select arbitrary KV block subsets.
2. Add partial-sparse prefill tests against a dense masked reference.
3. Add one repo-local Python smoke test using the public `sage_sparse(...)` API.
4. Run the new coverage under `source /opt/intel/oneapi/setvars.sh`.

## Correctness Model

- Sparse routing is block-level.
- If a KV block is selected for a given Q block, all tokens in that block remain visible.
- If it is not selected, all tokens in that block are masked in the dense reference.
- The dense reference uses the existing masked attention path, not a separate custom attention implementation.

## Acceptance Criteria

- Existing all-selected sparse tests continue to pass.
- New partial-sparse prefill tests pass for `D=64` and `D=128`.
- At least one test uses a **non-contiguous** KV block selection pattern.
- Python `sage_sparse(...)` prefill smoke test passes for partial sparsity under the oneAPI runtime environment.
