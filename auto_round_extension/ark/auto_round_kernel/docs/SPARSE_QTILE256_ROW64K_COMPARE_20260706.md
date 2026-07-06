# Sparse qtile256/qblock256/kblock64 Comparison

## Goal

Record the measured delta between the older sparse implementations on `sparse-attn-clean` and the newer decoupled sparse-row path that uses:

- `q_tile = 256`
- `sparse_q_block_tokens = 256`
- `sparse_k_block_tokens = 64`

This note is meant as a result checkpoint, not an implementation plan.

## Benchmark Shape

Common shape:

- `B=1`
- `Hq=40`
- `Hkv=40`
- `S=75600`
- `D=128`
- `topk=0.5`
- `quant_block_size=64`

## Older Sparse Baselines

### 1. Older HND sparse path on the same Python benchmark

Measured with the previous HND sparse path (`q_tile=64`) on `bench_sparse_topk.py`:

- `dense_torch_sdpa`: `1293.127 ms`
- `dense_sagev1`: `789.985 ms`
- `sparse_kernel_only`: `696.911 ms`
- `sparse_e2e`: `788.595 ms`

This is the most direct baseline for the newer HND decoupled path.

### 2. Older sparse-dev kernel-only best number

Recorded earlier on `sparse-attn-clean` in `SPARSE_SAGE_KERNEL_STATUS.md`:

- `dense sagev1`: `755.776 ms`
- `sparse topk=0.5`: `543.436 ms`

This number is useful, but it comes from the focused sparse-dev harness rather than the Python end-to-end benchmark, so it should not be treated as strictly apples-to-apples with the numbers below.

## New Decoupled qtile256/qblock256/kblock64 Results

### NHD

Measured from the decoupled sparse-row path:

- `dense_torch_sdpa`: `1277.550 ms`
- `dense_sagev1`: `781.549 ms`
- `sparse_qtile256_row64k_kernel_only`: `482.575 ms`
- `sparse_qtile256_row64k_e2e`: `578.122 ms`

### HND

Measured from the same decoupled sparse-row path after enabling `HND`:

- `dense_torch_sdpa`: `1291.787 ms`
- `dense_sagev1`: `791.633 ms`
- `sparse_qtile256_row64k_kernel_only`: `413.202 ms`
- `sparse_qtile256_row64k_e2e`: `523.357 ms`

## Direct Delta vs Older HND Sparse Path

Comparing the old HND sparse path with the new HND decoupled path:

| Metric | Old HND sparse | New HND decoupled | Delta |
| --- | ---: | ---: | ---: |
| kernel-only | `696.911 ms` | `413.202 ms` | `-283.709 ms` |
| e2e | `788.595 ms` | `523.357 ms` | `-265.238 ms` |

Relative improvement:

- kernel-only: about `1.69x` faster
- e2e: about `1.51x` faster

## Takeaway

The decoupled sparse-row path is materially faster than the older sparse implementation on the same long-shape `topk=0.5` workload.

The strongest practical result so far is:

- `HND`
- `q_tile=256`
- `sparse_q_block_tokens=256`
- `sparse_k_block_tokens=64`

with:

- `413.202 ms` kernel-only
- `523.357 ms` e2e

That is the current reference point when comparing against the earlier sparse implementation line in `sparse-attn-clean`.
