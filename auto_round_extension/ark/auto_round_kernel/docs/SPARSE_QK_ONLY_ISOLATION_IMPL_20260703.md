# Sparse QK-Only Isolation Implementation (2026-07-03)

## What Changed

The profiling-only sparse row-linear path now splits `QkOnly` / `QkSoftmaxOnly`
 away from the PV path inside
`wrapper/include/stla/xe_sparse_sagev1_fwd_mainloop.hpp`.

The key structural change is:

- `QkOnly` and `QkSoftmaxOnly` now run a dedicated QK-only loop body
- the dedicated path does not enter the PV GEMM path
- the dedicated path does not use V prefetch/copy/reorder state
- the profile sink now writes a minimal proxy into `tArA(0)` instead of
  spreading through the full output accumulator

The public profiling interface is unchanged:

- `sage_sparse_row_linear_profile(..., profile_mode="qk_only")`
- `sage_sparse_row_linear_profile(..., profile_mode="qk_softmax_only")`

## Validation

Smoke runs completed on:

- `B=1`
- `Hq=40`
- `Hkv=40`
- `S=75000`
- `D=128`
- `topk=0.5`
- `layout=NHD`
- `q_tile_override=64`

Both completed successfully:

- `profile_mode=qk_only`
- `profile_mode=qk_softmax_only`

## Fresh Unitrace Results

Folder:

`/home/yiliu4/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/profiles/qk_isolation_impl_20260703`

Files:

- `full_metric.2108917`
- `full_metric.metrics.2108917`
- `qk_only_metric.2109683`
- `qk_only_metric.metrics.2109683`

## Kernel Properties

| Mode | Private/thread | Spill/thread | GRF/thread |
| --- | ---: | ---: | ---: |
| `full` | 0 | 640 | 256 |
| `qk_only` | 0 | 0 | 256 |

This is the main success criterion for the isolation pass:

- the old `qk_only` path compiled to the same `spill=640` footprint as `full`
- the new `qk_only` path compiles with `spill=0`

## ComputeBasic Metrics

| Mode | GpuTime (ns) | XVE Active | XVE Stall | Occupancy | Multi-pipe | Read GB/s | Write GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | 829,127,447 to 841,737,656 | 36.40% to 36.64% | 63.03% to 63.27% | 99.53% to 99.54% | 4.92% to 4.93% | 538.39 to 541.99 | 1.13 to 1.15 |
| `qk_only` | 299,007,968 to 299,513,437 | 33.62% to 33.67% | 65.86% to 66.00% | 99.38% to 99.48% | 2.93% to 2.94% | 87.17 to 88.69 | 0.61 |

## Interpretation

The isolation is now real:

- `qk_only` no longer matches `full` in compiled resource footprint
- `qk_only` no longer matches `full` in runtime or memory traffic
- the spill-heavy PV-side live state was successfully removed from the
  `qk_only` specialization

The remaining `qk_only` behavior is still stall-heavy, but that is now a QK /
 sparse traversal result rather than a side effect of PV state surviving in the
 compiled kernel.
