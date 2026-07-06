# Sparse Dev Fast Build

## Goal

Shorten the sparse-kernel inner-loop rebuild when editing `wrapper/include/sycl_tla_sdpa.hpp` and related sparse mainloop headers.

The main Python extension target is too expensive for this loop because one sparse-kernel header touch invalidates:

- `sdpa.cpp`
- `ark.cpp`
- all generated `generated/sdpa/prefill_*`
- all generated `generated/sdpa/decode_*`

On this node, the full `auto_round_kernel_xpu` rebuild was still in the `sdpa.cpp` SYCL device compile after `11+ min`, so it is not the right iteration path for sparse kernel work.

## Recommended Inner Loop

Use the dedicated sparse-dev benchmark target with two compile-time restrictions:

- `ARK_SPARSE_DEV_FOCUS_128_BF16=ON`
  - instantiate only `D128 + BF16`
- `ARK_SPARSE_DEV_SPARSE_ONLY=ON`
  - drop dense launcher instantiations
  - skip the dense benchmark run

This is the right tradeoff now because dense reference numbers were already collected and current work is sparse-kernel-only tuning.

## Build Commands

Configure:

```bash
cmake -S /home/yiliu4/workspace/auto-round/auto_round_extension/ark/auto_round_kernel \
      -B /home/yiliu4/workspace/auto-round/auto_round_extension/ark/xbuild \
      -DARK_SPARSE_DEV_BENCH=ON \
      -DARK_SPARSE_DEV_FOCUS_128_BF16=ON \
      -DARK_SPARSE_DEV_SPARSE_ONLY=ON
```

Build:

```bash
cmake --build /home/yiliu4/workspace/auto-round/auto_round_extension/ark/xbuild \
      --target bench_ARK_XPU_sparse_dev -j 32
```

Typical header-refresh loop:

```bash
touch /home/yiliu4/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/wrapper/include/sycl_tla_sdpa.hpp
cmake --build /home/yiliu4/workspace/auto-round/auto_round_extension/ark/xbuild \
      --target bench_ARK_XPU_sparse_dev -j 32
```

## Timing Summary

The key metric is the same-header-touch incremental rebuild.

| Path | Config | Header-touch rebuild |
| --- | --- | ---: |
| full sparse-dev | baseline sparse-dev target | `5:44.81` |
| focused sparse-dev | `D128 + BF16`, dense+sparse | `2:25.24` |
| focused sparse-only sparse-dev | `D128 + BF16`, sparse-only | `2:07.57` |

Clean rebuilds of the sparse-dev target also improved:

| Path | Config | Build time |
| --- | --- | ---: |
| focused sparse-dev | `D128 + BF16`, dense+sparse | `2:22.28` |
| focused sparse-only sparse-dev | `D128 + BF16`, sparse-only | `2:03.63` |

## Practical Notes

- This fast path is only for the sparse-dev benchmark executable.
- It does not change the full Python extension build.
- The focused sparse-dev path is `BF16` only, so run the benchmark with `--dtype bf16`.
- The benchmark now accepts `--dtype f16|bf16` so the fast path is directly usable.

Example smoke run:

```bash
/home/yiliu4/workspace/auto-round/auto_round_extension/ark/xbuild/bench_ARK_XPU_sparse_dev \
  --preset wan_self \
  --dtype bf16 \
  --topk 0.5 \
  --warmup 1 \
  --iters 1 \
  --heads-q 4 \
  --heads-kv 4 \
  --seq-q 256 \
  --seq-kv 256 \
  --head-dim 128
```

Observed output:

```text
| wan_self | sparse_kernel_only | prefix | bf16 | 0.500 | 0.500000 | 2.000 | 0.076 | 1.779 | 0.889 |
```

## Current Recommendation

For sparse-kernel implementation and profiling work on this node:

- do not rebuild `auto_round_kernel_xpu` in the inner loop
- use `bench_ARK_XPU_sparse_dev`
- keep `ARK_SPARSE_DEV_FOCUS_128_BF16=ON`
- keep `ARK_SPARSE_DEV_SPARSE_ONLY=ON`
- run benchmark commands with `--dtype bf16`
