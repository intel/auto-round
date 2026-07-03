# Sparse Phase-Isolation Reprofile (2026-07-03)

## Scope

This rerun checks whether the existing sparse phase-isolation modes can identify
which phase is responsible for the spill/private-state pressure seen in the
row-linear sparse kernel path.

Important limitation:

- This is the profiling-only sparse path exposed via
  `scripts/profile_sparse_row_linear_kernel_only.py`.
- It is not the separate `xbuild_sparse_dev_prefetch/bench_ARK_XPU_sparse_dev`
  binary that measured the faster `~623 ms` sparse-dev kernel-only result.
- The goal here is phase attribution, not exact reproduction of the sparse-dev
  timing.

## Command

```bash
ZE_AFFINITY_MASK=4 ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
/home/yiliu4/workspace/pti-gpu/tools/unitrace/install_local/bin/unitrace \
  --metric-query -g ComputeBasic \
  --include-kernels XeSparseSageFwdKernel \
  --output <out_prefix> \
  /home/yiliu4/workspace/auto-round/auto_round_extension/ark/.venv/bin/python \
  /home/yiliu4/workspace/vllm-omni-fork/scripts/profile_sparse_row_linear_kernel_only.py \
    --topk 0.5 \
    --tensor-layout NHD \
    --warmup 1 \
    --iters 1 \
    --seq-len 75000 \
    --num-heads-q 40 \
    --num-heads-kv 40 \
    --head-dim 128 \
    --profile-mode <full|qk_only|qk_softmax_only|pv_only_synthetic>
```

## Shape

- `B=1`
- `Hq=40`
- `Hkv=40`
- `S=75000`
- `D=128`
- `layout=NHD`
- `topk=0.5`
- `q_tile_override=64`

## Results

| Mode | GpuTime (ns) | XVE Active | XVE Stall | Occupancy | Multi-pipe | Read GB/s | Write GB/s | Private/thread | Spill/thread | GRF/thread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | 828,776,718 to 838,681,093 | 36.46% to 36.65% | 63.03% to 63.12% | 99.45% to 99.55% | 4.93% to 4.94% | 538.63 to 539.93 | 1.13 to 1.15 | 0 | 640 | 256 |
| `qk_only` | 831,533,541 to 842,720,260 | 36.33% to 36.62% | 62.99% to 63.29% | 99.46% to 99.49% | 4.89% to 4.90% | 536.24 to 540.41 | 1.13 to 1.14 | 0 | 640 | 256 |
| `qk_softmax_only` | 824,519,687 to 841,703,125 | 36.52% to 36.62% | 62.99% to 63.11% | 99.47% to 99.50% | 4.92% to 4.97% | 539.38 to 541.56 | 1.13 to 1.15 | 0 | 640 | 256 |
| `pv_only_synthetic` | 827,243,333 to 842,628,541 | 36.49% to 36.68% | 63.01% to 63.20% | 99.55% to 99.56% | 4.91% to 4.94% | 537.78 to 541.09 | 1.13 to 1.15 | 0 | 640 | 256 |

## Raw Files

Folder:

`/home/yiliu4/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/profiles/phase_isolation_fastkernel_20260703`

Key files:

- `full_metric.1993310.txt`
- `full_metric.metrics.1993310.txt`
- `qk_only_metric.1997018.txt`
- `qk_only_metric.metrics.1997018.txt`
- `qk_softmax_only_metric.1997679.txt`
- `qk_softmax_only_metric.metrics.1997679.txt`
- `pv_only_metric.1998309.txt`
- `pv_only_metric.metrics.1998309.txt`

## What This Means

The current phase-isolation path does **not** separate the bottleneck cleanly.

All four modes show:

- nearly identical kernel time
- nearly identical `XVE_STALL`
- nearly identical `XVE_ACTIVE`
- nearly identical memory traffic rates
- identical kernel resource footprint: `private=0`, `spill=640`, `GRF=256`

That strongly suggests at least one of these is true:

1. The common sparse traversal/state machinery dominates all four modes.
2. The compile-time mode switches are not removing enough code/live state to
   materially change the generated kernel.
3. The current synthetic phase variants still preserve the same key live ranges
   and sparse metadata state, so the compiler keeps the same spill profile.

## Practical Conclusion

This rerun does **not** justify saying:

- "the spill comes mainly from QK", or
- "the spill comes mainly from PV"

based on the current phase-isolation path.

Instead, the actionable conclusion is:

- the current profiling variants are still too similar to isolate the source
- the next profiling step should reduce shared sparse traversal/state further
- if we want real attribution, we likely need more aggressive isolation:
  - phase-specialized kernels
  - phase-specialized compile units
  - or manual guard paths that also remove the corresponding sparse state/live
    variables rather than only bypassing the math

## Relation To Earlier Fast Sparse-Dev Result

This does **not** invalidate the earlier sparse-dev fast result:

- sparse-dev fast binary: sparse `~622.9 ms`
- this phase-isolation path: `~0.83 s`

They are different binaries / paths. The phase-isolation rerun only says that
within the current profiling-only row-linear path, the existing mode switches do
not isolate the stall/spill source well enough.
