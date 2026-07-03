# Sparse Prefetch Config Audit

Date: July 3, 2026  
Node: `b70`  
Shape: `B=1, Hq=40, Hkv=40, Sq=75000, Skv=75000, D=128, block=64, topk=0.5`

## Purpose

This note separates:

- the slow sparse-dev binary that was accidentally used for some recent comparisons
- the fast sparse-dev binary that should be treated as the current kernel-only reference

## Canonical Commands

Slow binary:

```bash
ZE_AFFINITY_MASK=4 ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
  /home/yiliu4/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/xbuild_sparse_dev/bench_ARK_XPU_sparse_dev \
  --topk 1.0,0.5 --warmup 1 --iters 3 \
  --batch 1 --heads-q 40 --heads-kv 40 \
  --seq-q 75000 --seq-kv 75000 --head-dim 128 --block-size 64
```

Fast binary:

```bash
ZE_AFFINITY_MASK=4 ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
  /home/yiliu4/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/xbuild_sparse_dev_prefetch/bench_ARK_XPU_sparse_dev \
  --topk 1.0,0.5 --warmup 1 --iters 3 \
  --batch 1 --heads-q 40 --heads-kv 40 \
  --seq-q 75000 --seq-kv 75000 --head-dim 128 --block-size 64
```

## Benchmark Result

| Build tree | Dense | Sparse `topk=1.0` | Sparse `topk=0.5` | Read |
| --- | ---: | ---: | ---: | --- |
| `xbuild_sparse_dev` | `755.495 ms` | `1736.831 ms` | `848.745 ms` | wrong baseline for fast sparse |
| `xbuild_sparse_dev_prefetch` | `755.437 ms` | `1248.092 ms` | `622.911 ms` | current fast sparse reference |

The key correction is:

- the `~849 ms` sparse result is not the intended fast sparse kernel baseline
- the corrected sparse-dev kernel-only baseline is `~623 ms`

## What To Trust

Valid as the current sparse-dev kernel-only baseline:

- `xbuild_sparse_dev_prefetch/bench_ARK_XPU_sparse_dev`
- unitrace captures collected from that binary in this audit folder
- VTune `gpu-offload` capture collected from that binary in this audit folder

Not valid as the fast sparse baseline:

- any sparse-dev report tied to the `~849 ms` `topk=0.5` result
- any sparse-dev report collected from `xbuild_sparse_dev` and then compared against the fast sparse path

Still directionally useful, but should be rechecked on the fast binary before using as a final conclusion:

- qualitative claims about sparse control overhead
- qualitative claims about spill pressure
- qualitative claims about memory-vs-execution balance

## Corrected Profiling Artifacts

Saved under:

`/home/yiliu4/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/profiles/prefetch_config_audit_20260703`

### unitrace metric-query

Files:

- `unitrace_fast_topk05_metric.1818321.txt`
- `unitrace_fast_topk05_metric.metrics.1818321.txt`

Observed benchmark:

- dense: `756.830 ms`
- sparse `topk=0.5`: `624.363 ms`

Useful kernel properties from the saved report:

- dense Sage kernel:
  - private memory per thread: `0`
  - spill memory per thread: `576`
  - register file size per thread: `256`
- sparse Sage kernel:
  - private memory per thread: `2048`
  - spill memory per thread: `640`
  - register file size per thread: `256`

Representative sparse-kernel metric-query rows:

- `GpuTime`: about `622.2-623.1 ms`
- `XVE_ACTIVE`: about `52.69-52.71%`
- `XVE_STALL`: about `47.06-47.09%`
- `XVE_THREADS_OCCUPANCY_ALL`: about `99.67%`
- `GPU_MEMORY_BYTE_READ_RATE`: about `9.75-9.83 GB/s`
- `GPU_MEMORY_BYTE_WRITE_RATE`: about `1.62-1.64 GB/s`

Read:

- occupancy is already high
- sparse is still spending nearly half of active GPU time stalled
- sparse still carries more private state and slightly more spill than dense

### unitrace stall-sampling

File:

- `unitrace_fast_topk05_stall.1818349.txt`

Observed benchmark:

- dense: `754.748 ms`
- sparse `topk=0.5`: `622.770 ms`

This run saved the timing log successfully. On this node and tool build, the standalone stall output did not emit a separate parsed stall-metric text file beyond the timing log path that unitrace reported during collection.

### VTune gpu-offload fallback

Result dir:

- `vtune_gpu_offload_fast_topk05`

Observed benchmark during collection:

- dense: `756.193 ms`
- sparse `topk=0.5`: `622.928 ms`

Summary:

- `GPU Time, % of Elapsed time`: `17.4%`
- target adapter used: `GPU 4`
- sparse compute task total time: `1.245 s` across `2` instances
- dense compute task total time: `1.495 s` across `2` instances
- recommendation text flagged `XVE Array Stalled/Idle: 37.9% of Elapsed time with GPU busy`

Read:

- VTune `gpu-offload` is good enough here to confirm the corrected fast sparse baseline
- it is not a replacement for in-kernel stall breakdown

## VTune Limitation On This Node

The attempted VTune `gpu-hotspots` collection is saved as:

- `vtune_gpu_hotspots_fast_topk05-bad`

The failure mode is documented in:

- `vtune_gpu_hotspots_fast_topk05-bad/log/perfrun-2026.07.03-01h49m27s.1812137.log`

Important lines:

- `Failed to connect to PMU reservation service (PAX)`
- `plugin [l0_metrics] was disabled`
- `plugin [stall_reasons] was disabled`

Meaning:

- VTune in-kernel GPU metrics are still not available on this node for us
- current VTune usage should be treated as:
  - `gpu-offload`: usable
  - `gpu-hotspots` / `stall-reasons`: blocked by PMU/PAX setup

## Actionable Conclusion

For the current sparse Sage work:

- use `xbuild_sparse_dev_prefetch/bench_ARK_XPU_sparse_dev` as the reference sparse-dev binary
- do not mix `xbuild_sparse_dev` timing with fast sparse conclusions
- use the saved unitrace metric-query data for kernel-level comparisons
- use VTune `gpu-offload` only as a workload/timeline sanity check until PMU/PAX access is fixed
