# Sparse Fast vs Slow Reprofile

Date: July 3, 2026  
Node: `b70`  
GPU: `ZE_AFFINITY_MASK=4`  
Shape: `B=1, Hq=40, Hkv=40, Sq=75000, Skv=75000, D=128, block=64, topk=0.5`

## Scope

This note reprofiles the two sparse-dev binaries with the same commands and tools:

- slow binary:
  - `xbuild_sparse_dev/bench_ARK_XPU_sparse_dev`
- fast binary:
  - `xbuild_sparse_dev_prefetch/bench_ARK_XPU_sparse_dev`

The first parallel reruns from this session were discarded because both binaries were launched on `GPU 4` at the same time and inflated latency. Only the clean single-run results below are used.

## Benchmark Comparison

Clean kernel-only reruns:

| Binary | Dense | Sparse `topk=0.5` | Sparse vs dense |
| --- | ---: | ---: | ---: |
| slow | `755.468 ms` | `848.997 ms` | `1.124x` slower |
| fast | `755.207 ms` | `622.871 ms` | `1.212x` faster |

Direct sparse-to-sparse improvement:

- `848.997 ms -> 622.871 ms`
- absolute gain: `226.126 ms`
- relative speedup: `1.363x`
- latency reduction: about `26.6%`

## unitrace Metric-Query Comparison

Artifacts:

- slow:
  - `profiles/prefetch_reprofile_compare_20260703/unitrace_slow_topk05_metric.1853100.txt`
  - `profiles/prefetch_reprofile_compare_20260703/unitrace_slow_topk05_metric.metrics.1853100.txt`
- fast:
  - `profiles/prefetch_reprofile_compare_20260703/unitrace_fast_topk05_metric.1853964.txt`
  - `profiles/prefetch_reprofile_compare_20260703/unitrace_fast_topk05_metric.metrics.1853964.txt`

Representative sparse-kernel rows:

| Metric | slow | fast | Read |
| --- | ---: | ---: | --- |
| `GpuTime` | `848.4-848.9 ms` | `621.9-622.8 ms` | fast matches benchmark delta |
| `XVE_ACTIVE` | `43.01-43.04%` | `52.72-52.80%` | fast keeps more useful work in flight |
| `XVE_STALL` | `56.74-56.77%` | `47.00-47.04%` | fast cuts stall share by about `9.7` points |
| `XVE_THREADS_OCCUPANCY_ALL` | `99.65-99.66%` | `99.66-99.70%` | occupancy was already saturated in both |
| `XVE_MULTIPLE_PIPE_ACTIVE` | `4.58-4.59%` | `8.84-8.86%` | fast uses more execution pipeline overlap |
| `GPU_MEMORY_BYTE_READ_RATE` | `23.54-23.73 GB/s` | `8.44-9.18 GB/s` | fast path does much less memory traffic |
| `GPU_MEMORY_BYTE_WRITE_RATE` | `4.51-4.52 GB/s` | `1.61-1.61 GB/s` | same direction |

## Kernel Property Comparison

From the saved unitrace kernel-property sections:

Dense Sage kernel in both binaries:

- private memory per thread: `0`
- spill memory per thread: `576`
- register file size per thread: `256`

Sparse kernel, slow binary:

- private memory per thread: `1536`
- spill memory per thread: `2880`
- register file size per thread: `256`

Sparse kernel, fast binary:

- private memory per thread: `2048`
- spill memory per thread: `640`
- register file size per thread: `256`

Read:

- the fast sparse kernel carries slightly more private state than the slow one
- but it cuts spill memory per thread from `2880` down to `640`
- that spill reduction is the biggest structural difference visible in the profiler output

## VTune gpu-offload Comparison

Artifacts:

- slow:
  - `profiles/prefetch_reprofile_compare_20260703/vtune_gpu_offload_slow_topk05`
- fast:
  - `profiles/prefetch_reprofile_compare_20260703/vtune_gpu_offload_fast_topk05`

Observed sparse compute-task totals:

| VTune view | slow | fast |
| --- | ---: | ---: |
| sparse compute task total time | `1.696 s` | `1.246 s` |
| dense compute task total time | `1.494 s` | `1.495 s` |

VTune recommendation summary:

| VTune item | slow | fast |
| --- | ---: | ---: |
| `GPU Time, % of Elapsed time` | `19.3%` | `17.0%` |
| `XVE Array Stalled/Idle` while GPU busy | `77.2%` | `37.9%` |

Read:

- dense stayed effectively unchanged
- the sparse task improved substantially
- the biggest visible VTune delta is the stall/idle fraction while the GPU is busy:
  - `77.2% -> 37.9%`

## Main Conclusion

The fast sparse binary is genuinely the right baseline.

What changed in profiler-visible behavior:

- sparse kernel time dropped by about `226 ms`
- active execution rose from about `43%` to about `53%`
- stall share dropped from about `56.8%` to about `47.0%`
- spill memory per thread dropped from `2880` to `640`
- memory traffic rates dropped sharply

So the fast-path win is not just noise or benchmark drift. It lines up with:

- much lower spill pressure
- lower memory traffic
- better execution utilization

## Practical Takeaway

For future sparse-kernel work on this shape:

- use `xbuild_sparse_dev_prefetch/bench_ARK_XPU_sparse_dev` as the sparse-dev reference binary
- treat `xbuild_sparse_dev` as a separate slow-path build, not the current sparse baseline
- compare future optimizations against the fast binary first, because that is the real target state now
