# SYCL TLA Commit Benchmark Compare

This note compares the sparse `HND` benchmark across the original bad build, the older good upstream point, and the local no-LSE sparse epilogue fix.

## Builds

- `auto_round_kernel/xbuild-0630-fedbb`
  Bad upstream point based on `fedbba40`
- `auto_round_kernel/xbuild-25ace5-0406`
  Good upstream point based on `25ace5`
- `auto_round_kernel/xbuild-0630-fedbb-nolse`
  Same `fedbba40` upstream point, but with the sparse path rewired to a local no-LSE epilogue

## Scope

Common benchmark settings:

- layout: `HND`
- `q_tile_override=256`
- `sparse_q_block_tokens=256`
- `sparse_k_block_tokens=64`
- device: `ZE_AFFINITY_MASK=1`

Coverage:

- `xbuild-0630-fedbb` and `xbuild-25ace5-0406`
  Compared at `topk=0.5`, `0.3`, and `0.1`
- `xbuild-0630-fedbb-nolse`
  Spot-checked at the target case `topk=0.5`

## Loader Note

`test/bench_sparse_topk.py` now supports:

- `--xbuild-dir`
- `--xpu-so`

That means each benchmark run can target an explicit `.so` instead of silently reloading a default build.

## Baseline Results

### xbuild-0630-fedbb

- dense torch: `2110.954 ms`
- dense sage: `1213.869 ms`
- `topk=0.5`: kernel `1812.282 ms`, e2e `2015.860 ms`
- `topk=0.3`: kernel `1094.826 ms`, e2e `1296.630 ms`
- `topk=0.1`: kernel `376.031 ms`, e2e `576.689 ms`

CSV:

- [bench_sparse_topk_xbuild-0630-fedbb_hnd_gpu1.csv](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/bench_sparse_topk_xbuild-0630-fedbb_hnd_gpu1.csv)

### xbuild-25ace5-0406

- dense torch: `2111.008 ms`
- dense sage: `1218.095 ms`
- `topk=0.5`: kernel `637.953 ms`, e2e `841.896 ms`
- `topk=0.3`: kernel `394.873 ms`, e2e `598.060 ms`
- `topk=0.1`: kernel `145.836 ms`, e2e `348.595 ms`

CSV:

- [bench_sparse_topk_xbuild-25ace5-0406_hnd_gpu1_rerun.csv](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/bench_sparse_topk_xbuild-25ace5-0406_hnd_gpu1_rerun.csv)

## Post-Fix Spot Check

### xbuild-0630-fedbb-nolse

Target case:

- layout: `HND`
- `topk=0.5`
- `q_tile_override=256`
- `sparse_q_block_tokens=256`
- `sparse_k_block_tokens=64`

Result:

- dense torch: `2111.371 ms`
- dense sage: `1213.433 ms`
- sparse kernel-only: `643.092 ms`
- sparse e2e: `847.798 ms`

CSV:

- [bench_sparse_topk_xbuild-0630-fedbb-nolse_hnd_topk05_gpu1.csv](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/bench_sparse_topk_xbuild-0630-fedbb-nolse_hnd_topk05_gpu1.csv)

## Delta

For the target `topk=0.5` case:

- bad `0630`: kernel `1812.282 ms`, e2e `2015.860 ms`
- fixed `0630-nolse`: kernel `643.092 ms`, e2e `847.798 ms`
- good `25ace5`: kernel `637.953 ms`, e2e `841.896 ms`

Interpretation:

- the local no-LSE sparse epilogue recovers almost all of the lost sparse performance
- the fixed `fedbba40` build is now very close to `25ace5`
- the remaining gap is small enough that the original large regression is no longer the dominant issue

## Unitrace Spill Check

The main target for profiling was:

- layout: `HND`
- `topk=0.5`
- `q_tile_override=256`
- `sparse_q_block_tokens=256`
- `sparse_k_block_tokens=64`
- `warmup=0`
- `iters=1`

Spill summary:

| Build | Sparse Epilogue | Sparse Private Mem / Thread | Sparse Spill / Thread | Dense Spill / Thread | Sparse Register File / Thread |
|---|---|---:|---:|---:|---:|
| `xbuild-0630-fedbb` | `FMHAFwdEpilogue` | `2048 B` | `28992 B` | `128 B` | `256` |
| `xbuild-0630-fedbb-nolse` | `SparseFMHAFwdEpilogue` | `2048 B` | `640 B` | `128 B` | `256` |
| `xbuild-25ace5-0406` | old stateless shape | `2048 B` | `640 B` | `128 B` | `256` |

Interpretation:

- the bad `0630` build shows a severe sparse-only spill explosion
- after the no-LSE sparse epilogue fix, sparse spill returns to the same `640 B/thread` level as `25ace5`
- dense spill stays flat, so the effect is isolated to the sparse path

Relevant unitrace logs:

- [unitrace_0630_hnd_topk05_d0.1254552](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/unitrace_0630_hnd_topk05_d0.1254552)
- [unitrace_nolse_hnd_topk05.2005344](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/unitrace_nolse_hnd_topk05.2005344)
- [unitrace_25ace5_hnd_topk05_d0.1262743](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/unitrace_25ace5_hnd_topk05_d0.1262743)

## Lower-Level Check

To confirm that this was really an epilogue/codegen issue and not only a benchmark-level symptom, we also checked the compiled kernel one layer lower.

### Upstream Epilogue Shape

- `25ace5` epilogue is stateless:
  [xe_fmha_fwd_epilogue.hpp](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/xbuild-25ace5-0406/_deps/sycl_tla-src/applications/flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp:106)
- `fedbba40` epilogue adds `lse_ptr`, `seq_len_qo`, `num_heads_q`, and extra `head_q` / `idx_b` inputs for optional LSE writeback:
  [xe_fmha_fwd_epilogue.hpp](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/xbuild-0630-fedbb/_deps/sycl_tla-src/applications/flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp:106)
  [xe_fmha_fwd_epilogue.hpp](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/xbuild-0630-fedbb/_deps/sycl_tla-src/applications/flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp:156)
- local sparse fix restores a stateless sparse-only epilogue:
  [xe_sparse_fmha_fwd_epilogue.hpp](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/wrapper/include/stla/xe_sparse_fmha_fwd_epilogue.hpp:71)

### Embedded SPIR-V Check

We extracted the embedded SPIR-V bundle from each `.so` and compared the type names inside it.

| Build | SPIR-V Bundle Size | `SparseFMHAFwdEpilogue` Mentions | `FMHAFwdEpilogue` Mentions | `XeSparseSageFwdKernel` Mentions |
|---|---:|---:|---:|---:|
| `xbuild-0630-fedbb` | `11010120` | `0` | `756` | `148` |
| `xbuild-0630-fedbb-nolse` | `10946344` | `196` | `752` | `148` |

Interpretation:

- both builds still carry the same sparse kernel family
- only the fixed build contains sparse kernel instantiations that actually mention `SparseFMHAFwdEpilogue`
- the bad build keeps sparse tied to dense `FMHAFwdEpilogue`

That confirms the device image changed exactly in the direction suggested by the root-cause theory.

### Runtime Counter Check

After setting `/proc/sys/dev/xe/observation_paranoid=0`, the runtime metric path is usable on this machine.

- `unitrace -d -k -g ComputeBasic ...` now produces populated sampling output
- `unitrace -d -q -g ComputeBasic ...` also works and is easier to compare per sparse kernel call
- `-q` should be run sequentially; overlapping query-mode runs may fail with
  `Failed to create metric query pool (status = 0x78000004)`

Average `XeSparseSageFwdKernel` query metrics over the two sparse kernel calls:

| Build | Avg GPU Time | XVE Active | XVE Stall | Threads Occupancy | SEND Inst | LSC Read | LSC Write | GPU Mem Read | GPU Mem Write |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `xbuild-0630-fedbb` | `1.803 ms` | `34.8%` | `64.9%` | `99.5%` | `4.97e10` | `5.67e12 B` | `1.24e12 B` | `1.95e11 B` | `3.93e11 B` |
| `xbuild-0630-fedbb-nolse` | `0.634 ms` | `69.5%` | `30.0%` | `99.4%` | `6.43e9` | `3.21e12 B` | `7.24e9 B` | `1.55e10 B` | `8.30e8 B` |

Interpretation:

- occupancy stays flat, so this is not an occupancy-driven recovery
- the fixed kernel spends much less time stalled and much more time actively executing
- SEND count and memory traffic drop sharply, especially write traffic
- this matches the sparse spill collapse seen in the unitrace kernel-property report

Relevant metric artifacts:

- [metricq_sparse_fedbb_retry.metrics.2408427](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/metricq_sparse_fedbb_retry.metrics.2408427)
- [metricq_sparse_nolse_retry.metrics.2412421](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/metricq_sparse_nolse_retry.metrics.2412421)
- [metric_sparse_fedbb_retry.metrics.2399447](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/metric_sparse_fedbb_retry.metrics.2399447)
- [metric_sparse_nolse_retry.metrics.2403841](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/metric_sparse_nolse_retry.metrics.2403841)

So the strongest confirmed chain today is:

- epilogue source shape changed upstream
- sparse device image specialization changed
- sparse runtime counters moved in the exact direction expected from spill removal
- sparse spill exploded only in the bad build
- restoring the stateless sparse epilogue removed the spill explosion and recovered performance

## Conclusion

- `fedbba40` by itself is the bad-performance case for sparse `HND`
- the regression is explained by sparse-kernel codegen shape, not by dense behavior
- rewiring sparse to a no-LSE epilogue removes the spill explosion
- after the fix, `fedbba40` sparse performance is effectively back to the `25ace5` level for the target `topk=0.5` benchmark
