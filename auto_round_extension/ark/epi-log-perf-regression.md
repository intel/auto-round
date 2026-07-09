# Sparse Epilogue Perf Regression

This note explains why the sparse `HND` regression was traced to the epilogue choice, with extra background for readers who are not familiar with CUTE or SYCL-TLA template code.

## Short Answer

Yes, the sparse regression came from using the newer dense `FMHAFwdEpilogue` in a sparse kernel that does not need LSE output.

The key point is not "LSE is expensive at runtime." The key point is "the epilogue type changed the generated kernel shape." In CUTE-heavy code, changing one template type often changes register lifetimes, temporary fragments, kernel arguments, and inlining decisions even when the runtime branch is rarely taken.

## What "Epilogue" Means Here

For these kernels, the work is split into two large compile-time pieces:

- `CollectiveMainloop`
  This is the tile-by-tile attention math: load Q/K/V tiles, run MMA, accumulate partial results.
- `CollectiveEpilogue`
  This is the final step: normalize the accumulator, reorder fragments, and write the output tile to global memory.

In CUTE / CUTLASS style code, both pieces are C++ template types. The kernel is instantiated from those types, so changing either type is not a small local edit. It can produce a materially different device kernel after inlining and optimization.

## Why CUTE Makes This Sensitive

If you do not work in CUTE often, the important mental model is:

- shapes are types
- layouts are types
- copy strategies are types
- the mainloop is a type
- the epilogue is a type
- the final GPU kernel is built from those types

So when we say "the sparse path used the wrong epilogue," that does not only mean "it called the wrong helper function." It means the sparse kernel was instantiated with a different compile-time object graph.

That can affect:

- how many values stay live across the epilogue
- whether extra state has to be carried in `Params`
- whether an extra conditional path is present
- how many fragment temporaries the compiler keeps
- whether register pressure crosses the spill threshold

That last point is what hurt us.

## What Changed Upstream

Between the good upstream point `25ace5` and the bad upstream point `fedbba40`, `FMHAFwdEpilogue` changed shape:

- `25ace5`
  The epilogue is effectively stateless for our use case.
- `fedbba40`
  The epilogue carries LSE-related state such as `lse_ptr`, `seq_len_qo`, and `num_heads_q`, and includes logic to optionally write LSE.

That change is reasonable for dense attention, because dense kernels may want LSE output.

It is not a good fit for our sparse prefill path, because sparse Sage does not currently consume LSE output at all.

## Why the Sparse Path Regressed

Our sparse wrapper was still instantiating the newer dense epilogue type:

- semantically, sparse did not use LSE
- structurally, the kernel still carried the LSE-capable epilogue type

That matters because the compiler still had to compile the heavier epilogue form into the sparse kernel.

The sparse kernel was already close to the register-pressure limit. The extra epilogue state and code path pushed it over the edge, which caused a large spill increase.

## Why Dense Did Not Regress

Dense still uses `FMHAFwdEpilogue`, and dense stayed healthy.

That does not contradict the root-cause claim. It actually supports it:

- dense kernels have different resource balance
- dense kernels can afford this epilogue change
- sparse kernels are more register-sensitive in this configuration

So the issue is not "the new epilogue is always bad." The issue is "the new dense epilogue is bad for this sparse instantiation."

## Evidence

### Before Fix

Bad build: `xbuild-0630-fedbb`

- sparse kernel type used `FMHAFwdEpilogue`
- sparse spill: `28992 B/thread`
- dense spill: `128 B/thread`
- `topk=0.5` sparse kernel-only latency: about `1812 ms`
- `topk=0.5` sparse e2e latency: about `2016 ms`

### After Fix

Fixed build: `xbuild-0630-fedbb-nolse`

- sparse kernel type uses `SparseFMHAFwdEpilogue`
- sparse spill: `640 B/thread`
- dense spill: `128 B/thread`
- `topk=0.5` sparse kernel-only latency: `643.092 ms`
- `topk=0.5` sparse e2e latency: `847.798 ms`

### Unitrace Interpretation

The important signal is not only that performance improved. It is that the generated sparse kernel changed in the expected direction:

- the kernel name now shows `SparseFMHAFwdEpilogue`
- spill dropped from `28992` to `640 B/thread`
- dense stayed unchanged

That is exactly the pattern we would expect if the epilogue type was the root cause.

### Lower-Level Confirmation

We were also able to confirm this one layer below the top-down benchmark.

#### 1. Source-Level Shape Change

The older good upstream epilogue in `25ace5` is stateless:

- [auto_round_kernel/xbuild-25ace5-0406/_deps/sycl_tla-src/applications/flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/xbuild-25ace5-0406/_deps/sycl_tla-src/applications/flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp:106)
  `Arguments {}` and `Params {}` are empty, and `operator()` only takes `(O, tArA, tA_max, tA_sum, blk_qv, thr_id)`.

The bad upstream epilogue in `fedbba40` is no longer stateless:

- [auto_round_kernel/xbuild-0630-fedbb/_deps/sycl_tla-src/applications/flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/xbuild-0630-fedbb/_deps/sycl_tla-src/applications/flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp:106)
  `Arguments` / `Params` now carry `lse_ptr`, `seq_len_qo`, and `num_heads_q`.
- [auto_round_kernel/xbuild-0630-fedbb/_deps/sycl_tla-src/applications/flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/xbuild-0630-fedbb/_deps/sycl_tla-src/applications/flash_attention_v2/collective/xe_fmha_fwd_epilogue.hpp:156)
  `operator()` also grows `head_q` and `idx_b`, and the body contains optional LSE writeback logic.

Our local sparse-only epilogue restores the stateless shape:

- [auto_round_kernel/wrapper/include/stla/xe_sparse_fmha_fwd_epilogue.hpp](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/wrapper/include/stla/xe_sparse_fmha_fwd_epilogue.hpp:71)
  `Arguments {}` and `Params {}` are empty again.

That means the root-cause claim is not based only on timing. The compile-time kernel template really changed shape in the exact place we suspected.

#### 2. Device-Image Confirmation

We extracted the embedded SPIR-V device image from both `.so` files and compared them directly.

- bad `xbuild-0630-fedbb`: `/tmp/fedbb.bundle`, size `11010120`
- fixed `xbuild-0630-fedbb-nolse`: `/tmp/nolse.bundle`, size `10946344`
- delta: fixed build is `63776` bytes smaller

More importantly, the type names inside the embedded SPIR-V changed exactly as expected:

- bad bundle:
  `SparseFMHAFwdEpilogue` count `0`
  `FMHAFwdEpilogue` count `756`
  `XeSparseSageFwdKernel` count `148`
- fixed bundle:
  `SparseFMHAFwdEpilogue` count `196`
  `FMHAFwdEpilogue` count `752`
  `XeSparseSageFwdKernel` count `148`

Interpretation:

- both builds still contain the same sparse kernel family
- only the fixed build contains sparse-kernel instantiations that mention `SparseFMHAFwdEpilogue`
- the bad build contains sparse-kernel instantiations tied to dense `FMHAFwdEpilogue`

So the device image itself confirms that the sparse kernel specialization changed. This is stronger than a runtime name-only observation.

#### 3. Runtime Counter Confirmation

After setting `/proc/sys/dev/xe/observation_paranoid=0`, both `unitrace -d -k -g ComputeBasic ...` and
`unitrace -d -q -g ComputeBasic ...` became usable on this machine.

For root-cause comparison, the query mode is the cleaner signal because it gives per-kernel metrics directly.
One practical caveat is that `-q` should be run sequentially. If two query-mode runs overlap, one of them may fail with
`Failed to create metric query pool (status = 0x78000004)`.

Relevant artifacts:

- bad build query metrics:
  [metricq_sparse_fedbb_retry.metrics.2408427](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/metricq_sparse_fedbb_retry.metrics.2408427)
- fixed build query metrics:
  [metricq_sparse_nolse_retry.metrics.2412421](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/metricq_sparse_nolse_retry.metrics.2412421)
- bad build sampling metrics:
  [metric_sparse_fedbb_retry.metrics.2399447](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/metric_sparse_fedbb_retry.metrics.2399447)
- fixed build sampling metrics:
  [metric_sparse_nolse_retry.metrics.2403841](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/metric_sparse_nolse_retry.metrics.2403841)

Average `XeSparseSageFwdKernel` query metrics over the two sparse kernel calls:

| Build | Avg GPU Time | XVE Active | XVE Stall | Threads Occupancy | SEND Inst | LSC Read | LSC Write | GPU Mem Read | GPU Mem Write |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `xbuild-0630-fedbb` | `1.803 ms` | `34.8%` | `64.9%` | `99.5%` | `4.97e10` | `5.67e12 B` | `1.24e12 B` | `1.95e11 B` | `3.93e11 B` |
| `xbuild-0630-fedbb-nolse` | `0.634 ms` | `69.5%` | `30.0%` | `99.4%` | `6.43e9` | `3.21e12 B` | `7.24e9 B` | `1.55e10 B` | `8.30e8 B` |

Interpretation:

- occupancy stays essentially flat, so the improvement is not from launching a fundamentally different amount of work
- active execution nearly doubles, from `34.8%` to `69.5%`
- stall percentage drops by more than half, from `64.9%` to `30.0%`
- SEND instruction count drops by about `7.7x`
- write traffic collapses, especially `LOAD_STORE_CACHE_BYTE_WRITE` and `GPU_MEMORY_BYTE_WRITE`

That is exactly the direction expected from removing a spill-heavy sparse epilogue shape. It strengthens the conclusion that
the dense LSE-capable epilogue was inflating register pressure and forcing excessive spill traffic in the sparse kernel.

## What We Changed

We added a sparse-only stateless epilogue:

- [auto_round_kernel/wrapper/include/stla/xe_sparse_fmha_fwd_epilogue.hpp](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/wrapper/include/stla/xe_sparse_fmha_fwd_epilogue.hpp:1)

This epilogue intentionally keeps the older no-LSE codegen shape for sparse kernels.

Then we rewired the active sparse launch path to use it:

- [auto_round_kernel/wrapper/include/sycl_tla_sdpa_sparse.hpp](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/wrapper/include/sycl_tla_sdpa_sparse.hpp:979)
- [auto_round_kernel/wrapper/include/sycl_tla_sdpa_sparse.hpp](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/wrapper/include/sycl_tla_sdpa_sparse.hpp:995)
- [auto_round_kernel/wrapper/include/sycl_tla_sdpa_sparse.hpp](/home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/wrapper/include/sycl_tla_sdpa_sparse.hpp:1010)

Dense launch sites were left on the normal dense epilogue.

## Practical CUTE Takeaway

If you are reading template code like this, a useful rule is:

- if a type participates in the kernel template, treat it as part of the kernel ABI and code shape

Even if two epilogues look functionally similar at a high level, they can produce very different generated kernels.

For performance debugging in CUTE code, these are often more important than small runtime branches:

- extra fields in `Arguments` / `Params`
- extra fragment transforms
- optional output paths
- different shared-memory staging structure
- different copy helper types

## Bottom Line

The sparse regression was caused by using the dense LSE-capable epilogue type in the sparse path.

The fix was not to "optimize LSE." The fix was to restore the sparse path to a no-LSE epilogue type so the compiler could generate the lighter kernel again.
