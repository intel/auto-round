# Sparse Sage Kernel Status

## Build-Config Audit

As of July 3, 2026 on `b70`, this file contains both:

- older measurements from exploratory sparse-dev builds
- corrected measurements from the fast prefetch-enabled sparse-dev binary

The important correction is:

- `xbuild_sparse_dev/bench_ARK_XPU_sparse_dev` is currently the slow path for the `75k / 40 heads / D128 / topk=0.5` kernel-only benchmark:
  - dense `755.495 ms`
  - sparse `topk=0.5` `848.745 ms`
- `xbuild_sparse_dev_prefetch/bench_ARK_XPU_sparse_dev` is the fast path that should be treated as the current reference:
  - dense `755.437 ms`
  - sparse `topk=0.5` `622.911 ms`

So:

- any profiling or conclusions tied to the slow `~849 ms` sparse-dev binary are not representative of the intended fast sparse kernel
- the corrected baseline for the sparse-dev kernel-only `topk=0.5` case is about `623 ms`

See `SPARSE_PREFETCH_CONFIG_AUDIT_20260703.md` for the exact commands, profiler outputs, and validity notes.

## Current Snapshot

Kernel:

- sparse prefill and cached decode paths exist:
  - `sage_sparse(...)`
  - `sage_sparse_decode(...)`
- kernel contract is stable:
  - `lut + valid_block_num + qscale + kscale + int8 Q/K + fp16/bf16 V`
- `xe_sparse_sagev1_fwd_mainloop.hpp` is currently a copy-paste dense baseline
- this restored performance versus the earlier sparse-native refactor
- sparse-aware `K` lookahead/prefetch now exists for LUT-driven sparse rows
- dense-style `K` prefetch still applies on the dense/all-selected path when `lut_row == nullptr`

Correctness:

- verified:
  - all-selected sparse = dense parity
  - partial sparse prefill = dense masked reference
  - causal sparse prefill = dense causal+masked reference
  - non-contiguous rows like `[0,3,5]`, `[0,2]`, `[1,3,5]`
  - cached sparse decode smoke
- model-level:
  - Qwen sparse prefill smoke passes
  - Qwen sparse decode smoke passes
  - Wan sparse smoke passes
  - Flux sparse integration exists

Preprocess:

- Triton-XPU is used for:
  - fused `pool + sim + quant`
  - `fill_block_map`
  - `block_map_to_lut`
- remaining routing/math is still partly torch-side
- current main preprocess cost is still routing/pooling related, not LUT conversion

Performance:

- long synthetic benchmark (`B=1, H=40, S=75600, D=128, topk=0.5`):
  - dense `sagev1`: `1172.91 ms`
  - sparse kernel-only: `591.08 ms`
  - sparse e2e: `787.31 ms`
- so:
  - sparse kernel-only: about `1.98x` faster than dense
  - sparse e2e: about `1.49x` faster than dense
- latest `75k / 40 heads / D128 / NHD / topk=0.5 / q_tile=128` kernel-only rerun on `b70` after adding sparse LUT `K` lookahead:
  - dense `sagev1`: `769.735 ms`
  - sparse kernel-only: `857.471 ms`
  - sparse e2e: `956.003 ms`
  - vs the earlier sparse kernel-only baseline (`769.870 ms`), the new sparse-aware `K` prefetch path regressed by about `87.6 ms`

Latest profile delta for the same `75k / 40 heads / D128 / NHD / topk=0.5 / q_tile=128` sparse-kernel-only case:

- timing:
  - previous sparse kernel average: `765.617 ms`
  - new sparse kernel average: `853.131 ms`
- stall mix moved in the expected direction for data wait, but not enough to offset the added work:
  - `SBID`: `52.94% -> 48.83%`
  - `ALUWR`: `33.34% -> 42.19%`
  - `BARRIER`: `14.95% -> 13.48%`
  - `XVE_ACTIVE`: `44.99% -> 46.98%`
  - `XVE_STALL`: `54.74% -> 52.70%`
- cache / traffic changed only modestly:
  - `L3 hit rate`: `84.38% -> 84.75%`
  - `LSC hit rate`: `82.54% -> 81.17%`
  - derived `gpu_mem_bandwidth_gbps`: `399.64 -> 344.35`
- current read:
  - the sparse LUT `K` lookahead reduced scoreboard wait somewhat
  - but the extra traversal / scheduling overhead increased downstream compute dependency cost
  - the bottleneck is still better described as sparse pipeline scheduling than raw bandwidth

Current failed prefetch design:

- implementation:
  - the current sparse LUT prefetch path uses `peek_sparse_block(...)` in `xe_sparse_sagev1_fwd_mainloop.hpp`
  - it copies live sparse traversal state:
    - `row_pos[]`
    - `row_cur_block[]`
  - it then replays the sparse merge order for `lookahead_steps` steps:
    - find the minimum current logical block across active sparse rows
    - treat that block as the next emitted sparse block
    - advance every row currently pointing at that block
    - repeat until the requested future step is reached
  - the returned logical block id is used by the sparse-path `K` prefetch
- why it failed:
  - it adds control-heavy speculative work directly into the hot loop
  - it pays extra temporary-array copy cost, per-row comparisons, and repeated hypothetical row advancement on every real sparse block
  - the cost grows with sparse rows per tile and requested lookahead depth
  - it lowers `SBID`, but not cheaply enough, so total kernel time regresses and the stall mix shifts into `ALUWR`
- practical conclusion:
  - this version should be treated as an experiment, not the new default sparse path
  - the current sparse `peek_sparse_block(...)` prefetch should be disabled unless a later revision proves a net kernel win

Lower-overhead replacement direction:

- design goal:
  - do not replay future sparse traversal state from scratch inside the hot loop
- preferred model:
  - maintain the merged sparse frontier incrementally as the real traversal advances
  - when the current `next_block` is emitted, update only the rows that matched it
  - compute and carry the next merged block forward once, instead of cloning state and replaying future steps
- if deeper lookahead is needed:
  - use a tiny ready queue or ring of future logical blocks
  - pop the current block
  - advance only the touched rows
  - push one newly discovered future block
  - avoid full speculative rescans for each lookahead distance
- why this is the right direction:
  - removes repeated temp-state copies
  - removes repeated hypothetical advancement of the same sparse rows
  - makes overhead proportional to actual sparse progression rather than speculative replay
  - fits the profile result: the sparse kernel can afford only very cheap scheduling logic on the hot path

Latest low-overhead follow-up:

- implementation:
  - the replay-style `peek_sparse_block(...)` was replaced with an incremental sparse-frontier pop model
  - separate `prefetch_pos[]` / `prefetch_cur_block[]` state is initialized once from the live sparse frontier
  - warmup prefetch advances only that prefetch frontier
  - each main sparse iteration pops one future block from the prefetch frontier and advances the live frontier once
  - this keeps the future-window state incrementally instead of cloning and replaying sparse traversal every iteration
- benchmark result on the same `75k / 40 heads / D128 / NHD / topk=0.5 / q_tile=128` case:
  - dense `sagev1`: `769.109 ms`
  - sparse kernel-only: `840.848 ms`
  - sparse e2e: `941.157 ms`
- comparison:
  - better than the replay-style sparse-prefetch attempt:
    - `857.471 ms -> 840.848 ms`
  - still worse than the no-sparse-prefetch baseline:
    - `769.870 ms -> 840.848 ms`
- current read:
  - the incremental frontier version is the right structural direction
  - but even this cheaper sparse-prefetch logic still adds more overhead than the latency it hides
- current recommendation remains:
    - keep sparse-prefetch disabled by default
    - use the incremental-frontier model only as the base if sparse lookahead is revisited later

Current implementation status after the latest sparse-control cleanup:

- sparse-K prefetch is now disabled by default in the kernel through an internal compile-time switch:
  - `ARK_SPARSE_SAGE_ENABLE_K_PREFETCH=0`
- the default sparse path no longer carries duplicate live/prefetch frontier state in the hot loop
- sparse row traversal now hoists:
  - per-row LUT base pointers
  - active-row compaction for rows with `valid_block_num > 0`
- sparse block search / advance now iterate only active rows instead of every potential row slot in the tile
- the dense MMA body and public sparse API contract are unchanged

Expected effect of this implementation:

- remove the extra scheduling work from the default sparse path before trying additional latency-hiding tricks
- shrink scalar sparse metadata churn in the `find -> advance -> consume` chain
- preserve the experimental sparse-prefetch path for future profiling, but keep it out of the default performance path

Sparse-dev kernel-only validation after this cleanup:

- command shape:
  - `B=1, Hq=40, Hkv=40, S=75000, D=128, block_size=64`
  - benchmark target:
    - `bench_ARK_XPU_sparse_dev`
  - device:
    - `ZE_AFFINITY_MASK=4`
- measured latencies:
  - dense `sagev1`: `755.776 ms`
  - sparse `topk=1.0`: `1118.766 ms`
  - sparse `topk=0.5`: `543.436 ms`
- immediate read:
  - the cleaned-up sparse default path is now clearly better than dense at `topk=0.5` in the sparse-dev kernel-only harness
  - `topk=1.0` remains slower than dense, which is expected because it still pays sparse metadata cost without reducing work

New sparse-kernel profile after the sparse-control cleanup:

- profile target:
  - benchmark:
    - `bench_ARK_XPU_sparse_dev`
  - shape:
    - `B=1, Hq=40, Hkv=40, Sq=75000, Skv=75000, D=128, block=64, topk=0.5`
  - device:
    - `ZE_AFFINITY_MASK=4`
  - outputs:
    - `/home/yiliu4/workspace/vllm-omni-fork/metrics_computebasic.metrics.562608.csv`
    - `/home/yiliu4/workspace/vllm-omni-fork/stalls_sampling.metrics.594035.csv`
    - `/home/yiliu4/workspace/vllm-omni-fork/profile_sparse_dev_topk05_control_cleanup_stalls/bench_ARK_XPU_s.867452.json`
- benchmark latencies during profiling:
  - sparse `topk=0.5`: about `543-545 ms`
- `ComputeBasic` aggregate for `SPARSESAGEV1FwdMainloop`:
  - sample count:
    - `251556`
  - `GpuTime` per kernel instance:
    - `13.317 us`
  - `XVE_ACTIVE`:
    - `44.46%`
  - `XVE_STALL`:
    - `55.25%`
  - `XVE_THREADS_OCCUPANCY_ALL`:
    - `99.56%`
  - `XVE_MULTIPLE_PIPE_ACTIVE`:
    - `5.66%`
  - `LSC hit rate`:
    - `79.83%`
  - `L3 hit rate`:
    - `85.38%`
- stall-sampling aggregate for `XeSparseSageFwdKernel`:
  - sample count:
    - `2351`
  - `SbidStall / Active`:
    - `84.73%`
  - `DistStall / Active`:
    - `29.05%`
  - `SyncStall / Active`:
    - `12.53%`
  - `InstrFetchStall / Active`:
    - `7.11%`
  - `PipeStall / Active`:
    - `4.81%`
  - `ControlStall / Active`:
    - `2.38%`
  - `SendStall / Active`:
    - effectively `0%`
- current read:
  - the cleaned-up sparse kernel still spends more time stalled than active even though occupancy is already near full
  - the dominant problem remains scoreboard dependency wait, not send-queue pressure
  - cache hit rates are reasonable, so the first optimization target should remain sparse metadata dependency chains and synchronization points
  - in practice this means:
    - keep reducing dependent LUT/frontier loads in the hot loop
    - increase distance between metadata loads and their consumers
    - avoid tiny sparse-path sync/barrier regions where correctness allows

Detailed low-level profile for the current `topk=0.5 / q_tile=128` sparse kernel on `b70`:

- profiled kernel:
  - `cutlass::fmha::kernel::XeSparseSageFwdKernel<...SPARSESAGEV1FwdMainloop... tuple<C<128>, C<128>> ...>[SIMD16 {1; 586; 40} {128; 1; 1}]`
  - dumped IGC shader matches:
    - `igc_dump_sparse_low_overhead/OCL_asm4732923733479b43_simd16_entry_0030.asm`
- top-level counters:
  - `xve_active`: `44.45%`
  - `xve_stall`: `55.26%`
  - `SBID`: `54.78%`
  - `ALUWR`: `41.41%`
  - `BARRIER`: `16.48%`
  - `INSTFETCH`: `10.71%`
  - `SENDWR`: effectively `0%`
  - `gpu_mem_bandwidth`: `371.61 GB/s`
  - `L3 hit rate`: `85.38%`
  - `LSC hit rate`: `79.83%`
- stall-sampling summary for `XeSparseSageFwdKernel`:
  - `SbidStall`: `60.25%`
  - `DistStall`: `20.65%`
  - `SyncStall`: `8.91%`
  - `InstrFetchStall`: `5.06%`
  - `PipeStall`: `3.42%`
  - `SendStall`: effectively `0%`
- strongest conclusion from counters:
  - this kernel is not primarily blocked on send-queue pressure or raw DRAM throughput
  - the dominant cost is dependency wait after memory/control operations, plus synchronization overhead

What the shader dump shows:

- the sparse-control region is heavy:
  - many scalar/global loads and scratch-backed loads/stores appear in the sparse metadata path:
    - `load.ugm.d32x1t.a64`
    - `load.ugm.d32x1t.a32.ca.ca`
    - `store.ugm.d32x1t.a32.wb.wb`
  - this region also contains repeated:
    - `sync.allrd`
    - `sync.nop`
  - and two full workgroup barriers:
    - `sync.bar // $2591`
    - `sync.bar // $4851`
- the MMA region is comparatively regular:
  - int8 score MMA and fp16 value MMA are issued as dense `dpas.8x8` blocks
  - representative issue windows:
    - `$1054-$1186`
    - `$2397-$2571`
  - these regions still show scoreboard waits, but they are much more structured than the sparse-control section

Current read on the biggest bottleneck:

- the current sparse kernel is bottlenecked more by sparse metadata/control scheduling than by the MMA math core itself
- specifically:
  - sparse row/block traversal produces many short-latency dependent scalar loads
  - those loads are followed quickly by consumers, so the scheduler spends time in `SBID`
  - the metadata path also introduces repeated `sync.allrd` / barrier points, which show up as elevated `BARRIER` and `SyncStall`
  - because `SENDWR` is near zero, the issue is not "too many sends in flight"; it is "dependent consumers arrive before the dataflow clears"

Optimization priority from this profile:

1. reduce sparse metadata/control traffic in the hot loop
   - fewer scalar LUT / scratch round-trips
   - fewer read-modify-write steps on scratch-backed temporary state
   - fewer repeated sparse-row frontier updates per emitted block
2. increase producer-consumer distance in the sparse metadata path
   - hoist address generation and next-block preparation earlier
   - overlap sparse metadata work with existing MMA windows where possible
3. remove or narrow synchronization windows
   - especially the sparse-path `sync.allrd` chains and any barrier that protects only tiny metadata exchanges
4. treat additional prefetch as secondary
   - current evidence does not support more aggressive sparse-prefetch as the first move
   - prefetch may help later, but only after the sparse-control path becomes cheaper

Flux block `v2` profile summary:

- dense block: `152.05 ms`
- sparse `topk=0.5`: `122.33 ms`
- sparse `topk=0.1`: `112.96 ms`
- `_triton_bmm_pool_sim_simmean_fuse_quant_xpu`:
  - about `21.3 ms` in both sparse traces
  - largest named Triton preprocess kernel
  - but not the only preprocess bottleneck

Benchmark harness:

- C++ kernel-only benchmark exists:
  - `wrapper/test/bench_sparse_kernel.cpp`
  - target: `bench_ARK_XPU`
- model-shaped presets:
  - `wan_self`
  - `flux_joint`
  - `flux_single`

Sparse prefill vs decode:

| Aspect | Sparse prefill | Sparse decode |
| --- | --- | --- |
| Public API | `sage_sparse(...)` | `sage_sparse_decode(...)` |
| C++ entrypoint | `sdpa_impl_qks8_sparse_pvhalf(...)` | `sdpa_impl_qks8_sparse_decode_pvhalf(...)` |
| Main kernel wrapper | `XeSparseSageFwdKernel` | `XeSparseSageFwdKernel` |
| Mainloop family | `SPARSESAGEV1FwdMainloop` | `SPARSESAGEV1FwdMainloop` with cached-KV path |
| Q shape contract | general prefill, typically `seq_len_q == seq_len_kv` | current v1 requires `seq_len_q == 1` |
| KV source | current `K/V` only | current `K/V` plus `K_cache/V_cache` |
| Sparse block-id meaning | `lut` rows index only the current KV sequence | `lut` rows index the logical concatenation `cache blocks + current blocks` |
| Scale contract | `qscale` and `kscale` over current Q/K blocks | `qscale` plus `kscale` over total visible KV blocks, including cache |
| Extra required inputs | `lut`, `valid_block_num` | `lut`, `valid_block_num`, `K_cache`, `V_cache`, `seq_len_kv_cache` |
| Typical call path | sparse prefill kernel launch | cached sparse decode kernel launch |
| Why separate from dense decode | sparse metadata must be consumed in-kernel | sparse metadata must also resolve cache-vs-current KV blocks |

What is still missing:

- deeper sparse-vs-dense pipeline alignment after the first sparse `K` lookahead pass
- more dense-vs-sparse hot-path alignment work
- fully Tritonized preprocess routing
- broader decode/GQA coverage
- fp8-V / sparse int8-PV support

## Summary

The ARK sparse Sage attention path is now in a good **prefill-ready** state for integrating Sparge-style preprocess output, and the preprocess side has partial Triton-XPU acceleration.

Current status:

- sparse prefill and cached-decode kernel paths exist and are callable through `sage_sparse(...)` and `sage_sparse_decode(...)`
- kernel contract is `lut + valid_block_num + qscale + kscale + int8 Q/K + fp16/bf16 V`
- sparse traversal is now implemented through a dedicated sparse Sage mainloop in `xe_sparse_sagev1_fwd_mainloop.hpp`
- causal masking is handled in-kernel
- Sparge-style preprocess exists on XPU through `sparge_preprocess_topk(...)` / `sparge_sage2_attn_meansim_topk_xpu(...)`
- the preprocess backend now uses Triton-XPU for:
  - fused `pool + sim + quant`
  - `fill_block_map`
  - `block_map_to_lut`
- remaining preprocess routing/math still uses torch tensor ops
- a dedicated C++ kernel-only benchmark executable now exists as `bench_ARK_XPU` for Wan- and Flux-shaped sparse-kernel tuning

This means the kernel is already usable as the ARK/XPU counterpart of the attention-execution stage of `spas_sage2_attn_meansim_topk_cuda`.

## Dedicated Sparse Mainloop Status

The sparse mainloop split is now partially implemented:

- `xe_sparse_sagev1_fwd_mainloop.hpp` is now a copy-paste baseline from `xe_sagev1_fwd_mainloop.hpp`, not the earlier inheritance-based sparse-native refactor
- this was done because the earlier sparse-native refactor regressed kernel performance by drifting too far from the dense hot path
- the current sparse file is intentionally dense-structured first, so later sparse optimization can start from a performance-stable baseline
- dense-style `K` prefetch is still guarded by `if (lut_row == nullptr)`, but the sparse LUT path now has its own first-pass sparse `K` lookahead/prefetch path

This means:

- performance came back after restoring the dense mainloop structure
- the first sparse `K` lookahead/prefetch pass is now in place
- the first sparse `K` lookahead/prefetch pass does reduce `SBID`, but it currently regresses end-to-end kernel time because `ALUWR` grows more than the latency win
- further sparse hot-path scheduling work is still a future optimization
- the current copy-paste sparse mainloop is the baseline that later sparse-prefetch work should optimize from

## What Is Verified

Kernel and API validation completed:

- all-selected sparse rows match dense Sage exactly
- partial-sparse prefill matches dense masked reference
- causal sparse prefill matches dense causal+masked reference
- non-contiguous sparse rows match dense masked reference directly, including patterns such as:
  - `[0, 3, 5]`
  - `[0, 2]`
  - `[1, 3, 5]`
  - `[0, 1, 4]`
- cached sparse decode coverage exists and the model-level all-selected cached path remains healthy
- Python `sage_sparse(...)` smoke coverage exists for:
  - all-selected
  - partial sparse
  - causal partial sparse

Model-level prefill-style smoke coverage completed:

- Qwen no-cache generation test uses the public sparse path
- `all_selected` mode gives exact dense parity for 30 steps
- original `partial_sparse` routing mode is preserved and runs end-to-end
- a preprocess-generated `sparse_preprocess` path is wired into the same smoke harness

Current caveat for preprocess integration:

- preprocess-generated partial sparsity is still an approximation path, so model-level `sparse_preprocess` output can diverge from dense full attention by design
- low-level and tensor-level preprocess replay against the same sparse metadata are validated
- Triton-XPU preprocess metadata matches the torch reference on discrete outputs; small scale-tensor differences remain at floating-point noise level only

Current caveat for the dedicated sparse-mainloop validation:

- the extension build is green
- Python/runtime sparse regressions are green
- the heavyweight `xbuild_ut_icpx --target test_ARK_XPU` rebuild can still stall or get resource-killed on the giant `sdpa.cpp` TU in this shell, so that unit binary is not the primary verification signal for this specific refactor pass
- the previous inheritance-based sparse-native refactor was intentionally replaced by the copy-paste baseline because its hot path underperformed on kernel benchmarks

Current caveat for the new C++ benchmark harness:

- the benchmark target reuses the generated SYCL-TLA SDPA sources, which still include a CUTLASS utility header that expects `oneapi/mkl/rng/device.hpp`
- this node currently does not have a full oneMKL install, so the repo now carries a small local compatibility shim at `compat/include/oneapi/mkl/rng/device.hpp`
- that shim is only present to satisfy the unused CUTLASS utility include during compile; the benchmarked ARK sparse kernel path does not depend on oneMKL RNG behavior

## What The Kernel Already Covers

- prefill only
- prefill with `seq_len_q == seq_len_kv`
- cached decode with `seq_len_q == 1`
- `head_dim = 64` and `128`
- non-causal and causal
- LUT-driven sparse K-block traversal
- int8 `QK`
- fp16/bf16 `V`
- dense-equivalent sparse routing, real partial sparsity, and raw non-contiguous sparse rows

## Current Preprocess Backend

The current preprocess stack is split into:

- Triton-XPU:
  - fused block pooling / similarity / int8 quantization
  - block-map fill
  - LUT generation
- torch tensor ops:
  - pooled `QK` routing score
  - softmax + sort
  - query-tile routing reduction / remap
  - some small mask/index glue

So the preprocess is no longer “torch-only”, but it is not yet fully Tritonized either.

## What Is Still Missing

The following are not complete yet:

- fully Tritonized preprocess routing
- broader sparse hot-path scheduling work beyond the first `K` lookahead implementation
- `seq_len_q != seq_len_kv` prefill
- fp8-V sparse path
- sparse int8 `PV`
- broader GQA-specific sparse validation
- more decode-side sparse coverage beyond the current cached path

## Performance Status

Current benchmark artifacts:

- `bench_sparse_topk_results_triton_xpu.csv`
- `bench_sparse_topk_results_triton_xpu_75600_w3_i5_v3.csv`

On `B=1, H=40, S=75600, D=128, topk=0.5`:

- `dense_torch_sdpa`: `2119.59 ms`
- `dense_sagev1`: `1172.91 ms`
- `sparse_kernel_only`: `591.08 ms`
- `sparse_e2e`: `787.31 ms`

So at this shape:

- sparse kernel-only is about `1.98x` faster than dense `sagev1`
- sparse end-to-end is about `1.49x` faster than dense `sagev1`

After porting Triton-XPU `fill_block_map` and `block_map_to_lut`, the dominant preprocess stage is now query-tile routing pooling rather than block-map construction.

## C++ Kernel Benchmark Harness

The repo now also has a dedicated C++ kernel-only benchmark executable:

- target: `bench_ARK_XPU`
- source: `wrapper/test/bench_sparse_kernel.cpp`

Current benchmark presets are shaped around the active diffusion integrations:

- `wan_self`
- `flux_joint`
- `flux_single`

This harness is intentionally:

- prefill-only
- kernel-only
- dense baseline = `sagev1`
- sparse mode = `sdpa_impl_qks8_sparse_pvhalf(...)` with metadata built outside the timed region

It supports both:

- top-k style block-count sweeps
- explicit sparse row patterns such as:
  - `all_selected`
  - `prefix`
  - `stride2`
  - `custom_02`
  - `custom_035`
  - `custom_135`

Smoke validation completed on the built executable:

- `wan_self` + `all_selected`
- `flux_single` + `prefix` + `topk=1.0,0.5`
- `flux_joint` + `custom_035`

Observed smoke outputs:

- `wan_self`, dense `sagev1`: `14723.477 ms`
- `wan_self`, sparse `all_selected`: `157.020 ms`
- `flux_single`, dense `sagev1`: `32.653 ms`
- `flux_single`, sparse `prefix`, `topk=1.0`: `76.785 ms`
- `flux_single`, sparse `prefix`, `topk=0.5`: `37.250 ms`
- `flux_joint`, sparse `custom_035`: `1.329 ms`

Current usage contract:

- `--preset` selects `wan_self`, `flux_joint`, or `flux_single`
- `--pattern` selects `all_selected`, `prefix`, `stride2`, `custom_02`, `custom_035`, or `custom_135`
- `--topk` accepts a comma-separated list such as `1.0,0.75,0.5`
- `--warmup` and `--iters` control measurement repeat count
- `--csv` writes machine-readable output
- raw overrides exist for `--batch`, `--heads-q`, `--heads-kv`, `--seq-q`, `--seq-kv`, `--head-dim`, and `--block-size`

Practical caveat from the current smoke runs:

- `wan_self` dense timing is suspiciously slow in the current single-shot smoke run and should be rechecked before using it as a tuning baseline
- the harness itself is working, and the sparse rows execute correctly on all three preset families

## Change Inventory

The sparse-attention bring-up touched these areas:

- public Python APIs
  - `sage_sparse(...)`
  - `sage_sparse_decode(...)`
  - `sparge_preprocess_topk(...)`
  - `sparge_sage2_attn_meansim_topk_xpu(...)`
  - `sparge_preprocess_topk_decode(...)`
  - `sparge_sage2_decode_meansim_topk_xpu(...)`
- pybind / C++ entrypoints
  - sparse prefill and sparse decode bindings in `ark.cpp`
  - sparse SDPA entrypoints in `sdpa.cpp`
  - declarations in `wrapper/include/sycl_tla_common.hpp`
- kernel launch plumbing
  - sparse routing fields (`lut`, `valid_block_num`, `num_q_blocks`, `num_k_blocks`) in `wrapper/include/sycl_tla_sdpa.hpp`
  - sparse launcher selection and config wiring for prefill and cached decode
- kernel implementation
  - sparse row mapping and scale indexing in `wrapper/include/stla/xe_sage_fwd_kernel.hpp`
  - sparse traversal support in `wrapper/include/stla/xe_sagev1_fwd_mainloop.hpp`
  - dedicated sparse-native mainloop implementation in `wrapper/include/stla/xe_sparse_sagev1_fwd_mainloop.hpp`
- preprocess implementation
  - torch reference preprocess in `__init__.py`
  - Triton-XPU preprocess backend in `sparge_preprocess_triton.py`
  - Triton-XPU ports for fused `pool + sim + quant`, `fill_block_map`, and `block_map_to_lut`
- model / application integration
  - Qwen prefill and decode smoke tests
  - Wan self-attention sparse patch in `wan_sparse_patch.py`
  - Wan runner integration in `run_wan.py`
  - Wan top-k sweep helper in `run_spa.sh`
- validation and benchmarking
  - sparse kernel unit coverage in `wrapper/test/test_sdpa.hpp`
  - preprocess e2e checks in `wrapper/test/test_sparge_preprocess_topk_e2e.py`
  - decode preprocess smoke in `wrapper/test/test_sparge_decode_topk_e2e.py`
  - model-level Qwen tests in `../test/test_qwen_sparse_prefill_e2e.py` and `../test/test_qwen_sparse_decode_e2e.py`
  - top-k benchmark harness in `../test/bench_sparse_topk.py`
  - C++ kernel-only benchmark harness in `wrapper/test/bench_sparse_kernel.cpp`
- docs
  - implementation plans, benchmark runbook, Wan note, and this status note under `docs/`

## Practical Conclusion

The kernel side is usable for raw Sparge-style sparse metadata without the old compatibility densification step.

The preprocess side is now good enough for meaningful end-to-end benchmarking, and the next optimization milestone should focus on the remaining routing-heavy torch stages:

- query-tile routing pooling
- pooled routing score / select path
- any remaining torch-side routing glue that prevents preprocess cost from scaling down with sparsity

Those outputs already feed the current `sage_sparse(...)` path directly without changing the sparse kernel contract.

## Build And Test Commands

Commands used for the latest dedicated sparse-mainloop validation:

Build:

```bash
source /opt/intel/oneapi/setvars.sh
/home/yiliu7/workspace/venvs/ark/bin/cmake --build xbuild -j 4
```

Optional heavier unit-target rebuild:

```bash
source /opt/intel/oneapi/setvars.sh
/home/yiliu7/workspace/venvs/ark/bin/cmake --build xbuild_ut_icpx --target test_ARK_XPU -j 1
```

Build the dedicated C++ kernel benchmark:

```bash
source /opt/intel/oneapi/setvars.sh
/home/yiliu7/workspace/venvs/ark/bin/cmake -B xbuild -DARK_BENCH=ON
/home/yiliu7/workspace/venvs/ark/bin/cmake --build xbuild --target bench_ARK_XPU -j 4
```

Sparse runtime regression checks:

```bash
source /opt/intel/oneapi/setvars.sh
/home/yiliu7/workspace/venvs/ark/bin/python wrapper/test/test_sparge_preprocess_topk_e2e.py
```

```bash
source /opt/intel/oneapi/setvars.sh
/home/yiliu7/workspace/venvs/ark/bin/python wrapper/test/test_sage_sparse_prefill_e2e.py
```

```bash
source /opt/intel/oneapi/setvars.sh
/home/yiliu7/workspace/venvs/ark/bin/python -m pytest -q ../test/test_qwen_sparse_prefill_e2e.py -s
```

C++ kernel-benchmark smoke checks:

```bash
source /opt/intel/oneapi/setvars.sh
./xbuild/bench_ARK_XPU --preset flux_single --pattern prefix --topk 1.0,0.5 --warmup 1 --iters 1
```

```bash
source /opt/intel/oneapi/setvars.sh
./xbuild/bench_ARK_XPU --preset flux_joint --pattern custom_035 --warmup 1 --iters 1
```

```bash
source /opt/intel/oneapi/setvars.sh
./xbuild/bench_ARK_XPU --preset wan_self --pattern all_selected --warmup 0 --iters 1
```

Interpretation:

- `xbuild` shared-module build is the primary compile check for this refactor
- the Python sparse replay and Qwen prefill tests are the primary runtime checks
- `test_ARK_XPU` remains useful, but its rebuild path may be limited by local compile-resource pressure on `sdpa.cpp`
