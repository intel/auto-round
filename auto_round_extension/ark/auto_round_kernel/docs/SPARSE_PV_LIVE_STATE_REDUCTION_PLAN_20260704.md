# Sparse PV Live-State Reduction Plan

## Goal

Reduce the sparse Sage PV-path live state in the current long-sequence `NHD` workload before doing any more prefetch work.

Primary target:

- layout: `NHD`
- shape: `B=1, Hq=40, Hkv=40, Sq=Skv=75600, D=128`
- sparse mode: `topk=0.5`
- sparse launcher: `Q=64`
- prefetch: `OFF`

This is the path that matters for the current video-generation workload and the current sparse row-linear kernel work.

## Why This Is The Right Focus

Existing profiling already narrows the problem down:

1. PV MMA is the first isolated mode that spills hard.

- In `PV_MICRO_MODE_RESULTS_20260703.md`:
  - `pv_reorder_only`: `spill/thread = 0`
  - `pv_load_v_only`: `spill/thread = 0`
  - `pv_mma_only`: `spill/thread = 7168`
- File: `auto_round_kernel/docs/PV_MICRO_MODE_RESULTS_20260703.md`

2. The full sparse kernel still carries more off-register state than dense.

- In `SPARSE_PREFETCH_CONFIG_AUDIT_20260703.md`:
  - dense: `private/thread = 0`, `spill/thread = 576`
  - sparse: `private/thread = 2048`, `spill/thread = 640`
- File: `auto_round_kernel/docs/SPARSE_PREFETCH_CONFIG_AUDIT_20260703.md`

3. Prefetch is not the next lever.

- Current sparse-dev A/B on the same focused kernel-only setup:
  - `prefetch OFF`: `557.723 ms`
  - `prefetch ON`: `617.068 ms`
- So the next pass should not chase sparse K prefetch first.

4. The code structure matches the profiling story.

- The PV body still keeps the following state overlapping in the hot loop:
  - `tArP` / `tArP_i8`
  - `tVrV` / `tVrV_i8`
  - `tArV` / `tArV_i8`
  - `tArA(_, _, _, VV)` / `tArA_v`
  - `tArAcc`
  - sparse row / block index state
  - rescale state on non-first blocks
- Main touch points:
  - `xe_sparse_sagev1_fwd_mainloop.hpp:780-918`
  - `xe_sparse_sagev1_fwd_mainloop.hpp:1050-1103`

## Working Hypothesis

The current sparse slowdown is mainly a live-range problem in the PV stage:

- sparse traversal metadata remains live too long
- PV accumulator / reorder state overlaps with that metadata
- non-first-block rescale logic extends the lifetime of output fragments
- the compiler pushes part of that state out of GRF into private / spill storage
- later PV work then waits on those dependencies, which shows up as `SBID`

The optimization target is therefore:

- reduce the amount of state live at the same time
- reduce the duration that it stays live
- keep the hottest PV loop body holding only the state needed for the current `VV`

## Non-Goals For This Pass

- do not tune sparse K prefetch
- do not tune occupancy
- do not rework the sparse pattern generation format
- do not switch the main target away from `NHD`

Those can be revisited after PV live-state reduction is measured.

## Code Areas To Change

Primary file:

- `auto_round_kernel/wrapper/include/stla/xe_sparse_sagev1_fwd_mainloop.hpp`

Hot regions:

1. PV-only helper path:

- `run_pv_only_math`
- roughly `780-918`

2. Full sparse PV path after softmax:

- roughly `1050-1103`

3. Similar duplicated PV code in the real-QK branch:

- roughly `1290-1333`

Secondary file for quick A/B:

- `auto_round_kernel/sdpa_sparse_dev.cpp`

This remains the fastest way to rebuild only the sparse benchmark path while iterating.

## Action Plan

### Phase 0: Lock The Baseline

Keep development pinned to:

- `Q=64`
- `topk=0.5`
- `NHD`
- `prefetch OFF`

Baseline kernel-only command:

```bash
ZE_AFFINITY_MASK=4 \
/home/yiliu4/workspace/auto-round/auto_round_extension/ark/xbuild_sparse_dev_prefetch/bench_ARK_XPU_sparse_dev \
  --sparse-mode sparse_row_linear \
  --topk 0.5 \
  --dtype bf16 \
  --warmup 1 \
  --iters 3 \
  --batch 1 \
  --heads-q 40 \
  --heads-kv 40 \
  --seq-q 75600 \
  --seq-kv 75600 \
  --head-dim 128 \
  --block-size 64 \
  --q-tile 64
```

Current reference:

- `557.723 ms`

Success is judged against this number first, then against profiling metrics.

### Phase 1: Shrink Lexical Scope Inside The PV `VV` Loop

Change:

- split the per-`VV` PV update into tiny helpers with minimal inputs
- keep `tVrV`, `tArV`, `tArAcc`, `tArA_v` inside the innermost scope only

Concrete edits:

1. Introduce dedicated helper bodies for one `VV` slice:

- one FP path helper
- one INT8 PV helper

2. Pass only the values needed for the current `VV`:

- current `VV`
- `k_idx`
- `first_block`
- current reordered `P`
- current output slice
- current scale base

3. Do not let helper bodies capture sparse traversal state they do not use.

Expected benefit:

- shorter live range for `tVrV*`, `tArV*`, `tArAcc`, `tArA_v`
- less overlap between V-load/reorder temporaries and output accumulators

Why this is worth doing first:

- it is low-risk
- it matches the profiling result that PV MMA is the first spill point
- it does not change the math

### Phase 2: Split First-Block And Rescaled-Block PV Paths

Today the non-first-block rescale logic is fused into the `VV` loop.

Change:

- create two separate PV bodies:
  - first-block path with no output rescale
  - later-block path with rescale

Concrete edits:

1. Replace:

- `if (!first_block) { ... rescale ... }`

with:

- a `first_block` specialized loop
- a `rescaled_update` specialized loop

2. In the first-block path:

- do not materialize or carry rescale-related state through the `VV` loop

3. In the later-block path:

- apply rescale as locally as possible to the current `VV` output slice

Expected benefit:

- shorter live range for `rescale`
- fewer loop-carried predicates
- less overlap between `rescale`, `tArA_v`, and `tArAcc`

### Phase 3: Detach Sparse Traversal Metadata From The PV Body

The sparse PV body should receive scalar indices, not broad traversal context.

Change:

- compute sparse block selection state before entering the PV helper
- pass only resolved scalar values into PV

Concrete edits:

1. Resolve before PV helper call:

- `k_idx`
- `scalev_block_base`
- `v_block_base`
- `subgroup_selected`

2. For the `Q=64` one-row path:

- keep row-local sparse cursor variables outside the PV helper
- do not let PV helpers depend on LUT/frontier arrays or row-iteration machinery

3. Keep any optional sparse-prefetch bookkeeping outside the PV helper as well.

Expected benefit:

- fewer sparse-control variables live during PV MMA
- less chance that sparse traversal state extends the lifetime of PV fragments

### Phase 4: Simplify The INT8 PV Scale / Writeback Path

The INT8 PV path currently recomputes per-element scale addressing inside the writeback loop.

Change:

- reduce integer/live-state pressure around:
  - `scalev_head_dim`
  - `v_block_base`
  - `scalev_block_base`
  - `scalev_idx`
  - `get<1>(tCrA(i))`

Concrete edits:

1. Hoist loop-invariant base values outside the inner element loop.

2. Precompute the per-`VV` scale base once.

3. If practical, replace repeated element-wise address reconstruction with a compact local map or helper that does not keep extra intermediate values live across the whole loop.

Expected benefit:

- less integer/control live state during writeback
- lower pressure when `tArAcc` and `tArA_v` are already live

Note:

- this is not the first change because the biggest signal still points to PV MMA footprint itself
- but it is a good second-order cleanup once Phase 1-3 are in place

### Phase 5: If Phase 1-4 Do Not Move The Needle, Use A Structural Fallback

Fallback, in order:

1. Keep QK once, split PV into smaller output slices per pass.

- this is the existing “single-QK, split-PV” direction
- the goal is to hold fewer output accumulators alive per pass

2. Add an experiment knob for smaller PV output work per pass.

- a development-only compile-time option is fine
- use sparse-dev benchmark first

3. Re-run the half-width probe only if simpler live-range cuts fail.

Reason:

- this is more intrusive
- it changes structure, not just scope
- it should be justified by failed low-risk passes first

## Measurement Gates After Each Phase

### Fast gate

Run the sparse-dev kernel-only benchmark command above.

Minimum acceptable outcome:

- no regression larger than `2%`

Good outcome:

- at least `3-5%` runtime improvement

### Profiling gate

Re-check:

- `private/thread`
- `spill/thread`
- `SBID`
- `XVE_STALL`
- `XVE_MULTIPLE_PIPE_ACTIVE`

Desired movement:

1. `pv_mma_only`

- `spill/thread` materially below `7168`

2. full sparse kernel

- `private/thread` below `2048`
- `spill/thread` at or below current `640`
- `SBID` / stall meaningfully down

3. if runtime improves but `private/thread` stays flat:

- keep the change if stall metrics also improve
- the goal is performance, not the private metric by itself

## Recommended Execution Order

1. Phase 1
2. Phase 2
3. Phase 3
4. benchmark
5. profile
6. Phase 4 if still needed
7. structural fallback only if the first four phases do not improve the kernel

## Stop / Go Criteria

### Stop and keep the change

If any of the following happen without correctness regression:

- kernel-only runtime improves by `>= 5%`
- `SBID` clearly drops
- `private/thread` drops materially
- `pv_mma_only spill/thread` drops materially

### Stop and revert the change

If:

- runtime regresses by `> 2%`
- compile time explodes without profile benefit
- the change only moves `private/thread` cosmetically but does not help runtime or stalls

## Practical Recommendation

Start with the lowest-risk refactor:

- Phase 1 lexical-scope split
- Phase 2 first-block vs rescaled-block split

That is the highest-signal next step because it attacks the exact overlap the profiles are pointing at, without changing the algorithm yet.
