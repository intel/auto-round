# Sparse Phase Split Results (2026-07-03)

## Scope

This records the new sparse phase-isolation modes added to the row-linear
profiling path and the first `unitrace` result on the real long-sequence shape.

Goal:

- separate QK / softmax cost from PV-side cost
- check which side owns the spill-heavy compiled footprint
- give the next optimization pass a concrete target

## Shape

- `B=1`
- `S=75000`
- `Hq=40`
- `Hkv=40`
- `D=128`
- `layout=NHD`
- `topk=0.5`
- `q_tile_override=64`

## New Profiling Modes

- `softmax_only_synth`
  - skips real QK
  - fills synthetic scores
  - runs softmax path only
- `pv_only_realish`
  - skips real QK and softmax
  - fills deterministic positive probability-like values
  - runs PV path with more realistic access/accumulation than the old synthetic mode
- `qk_plus_pv_no_softmax`
  - runs real QK
  - skips softmax
  - clamps/scales QK scores into `[0, 1]`
  - feeds them directly into PV

## Command

```bash
OUTDIR=/home/yiliu4/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/profiles/phase_isolation_split_20260703
mkdir -p "${OUTDIR}"

ZE_AFFINITY_MASK=4 ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
/home/yiliu4/workspace/pti-gpu/tools/unitrace/install_local/bin/unitrace \
  --metric-query -g ComputeBasic \
  --include-kernels XeSparseSageFwdKernel \
  --output "${OUTDIR}/<mode>_metric" \
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
    --profile-mode <mode>
```

Modes used:

- `full`
- `qk_only`
- `qk_softmax_only`
- `softmax_only_synth`
- `pv_only_realish`
- `qk_plus_pv_no_softmax`

## Results

| Mode | GpuTime (ns) | XVE Active | XVE Stall | Occupancy | Multi-pipe | Read GB/s | Write GB/s | Private/thread | Spill/thread | GRF/thread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full` | 837,732,448 | 36.44% | 63.24% | 99.55% | 4.90% | 540.06 | 1.14 | 0 | 640 | 256 |
| `qk_only` | 297,935,000 | 33.80% | 65.82% | 99.48% | 2.95% | 88.46 | 0.61 | 0 | 0 | 256 |
| `qk_softmax_only` | 364,371,927 | 48.16% | 51.43% | 99.44% | 10.28% | 72.69 | 0.51 | 0 | 0 | 256 |
| `softmax_only_synth` | 840,770,312 | 45.66% | 53.41% | 99.02% | 2.98% | 1.20 | 0.19 | 0 | 0 | 256 |
| `pv_only_realish` | 8,191,775,180 | 9.17% | 90.70% | 99.74% | 0.06% | 190.48 | 237.13 | 4096 | 9600 | 256 |
| `qk_plus_pv_no_softmax` | 9,412,275,676 | 5.48% | 94.45% | 99.79% | 0.03% | 264.69 | 170.12 | 0 | 10880 | 256 |

## Raw Files

Folder:

`/home/yiliu4/workspace/auto-round/auto_round_extension/ark/auto_round_kernel/profiles/phase_isolation_split_20260703`

Examples:

- `full_metric.2275680`
- `full_metric.metrics.2275680`
- `qk_only_metric.2276008`
- `qk_only_metric.metrics.2276008`
- `qk_softmax_only_metric.2276277`
- `qk_softmax_only_metric.metrics.2276277`
- `softmax_only_synth_metric.2276793`
- `softmax_only_synth_metric.metrics.2276793`
- `pv_only_realish_metric.2277131`
- `pv_only_realish_metric.metrics.2277131`
- `qk_plus_pv_no_softmax_metric.2277878`
- `qk_plus_pv_no_softmax_metric.metrics.2277878`

## Main Conclusions

1. QK is not the spill source.

- `qk_only` compiles with `spill=0`
- `qk_softmax_only` also compiles with `spill=0`
- both are far smaller than `full`

2. Softmax by itself is not the spill source.

- `softmax_only_synth` also compiles with `spill=0`

3. The spill-heavy state is on the PV side.

- `pv_only_realish` jumps to `private=4096`, `spill=9600`
- `qk_plus_pv_no_softmax` jumps further to `spill=10880`

4. The worst runtime behavior now lines up with the PV path.

- `pv_only_realish` and `qk_plus_pv_no_softmax` are extremely stall-heavy
- `multi-pipe` activity collapses to almost zero
- occupancy stays high, so this is not a launch-occupancy problem

## Interpretation

The prior question was whether the spill came from:

- QK accumulation / MMA, or
- PV accumulation / sparse traversal state

This split is now decisive enough:

- QK-only and QK+softmax-only are spill-free
- PV-containing variants are the ones that explode in spill/private memory

So the next optimization pass should focus on the PV-side live state, especially:

- per-row / per-head accumulator lifetime
- temporary probability / reordered-score buffers used before PV MMA
- sparse traversal metadata kept live across the PV loop
- any extra state introduced by NHD handling in the PV path

## Caution

`pv_only_realish` and `qk_plus_pv_no_softmax` are profiling variants, not valid
attention math. Their value here is compile/runtime attribution, not numerical
correctness or end-to-end timing.
