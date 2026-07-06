# Sparse Half-Width Full-Kernel Benchmark

## Goal

Check whether exposing the existing `ShapeOut=[64,64]` sparse row-linear full kernel as a normal benchmark mode improves the long-shape `topk=0.5` sparse path.

This is a structural probe for the question:

- does a smaller PV output tile reduce enough live state / spill pressure to help the real sparse kernel?

## Implementation

Added a real sparse-dev benchmark mode:

- `--sparse-mode sparse_row_linear_halfwidth`

This routes to the existing half-width launcher:

- `ShapeQK = [64, 64, 32]`
- `ShapePV = [64, 32, 64]`
- `ShapeOut = [64, 64]`
- `q_tile = 64`

The baseline mode remains:

- `--sparse-mode sparse_row_linear`
- `ShapeOut = [64, 128]`

## Build

```bash
cmake --build /home/yiliu4/workspace/auto-round/auto_round_extension/ark/xbuild_sparse_dev_prefetch \
      --target bench_ARK_XPU_sparse_dev -j 32
```

## Benchmark Commands

HND baseline:

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

HND half-width:

```bash
ZE_AFFINITY_MASK=4 \
/home/yiliu4/workspace/auto-round/auto_round_extension/ark/xbuild_sparse_dev_prefetch/bench_ARK_XPU_sparse_dev \
  --sparse-mode sparse_row_linear_halfwidth \
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

NHD baseline:

```bash
ZE_AFFINITY_MASK=4 \
/home/yiliu4/workspace/auto-round/auto_round_extension/ark/xbuild_sparse_dev_prefetch/bench_ARK_XPU_sparse_dev \
  --tensor-layout NHD \
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

NHD half-width:

```bash
ZE_AFFINITY_MASK=4 \
/home/yiliu4/workspace/auto-round/auto_round_extension/ark/xbuild_sparse_dev_prefetch/bench_ARK_XPU_sparse_dev \
  --tensor-layout NHD \
  --sparse-mode sparse_row_linear_halfwidth \
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

## Results

| Layout | Mode | Latency (ms) | Delta vs baseline |
| --- | --- | ---: | ---: |
| HND | `sparse_row_linear` | `557.075` | baseline |
| HND | `sparse_row_linear_halfwidth` | `653.126` | `+96.051 ms` |
| NHD | `sparse_row_linear` | `560.780` | baseline |
| NHD | `sparse_row_linear_halfwidth` | `712.816` | `+152.036 ms` |

## Conclusion

Reducing `ShapeOut` from `[64,128]` to `[64,64]` does not help the real sparse row-linear full kernel on this workload.

What it tells us:

- smaller PV output tiles can remove some isolated PV spill pressure
- but the full sparse kernel loses more from the extra PV/output slicing work than it gains from the smaller accumulator footprint
- the next useful direction is not "replace the kernel with half-width ShapeOut"
- the real target is still a structural sparse PV live-state reduction that preserves the full-width output path

In short:

- `half-width ShapeOut` is a useful diagnostic
- it is not a winning end-to-end sparse-kernel configuration for the current `topk=0.5`, `D=128`, long-shape workload
