# Sparse Query-Tile 256 Support

## Goal

Add an explicit `256`-token query-tile mode to the SPARGE preprocess path and keep it aligned with the existing sparse kernel `q_tile_override=256` path.

## Current Kernel Status

The generic sparse SYCL kernel already supports `q_tile_override=256` for `head_dim=128`.

Relevant code:

- `auto_round_kernel/sdpa.cpp`
- `auto_round_kernel/wrapper/include/sycl_tla_sdpa.hpp`

That kernel path uses:

- `ShapeQK = [256, 64, 32]`
- `ShapeOut = [256, 128]`

So this implementation does not add a new sparse kernel. It adds preprocess-side control and preprocess/kernel coupling.

## API Changes

Added public preprocess control:

- `sparge_preprocess_topk(..., query_tile_tokens: int | None = None)`
- `sparge_sage2_attn_meansim_topk_xpu(..., query_tile_tokens: int | None = None, q_tile_override: int = 0)`

Accepted `query_tile_tokens` values:

- `64`
- `128`
- `256`

It must be a positive multiple of `quant_block_size`.

## Coupling Rules

Default behavior stays unchanged when both are omitted:

- preprocess uses the old head-dim-based default query routing tile
- kernel uses the old default launch choice

When one side is set explicitly, the e2e helper now aligns the other side:

1. If only `q_tile_override` is set:

- preprocess uses `query_tile_tokens = q_tile_override`

2. If only `query_tile_tokens` is set:

- kernel uses `q_tile_override = query_tile_tokens`

3. If both are set:

- they must match
- mismatch raises a `ValueError`

This gives an aligned `256` mode with either of these calls:

```python
ark.sparge_sage2_attn_meansim_topk_xpu(..., q_tile_override=256)
```

or

```python
ark.sparge_sage2_attn_meansim_topk_xpu(..., query_tile_tokens=256)
```

## Metadata Contract

The kernel metadata contract is unchanged:

- `lut`: `[B, Hq, num_q_blocks, num_k_blocks]`
- `valid_block_num`: `[B, Hq, num_q_blocks]`

where `num_q_blocks = ceil(Sq / quant_block_size)`.

`query_tile_tokens=256` only changes preprocess routing granularity:

- query routing is computed on `256`-token tiles
- the result is expanded back to the `64`-token sparse-row grid before LUT generation

## Validation

Targeted coverage for this slice:

- preprocess context helper accepts explicit `query_tile_tokens=256`
- preprocess e2e replay covers explicit `query_tile_tokens=256`
- preprocess e2e replay covers `q_tile_override=256` auto-coupling into preprocess
- benchmark harness passes `q_tile_override` into preprocess metadata generation so kernel-only sparse benchmark uses aligned metadata

## Benchmark

Long-shape benchmark on `GPU 4`:

- layout: `NHD`
- shape: `B=1, Hq=40, Hkv=40, S=75600, D=128`
- `topk=0.5`
- `quant_block_size=64`
- `warmup=1`, `iters=3`

Command shape:

```bash
ZE_AFFINITY_MASK=4 \
PYTHONDONTWRITEBYTECODE=1 \
/home/yiliu4/workspace/auto-round/auto_round_extension/ark/.venv/bin/python \
  /home/yiliu4/workspace/auto-round/auto_round_extension/ark/test/bench_sparse_topk.py \
  --batch 1 \
  --num-heads-q 40 \
  --num-heads-kv 40 \
  --seq-len 75600 \
  --head-dim 128 \
  --topk 0.5 \
  --tensor-layout NHD \
  --warmup 1 \
  --iters 3 \
  --q-tile-override <0-or-256>
```

Results:

| Config | selected_ratio | sparse_kernel_only | sparse_e2e |
| --- | ---: | ---: | ---: |
| default `q_tile_override=0` | `0.501268` | `867.872 ms` | `953.648 ms` |
| explicit `q_tile_override=256` | `0.501691` | `1016.478 ms` | `1162.142 ms` |

Observed delta for the generic sparse path:

- kernel-only: `+148.606 ms`
- e2e: `+208.494 ms`

Interpretation:

- the explicit preprocess+kernel `256` mode works correctly
- on this real `NHD` long-shape `topk=0.5` case, the generic sparse `q_tile=256` path is slower than the default path
- the slight `selected_ratio` drift is expected because preprocess routing granularity changed from the old default to explicit `256`
