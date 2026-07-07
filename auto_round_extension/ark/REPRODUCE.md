# Sparse Prefill Clean Branch

This branch keeps the sparse prefill runtime on top of `main` with two supported query-tile modes:

- `q_tile=64` via the generic sparse prefill path
- `q_tile=256` with `sparse_q_block_tokens=256` and `sparse_k_block_tokens=64`

## Build

```bash
cd auto_round_extension/ark
source /opt/intel/oneapi/setvars.sh
python -m cmake -S auto_round_kernel -B auto_round_kernel/xbuild -DARK_XPU=ON -DARK_SYCL_TLA=ON
python -m cmake --build auto_round_kernel/xbuild --target auto_round_kernel_xpu -j 4
```

## Tests

```bash
cd auto_round_extension/ark
python \
  auto_round_kernel/wrapper/test/test_sage_sparse_prefill_e2e.py
python \
  auto_round_kernel/wrapper/test/test_sparge_preprocess_topk_e2e.py
python -m pytest -q \
  test/test_sparge_preprocess_helpers.py
```

## Benchmark

`q_tile=64`:

```bash
ZE_AFFINITY_MASK=4 \
python \
  test/bench_sparse_topk.py \
  --batch 1 --num-heads-q 40 --num-heads-kv 40 --seq-len 75600 --head-dim 128 \
  --tensor-layout NHD --topk 0.5 0.3 --q-tile-override 64
```

`q_tile=256`, decoupled sparse rows:

```bash
ZE_AFFINITY_MASK=4 \
python \
  test/bench_sparse_topk.py \
  --batch 1 --num-heads-q 40 --num-heads-kv 40 --seq-len 75600 --head-dim 128 \
  --tensor-layout NHD --topk 0.5 0.3 \
  --q-tile-override 256 \
  --sparse-q-block-tokens 256 \
  --sparse-k-block-tokens 64
```

## Wan Example

`q_tile=64`:

```bash
cd auto_round_extension/ark/examples
WAN_USE_SPARSE=1 \
WAN_SPARSE_TOPK=0.5 \
WAN_SPARSE_Q_TILE_OVERRIDE=64 \
python run_wan.py
```

`q_tile=256`:

```bash
cd auto_round_extension/ark/examples
WAN_USE_SPARSE=1 \
WAN_SPARSE_TOPK=0.5 \
WAN_SPARSE_Q_TILE_OVERRIDE=256 \
WAN_SPARSE_Q_BLOCK_TOKENS=256 \
WAN_SPARSE_K_BLOCK_TOKENS=64 \
python run_wan.py
```

## Flux Example

`q_tile=64`:

```bash
cd auto_round_extension/ark/examples
FLUX_USE_SPARSE=1 \
FLUX_SPARSE_TOPK=0.5 \
FLUX_SPARSE_Q_TILE_OVERRIDE=64 \
python run_flux.py
```

`q_tile=256`:

```bash
cd auto_round_extension/ark/examples
FLUX_USE_SPARSE=1 \
FLUX_SPARSE_TOPK=0.5 \
FLUX_SPARSE_Q_TILE_OVERRIDE=256 \
FLUX_SPARSE_Q_BLOCK_TOKENS=256 \
FLUX_SPARSE_K_BLOCK_TOKENS=64 \
python run_flux.py
```

## Flux Sweep

Dense baseline plus sparse `topk=1.0..0.1`, using 8 devices and the `q_tile=256`,
`sparse_q_block_tokens=256`, `sparse_k_block_tokens=64` kernel:

```bash
cd auto_round_extension/ark/examples
FLUX_SWEEP_PYTHON=/home/yiliu4/workspace/auto-round-py/.venv/bin/python \
FLUX_MODEL=/home/yiliu4/workspace/models/black-forest-labs/FLUX.1-dev \
FLUX_SWEEP_DEVICES=0,1,2,3,4,5,6,7 \
bash run_flux_sweep.sh
```
