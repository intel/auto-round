# Kernel Build / Benchmark BKC

This is the shortest known-good flow to compile the XPU sparse prefill kernel and run the kernel benchmark on this branch.

## Environment

```bash
cd /home/yiliu4/workspace/auto-round-prefill-clean-sparse-pr/auto_round_extension/ark
source /opt/intel/oneapi/setvars.sh
source /home/yiliu4/workspace/auto-round-py/.venv/bin/activate
```

## Build

Recommended build for the current sparse prefill path:

```bash
python -m cmake -S auto_round_kernel -B auto_round_kernel/xbuild \
  -DARK_XPU=ON \
  -DARK_SYCL_TLA=ON

python -m cmake --build auto_round_kernel/xbuild \
  --target auto_round_kernel_xpu \
  -j 4
```

## Sanity

```bash
python auto_round_kernel/wrapper/test/test_sage_sparse_prefill_e2e.py
python auto_round_kernel/wrapper/test/test_sparge_preprocess_topk_e2e.py
python -m pytest -q test/test_sparge_preprocess_helpers.py
```

## Benchmark

Recommended benchmark for the supported `q_tile=256` kernel:

```bash
ZE_AFFINITY_MASK=4 \
python test/bench_sparse_topk.py \
  --batch 1 \
  --num-heads-q 40 \
  --num-heads-kv 40 \
  --seq-len 75600 \
  --head-dim 128 \
  --tensor-layout NHD \
  --topk 0.5 0.3 0.1 \
  --q-tile-override 256 \
  --sparse-q-block-tokens 256 \
  --sparse-k-block-tokens 64 \
  --warmup 2 \
  --iters 3 \
  --output-csv bench_sparse_topk_bkc_qtile256_k64.csv
```

The script prints a summary table and writes the CSV to:

```bash
bench_sparse_topk_bkc_qtile256_k64.csv
```

## Optional `q_tile=64` Reference

```bash
ZE_AFFINITY_MASK=4 \
python test/bench_sparse_topk.py \
  --batch 1 \
  --num-heads-q 40 \
  --num-heads-kv 40 \
  --seq-len 75600 \
  --head-dim 128 \
  --tensor-layout NHD \
  --topk 0.5 0.3 0.1 \
  --q-tile-override 64 \
  --warmup 2 \
  --iters 3 \
  --output-csv bench_sparse_topk_bkc_qtile64.csv
```
