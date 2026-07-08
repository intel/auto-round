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
ZE_AFFINITY_MASK=1 \
python test/bench_sparse_topk.py \
  --batch 1 \
  --num-heads-q 40 \
  --num-heads-kv 40 \
  --seq-len 75600 \
  --head-dim 128 \
  --tensor-layout NHD \
  --topk 0.5 \
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

## Pick A Free GPU

Before running the benchmark, check that the target XPU is idle:

```bash
xpu-smi ps
xpu-smi dump -d -1 -m 18
```

What to look for:

- no `bench_sparse_topk.py` or other user compute process attached in `xpu-smi ps`
- flat, low memory on all devices in `xpu-smi dump`

On this node, `xpu-smi` compute-util counters were not available without extra privilege, so the practical check was:

- no active XPU process attached
- steady low memory footprint across devices

The validated rerun used `ZE_AFFINITY_MASK=6`.

## Sparse K-Prefetch A/B

Build with sparse `K` prefetch enabled:

```bash
python -m cmake -S auto_round_kernel -B auto_round_kernel/xbuild \
  -DARK_XPU=ON \
  -DARK_SYCL_TLA=ON \
  -DARK_UT=OFF \
  -DARK_RESCALE=OFF \
  -DARK_SPARSE_SAGE_ENABLE_K_PREFETCH=ON

python -m cmake --build auto_round_kernel/xbuild \
  --target auto_round_kernel_xpu \
  -j 4
```

Run `NHD` and `HND` on the chosen device:

```bash
ZE_AFFINITY_MASK=6 \
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
  --output-csv bench_sparse_topk_prefetch_on_qtile256_k64_nhd_gpu6.csv

ZE_AFFINITY_MASK=6 \
python test/bench_sparse_topk.py \
  --batch 1 \
  --num-heads-q 40 \
  --num-heads-kv 40 \
  --seq-len 75600 \
  --head-dim 128 \
  --tensor-layout HND \
  --topk 0.5 0.3 0.1 \
  --q-tile-override 256 \
  --sparse-q-block-tokens 256 \
  --sparse-k-block-tokens 64 \
  --warmup 2 \
  --iters 3 \
  --output-csv bench_sparse_topk_prefetch_on_qtile256_k64_hnd_gpu6.csv
```

Rebuild with sparse `K` prefetch disabled:

```bash
python -m cmake -S auto_round_kernel -B auto_round_kernel/xbuild \
  -DARK_XPU=ON \
  -DARK_SYCL_TLA=ON \
  -DARK_UT=OFF \
  -DARK_RESCALE=OFF \
  -DARK_SPARSE_SAGE_ENABLE_K_PREFETCH=OFF

python -m cmake --build auto_round_kernel/xbuild \
  --target auto_round_kernel_xpu \
  -j 4
```

Then rerun the same `NHD` and `HND` commands with:

```bash
--output-csv bench_sparse_topk_prefetch_off_qtile256_k64_nhd_gpu6.csv
--output-csv bench_sparse_topk_prefetch_off_qtile256_k64_hnd_gpu6.csv
```

## Validated Results

GPU6 rerun, same node, same shape, same `q_tile=256` / `k_block=64` kernel.

`NHD`, prefetch `ON` vs `OFF`:

- `topk=0.5`: kernel `796.465 -> 482.584 ms` (`39.4%` faster), e2e `916.359 -> 577.503 ms` (`37.0%` faster)
- `topk=0.3`: kernel `478.918 -> 291.787 ms` (`39.1%` faster), e2e `599.025 -> 387.981 ms` (`35.2%` faster)
- `topk=0.1`: kernel `161.504 -> 98.049 ms` (`39.3%` faster), e2e `282.359 -> 213.140 ms` (`24.5%` faster)

`HND`, prefetch `ON` vs `OFF`:

- `topk=0.5`: kernel `694.479 -> 409.535 ms` (`41.0%` faster), e2e `823.376 -> 517.995 ms` (`37.1%` faster)
- `topk=0.3`: kernel `417.675 -> 256.283 ms` (`38.6%` faster), e2e `544.377 -> 367.859 ms` (`32.4%` faster)
- `topk=0.1`: kernel `144.716 -> 89.758 ms` (`38.0%` faster), e2e `272.264 -> 216.839 ms` (`20.4%` faster)

Cross-device consistency against the earlier GPU4 run:

- `NHD`: GPU4 vs GPU6 delta stayed within `0.0-2.7%` for prefetch `ON`, and within `0.2-0.6%` for prefetch `OFF`
- `HND`: GPU4 vs GPU6 delta stayed within `1.0-4.2%` for prefetch `ON`, and within `0.0-0.3%` for prefetch `OFF`

Saved CSVs:

- `bench_sparse_topk_prefetch_on_qtile256_k64_nhd_gpu6.csv`
- `bench_sparse_topk_prefetch_on_qtile256_k64_hnd_gpu6.csv`
- `bench_sparse_topk_prefetch_off_qtile256_k64_nhd_gpu6.csv`
- `bench_sparse_topk_prefetch_off_qtile256_k64_hnd_gpu6.csv`

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
