# K32 qtile256 BF16/FP16 Sparse SDPA Sweep — 2026-08-14

## Setup

- Branch: `feat/sparse-bf16-prefill-v2`
- Device: Intel Battlemage XPU, `ZE_AFFINITY_MASK=5`
- Kernel shape: BF16/FP16 sparse SDPA selected-block K32, qtile256
- Benchmark: `benchmarks/bench_sparse_topk.py`
- Shape: `seq_len=75000`, 40 query heads, 40 KV heads, head_dim 128
- Layouts: HND, NHD
- Dtypes: BF16, FP16
- Topk: 0.5, 0.3, 0.1
- Options: `--warmup 2 --iters 3`

## Command

```bash
source /opt/intel/oneapi/setvars.sh --force
export ZE_AFFINITY_MASK=5
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export SAGE_ATTN_SPARSE_PREPROCESS_BACKEND=triton_xpu

.venv/bin/python benchmarks/bench_sparse_topk.py \
  --dtype bf16 --seq-len 75000 \
  --topk 0.5 0.3 0.1 \
  --tensor-layout HND NHD \
  --q-tile-override 256 \
  --sparse-q-block-tokens 256 \
  --sparse-k-block-tokens 64 \
  --warmup 2 --iters 3 \
  --output-csv benchmarks/results/k32_qtile256_sweep_20260814_163445/bf16_k32_qtile256_topk_0p5_0p3_0p1.csv

.venv/bin/python benchmarks/bench_sparse_topk.py \
  --dtype fp16 --seq-len 75000 \
  --topk 0.5 0.3 0.1 \
  --tensor-layout HND NHD \
  --q-tile-override 256 \
  --sparse-q-block-tokens 256 \
  --sparse-k-block-tokens 64 \
  --warmup 2 --iters 3 \
  --output-csv benchmarks/results/k32_qtile256_sweep_20260814_163445/fp16_k32_qtile256_topk_0p5_0p3_0p1.csv
```

## BF16 Sparse SDPA

| layout | topk | selected ratio | kernel latency | kernel vs ark | effective TFLOPS | e2e latency | e2e vs ark |
|---|---:|---:|---:|---:|---:|---:|---:|
| HND | 0.5 | 0.502557 | 859.097 ms | 1.600x | 67.390 | 1020.700 ms | 1.347x |
| HND | 0.3 | 0.302729 | 554.850 ms | 2.477x | 62.854 | 713.625 ms | 1.926x |
| HND | 0.1 | 0.103752 | 198.340 ms | 6.930x | 60.261 | 359.291 ms | 3.825x |
| NHD | 0.5 | 0.502557 | 872.543 ms | 1.647x | 66.352 | 1035.034 ms | 1.388x |
| NHD | 0.3 | 0.302729 | 544.671 ms | 2.638x | 64.028 | 709.006 ms | 2.027x |
| NHD | 0.1 | 0.103752 | 187.604 ms | 7.659x | 63.710 | 348.802 ms | 4.120x |

## FP16 Sparse SDPA

| layout | topk | selected ratio | kernel latency | kernel vs ark | effective TFLOPS | e2e latency | e2e vs ark |
|---|---:|---:|---:|---:|---:|---:|---:|
| HND | 0.5 | 0.502557 | 860.806 ms | 1.607x | 67.256 | 1052.201 ms | 1.315x |
| HND | 0.3 | 0.302729 | 554.388 ms | 2.495x | 62.906 | 745.712 ms | 1.855x |
| HND | 0.1 | 0.103752 | 198.106 ms | 6.982x | 60.333 | 388.876 ms | 3.557x |
| NHD | 0.5 | 0.502557 | 872.341 ms | 1.648x | 66.367 | 1064.251 ms | 1.351x |
| NHD | 0.3 | 0.302729 | 545.090 ms | 2.637x | 63.979 | 740.416 ms | 1.942x |
| NHD | 0.1 | 0.103752 | 186.927 ms | 7.690x | 63.941 | 378.443 ms | 3.799x |

## Notes

- BF16 and FP16 sparse SDPA K32 performance is nearly identical for kernel-only latency.
- NHD is slightly faster than HND at lower topk for sparse SDPA kernel-only results.
- At topk 0.5, K32 sparse SDPA is already faster than dense `ark.sdpa` for both dtypes and layouts.
- At topk 0.1, kernel-only speedup vs dense `ark.sdpa` reaches about 7x.
