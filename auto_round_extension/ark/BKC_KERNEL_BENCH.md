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

## Historical Sparse K-Prefetch A/B

The current branch keeps sparse `K` prefetch enabled by default, so there is no
longer a build-time toggle in `CMakeLists.txt`.

The `prefetch=ON/OFF` comparisons below are historical A/B data captured before
that cleanup, when both paths were still build-selectable.

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

For the old `prefetch=OFF` path, rerun the same `NHD` and `HND` commands with:

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

## B70 Profiling Snapshot

This node is `0xe223` (`B70` / `bmg_g31`). The following `unitrace` A/B was run
for the same sparse case in `HND` layout:

- `batch=1`
- `num_heads_q=40`
- `num_heads_kv=40`
- `seq_len=75600`
- `head_dim=128`
- `topk=0.5`
- `q_tile=256`
- `sparse_q_block_tokens=256`
- `sparse_k_block_tokens=64`
- `ZE_AFFINITY_MASK=0`

Historical build toggle used for this A/B:

- `ARK_SPARSE_SAGE_ENABLE_K_PREFETCH=OFF`
- `ARK_SPARSE_SAGE_ENABLE_K_PREFETCH=ON`

Machine-specific profiling command:

```bash
sg render -c 'bash -lc "
cd /home/yiliu4/workspace/auto-round-prefill-clean-sparse-pr/auto_round_extension/ark
export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/2025.3/lib:/opt/intel/oneapi/2025.3/lib:${LD_LIBRARY_PATH}
ZE_AFFINITY_MASK=0 \
/home/yiliu4/workspace/pti-gpu/tools/unitrace/install_local/bin/unitrace \
  -d -s --chrome-kernel-logging --demangle \
  --devices-to-sample 0 \
  --output sparse_vecstall_sampling_prefetch_<on_or_off> \
  /home/yiliu4/workspace/auto-round-py/.venv/bin/python \
  test/bench_sparse_topk.py \
    --batch 1 \
    --num-heads-q 40 \
    --num-heads-kv 40 \
    --seq-len 75600 \
    --head-dim 128 \
    --tensor-layout HND \
    --topk 0.5 \
    --q-tile-override 256 \
    --sparse-q-block-tokens 256 \
    --sparse-k-block-tokens 64 \
    --warmup 0 \
    --iters 1
"'
```

Kernel properties from `unitrace`:

| Setting | Sparse Private Mem / Thread | Sparse Spill / Thread | Dense Spill / Thread | Notes |
|---|---:|---:|---:|---|
| `prefetch=OFF` | `1536 B` | `3392 B` | `128 B` | large sparse spill remains |
| `prefetch=ON` | `2048 B` | `640 B` | `128 B` | spill drops a lot, private mem rises |

Sparse kernel timing from the same profiled runs:

| Setting | XeSparseSageFwdKernel Calls | Total Time (ns) | Avg Time (ns) |
|---|---:|---:|---:|
| `prefetch=OFF` | `2` | `1377784270` | `688892135` |
| `prefetch=ON` | `2` | `868062916` | `434031458` |

Profiled benchmark output from the same runs:

| Setting | dense_torch_sdpa | dense_sagev1 | sparse kernel_only | sparse e2e |
|---|---:|---:|---:|---:|
| `prefetch=OFF` | `2160.233 ms` | `8896.800 ms` | `40102.633 ms` | `837.385 ms` |
| `prefetch=ON` | `2014.259 ms` | `873.501 ms` | `42590.231 ms` | `620.531 ms` |

Interpretation:

- On this `B70` node, enabling sparse `K` prefetch increases sparse private
  memory but reduces sparse spill substantially.
- The sparse kernel also ran faster under `unitrace` with prefetch enabled.
- Treat the benchmark latencies in this section as profiler-distorted. Use them
  only for `prefetch ON/OFF` comparison inside `unitrace`, not as normal
  benchmark numbers.

Saved profiling artifacts:

- `sparse_vecstall_sampling_prefetch_off.3716760`
- `python.3716760.json`
- `sparse_vecstall_sampling_prefetch_on.3756979`
- `python.3756979.json`

## B70 `sycl_tla` Tag A/B

On the same `B70` node, with sparse `K` prefetch kept enabled, a separate build
was created with:

- baseline tree: `auto_round_kernel/xbuild`
- baseline tag: `SYCL_TLA_GIT_TAG=260409`
- comparison tree: `auto_round_kernel/xbuild_tla260630_prefetch_on`
- comparison tag: `SYCL_TLA_GIT_TAG=260630`

The benchmark script only loads the extension from `xbuild/`, so the rerun used
the `xbuild` binary first, then temporarily swapped in the `260630` `.so`, ran
the same benchmark, and restored the original `xbuild` binary afterward.

Rerun conditions:

- free device chosen from `xpu-smi ps`: `GPU7`
- `ZE_AFFINITY_MASK=7`
- `batch=1`
- `num_heads_q=40`
- `num_heads_kv=40`
- `seq_len=75600`
- `head_dim=128`
- `topk=0.5`
- `q_tile=256`
- `sparse_q_block_tokens=256`
- `sparse_k_block_tokens=64`
- `warmup=2`
- `iters=3`

Sparse benchmark comparison:

| Layout | `260409` kernel-only | `260630` kernel-only | Delta |
|---|---:|---:|---:|
| `HND` | `409.834 ms` | `399.086 ms` | `2.6%` faster |
| `NHD` | `481.936 ms` | `465.911 ms` | `3.3%` faster |

| Layout | `260409` e2e | `260630` e2e | Delta |
|---|---:|---:|---:|
| `HND` | `525.080 ms` | `515.407 ms` | `1.8%` faster |
| `NHD` | `582.289 ms` | `564.575 ms` | `3.0%` faster |

Interpretation:

- On this rerun, `SYCL_TLA_GIT_TAG=260630` was consistently but modestly faster
  than `260409`.
- The improvement was small on `HND` and slightly larger on `NHD`.
- This comparison was done with sparse `K` prefetch enabled in both builds.

Saved CSVs:

- `bench_sparse_topk_prefetch_on_tla260409_hnd_gpu7.csv`
- `bench_sparse_topk_prefetch_on_tla260409_nhd_gpu7.csv`
- `bench_sparse_topk_prefetch_on_tla260630_hnd_gpu7.csv`
- `bench_sparse_topk_prefetch_on_tla260630_nhd_gpu7.csv`

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
