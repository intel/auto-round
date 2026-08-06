# Sparse BF16 Handoff (updated 2026-08-06)

> **Note:** This is a refresh of the 2026-08-05 handoff. The earlier version described a
> pre-refactor state and reported the BF16 benchmark as failing; both are stale. The BF16
> sparse path has been cleanly separated from the INT8 sparse path, and recent benchmark
> runs complete successfully.

## Scope

Handoff for the `sparse BF16` work in `auto_round_extension/ark` after the INT8/BF16
path-separation refactor on branch `feat/sparse-bf16-prefill-v2`.

Covers:

- the final architecture (two clean, per-dtype sparse attention paths)
- the public API
- build/verify/reproduce steps
- current benchmark status and known caveats

## Goal

Two independent sparse SDPA paths, one per precision:

- **INT8** — `sage_sparse` / `sparge_sage2_attn_meansim_topk_xpu`
- **BF16 / FP16** — `sage_sparse_sdpa` / `sparge_sage2_attn_meansim_topk_xpu_sdpa`

Previously the BF16 native path was mixed into the INT8-centric SAGE kernel path
(`sage_sparse_bf16` routed native Q/K through `SparseSageConfig` /
`SPARSESAGEV1FwdMainloop`). The refactor removed that coupling.

## Current Environment

- repo: `/home/yiliu4/workspace/auto-round/auto_round_extension/ark`
- Python: `.venv/bin/python`
- PyTorch: `2.13.0+xpu`, `torch.version.xpu=20260000`
- oneAPI compiler: `2026.1.1` (`/opt/intel/oneapi/compiler/2026.1`), MKL `2026.1`
- CMake XPU build target: `intel_gpu_bmg_g21`, `ARK_SYCL_TLA=ON`, icx

## Architecture (post-refactor)

**INT8 path:** `sparse_attention.py:sage_sparse` → pybind `ark.cpp:sage_sparse` →
`sdpa_sparse.cpp` (`sdpa_impl_qks8_sparse_{d64,row_linear,qtile256_row64k}_pvhalf`) →
`SparseSageConfig` → `SPARSESAGEV1FwdMainloop`
(`wrapper/include/stla/xe_sparse_sagev1_fwd_mainloop.hpp`).

- Q/K are `int8_t` (dequantized via qscale/kscale); V/PV is fp16/bf16.
- Q*K MMA is hardcoded to int32/int8 DPAS; `SparseSageConfig` carries a
  `static_assert` that `ElementQ == int8_t` (INT8-only).

**Native BF16/FP16 path:** `sparse_attention.py:sage_sparse_sdpa` → pybind
`ark.cpp:sage_sparse_sdpa` → `sdpa_sparse_sdpa.cpp`
(`sdpa_impl_{bf16,fp16}_sparse_sdpa_{d64,row_linear,qtile256_row64k}`) →
`SparseSDPAConfig` → `SPARSESDPAFwdMainloop`
(`wrapper/include/stla/xe_sparse_sdpa_fwd_mainloop.hpp`).

- Q/K/V native precision; `scale_block_size = 0`, no dequant scales; softmax scale applied directly.
- Head dim 64/128, q_tile_override 0/64/256 (d64 / row_linear / qtile256_row64k).

**Shared:** `sparge_preprocess_topk` (torch or triton preprocess backend),
`XeSparseSageFwdKernel` (`xe_sparse_sage_fwd_kernel.hpp`), `SparseFMHAFwdEpilogue`.

**Removed by the refactor:**

- `auto_round_kernel/sdpa_sparse_bf16.cpp` (superseded by `sdpa_sparse_sdpa.cpp`)
- `sage_sparse_bf16` (pybind binding + Python alias; was an alias of `sage_sparse_sdpa`)
- `sparge_sage2_attn_meansim_topk_xpu_bf16` (e2e wrapper)
- `sdpa_impl_bf16_sparse_{d64,row_linear,qtile256_row64k}` declarations
- native softmax-scale branch in the INT8 SAGE mainloop

**Follow-up fix (verification pass):** `_query_tile_tokens_for_head_dim(64)` now returns
64 (was 128) so the preprocess None-default for head_dim 64 matches the sparse e2e
wrappers, fixing a pre-existing `tile_block_map` metadata mismatch that failed
`test_sparge_preprocess_topk_e2e.py`.

## Public API Status

BF16/FP16 sparse is exposed as:

- `sage_sparse_sdpa(query, key, value, lut, valid_block_num, ...)`
- `sparge_sage2_attn_meansim_topk_xpu_sdpa(query, key, value, ...)` (end-to-end)

INT8 sparse remains `sage_sparse` / `sparge_sage2_attn_meansim_topk_xpu`.
There is no `sage_sparse_bf16` anymore.

## Build Status

The XPU extension is rebuilt into `auto_round_kernel/ark-xbuild`:

```bash
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
export MKLROOT=/opt/intel/oneapi/mkl/2026.1
cmake -S auto_round_kernel -B auto_round_kernel/ark-xbuild \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icx \
  -DARK_XPU=ON -DARK_SYCL_TLA=ON -DARK_DNNL=OFF -DARK_JOINT_MATRIX=OFF \
  -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21
cmake --build auto_round_kernel/ark-xbuild -j 16
```

**Important:** `auto_round_kernel/CMakeLists.txt` collects sources with
`file(GLOB SRCS .../*.cpp)` and has **no** `CONFIGURE_DEPENDS`. After adding/removing a
`.cpp` (e.g. the deleted `sdpa_sparse_bf16.cpp`), you MUST re-run `cmake -S ... -B ...`
before `cmake --build`, otherwise the stale build references the removed file and the old
`.so` still exports the old symbols.

## Reproduce / Verify

```bash
cd /home/yiliu4/workspace/auto-round/auto_round_extension/ark
# correctness (INT8 + BF16 + FP16). Set torch preprocess backend to avoid an XPU
# scatter/gather OOB assertion that the triton preprocess backend triggers on this host.
export SAGE_ATTN_SPARSE_PREPROCESS_BACKEND=torch
.venv/bin/python auto_round_kernel/wrapper/test/test_sage_sparse_prefill_e2e.py
.venv/bin/python auto_round_kernel/wrapper/test/test_sparge_preprocess_topk_e2e.py

# benchmark (script already exports SAGE_ATTN_SPARSE_PREPROCESS_BACKEND=torch, ZE_AFFINITY_MASK=6)
./tools/repro_sparse_bf16_sdpa_bench.sh
# CSV default: /tmp/bench_sparse_topk_bf16_dev6.csv
```

## Benchmark Status (2026-08-06)

The BF16 sparse benchmark **completes successfully** (`status: ok`). Full sweep results
are in `benchmarks/results/bf16_sparse_20260806_full.csv`. Native-path
(`sparse_sdpa_bf16_*` via `sage_sparse_sdpa`) speedups vs dense torch SDPA:

| case | kernel-only | e2e |
|---|---|---|
| seq 32768, topk 0.5 | 1.40–1.44x | 0.85–0.87x |
| seq 32768, topk 0.125 | 5.38–5.67x | 1.56–1.57x |
| seq 75600, topk 0.5 | 1.51–1.53x | 1.15x |
| seq 75600, topk 0.125 | 5.71–5.83x | 2.64–2.65x |

The e2e mode is preprocess-bound at moderate sizes (topk 0.5 @ seq 32768) but wins at
larger seq_len / lower topk. These match the earlier mixed-path `sparse_bf16_*` numbers,
so the separation did not regress performance.

## Known Caveats

- **Preprocess backend:** use `SAGE_ATTN_SPARSE_PREPROCESS_BACKEND=torch`. The triton
  preprocess backend hits a PyTorch XPU scatter/gather "index out of bounds" assert on
  this host (`ScatterGatherKernels.cpp:233`). The repro script already sets `torch`;
  `tools/run_bf16_sparse_bench.sh` defaults to `triton_xpu` and should be overridden.
- **`scale_block_size == 0` on the native path:** `xe_sparse_sage_fwd_kernel.hpp`
  guards the `seq_*_pad` division by `scale_block_size` to avoid div-by-zero UB; pads are
  only used inside `scale_block_size ? ... : nullptr` ternaries.
- **INT8-only SAGE config:** `SparseSageConfig` enforces `ElementQ == int8_t` at compile
  time. Use `SparseSDPAConfig` for native-precision BF16/FP16.
- **Stale docs:** `SPARSE_BF16_HANDOFF_20260805.md` (this file) supersedes the
  pre-refactor handoff; `SPARSE_BF16_STATUS_20260729.md` / `SPARSE_BF16_BENCH_20260805.md`
  are historical and refer to older builds/symbols.

## Summary

The BF16 native sparse path is fully separated from the INT8 SAGE path. The build wiring,
public API, correctness tests, and benchmark modes are in place; BF16 sparse completes and
outperforms dense torch SDPA. The main runtime caveat is the preprocess backend choice
(torch, not triton). Remaining work is any follow-up perf tuning or integration into the
Flux/Wan sparse examples via `sage_sparse_sdpa`.
