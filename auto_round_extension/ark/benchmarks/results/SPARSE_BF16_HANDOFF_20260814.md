# Handoff: BF16 native sparse SDPA (sparge) — 2026-08-14

Status snapshot for a developer picking up this work. Everything here was verified on
**Intel Battlemage XPU (24.4 GB, `intel_gpu_bmg_g21`)**, oneAPI 2026.1, PyTorch 2.13.0+xpu.

## 1. Where the code lives

- **Branch `feat/sparse-bf16-prefill-v2`** — all sparse work (INT8 SAGE **and** native BF16/FP16
  sparse SDPA). **`main` only has the INT8 SAGE path** (empty `ark.cpp`/`sparse_attention.py` were
  false alarms; check paths against the repo root — `main` has a working int8 `sparge_sage2_attn_meansim_topk_xpu`
  in `auto_round_extension/ark/auto_round_kernel/sparse_attention.py`, 1440 lines).
- Switching branches requires a **rebuild** every time (the `.so` in `auto_round_kernel/ark-xbuild/`
  only matches the branch it was built on).

### Key changes in this handoff
- BF16 sparse SDPA now has a separate native BF16/FP16 path (`SparseSDPAConfig`) instead of reusing
  the INT8 SAGE DPAS path.
- `benchmarks/bench_sparse_topk.py` now includes dense `ark.sdpa` as a baseline and reports
  `speedup_vs_ark`.
- The BF16 sparse selected-block microtile was A/B tested at K64 vs K32. K32 is the current
  experimental direction and the locally rebuilt `.so` is back on K32.
- `xe_sparse_sdpa_fwd_mainloop.hpp` supports expanding one 64-token sparse route block into multiple
  physical K microtiles, which is required for K32 while keeping the existing 64-token LUT format.
- Unitrace profiles show K64 regresses because of compiler-reported spill plus SEND/SBID and
  memory-write pressure.
- A FLUX.1-dev sparse BF16 topk 0.9 K32 image was generated successfully with zero fallbacks.

### Two separate paths (do not mix)

| path | wrapper | config | DPAS |
|---|---|---|---|
| INT8 sparse | `sparge_sage2_attn_meansim_topk_xpu` / `sage_sparse` | `SparseSageConfig` | `XE_DPAS_TT<8, int, int8, int8, int>` (+ `static_assert(ElementQ==int8_t)`) |
| BF16/FP16 sparse | `sparge_sage2_attn_meansim_topk_xpu_sdpa` / `sage_sparse_sdpa` | `SparseSDPAConfig` | `XE_DPAS_TT<8, float, bf16, bf16, float>` |

**Using the int8 DPAS on bf16 fragments hangs the device** (fixed by the split; see
`SPARSE_BF16_DPAS_HANG.md`).

### Key files
- `auto_round_kernel/sdpa_sparse_sdpa.cpp` — BF16 launcher. `sdpa_sparse.cpp` — INT8 launcher.
- `auto_round_kernel/sparse_attention.py` — Python wrappers + `sparge_preprocess_topk`.
- `auto_round_kernel/sparge_preprocess_triton.py` — triton_xpu preprocess (+ torch fallback).
- `wrapper/include/stla/xe_sparse_sdpa_fwd_mainloop.hpp`, `sycl_tla_sdpa_sparse.hpp` — kernels.
- `auto_round_kernel/sdpa.cpp` — **dense** SYCL-TLA SDPA → Python `ark.sdpa(...)`
  (`xe_sdpa_fwd_mainloop.hpp`).

## 2. Correctness — verified

- **Repo e2e tests pass** (`test_sage_sparse_prefill_e2e.py`, `test_sparge_preprocess_topk_e2e.py`).
- **topk=1 (all blocks selected) == dense SDPA**: max diff **0.0039** (bf16 rounding) vs an exact
  CPU fp32 reference. This is the "dense gate" test.
- **INT8 == BF16 kernels**: on identical Q/K/V they select the *same* blocks (bit-identical LUT)
  and produce outputs within 0.008; verified across separate processes (bit-identical).
- **Preprocess + kernel are deterministic** (same QKV → identical results, in-process and
  cross-process).

## 3. Performance (bench `benchmarks/bench_sparse_topk.py`, seq 75000, 40h×128)

New mode `dense_ark_sdpa` (dense SYCL-TLA SDPA via `ark.sdpa`) added as a third baseline; every
row gets a `speedup_vs_ark` column. Full tables in `BENCH_SPARSE_ARK_SDPA_20260813.md`.

| mode | topk | vs torch | vs sagev1 | vs ark_sdpa |
|---|---|---|---|---|
| dense_ark_sdpa | — | 1.5–1.6× | 0.82–0.87× | 1.0 |
| INT8 sparse qtile256 (kernel) | 0.5 / 0.25 / 0.125 | 3.5 / 6.9 / 13.2× | — | 2.2 / 4.3 / 8.2× |
| BF16 sparse (kernel) | 0.5 / 0.25 / 0.125 | 1.5 / 2.9 / 5.6× | — | 0.92 / 1.8 / 3.5× |

Notes: `dense_ark_sdpa` is faster than torch SDPA but slower than `dense_sagev1`; the BF16 sparse
kernel only beats the dense ark kernel at topk ≤ 0.25. The 1.5×-not-2× ceiling at topk 0.5 is the
**SBID scoreboard stall** (memory-SEND concurrency bound; INT8 avoids it because its K is half the
size) — earlier profiling details in `SPARSE_BF16_KERNEL_OPTIMIZATION_JOURNEY.md` /
`SPARSE_SDPA_KERNEL_OPTIMIZATION_OPTIONS.md`.

**2026-08-14 A/B update:** changing the BF16 sparse qtile256 selected-block microtile from K64 to
K32 improved kernel-only latency by **1.69–1.77×**. At topk 0.5, BF16 sparse improved from
1511 ms / **0.91× vs ark** to 863 ms / **1.59× vs ark**. Correctness e2e still passed. See
`SPARSE_BF16_K32_AB_20260814.md`.

**2026-08-14 unitrace update:** profiling K32 vs K64 at topk 0.5 confirmed the K64 regression is
compiler-reported spill plus memory/SEND scoreboard pressure, not lower occupancy. Kernel properties
from `unitrace --device-timing --verbose` showed `Spill Memory Per Thread = 0 B` for K32 and
`3840 B` for K64, with the same `Private Memory Per Thread = 2432 B` and register file size 256.
Aggregated `SPARSESDPAFwdMainloop` GPU time was 1684.6 ms (K32) vs 3072.4 ms (K64), XVE active
64.7% vs 39.9%, XVE stall 35.1% vs 59.9%, and SBID stall 31.4% vs 55.7%. SEND instructions rose
4.4× and GPU memory writes rose ~5.0× with K64. SLM read/write was 0/0 in both profiles because the
spill is private/scratch traffic, not SLM-backed storage. Raw profiles are in
`unitrace_k32_compute_20260814/`, `unitrace_k32_stalls_20260814/`,
`unitrace_k32_verbose_20260814/`, `unitrace_k64_compute_real_20260814/`, and
`unitrace_k64_stalls_real_20260814/`.

## 4. FLUX.1-dev end-to-end — the critical gotcha

**Use `enable_sequential_cpu_offload()`, NOT `enable_model_cpu_offload()`.**

- `enable_model_cpu_offload()` streams at the *component* level: the whole ~46 GB transformer gets
  pulled onto the 24.4 GB device (peak ~24 GB, ~350 MB headroom). The sparse launch then flakily
  OOMs (`OUT_OF_DEVICE_MEMORY` / `OUT_OF_RESOURCES`), and can **reset the GPU**
  (`dmesg: exec queue reset` on `0000:ba:00.0` = dev5; the `DEVICE_LOST` error).
- Sequential offload keeps the peak at **~0.16 GB** → triton_xpu sparse runs reliably (0 fallbacks).
  Cost: ~4 s/step vs ~1 s/step (50-step generation ≈ 4–5 min).

### FLUX image-quality findings
- bf16 sparse and int8 sparse produce **identical images** at the same config (verified, 0.0 diff).
- `topk=0.9` drops **12.5–16.7%** of blocks (route-block rounding + sim threshold), not 10% →
  images sit ~15–18 from dense. Attention-mass coverage: topk 0.9 keeps **91.5%** of the mass,
  topk 0.5 keeps **64.6%** — the renormalized output shifting is inherent to topk-sparse on
  FLUX's spread-out attention.
- **Latest K32 BF16 sparse topk 0.9 run:** saved PNG at
  `benchmarks/results/flux_bf16_topk09_k32_20260814_112445/flux_bf16_qtile256_topk0.9_512_dev5.png`.
  Config: BF16 sparse qtile256, selected-block K32, `topk=0.9`, 512×512, 50 steps, seed 0,
  `ZE_AFFINITY_MASK=5`. Result: wall **238.884 s**, sparsity **0.125**, calls **2850**,
  sparse_calls **2850**, runtime_fallbacks **0**.
- **Open question: run-to-run non-determinism.** Same config (bf16 qtile256 topk0.5, seed 0) gave
  images differing by **48.5** across two runs (means 97.6 vs 101.3), while topk 0.9 was
  deterministic (0.0). QKV, preprocess and kernel are all verified deterministic → suspected
  diffusion-trajectory amplification of a tiny perturbation elsewhere, not a sparse-kernel bug.
  Worth re-checking before trusting single-run image comparisons.

### Dev tooling (untracked)
- `examples/flux_gen_bf16_sweep.py` — offload-only runner (loop over topk, saves PNGs + summary).
- `tools/sweep_flux_bf16_topk.sh` — one-GPU wrapper (qtile256 env). **Note: `FLUX_SPARSE_KERNEL`
  is now env-overridable (default bf16)**; earlier it hardcoded bf16 and silently made an "int8"
  run bf16. `main`'s `flux_sparse_patch.py` is int8-only and ignores the block-token envs.
- `tools/bench_sparse_bf16_fp16_sweep.sh` — the bench sweep driver.

## 5. Current repo state (uncommitted)

- **Branch**: `feat/sparse-bf16-prefill-v2`.
- **Modified (tracked)**: `benchmarks/bench_sparse_topk.py` (added `dense_ark_sdpa` mode +
  `speedup_vs_ark` column), sparse SDPA headers for K32 route-block expansion — not yet committed.
- **Untracked**: `examples/flux_gen_bf16_sweep.py`, `tools/sweep_flux_bf16_topk.sh`,
  `tools/bench_sparse_bf16_fp16_sweep.sh`.
- `.so`: `auto_round_kernel/ark-xbuild/auto_round_kernel_xpu.cpython-313-x86_64-linux-gnu.so`
  (feature-branch build). A stale **main** `.so` copy sits in `auto_round_kernel/xbuild/` (harmless;
  feature's loader uses `ark-xbuild`).
- `benchmarks/results/` is gitignored → all result CSVs, PNGs, and docs below are local-only.

## 6. How to run

```bash
# rebuild (after oneAPI: source /opt/intel/oneapi/setvars.sh --force)
cd auto_round_extension/ark/auto_round_kernel/ark-xbuild && cmake --build . -j 16

# correctness tests
export ZE_AFFINITY_MASK=5 SAGE_ATTN_SPARSE_PREPROCESS_BACKEND=torch
.venv/bin/python auto_round_kernel/wrapper/test/test_sage_sparse_prefill_e2e.py
.venv/bin/python auto_round_kernel/wrapper/test/test_sparge_preprocess_topk_e2e.py

# FLUX sweep (bf16 qtile256; set FLUX_SPARSE_KERNEL=int8 for SAGE)
export ZE_AFFINITY_MASK=5 SAGE_ATTN_SPARSE_PREPROCESS_BACKEND=triton_xpu
export FLUX_SPARSE_KERNEL=bf16 FLUX_SPARSE_TOPKS="0.9 0.5" FLUX_RUN_DENSE=0
export FLUX_OUTPUT_DIR=benchmarks/results/flux_<stamp>
.venv/bin/python examples/flux_gen_bf16_sweep.py

# bench (needs feature branch — main lacks bf16 symbols)
export ZE_AFFINITY_MASK=5 SAGE_ATTN_SPARSE_PREPROCESS_BACKEND=triton_xpu
.venv/bin/python benchmarks/bench_sparse_topk.py --dtype bf16 --seq-len 75000 \
  --topk 0.5 0.25 0.125 --tensor-layout HND NHD --q-tile-override 256 \
  --sparse-q-block-tokens 256 --sparse-k-block-tokens 64 --warmup 2 --iters 3 \
  --output-csv benchmarks/results/bench_<stamp>.csv
```

## 7. Open items / suggestions

1. **Resolve the topk-0.5 cross-run non-determinism** before trusting single-run image diffs
   (capture QKV per run; all sparse ops verified deterministic).
2. **Clean up the K32 BF16 sparse SDPA experiment** (`SPARSE_BF16_K32_AB_20260814.md`) into a final
   implementation. The unitrace A/B says to keep the selected-block microtile at K32 unless a later
   compiler/assembly pass removes the K64 write/SBID pressure.
3. Consider making the FLUX runner's `_512` filename suffix derive from the actual size (cosmetic;
   the 1024×1024 run is mislabeled `_512`).
4. Decide whether to commit the bench `dense_ark_sdpa` change + dev tooling (currently uncommitted).
