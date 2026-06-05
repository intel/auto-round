# sycl-tla Build BKC + ARK mixed-dtype GEMM comparison (Battlemage G21)

Best Known Configuration for building [intel/sycl-tla](https://github.com/intel/sycl-tla)
(the CUTLASS/CuTe SYCL fork) on the `inc101` Battlemage G21 node, plus a measured
comparison of its **fused in-register u4 dequant** GEMM against ARK's **two-pass int8-XMX**
`woq_s4` prefill path.

Validated against sycl-tla `v0.9-17-g14859073` (commit `14859073`, 2026-06-01).

## TL;DR finding (the reason this comparison was run)

For m>1 prefill, **ARK's existing two-pass int8-XMX path is ~4.6× faster than sycl-tla's
fused fp16 path** at the same GEMM shape (m512/n4096/k11008: ARK 0.71 ms vs sycl-tla 3.25 ms).

The fused path is *not* automatically a win. sycl-tla's `f16×u4` collective dequantizes the
u4 weight to **fp16 in register**, then runs an **fp16 MMA** (`XE_8x16x16_F32F16F16F32_TT`).
fp16 DPAS peaks at ~98 TFLOP/s on this part — **half** the int8 XMX peak (~197 INT8 TOPS).
ARK pays a DRAM int8 round-trip but keeps the matmul on the **2×-faster int8 DPAS**, and that
wins decisively. So a naive "port ARK woq_s4 m>1 to a 02-style fused collective" would
**regress** prefill throughput. A fused path only beats ARK if it dequants u4→**int8** in
register and stays on int8 DPAS (the `*U4S8S8*` variant — but sycl-tla's current int8-MMA
mixed config is poorly tuned, see numbers below).

## Validated environment

| Component       | Version / detail                                       |
|-----------------|--------------------------------------------------------|
| GPU             | Intel Battlemage G21 (`0xe211`), Level Zero backend     |
| oneAPI compiler | Intel oneAPI DPC++/C++ `icx`/`icpx` 2025.3              |
| CMake / Ninja   | 4.3.1 / 1.13.0 (from the `ark` venv)                    |
| sycl-tla        | `v0.9-17-g14859073` at `/home/yiliu7/workspace/sycl-tla`|
| SYCL target     | `intel_gpu_bmg_g21`                                      |

## Build

sycl-tla is already cloned at `/home/yiliu7/workspace/sycl-tla` (it is also an ARK
FetchContent dep). Configure + build only the targets you need — a full suite build is large.

```bash
cd /home/yiliu7/workspace/sycl-tla
source /opt/intel/oneapi/setvars.sh                       # icx/icpx + SYCL runtime
export PATH="/home/yiliu7/workspace/venvs/ark/bin:$PATH"   # cmake / ninja

# Proxy only if FetchContent needs the network (sycl-tla itself has no extra clones here):
# export http_proxy=http://proxy.ims.intel.com:911 https_proxy=$http_proxy \
#        no_proxy="intel.com,.intel.com,localhost,127.0.0.1"

CC=icx CXX=icpx cmake -S . -B build -G Ninja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21 \
  -DCUTLASS_SYCL_PROFILING_ENABLED=ON \
  -DCUTLASS_ENABLE_EXAMPLES=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-I/home/yiliu7/workspace/venvs/ark/include"
```

Notes / gotchas:
- `DPCPP_SYCL_TARGET=intel_gpu_bmg_g21` is the device. `bmg_g31` or `bmg` (both) also valid.
- `CUTLASS_SYCL_PROFILING_ENABLED=ON` makes the harness time with SYCL events (device time).
- The `-I.../venvs/ark/include` flag pre-empts the same **`oneapi/mkl/rng/device.hpp` not
  found** blocker documented in `ark_xpu_build_bkc.md` (cutlass reference RNG pulls it in).
- Configure prints harmless `libiomp5.so ... may be hidden` RPATH warnings — ignore them.
- The toolchain `g++` is forced by oneAPI; to pin a host g++ add `-DDPCPP_HOST_COMPILER=g++-13`.

### Build the targets

```bash
# (1) The standalone learning example: fp16 act x u4 weight -> fp16, self-verifying.
cmake --build build --target 02_bmg_gemm_f16_u4_f16 -j 4

# (2) The TUNED benchmark harness (use this for perf numbers, NOT the example):
cmake --build build --target cutlass_benchmarks_gemm_sycl_legacy -j 4
```

> **Why two binaries?** The example hardcodes a `256×256×32` workgroup tile that **spills
> registers** ("compiled SIMD16 ... spilled around 141" at link) and reports non-monotonic,
> low TFLOP/s — fine for correctness, useless for perf. The `*_legacy` benchmark uses the
> tuned `Shape<_8,_128,_32>` mixed-dtype config (`benchmarks/gemm/legacy/benchmarks_sycl.hpp`,
> `PvcMixedPrecisionGemm*`) and reports best/avg/worst over ~hundreds of iters. Trust (2).

## Run

### The example (correctness demo)

```bash
cd /home/yiliu7/workspace/sycl-tla
source /opt/intel/oneapi/setvars.sh
export PATH="/home/yiliu7/workspace/venvs/ark/bin:$PATH"
BIN=build/examples/02_bmg_gemm_mixed_dtype/02_bmg_gemm_f16_u4_f16

# mode: 0=convert-only, 1=convert+scale (≈ ARK symmetric s4), 2=+zero-point.
# g = groupwise scale block (128 ≈ ARK blocksize); g=k => tensorwise.
ONEAPI_DEVICE_SELECTOR=level_zero:gpu "$BIN" \
  --m=512 --n=4096 --k=11008 --l=1 --mode=1 --g=128 --iterations=50
# -> "Disposition: Passed" + a (register-spilled, low) TFLOP/s line.
```

`--mode=1` zero-fills the zero-point, so dequant is pure `scale × value` — the right match for
ARK's symmetric s4 (the −8 centering ARK uses costs nothing extra at runtime).

### The tuned benchmark (perf numbers)

The harness is config-file driven. A config with ARK's exact gemm-bench shapes lives at
`benchmarks/device/bmg/input_files/input_ark_compare_mixed_dtype.in`:

```bash
BIN=build/benchmarks/gemm/legacy/cutlass_benchmarks_gemm_sycl_legacy
CFG=benchmarks/device/bmg/input_files/input_ark_compare_mixed_dtype.in
ONEAPI_DEVICE_SELECTOR=level_zero:gpu "$BIN" --config_file="$CFG" 2>&1 | grep ark_m
# read the best_tflop= and best_runtime_ms= fields.
```

The stock sglang config (`input_sglang_gemm_mixed_dtype.in`, m=32 only) runs all 8 variants
(fp16/bf16 × {fp16,s8,s8s8} MMA) — useful to see the MMA-datapath spread.

## Measured numbers (Battlemage G21)

**sycl-tla `FP16U4FP16F16FP16S4` (fp16 act × u4 weight, in-register dequant, fp16 MMA):**

| shape (m·n·k)      | best ms | best TFLOP/s | % of ~98 fp16-peak |
|--------------------|---------|--------------|--------------------|
| m32  n4096 k4096   | 0.300   |  3.6         |  4%                |
| m128 n4096 k4096   | 0.388   | 11.1         | 11%                |
| m512 n4096 k11008  | 3.25    | 14.2         | 14%                |
| m1024 n4096 k11008 | 6.34    | 14.6         | 15%                |

**ARK `woq_s4` (two-pass: unpack int4→int8 in DRAM, then s8×s8→s32 XMX)** — from
`ark_xpu_build_bkc.md`:

| shape (m·n·k)      | best ms | best TFLOP/s | % of ~197 int8-peak |
|--------------------|---------|--------------|---------------------|
| m32  n4096 k4096   | —       |  7.97        |  4%                 |
| m128 n4096 k4096   | 0.177   | 32.8         | 17%                 |
| m512 n4096 k11008  | 0.710   | 65.1         | 33%                 |

**Head-to-head, identical m512/n4096/k11008 GEMM:** ARK **0.71 ms** vs sycl-tla **3.25 ms** —
ARK is **4.6× faster in wall-clock**.

sglang m=32 spread (shows the MMA datapath cost directly):

```
FP16U4FP16F16FP16S4 (fp16 MMA) : best 10.1 TFLOP/s   <- fastest mixed variant
FP16U4FP16S8FP16S4  (s8  MMA)  : best  3.2 TFLOP/s   <- int8-MMA mixed path, poorly tuned
FP16S8FP16S8FP16S8  (pure s8)  : best  4.3 TFLOP/s
```

## Interpretation

1. **The MMA datapath dominates, not the dequant location.** sycl-tla's fused path avoids
   ARK's DRAM int8 round-trip but runs **fp16** DPAS (½ the int8 peak). Net: it loses ~4.6×.
   The round-trip ARK "wastes" is cheaper than halving MMA throughput.
2. **ARK's two-pass design is the right call today** for s4 prefill on this part, *because*
   the int4→int8 unpack lets the matmul stay on the fast int8 DPAS. The earlier hypothesis
   ("fused in-register dequant must be faster") is **wrong for fp16-MMA fusion**.
3. **A fused path could still win** only if it dequants u4→**int8** in register and keeps int8
   DPAS. sycl-tla *has* such a variant (`*U4S8S8*`) but it currently measures slower than its
   own fp16 path (3.2 vs 10.1 TFLOP/s @ m32) — its int8-MMA mixed config is not tuned. That,
   not the fp16 example, is the path worth pursuing if ARK ever fuses.
4. **Caveat — the tuned config targets small-m decode.** The `Shape<_8,_128,_32>` mixed-dtype
   config is tuned for sglang's m=32 shapes; its large-m numbers (flat 14.2→14.6 from
   m512→m1024) likely undersell a prefill-tuned ceiling. Even doubled it would not close a
   4.6× gap against int8 DPAS, but the absolute fp16 numbers here are a floor, not a ceiling.

## Files added by this exercise

- `benchmarks/device/bmg/input_files/input_ark_compare_mixed_dtype.in` — ARK-shape u4 config
  (in the sycl-tla tree, not committed to auto-round).
