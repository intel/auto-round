# ARK XPU Build BKC (Battlemage G21)

Best Known Configuration for building and testing the AutoRound ARK kernel for Intel XPU,
validated on an Intel Battlemage G21 (`0xe211`) node (`inc101`).

## Validated environment

| Component         | Version / detail                                   |
|-------------------|----------------------------------------------------|
| GPU               | Intel Battlemage G21 (`0xe211`), Level Zero backend |
| oneAPI compiler   | Intel oneAPI DPC++/C++ `icx` 2025.3.3              |
| PyTorch           | 2.11.0+xpu (XPU available)                          |
| CMake             | 4.3.1                                               |
| Ninja             | 1.13.0                                              |
| Python            | 3.13                                                |
| SYCL target       | `intel_gpu_bmg_g21` (device name `bmg_g21`)         |

The cmake/ninja/torch toolchain lives in the `ark` venv at
`/home/yiliu7/workspace/venvs/ark`. The oneAPI compiler is at `/opt/intel/oneapi`.

> **Per-node SYCL target (`bmg_g21` vs `bmg_g31`).** This BKC is the B60 / Battlemage
> G21 node (`inc101`, `0xe211`), hence `DPCPP_SYCL_TARGET=intel_gpu_bmg_g21`. The B70 /
> Battlemage G31 node (`0xe223`) is a different silicon (`bmg_g31`). **`DPCPP_SYCL_TARGET`
> only feeds the `-Xs -device` JIT tuning hint (`CMakeLists.txt:94-109`) — it does NOT
> gate the m>1 `joint_matrix` capability.** That gate is the `libsycl` runtime version
> (2025.3's `.so.8` matrix table predates `bmg_g31`; a `.so.9`/2026 runtime carries it).
> So on b70 the m>1 int8-XMX path needs a 2026 DPC++ runtime, not a target change — see
> `ark_xpu_joint_matrix_finding.md` (CORRECTION #3 + the `DPCPP_SYCL_TARGET` verdict).
> For a copy-paste verification of the released-2026 fix on b70, see
> `ark_xpu_verify_2026_kernel.md` (gate-check + real-kernel harness, with expected output).

> **Bringing up a fresh node?** This BKC assumes Python dev headers and a complete oneAPI MKL are
> already present. On a clean machine they are often missing and `setup.py build_ext` fails at the
> pybind11 configure (`Development.Module`) or the SYCL-TLA SDPA compile
> (`oneapi/mkl/rng/device.hpp` not found). See `ark_xpu_fresh_node_bkc.md` for those two blockers
> and their fixes.

## Prerequisites

1. **Source the oneAPI environment** (provides `icx`/`icpx`, SYCL runtime, headers):

   ```bash
   source /opt/intel/oneapi/setvars.sh
   ```

2. **Put the venv toolchain on PATH** (cmake, ninja, the XPU-enabled python):

   ```bash
   export PATH="/home/yiliu7/workspace/venvs/ark/bin:$PATH"
   ```

3. **Proxy** (only if you hit network errors during the FetchContent clones of
   oneDNN / sycl-tla / pybind11):

   ```bash
   export http_proxy=http://proxy.ims.intel.com:911
   export https_proxy=http://proxy.ims.intel.com:911
   export no_proxy="intel.com,.intel.com,localhost,127.0.0.1"
   export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy NO_PROXY=$no_proxy
   ```

4. **Verify the GPU and compiler are visible:**

   ```bash
   sycl-ls | grep level_zero:gpu     # expect Intel(R) Graphics [0xe211]
   icx --version                     # expect oneAPI DPC++/C++ Compiler 2025.3.x
   ```

## Critical gotcha — MKL RNG header

The cutlass / sycl-tla SDPA path includes `oneapi/mkl/rng/device.hpp`. That header ships with
the **pip MKL-devel** package but **not** with the oneAPI compiler bundle, so the build fails
with:

```
fatal error: 'oneapi/mkl/rng/device.hpp' file not found
```

Fix: add the venv include dir to the compile flags. The header set lives at
`/home/yiliu7/workspace/venvs/ark/include`:

```
-DCMAKE_CXX_FLAGS="-I/home/yiliu7/workspace/venvs/ark/include"
```

> Do **not** try to work around this by disabling SYCL-TLA (`-DARK_SYCL_TLA=OFF`). That path
> does not build either: `wrapper/include/xpu_wrapper.hpp::sagev1_impl` calls the SDPA impls
> (`sdpa_impl_qks8_pvi8` / `sdpa_impl_qks8_pvhalf`) unconditionally, so the full TLA build is
> required.

## Build the Python extension (production)

This is the path `setup.py` drives (`CMakeBuild` builds a CPU `.so` then an XPU `.so`):

```bash
cd /home/yiliu7/workspace/auto-round/auto_round_extension/ark
source /opt/intel/oneapi/setvars.sh
export PATH="/home/yiliu7/workspace/venvs/ark/bin:$PATH"

CMAKE_ARGS="-DCMAKE_CXX_FLAGS=-I/home/yiliu7/workspace/venvs/ark/include" \
  python setup.py bdist_wheel
pip install dist/*.whl
```

(If `setup.py` does not forward `CMAKE_CXX_FLAGS`, configure the XPU build manually as below and
copy the resulting `auto_round_kernel_xpu*.so` into the package, or use the unit-test build to
validate the kernel first.)

## Build the C++ unit test (kernel validation)

Builds the standalone `test_ARK_XPU` executable (gated behind `ARK_UT`). This is the fastest way
to validate the SYCL kernels without a quantized checkpoint.

```bash
cd /home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel
source /opt/intel/oneapi/setvars.sh
export PATH="/home/yiliu7/workspace/venvs/ark/bin:$PATH"

cmake -S . -B xbuild_ut \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=icx \
  -DARK_XPU=ON -DARK_UT=ON -DARK_SYCL_TLA=ON \
  -DCMAKE_CXX_FLAGS="-I/home/yiliu7/workspace/venvs/ark/include" \
  -GNinja

# SYCL-TLA objects are memory-heavy (~5 GB/job). Keep -j modest (4 is safe on a 64 GB node).
cmake --build xbuild_ut --target test_ARK_XPU -j 4
```

First configure clones oneDNN, sycl-tla, and pybind11 via FetchContent (slow; needs network /
proxy). The SDPA `.cpp` instantiations dominate build time.

## Run the test

```bash
cd /home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel
source /opt/intel/oneapi/setvars.sh
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
./xbuild_ut/test_ARK_XPU
```

Expected int3 (S3) WOQ GEMV accuracy output (max_diff at machine epsilon, ~1e-7):

```
[woq_s3][accuracy] n=128 k=128 blk=128 max_diff=5.96046e-07
[woq_s3][accuracy] n=256 k=96  blk=32  max_diff=2.98023e-07
[woq_s3][accuracy] n=64  k=256 blk=64  max_diff=4.76837e-07
[woq_s3][gemm] m=4  n=128 k=128 blk=128 max_diff=3.57628e-07
[woq_s3][gemm] m=8  n=256 k=128 blk=128 max_diff=3.57628e-07
[woq_s3][gemm] m=16 n=64  k=256 blk=64  max_diff=1.07288e-06
```

The `[woq_s3][gemm]` lines exercise the m>1 (batched) int3 path: they drive `woq_gemm` with
`compute_type=S8` (matching the Python `cdt="int8"`) so the S3 fp-compute fallback is taken.

The run also prints int3 throughput benchmarks (warmup + timed iters, weight-bound GB/s for the
GEMV, FLOPs/s for both), with **int4 (S4) at the same shapes as a reference**: S4's GEMV uses the
dedicated S4 kernel and its m>1 GEMM takes the native int8-XMX path (`woq_s8`), whereas S3 m>1 falls
back to fp-dequant + fp GEMM — so the S4 numbers are the speed-of-light reference S3 is measured
against. Absolute numbers vary by node; on Battlemage G21 they look like:

```
[woq_s3][gemv_bench] bench_s3_gemv_n4096_k4096  n=4096 k=4096  blk=128 ms=0.0787 TFLOPS=0.43 GBps=86.6
[woq_s3][gemv_bench] bench_s3_gemv_n4096_k11008 n=4096 k=11008 blk=128 ms=0.1172 TFLOPS=0.77 GBps=156.3
[woq_s4][gemv_bench] bench_s4_gemv_n4096_k4096  n=4096 k=4096  blk=128 ms=0.0185 TFLOPS=1.81 GBps=481.8
[woq_s4][gemv_bench] bench_s4_gemv_n4096_k11008 n=4096 k=11008 blk=128 ms=0.0503 TFLOPS=1.79 GBps=476.1
[woq_s3][gemm_bench] bench_s3_gemm_m32_n4096_k4096   m=32  n=4096 k=4096  blk=128 ms=0.3546 TFLOPS=3.03
[woq_s3][gemm_bench] bench_s3_gemm_m128_n4096_k4096  m=128 n=4096 k=4096  blk=128 ms=0.5638 TFLOPS=7.62
[woq_s3][gemm_bench] bench_s3_gemm_m512_n4096_k11008 m=512 n=4096 k=11008 blk=128 ms=4.3748 TFLOPS=10.55
[woq_s4][gemm_bench] bench_s4_gemm_m32_n4096_k4096   m=32  n=4096 k=4096  blk=128 ms=0.1820 TFLOPS=5.90
[woq_s4][gemm_bench] bench_s4_gemm_m128_n4096_k4096  m=128 n=4096 k=4096  blk=128 ms=0.1768 TFLOPS=24.30
[woq_s4][gemm_bench] bench_s4_gemm_m512_n4096_k11008 m=512 n=4096 k=11008 blk=128 ms=0.7095 TFLOPS=65.07
```

GEMV is weight-bandwidth bound (the packed low-bit blob dominates I/O); the GEMM path's TFLOPS rises
with `m` as the dequant cost amortizes over more output rows. The large S3-vs-S4 GEMM gap at high
`m` is the cost of S3's fp fallback vs S4's int8-XMX path — the headroom a future int8-XMX S3 path
would recover.

The S3 GEMV reads the dense 3-bit blob (lane `L`'s three straddle words `3L,3L+1,3L+2`) through a
**coalesced staging load into SLM** before the per-lane straddle decode in `WeightS3T::gemv`
(`bestla/bestla/sycl/sycl_prologue_b.h`); decoding the strided words straight from global memory
compiles to a gather and is ~2x slower. S3 GEMV still trails the S4 reference because the 3-bit
straddle layout can't use S4's branchless nibble unpack and pays per-work-group SLM-barrier overhead
that only amortizes at large `k` (hence k=11008 lands far closer to S4 than k=4096).


To run **only** these benchmarks (skip every functional case and the multi-minute SDPA suite),
set `ARK_S3_BENCH=1` — it finishes in a few seconds. The benchmark lives in a header
(`wrapper/test/test_gemm.hpp`), so after editing shapes you must re-link the binary. Full
rebuild + bench cycle from the `auto_round_kernel` dir:

```bash
cd /home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel
source /opt/intel/oneapi/setvars.sh                       # SYCL runtime (needed to build AND run)
export PATH="/home/yiliu7/workspace/venvs/ark/bin:$PATH"  # cmake / ninja toolchain

# 1. Incremental rebuild (only test_main.cpp recompiles + link; ~seconds, no FetchContent).
cmake --build xbuild_ut --target test_ARK_XPU -j 4

# 2. Run the int3 (S3) + int4 (S4 reference) benchmarks only.
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ARK_S3_BENCH=1 ./xbuild_ut/test_ARK_XPU

# (optional) keep just the throughput lines:
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ARK_S3_BENCH=1 ./xbuild_ut/test_ARK_XPU 2>&1 | grep _bench
```

Edit the shapes (and weight types) in `TestGemm::run_s3_benchmarks()` (`wrapper/test/test_gemm.hpp`)
then repeat step 1+2 to sweep other `m`/`n`/`k`. Without `ARK_S3_BENCH`, a plain
`./xbuild_ut/test_ARK_XPU` runs the full suite (functional cases + these benchmarks + SDPA).

## Benchmark S2 vs S4 (GEMV decode + GEMM prefill)

The same `ARK_S3_BENCH=1` run also benches **int2 (S2)** against the **int4 (S4)** reference at both
m=1 (GEMV / decode) and m>1 (GEMM / prefill). Filter by regime:

```bash
cd /home/yiliu7/workspace/auto-round/auto_round_extension/ark/auto_round_kernel
source /opt/intel/oneapi/setvars.sh
export PATH="/home/yiliu7/workspace/venvs/ark/bin:$PATH"
cmake --build xbuild_ut --target test_ARK_XPU -j 4          # only if shapes/types were edited

# m=1 decode (S2/S3/S4):
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ARK_S3_BENCH=1 ./xbuild_ut/test_ARK_XPU 2>&1 | grep gemv_bench
# m>1 prefill (S2/S4):
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ARK_S3_BENCH=1 ./xbuild_ut/test_ARK_XPU 2>&1 | grep gemm_bench
```

Knobs (all in `run_s3_benchmarks()`, `wrapper/test/test_gemm.hpp`):

- **dtypes** — GEMM loop uses `gemm_wtypes[] = {S2, S4}`; add `{S3,"s3"}` for the fp-fallback comparison.
- **shapes** — `gemm_shapes[]` (m32/m128 @ k4096, m512 @ k11008); add e.g.
  `{"m1024_n4096_k11008",1024,4096,11008,128}` to probe past the S2/S4 crossover.
- **warmup/iters** — trailing `5, 20` on each `bench_woq_gemm` call; raise iters to cut noise.

Any shape/type edit needs the rebuild (step above) before re-running. Measured on Battlemage G21:

```
[woq_s2][gemv_bench] n4096_k4096   ms=0.0440 TFLOPS=0.76 GBps=107    [woq_s4] ms=0.0185 TFLOPS=1.81 GBps=481
[woq_s2][gemm_bench] m32 TFLOPS=4.95 GBps=21.7   m128 20.3/22.3   m512 70.8/19.5
[woq_s4][gemm_bench] m32 TFLOPS=7.97 GBps=66.2   m128 32.8/68.1   m512 65.0/33.7
```

Reading (TFLOPS is the cross-dtype metric; GBps over the blob is NOT comparable across bit-widths —
a fatter blob shows higher GBps at equal speed):

- **Decode (GEMV, m=1)** is bandwidth/latency-bound: S4 reaches 481 GBps (66% of the 725 GB/s bus)
  and wins 2.4x; S2/S3 sit at ~15% of the bus (occupancy-bound, the gap the optimization journey chased).
- **Prefill (GEMM, m>1)** is **compute-bound**, not bandwidth-bound: every line is at 3-9% of the bus,
  so XMX throughput gates it. S4 wins small/medium m (m32/m128, ~1.6x); **S2 edges S4 at large m**
  (m512: 70.8 vs 65.0 TFLOPS, ~9%, reproducible). The GBps rules out bus saturation as the cause; the
  most likely mechanism is S2's cheaper dequant (half the weight bytes to unpack), but this is not yet
  profiled. Crossover is ~m=256-512.
- S2 is NOT in S3's fp-dequant fallback — it is 2.7-6.7x faster than S3 at GEMM.



The binary exits 0 when all GEMM / WOQ / SDPA cases pass. The SDPA benchmarks at the end
(`bench_*` with 4096/8192 seq len) run for a few minutes — this is expected.

> `test_woq_s3(n, k, blocksize)` requires `k % 32 == 0`, `k % blocksize == 0`, and
> `blocksize % 32 == 0`. The int3 kernel itself assumes each 32-element group shares one scale
> (`blocksize % 32 == 0`).

## Notes

- Build tip: stdout from a long `cmake --build`/test run is line-buffered through pipes; if you
  pipe through `grep`/`tail` you may see nothing until exit. Redirect to a file with
  `stdbuf -oL -eL ... > run.log 2>&1` to watch progress live.
- int3 status: **m=1 GEMV + m>1 GEMM** both supported. m=1 uses the dedicated S3 GEMV kernel; m>1
  (batched / multi-token prefill) runs through an fp-dequant + DNNL fp GEMM fallback in `woq_gemm`
  (the dense 3-bit blob is decoded by `WeightS3T::dequant`, the same `unpack32` decode as GEMV, so
  it is numerically exact). The int8-XMX m>1 path (mirroring S4's `dequantS8`) is a future
  optimization. Asymmetric int3 is unsupported (XPU is symmetric-only).
