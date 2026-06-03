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
