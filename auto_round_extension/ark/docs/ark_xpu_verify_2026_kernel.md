# Verify m>1 joint_matrix on released oneAPI 2026 (b70 / bmg_g31)

Standalone, copy-paste verification that the **apt-released** oneAPI 2026 DPC++ runtime
makes ARK's m>1 int8-XMX WOQ GEMM (`IGemmDQCore` `joint_matrix`) work on the **B70 /
Battlemage G31** node (`root@10.239.98.43`, PCI `0xe223`, arch `intel_gpu_bmg_g31`).

This is the **formal fix** for the failure documented in `ark_xpu_joint_matrix_finding.md`:

```
RuntimeError: no matrix hardware on the target device, joint_matrix is not supported
```

**Root cause (one-line):** oneAPI 2025.3.2's `libsycl.so.8` matrix `supported_archs[]`
table predates `bmg_g31`, so `has(aspect::ext_intel_matrix)` returns NO even though
`architecture_is(bmg_g31)` returns YES. The released **2026.0.0** runtime (`libsycl.so.9`)
carries the entry → aspect YES → the kernel runs. Driver (L0 1.15.37833) and IGC (2.32.7)
are NOT the determinant — they were held fixed across the FAIL→PASS flip. `DPCPP_SYCL_TARGET`
does **not** gate this (it only feeds the `-Xs -device` JIT hint). See the finding doc.

> **Scope:** kernel UT only. The torch end-to-end path is out of scope — torch 2.11+xpu
> hard-pins `intel-sycl-rt==2025.3.2` (`.so.8`) and shares the process, so it cannot load
> `.so.9` without a torch rebuild. This verifies the **torch-free** kernel via the
> standalone harness that calls the production `IGemmDQCore` launch directly.

## Prerequisites on the node

1. **Released oneAPI 2026 compiler** at `/opt/intel/oneapi/compiler/2026.0`. Install the
   **versioned** package (NOT the umbrella `intel-basekit` / `intel-oneapi-compiler-dpcpp-cpp`,
   which flips `compiler/latest`→2026.0 and would disturb torch-xpu's 2025.3 pin):

   ```bash
   # one-time, requires apt (shared node — get user go-ahead before installing)
   apt-get install -y intel-oneapi-compiler-dpcpp-cpp-2026.0=2026.0.0-947
   /opt/intel/oneapi/compiler/2026.0/bin/icpx --version   # expect 2026.0.0 (2026.0.0.20260331)
   ```

   > NOTE (observed): even the versioned install ran a postinst trigger that flipped
   > `compiler/latest` 2025.3→2026.0. If you need `latest`→2025.3 for torch, restore it:
   > `ln -sfn 2025.3 /opt/intel/oneapi/compiler/latest`. 2026.0 stays usable via its
   > explicit path; the verify script below always calls `compiler/2026.0/bin/icpx`
   > directly, so it never depends on `latest`.

2. **Probe + harness sources on the node** (byte-identical copies live in this `docs/` dir):
   - `/root/arch_resolve.cpp` — the aspect gate-check (`docs/arch_resolve.cpp`)
   - `/root/ark_jm_harness.cpp` — the real `IGemmDQCore` joint_matrix launch (`docs/ark_jm_harness.cpp`)
   - `/root/auto-round` checked out at `feat/ark-xpu-int3-woq-gemm` (provides
     `auto_round_kernel/bestla` includes)

   If they are missing, copy them over (from this docs dir):

   ```bash
   python3 /tmp/scp_run.py auto_round_extension/ark/docs/arch_resolve.cpp   /root/arch_resolve.cpp
   python3 /tmp/scp_run.py auto_round_extension/ark/docs/ark_jm_harness.cpp /root/ark_jm_harness.cpp
   ```

## The runtime layering (why LD_LIBRARY_PATH has three dirs)

The released 2026 `compiler/2026.0/lib` tree provides `libsycl.so.9`, `libur_loader.so.0`,
and the Level-Zero **UR adapter** — but NOT the Level-Zero loader/driver themselves. Those
are the **system** ones, and `hwloc` comes from the torch venv. So the run needs, in order:

| Source dir                              | Provides                                        |
|-----------------------------------------|-------------------------------------------------|
| `/opt/intel/oneapi/compiler/2026.0/lib` | `libsycl.so.9`, `libur_loader.so.0`, L0 UR adapter |
| `/usr/lib/x86_64-linux-gnu`             | `libze_loader.so.1`, `libze_intel_gpu.so.1`     |
| `/root/torch-xpu-setup/.venv/lib`       | `libhwloc.so.15`                                |

Without the system L0 dir, UR finds **0 adapters** → `No device of requested type 'gpu'
available`. (Diagnose with `SYCL_UR_TRACE=2` → `urAdapterGet → 0 adapters`.)

## Run the verification

Copy the script over and run it (it is self-contained — sets the layering, builds both
binaries with the 2026 `icpx`, runs both):

```bash
# from a machine that can reach the node (this docs/ dir has verify_released_2026.sh):
python3 /tmp/scp_run.py auto_round_extension/ark/docs/verify_released_2026.sh /root/verify_released_2026.sh
python3 /tmp/ssh_run.py 'bash /root/verify_released_2026.sh'
```

Or, already on the node:

```bash
bash /root/verify_released_2026.sh
```

The script (`docs/verify_released_2026.sh`) does exactly:

```bash
ONEAPI2026=/opt/intel/oneapi/compiler/2026.0
ICPX=$ONEAPI2026/bin/icpx
SRCROOT=/root/auto-round/auto_round_extension/ark/auto_round_kernel

export LD_LIBRARY_PATH=$ONEAPI2026/lib:/usr/lib/x86_64-linux-gnu:/root/torch-xpu-setup/.venv/lib

# 1) aspect gate-check — want has(ext_intel_matrix)=YES
$ICPX -fsycl -O2 /root/arch_resolve.cpp -o /root/arch_resolve_rel2026
ONEAPI_DEVICE_SELECTOR=level_zero:gpu /root/arch_resolve_rel2026

# 2) real ARK joint_matrix kernel — want PASS, exit=0
$ICPX -fsycl -fno-sycl-instrument-device-code -O2 -std=c++20 -DBTLA_SYCL \
  -I$SRCROOT/bestla -fsycl-targets=spir64 \
  /root/ark_jm_harness.cpp -o /root/ark_jm_harness_rel2026
ONEAPI_DEVICE_SELECTOR=level_zero:gpu /root/ark_jm_harness_rel2026 128 128 128
```

## Expected output (PASS)

**Gate-check** (`arch_resolve_rel2026`) — the decisive line is the last one:

```
name: Intel(R) Arc(TM) Pro B70 Graphics
architecture enum int = 21483225088 (hex 0x500800000)
  is intel_gpu_bmg_g21 ? no
  is intel_gpu_bmg_g31 ? YES
has(ext_intel_matrix) = YES        <-- was NO on 2025.3.2 .so.8; YES is the fix
```

**Harness** (`ark_jm_harness_rel2026 128 128 128`) — links `.so.9`, kernel runs:

```
  SONAME: libsycl.so.9
  which libsycl resolves:
    libsycl.so.9 => /opt/intel/oneapi/compiler/2026.0/lib/libsycl.so.9
  --- run: want PASS, exit=0 ---
  [harness] ARK IGemmDQCore (joint_matrix) m=128 n=128 k=128
  [harness] device: Intel(R) Arc(TM) Pro B70 Graphics
  [harness] launching joint_matrix kernel...
  [harness] PASS: joint_matrix kernel launched and completed.
  exit=0
```

(On the broken 2025.3 `.so.8` runtime the same harness instead prints
`[harness] FAIL: sycl::exception: ... joint_matrix is not supported` and `exit=2`.)

Both signals together (`has(ext_intel_matrix)=YES` **and** harness `exit=0` linking
`libsycl.so.9`) confirm the released 2026 runtime fixes the m>1 joint_matrix path on b70.

## If it FAILS

- `has(ext_intel_matrix) = NO` → you built against a `.so.8` runtime, not 2026. Check
  `ldd /root/arch_resolve_rel2026 | grep sycl` resolves to `compiler/2026.0/lib/libsycl.so.9`.
- `No device of requested type 'gpu' available` → the system L0 dir is missing from
  `LD_LIBRARY_PATH`; confirm `/usr/lib/x86_64-linux-gnu/libze_loader.so.1` exists.
- `joint_matrix is not supported` thrown at runtime → the runtime in the SONAME chain is
  still the 2025.3 `.so.8`; a stray `compiler/latest` or venv path is shadowing 2026.

## Building the full `test_ARK_XPU` UT against 2026 (not just the harness)

Everything above verifies the **kernel** via the standalone harness. To run the **full
`test_ARK_XPU` unit-test executable** against the 2026 runtime, build it with 2026's `icx`
and its bundled `IntelSYCL` cmake module.

> **STATUS: VALIDATED end-to-end on b70 (2026-06-06).** The full `test_ARK_XPU` was
> configured + built (549/549, exit=0) with 2026's `icx` and run on `libsycl.so.9`: the
> `TestGemm` cases `test_s8s8` (`igemm_s8s8`) and `test_woqs8` (`woq_s8` — the m>1
> int8-XMX `joint_matrix` path) both completed with **no `joint_matrix is not supported`
> throw** and **exit=0**. This upgrades the harness-only proof to a real UT proof. The
> two non-obvious specifics found while doing it — the OpenCL configure flags and the
> `TestGemm`-is-commented-out gotcha — are baked into the steps below.

What changed since the harness-only session, confirmed read-only on b70 (2026-06-06):

- **The old blocker is gone.** The released 2026 ships the cmake module the nightly lacked:
  `/opt/intel/oneapi/compiler/2026.0/lib/cmake/IntelSYCL/IntelSYCLConfig.cmake`. (The
  nightly clang had none, so `CMakeLists.txt:55`'s `find_package(IntelSYCL REQUIRED)` could
  only resolve to the 2025.3 module → broke back to `.so.8`. That forced the harness route.)
- **MKL-RNG header** (`oneapi/mkl/rng/device.hpp`, the known fresh-node blocker) is at
  `/opt/mkl-include` → pass `-I/opt/mkl-include`.
- **Node defaults that must be overridden:** the checkout sits on `main` (check out the
  branch you intend to test), and `compiler/latest`→2025.3 (restored for torch), so 2026
  must be selected by **explicit path**, never via `latest`.

### Build + run

```bash
# on the node: root@10.239.98.43
cd /root/auto-round/auto_round_extension/ark/auto_round_kernel
git -C /root/auto-round checkout feat/ark-xpu-int3-woq-gemm   # the branch under test (node is on main)

ONEAPI2026=/opt/intel/oneapi/compiler/2026.0
export http_proxy=http://proxy.ims.intel.com:911 https_proxy=http://proxy.ims.intel.com:911
export no_proxy="intel.com,.intel.com,localhost,127.0.0.1"

# Select 2026's icx + its IntelSYCL module explicitly; MKL-RNG header via -I; target b70 silicon.
# oneDNN's find_package(OpenCL) needs OpenCL_LIBRARY + OpenCL_INCLUDE_DIR — both live in the
# 2026 tree; without them configure dies at dnnl-src/CMakeLists.txt with "Could NOT find OpenCL".
cmake -S . -B xbuild_b70_2026 -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=$ONEAPI2026/bin/icx \
  -DIntelSYCL_DIR=$ONEAPI2026/lib/cmake/IntelSYCL \
  -DARK_XPU=ON -DARK_UT=ON -DARK_SYCL_TLA=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g31 \
  -DOpenCL_LIBRARY=$ONEAPI2026/lib/libOpenCL.so \
  -DOpenCL_INCLUDE_DIR=$ONEAPI2026/include \
  -DCMAKE_CXX_FLAGS="-I/opt/mkl-include"

# SYCL-TLA / SDPA objects are memory-heavy (~5 GB/job). Keep -j modest. First configure
# clones oneDNN / sycl-tla / pybind11 via FetchContent (slow; needs the proxy above).
cmake --build xbuild_b70_2026 --target test_ARK_XPU -j 4

# The binary links libsycl.so.9 → apply the SAME 3-dir runtime layering as the harness.
export LD_LIBRARY_PATH=$ONEAPI2026/lib:/usr/lib/x86_64-linux-gnu:/root/torch-xpu-setup/.venv/lib
ONEAPI_DEVICE_SELECTOR=level_zero:gpu ./xbuild_b70_2026/test_ARK_XPU
```

> **GOTCHA — `TestGemm` is commented out on `main`.** `wrapper/test/test_main.cpp` ships
> with only `TestSDPA` active; `TestGemm` (and `TestQuant`) are commented out. A plain
> build+run therefore exercises **only SDPA** and exits 0 **without ever touching the
> joint_matrix path** — a misleading green. To prove the m>1 int8-XMX case you must enable
> `TestGemm` before building (back the file up first — shared node):
>
> ```bash
> cd /root/auto-round/auto_round_extension/ark/auto_round_kernel
> cp wrapper/test/test_main.cpp wrapper/test/test_main.cpp.bak
> sed -i 's|^  // TestGemm test_gemm;|  TestGemm test_gemm;|' wrapper/test/test_main.cpp
> # (optional) comment out TestSDPA to skip the multi-minute SDPA benches — not needed for the proof.
> ```
>
> `TestGemm`'s constructor (`wrapper/test/test_gemm.hpp`) runs `test_s8s8<float>(128,128,128)`
> → `igemm_s8s8`, and `test_woqs8<float>(128,128,128)` → `woq_s8` (line 59) — the m>1
> int8-XMX `joint_matrix` path. Revert with the `.bak` when done.

### Success signal

The decisive case is the **S4 / int8-XMX m>1 GEMM** — the direct `DnnlWrapper::woq_s8`
call (`wrapper/test/test_gemm.hpp:59`), which is what actually reaches `joint_matrix`. It
must complete with **no `joint_matrix is not supported` throw**. With `TestGemm` enabled,
the validated run (2026-06-06, b70) printed:

```
=== SONAME / libsycl resolution (want libsycl.so.9 from compiler/2026.0) ===
  NEEDED libsycl.so.9
    libsycl.so.9 => /opt/intel/oneapi/compiler/2026.0/lib/libsycl.so.9
    libur_loader.so.0 => /opt/intel/oneapi/compiler/2026.0/lib/libur_loader.so.0
Welcome to ARK TEST
test:L18
test:L18
test:L18
test_s8s8:L35      <- igemm_s8s8 completed (no joint_matrix throw)
test_woqs8:L53     <- woq_s8 m>1 int8-XMX joint_matrix completed
=== test_ARK_XPU exit=0 ===
```

On the broken 2025.3 `.so.8` runtime the same `test_woqs8` instead throws
`joint_matrix is not supported` and the process aborts non-zero.

> The `[woq_s3][gemm]` lines (if you also run the S3 suite) are the **fp-dequant fallback**
> (`compute_type=S8`) and pass regardless of the matrix aspect — they are NOT sufficient
> proof on their own. Confirm the **S4/int8-XMX** `test_woqs8` case specifically.

### Notes / risks

- **Heavy build on a shared node** (multi-minute, ~5 GB/job) — hence `-j 4`.
- **Harmless link-time warnings:** `libOpenCL.so` triggers `rpath-link` warnings for
  `libsvml/libirng/libimf/libintlc.so.5` at the final link. They resolve at runtime via
  the 3-dir `LD_LIBRARY_PATH` (2026 `lib` first) — ignore them; the binary links fine.
- If `find_package(IntelSYCL)` still resolves to 2025.3 despite `-DIntelSYCL_DIR=...`,
  prepend `2026.0` to the cmake search: add `-DCMAKE_PREFIX_PATH=$ONEAPI2026`.
- Do NOT rely on `source $ONEAPI2026/env/vars.sh` for the *run* — it sets compile-time
  vars but the run still needs the explicit 3-dir `LD_LIBRARY_PATH` layering above
  (2026 lib provides `libsycl.so.9` + UR adapter, but the L0 loader/driver are the system
  ones and `hwloc` is the venv's — see the layering table earlier in this doc).

## Related

- `ark_xpu_joint_matrix_finding.md` — full investigation, CORRECTION #3 (libsycl is the
  gate), and the `DPCPP_SYCL_TARGET` A/B verdict.
- `ark_xpu_build_bkc.md` — per-node SYCL target note (bmg_g21 vs bmg_g31).
- Memory: `ark-xpu-joint-matrix-libsycl-gate`.
