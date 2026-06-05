# ARK XPU Fresh-Node Build BKC (prerequisites the main BKC assumes)

Companion to `ark_xpu_build_bkc.md`. The main BKC assumes a node that already has Python dev
headers and a complete oneAPI MKL. On a fresh node those two are often missing and the
`setup.py` production build fails partway through. This doc captures the two blockers and their
fixes so the next bring-up doesn't rediscover them.

Validated on node `b70-pc6` (Battlemage G21, Debian 13 trixie, 32 GB RAM, 12 cores),
oneAPI `icx` 2025.3.3, PyTorch 2.11.0+xpu, Python 3.13.5, building via:

```bash
cd <repo>/auto_round_extension/ark
source /opt/intel/oneapi/setvars.sh
export PATH="<xpu-venv>/bin:$PATH"
python setup.py build_ext
```

`setup.py` builds **two** extensions in sequence: a CPU `.so` (host `c++`, Unix Makefiles) then
an XPU `.so` (`icx`, `-DARK_XPU=ON -DARK_SYCL_TLA=ON`). Each runs its own FetchContent clone +
build of oneDNN, so a full from-scratch build is ~60–90 min on a 12-core/32 GB node (the XPU
SYCL-TLA SDPA instantiations are `-j 1` here — see "Job count" below).

---

## Blocker 1 — missing Python development headers

**Symptom** (during the *first*, CPU cmake configure, inside pybind11):

```
CMake Error at .../FindPackageHandleStandardArgs.cmake:290 (message):
  Could NOT find Python (missing: Python_INCLUDE_DIRS Development.Module)
  (found suitable version "3.13.5", minimum required is "3.8")
Call Stack (most recent call first):
  .../pybind11NewTools.cmake:54 (find_package)
```

**Root cause:** the XPU venv's interpreter is the system `/usr/bin/python3.13` (`base_prefix=/usr`),
and `python3.13-dev` is not installed — `/usr/include/python3.13/Python.h` is absent. pybind11
requires the `Development.Module` component, which needs the headers. A venv alone never provides
them; they come from the system package matching the base interpreter.

**Diagnose:**

```bash
<xpu-venv>/bin/python - <<'PY'
import sysconfig, os
inc = sysconfig.get_config_var("INCLUDEPY")
print("INCLUDEPY:", inc, "Python.h exists:", os.path.exists(os.path.join(inc, "Python.h")))
PY
```

**Fix (Debian/Ubuntu)** — install the dev package whose version matches the runtime exactly
(here `libpython3.13-stdlib` was `3.13.5-2+deb13u2`, so install the same):

```bash
# proxy only if behind the Intel firewall
export http_proxy=http://proxy.ims.intel.com:911 https_proxy=http://proxy.ims.intel.com:911
apt-get update
apt-get install -y libpython3.13-dev python3.13-dev
```

Confirm `/usr/include/python3.13/Python.h` exists. The second (XPU) configure then reports
`Found Python: ... found components: Interpreter Development.Module Development.Embed`.

> If `apt-cache policy python3-dev` shows `Candidate: (none)`, the apt indexes are stale — run
> `apt-get update` first (the `cdrom:` source line in `/etc/apt/sources.list` errors harmlessly;
> the deb.debian.org sources still refresh).

---

## Blocker 2 — missing MKL header `oneapi/mkl/rng/device.hpp`

**Symptom** (during the *second*, XPU build, ~95%, compiling the SYCL-TLA SDPA path):

```
.../sycl_tla-src/tools/util/include/cutlass/util/reference/device/sycl_tensor_fill.h:41:10:
  fatal error: 'oneapi/mkl/rng/device.hpp' file not found
   41 | #include <oneapi/mkl/rng/device.hpp>
```

This is the same gotcha the main BKC calls out, but its fix (`-I<venv>/include`) assumes the venv
ships the MKL headers. On a fresh node neither source exists:

- The **host oneAPI** at `/opt/intel/oneapi` has `compiler/`, `tbb/`, `dpl/` … but **no `mkl/`
  component** — `setvars.sh` leaves `MKLROOT` empty.
- The **pip MKL** packages in the venv (`mkl`, `onemkl_sycl_rng`, …) are **runtime-only**: their
  RECORD lists no `.hpp`, only shared libs. (`mkl-include` would provide headers, but the venv may
  have no `pip`.)

**Fix A — proper, if you can install packages:** install the matching oneAPI MKL devel so the
headers land under `/opt/intel/oneapi/mkl/<ver>/include` and `setvars.sh` sets `MKLROOT`:

```bash
apt-get install -y intel-oneapi-mkl-devel-2025.3   # version must match the sourced compiler
```

**Fix B — header-only workaround (what was used on `b70-pc6`):** stage just the MKL include tree
to a stable path and put it on `CPATH` (which `icx` honours, and which works even though
`setup.py` does **not** forward `CMAKE_CXX_FLAGS`):

```bash
# A complete MKL 2025.3 include tree existed in a docker overlay on this node; any matching
# oneapi/mkl/<ver>/include works. Copy it somewhere stable:
mkdir -p /opt/mkl-include
cp -r <some>/oneapi/mkl/2025.3/include/* /opt/mkl-include/
ls /opt/mkl-include/oneapi/mkl/rng/device.hpp   # sanity check

# Then build with CPATH set (add to the build script, after sourcing setvars):
export CPATH="/opt/mkl-include:$CPATH"
python setup.py build_ext
```

`CPATH` is the key insight: the main BKC's `-DCMAKE_CXX_FLAGS=-I...` route does nothing through
`setup.py` (it never forwards the flag), but `CPATH` is read by the compiler directly, so it
reaches every sub-build including the SYCL-TLA instantiations.

> Header version must match the sourced compiler/MKL ABI. 2025.3 headers + `icx` 2025.3 is the
> validated pairing. Do **not** mix a 2021.x apt MKL with a 2025.x compiler.

---

## Blocker 3 — stale CMake cache: `BTLA_SYCL` desyncs from `ARK_XPU`

**Symptom** (during the XPU compile of `ark.cpp`):

```
.../wrapper/include/dnnl_wrapper.hpp:123:27: error:
  no member named 'sycl_prologue_a' in namespace 'bestla'; did you mean 'prologue_a'?
  123 |       using Pro = bestla::sycl_prologue_a::ActivationBase<T>;
```

**Root cause:** two CMake switches must agree but can drift apart in a reused build dir:

- `dnnl_wrapper.hpp` references `bestla::sycl_prologue_a` under `#if ARK_XPU`
  (`auto_round_kernel/CMakeLists.txt` sets `ARK_TYPE=ARK_XPU` when `ARK_XPU=ON`).
- That namespace only exists when bestla is compiled with `BTLA_SYCL` — every bestla SYCL header
  (`sycl_prologue_a.h`, `sycl_prologue_b.h`, `sycl_gemm.h`, …) is wrapped in `#ifdef BTLA_SYCL`.

`auto_round_kernel/CMakeLists.txt` enables it with `set(BTLA_SYCL ON CACHE BOOL ...)` inside
`if(ARK_XPU)`. The **`CACHE` keyword makes this a no-op if `BTLA_SYCL` already exists in
`CMakeCache.txt`** — and bestla's own `CMakeLists.txt` declares `option(BTLA_SYCL ... OFF)`. So if
the `xbuild/` cache ever recorded `BTLA_SYCL=OFF` (e.g. a configure where `ARK_XPU` wasn't yet
set), re-running configure with `-DARK_XPU=ON` leaves the stale `OFF` in place. Result:
`ARK_XPU=ON` (wrapper wants the SYCL namespace) but `BTLA_SYCL=OFF` (bestla compiled it out).

A fresh `xbuild/` works; the desync only appears in a **reused/partially-configured** dir — which
is exactly the state you land in after iterating on the build.

**Diagnose** — the cache tells you directly:

```bash
grep -iE "ARK_XPU|BTLA_SYCL" xbuild/CMakeCache.txt
# BAD:  ARK_XPU:BOOL=ON  +  BTLA_SYCL:BOOL=OFF   <- the desync
# GOOD: ARK_XPU:BOOL=ON  +  BTLA_SYCL:BOOL=ON
```

**Fix** — a flag can't override a cached value; you must drop the cache entry and reconfigure
(this keeps the expensive `_deps/` oneDNN + sycl-tla checkouts, only the configure state is reset):

```bash
rm -f  xbuild/CMakeCache.txt
rm -rf xbuild/CMakeFiles
cmake -S auto_round_kernel -B xbuild \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icx \
  -DARK_XPU=ON -DARK_SYCL_TLA=ON
grep BTLA_SYCL xbuild/CMakeCache.txt          # must now read BTLA_SYCL:BOOL=ON
cmake --build xbuild -j 2                      # -j 2 — see Job count below
```

---

## Job count for the XPU / SYCL-TLA build

`setup.py:get_sycl_tla_job_count()` caps XPU parallelism at `memory_gb // 16` (~5 GB per
SYCL-TLA object). On this 32 GB node that is **`-j 1`** — the SDPA `prefill_*` / `decode_*`
instantiations compile serially, which is the bulk of the wall-clock. This is expected, not a
hang; watch progress with `tail -f` on the build log. (The 64 GB `inc101` node in the main BKC
gets `-j 4`.)

> **Do not override this with a high `-j` when driving cmake directly.** On `b70-pc6`, running
> `cmake --build xbuild -j 16` during the `generated/sdpa/*` stage exhausted all 32 GB and the
> **kernel OOM-killer rebooted the box** (`uptime` showed it back at 2 min; `/tmp` was wiped).
> Each SDPA `.cpp` peaks ~5 GB, so 16 parallel jobs ≈ 80 GB demand. The early oneDNN `.cpp`
> files are light and tolerate a high `-j`, but the SDPA stage does not — there is no per-stage
> throttle, so size `-j` for the **worst** stage: **`-j 2` is the safe blanket value on 32 GB**
> (`-j 1` is what `setup.py` itself picks). Watch `free -g` as the build crosses ~95%.
>
> `setup.py` has no `MAX_JOBS` env hook — the job count is computed internally (CPU build:
> `cpu_count()//2`; XPU build: `min(that, mem_gb//16)`). To control it, either build via cmake
> directly with your own `-j`, or patch `setup.py` to read `os.environ.get("MAX_JOBS")`.

---

## Installing the built artifacts into the package

There are two distinct consumers, and they need different things:

1. **Direct `import auto_round_kernel`** (running from the source tree) only needs the `.so` files
   in the package dir. `setup.py build_ext` emits them under `build/lib.linux-*/auto_round_kernel/`,
   but the package imports from its **own** directory (`auto_round_kernel/__init__.py` does
   `from . import auto_round_kernel_cpu` / `auto_round_kernel_xpu`). A bare `build_ext` (or a
   direct `cmake --build`) leaves the package dir without the `.so`, so copy them in:

   ```bash
   cp build/lib.linux-*/auto_round_kernel/auto_round_kernel_{cpu,xpu}*.so auto_round_kernel/
   # (direct cmake build:  cp xbuild/auto_round_kernel_xpu*.so auto_round_kernel/)
   ```

2. **AutoRound's inference backend** additionally requires the **`auto-round-lib` dist metadata**
   to be installed. `auto_round/inference/backend.py` gates every ARK backend on
   `require_version("auto-round-lib")` (and then does a top-level `import auto_round_kernel`). With
   only the copied `.so` and no installed distribution you get, at model-load time:

   ```
   ERROR backend.py: Please install auto-round-lib for CPU/XPU, e.g.: pip install "auto-round-lib"
   ```

   even though the kernel imports fine by hand. The copied `.so` satisfies the import but **not**
   the metadata check — you must install the distribution.

### Installing `auto-round-lib` without a from-scratch rebuild

> **Do NOT use `uv pip install .` (non-editable) to install the kernel.** PEP 517 copies the
> source tree into a temp build dir, where ark's `setup.py` recomputes `BUILD_DIR`/`XBUILD_DIR`
> relative to that copy — so your existing `build/`/`xbuild/` (with all the compiled oneDNN +
> SDPA objects) are **invisible**, and it reconfigures + rebuilds oneDNN from scratch (~60–90 min,
> and the OOM risk from "Job count" above). The tell-tale sign is build output restarting at
> `_deps/dnnl-build/...`.

Build the wheel **in-tree** instead — `python setup.py bdist_wheel` runs in the source tree, so
`XBUILD_DIR` points at your existing `xbuild/` and cmake is incremental (seconds: reconfigure +
relink check, no recompile). Then install the wheel:

```bash
cd <repo>/auto_round_extension/ark
source /opt/intel/oneapi/setvars.sh
export PATH="<xpu-venv>/bin:$PATH"
export CPATH="/opt/mkl-include:$CPATH"     # Blocker 2 fix (still needed by the build_ext step)

python setup.py bdist_wheel                 # reuses build/ + xbuild/ -> fast
uv pip install --no-build-isolation dist/*.whl
```

This installs both `.so` files **and** the `auto-round-lib` metadata into site-packages.
`--no-build-isolation` lets the `build_ext` step see the venv's `+xpu` torch instead of pulling a
fresh one.

> If `bdist_wheel` unexpectedly starts recompiling oneDNN (output at `_deps/dnnl-build/...`), a
> cache got touched — stop and re-check `xbuild/CMakeCache.txt` (see Blocker 3) before continuing.

## Verify

Source the oneAPI runtime (the XPU `.so` needs the SYCL libs at import) and load both modules:

```bash
source /opt/intel/oneapi/setvars.sh
cd <repo>
<xpu-venv>/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "xpu", torch.xpu.is_available())
import auto_round_extension.ark.auto_round_kernel as ark
inst = ark._ark_instance()
print("cpu_lib:", inst.cpu_lib)
print("xpu_lib:", inst.xpu_lib)
PY
```

Expected: `xpu True`, and both `cpu_lib` / `xpu_lib` print as loaded `...so` modules (not `None`).
A `None` lib means that `.so` failed to import — re-check the copy step and that `setvars.sh` was
sourced.

If you installed the distribution (for the AutoRound backend), also confirm the two checks
`backend.py` performs — both must succeed, or the ARK backend is silently filtered out:

```bash
source /opt/intel/oneapi/setvars.sh
python -c "import auto_round_kernel; print(auto_round_kernel.__file__)"          # must NOT be None
python -c "import importlib.metadata as m; print(m.version('auto-round-lib'))"    # prints a version
```

A `__file__` of `None` means `import auto_round_kernel` resolved to an empty **namespace** dir in
site-packages (e.g. a leftover `auto_round_kernel/` holding only a `.bak`) shadowing the real
package — remove that dir and reinstall. `PackageNotFoundError` for `auto-round-lib` means only
the `.so` was copied, never the distribution — run the `bdist_wheel` + `uv pip install` step above.
