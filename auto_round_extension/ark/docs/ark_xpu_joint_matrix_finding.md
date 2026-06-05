# ARK XPU m>1 WOQ GEMM Failure — Root Cause Analysis

> **CORRECTION #3 (2026-06-05, DEFINITIVE — supersedes #2, #1, and the original).**
> A controlled single-machine, single-variable experiment on `b70-pc6`
> (root@10.239.98.43) **proves the determinant is the `libsycl` (DPC++ runtime)
> version**, not the GPU/driver (#2's claim) and not IGC (#1's claim). Holding the
> GPU, kernel driver (L0 `1.15.37833`), AND IGC (`2.32.7`) all FIXED, and swapping
> ONLY `libsycl` from oneAPI 2025.3.2 (`libsycl.so.8`) to an intel/llvm nightly
> (`2026-06-04`, `libsycl.so.9`) flips the result from FAIL to PASS.
>
> ## The experiment (one machine, ONLY libsycl changed)
> The SAME `arch_resolve` and `jm_probe` sources (the int8 M8×N16×K32 DPAS tile ARK
> uses) were rebuilt against each libsycl and run on the SAME physical GPU:
>
> | libsycl runtime | GPU/driver | IGC | arch is `bmg_g31` | `has(ext_intel_matrix)` | jm_probe |
> |---|---|---|---|---|---|
> | oneAPI 2025.3.2 (`.so.8`, `20260112`) | 0xe223 / 1.15.37833 | 2.32.7 | **YES** | **NO** | ❌ FAIL |
> | intel/llvm nightly-2026-06-04 (`.so.9`) | 0xe223 / 1.15.37833 | 2.32.7 | **YES** | **YES** | ✅ **JOINT_MATRIX_OK** |
>
> Driver and IGC are byte-identical across rows — only `libsycl` differs.
>
> ## The smoking gun — a self-contradiction inside 2025.3 libsycl
> On 2025.3, the runtime simultaneously reports:
> - `ext_oneapi_architecture_is(intel_gpu_bmg_g31)` → **YES**
> - `has(aspect::ext_intel_matrix)` → **NO**
>
> But upstream `device_impl.hpp` `CASE(ext_intel_matrix)` returns
> `any_of(supported_archs, architecture_is)` where `supported_archs[]` INCLUDES
> `arch::intel_gpu_bmg_g31`. If arch-is-bmg_g31 is YES, matrix MUST be YES. 2025.3
> returns NO → its compiled `supported_archs[]` matrix table PRE-DATES the
> `bmg_g31` entry. The nightly's table includes it → YES. (The `bmg_g31` *string*
> is present in BOTH `.so` files — it backs `architecture_is` — so `strings | grep`
> does NOT distinguish them; only the runtime `has()` query does.)
>
> ## Why #1 (IGC) and #2 (driver/silicon) were both wrong
> - **#2 (this device lacks matrix HW / driver gap):** disproven — same driver,
>   newer libsycl → matrix aspect appears and the DPAS kernel runs. The hardware and
>   kernel driver were always capable; libsycl just refused to enumerate it.
> - **#1 (IGC too old):** disproven as the *root* determinant — IGC stayed 2.32.7
>   across the PASS row. IGC 2.34 "fixing" it on another machine was a co-varying
>   proxy: that machine also had a newer libsycl. libsycl is the gate; once it
>   advertises the aspect, IGC 2.32.7 lowers the tile fine.
>
> ## THE FIX (definitive)
> **Upgrade the DPC++/libsycl the ARK runtime loads to one whose matrix
> `supported_archs[]` includes `bmg_g31`** (intel/llvm nightly ≥ 2026-06-04, or a
> oneAPI release that carries that table entry). No driver, IGC, or hardware change
> is required on b70. Caveat: the nightly bumps the SONAME (`libsycl.so.8` →
> `.so.9`), so an `LD_LIBRARY_PATH` drop-in over the existing `.so.8` binary will
> NOT bind — the consuming binary (torch-xpu / the ARK `.so`) must be built/linked
> against the `.so.9` runtime, or use a oneAPI point release that keeps `.so.8` but
> ships the updated table. ARK-side bypasses (oneDNN fp GEMM for m>1, or inline-dpas
> `XE_DPAS_TT`) remain valid escapes if the runtime can't be upgraded.
>
> ## FORMAL SOLUTION VERIFIED (2026-06-05) — released oneAPI 2026, not just the nightly
> The "or a oneAPI release that carries that table entry" hypothesis above is now
> **confirmed**. Installed the **apt-released** `intel-oneapi-compiler-dpcpp-cpp-2026.0`
> (`2026.0.0-947`, `icpx 2026.0.0.20260331`) on b70 side-by-side with 2025.3 (additive:
> `0 to remove`, lands at `/opt/intel/oneapi/compiler/2026.0`). Rebuilt BOTH probes with
> that released `icpx` and ran on the SAME GPU/driver:
> - `arch_resolve` → `is intel_gpu_bmg_g31 ? YES`, **`has(ext_intel_matrix) = YES`**
> - `ark_jm_harness` (real `IGemmDQCore` joint_matrix) → **PASS, exit=0**
>
> So the fix does NOT require the intel/llvm nightly — the released 2026.0.0 ships the
> `bmg_g31` matrix-table entry (`libsycl.so.9`). Runtime layering that works: the 2026
> `compiler/2026.0/lib` (libsycl.so.9 + libur_loader + L0 UR adapter) provides the SYCL
> side, but the **Level-Zero loader/driver are the system ones** — append
> `/usr/lib/x86_64-linux-gnu` (`libze_loader.so.1`, `libze_intel_gpu.so.1`) and a
> `libhwloc.so.15` (e.g. the venv's) to `LD_LIBRARY_PATH`, else UR finds 0 adapters
> (`urAdapterGet → 0`) and throws "No device of requested type 'gpu' available". Repro
> script on the node: `/root/verify_released_2026.sh`. Note: the apt umbrella package
> flips `compiler/latest` to `2026.0`; install the **versioned** package
> (`...-dpcpp-cpp-2026.0`) to leave `latest`→`2025.3` for torch-xpu's pin.
>
> ## Reproduce (single machine, harness already on the node under /root)
> ```bash
> # arch_resolve.cpp + jm_probe.cpp are on root@10.239.98.43:/root
> # 1) baseline 2025.3 (FAIL): matrix aspect NO
> LD_LIBRARY_PATH=/root/torch-xpu-setup/.venv/lib ONEAPI_DEVICE_SELECTOR=level_zero:gpu /root/arch_resolve
> # 2) fetch nightly DPC++ runtime, rebuild against it, re-run (PASS): matrix aspect YES
> #    (see /root/fetch_nightly.sh, /root/test_nightly.sh, /root/test_jm_nightly.sh)
> N=/root/sycl_nightly
> export PATH=$N/bin:$PATH LD_LIBRARY_PATH=$N/lib:/opt/intel/oneapi/compiler/2025.3/lib
> $N/bin/clang++ -fsycl -O2 /root/arch_resolve.cpp -o /root/arch_resolve_nightly
> ONEAPI_DEVICE_SELECTOR=level_zero:gpu /root/arch_resolve_nightly   # -> has(ext_intel_matrix)=YES
> $N/bin/clang++ -fsycl -fsycl-targets=spir64 -O2 /root/jm_probe.cpp -o /root/jm_probe_nightly
> ONEAPI_DEVICE_SELECTOR=level_zero:gpu /root/jm_probe_nightly       # -> JOINT_MATRIX_OK
> ```
> Note: the nightly L0 UR adapter needs `libhwloc.so.15` on the path (any oneAPI
> `compiler/2025.3/lib` copy works).
>
> The #2 and #1 corrections below are retained as a record; both are superseded by
> this single-variable libsycl experiment.

> **ROUTE B VERIFICATION (2026-06-05) — the REAL ARK kernel, not just `jm_probe`.**
> CORRECTION #3 proved it on `jm_probe` (a hand-written DPAS tile). This step proves
> the FAIL→PASS flip on **ARK's actual production kernel** — the exact
> `Launcher<xmx::IGemmDQCfg<float>, xmx::IGemmDQCore>::run` joint_matrix launch that
> `DnnlWrapper::sycl_igemm_s8s8` instantiates for the m>1 WOQ GEMM path.
>
> ## Why a torch-free harness was needed
> The full ARK UT links torch 2.11.0+xpu, which hard-pins `intel-sycl-rt==2025.3.2`
> and links `libsycl.so.8`. torch and ARK share one process, so the 2026 runtime
> (`.so.9`) can't be swapped in under the torch UT without also rebuilding torch.
> Route B sidesteps torch entirely: `bestla/bestla/sycl/sycl_gemm.h` (where
> `IGemmDQCore` lives) is **header-only SYCL — no oneDNN, no torch**. A ~70-line
> `main()` constructs a `sycl::queue`, allocates int8 A/B + float C/scales, fills the
> `IGemmDQParam` exactly as `sycl_igemm_s8s8` does
> (`{A,B,C, m,n,k, lda=k,ldb=k,ldc=n, bias=nullptr, scaleA, scaleB}`), and calls the
> real `Launcher::run`. Source: `auto_round_kernel/wrapper/.../ark_jm_harness.cpp`
> (kept on the node at `/root/ark_jm_harness.cpp`).
>
> ## The experiment (same source, same GPU/driver/IGC — only libsycl differs)
> | Harness build | libsycl runtime | Result |
> |---|---|---|
> | `icpx 2025.3.3`, `-fsycl-targets=spir64` → links `.so.8` | 2025.3.2 venv (`.so.8`) | ❌ **FAIL** — `no matrix hardware on the target device, joint_matrix is not supported` |
> | nightly `clang++` 2026-06-04, `-fsycl-targets=spir64` → links `.so.9` | 2026.0.0 (`.so.9`) | ✅ **PASS** — kernel launched & completed |
>
> Device `Intel(R) Graphics [0xe223]` (bmg_g31) and L0 driver `1.15.37833` identical
> across both rows. This is the production `IGemmDQCore` joint_matrix kernel, not a
> proxy — confirming the runtime bump fixes ARK's real m>1 WOQ GEMM end-to-end
> (minus torch, which is the only remaining integration blocker).
>
> ## Reproduce (harness + build/run scripts on the node under /root)
> ```bash
> # build .so.8 variant (system icpx) and run on default 2025.3.2 venv -> FAIL
> bash /root/build_harness_so8.sh && bash /root/run_harness_default.sh
> # build .so.9 variant (nightly clang) and run on 2026.0.0 runtime -> PASS
> bash /root/build_harness_so9.sh && bash /root/run_harness_2026.sh
> ```
> Build flags that matter: `-std=c++20` (header uses C++20 dependent-type syntax),
> `-DBTLA_SYCL` (gates the whole `sycl_gemm.h` body), `-I<kernel>/bestla`,
> include `bestla/sycl/sycl_wrapper.h` (pulls `sycl_utils.h` for `nd_item_helper`).


> **`DPCPP_SYCL_TARGET` VERDICT (2026-06-05) — it does NOT gate the matrix aspect.**
> Second question from the same investigation: since the CMake default is
> `intel_gpu_bmg_g21` but b70 is `bmg_g31`, does the SYCL target/device hint matter
> for the `joint_matrix` failure? **No — empirically settled.**
>
> - **Code path:** `DPCPP_SYCL_TARGET` → `SYCL_DEVICE_NAME` (`CMakeLists.txt:94-101`)
>   feeds **only** `-Xs "-device ${SYCL_DEVICE_NAME}"` inside `SYCL_TLA_LINK_FLAGS`
>   (`:107`). It is never a `target_compile_definition`, never an `#ifdef`, and never
>   reaches an aspect/`architecture_is` query in the headers. `-Xs -device` is a JIT
>   *tuning hint* for SPIR-V codegen, not the runtime capability gate.
> - **Empirical proof (stronger than a bmg_g21-vs-bmg_g31 A/B):** the Route B harness
>   that PASSES on `.so.9` is built with `-fsycl -fsycl-targets=spir64` and **no
>   `-Xs -device` hint at all** (`/root/build_harness_so9.sh`). The matrix aspect is
>   advertised and `IGemmDQCore` runs on bmg_g31 with *zero* device hint — so the hint
>   (all `DPCPP_SYCL_TARGET` controls) cannot be the gate. The libsycl version is.
> - **Direct A/B on 2025.3 `.so.8` (2026-06-05, `/root/ab_dpcpp_target_2025.sh`):**
>   the same harness, rebuilt with `icpx 2025.3.3` three ways and all run on the
>   2025.3.2 `.so.8` runtime — the device hint is the ONLY variable:
>
>   | `-Xs -device` hint | Result on `.so.8` |
>   |---|---|
>   | none | ❌ FAIL `joint_matrix is not supported` |
>   | `bmg_g31` (matches b70 silicon) | ❌ FAIL — byte-identical |
>   | `bmg_g21` (CMake default) | ❌ FAIL — byte-identical |
>
>   All three fail identically on `[0xe223]`. **On oneAPI 2025, changing
>   `DPCPP_SYCL_TARGET` does NOT fix m>1** — even hinting the exact b70 arch
>   (`bmg_g31`) is inert against the runtime aspect gate. Only the `.so.8`→`.so.9`
>   runtime swap flips FAIL→PASS, in either device-hint direction.
> - **Consequence:** leaving the CMake default at `intel_gpu_bmg_g21` is fine for b70
>   functionally; it only affects JIT tuning, never the m>1 `joint_matrix` pass/fail.
>   The earlier AOT-on-`.so.8` experiment (below) already showed compiling *for*
>   `bmg-g31` does not move the runtime aspect gate either — same conclusion from the
>   AOT direction.
>
> Practical note for reproducing at the **full `test_ARK_XPU` UT** level (vs the
> harness): the intel/llvm nightly (`/root/sycl_nightly`) ships `libsycl.so.9` but
> **no `IntelSYCLConfig.cmake`**, while ARK's `CMakeLists.txt:55` hard-requires
> `find_package(IntelSYCL REQUIRED)` — which on this node only resolves to the 2025.3
> module that links the broken `.so.8`. So the UT CMake build needs a cross-toolchain
> shim to use the nightly; the torch-free harness (built with the nightly `clang++`
> directly, no CMake) is the clean authoritative int8-XMX proof and avoids that.


> **CORRECTION #2 (2026-06-05, supersedes BOTH the IGC conclusion and the original
> analysis).** A controlled single-machine experiment on `b70-pc6` (root@10.239.98.43)
> **disproves the "IGC version is the determinant" conclusion** below.
>
> **Real determinant: the GPU itself does not advertise the `joint_matrix`
> capability.** This node's device is **`intel_gpu_bmg_g31`** (Arc Pro B70, PCI
> `0xe223`), and it reports **`ext_oneapi_matrix` = NO** in `sycl-ls --verbose`.
> When that aspect is absent, `libsycl` throws
> `no matrix hardware on the target device, joint_matrix is not supported`
> **regardless of IGC version** — there is no `joint_matrix` lowering path to invoke.
>
> ## The experiment (one machine, only IGC changed)
> The SAME host-built generic-SPIR-V `jm_probe` (int8 M8×N16×K32 DPAS tile) was run
> against the SAME physical GPU, varying ONLY the user-space IGC/runtime via a
> container with GPU passthrough (`--device /dev/dri`):
>
> | Environment | IGC (`libigc.so` init-loaded) | compute-runtime / L0 | `ext_oneapi_matrix` | jm_probe |
> |---|---|---|---|---|
> | Host (bare metal) | 2.32.7 | 26.14 / 1.15.37833 | **NO** | ❌ FAIL |
> | Container `igc-newer:noble` | **2.34.4** (`/usr/local/lib/libigc.so.2`, confirmed via `LD_DEBUG=libs` → `calling init: /usr/local/lib/libigc.so.2`) | **26.18 / 1.15.38308** | **NO** | ❌ **FAIL** |
>
> Both rows are the SAME `intel_gpu_bmg_g31` device (`sycl-ls` Architecture line,
> identical on host and in container). The container stack is the EXACT pairing the
> 2026-06-03 correction called the "PASS" set (IGC 2.34.4, runtime 38308) — yet it
> **still FAILS here.** So IGC 2.34.x is **necessary-but-not-sufficient**, or
> irrelevant; the device's missing `ext_oneapi_matrix` aspect is what gates it.
>
> ## Why the 2026-06-03 IGC conclusion was wrong
> That correction compared **two different physical machines** (a PASS node with an
> Arc Pro B60 / different BMG stepping, and this FAIL node) and attributed the
> difference to IGC — but **hardware co-varied with IGC** across those nodes. The
> single-machine container experiment isolates the variable: holding the GPU fixed
> and upgrading only IGC to the "PASS" version does **not** fix it. The deciding
> factor was the **device** (and its driver's advertised aspects), not IGC.
>
> > The PASS node's B60 advertises `ext_oneapi_matrix`; this node's B70
> > (`bmg_g31`, `0xe223`) does not — with the current `xe` KMD + Battlemage
> > firmware on this box. Whether that is a permanent silicon limitation or a
> > driver/firmware gap on G31 is the open question (see below).
>
> ## What this means for the ARK m>1 GEMM on THIS node
> - **An IGC upgrade will NOT fix it** (experimentally shown). Don't chase the
>   `libigc2 >= 2.34` apt/deb route for this device.
> - The working levers are the ARK-side ones that **avoid the `joint_matrix`
>   abstraction**: route w4 m>1 through the oneDNN fp GEMM, or replace ARK's
>   `joint_matrix` int8 kernel with inline-`dpas` asm (sycl-tla `XE_DPAS_TT`) — see
>   "a third ARK-side fix option" below. Those bypass the missing-aspect check.
> - Open follow-up: confirm whether a newer `xe` KMD / Battlemage GuC firmware makes
>   `bmg_g31` advertise `ext_oneapi_matrix`. The host kernel driver (not the
>   container user-space) owns that aspect, so a container can never add it.
>
> ## Reproduce (container experiment)
> ```bash
> # 1. host-built generic SPIR-V probe (portable across drivers):
> source /opt/intel/oneapi/setvars.sh
> icpx -fsycl -fsycl-targets=spir64 -O2 docs/jm_probe.cpp -o /root/jm_probe_new
> ONEAPI_DEVICE_SELECTOR=level_zero:gpu /root/jm_probe_new        # FAIL on host
> # 2. image with IGC 2.34.4 + runtime 26.18 over the local intel/vllm base
> #    (debs from intel/intel-graphics-compiler v2.34.4 + intel/compute-runtime 26.18.38308.1,
> #     installed to /usr/local/lib; see /root/igc-test/Dockerfile on the node)
> # 3. run the SAME probe inside, host GPU passed through:
> docker run --rm --device /dev/dri:/dev/dri \
>   --group-add "$(getent group render | cut -d: -f3)" \
>   -v /root/jm_probe_new:/jm_probe:ro -e ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
>   igc-newer:noble /jm_probe                                     # STILL FAILS
> # 4. confirm the device advertises no matrix aspect (host OR container):
> sycl-ls --verbose | grep -E "Architecture|ext_oneapi_matrix"   # bmg_g31, no matrix aspect
> ```
>
> The 2026-06-03 IGC analysis is retained below as a record; treat its "THE FIX:
> upgrade IGC" as **disproven for this device**.

> **CORRECTION (2026-06-03, supersedes the analysis below).**
> The original conclusion in this doc — "missing SPV_INTEL matrix extension /
> ARK_SYCL_TLA=OFF build bug" — is **WRONG**, and the *first* correction's
> "upgrade the L0 driver" framing was a **proxy** for the true determinant.
>
> **Real root cause: the Intel Graphics Compiler (IGC, `libigc2`) is too old.**
> IGC is the JIT compiler (bundled with, but versioned separately from, the
> compute-runtime) that lowers `joint_matrix` SPIR-V into Battlemage DPAS
> instructions. The ability to lower this int8 M8×N16×K32 DPAS tile landed in IGC
> **between 2.32.7 and 2.34.4**.
>
> The SAME wheel binary (`auto-round-lib 0.13.1`, TLA-OFF,
> md5 `88c363a4a736fc468a59b97870ab2bff`) **PASSES** on IGC `2.34.4` and **FAILS**
> on IGC `2.32.7`. Same DPC++ runtime (libsycl `2025.3.2.20260112`), same
> Battlemage family, same TLA-OFF binary — the only variable is **IGC**.
>
> | | PASS node | FAIL node |
> |---|---|---|
> | Wheel md5 | `88c363a4…` | `88c363a4…` (identical) |
> | libsycl (DPC++) | `2025.3.2.20260112` | `2025.3.2.20260112` (identical) |
> | **IGC (`libigc2`)** | **2.34.4** (`libigc.so.2.34.4`) | **2.32.7** (`libigc.so.2.32.7`) |
> | compute-runtime | `26.18.38308` | `26.14.37833` |
> | L0 driver soname | `1.15.38308` | `1.15.37833` |
> | Device | Arc Pro B60 (BMG) | Graphics 0xe223 (BMG) |
> | Result | PASSED | FAILED |
>
> ## THE FIX (experimentally validated)
> **Upgrade IGC to `libigc2` >= 2.34.x.** IGC ships version-locked inside the
> compute-runtime bundle, so upgrade the set together:
> ```bash
> # On the FAIL node (root@10.239.98.43). The newer set is the 38308 / IGC 2.34.x
> # release. Upgrade IGC AND the runtime packages together — bumping libigc2 alone
> # is not enough; intel-opencl-icd / libze-intel-gpu1 embed and expect the
> # matching IGC.
> apt-get install --only-upgrade \
>     libigc2 intel-opencl-icd libze-intel-gpu1 intel-ocloc
> # Verify afterwards:
> dpkg-query -W -f='${Version}\n' libigc2        # want >= 2.34.x
> readlink -f /usr/lib/x86_64-linux-gnu/libigc.so*
> ```
> ⚠️ The FAIL node is Ubuntu **22.04**, whose Intel GPU channel may not yet
> package IGC 2.34.x. If `apt` has no candidate, install the matching `~22.04`
> .deb set directly from Intel's GitHub releases (intel-graphics-compiler +
> compute-runtime), or move the node to a channel that carries 2.34.x.
> The L0 build number (`38308`) is only the visible bundle label — the operative
> payload is the IGC version it carries.
>
> **DISPROVEN: rebuilding the wheel with the matrix SPIR-V extension does NOT
> help.** A `jm_probe` built *with*
> `+SPV_INTEL_subgroup_matrix_multiply_accumulate` is **byte-identical** to one
> built without it (the C++ `joint_matrix` API already maps to
> `+SPV_INTEL_joint_matrix`, which the compiler emits by DEFAULT — the extra
> `-spirv-ext` adds an unused, different extension). Both FAIL identically on IGC
> 2.32.7. So no build flag — and no `ARK_SYCL_TLA=ON` rebuild — can rescue an old
> IGC; only an IGC upgrade fixes it. `ARK_SYCL_TLA` is a separate concern
> (CUTLASS/TLA path) and is NOT what gates this m>1 GEMM.
>
> **Definitive experiment (2026-06-03) — flags don't matter, IGC does.**
> Two standalone `jm_probe` binaries (the SAME int8 M8×N16×K32 DPAS tile ARK
> uses), built to generic SPIR-V with oneAPI 2025.3, run on the FAIL node:
>
> | Probe | extra SPIR-V ext | IGC 2.32.7 (remote) | IGC 2.34.4 (local) |
> |---|---|---|---|
> | implicit | default (`+SPV_INTEL_joint_matrix` only) | ❌ FAIL | ✅ PASS |
> | explicit | `+SPV_INTEL_subgroup_matrix_multiply_accumulate` | ❌ FAIL | — |
>
> **Version decoding.** compute-runtime uses `YY.WW.BUILD.REV`
> (year . work-week . build); IGC uses plain upstream `MAJOR.MINOR.PATCH`. The
> apt suffix `-NNNN~DD.DD` is Intel's package revision + target Ubuntu release
> (`25.10` = Ubuntu 25.10, `22.04` = Ubuntu 22.04 LTS — NOT a date):
>
> | Node | compute-runtime pkg | IGC pkg | Ubuntu |
> |---|---|---|---|
> | FAIL | `26.14.37833.4-1256~22.04` (WW14 ’26) | `2.32.7-1256~22.04` | 22.04 |
> | PASS | `26.18.38308.1-1~25.10~ppa1` (WW18 ’26) | `2.34.4-1260~25.10` | 25.10 |
>
> Both are recent 2026 builds, ~1 monthly drop apart.
> Ref: https://github.com/intel/intel-graphics-compiler/releases
>      https://github.com/intel/compute-runtime/releases

> **Check the determinant on the remote node** (root@10.239.98.43) — copy/paste:
> ```bash
> # 1) IGC version (the DECISIVE number — want >= 2.34.x):
> ssh root@10.239.98.43 'dpkg-query -W -f="${Version}\n" libigc2; \
>   readlink -f /usr/lib/x86_64-linux-gnu/libigc.so*'
> #    -> 2.32.7-1256~22.04  /  libigc.so.2.32.7+0   (too old, FAILS)
>
> # 2) compute-runtime / L0 bundle that carries it (proxy, year.workweek.build):
> ssh root@10.239.98.43 'dpkg -l | grep -E "libze-intel-gpu1|intel-opencl-icd|intel-ocloc"'
> #    -> 26.14.37833.4-1256~22.04
> ssh root@10.239.98.43 'readlink -f /lib/x86_64-linux-gnu/libze_intel_gpu.so.1'
> #    -> .../libze_intel_gpu.so.1.15.37833
>
> # 3) as torch sees it:
> ssh root@10.239.98.43 '/root/torch-xpu-setup/.venv/bin/python -c \
>   "import torch;p=torch.xpu.get_device_properties(0);print(p.name,p.driver_version)"'
> #    -> Intel(R) Graphics [0xe223] 1.15.37833+4
>
> # 4) full report via the saved script (already copied to /root on the remote):
> ssh root@10.239.98.43 'PYTHON=/root/torch-xpu-setup/.venv/bin/python bash /root/check_xpu_driver.sh'
> ```
>
> **Check a node:** run `bash docs/check_xpu_driver.sh`
> (`PYTHON=/path/to/venv/bin/python bash check_xpu_driver.sh`). It reports the IGC
> (`libigc2`) version and the compute-runtime bundle, and flags whether they are
> new enough.
>
> **Definitive one-click capability check:** run `bash docs/check_joint_matrix.sh`.
> It builds a minimal standalone `joint_matrix` program (`docs/jm_probe.cpp`, the
> SAME int8 M8xN16xK32 DPAS tile ARK uses) to generic SPIR-V and runs it, so the
> result depends ONLY on the IGC's JIT-lowering ability — nothing from auto-round
> is involved. Point it at a venv's runtime with
> `RUNTIME_LIB=/path/to/venv/lib bash docs/check_joint_matrix.sh`.
> Confirmed with byte-identical SPIR-V binaries (implicit AND explicit-extension
> builds behave the same — see the experiment table above):
>   - IGC `2.34.4` -> `JOINT_MATRIX_OK` (PASS)
>   - IGC `2.32.7` -> `no matrix hardware ... joint_matrix is not supported` (FAIL)
>
> The "ROOT CAUSE" section below about the SPIR-V extension is retained only as a
> record of the (incorrect) initial investigation.

---

## The lowering process, and why vllm-xpu-kernels does NOT hit this bug

This was a confusing point ("vllm also uses XMX / DPAS on the same hardware — why
does it work on the old IGC?"), so here is the verified mechanism.

### How a matrix op becomes a DPAS instruction

There are TWO different ways to reach the SAME Battlemage DPAS (XMX) hardware,
and they go through DIFFERENT compiler stages:

```
                                          ┌─────────────────────────────────────┐
  ARK IGemmDQCore (joint_matrix)          │  the failing path                   │
  ──────────────────────────────          └─────────────────────────────────────┘
  C++  joint_matrix<...> + joint_matrix_mad
    │   (high-level matrix abstraction)
    ▼
  DPC++/libsycl emits SPIR-V op  OpJointMatrixMad  (capability SPV_INTEL_joint_matrix)
    │
    ▼
  *** IGC LOWERING PASS ***  <-- the runtime IGC must translate the abstract
    │                            joint_matrix op INTO a concrete `dpas` ISA op.
    │                            IGC 2.32.7 CANNOT do this for the int8 tile
    │                            and gives up -> libsycl throws
    │                            "no matrix hardware ... joint_matrix is not supported"
    ▼
  dpas instruction on XMX        (only reached on IGC >= 2.34.x)


                                          ┌─────────────────────────────────────┐
  sycl-tla XE_DPAS_TT (CuTe)              │  vllm-xpu-kernels path — immune     │
  ────────────────────────────           └─────────────────────────────────────┘
  C++  XE_DPAS_TT<M, s8, s8, d>::fma()
    │   lowers to INLINE ASSEMBLY, not a matrix abstraction:
    │     asm("dpas.s8.s8.8.%3 (M1,16) DST.0 SRC0.0 SRC1_UD.0 SRC2_UD(0,0)")
    │   (verified: sycl-tla include/cute/arch/mma_xe.hpp lines 78-115)
    ▼
  the `dpas` instruction is ALREADY WRITTEN by hand in the header
    │   -> there is NO joint_matrix op, NO SPV_INTEL_joint_matrix capability,
    │      and NOTHING for IGC's matrix-lowering pass to translate.
    ▼
  IGC just assembles / register-allocates the given `dpas`  (any IGC handles this)
    ▼
  dpas instruction on XMX        (works on IGC 2.32.7 — the broken pass is skipped)
```

### The key insight

The ARK failure is NOT "the hardware lacks XMX" and NOT "DPAS is unsupported".
It is specifically: **IGC 2.32.7's pass that lowers the high-level `joint_matrix`
abstraction into a `dpas` machine instruction is missing/broken for the int8 tile.**

- ARK's `IGemmDQCore` **depends on that pass** (it hands IGC an abstract
  `joint_matrix` op and asks IGC to produce the `dpas`). Old IGC can't -> FAIL.
- sycl-tla's `XE_DPAS_TT` **emits the `dpas` instruction itself, as inline asm**
  (`include/cute/arch/mma_xe.hpp:93-112`; the int8 variant is
  `CUTE_DECLARE_XE_DPAS_TT(d, s8, s8, d)` at line 151 — the SAME int8->int32 math
  ARK does). The broken lowering pass is never invoked, so IGC version is
  irrelevant to this op.

So both compute identical int8 DPAS math on identical silicon; only ARK routes
through the IGC lowering pass that 2.32.7 lacks.

### Two independent reasons vllm-xpu-kernels is unaffected

1. **No `joint_matrix` anywhere.** `grep -rI joint_matrix` over the whole
   vllm-xpu-kernels tree returns ZERO hits. Its matrix math uses either CuTe
   `XE_DPAS_TT` (inline `dpas` asm, above) or oneDNN `dnnl::matmul` (oneDNN's own
   internal DPAS codegen) — neither emits the `SPV_INTEL_joint_matrix` op that
   IGC 2.32.7 chokes on. (This is also why `torch` bf16/f16 matmul, which is
   oneDNN, already PASSES on the FAIL node.)
2. **AOT compilation.** vllm-xpu-kernels builds with
   `-fsycl-targets=spir64_gen -device bmg` (CMakeLists.txt:257,276,286) — device
   code is lowered to real Battlemage ISA AT BUILD TIME by the build container's
   toolchain and baked into the binary. The deploy node's runtime IGC mostly just
   loads precompiled ISA; it does little/no JIT lowering. ARK, by contrast, ships
   GENERIC SPIR-V (`spir64`) and JIT-lowers on the deploy node — which is the very
   reason the deploy node's IGC version became the deciding factor.

Either reason alone is sufficient; vllm has both.

### Consequence: a third ARK-side fix option

Beyond "upgrade IGC >= 2.34" and "route w4 m>1 through the oneDNN fp GEMM"
(see Fix options below), this verified mechanism unlocks a third route that keeps
fused int8 DPAS performance AND works on IGC 2.32.7:

3. **Replace ARK's `joint_matrix` int8 kernel with an inline-`dpas` implementation**
   (hand-written asm like sycl-tla's `XE_DPAS_TT`, or adopt `XE_DPAS_TT` directly).
   It bypasses the broken IGC lowering pass entirely, so it needs no driver
   upgrade and keeps the fused int8 path. Cost: more engineering than the env-var
   fallback; benefit: no IGC dependency, no perf regression. Switching ARK to AOT
   (`spir64_gen -device bmg`) is a complementary lever that also removes runtime
   IGC sensitivity.

---

## AOT-on-`.so.8` experiment — does AOT alone rescue the vllm 2025.3 docker?

**Question.** The vllm-xpu docker ships oneAPI **2025.3 / libsycl.so.8**. Of the two
cheap fixes, "rebuild ARK vs `.so.9`" is dead on arrival there (no `.so.9` in the
image). That leaves AOT (`spir64_gen -device bmg`). AOT only helps if the failure is a
**JIT-compile** error (lowering happens on the deploy node). If instead it's a
**runtime aspect** throw inside libsycl, AOT cannot help — the gate fires before the
kernel runs. Route B couldn't distinguish these; this experiment does.

**Method (one variable).** Same `ark_jm_harness.cpp`, same `.so.8` runtime. Only the
*compile mode* changed: AOT via icpx 2025.3 with `-fsycl-targets=spir64_gen -Xs
"-device bmg-g31"` (vs the generic `spir64` build that JIT-lowers on deploy).

```
build_harness_so8_aot.sh : icpx 2025.3.3, -fsycl-targets=spir64_gen -Xs "-device bmg-g31"
  -> build SUCCESS, no compile error, binary NEEDED libsycl.so.8

run on .venv 2025.3.2 (.so.8):
  [harness] launching joint_matrix kernel...
  [harness] FAIL: sycl::exception: no matrix hardware on the target device,
            joint_matrix is not supported          exit=2
```

**Result — AOT does NOT work on `.so.8`.** Two facts pin the failure to the runtime,
not codegen:
- The matrix kernel **compiled cleanly ahead-of-time** for `bmg-g31` (IGC accepts the
  arch, DPAS is baked into the binary) — so it was never a JIT/codegen problem.
- It still **throws at runtime**, because `libsycl.so.8` consults its compiled
  `supported_archs[]` table via `has(aspect::ext_intel_matrix)`, gets NO for bmg_g31,
  and bails *before the kernel launches*. AOT moves the compile off the node; it does
  not touch the libsycl aspect gate.

**Correction to option 2 / the "complementary lever" note above:** AOT does **not**
remove the runtime aspect sensitivity on a `.so.8` runtime. It is only a fix when the
runtime is already `.so.9` (where it additionally avoids deploy-node JIT). On the
2025.3 docker, AOT alone is insufficient.

**Decision matrix for the vllm 2025.3 / `.so.8` container:**

| Option | Works in vllm 2025.3 docker? | Why |
|---|---|---|
| Rebuild ARK vs `.so.9` | ❌ | no `.so.9` in the image |
| AOT compile (`spir64_gen -device bmg-g31`) | ❌ | proven here — runtime aspect throw on `.so.8` |
| Upgrade in-image runtime to `.so.9` | ⚠️ works, but mutates shared image + forces a torch-xpu rebuild |
| **Inline-`dpas` rewrite (sycl-tla `XE_DPAS_TT`)** | ✅ | bypasses the `joint_matrix` aspect gate; emits DPAS directly |

So to ship inside the unmodified vllm 2025.3/`.so.8` image, the **inline-`dpas`
rewrite (option 3 above) is the only viable path** — both cheap alternatives are ruled
out.

**Date:** 2026-06-04
**Node:** root@10.239.98.43 (b70-pc6)

---

**Date:** 2026-06-03
**Node:** root@10.239.98.43 (b70-pc6)
**Device:** Intel(R) Graphics [0xe223] — Battlemage G21, 256 EU
**Driver:** libze_intel_gpu.so.1.15.37833 (Level-Zero)
**Python env:** /root/torch-xpu-setup/.venv (Python 3.13, torch 2.11.0+xpu)

## Symptom

UT:
```
LD_LIBRARY_PATH=/root/torch-xpu-setup/.venv/lib:$LD_LIBRARY_PATH \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/root/torch-xpu-setup/.venv/bin/python -m pytest \
  test_model.py::TestAutoRoundARKBackend::test_formats'[xpu-dtype0-4-128-True-auto_round]' -vv
```
(Run from /root/auto-round/test/test_ark/)

Quantization (w4g128), shard-write, and repack-to-XPU all SUCCEED.
Fails during model.generate() at inference:
```
auto_round_kernel/qlinear.py:242  -> ark.woqgemm(...)
auto_round_kernel/__init__.py:246 -> lib.woqgemm(...)
RuntimeError: no matrix hardware on the target device, joint_matrix is not supported
```

NOTE: the original command (HF_ENDPOINT=https://hf-mirror.com) failed earlier on a
*network timeout* fetching facebook/opt-125m. The model is already cached locally,
so use HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 to bypass. That is unrelated to the
real kernel failure below.

## Conclusion

**This is an auto_round_kernel BUILD/PACKAGING issue, NOT a driver/hardware issue.**

### Proof the driver/HW are fine
PyTorch's own XMX matmul works on this exact device:
```
torch matmul bfloat16: OK
torch matmul float16:  OK
```
0xe223 (Battlemage) has full XMX/DPAS joint_matrix hardware, and the L0 driver exposes it.

### Failing code path (m>1 / prefill)
- qlinear.py:242 -> ark.woqgemm -> XpuWrapper::woq_gemm (xpu_wrapper.hpp:668)
- woq_gemv() returns -2 for m>1 (xpu_wrapper.hpp:510) -> falls through to oneDNN woq_s8
- DnnlWrapper::woq_s8 (XPU branch, dnnl_wrapper.hpp:229) -> sycl_igemm_s8s8 (dnnl_wrapper.hpp:139)
- sycl_igemm_s8s8 -> bestla::sycl_gemm::xmx::IGemmDQCore
- IGemmDQCore (bestla/bestla/sycl/sycl_gemm.h:760) uses joint_matrix / joint_matrix_mad
  DIRECTLY (int8 M8xN16xK32, sub_group=16, B col_major)

No oneDNN verbose is emitted (ONEDNN_VERBOSE=2 silent) -> the failing kernel is ARK's
OWN hand-written joint_matrix kernel, not an oneDNN primitive.

The m==1 (decode/GEMV) path works because BesTLA's GEMV does NOT use joint_matrix.

### ROOT CAUSE
The installed .so was built WITHOUT the matrix-MMA SPIR-V extension.

Inspecting the installed binary:
  .so CONTAINS:  SPV_INTEL_split_barrier
  .so MISSING:   SPV_INTEL_subgroup_matrix_multiply_accumulate   <-- joint_matrix needs this
  .so MISSING:   SPV_INTEL_2d_block_io
  .so MISSING:   any sdpa/sage symbols  -> built with ARK_SYCL_TLA=OFF

In auto_round_kernel/CMakeLists.txt the matrix-MMA extension is declared ONLY inside
SYCL_TLA_LINK_FLAGS:
```
"-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate"
```
which is applied ONLY under `if(ARK_XPU AND ARK_SYCL_TLA)`.

But IGemmDQCore (the m>1 GEMM) uses joint_matrix UNCONDITIONALLY (gated on ARK_XPU,
not on ARK_SYCL_TLA). So a TLA-off build emits joint_matrix ops in the device image
WITHOUT declaring the matrix-MMA capability. At runtime the DPC++ device library
cannot lower them and hits the generic stub that throws:
  "no matrix hardware on the target device, joint_matrix is not supported"
(That string lives in libsycl.so.8 — the DPC++ runtime — NOT in ARK and NOT in the driver.)

### Where the broken .so came from
auto-round-lib is a PREBUILT manylinux wheel pulled from a REMOTE pip index
(installed by uv), NOT a local source build:
  Package:   auto-round-lib  (dist: auto_round_lib-0.13.1.dist-info)
  Version:   0.13.1
  INSTALLER: uv
  WHEEL Tag: cp313-cp313-manylinux_2_28_x86_64
  direct_url.json: ABSENT  -> came from an index/registry (not local path / not VCS)
  Home-page: https://github.com/intel/auto-round/auto_round_extension/ark

=> The published 0.13.1 wheel was built with ARK_SYCL_TLA=OFF, so its m>1 GEMM
   device image lacks the matrix-MMA capability and fails on any XMX device.

Note: setup.py gates TLA on oneAPI >= 2025.3 (enable_sycl_tla). The build compiler
in the docker images is icpx 2025.3.2, which SHOULD enable TLA — yet the shipped
wheel has it off. The published wheel build env evidently did not satisfy that gate
(older/different oneAPI), producing a TLA-off binary.

## Fix options

> ⚠️ **SUPERSEDED — both options below are DISPROVEN.** See the CORRECTION block
> at the top of this file. Rebuilding with `ARK_SYCL_TLA=ON` or moving the
> matrix-MMA extension into the always-on link flags produces a **byte-identical**
> device image (the `joint_matrix` API already emits `+SPV_INTEL_joint_matrix` by
> default) and still **FAILS** on IGC 2.32.7. The real, validated fix is to
> **upgrade IGC (`libigc2`) to >= 2.34.x** — see "THE FIX" at the top. The two
> options are retained only as a record of the incorrect initial investigation.

1. (Quick) Rebuild & reinstall the kernel with ARK_SYCL_TLA=ON. The matrix-MMA
   SPIR-V extension is then linked in and the m>1 GEMM works.

2. (Structural, preferred) In auto_round_kernel/CMakeLists.txt, decouple the
   matrix-MMA SPIR-V extension from TLA. BesTLA's IGemmDQCore needs
   +SPV_INTEL_subgroup_matrix_multiply_accumulate whenever ARK_XPU is on,
   regardless of ARK_SYCL_TLA. Move that extension into the always-applied XPU
   link flags so a no-TLA build still produces a working m>1 GEMM. Then republish
   the wheel.

## Repro / verification commands

```bash
# torch XMX works on this device (driver OK):
LD_LIBRARY_PATH=/root/torch-xpu-setup/.venv/lib:$LD_LIBRARY_PATH \
  /root/torch-xpu-setup/.venv/bin/python -c \
  'import torch;a=torch.randn(32,768,device="xpu",dtype=torch.bfloat16);\
   b=torch.randn(768,768,device="xpu",dtype=torch.bfloat16);\
   print((a@b).float().sum())'

# missing matrix-MMA extension in shipped .so:
SO=/root/torch-xpu-setup/.venv/lib/python3.13/site-packages/auto_round_kernel/auto_round_kernel_xpu.cpython-313-x86_64-linux-gnu.so
strings "$SO" | grep -iE 'SPV_INTEL_subgroup_matrix|2d_block_io|split_barrier' | sort -u
# -> only SPV_INTEL_split_barrier present (matrix MMA + 2d_block_io absent)

# error string origin = DPC++ runtime, not ARK, not driver:
strings /root/torch-xpu-setup/.venv/lib/libsycl.so.8 | grep 'no matrix hardware'
```


```
(ark) yiliu7@inc101:~/workspace/auto-round$ dpkg -l | grep libigc; readlink -f /usr/lib/x86_64-linux-gnu/libigc.so*
ii  libigc2                                        2.34.4-1260~25.10                          amd64        Core libraries for Intel(R) Graphics Compiler for OpenCL(TM)
/usr/lib/x86_64-linux-gnu/libigc.so.2.34.4+0
/usr/lib/x86_64-linux-gnu/libigc.so.2.34.4+0

root@b70-pc6:~# dpkg -l | grep libigc; readlink -f  /usr/lib/x86_64-linux-gnu/libigc.so*
ii  libigc-dev                                2.32.7-1256~22.04                   amd64        Core development files for Intel(R) Graphics Compiler for OpenCL(TM)
ii  libigc-tools                              2.32.7-1256~22.04                   amd64        Media driver tools for Intel(R) Graphics Compiler for OpenCL(TM)
ii  libigc2                                   2.32.7-1256~22.04                   amd64        Core libraries for Intel(R) Graphics Compiler for OpenCL(TM)
/usr/lib/x86_64-linux-gnu/libigc.so.2.32.7+0
/usr/lib/x86_64-linux-gnu/libigc.so.2.32.7+0
/usr/lib/x86_64-linux-gnu/libigc.so.2.32.7+0
```