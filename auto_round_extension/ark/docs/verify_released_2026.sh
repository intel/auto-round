#!/bin/bash
# Verify the RELEASED oneAPI 2026 fixes m>1 joint_matrix on b70 (bmg_g31).
# Uses the freshly-installed /opt/intel/oneapi/compiler/2026.0 icpx (NOT the nightly clang,
# NOT the extracted srt wheels). This proves the formal apt-released 2026 carries the
# bmg_g31 matrix table end-to-end: build + link + run, one self-contained toolchain.
#
# Prereqs on the node (root@10.239.98.43, Arc Pro B70 / bmg_g31 / PCI 0xe223):
#   - /opt/intel/oneapi/compiler/2026.0  (apt pkg intel-oneapi-compiler-dpcpp-cpp-2026.0=2026.0.0-947)
#   - /root/arch_resolve.cpp  /root/ark_jm_harness.cpp  (copies live in this docs/ dir)
#   - /root/auto-round checked out at feat/ark-xpu-int3-woq-gemm
ONEAPI2026=/opt/intel/oneapi/compiler/2026.0
ICPX=$ONEAPI2026/bin/icpx
SRCROOT=/root/auto-round/auto_round_extension/ark/auto_round_kernel
SRC=/root/ark_jm_harness.cpp

echo "=== compiler ==="
$ICPX --version | head -2
echo

# The released 2026 compiler/lib tree provides libsycl.so.9 + libur_loader.so.0 +
# the L0 UR *adapter*, but the actual Level-Zero loader (libze_loader.so.1) and the
# L0 GPU driver (libze_intel_gpu.so.1) are the system ones in /usr/lib/x86_64-linux-gnu,
# and hwloc.so.15 comes from the venv. Layer: 2026 first, then system L0, then hwloc.
export LD_LIBRARY_PATH=$ONEAPI2026/lib:/usr/lib/x86_64-linux-gnu:/root/torch-xpu-setup/.venv/lib

echo "=== build arch_resolve (aspect gate-check) ==="
$ICPX -fsycl -O2 /root/arch_resolve.cpp -o /root/arch_resolve_rel2026 2>&1 | grep -iE "error" | head
echo "  --- run: want has(ext_intel_matrix)=YES ---"
ONEAPI_DEVICE_SELECTOR=level_zero:gpu /root/arch_resolve_rel2026 2>&1 | sed 's/^/  /'
echo

echo "=== build harness (real ARK IGemmDQCore joint_matrix) ==="
$ICPX -fsycl -fno-sycl-instrument-device-code -O2 -std=c++20 -DBTLA_SYCL \
  -I$SRCROOT/bestla -fsycl-targets=spir64 \
  $SRC -o /root/ark_jm_harness_rel2026 2>&1 | grep -iE "error" | head -40
echo "  SONAME: $(objdump -p /root/ark_jm_harness_rel2026 2>/dev/null | awk '/NEEDED/ && /sycl/ {print $2}')"
echo "  which libsycl resolves:"
ldd /root/ark_jm_harness_rel2026 2>/dev/null | grep -iE "sycl|libur_loader" | sed 's/^/    /'
echo "  --- run: want PASS, exit=0 ---"
ONEAPI_DEVICE_SELECTOR=level_zero:gpu /root/ark_jm_harness_rel2026 128 128 128 2>&1 | sed 's/^/  /'
echo "  exit=${PIPESTATUS[0]}"
echo "=== DONE ==="
