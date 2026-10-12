#!/usr/bin/env bash
# 续接 resolve.sh：解决 CMakeLists.txt 与 __init__.py 冲突
set -euo pipefail

CMAKE=auto_round_extension/ark/auto_round_kernel/CMakeLists.txt
INIT=auto_round_extension/ark/auto_round_kernel/__init__.py

# 必须处于 merge 中
git rev-parse -q --verify MERGE_HEAD >/dev/null || { echo "当前不在 merge 状态，请先运行 resolve.sh"; exit 1; }

# ---------------------------------------------------------------------------
# 1) CMakeLists.txt：以 main 为准，补回 PR 的 4 处新增
# ---------------------------------------------------------------------------
git checkout --theirs -- "$CMAKE"
python3 - "$CMAKE" <<'PY'
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()

def insert_after(s, anchor, text, name):
    if text.strip() in s:
        print(f"[cmake] {name}: 已存在，跳过"); return s
    assert s.count(anchor) == 1, f"[cmake] {name}: 锚点出现 {s.count(anchor)} 次"
    print(f"[cmake] {name}: 已插入")
    return s.replace(anchor, anchor + text, 1)

def insert_before(s, anchor, text, name):
    if text.strip() in s:
        print(f"[cmake] {name}: 已存在，跳过"); return s
    assert s.count(anchor) == 1, f"[cmake] {name}: 锚点出现 {s.count(anchor)} 次"
    print(f"[cmake] {name}: 已插入")
    return s.replace(anchor, text + anchor, 1)

s = insert_after(s,
    '  option(ARK_JOINT_MATRIX "Build BestLA joint_matrix S8 backend" OFF)\n',
    '  set(ARK_MXFP_BDPAS_CUTLASS_DIR "" CACHE PATH "Optional CUTLASS tree containing Xe BDPAS block-scaled grouped GEMM headers")\n',
    "ARK_MXFP_BDPAS_CUTLASS_DIR option")

s = insert_before(s,
    '  elseif(DPCPP_SYCL_TARGET STREQUAL "intel_gpu_pvc" OR DPCPP_SYCL_TARGET STREQUAL "pvc")\n',
    '  elseif(DPCPP_SYCL_TARGET STREQUAL "intel_gpu_cri" OR DPCPP_SYCL_TARGET STREQUAL "cri")\n'
    '    set(SYCL_DEVICE_NAME "cri")\n',
    "cri target")

s = insert_after(s,
    '  target_compile_definitions(${PY_NAME} PRIVATE ARK_SYCL_TLA=1 CUTLASS_ENABLE_SYCL=1 SYCL_INTEL_TARGET=1)\n',
    '  if(ARK_MXFP_BDPAS_CUTLASS_DIR)\n'
    '    if(NOT EXISTS "${ARK_MXFP_BDPAS_CUTLASS_DIR}/include/cutlass/gemm/collective/xe_array_mma_blockscaled_native.hpp")\n'
    '      message(FATAL_ERROR "ARK_MXFP_BDPAS_CUTLASS_DIR must point to a CUTLASS tree with Xe block-scaled grouped GEMM headers")\n'
    '    endif()\n'
    '    target_compile_definitions(${PY_NAME} PRIVATE ARK_MXFP_BDPAS_CUTLASS=1)\n'
    '    set_source_files_properties(\n'
    '      ${CMAKE_CURRENT_SOURCE_DIR}/sycl_tla_moe_prefill_mxfp8_mxfp4_bdpas.cpp\n'
    '      PROPERTIES\n'
    '        COMPILE_DEFINITIONS "SYCL_TARGET_INTEL_GPU_CRI;SYCL_INTEL_TARGET;CUTLASS_ENABLE_SYCL;CUTE_ENABLE_XE_BLOCK_2D_ASSERT"\n'
    '        COMPILE_OPTIONS "-std=c++17;-DCUTLASS_VERSIONS_GENERATED;-I${ARK_MXFP_BDPAS_CUTLASS_DIR}/include;-I${ARK_MXFP_BDPAS_CUTLASS_DIR}/build/include;-I${ARK_MXFP_BDPAS_CUTLASS_DIR}/tools/util/include")\n'
    '  endif()\n',
    "MXFP BDPAS block")

s = insert_after(s,
    '  target_include_directories(${PY_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated/sdpa)\n',
    '  if(ARK_ONEMKL_INSTALL_DIR AND EXISTS "${ARK_ONEMKL_INSTALL_DIR}/include")\n'
    '    target_include_directories(${PY_NAME} SYSTEM PRIVATE "${ARK_ONEMKL_INSTALL_DIR}/include")\n'
    '  endif()\n',
    "ARK_ONEMKL_INSTALL_DIR block")

assert "<<<<<<<" not in s and ">>>>>>>" not in s
open(p, "w", encoding="utf-8").write(s)
PY
git add "$CMAKE"

# ---------------------------------------------------------------------------
# 2) __init__.py：冲突块取 PR 侧（ours），并把块内 Optional[X] 改为 X | None
# ---------------------------------------------------------------------------
python3 - "$INIT" <<'PY'
import re, sys
p = sys.argv[1]
lines = open(p, encoding="utf-8").read().splitlines(keepends=True)

def deoptional(text):
    out, i = [], 0
    while True:
        j = text.find("Optional[", i)
        if j < 0:
            out.append(text[i:]); break
        out.append(text[i:j])
        k, depth = j + len("Optional["), 1
        while depth:
            c = text[k]
            depth += (c == "[") - (c == "]")
            k += 1
        inner = deoptional(text[j + len("Optional["):k - 1])
        out.append(f"{inner} | None")
        i = k
    return "".join(out)

out, state, ours, theirs, n = [], None, [], [], 0
for ln in lines:
    if ln.startswith("<<<<<<< "):
        state, ours, theirs = "ours", [], []
    elif ln.startswith("||||||| ") and state == "ours":
        state = "base"
    elif ln.startswith("=======") and state in ("ours", "base"):
        state = "theirs"
    elif ln.startswith(">>>>>>> ") and state == "theirs":
        n += 1
        block = deoptional("".join(ours))
        print(f"----- 冲突块 #{n}：采用 PR 侧（已转 X | None）-----")
        print(block, end="")
        out.append(block)
        state = None
    elif state == "ours":
        ours.append(ln)
    elif state == "theirs":
        theirs.append(ln)
    elif state == "base":
        pass
    else:
        out.append(ln)

assert state is None, "冲突标记不完整"
src = "".join(out)
assert "<<<<<<<" not in src and ">>>>>>>" not in src
compile(src, p, "exec")  # 语法检查
open(p, "w", encoding="utf-8").write(src)
print(f"[__init__.py] 共解决 {n} 个冲突块，语法检查通过")
PY

# 关键检查：PR 的 scale_dtype 改动与 main 的新增都在
for pat in 'scale_dtype_id' 'def moe_gemm_prefill_mxfp8_mxfp4' 'def moe_gemm_prefill_hmt_mxfp4_mxfp4' \
           '_MOE_PREFILL_MXFP8_MXFP4_WEIGHT_STAGING_CACHE.clear()' 'def _xpu_capturing' 'block_sparse_sdpa,'; do
  grep -q "$pat" "$INIT" || { echo "[check] 缺失: $pat"; exit 1; }
done
echo "[check] __init__.py 关键内容齐全"
git add "$INIT"

# ---------------------------------------------------------------------------
# 3) 全局检查并提交
# ---------------------------------------------------------------------------
if [ -n "$(git diff --name-only --diff-filter=U)" ]; then
  echo "仍有未解决文件："; git diff --name-only --diff-filter=U; exit 2
fi
if grep -rn '^<<<<<<< \|^>>>>>>> ' auto_round_extension/; then
  echo "仍有冲突标记"; exit 1
fi
echo "[check] MoePrefillParams 构造处（确认都带 scale_dtype）："
grep -rn "MoePrefillParams params{" auto_round_extension/ || true

git commit -m "Merge main into PR #2371; resolve CMake/__init__ conflicts; move MXFP BDPAS scratch slots to 13/14"
echo
echo "提交完成。接下来："
echo "  pre-commit run --all-files   # 若有自动修改，再 git add -A \u0026\u0026 git commit"
echo "  git push origin feat/mxfp8-mxfp4-moe-bdpas"
