#!/usr/bin/env bash
# Resolve merge conflicts of intel/auto-round PR #2371 against main.
set -euo pipefail

PR_BRANCH="${PR_BRANCH:-$(gh pr view 2371 --repo intel/auto-round --json headRefName -q .headRefName 2>/dev/null || true)}"
if [[ -z "${PR_BRANCH}" ]]; then
  echo "请设置 PR_BRANCH=<PR #2371 的分支名> 后重试"; exit 1
fi

UTILS=auto_round_extension/ark/auto_round_kernel/wrapper/include/utils.hpp
BDPAS=auto_round_extension/ark/auto_round_kernel/sycl_tla_moe_prefill_mxfp8_mxfp4_bdpas.cpp

git fetch origin main "${PR_BRANCH}"
git checkout "${PR_BRANCH}"
git pull --ff-only origin "${PR_BRANCH}"

# 合并 main（冲突时不中断脚本）
git merge --no-ff --no-commit origin/main || true

# ---------------------------------------------------------------------------
# 1) utils.hpp：以 main 为准（main 占用 slot 12），MXFP 用 13/14，MaxLocNum=15
#    在 PR 分支上 merge main 时，--theirs = main
# ---------------------------------------------------------------------------
if git ls-files -u -- "${UTILS}" | grep -q .; then
  git checkout --theirs -- "${UTILS}"
fi
python3 - "${UTILS}" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
new_tail = (
    "  // prefill workspace (`kPrefillScratchLoc` in `sycl_tla_wrapper.hpp`);\n"
    "  // slots 13 and 14 are the MXFP block-scaled grouped-GEMM metadata and\n"
    "  // scheduler workspace buffers (`moe_mxfp_bdpas_detail::kMetadataScratchLoc` /\n"
    "  // `kWorkspaceScratchLoc` in `sycl_tla_moe_prefill_mxfp8_mxfp4_bdpas.cpp`).\n"
    "  static constexpr int MaxLocNum = 15;\n"
)
pat = re.compile(
    r"  // prefill workspace \(`kPrefillScratchLoc` in `sycl_tla_wrapper\.hpp`\)[.;]\n"
    r"(?:  //[^\n]*\n)*"
    r"  static constexpr int MaxLocNum = \d+;\n"
)
s2, n = pat.subn(new_tail, s, count=1)
if n == 0:
    sys.exit(f"[utils.hpp] 未找到 main 的 MaxLocNum 注释块，请手动修改 {p}")
open(p, "w", encoding="utf-8").write(s2)
print("[utils.hpp] MaxLocNum -> 15, MXFP slots 13/14")
PY
git add "${UTILS}"

# ---------------------------------------------------------------------------
# 2) BDPAS 新文件：槽位 12/13 -> 13/14
# ---------------------------------------------------------------------------
sed -i \
  -e 's/inline constexpr size_t kMetadataScratchLoc = 12;/inline constexpr size_t kMetadataScratchLoc = 13;/' \
  -e 's/inline constexpr size_t kWorkspaceScratchLoc = 13;/inline constexpr size_t kWorkspaceScratchLoc = 14;/' \
  "${BDPAS}"
grep -n "kMetadataScratchLoc = 13;" "${BDPAS}" >/dev/null || { echo "[bdpas] 修改失败"; exit 1; }
grep -n "kWorkspaceScratchLoc = 14;" "${BDPAS}" >/dev/null || { echo "[bdpas] 修改失败"; exit 1; }
git add "${BDPAS}"
echo "[bdpas] kMetadataScratchLoc=13, kWorkspaceScratchLoc=14"

# ---------------------------------------------------------------------------
# 3) 检查剩余冲突
# ---------------------------------------------------------------------------
REMAINING=$(git diff --name-only --diff-filter=U || true)
if [[ -n "${REMAINING}" ]]; then
  echo
  echo "以下文件仍有冲突，需要手动处理（原则：两边改动都保留）："
  echo "${REMAINING}"
  echo
  echo "处理完后执行："
  echo "  grep -rn '<<<<<<<\\|>>>>>>>' auto_round_extension/   # 应无输出"
  echo "  git add -A && git commit -m 'Merge main into PR #2371; move MXFP BDPAS scratch slots to 13/14'"
  echo "  pre-commit run --all-files && git push origin ${PR_BRANCH}"
  exit 2
fi

if grep -rn '<<<<<<<\|>>>>>>>' auto_round_extension/; then
  echo "仍有冲突标记，请检查"; exit 1
fi

git commit -m "Merge main into PR #2371; move MXFP BDPAS scratch slots to 13/14"
pre-commit run --all-files || { echo "pre-commit 有改动或报错，修复后再 commit/push"; exit 1; }
git push origin "${PR_BRANCH}"
echo "完成：PR #2371 已更新"
