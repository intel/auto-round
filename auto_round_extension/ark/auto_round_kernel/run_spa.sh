#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# oneAPI setvars.sh is not compatible with `set -u`; relax nounset only for env bootstrap.
# set +u
# source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
# source /home/yiliu7/workspace/venvs/diffuser/bin/activate
# set -u

TOPKS=("${@}")
if [ ${#TOPKS[@]} -eq 0 ]; then
  TOPKS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
fi

WAN_STEPS="${WAN_STEPS:-50}"
WAN_HEIGHT="${WAN_HEIGHT:-480}"
WAN_WIDTH="${WAN_WIDTH:-832}"
WAN_NUM_FRAMES="${WAN_NUM_FRAMES:-81}"
WAN_GUIDANCE_SCALE="${WAN_GUIDANCE_SCALE:-5.0}"
WAN_OUTPUT_DIR="${WAN_OUTPUT_DIR:-${SCRIPT_DIR}}"

mkdir -p "${WAN_OUTPUT_DIR}"

for topk in "${TOPKS[@]}"; do
  output_file="${WAN_OUTPUT_DIR}/wan_sparse_topk_${topk}.mp4"
  log_file="${WAN_OUTPUT_DIR}/wan_sparse_topk_${topk}.log"

  echo "[$(date '+%F %T')] [run_spa] start topk=${topk} output=${output_file}"
  echo "[$(date '+%F %T')] [run_spa] log=${log_file} steps=${WAN_STEPS} frames=${WAN_NUM_FRAMES} size=${WAN_HEIGHT}x${WAN_WIDTH}"
  WAN_USE_SPARSE=1 \
  WAN_SPARSE_TOPK="${topk}" \
  WAN_STEPS="${WAN_STEPS}" \
  WAN_HEIGHT="${WAN_HEIGHT}" \
  WAN_WIDTH="${WAN_WIDTH}" \
  WAN_NUM_FRAMES="${WAN_NUM_FRAMES}" \
  WAN_GUIDANCE_SCALE="${WAN_GUIDANCE_SCALE}" \
  WAN_OUTPUT="${output_file}" \
  PYTHONUNBUFFERED=1 stdbuf -oL -eL python -u run_wan.py 2>&1 | tee "${log_file}"
  echo "[$(date '+%F %T')] [run_spa] finished topk=${topk}"
done
