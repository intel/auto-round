#!/usr/bin/env bash
# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -Eeuo pipefail

PROFILE="${1:-fast}"
GPUS="${2:-${GPUS:-0,1,2}}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(dirname "$SCRIPT_DIR")}" 
ROOT="${ROOT:-/mnt/disk3/changwa1}"
PY="${PY:-$ROOT/nunchaku-torch-cu130-ubuntu22/.venv/bin/python}"
MODEL="${MODEL:-$ROOT/Wan2.2-T2V-A14B-Diffusers}"
DATASET="${DATASET:-$ROOT/coco2017-captions.tsv}"
RANK="${RANK:-32}"

case "$PROFILE" in
  smoke)
    DEFAULT_RESIDUAL_ITERS=1
    DEFAULT_NSAMPLES=1
    DEFAULT_STEPS=2
    DEFAULT_SIGNROUND_ITERS=1
    DEFAULT_SMOOTH_GRIDS=2
    DEFAULT_SMOOTH_CALLS=1
    ;;
  fast)
    DEFAULT_RESIDUAL_ITERS=1
    DEFAULT_NSAMPLES=2
    DEFAULT_STEPS=2
    DEFAULT_SIGNROUND_ITERS=20
    DEFAULT_SMOOTH_GRIDS=6
    DEFAULT_SMOOTH_CALLS=4
    ;;
  quality)
    DEFAULT_RESIDUAL_ITERS=3
    DEFAULT_NSAMPLES=8
    DEFAULT_STEPS=4
    DEFAULT_SIGNROUND_ITERS=200
    DEFAULT_SMOOTH_GRIDS=20
    DEFAULT_SMOOTH_CALLS=16
    ;;
  *)
    echo "Usage: $0 [smoke|fast|quality] [GPU_LIST]" >&2
    exit 2
    ;;
esac

RESIDUAL_ITERS="${RESIDUAL_ITERS:-$DEFAULT_RESIDUAL_ITERS}"
NSAMPLES="${NSAMPLES:-$DEFAULT_NSAMPLES}"
STEPS="${STEPS:-$DEFAULT_STEPS}"
SIGNROUND_ITERS="${SIGNROUND_ITERS:-$DEFAULT_SIGNROUND_ITERS}"
SMOOTH_GRIDS="${SMOOTH_GRIDS:-$DEFAULT_SMOOTH_GRIDS}"
SMOOTH_MAX_CALLS="${SMOOTH_MAX_CALLS:-$DEFAULT_SMOOTH_CALLS}"

TAG="b200-smooth-signround-r${RANK}-ri${RESIDUAL_ITERS}-n${NSAMPLES}-s${STEPS}-i${SIGNROUND_ITERS}-g${SMOOTH_GRIDS}-c${SMOOTH_MAX_CALLS}"
OUT="${OUT:-$ROOT/Wan2.2-T2V-A14B-AutoRound-SVDQuant-MXFP4-$TAG-Nunchaku}"
LOG="${LOG:-$ROOT/launch-logs/wan22-$TAG.log}"

for path in "$REPO" "$PY" "$MODEL" "$DATASET"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path does not exist: $path" >&2
    exit 1
  fi
done
if [[ -e "$OUT" ]]; then
  echo "Output already exists; refusing to overwrite: $OUT" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG")" "$ROOT/.cache/huggingface"
cd "$REPO"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPUS"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export PYTHONPATH="$ROOT/nunchaku-torch-cu130-ubuntu22/nunchaku:$REPO${PYTHONPATH:+:$PYTHONPATH}"

printf 'profile=%s physical_gpus=%s\n' "$PROFILE" "$GPUS"
printf 'residual_iters=%s nsamples=%s steps=%s signround_iters=%s smooth_grids=%s smooth_calls=%s\n' \
  "$RESIDUAL_ITERS" "$NSAMPLES" "$STEPS" "$SIGNROUND_ITERS" "$SMOOTH_GRIDS" "$SMOOTH_MAX_CALLS"
printf 'output=%s\nlog=%s\n' "$OUT" "$LOG"

"$PY" -u -m auto_round \
  --model "$MODEL" \
  --model_dtype bf16 \
  --scheme MXFP4 \
  --algorithm signround,svdquant \
  --iters "$SIGNROUND_ITERS" \
  --nsamples "$NSAMPLES" \
  --batch_size 1 \
  --dataset "$DATASET" \
  --num_inference_steps "$STEPS" \
  --svdquant-rank "$RANK" \
  --svdquant-residual-iters "$RESIDUAL_ITERS" \
  --enable-svdquant-residual-early-stop \
  --enable-svdquant-smooth \
  --svdquant-smooth-num-grids "$SMOOTH_GRIDS" \
  --svdquant-smooth-max-calibration-calls "$SMOOTH_MAX_CALLS" \
  --svdquant-low-rank-dtype bf16 \
  --svdquant-model-adapter wan \
  --format svdquant_nunchaku \
  --device auto \
  --disable_low_cpu_mem_usage \
  --output_dir "$OUT" \
  2>&1 | tee "$LOG"
