#!/usr/bin/env bash

set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <physical_gpu> <nsamples> <num_inference_steps>" >&2
  exit 2
fi

GPU="$1"
NSAMPLES="$2"
STEPS="$3"

ROOT="${ROOT:-/home/user2/data/xixi}"
REPO="${REPO:-$ROOT/auto-round-svdquant}"
PY="${PY:-$ROOT/torch213-cu130-env/.venv/bin/python}"
MODEL="${MODEL:-$ROOT/FLUX.1-dev}"
DATASET="${DATASET:-$ROOT/coco2017-captions.tsv}"
TAG="n${NSAMPLES}-s${STEPS}"
OUT="${OUT:-$ROOT/autoround-flux-mxfp4-r32-smooth-r100-early-rtn-$TAG}"
LOG="${LOG:-$ROOT/autoround-flux-mxfp4-r32-smooth-r100-early-rtn-$TAG.log}"

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

cd "$REPO"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/torch213-cu130-env/.uv-cache}"
export TMPDIR="${TMPDIR:-$ROOT/torch213-cu130-env/.tmp}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export PYTHONPATH="$ROOT/torch213-cu130-env/nunchaku${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$UV_CACHE_DIR" "$TMPDIR" "$HF_HOME"

echo "GPU=$GPU nsamples=$NSAMPLES steps=$STEPS"
echo "Output: $OUT"
echo "Log: $LOG"

"$PY" -u -m auto_round \
  --model "$MODEL" \
  --model_dtype bf16 \
  --scheme MXFP4 \
  --algorithm rtn \
  --iters 0 \
  --disable_opt_rtn \
  --nsamples "$NSAMPLES" \
  --batch_size 1 \
  --dataset "$DATASET" \
  --num_inference_steps "$STEPS" \
  --enable_svdquant \
  --svdquant_rank 32 \
  --svdquant_residual_iters 100 \
  --enable_svdquant_residual_early_stop \
  --svdquant_residual_quant_method rtn \
  --enable_svdquant_smooth \
  --svdquant_smooth_num_grids 20 \
  --svdquant_low_rank_dtype bf16 \
  --svdquant_model_adapter flux \
  --format svdquant_nunchaku \
  --device 0 \
  --disable_low_cpu_mem_usage \
  --output_dir "$OUT" \
  2>&1 | tee "$LOG"

