#!/usr/bin/env bash
set -Eeuo pipefail

# Recommended 8B MXFP4 W4A4 fake-format HF lm-eval runs for comparing refined
# TEQ+AutoRound against plain AutoRound. Supports Qwen3-8B and Llama 3 8B
# Instruct-style model paths.
#
# Usage:
#   MODEL_PRESET=qwen3-8b scripts/run_qwen3_8b_teq_ar_compare.sh main
#   MODEL_PRESET=llama3-8b-instruct scripts/run_qwen3_8b_teq_ar_compare.sh main
#   MODEL_PRESET=both scripts/run_qwen3_8b_teq_ar_compare.sh main
#   scripts/run_qwen3_8b_teq_ar_compare.sh main
#   scripts/run_qwen3_8b_teq_ar_compare.sh ar-only
#   scripts/run_qwen3_8b_teq_ar_compare.sh teq-ar-only
#   scripts/run_qwen3_8b_teq_ar_compare.sh ablation-no-awq-init
#   scripts/run_qwen3_8b_teq_ar_compare.sh ablation-longer-teq
#   scripts/run_qwen3_8b_teq_ar_compare.sh ablation-low-lr
#   scripts/run_qwen3_8b_teq_ar_compare.sh ablation-high-lr
#   scripts/run_qwen3_8b_teq_ar_compare.sh ablation-few-iters
#   scripts/run_qwen3_8b_teq_ar_compare.sh ablation-more-iters
#   scripts/run_qwen3_8b_teq_ar_compare.sh ablation-fewer-samples
#   scripts/run_qwen3_8b_teq_ar_compare.sh ablation-more-samples
#   scripts/run_qwen3_8b_teq_ar_compare.sh ablation-long-seq
#   scripts/run_qwen3_8b_teq_ar_compare.sh ablation-sqrt-init
#   scripts/run_qwen3_8b_teq_ar_compare.sh ablation-scale-tight
#   scripts/run_qwen3_8b_teq_ar_compare.sh teq-ablations
#   scripts/run_qwen3_8b_teq_ar_compare.sh all
#
# Override GPUS/MODELS/TASKS/EVAL_BS/etc. via environment variables as needed.
# If the local Llama 3 8B Instruct path is different, set
# LLAMA3_8B_INSTRUCT_PATH=/path/to/model.

AUTOROUND_ROOT=${AUTOROUND_ROOT:-/data6/zww/round_teq}
BENCH_SCRIPT=${BENCH_SCRIPT:-"$AUTOROUND_ROOT/scripts/bench_teq_mxfp4_vllm.sh"}
AR_VARIANT=${AR_VARIANT:-auto-round}
TEQ_AR_VARIANT=${TEQ_AR_VARIANT:-teq-ar}
COMPARE_VARIANTS=${VARIANTS:-"$AR_VARIANT $TEQ_AR_VARIANT"}
SOURCE_ROOT_EXPECTED=${SOURCE_ROOT_EXPECTED:-"$AUTOROUND_ROOT/auto_round"}
MODEL_PRESET=${MODEL_PRESET:-qwen3-8b}
QWEN3_8B_PATH=${QWEN3_8B_PATH:-/models/Qwen3-8B}
LLAMA3_8B_INSTRUCT_PATH=${LLAMA3_8B_INSTRUCT_PATH:-}

resolve_llama3_8b_instruct_path() {
  if [[ -n "$LLAMA3_8B_INSTRUCT_PATH" ]]; then
    echo "$LLAMA3_8B_INSTRUCT_PATH"
    return
  fi

  local candidate
  for candidate in \
    /models/Llama-3.1-8B-Instruct \
    /models/Meta-Llama-3.1-8B-Instruct \
    /models/Meta-Llama-3-8B-Instruct \
    /models/Llama-3.1-8B \
    /models/Meta-Llama-3-8B; do
    if [[ -d "$candidate" ]]; then
      echo "$candidate"
      return
    fi
  done

  echo /models/Llama-3.1-8B-Instruct
}

resolve_models() {
  if [[ -n "${MODELS:-}" ]]; then
    echo "$MODELS"
    return
  fi

  local llama_path
  llama_path="$(resolve_llama3_8b_instruct_path)"
  case "$MODEL_PRESET" in
    qwen3-8b|qwen3)
      echo "qwen3-8b=$QWEN3_8B_PATH"
      ;;
    llama3-8b-instruct|llama3-8b|llama3)
      echo "llama3-8b-instruct=$llama_path"
      ;;
    both|qwen3-llama3)
      echo "qwen3-8b=$QWEN3_8B_PATH llama3-8b-instruct=$llama_path"
      ;;
    *)
      echo "Unknown MODEL_PRESET='$MODEL_PRESET'. Use qwen3-8b, llama3-8b-instruct, both, or set MODELS." >&2
      exit 2
      ;;
  esac
}

SELECTED_MODELS="$(resolve_models)"

COMMON_ENV=(
  AUTOROUND_ROOT="$AUTOROUND_ROOT"
  PYTHONPATH="$AUTOROUND_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  KEEP_MODELS=0
  KEEP_FAILED_MODELS=0
  ENABLE_TORCH_COMPILE=1
  LOW_GPU_MEM_USAGE=0
  LOW_CPU_MEM_USAGE=0
  GPUS="${GPUS:-0}"
  MODELS="$SELECTED_MODELS"
  SCHEME=MXFP4
  FORMAT=fake
  EVAL_BACKEND=hf
  NSAMPLES="${NSAMPLES:-128}"
  SEQLEN="${SEQLEN:-2048}"
  BATCH_SIZE="${BATCH_SIZE:-8}"
  ITERS="${ITERS:-200}"
  EVAL_BS="${EVAL_BS:-8}"
  TASKS="${TASKS:-mmlu,gsm8k,piqa,hellaswag,winogrande}"
)

run_case() {
  local name=$1
  local variants=$2
  shift 2

  echo "Running ${name}"
  echo "Variants: ${variants} (plain AR is '${AR_VARIANT}', TEQ+AR is '${TEQ_AR_VARIANT}')"
  env "${COMMON_ENV[@]}" VARIANTS="$variants" "$@" "$BENCH_SCRIPT" run
}

run_teq_ar_case() {
  local name=$1
  shift

  run_case "$name" "$COMPARE_VARIANTS" "$@"
}

run_teq_only_case() {
  local name=$1
  shift

  run_case "$name" "$TEQ_AR_VARIANT" "$@"
}

check_autoround_source() {
  local actual
  actual="$(
    PYTHONPATH="$AUTOROUND_ROOT${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
from pathlib import Path
import auto_round

print(Path(auto_round.__file__).resolve())
PY
  )"

  case "$actual" in
    "$SOURCE_ROOT_EXPECTED"/*)
      echo "AutoRound source: $actual"
      ;;
    *)
      echo "ERROR: auto_round is imported from '$actual', expected under '$SOURCE_ROOT_EXPECTED'." >&2
      exit 3
      ;;
  esac
}

check_model_paths() {
  local item
  local name
  local path
  for item in $SELECTED_MODELS; do
    name="${item%%=*}"
    path="${item#*=}"
    if [[ "$name" == "$path" || -z "$path" ]]; then
      echo "ERROR: invalid MODELS entry '$item'. Expected name=/path." >&2
      exit 2
    fi
    if [[ ! -d "$path" ]]; then
      echo "ERROR: model path for '$name' does not exist: $path" >&2
      echo "Set MODELS='name=/path' or LLAMA3_8B_INSTRUCT_PATH=/path/to/model." >&2
      exit 4
    fi
  done

  if [[ "$MODEL_PRESET" == llama3* && "$SELECTED_MODELS" != *Instruct* ]]; then
    echo "Warning: no local Llama 3 8B Instruct directory was found; using: $SELECTED_MODELS" >&2
    echo "Set LLAMA3_8B_INSTRUCT_PATH=/path/to/instruct/model if you need the instruct checkpoint." >&2
  fi
}

check_autoround_source
check_model_paths

run_mode() {
  local mode="${1:-main}"

  case "$mode" in
  main)
    run_teq_ar_case \
      "main: AR vs TEQ+AR with AWQ init" \
      TEQ_AWQ_INIT=1 \
      TEQ_AWQ_INIT_N_GRID=20 \
      TEQ_ITERS=20 \
      TEQ_LR=5e-4 \
      TEQ_NSAMPLES=32 \
      TEQ_BATCH_SIZE=1 \
      TEQ_SAMPLE_SEQLEN=512
    ;;
  ar-only)
    run_case \
      "plain AR only" \
      "$AR_VARIANT"
    ;;
  teq-ar-only)
    run_case \
      "TEQ+AR only with AWQ init" \
      "$TEQ_AR_VARIANT" \
      TEQ_AWQ_INIT=1 \
      TEQ_AWQ_INIT_N_GRID=20 \
      TEQ_ITERS=20 \
      TEQ_LR=5e-4 \
      TEQ_NSAMPLES=32 \
      TEQ_BATCH_SIZE=1 \
      TEQ_SAMPLE_SEQLEN=512
    ;;
  ablation-no-awq-init)
    run_teq_only_case \
      "ablation: TEQ+AR without AWQ init" \
      TEQ_AWQ_INIT=0 \
      TEQ_ITERS=20 \
      TEQ_LR=5e-4 \
      TEQ_NSAMPLES=32 \
      TEQ_BATCH_SIZE=1 \
      TEQ_SAMPLE_SEQLEN=512
    ;;
  ablation-longer-teq)
    run_teq_only_case \
      "ablation: TEQ+AR with larger TEQ replay context" \
      TEQ_AWQ_INIT=0 \
      TEQ_ITERS=20 \
      TEQ_LR=5e-4 \
      TEQ_NSAMPLES=64 \
      TEQ_BATCH_SIZE=1 \
      TEQ_SAMPLE_SEQLEN=1024
    ;;
  ablation-low-lr)
    run_teq_only_case \
      "ablation: TEQ+AR low TEQ learning rate" \
      TEQ_AWQ_INIT=0 \
      TEQ_ITERS=20 \
      TEQ_LR=2e-4 \
      TEQ_NSAMPLES=32 \
      TEQ_BATCH_SIZE=1 \
      TEQ_SAMPLE_SEQLEN=512
    ;;
  ablation-high-lr)
    run_teq_only_case \
      "ablation: TEQ+AR high TEQ learning rate" \
      TEQ_AWQ_INIT=0 \
      TEQ_ITERS=20 \
      TEQ_LR=1e-3 \
      TEQ_NSAMPLES=32 \
      TEQ_BATCH_SIZE=1 \
      TEQ_SAMPLE_SEQLEN=512
    ;;
  ablation-few-iters)
    run_teq_only_case \
      "ablation: TEQ+AR fewer TEQ iterations" \
      TEQ_AWQ_INIT=0 \
      TEQ_ITERS=10 \
      TEQ_LR=5e-4 \
      TEQ_NSAMPLES=32 \
      TEQ_BATCH_SIZE=1 \
      TEQ_SAMPLE_SEQLEN=512
    ;;
  ablation-more-iters)
    run_teq_only_case \
      "ablation: TEQ+AR more TEQ iterations" \
      TEQ_AWQ_INIT=0 \
      TEQ_ITERS=40 \
      TEQ_LR=5e-4 \
      TEQ_NSAMPLES=32 \
      TEQ_BATCH_SIZE=1 \
      TEQ_SAMPLE_SEQLEN=512
    ;;
  ablation-fewer-samples)
    run_teq_only_case \
      "ablation: TEQ+AR fewer TEQ replay samples" \
      TEQ_AWQ_INIT=0 \
      TEQ_ITERS=20 \
      TEQ_LR=5e-4 \
      TEQ_NSAMPLES=16 \
      TEQ_BATCH_SIZE=1 \
      TEQ_SAMPLE_SEQLEN=512
    ;;
  ablation-more-samples)
    run_teq_only_case \
      "ablation: TEQ+AR more TEQ replay samples" \
      TEQ_AWQ_INIT=0 \
      TEQ_ITERS=20 \
      TEQ_LR=5e-4 \
      TEQ_NSAMPLES=64 \
      TEQ_BATCH_SIZE=1 \
      TEQ_SAMPLE_SEQLEN=512
    ;;
  ablation-long-seq)
    run_teq_only_case \
      "ablation: TEQ+AR longer TEQ replay sequence" \
      TEQ_AWQ_INIT=0 \
      TEQ_ITERS=20 \
      TEQ_LR=5e-4 \
      TEQ_NSAMPLES=32 \
      TEQ_BATCH_SIZE=1 \
      TEQ_SAMPLE_SEQLEN=1024
    ;;
  ablation-sqrt-init)
    run_teq_only_case \
      "ablation: TEQ+AR sqrt-weight init instead of AWQ init" \
      TEQ_AWQ_INIT=0 \
      TEQ_SQRT_W_INIT=1 \
      TEQ_ITERS=20 \
      TEQ_LR=5e-4 \
      TEQ_NSAMPLES=32 \
      TEQ_BATCH_SIZE=1 \
      TEQ_SAMPLE_SEQLEN=512
    ;;
  ablation-scale-tight)
    run_teq_only_case \
      "ablation: TEQ+AR tighter TEQ scale bounds" \
      TEQ_AWQ_INIT=0 \
      TEQ_ITERS=20 \
      TEQ_LR=5e-4 \
      TEQ_NSAMPLES=32 \
      TEQ_BATCH_SIZE=1 \
      TEQ_SAMPLE_SEQLEN=512 \
      TEQ_MIN_SCALE=0.05 \
      TEQ_MAX_SCALE=3.0
    ;;
  teq-ablations)
    # run_mode ar-only
    # run_mode ablation-no-awq-init
    run_mode ablation-longer-teq
    run_mode ablation-low-lr
    run_mode ablation-high-lr
    run_mode ablation-few-iters
    run_mode ablation-more-iters
    run_mode ablation-fewer-samples
    run_mode ablation-more-samples
    run_mode ablation-long-seq
    run_mode ablation-sqrt-init
    run_mode ablation-scale-tight
    ;;
  all)
    run_mode main
    run_mode ablation-no-awq-init
    run_mode ablation-longer-teq
    run_mode ablation-low-lr
    run_mode ablation-high-lr
    run_mode ablation-few-iters
    run_mode ablation-more-iters
    run_mode ablation-fewer-samples
    run_mode ablation-more-samples
    run_mode ablation-long-seq
    run_mode ablation-sqrt-init
    run_mode ablation-scale-tight
    ;;
  *)
    echo "Usage: $0 {main|ar-only|teq-ar-only|ablation-no-awq-init|ablation-longer-teq|ablation-low-lr|ablation-high-lr|ablation-few-iters|ablation-more-iters|ablation-fewer-samples|ablation-more-samples|ablation-long-seq|ablation-sqrt-init|ablation-scale-tight|teq-ablations|all}" >&2
    exit 2
    ;;
  esac
}

run_mode "${1:-main}"
