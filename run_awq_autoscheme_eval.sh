#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# AutoScheme + SFMP-style AWQ (fake quant) for Qwen3, then lm-eval.
# Loops over a list of models; for each model & bits-config:
#   1. Runs autoscheme_awq_qwen3.py to fake-quantize the model.
#   2. Evaluates the saved fake-quantized model with lm-eval.
#
# Usage:
#   bash run_awq_autoscheme_eval.sh

set -euo pipefail

# ── configuration ──────────────────────────────────────────────────────────────
DEVICE=${DEVICE:-3}            # CUDA device index used for both quant & eval
EVAL_BS=${EVAL_BS:-32}         # lm-eval batch size
GROUP_SIZE=${GROUP_SIZE:-128}  # weight quantization group size
NSAMPLES=${NSAMPLES:-128}      # number of calibration samples
SEQLEN=${SEQLEN:-512}          # calibration sequence length
DTYPE=${DTYPE:-"bfloat16"}     # model dtype: float16 | bfloat16
OUTPUT_ROOT=${OUTPUT_ROOT:-"/storage/wenhuach/awq_autoscheme_fake_asym"}
LOG_FILE=${LOG_FILE:-"awq_autoscheme_eval_asym.txt"}

# ── model list ─────────────────────────────────────────────────────────────────
MODEL_LIST=(
  "/storage/models/Qwen3-8B"
  "/storage/models/Qwen3-14B"
#  "/storage/models/Qwen3-32B"
)

# ── eval tasks ─────────────────────────────────────────────────────────────────
MMLU_FEWSHOT=5

# ── bits config list: "target_bits|options" ───────────────────────────────────
BITS_CONFIGS=(
  "2.5|W2A16,W3A16"
  "3.0|W2A16,W3A16"
  "3.5|W3A16,W4A16"
  "4.0|W3A16,W4A16"
)

# ── helpers ────────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"; }
sep() { echo "==========================================================" | tee -a "${LOG_FILE}"; }

# ── main loop ──────────────────────────────────────────────────────────────────
for model_name in "${MODEL_LIST[@]}"; do
  model_tag=$(basename "${model_name}")

  for bits_cfg in "${BITS_CONFIGS[@]}"; do
    target_bits="${bits_cfg%%|*}"
    options="${bits_cfg##*|}"
    bits_tag="${target_bits//./p}bits"   # e.g. 2.5 → 2p5bits

    sep
    log "Model: ${model_name}  |  target_bits=${target_bits}  |  options=${options}"
    sep

    output_dir="${OUTPUT_ROOT}/${model_tag}/${bits_tag}"
    mkdir -p "${output_dir}"

    # ── step 1: quantize (AutoScheme bits + SFMP-style AWQ, fake) ─────────────
    log "Step 1/2 — quantization"

    CUDA_VISIBLE_DEVICES=${DEVICE} python3 autoscheme_awq_qwen3.py \
      --model      "${model_name}" \
      --avg_bits   "${target_bits}" \
      --options    "${options}" \
      --group_size "${GROUP_SIZE}" \
      --nsamples   "${NSAMPLES}" \
      --quant_sym asym \
      --seqlen     "${SEQLEN}" \
      --dtype      "${DTYPE}" \
      --device     0 \
      --save_dir   "${output_dir}" \
      2>&1 | tee -a "${LOG_FILE}"

    log "Quantization done → ${output_dir}"

    quant_model_dir="${output_dir}"
    log "Quantized model directory: ${quant_model_dir}"

    # ── step 2a: MMLU — 5-shot ───────────────────────────────────────────────
    log "Step 2a/2 — lm-eval MMLU (5-shot)"

    CUDA_VISIBLE_DEVICES=${DEVICE} lm-eval \
      --model hf \
      --model_args "pretrained=${quant_model_dir},trust_remote_code=True" \
      --tasks       mmlu \
      --num_fewshot "${MMLU_FEWSHOT}" \
      --batch_size  "${EVAL_BS}" \
      2>&1 | tee -a "${LOG_FILE}"

    # ── step 2b: GSM8K + commonsense — default fewshot ───────────────────────
    log "Step 2b/2 — lm-eval GSM8K + commonsense"

    CUDA_VISIBLE_DEVICES=${DEVICE} lm-eval \
      --model hf \
      --model_args "pretrained=${quant_model_dir},trust_remote_code=True" \
      --tasks       "gsm8k,arc_challenge,arc_easy,piqa,boolq,hellaswag,winogrande" \
      --batch_size  "${EVAL_BS}" \
      2>&1 | tee -a "${LOG_FILE}"

    log "Evaluation done for: ${model_name} @ ${target_bits}bits"
  done
done

sep
log "All models finished. Full log: ${LOG_FILE}"

