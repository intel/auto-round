#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/data/xinhe/auto-round-main"
OUTPUT_ROOT="${AR_TOPK_BENCH_OUTPUT_ROOT:-/data/xinhe/opt_rtn_topk_bench}"
IDLE_MAX_UTIL="${AR_TOPK_IDLE_MAX_UTIL:-5}"
IDLE_MAX_MEMORY_MIB="${AR_TOPK_IDLE_MAX_MEMORY_MIB:-4096}"
POLL_SECONDS="${AR_TOPK_IDLE_POLL_SECONDS:-60}"
TOP1_GPUS="${AR_TOP1_GPUS:-0,1,2,3}"
TOP10_GPUS="${AR_TOP10_GPUS:-4,5,6,7}"
TASKS="lambada_openai,hellaswag,piqa,winogrande,truthfulqa_mc1,openbookqa,boolq,arc_easy,arc_challenge,mmlu"
MODELS=("Qwen/Qwen3-8B" "meta-llama/Llama-3.1-8B-Instruct")
RUN_ID="$(date +%Y%m%d_%H%M%S)"

cd /data/xinhe
source .bashrc
cd "$REPO_DIR"
export PATH="$REPO_DIR/.venv/bin:$PATH"

wait_for_all_gpus_idle() {
    local rows count busy
    while true; do
        rows="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"
        count="$(printf '%s\n' "$rows" | wc -l)"
        if [[ "$count" -ne 8 ]]; then
            printf 'Expected 8 GPUs, found %s; refusing to start.\n' "$count" >&2
            return 1
        fi
        busy="$(printf '%s\n' "$rows" | awk -F, -v max_mem="$IDLE_MAX_MEMORY_MIB" -v max_util="$IDLE_MAX_UTIL" '
            { gsub(/[[:space:]]/, "", $2); gsub(/[[:space:]]/, "", $3) }
            $2 > max_mem || $3 > max_util { print; found=1 }
            END { if (!found) exit 0; exit 1 }
        ' || true)"
        if [[ -z "$busy" ]]; then
            printf 'All 8 GPUs are idle.\n'
            return 0
        fi
        printf 'GPUs are still busy (threshold: <=%s MiB and <=%s%% utilization); checking again in %ss.\n' \
            "$IDLE_MAX_MEMORY_MIB" "$IDLE_MAX_UTIL" "$POLL_SECONDS"
        printf '%s\n' "$busy"
        sleep "$POLL_SECONDS"
    done
}

run_quantize_with_top_k() {
    local gpu_ids="$1"
    local top_k="$2"
    local model_log="$3"
    shift 3

    CUDA_VISIBLE_DEVICES="$gpu_ids" AR_SEARCH_SCALE_TOP_K="$top_k" "$REPO_DIR/.venv/bin/python" - "$@" \
        2>&1 <<'PY' | tee -a "$model_log"
import os
import sys

import auto_round.data_type.int as int_dtype

top_k = int(os.environ["AR_SEARCH_SCALE_TOP_K"])
original_search_scales = int_dtype.search_scales
print(f"[candidate-check] AR_SEARCH_SCALE_TOP_K={top_k}", flush=True)


def configured_search_scales(data, bits, qw=None, candidate_count=1):
    return original_search_scales(data, bits, qw=qw, candidate_count=top_k)


int_dtype.search_scales = configured_search_scales
from auto_round.__main__ import run

sys.argv = ["auto-round", *sys.argv[1:]]
run()
PY
}

run_group() {
    local top_k="$1"
    local gpu_ids="$2"
    local group_name="top${top_k}"
    local group_dir="$OUTPUT_ROOT/$RUN_ID/$group_name"
    mkdir -p "$group_dir"

    for model in "${MODELS[@]}"; do
        local model_slug="${model//\//_}"
        local output_dir="$group_dir/$model_slug"
        local model_log="$group_dir/${model_slug}.log"
        local quantized_model
        mkdir -p "$output_dir"

        printf '[%s] model=%s GPUs=%s K=%s tasks=%s\n' "$group_name" "$model" "$gpu_ids" "$top_k" "$TASKS" \
            | tee -a "$model_log"
        run_quantize_with_top_k "$gpu_ids" "$top_k" "$model_log" \
            --model_name "$model" \
            --scheme W4A16 \
            --format fake \
            --iters 0 \
            --device_map 0,1,2,3 \
            --enable_deterministic_algorithms \
            --enable_torch_compile \
            --tasks "$TASKS" \
            --output_dir "$output_dir" \
            2>&1 | tee -a "$model_log"

        quantized_model="$(find "$output_dir" -mindepth 1 -maxdepth 2 -type f -name config.json -printf '%h\n' -quit)"
        if [[ -z "$quantized_model" ]]; then
            printf 'Could not locate quantized model config under %s\n' "$output_dir" >&2
            return 1
        fi

        printf '[%s] model=%s task=gsm8k\n' "$group_name" "$model" | tee -a "$model_log"
        CUDA_VISIBLE_DEVICES="$gpu_ids" auto-round eval \
            --model "$quantized_model" \
            --device_map 0,1,2,3 \
            --tasks gsm8k \
            --seed 42 \
            2>&1 | tee -a "$model_log"
    done
}

printf 'Run ID: %s\nOutput: %s/%s\n' "$RUN_ID" "$OUTPUT_ROOT" "$RUN_ID"
printf 'TOP1 GPUs: %s; TOP10 GPUs: %s\n' "$TOP1_GPUS" "$TOP10_GPUS"
printf 'Waiting until all 8 GPUs satisfy the idle thresholds before launching.\n'
wait_for_all_gpus_idle

run_group 1 "$TOP1_GPUS" &
top1_pid=$!
run_group 10 "$TOP10_GPUS" &
top10_pid=$!

set +e
wait "$top1_pid"
top1_status=$?
wait "$top10_pid"
top10_status=$?
set -e

if [[ "$top1_status" -ne 0 || "$top10_status" -ne 0 ]]; then
    printf 'Benchmark failed: top1=%s top10=%s\n' "$top1_status" "$top10_status" >&2
    exit 1
fi

printf 'All top-1 and top-10 runs completed. Results: %s/%s\n' "$OUTPUT_ROOT" "$RUN_ID"