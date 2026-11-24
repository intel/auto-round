#!/usr/bin/env bash
set -euo pipefail

# Quick W4A4 re-test after the activation-format and [-7, 7] fixes: one model, no GSM8K.
RUN_ROOT="${RUN_ROOT:-/data/xinhe/exp_w4a4_llama_quick_20261119}"
REPO=/data/xinhe/auto-round
PYTHON=/data/xinhe/.venv/bin/python
AUTO_ROUND=/data/xinhe/.venv/bin/auto-round
MODEL_KEY=llama31-8b
MODEL_PATH=/data/xinhe/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
SCHEMES=(
    EXP-FP4-UE5M3
    EXP-INT4-E4M3
    EXP-INT4-UE5M3
    EXP-UINT4-ZP-UE5M3
    EXP-FP4-BF16
    EXP-FP4-FP16
    EXP-INT4-UBF16
    EXP-INT4-UFP16
    EXP-UINT4-ZP-BF16
    EXP-UINT4-ZP-FP16
    EXP-INT4-BF16
    EXP-INT4-FP16
)

write_run_metadata() {
    local log_path="$1"
    local scheme="$2"
    local stage="$3"
    {
        printf 'canonical_scheme=%s\ncli_scheme=%s\n' "$scheme" "$scheme"
        printf 'weight_act=W4A4 (act format = weight format)\n'
        printf 'model_key=%s\nmodel_path=%s\nstage=%s\nscript=%s\nrun_root=%s\nhost=%s\n' \
            "$MODEL_KEY" "$MODEL_PATH" "$stage" "${BASH_SOURCE[0]}" "$RUN_ROOT" "$(hostname)"
        printf 'git_head=%s\n' "$(git -C "$REPO" rev-parse --short HEAD)"
        printf 'quantization=RTN,group_size=16,seed=42\n\n'
    } > "$log_path"
}

run_one() {
    local gpu="$1"
    local scheme="$2"
    local combo_dir="$RUN_ROOT/full/$MODEL_KEY/$scheme"
    local model_out="$combo_dir/model"
    local quant_log="$RUN_ROOT/logs/$MODEL_KEY-$scheme-quant.log"
    local eval_log="$RUN_ROOT/logs/$MODEL_KEY-$scheme-likelihood.log"
    local quant_model

    if [[ -f "$combo_dir/COMPLETE" ]]; then
        printf 'skip complete: %s/%s\n' "$MODEL_KEY" "$scheme"
        return 0
    fi
    mkdir -p "$combo_dir"
    if [[ ! -f "$combo_dir/QUANT_COMPLETE" ]]; then
        write_run_metadata "$quant_log" "$scheme" quantize
        "$PYTHON" -m auto_round "$MODEL_PATH" \
            --scheme "$scheme" \
            --format fake \
            --nsamples 128 \
            --seqlen 2048 \
            --batch_size 8 \
            --dataset NeelNanda/pile-10k \
            --seed 42 \
            --device_map 0 \
            --enable_deterministic_algorithms \
            --disable_opt_rtn \
            --iters 0 \
            --output_dir "$model_out" \
            >> "$quant_log" 2>&1 || return 1
        touch "$combo_dir/QUANT_COMPLETE"
    fi
    quant_model="$(find "$model_out" -maxdepth 3 -type f -name config.json -printf '%h\n' -quit)"
    if [[ -z "$quant_model" ]]; then
        printf 'missing quantized artifact: %s\n' "$model_out" >&2
        return 1
    fi
    cp "$quant_model/config.json" "$combo_dir/config.json"
    write_run_metadata "$eval_log" "$scheme" likelihood
    "$AUTO_ROUND" eval "$quant_model" \
        --tasks mmlu,hellaswag,piqa \
        --device_map 0 \
        --seed 42 \
        --eval_bs 32 \
        --eval_model_dtype bfloat16 \
        --eval_task_by_task \
        >> "$eval_log" 2>&1 || return 1
    printf 'canonical_scheme=%s\nmodel_key=%s\nlog_path=%s\n' "$scheme" "$MODEL_KEY" "$eval_log" > "$combo_dir/COMPLETE"
    # Shared /data is nearly full; keep config/logs, drop the fake weights once evaluated.
    rm -rf "$model_out"
    printf 'complete: %s/%s on gpu=%s\n' "$MODEL_KEY" "$scheme" "$gpu"
}

worker() {
    local gpu="$1"
    local status=0
    local job_index

    export HTTP_PROXY=http://127.0.0.1:2080
    export HTTPS_PROXY=http://127.0.0.1:2080
    export http_proxy="$HTTP_PROXY"
    export https_proxy="$HTTPS_PROXY"
    export HF_HOME=/data/xinhe/huggingface
    export HUGGINGFACE_HUB_CACHE=/data/xinhe/huggingface/hub
    export HF_DATASETS_CACHE=/data/xinhe/huggingface/datasets
    export XDG_CACHE_HOME=/data/xinhe/huggingface/.cache
    export HF_HUB_DISABLE_XET=1
    export PYTHONPATH="$REPO"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export TOKENIZERS_PARALLELISM=false
    export TMPDIR="$RUN_ROOT/tmp"
    cd "$RUN_ROOT"
    for ((job_index = gpu; job_index < ${#SCHEMES[@]}; job_index += NUM_GPUS)); do
        if ! run_one "$gpu" "${SCHEMES[$job_index]}"; then
            printf 'failed: %s/%s on gpu=%s\n' "$MODEL_KEY" "${SCHEMES[$job_index]}" "$gpu" >&2
            status=1
        fi
    done
    return "$status"
}

NUM_GPUS="${NUM_GPUS:-8}"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/tmp"
pids=()
for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    worker "$gpu" >> "$RUN_ROOT/orchestrator.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
printf 'all done, status=%s\n' "$status" >> "$RUN_ROOT/orchestrator.log"
exit "$status"
