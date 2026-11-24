#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/xinhe/exp_w4a4_llama_quick_20261119
RUN_ROOT="$ROOT/recovery_h20_120"
GSM_ROOT="$ROOT/gsm8k"
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
RETRY_SCHEMES=(EXP-UINT4-ZP-BF16 EXP-UINT4-ZP-FP16 EXP-INT4-BF16 EXP-INT4-FP16)

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

setup_env() {
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
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export TOKENIZERS_PARALLELISM=false
    export TMPDIR="$RUN_ROOT/tmp"
}

run_likelihood() {
    local gpu="$1" scheme="$2"
    local model_dir="$ROOT/full/$MODEL_KEY/$scheme/model"
    local quant_model
    local log_file="$RUN_ROOT/logs/$MODEL_KEY-$scheme-likelihood.log"
    export CUDA_VISIBLE_DEVICES="$gpu"
    quant_model="$(find "$model_dir" -maxdepth 3 -type f -name config.json -printf '%h\n' -quit)"
    if [[ -z "$quant_model" ]]; then
        log "missing quantized model: $scheme" >&2
        return 1
    fi
    {
        printf 'canonical_scheme=%s\nhost=%s\ngpu=%s\nweight_act=W4A4\nstage=likelihood-recovery\n' "$scheme" "$(hostname)" "$gpu"
        printf 'git_head=%s\n\n' "$(git -C "$REPO" rev-parse --short HEAD)"
    } > "$log_file"
    "$AUTO_ROUND" eval "$quant_model" \
        --tasks mmlu,hellaswag,piqa \
        --device_map 0 \
        --seed 42 \
        --eval_bs 16 \
        --eval_model_dtype bfloat16 \
        --eval_task_by_task \
        >> "$log_file" 2>&1
    printf 'canonical_scheme=%s\nmodel_key=%s\nlog_path=%s\n' "$scheme" "$MODEL_KEY" "$log_file" \
        > "$RUN_ROOT/complete/$scheme"
    log "likelihood complete: $scheme on gpu=$gpu"
}

run_gsm8k() {
    local gpu="$1" scheme="$2"
    local combo_dir="$GSM_ROOT/full/$MODEL_KEY/$scheme"
    local model_out="$combo_dir/model"
    local quant_log="$GSM_ROOT/logs/$MODEL_KEY-$scheme-quant.log"
    local gsm_log="$GSM_ROOT/logs/$MODEL_KEY-$scheme-gsm8k.log"
    local quant_model
    export CUDA_VISIBLE_DEVICES="$gpu"
    mkdir -p "$combo_dir" "$GSM_ROOT/logs"
    if [[ ! -f "$combo_dir/QUANT_COMPLETE" ]]; then
        {
            printf 'canonical_scheme=%s\nhost=%s\ngpu=%s\nweight_act=W4A4\nstage=quantize\n' "$scheme" "$(hostname)" "$gpu"
            printf 'git_head=%s\n\n' "$(git -C "$REPO" rev-parse --short HEAD)"
        } > "$quant_log"
        "$PYTHON" -m auto_round "$MODEL_PATH" \
            --scheme "$scheme" --format fake --nsamples 128 --seqlen 2048 \
            --batch_size 8 --dataset NeelNanda/pile-10k --seed 42 --device_map 0 \
            --enable_deterministic_algorithms --disable_opt_rtn --iters 0 \
            --output_dir "$model_out" >> "$quant_log" 2>&1
        touch "$combo_dir/QUANT_COMPLETE"
    fi
    quant_model="$(find "$model_out" -maxdepth 3 -type f -name config.json -printf '%h\n' -quit)"
    [[ -n "$quant_model" ]] || { log "missing GSM8K model: $scheme" >&2; return 1; }
    {
        printf 'canonical_scheme=%s\nhost=%s\ngpu=%s\nweight_act=W4A4\nstage=gsm8k\n' "$scheme" "$(hostname)" "$gpu"
        printf 'git_head=%s\n\n' "$(git -C "$REPO" rev-parse --short HEAD)"
    } > "$gsm_log"
    "$AUTO_ROUND" eval "$quant_model" --tasks gsm8k --device_map 0 --seed 42 \
        --eval_bs 8 --num_fewshot 5 --eval_model_dtype bfloat16 \
        --eval_gen_kwargs max_gen_toks=1024,do_sample=False >> "$gsm_log" 2>&1
    printf 'canonical_scheme=%s\nmodel_key=%s\ngsm8k_log=%s\n' "$scheme" "$MODEL_KEY" "$gsm_log" \
        > "$combo_dir/COMPLETE"
    rm -rf "$model_out"
    log "GSM8K complete: $scheme on gpu=$gpu"
}

likelihood_worker() {
    setup_env
    local gpu="$1" scheme="$2"
    run_likelihood "$gpu" "$scheme"
}

gsm_worker() {
    setup_env
    local gpu="$1" scheme="$2"
    run_gsm8k "$gpu" "$scheme"
}

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/complete" "$RUN_ROOT/tmp" "$GSM_ROOT/logs" "$GSM_ROOT/tmp"
cd "$RUN_ROOT"

likelihood_pids=()
for i in "${!RETRY_SCHEMES[@]}"; do
    gpu="$i"
    scheme="${RETRY_SCHEMES[$i]}"
    likelihood_worker "$gpu" "$scheme" &
    likelihood_pids+=("$!")
    log "deployed likelihood: $scheme on gpu=$gpu pid=${likelihood_pids[-1]}"
done

for pid in "${likelihood_pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
        log "likelihood worker exited before GSM8K deployment: pid=$pid" >&2
        exit 1
    fi
done
log "all four missing likelihood jobs deployed; GSM8K begins on separate GPUs 4-7"

for i in 0 1 2 3; do
    (
        setup_env
        gpu=$((i + 4))
        for ((idx = i; idx < ${#SCHEMES[@]}; idx += 4)); do
            scheme="${SCHEMES[$idx]}"
            if [[ -f "$GSM_ROOT/full/$MODEL_KEY/$scheme/COMPLETE" ]]; then
                log "skip GSM8K complete: $scheme"
                continue
            fi
            if ! run_gsm8k "$gpu" "$scheme"; then
                log "GSM8K failed: $scheme on gpu=$gpu" >&2
            fi
        done
    ) &
done
wait
for pid in "${likelihood_pids[@]}"; do
    wait "$pid"
done
log "recovery and GSM8K workers finished"
