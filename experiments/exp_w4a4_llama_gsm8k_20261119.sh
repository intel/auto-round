#!/usr/bin/env bash
set -euo pipefail

# GSM8K follow-up for exp_w4a4_llama_quick_20261119: starts only after all 12 quick runs are deployed,
# then fills GPUs that have no compute process.
QUICK_ROOT="${QUICK_ROOT:-/data/xinhe/exp_w4a4_llama_quick_20261119}"
RUN_ROOT="${RUN_ROOT:-$QUICK_ROOT/gsm8k}"
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
NUM_GPUS="${NUM_GPUS:-8}"
POLL_SECONDS="${POLL_SECONDS:-60}"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

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
    local scheme="$1"
    local combo_dir="$RUN_ROOT/full/$MODEL_KEY/$scheme"
    local model_out="$combo_dir/model"
    local quant_log="$RUN_ROOT/logs/$MODEL_KEY-$scheme-quant.log"
    local gsm8k_log="$RUN_ROOT/logs/$MODEL_KEY-$scheme-gsm8k.log"
    local quant_model

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
    [[ -n "$quant_model" ]] || { printf 'missing quantized artifact: %s\n' "$model_out" >&2; return 1; }
    cp "$quant_model/config.json" "$combo_dir/config.json"
    write_run_metadata "$gsm8k_log" "$scheme" gsm8k
    "$AUTO_ROUND" eval "$quant_model" \
        --tasks gsm8k \
        --device_map 0 \
        --seed 42 \
        --eval_bs 8 \
        --num_fewshot 5 \
        --eval_model_dtype bfloat16 \
        --eval_gen_kwargs max_gen_toks=1024,do_sample=False \
        >> "$gsm8k_log" 2>&1 || return 1
    printf 'canonical_scheme=%s\nmodel_key=%s\ngsm8k_log=%s\n' "$scheme" "$MODEL_KEY" "$gsm8k_log" > "$combo_dir/COMPLETE"
    rm -rf "$model_out"
}

job() {
    local gpu="$1"
    local scheme="$2"
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
    if run_one "$scheme"; then
        log "complete: $MODEL_KEY/$scheme on gpu=$gpu"
    else
        log "failed: $MODEL_KEY/$scheme on gpu=$gpu" >&2
        return 1
    fi
}

all_quick_deployed() {
    local scheme
    for scheme in "${SCHEMES[@]}"; do
        [[ -d "$QUICK_ROOT/full/$MODEL_KEY/$scheme" ]] || return 1
    done
}

# A GPU is free when no compute process (ours or anyone else's) is on it.
gpu_is_idle() {
    local uuid apps
    uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$1")" || return 1
    apps="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)" || return 1
    [[ "$apps" != *"$uuid"* ]]
}

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/tmp"
until all_quick_deployed; do
    sleep "$POLL_SECONDS"
done
log "all ${#SCHEMES[@]} quick runs deployed; start scheduling GSM8K"

declare -A gpu_pid=()
pending=()
for scheme in "${SCHEMES[@]}"; do
    [[ -f "$RUN_ROOT/full/$MODEL_KEY/$scheme/COMPLETE" ]] || pending+=("$scheme")
done
status=0
while ((${#pending[@]} > 0)); do
    for ((gpu = 0; gpu < NUM_GPUS && ${#pending[@]} > 0; gpu++)); do
        if [[ -n "${gpu_pid[$gpu]:-}" ]]; then
            kill -0 "${gpu_pid[$gpu]}" 2>/dev/null && continue
            wait "${gpu_pid[$gpu]}" || status=1
            unset "gpu_pid[$gpu]"
        fi
        gpu_is_idle "$gpu" || continue
        scheme="${pending[0]}"
        pending=("${pending[@]:1}")
        job "$gpu" "$scheme" &
        gpu_pid[$gpu]=$!
        log "launch: $MODEL_KEY/$scheme on gpu=$gpu pid=${gpu_pid[$gpu]}"
        # Give the new process time to show up in nvidia-smi before probing again.
        sleep 30
    done
    if ((${#pending[@]} > 0)); then
        sleep "$POLL_SECONDS"
    fi
done
for pid in "${gpu_pid[@]}"; do
    wait "$pid" || status=1
done
log "all done, status=$status"
exit "$status"
