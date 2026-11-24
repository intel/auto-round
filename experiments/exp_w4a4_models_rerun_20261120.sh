#!/usr/bin/env bash
set -euo pipefail

REPO=/data/xinhe/auto-round
PYTHON=/data/xinhe/.venv/bin/python
AUTO_ROUND=/data/xinhe/.venv/bin/auto-round
IN_MEMORY_EVAL="$REPO/experiments/exp_w4a4_in_memory_eval.py"
MODEL_KEY="${1:?usage: $0 <phi-4|qwen2.5-7b|qwen3-32b> [num_gpus]}"
NUM_GPUS="${2:-8}"
DEVICE_MAP=0
GPU_SET=
RUN_ROOT="${RUN_ROOT:-/data/xinhe/exp_w4a4_models_rerun_20261120}"
CACHE=/data/xinhe/.cache/huggingface

case "$MODEL_KEY" in
    phi-4)
        MODEL_ID=microsoft/phi-4
        MODEL_CACHE=models--microsoft--phi-4
        NUM_GPUS="${2:-8}"
        ;;
    qwen2.5-7b)
        MODEL_ID=Qwen/Qwen2.5-7B-Instruct
        MODEL_CACHE=models--Qwen--Qwen2.5-7B-Instruct
        NUM_GPUS="${2:-8}"
        ;;
    qwen3-32b)
        MODEL_ID=Qwen/Qwen3-32B
        MODEL_CACHE=models--Qwen--Qwen3-32B
        NUM_GPUS="${2:-1}"
        DEVICE_MAP=0,1
        GPU_SET="${QWEN3_GPU_SET:-0,1}"
        ;;
    *) printf 'unsupported model key: %s\n' "$MODEL_KEY" >&2; exit 2 ;;
esac

snapshot_is_complete() {
    local snapshot="$1"
    [[ -f "$snapshot/config.json" && -f "$snapshot/model.safetensors.index.json" ]] || return 1
    "$PYTHON" -c 'import json, os, sys; path = sys.argv[1]; index = json.load(open(os.path.join(path, "model.safetensors.index.json"))); sys.exit(not all(os.path.isfile(os.path.join(path, shard)) for shard in set(index["weight_map"].values())))' "$snapshot"
}

MODEL_PATH=
for snapshot in "$CACHE/hub/$MODEL_CACHE"/snapshots/* "$CACHE/$MODEL_CACHE"/snapshots/*; do
    [[ -d "$snapshot" ]] || continue
    if snapshot_is_complete "$snapshot"; then
        MODEL_PATH="$snapshot"
        break
    fi
done
if [[ -z "$MODEL_PATH" ]]; then
    printf 'model snapshot is incomplete: %s\n' "$MODEL_ID" >&2
    exit 2
fi

SCHEMES=(
    EXP-FP4-UE5M3 EXP-INT4-E4M3 EXP-INT4-UE5M3 EXP-UINT4-ZP-UE5M3
    EXP-FP4-BF16 EXP-FP4-FP16 EXP-INT4-UBF16 EXP-INT4-UFP16
    EXP-UINT4-ZP-BF16 EXP-UINT4-ZP-FP16 EXP-INT4-BF16 EXP-INT4-FP16
)

write_metadata() {
    local path="$1" scheme="$2" stage="$3"
    {
        printf 'canonical_scheme=%s\nweight_act=W4A4 (act format = weight format)\n' "$scheme"
        printf 'model_key=%s\nmodel_id=%s\nmodel_path=%s\nstage=%s\n' "$MODEL_KEY" "$MODEL_ID" "$MODEL_PATH" "$stage"
        printf 'host=%s\ngit_head=%s\n' "$(hostname)" "$(git -C "$REPO" rev-parse --short HEAD)"
        printf 'quantization=RTN,group_size=16,seed=42\n\n'
    } > "$path"
}

run_one() {
    local gpu="$1" scheme="$2"
    local combo="$RUN_ROOT/full/$MODEL_KEY/$scheme"
    local likelihood_log="$RUN_ROOT/logs/$MODEL_KEY-$scheme-likelihood.log"
    local gsm8k_log="$RUN_ROOT/logs/$MODEL_KEY-$scheme-gsm8k.log"

    [[ -f "$combo/COMPLETE" ]] && return 0
    mkdir -p "$combo" "$RUN_ROOT/logs" "$RUN_ROOT/work"
    export CUDA_VISIBLE_DEVICES="${GPU_SET:-$gpu}"
    export TMPDIR="$RUN_ROOT/work"
    export TORCHINDUCTOR_CACHE_DIR="$RUN_ROOT/work/inductor-cache"
    export TOKENIZERS_PARALLELISM=false
    export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

    if [[ ! -f "$combo/LIKELIHOOD_COMPLETE" ]]; then
        write_metadata "$likelihood_log" "$scheme" quantize_and_likelihood
        "$PYTHON" "$IN_MEMORY_EVAL" --model "$MODEL_PATH" --scheme "$scheme" \
            --device-map "$DEVICE_MAP" --tasks mmlu,hellaswag,piqa --eval-bs 8 \
            >> "$likelihood_log" 2>&1 || return 1
        touch "$combo/LIKELIHOOD_COMPLETE"
    fi

    if [[ ! -f "$combo/GSM8K_COMPLETE" ]]; then
        write_metadata "$gsm8k_log" "$scheme" quantize_and_gsm8k
        "$PYTHON" "$IN_MEMORY_EVAL" --model "$MODEL_PATH" --scheme "$scheme" \
            --device-map "$DEVICE_MAP" --tasks gsm8k --eval-bs 8 --gsm8k \
            >> "$gsm8k_log" 2>&1 || return 1
        touch "$combo/GSM8K_COMPLETE"
    fi

    printf 'model_key=%s\nscheme=%s\n' "$MODEL_KEY" "$scheme" > "$combo/COMPLETE"
    printf '[%s] complete %s/%s gpu=%s\n' "$(date '+%F %T')" "$MODEL_KEY" "$scheme" "$gpu"
}

worker() {
    local gpu="$1" index status=0
    for ((index = gpu; index < ${#SCHEMES[@]}; index += NUM_GPUS)); do
        if ! run_one "$gpu" "${SCHEMES[$index]}"; then
            printf '[%s] failed %s/%s gpu=%s\n' "$(date '+%F %T')" "$MODEL_KEY" "${SCHEMES[$index]}" "$gpu" >&2
            status=1
        fi
    done
    return "$status"
}

wait_for_gpus() {
    local gpu uuid apps
    while true; do
        apps="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)" || return 1
        local busy=0
        IFS=, read -ra target_gpus <<< "${GPU_SET:-$(seq -s, 0 $((NUM_GPUS - 1)))}"
        for gpu in "${target_gpus[@]}"; do
            uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu" | tr -d '[:space:]')" || return 1
            if [[ "$apps" == *"$uuid"* ]]; then
                busy=1
                break
            fi
        done
        ((busy == 0)) && return 0
        printf '[%s] waiting: requested GPU set is occupied\n' "$(date '+%F %T')"
        sleep 60
    done
}

export HTTP_PROXY=http://127.0.0.1:2080 HTTPS_PROXY=http://127.0.0.1:2080
export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY"
export HF_HOME="$CACHE" HF_HUB_CACHE="$CACHE/hub" HF_HUB_DISABLE_XET=1
export HF_DATASETS_CACHE=/data/xinhe/huggingface/datasets XDG_CACHE_HOME="$CACHE/.cache"
export PYTHONPATH="$REPO"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/work"
wait_for_gpus
printf '[%s] start model=%s GPUs=%s path=%s\n' "$(date '+%F %T')" "$MODEL_KEY" "$NUM_GPUS" "$MODEL_PATH" | tee -a "$RUN_ROOT/orchestrator.log"
pids=()
for ((gpu = 0; gpu < NUM_GPUS; gpu++)); do
    worker "$gpu" >> "$RUN_ROOT/orchestrator.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
printf '[%s] finished model=%s status=%s\n' "$(date '+%F %T')" "$MODEL_KEY" "$status" | tee -a "$RUN_ROOT/orchestrator.log"
exit "$status"