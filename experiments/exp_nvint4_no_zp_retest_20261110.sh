#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/data/exp_int4_ue5m3_retest_20261110}"
REPO=/data/xinhe/auto-round
PYTHON=/data/xinhe/.venv/bin/python
MODEL_ROOT=/data/xinhe/autoround-rtn-matrix/runs/20261102-152238/model-sources
LLAMA=/data/xinhe/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
MODEL_KEYS=(llama31-8b phi-4 qwen2.5-7b qwen3-32b)
MODEL_PATHS=("$LLAMA" "$MODEL_ROOT/phi-4" "$MODEL_ROOT/qwen2.5-7b" "$MODEL_ROOT/qwen3-32b")
SCHEME=EXP-INT4-UE5M3

write_run_metadata() {
    local log_path="$1"
    local model_key="$2"
    local model_path="$3"
    local stage="$4"
    local mode="$5"
    {
        printf 'canonical_scheme=%s\ncli_scheme=%s\n' "$SCHEME" "$SCHEME"
        printf 'legacy_scheme_mapping=EXP-NVINT4->EXP-INT4-UE5M3\n'
        printf 'model_key=%s\nmodel_path=%s\nstage=%s\nmode=%s\nscript=%s\nrun_root=%s\n' \
            "$model_key" "$model_path" "$stage" "$mode" "${BASH_SOURCE[0]}" "$RUN_ROOT"
        printf 'quantization_group_size=16\nseed=42\n\n'
    } > "$log_path"
}

write_completion_marker() {
    local marker="$1"
    local model_key="$2"
    local log_path="$3"
    printf 'canonical_scheme=%s\nmodel_key=%s\nlog_path=%s\n' "$SCHEME" "$model_key" "$log_path" > "$marker"
}

run_matrix_model() {
    local gpu="$1"
    local model_index="$2"
    local model_key="${MODEL_KEYS[$model_index]}"
    local model_path="${MODEL_PATHS[$model_index]}"
    local eval_bs=8
    local combo_dir="$RUN_ROOT/full/$model_key/$SCHEME"
    local model_out="$combo_dir/model"
    local quant_model

    if [[ "$model_index" -eq 0 ]]; then
        eval_bs=32
    fi
    if [[ -f "$combo_dir/COMPLETE" ]]; then
        printf 'skip complete: %s/%s\n' "$model_key" "$SCHEME"
        return 0
    fi
    mkdir -p "$combo_dir" "$RUN_ROOT/logs" "$RUN_ROOT/tmp"
    if [[ ! -f "$model_out/config.json" ]]; then
        local quant_log="$RUN_ROOT/logs/$model_key-$SCHEME-rtn-quant.log"
        write_run_metadata "$quant_log" "$model_key" "$model_path" quantize rtn
        "$PYTHON" -m auto_round "$model_path" \
            --scheme "$SCHEME" \
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
            >> "$quant_log" 2>&1
    fi
    quant_model="$(find "$model_out" -maxdepth 3 -type f -name config.json -printf '%h\n' -quit)"
    if [[ -z "$quant_model" ]] || { [[ ! -f "$quant_model/model.safetensors" ]] && [[ ! -f "$quant_model/model.safetensors.index.json" ]]; }; then
        printf 'incomplete quantized artifact: %s\n' "$model_out" >&2
        return 1
    fi
    if [[ ! -f "$combo_dir/LIKELIHOOD_COMPLETE" ]]; then
        local likelihood_log="$RUN_ROOT/logs/$model_key-$SCHEME-rtn-likelihood.log"
        write_run_metadata "$likelihood_log" "$model_key" "$model_path" likelihood rtn
        auto-round eval "$quant_model" \
            --tasks mmlu,hellaswag,piqa \
            --device_map 0 \
            --seed 42 \
            --eval_bs "$eval_bs" \
            --eval_model_dtype bfloat16 \
            --eval_task_by_task \
            >> "$likelihood_log" 2>&1
        write_completion_marker "$combo_dir/LIKELIHOOD_COMPLETE" "$model_key" "$likelihood_log"
    fi
    if [[ ! -f "$combo_dir/GSM8K_COMPLETE" ]]; then
        local gsm8k_log="$RUN_ROOT/logs/$model_key-$SCHEME-rtn-gsm8k.log"
        write_run_metadata "$gsm8k_log" "$model_key" "$model_path" gsm8k rtn
        auto-round eval "$quant_model" \
            --tasks gsm8k \
            --device_map 0 \
            --seed 42 \
            --eval_bs 8 \
            --num_fewshot 5 \
            --eval_model_dtype bfloat16 \
            --eval_gen_kwargs max_gen_toks=1024,do_sample=False \
            >> "$gsm8k_log" 2>&1
        write_completion_marker "$combo_dir/GSM8K_COMPLETE" "$model_key" "$gsm8k_log"
    fi
    printf 'canonical_scheme=%s\nmodel_key=%s\nlikelihood_log=%s\ngsm8k_log=%s\n' \
        "$SCHEME" "$model_key" "$RUN_ROOT/logs/$model_key-$SCHEME-rtn-likelihood.log" \
        "$RUN_ROOT/logs/$model_key-$SCHEME-rtn-gsm8k.log" > "$combo_dir/COMPLETE"
    printf 'complete: %s/%s-rtn on gpu=%s\n' "$model_key" "$SCHEME" "$gpu"
}

run_llama_hellaswag_mode() {
    local gpu="$1"
    local mode="$2"
    local model_out="$RUN_ROOT/full/llama31-8b/$SCHEME-$mode/model"
    local log="$RUN_ROOT/logs/llama31-8b-$SCHEME-$mode.log"
    local mode_args=()

    if [[ -f "$RUN_ROOT/full/llama31-8b/$SCHEME-$mode/COMPLETE" ]] && [[ "${FORCE:-0}" != 1 ]]; then
        printf 'skip complete: llama31-8b/%s-%s\n' "$SCHEME" "$mode"
        return 0
    fi
    case "$mode" in
        opt-rtn)
            mode_args=(--iters 0)
            ;;
        tuned)
            mode_args=(--iters 200 --enable_minmax_tuning)
            ;;
        *)
            printf 'unsupported HellaSwag mode: %s\n' "$mode" >&2
            return 2
            ;;
    esac
    mkdir -p "$(dirname "$model_out")" "$RUN_ROOT/logs" "$RUN_ROOT/tmp"
    write_run_metadata "$log" llama31-8b "$LLAMA" hellaswag "$mode"
    "$PYTHON" -m auto_round "$LLAMA" \
        --scheme "$SCHEME" \
        --format fake \
        --tasks hellaswag \
        --eval_bs 32 \
        --eval_model_dtype bfloat16 \
        --nsamples 128 \
        --seqlen 2048 \
        --batch_size 8 \
        --dataset NeelNanda/pile-10k \
        --seed 42 \
        --device_map 0 \
        --enable_deterministic_algorithms \
        --output_dir "$model_out" \
        "${mode_args[@]}" \
        >> "$log" 2>&1
    printf 'canonical_scheme=%s\nmode=%s\nlog_path=%s\n' "$SCHEME" "$mode" "$log" > "$RUN_ROOT/full/llama31-8b/$SCHEME-$mode/COMPLETE"
    printf 'complete: llama31-8b/%s-%s on gpu=%s\n' "$SCHEME" "$mode" "$gpu"
}

worker() {
    local gpu="$1"
    local job_index="$2"

    if [[ ! "$gpu" =~ ^[0-6]$ ]]; then
        printf 'gpu id must be in [0, 6], got %s\n' "$gpu" >&2
        return 2
    fi
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
    export AR_WORK_SPACE="$RUN_ROOT/workspaces/job-$job_index"
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export OPENBLAS_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false
    export TMPDIR="$RUN_ROOT/tmp"
    mkdir -p "$AR_WORK_SPACE"

    if [[ "$job_index" =~ ^[0-3]$ ]]; then
        run_matrix_model "$gpu" "$job_index"
    else
        case "$job_index" in
            4)
                export AR_DISABLE_BATCHED_SEARCH=1
                run_llama_hellaswag_mode "$gpu" opt-rtn
                ;;
            5) run_llama_hellaswag_mode "$gpu" tuned ;;
            *) printf 'job index must be in [0, 5], got %s\n' "$job_index" >&2; return 2 ;;
        esac
    fi
}

if [[ "${1:-}" != "--job" ]] || [[ "$#" -ne 3 ]]; then
    printf 'usage: %s --job GPU_ID JOB_INDEX\n' "$0" >&2
    exit 2
fi

mkdir -p "$RUN_ROOT" "$RUN_ROOT/logs" "$RUN_ROOT/tmp"
for model_path in "${MODEL_PATHS[@]}"; do
    if [[ ! -f "$model_path/config.json" ]]; then
        printf 'missing model config: %s\n' "$model_path" >&2
        exit 2
    fi
done
source /data/xinhe/.venv/bin/activate
worker "$2" "$3"