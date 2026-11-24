#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/data/exp_scale_int4_no_zp_20261110}"
REPO=/data/xinhe/auto-round
PYTHON=/data/xinhe/.venv/bin/python
MODEL_ROOT=/data/xinhe/autoround-rtn-matrix/runs/20261102-152238/model-sources
LLAMA=/data/xinhe/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
MODEL_KEYS=(llama31-8b phi-4 qwen2.5-7b qwen3-32b)
MODEL_PATHS=("$LLAMA" "$MODEL_ROOT/phi-4" "$MODEL_ROOT/qwen2.5-7b" "$MODEL_ROOT/qwen3-32b")
SCHEMES=(EXP-INT4-UBF16 EXP-INT4-UFP16)

write_run_metadata() {
    local log_path="$1"
    local model_key="$2"
    local model_path="$3"
    local scheme="$4"
    local stage="$5"
    {
        printf 'canonical_scheme=%s\ncli_scheme=%s\n' "$scheme" "$scheme"
        printf 'legacy_scheme_mapping=EXP-INT4-BF16S/FP16S->EXP-INT4-UBF16/UFP16\n'
        printf 'model_key=%s\nmodel_path=%s\nstage=%s\nscript=%s\nrun_root=%s\n' \
            "$model_key" "$model_path" "$stage" "${BASH_SOURCE[0]}" "$RUN_ROOT"
        printf 'quantization=RTN,group_size=16,seed=42\n\n'
    } > "$log_path"
}

write_completion_marker() {
    local marker="$1"
    local scheme="$2"
    local model_key="$3"
    local log_path="$4"
    printf 'canonical_scheme=%s\nmodel_key=%s\nlog_path=%s\n' "$scheme" "$model_key" "$log_path" > "$marker"
}

run_one() {
    local gpu="$1"
    local job_index="$2"
    local scheme_index=$((job_index / 4))
    local model_index=$((job_index % 4))
    local model_key="${MODEL_KEYS[$model_index]}"
    local model_path="${MODEL_PATHS[$model_index]}"
    local scheme="${SCHEMES[$scheme_index]}"
    local eval_bs=8
    local combo_dir="$RUN_ROOT/full/$model_key/$scheme"
    local model_out="$combo_dir/model"
    local quant_model

    if [[ "$model_index" -eq 0 ]]; then
        eval_bs=32
    fi
    if [[ -f "$combo_dir/COMPLETE" ]]; then
        printf 'skip complete: %s/%s\n' "$model_key" "$scheme"
        return 0
    fi
    mkdir -p "$combo_dir" "$RUN_ROOT/logs" "$RUN_ROOT/tmp"
    if [[ ! -f "$model_out/config.json" ]]; then
        local quant_log="$RUN_ROOT/logs/$model_key-$scheme-quant.log"
        write_run_metadata "$quant_log" "$model_key" "$model_path" "$scheme" quantize
        "$PYTHON" -m auto_round "$model_path" \
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
    fi
    quant_model="$(find "$model_out" -maxdepth 3 -type f -name config.json -printf '%h\n' -quit)"
    if [[ -z "$quant_model" ]] || { [[ ! -f "$quant_model/model.safetensors" ]] && [[ ! -f "$quant_model/model.safetensors.index.json" ]]; }; then
        printf 'incomplete quantized artifact: %s\n' "$model_out" >&2
        return 1
    fi
    if [[ ! -f "$combo_dir/LIKELIHOOD_COMPLETE" ]]; then
        local likelihood_log="$RUN_ROOT/logs/$model_key-$scheme-likelihood.log"
        write_run_metadata "$likelihood_log" "$model_key" "$model_path" "$scheme" likelihood
        auto-round eval "$quant_model" \
            --tasks mmlu,hellaswag,piqa \
            --device_map 0 \
            --seed 42 \
            --eval_bs "$eval_bs" \
            --eval_model_dtype bfloat16 \
            --eval_task_by_task \
            >> "$likelihood_log" 2>&1 || return 1
        write_completion_marker "$combo_dir/LIKELIHOOD_COMPLETE" "$scheme" "$model_key" "$likelihood_log"
    fi
    if [[ ! -f "$combo_dir/GSM8K_COMPLETE" ]]; then
        local gsm8k_log="$RUN_ROOT/logs/$model_key-$scheme-gsm8k.log"
        write_run_metadata "$gsm8k_log" "$model_key" "$model_path" "$scheme" gsm8k
        auto-round eval "$quant_model" \
            --tasks gsm8k \
            --device_map 0 \
            --seed 42 \
            --eval_bs 8 \
            --num_fewshot 5 \
            --eval_model_dtype bfloat16 \
            --eval_gen_kwargs max_gen_toks=1024,do_sample=False \
            >> "$gsm8k_log" 2>&1 || return 1
        write_completion_marker "$combo_dir/GSM8K_COMPLETE" "$scheme" "$model_key" "$gsm8k_log"
    fi
    printf 'canonical_scheme=%s\nmodel_key=%s\nlikelihood_log=%s\ngsm8k_log=%s\n' \
        "$scheme" "$model_key" "$RUN_ROOT/logs/$model_key-$scheme-likelihood.log" \
        "$RUN_ROOT/logs/$model_key-$scheme-gsm8k.log" > "$combo_dir/COMPLETE"
    printf 'complete: %s/%s on gpu=%s\n' "$model_key" "$scheme" "$gpu"
}

worker() {
    local gpu="$1"
    local job_index="$2"

    if [[ ! "$job_index" =~ ^[0-7]$ ]]; then
        printf 'job index must be in [0, 7], got %s\n' "$job_index" >&2
        return 2
    fi
    cd "$RUN_ROOT"
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
    export OPENBLAS_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false
    export TMPDIR="$RUN_ROOT/tmp"
    run_one "$gpu" "$job_index"
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