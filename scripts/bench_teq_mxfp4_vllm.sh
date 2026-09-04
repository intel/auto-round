#!/usr/bin/env bash
set -Eeuo pipefail

# Benchmark BF16, MXFP4 baselines, and TEQ + {RTN, AutoRound, AutoRoundV2}.
# Defaults use fake format + HF eval so MXFP4 is benchmarked as W4A4.
# Set FORMAT=llm_compressor EVAL_BACKEND=vllm to benchmark exported vLLM inference.
#
# Examples:
#   scripts/bench_teq_mxfp4_vllm.sh dry-run
#   GPUS=2,4 LIMIT=0.001 scripts/bench_teq_mxfp4_vllm.sh run
#   RESULTS_DIR=/path/to/run scripts/bench_teq_mxfp4_vllm.sh summarize

AUTOROUND_ROOT=${AUTOROUND_ROOT:-/data6/zww/round_teq}
RESULTS_DIR=${RESULTS_DIR:-"$AUTOROUND_ROOT/runs/teq_mxfp4_w4a4_$(date +%Y%m%d_%H%M%S)"}

MODELS=${MODELS:-"qwen3-4b=/models/Qwen3-4B"}
SCHEME=${SCHEME:-MXFP4}
VARIANTS=${VARIANTS:-"bf16 rtn rtn-opt auto-round auto-round-v2 awq-rtn awq-ar awq-arv2 teq-rtn teq-ar teq-arv2"}
GPUS=${GPUS:-"2,4"}

CONDA_ENV=${CONDA_ENV:-}
TASKS=${TASKS:-"mmlu,gsm8k,piqa,hellaswag,winogrande"}
DATASET=${DATASET:-NeelNanda/pile-10k}
NSAMPLES=${NSAMPLES:-128}
SEQLEN=${SEQLEN:-2048}
ITERS=${ITERS:-200}
BATCH_SIZE=${BATCH_SIZE:-8}
EVAL_BS=${EVAL_BS:-16}
MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
BF16_MODEL_DTYPE=${BF16_MODEL_DTYPE:-bfloat16}
FORMAT=${FORMAT:-fake}
EVAL_BACKEND=${EVAL_BACKEND:-hf}
LIMIT=${LIMIT:-}
VLLM_ARGS=${VLLM_ARGS:-"gpu_memory_utilization=0.85,max_model_len=4096,enforce_eager=true"}

TEQ_ITERS=${TEQ_ITERS:-20}
TEQ_LR=${TEQ_LR:-1e-3}
TEQ_MIN_SCALE=${TEQ_MIN_SCALE:-1e-5}
TEQ_MAX_SCALE=${TEQ_MAX_SCALE:-10.0}
TEQ_SQRT_W_INIT=${TEQ_SQRT_W_INIT:-0}
TEQ_AWQ_INIT=${TEQ_AWQ_INIT:-0}
TEQ_AWQ_INIT_N_GRID=${TEQ_AWQ_INIT_N_GRID:-20}
TEQ_NSAMPLES=${TEQ_NSAMPLES:-}
TEQ_BATCH_SIZE=${TEQ_BATCH_SIZE:-1}
TEQ_SAMPLE_SEQLEN=${TEQ_SAMPLE_SEQLEN:-512}

AWQ_N_GRID=${AWQ_N_GRID:-20}
AWQ_SMOOTH_SEQLEN=${AWQ_SMOOTH_SEQLEN:-512}
AWQ_SMOOTH_BATCH_SIZE=${AWQ_SMOOTH_BATCH_SIZE:-}

LOW_GPU_MEM_USAGE=${LOW_GPU_MEM_USAGE:-0}
LOW_CPU_MEM_USAGE=${LOW_CPU_MEM_USAGE:-1}
ENABLE_TORCH_COMPILE=${ENABLE_TORCH_COMPILE:-1}
SAMPLE_INTERVAL=${SAMPLE_INTERVAL:-1}
KEEP_MODELS=${KEEP_MODELS:-0}
KEEP_FAILED_MODELS=${KEEP_FAILED_MODELS:-1}
DRY_RUN=${DRY_RUN:-0}

usage() {
  cat <<USAGE
Usage: $0 {run|dry-run|summarize|debug}

Default matrix:
  models:   $MODELS
  scheme:   $SCHEME
  variants: $VARIANTS
  gpus:     $GPUS
  format:   $FORMAT
  eval:     $EVAL_BACKEND

Important overrides:
  RESULTS_DIR=/path                    Output logs and summaries.
  GPUS=2,4                             Physical GPUs for workers. GPU 7 is always rejected.
  MODELS='name=/path name2=/path2'     Model list.
  VARIANTS='bf16 rtn rtn-opt auto-round auto-round-v2 awq-rtn awq-ar awq-arv2 teq-rtn teq-ar teq-arv2'
                                         Variant list.
                                         rtn is plain RTN; rtn-opt enables optimized RTN.
                                         awq-rtn applies AWQ clip and plain RTN.
                                         bf16 evaluates the original BF16 model without quantization.
  NSAMPLES=128 SEQLEN=2048 BATCH_SIZE=8
  TEQ_ITERS=20 TEQ_LR=1e-3 TEQ_NSAMPLES=32 TEQ_BATCH_SIZE=1 TEQ_SAMPLE_SEQLEN=512
  TEQ_AWQ_INIT=1 TEQ_AWQ_INIT_N_GRID=20
  AWQ_N_GRID=20 AWQ_SMOOTH_SEQLEN=512 AWQ_SMOOTH_BATCH_SIZE=1
  ITERS=200                            AutoRound/AutoRoundV2 iterations.
  FORMAT=fake                          Default fake-quant format for W4A4 HF eval.
  EVAL_BACKEND=hf                      Default eval backend. Use vllm only for exported inference.
  LIMIT=0.001                          Optional lm-eval limit for smoke runs.
  TASKS=mmlu,gsm8k,piqa,hellaswag,winogrande
  VLLM_ARGS='gpu_memory_utilization=0.85,max_model_len=4096,enforce_eager=true'
                                         Used only when EVAL_BACKEND=vllm.
  LOW_GPU_MEM_USAGE=0 LOW_CPU_MEM_USAGE=0
  KEEP_MODELS=1                        Keep output model dirs for non-BF16 recipes.
  DRY_RUN=1                            Print commands without executing.

Examples:
  $0 dry-run
  GPUS=2,4 LIMIT=0.001 EVAL_BS=8 $0 run
  FORMAT=llm_compressor EVAL_BACKEND=vllm GPUS=2,4 $0 run
  RESULTS_DIR=/path/to/run $0 summarize
USAGE
}

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*" >&2
}

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

activate_env() {
  if [[ -z "$CONDA_ENV" || -n "${VIRTUAL_ENV:-}" || "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  else
    log "conda was not found; continuing with the current Python environment."
  fi
}

require_paths() {
  [[ -d "$AUTOROUND_ROOT" ]] || { echo "Missing AUTOROUND_ROOT=$AUTOROUND_ROOT" >&2; exit 1; }
  for item in $MODELS; do
    local model_path=${item#*=}
    [[ -d "$model_path" ]] || { echo "Missing model path: $model_path" >&2; exit 1; }
  done
}

parse_gpus() {
  local raw=${1//,/ }
  for gpu in $raw; do
    [[ -z "$gpu" ]] && continue
    if [[ "$gpu" == "7" ]]; then
      log "Skipping GPU 7 because it is marked broken."
      continue
    fi
    printf '%s\n' "$gpu"
  done
}

safe_name() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+|-+$//g'
}

is_bf16_variant() {
  case "${1,,}" in
    bf16|bfloat16|fp16-baseline|full-precision|full_precision)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

write_jobs() {
  local jobs_file=$1
  : >"$jobs_file"
  local job_id=0
  for item in $MODELS; do
    local model_key=${item%%=*}
    local model_path=${item#*=}
    for variant in $VARIANTS; do
      local job_scheme=$SCHEME
      if is_bf16_variant "$variant"; then
        job_scheme=BF16
      fi
      printf '%s\t%s\t%s\t%s\t%s\n' "$job_id" "$model_key" "$model_path" "$job_scheme" "$variant" >>"$jobs_file"
      job_id=$((job_id + 1))
    done
  done
}

monitor_gpu() {
  local gpu=$1
  local output=$2
  local stop_file=$3
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return
  fi
  while [[ ! -f "$stop_file" ]]; do
    nvidia-smi --id="$gpu" \
      --query-gpu=timestamp,index,memory.used,utilization.gpu \
      --format=csv,noheader,nounits >>"$output" 2>/dev/null || true
    sleep "$SAMPLE_INTERVAL"
  done
}

peak_vram_mib() {
  local csv=$1
  if [[ ! -s "$csv" ]]; then
    echo 0
    return
  fi
  awk -F',' '{gsub(/ /, "", $3); if ($3 + 0 > max) max = $3 + 0} END {print max + 0}' "$csv"
}

max_rss_kb() {
  local time_file=$1
  if [[ ! -s "$time_file" ]]; then
    echo 0
    return
  fi
  awk -F': ' '/Maximum resident set size/ {print $2 + 0}' "$time_file"
}

cleanup_model_dir() {
  local output_dir=$1
  local status=$2
  if [[ "$KEEP_MODELS" == "1" ]]; then
    log "Keeping quantized model dir because KEEP_MODELS=1: $output_dir"
    return
  fi
  if [[ "$status" != "ok" && "$KEEP_FAILED_MODELS" == "1" ]]; then
    log "Keeping failed-run model dir because KEEP_FAILED_MODELS=1: $output_dir"
    return
  fi
  case "$output_dir" in
    "$RESULTS_DIR"/models/*)
      log "Deleting quantized model dir after run; logs are kept under $RESULTS_DIR/logs: $output_dir"
      rm -rf "$output_dir"
      ;;
    *)
      log "Refusing to delete unexpected output_dir=$output_dir"
      ;;
  esac
}

build_command_array() {
  local model_path=$1
  local variant=$2
  local output_dir=$3
  local scheme=$4

  if is_bf16_variant "$variant"; then
    RUN_CMD=(
      python -m auto_round eval "$model_path"
      --device_map 0
      --tasks "$TASKS"
      --eval_backend "$EVAL_BACKEND"
      --eval_bs "$EVAL_BS"
      --eval_model_dtype "$BF16_MODEL_DTYPE"
    )

    if [[ -n "$LIMIT" ]]; then
      RUN_CMD+=(--limit "$LIMIT")
    fi
    if [[ "$EVAL_BACKEND" == "vllm" && -n "$VLLM_ARGS" ]]; then
      RUN_CMD+=(--vllm_args "$VLLM_ARGS")
    fi
    if [[ "$model_path" == *Llama* || "$model_path" == *llama* ]]; then
      RUN_CMD+=(--add_bos_token)
    fi
    return 0
  fi

  RUN_CMD=(
    python -m auto_round quantize "$model_path"
    --model_dtype "$MODEL_DTYPE"
    --scheme "$scheme"
    --format "$FORMAT"
    --output_dir "$output_dir"
    --device_map 0
    --dataset "$DATASET"
    --nsamples "$NSAMPLES"
    --seqlen "$SEQLEN"
    --batch_size "$BATCH_SIZE"
    --tasks "$TASKS"
    --eval_backend "$EVAL_BACKEND"
    --eval_bs "$EVAL_BS"
  )

  if [[ "$LOW_GPU_MEM_USAGE" == "1" ]]; then
    RUN_CMD+=(--low_gpu_mem_usage)
  fi
  if [[ "$LOW_CPU_MEM_USAGE" == "0" ]]; then
    RUN_CMD+=(--disable_low_cpu_mem_usage)
  fi
  if [[ "$ENABLE_TORCH_COMPILE" == "1" ]]; then
    RUN_CMD+=(--enable_torch_compile)
  elif [[ "$ENABLE_TORCH_COMPILE" == "0" ]]; then
    RUN_CMD+=(--disable_torch_compile)
  fi
  if [[ -n "$LIMIT" ]]; then
    RUN_CMD+=(--limit "$LIMIT")
  fi
  if [[ "$EVAL_BACKEND" == "vllm" && -n "$VLLM_ARGS" ]]; then
    RUN_CMD+=(--vllm_args "$VLLM_ARGS")
  fi
  if [[ "$model_path" == *Llama* || "$model_path" == *llama* ]]; then
    RUN_CMD+=(--add_bos_token)
  fi

  case "$variant" in
    rtn|rtn-plain|rtn-no-opt|rtn-default)
      RUN_CMD+=(--algorithm rtn --disable_opt_rtn)
      ;;
    rtn-opt|rtn-with-opt|opt-rtn|opt_rtn)
      RUN_CMD+=(--algorithm rtn --enable_opt_rtn)
      ;;
    auto-round|auto_round|ar-default)
      RUN_CMD+=(--algorithm auto_round --iters "$ITERS" --no-enable_alg_ext)
      ;;
    auto-round-v2|auto_round_v2|ar-v2|arv2)
      RUN_CMD+=(--algorithm auto_round --iters "$ITERS" --enable_alg_ext)
      ;;
    awq-rtn)
      RUN_CMD+=(
        --awq-n-grid "$AWQ_N_GRID"
        --awq-smooth-seqlen "$AWQ_SMOOTH_SEQLEN"
        --awq-apply-clip
        --algorithm awq,rtn
        --disable_opt_rtn
      )
      ;;
    awq-ar)
      RUN_CMD+=(
        --awq-n-grid "$AWQ_N_GRID"
        --awq-smooth-seqlen "$AWQ_SMOOTH_SEQLEN"
        --algorithm awq,auto_round
        --iters "$ITERS"
        --no-enable_alg_ext
      )
      ;;
    awq-arv2)
      RUN_CMD+=(
        --awq-n-grid "$AWQ_N_GRID"
        --awq-smooth-seqlen "$AWQ_SMOOTH_SEQLEN"
        --algorithm awq,auto_round
        --iters "$ITERS"
        --enable_alg_ext
      )
      ;;
    teq-rtn)
      RUN_CMD+=(
        --teq-iters "$TEQ_ITERS"
        --teq-lr "$TEQ_LR"
        --teq-min-scale "$TEQ_MIN_SCALE"
        --teq-max-scale "$TEQ_MAX_SCALE"
        --algorithm teq,rtn
        --enable_opt_rtn
      )
      ;;
    teq-ar)
      RUN_CMD+=(
        --teq-iters "$TEQ_ITERS"
        --teq-lr "$TEQ_LR"
        --teq-min-scale "$TEQ_MIN_SCALE"
        --teq-max-scale "$TEQ_MAX_SCALE"
        --algorithm teq,auto_round
        --iters "$ITERS"
        --no-enable_alg_ext
      )
      ;;
    teq-arv2)
      RUN_CMD+=(
        --teq-iters "$TEQ_ITERS"
        --teq-lr "$TEQ_LR"
        --teq-min-scale "$TEQ_MIN_SCALE"
        --teq-max-scale "$TEQ_MAX_SCALE"
        --algorithm teq,auto_round
        --iters "$ITERS"
        --enable_alg_ext
      )
      ;;
    *)
      echo "Unknown variant=$variant. Expected one of: bf16 rtn rtn-opt auto-round auto-round-v2 awq-rtn awq-ar awq-arv2 teq-rtn teq-ar teq-arv2" >&2
      return 2
      ;;
  esac

  if [[ "$variant" == awq-* && -n "$AWQ_SMOOTH_BATCH_SIZE" ]]; then
    RUN_CMD+=(--awq-smooth-batch-size "$AWQ_SMOOTH_BATCH_SIZE")
  fi
  if [[ "$variant" == teq-* && "$TEQ_SQRT_W_INIT" == "1" ]]; then
    RUN_CMD+=(--teq-sqrt-w-init)
  fi
  if [[ "$variant" == teq-* && "$TEQ_AWQ_INIT" == "1" ]]; then
    RUN_CMD+=(--teq-awq-init --teq-awq-init-n-grid "$TEQ_AWQ_INIT_N_GRID")
  fi
  if [[ "$variant" == teq-* && -n "$TEQ_NSAMPLES" ]]; then
    RUN_CMD+=(--teq-nsamples "$TEQ_NSAMPLES")
  fi
  if [[ "$variant" == teq-* && -n "$TEQ_SAMPLE_SEQLEN" ]]; then
    RUN_CMD+=(--teq-sample-seqlen "$TEQ_SAMPLE_SEQLEN")
  fi
  if [[ "$variant" == teq-* && -n "$TEQ_BATCH_SIZE" ]]; then
    RUN_CMD+=(--teq-batch-size "$TEQ_BATCH_SIZE")
  fi
}

run_one() {
  local gpu=$1
  local model_key=$2
  local model_path=$3
  local scheme=$4
  local variant=$5

  local run_id
  run_id="$(safe_name "${model_key}_${scheme}_${variant}")"
  local output_dir="$RESULTS_DIR/models/$run_id"
  local log_file="$RESULTS_DIR/logs/${run_id}.log"
  local time_file="$RESULTS_DIR/logs/${run_id}.time"
  local gpu_csv="$RESULTS_DIR/logs/${run_id}.gpu.csv"
  local stop_file="$RESULTS_DIR/logs/${run_id}.gpu.stop"
  local metrics_file="$RESULTS_DIR/metrics.tsv"

  mkdir -p "$RESULTS_DIR/models" "$RESULTS_DIR/logs"
  rm -f "$stop_file"

  if ! build_command_array "$model_path" "$variant" "$output_dir" "$scheme"; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$model_key" "$scheme" "$variant" "failed" 0 0 0 "$log_file" "$output_dir" "$gpu" >>"$metrics_file"
    return 2
  fi

  log "[$run_id] GPU $gpu command: $(quote_cmd "${RUN_CMD[@]}")"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$model_key" "$scheme" "$variant" "dry-run" 0 0 0 "$log_file" "$output_dir" "$gpu" >>"$metrics_file"
    return 0
  fi

  monitor_gpu "$gpu" "$gpu_csv" "$stop_file" &
  local monitor_pid=$!
  local start_s
  start_s=$(date +%s)
  local status=ok

  if ! (
    cd "$AUTOROUND_ROOT"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export TOKENIZERS_PARALLELISM=false
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export HF_HUB_ENABLE_HF_TRANSFER=0
    /usr/bin/time -v -o "$time_file" "${RUN_CMD[@]}"
  ) >"$log_file" 2>&1; then
    status=failed
  fi

  local end_s
  end_s=$(date +%s)
  touch "$stop_file"
  wait "$monitor_pid" || true

  local wall_seconds=$((end_s - start_s))
  local rss_kb
  local peak_vram
  rss_kb=$(max_rss_kb "$time_file")
  peak_vram=$(peak_vram_mib "$gpu_csv")

  if ! is_bf16_variant "$variant"; then
    cleanup_model_dir "$output_dir" "$status"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$model_key" "$scheme" "$variant" "$status" "$wall_seconds" "$rss_kb" "$peak_vram" \
    "$log_file" "$output_dir" "$gpu" >>"$metrics_file"

  [[ "$status" == "ok" ]]
}

worker() {
  local worker_index=$1
  local worker_count=$2
  local gpu=$3
  local jobs_file=$4
  local failed=0

  while IFS=$'\t' read -r job_id model_key model_path scheme variant; do
    if (( job_id % worker_count != worker_index )); then
      continue
    fi
    if ! run_one "$gpu" "$model_key" "$model_path" "$scheme" "$variant"; then
      failed=1
    fi
  done <"$jobs_file"

  return "$failed"
}

summarize_results() {
  python - "$RESULTS_DIR" "$TASKS" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

root = Path(sys.argv[1])
tasks = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
metrics_path = root / "metrics.tsv"

preferred = {
    "hellaswag": ["acc_norm", "acc"],
    "piqa": ["acc_norm", "acc"],
    "gsm8k": ["exact_match", "acc"],
    "lambada_openai": ["acc", "perplexity"],
}


def to_float(value: str):
    try:
        return float(value.strip())
    except ValueError:
        return None


def parse_log(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    found: dict[str, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            continue
        task = cells[0].lower()
        if not task or set(task) <= {"-", ":"} or task in {"tasks", "groups"}:
            continue
        for idx, cell in enumerate(cells):
            metric = cell.split(",", 1)[0].strip().lower()
            if metric not in {"acc", "acc_norm", "exact_match", "perplexity"}:
                continue
            value = None
            for later in cells[idx + 1 :]:
                value = to_float(later)
                if value is not None:
                    break
            if value is not None:
                found.setdefault(task, {})[metric] = value

    results = {}
    for task in tasks:
        task_l = task.lower()
        candidates = []
        if task_l in found:
            candidates.append(found[task_l])
        if task_l == "mmlu":
            mmlu_rows = [value for key, value in found.items() if key.startswith("mmlu_")]
            if mmlu_rows:
                aggregate = {}
                for metric in {"acc", "acc_norm", "exact_match"}:
                    vals = [row[metric] for row in mmlu_rows if metric in row]
                    if vals:
                        aggregate[metric] = sum(vals) / len(vals)
                candidates.append(aggregate)
        choice = None
        for candidate in candidates:
            for metric in preferred.get(task_l, ["acc", "acc_norm", "exact_match", "perplexity"]):
                if metric in candidate:
                    choice = candidate[metric]
                    break
            if choice is not None:
                break
        if choice is not None:
            results[task] = choice
    return results


rows = []
if metrics_path.exists():
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("model\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 10:
            continue
        model, scheme, variant, status, seconds, rss_kb, vram_mib, log_file, output_dir, gpu = parts[:10]
        scores = parse_log(Path(log_file))
        present = [scores[task] for task in tasks if task in scores]
        avg = sum(present) / len(present) if present else None
        rows.append(
            {
                "model": model,
                "scheme": scheme,
                "variant": variant,
                "status": status,
                "seconds": int(seconds or 0),
                "rss_gib": int(rss_kb or 0) / 1024 / 1024,
                "vram_mib": int(vram_mib or 0),
                "gpu": gpu,
                "scores": scores,
                "avg": avg,
                "log": log_file,
            }
        )

header = ["Model", "Scheme", "Variant", "Status", *tasks, "Avg.", "Time(s)", "Peak VRAM(MiB)", "Max RSS(GiB)", "GPU"]
align = [":--", ":--:", ":--:", ":--:", *[":--:" for _ in tasks], ":--:", "--:", "--:", "--:", ":--:"]
lines = ["|" + "|".join(header) + "|", "|" + "|".join(align) + "|"]
for row in sorted(rows, key=lambda item: (item["model"], item["scheme"], item["variant"])):
    score_cells = [f"{row['scores'][task]:.4f}" if task in row["scores"] else "-" for task in tasks]
    avg = f"{row['avg']:.4f}" if row["avg"] is not None else "-"
    lines.append(
        "|"
        + "|".join(
            [
                row["model"],
                row["scheme"],
                row["variant"],
                row["status"],
                *score_cells,
                avg,
                str(row["seconds"]),
                str(row["vram_mib"]),
                f"{row['rss_gib']:.2f}",
                row["gpu"],
            ]
        )
        + "|"
    )

summary = "\n".join(lines) + "\n"
(root / "summary.md").write_text(summary, encoding="utf-8")
print(summary)
PY
}

debug_check() {
  require_paths
  activate_env
  log "AutoRound root: $AUTOROUND_ROOT"
  log "Results dir: $RESULTS_DIR"
  log "Conda env: ${CONDA_DEFAULT_ENV:-none}"
  log "Python: $(command -v python)"
  python - <<'PY'
import auto_round
from auto_round import TEQConfig, SignRoundConfig
from auto_round.algorithms.registry import resolve_algorithm_names

print("auto_round:", getattr(auto_round, "__version__", "unknown"))
print("alg teq,rtn:", resolve_algorithm_names("teq,rtn"))
print("alg teq,auto_round:", resolve_algorithm_names("teq,auto_round"))
print("TEQConfig:", TEQConfig(iters=1))
print("ARV2 flag:", SignRoundConfig(enable_alg_ext=True).enable_alg_ext)
PY
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.free,memory.total,utilization.gpu --format=csv
  fi
}

run_all() {
  require_paths
  activate_env
  mkdir -p "$RESULTS_DIR/logs" "$RESULTS_DIR/models"
  local metrics_file="$RESULTS_DIR/metrics.tsv"
  printf 'model\tscheme\tvariant\tstatus\twall_seconds\tmax_rss_kb\tpeak_vram_mib\tlog_file\toutput_dir\tgpu\n' >"$metrics_file"

  local jobs_file="$RESULTS_DIR/jobs.tsv"
  write_jobs "$jobs_file"

  mapfile -t gpu_list < <(parse_gpus "$GPUS")
  if [[ ${#gpu_list[@]} -eq 0 ]]; then
    echo "No usable GPUs configured. Set GPUS without card 7, for example GPUS=2,4." >&2
    exit 1
  fi

  local pids=()
  for idx in "${!gpu_list[@]}"; do
    local gpu=${gpu_list[$idx]}
    log "Launching worker $idx on physical GPU $gpu"
    worker "$idx" "${#gpu_list[@]}" "$gpu" "$jobs_file" >"$RESULTS_DIR/logs/worker_${idx}_gpu_${gpu}.log" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done

  summarize_results
  log "Summary: $RESULTS_DIR/summary.md"
  if [[ "$failed" != "0" ]]; then
    echo "One or more runs failed; inspect $RESULTS_DIR/logs." >&2
    exit 1
  fi
}

cmd=${1:-}
case "$cmd" in
  run)
    run_all
    ;;
  dry-run)
    DRY_RUN=1
    run_all
    ;;
  summarize)
    summarize_results
    ;;
  debug|check)
    debug_check
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
