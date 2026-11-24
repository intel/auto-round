#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/xinhe/exp_w4a4_llama_quick_20261119
LOG="$ROOT/gsm8k/h20-120-delayed-deploy.log"
REPO=/data/xinhe/auto-round
GSM_SCRIPT="$REPO/experiments/exp_w4a4_llama_gsm8k_20261119.sh"
TARGET_PREFIX='/data/drama_pairs/pilot/env_ds41/bin/vllm serve /data/models/deepseek-ai/DeepSeek-V4.1-Flash-1017'

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*"; }
log "delayed GSM8K deployment started on $(hostname)"

mapfile -t target_pids < <(ps -eo pid=,args= | awk -v prefix="$TARGET_PREFIX" '
    index($0, prefix) && index($0, "--port 8041") { print $1 }
')
if ((${#target_pids[@]} == 0)); then
    log "target vLLM service not found; refusing to terminate unrelated processes"
else
    for pid in "${target_pids[@]}"; do
        if [[ -r "/proc/$pid/cmdline" ]] && tr '\0' ' ' <"/proc/$pid/cmdline" | grep -Fq "$TARGET_PREFIX" && tr '\0' ' ' <"/proc/$pid/cmdline" | grep -Fq -- '--port 8041'; then
            log "sending SIGTERM to authorized vLLM service pid=$pid"
            kill -TERM "$pid"
        fi
    done
fi

released=0
for ((attempt = 1; attempt <= 60; attempt++)); do
    if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')" ]]; then
        released=1
        break
    fi
    sleep 10
done
if ((released == 0)); then
    log "GPU processes remain after 10 minutes; GSM8K deployment aborted without force-killing"
    exit 1
fi

log "all h20-120 GPUs are free; starting 12-scheme GSM8K dispatcher"
export NUM_GPUS=8
export POLL_SECONDS=60
export QUICK_ROOT="$ROOT"
export RUN_ROOT="$ROOT/gsm8k"
bash "$GSM_SCRIPT"
log "GSM8K dispatcher exited"
