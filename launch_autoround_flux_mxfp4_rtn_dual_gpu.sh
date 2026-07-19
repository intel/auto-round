#!/usr/bin/env bash

set -Eeuo pipefail

ROOT="${ROOT:-/home/user2/data/xixi}"
RUNNER="$ROOT/run_autoround_flux_mxfp4_rtn_calibration.sh"
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-$ROOT/launch-logs}"

if [[ ! -x "$RUNNER" ]]; then
  echo "Runner is missing or not executable: $RUNNER" >&2
  exit 1
fi

mkdir -p "$LAUNCH_LOG_DIR"

nohup "$RUNNER" 0 32 20 >"$LAUNCH_LOG_DIR/mxfp4-rtn-n32-s20.launch.log" 2>&1 &
PID0=$!

nohup "$RUNNER" 1 128 50 >"$LAUNCH_LOG_DIR/mxfp4-rtn-n128-s50.launch.log" 2>&1 &
PID1=$!

printf 'GPU 0: PID=%s, launcher log=%s\n' "$PID0" "$LAUNCH_LOG_DIR/mxfp4-rtn-n32-s20.launch.log"
printf 'GPU 1: PID=%s, launcher log=%s\n' "$PID1" "$LAUNCH_LOG_DIR/mxfp4-rtn-n128-s50.launch.log"

