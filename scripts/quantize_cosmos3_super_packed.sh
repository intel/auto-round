#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/yiliu7/workspace/venvs/ar/bin/python}"

exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/quantize_cosmos3_super_packed.py" \
  --model-root "/storage/yiliu7/nvidia/Cosmos3-Super" \
  --output-dir "/storage/yiliu7/nvidia/Cosmos3-Super-W4A16-packed" \
  --mode full \
  --iters 0 \
  "$@"
