#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${ARK_BENCH_PYTHON:-${REPO_ROOT}/ark-torch-212/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/bench_sparse_topk_sweeps}"
WARMUP="${WARMUP:-2}"
ITERS="${ITERS:-3}"

if [[ -z "${ZE_AFFINITY_MASK:-}" || -z "${CMPLR_ROOT:-}" || -z "${ONEAPI_ROOT:-}" ]]; then
    echo "Benchmark environment is not initialized." >&2
    echo "Run: source benchmarks/source_env_xpu67_oneapi2025.sh" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

seq_lens=(16000 32000 75600)
layouts=(NHD HND)
topks=(0.5 0.4 0.3 0.2 0.1)

for layout in "${layouts[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
        csv_path="${OUTPUT_DIR}/bench_sparse_topk_${layout}_seqlen${seq_len}_qtile256.csv"
        echo "Running layout=${layout} seq_len=${seq_len} -> ${csv_path}"
        "${PYTHON_BIN}" benchmarks/bench_sparse_topk.py \
            --seq-len "${seq_len}" \
            --tensor-layout "${layout}" \
            --topk "${topks[@]}" \
            --q-tile-override 256 \
            --sparse-q-block-tokens 256 \
            --sparse-k-block-tokens 64 \
            --warmup "${WARMUP}" \
            --iters "${ITERS}" \
            --output-csv "${csv_path}"
    done
done
