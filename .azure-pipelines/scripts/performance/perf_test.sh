#!/bin/bash
set -e

function setup_environment() {
    export TZ='Asia/Shanghai'
    export TQDM_MININTERVAL=60
    export HF_HUB_DISABLE_PROGRESS_BARS=1
    export UV_NO_PROGRESS=1
    export UV_SYSTEM_PYTHON=1

    LOG_DIR="/auto-round/log_dir"
    mkdir -p "${LOG_DIR}"

    model_name="Qwen/Qwen3-0.6B"
    hf download ${model_name}
    hf download NeelNanda/pile-10k --repo-type dataset
}

function install_requirements() {
    echo "##[group]set up env..."
    cd /auto-round
    uv pip uninstall auto-round || true
    BUILD_HPU_ONLY=1 uv pip install .
    echo "##[endgroup]"
}

function install_baseline_requirements() {
    echo "##[group]set up baseline env..."
    cd /auto-round
    uv pip uninstall auto-round || true
    BUILD_HPU_ONLY=1 uv pip install git+https://github.com/intel/auto-round.git
    echo "##[endgroup]"
}

function run_performance_test() {
    test_mode=$1
    cd /auto-round/.azure-pipelines/scripts/performance
    local log_file="perf_test_${test_mode}.log"
    rm -rf "saved" "${LOG_DIR}/${log_file}"
    echo "##[group]run ${test_mode} performance test..."
    auto-round --model_name ${model_name} --bits 4 --iters 200 --enable_torch_compile --device hpu --output_dir ./saved 2>&1 | tee -a "${LOG_DIR}/${log_file}"
    echo "##[endgroup]"
}

function main() {
    setup_environment

    install_requirements
    run_performance_test "current"

    install_baseline_requirements
    run_performance_test "baseline"

    cd /auto-round/.azure-pipelines/scripts/performance
    python check_performance.py
}

main
