#!/bin/bash
set -xe

function setup_environment() {
    export TZ='Asia/Shanghai'
    export TQDM_MININTERVAL=60
    export HF_HUB_DISABLE_PROGRESS_BARS=1
    export UV_NO_PROGRESS=1
    export UV_SYSTEM_PYTHON=1
}

function install_requirements() {
    uv pip uninstall auto-round || true
    BUILD_HPU_ONLY=1 uv pip install .
}

function install_baseline_requirements() {
    uv pip uninstall auto-round || true
    BUILD_HPU_ONLY=1 uv pip install git+https://github.com/intel/auto-round.git
}

function run_performance_test() {
    test_mode=$1
    log_file="perf_test_${test_mode}.log"
    rm -rf ./saved ${log_file}
    auto-round --model_name Qwen/Qwen3-8B --bits 4 --iters 200 --enable_torch_compile --output_dir ./saved | tee ${log_file}
}

function main() {
    setup_environment

    install_requirements
    run_performance_test "current"

    install_baseline_requirements
    run_performance_test "baseline"

    python check_performance.py
}

main
