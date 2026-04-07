#!/bin/bash
set -euo pipefail

PATTERN='[-a-zA-Z0-9_]*='

for i in "$@"; do
    case $i in
    --model_name=*)
        model_name=$(echo $i | sed "s/${PATTERN}//")
        ;;
    --scheme=*)
        scheme=$(echo $i | sed "s/${PATTERN}//")
        ;;
    *)
        echo "Parameter $i not recognized."
        exit 1
        ;;
    esac
done

readonly WORKSPACE_DIR="/auto-round"
readonly LOG_DIR="${WORKSPACE_DIR}/log_dir"
readonly PERF_SCRIPT_DIR="${WORKSPACE_DIR}/.azure-pipelines/scripts/performance"
readonly BASELINE_GIT_URL="git+https://github.com/intel/auto-round.git"
readonly ITERS=200

log_group_start() { echo "##[group]$1"; }
log_group_end()   { echo "##[endgroup]"; }
log_info()        { echo -e "[\033[32mINFO\033[0m] $1"; }
log_err()         { echo -e "[\033[31mERROR\033[0m] $1" >&2; }

function setup_environment() {
    log_group_start "Set up environment..."

    export TZ='Asia/Shanghai'
    export TQDM_MININTERVAL=60
    export HF_HUB_DISABLE_PROGRESS_BARS=1
    export UV_NO_PROGRESS=1
    export UV_SYSTEM_PYTHON=1

    log_info "Creating log directory: ${LOG_DIR}"
    mkdir -p "${LOG_DIR}"

    log_info "Downloading model: ${model_name}"
    hf download "${model_name}"

    log_group_end
}

function install_auto_round() {
    local install_source=$1
    local mode_name=$2

    log_group_start "Install requirements for [${mode_name}]..."

    (
        cd "${WORKSPACE_DIR}"
        log_info "Uninstalling existing auto-round..."
        uv pip uninstall auto-round || true
        
        log_info "Installing auto-round from: ${install_source}"
        BUILD_HPU_ONLY=1 uv pip install "${install_source}"
    )

    log_group_end
}

function run_performance_test() {
    local test_mode=$1
    local log_file="${LOG_DIR}/perf_test_${test_mode}.log"

    log_group_start "Run ${test_mode} performance test (${scheme})..."

    (
        cd "${PERF_SCRIPT_DIR}"
        log_info "Executing auto-round for ${scheme}. Logging to ${log_file}"
        auto-round \
            --model_name "${model_name}" \
            --scheme "${scheme}" \
            --iters "${ITERS}" \
            --enable_torch_compile \
            --device hpu \
            --output_dir "./${test_mode}" 2>&1 | tee -a "${log_file}"
    )

    log_group_end
}

function run_performance_check() {
    log_group_start "Check performance results..."

    (
        cd "${PERF_SCRIPT_DIR}"
        log_info "Executing check_performance.py"
        python check_performance.py
    )

    log_group_end
}

function main() {
    setup_environment

    install_auto_round "." "current"
    run_performance_test "current"

    install_auto_round "${BASELINE_GIT_URL}" "baseline"
    run_performance_test "baseline"

    run_performance_check

    log_info "All tasks completed successfully."
}

main "$@"