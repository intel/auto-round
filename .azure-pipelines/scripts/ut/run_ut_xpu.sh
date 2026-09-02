#!/bin/bash
set -e

test_part=${UT_MODE}

source /auto-round/.azure-pipelines/scripts/change_color.sh
source /auto-round/.azure-pipelines/scripts/ut/detect_changed_tests.sh
source /auto-round/.azure-pipelines/scripts/ut/retry_failed_tests.sh

TIMEOUT=30
SESSION_TIMEOUT=600

function setup_environment() {
    echo "##[group]set up UT env..."
    echo "Install unit report dependencies ..."
    uv pip install pytest-cov
    uv pip install -U chardet

    # Keep the GGUF conversion helpers in sync with the model conversion code.
    # The common GGUF tests exercise MODEL_ARCH entries that are only available
    # in the current llama.cpp master branch.
    echo "Install latest gguf for ut test ..."
    cd ~ || exit 1
    git clone -b master --quiet --single-branch https://github.com/ggml-org/llama.cpp.git \
        && cd llama.cpp/gguf-py \
        && uv pip install .
    echo "List final dependencies ..."
    uv pip list
    echo "##[endgroup]"

    git config --global --add safe.directory /auto-round
    cd /auto-round/test || exit 1

    echo "##[group]check xpu env..."
    echo "ZE_AFFINITY_MASK: ${ZE_AFFINITY_MASK}"
    python -c "import torch; print('torch:', torch.__version__); print('xpu available:', torch.xpu.is_available()); print('xpu count:', torch.xpu.device_count())"
    echo "##[endgroup]"

    export TZ='Asia/Shanghai'
    export TQDM_MININTERVAL=60
    export HF_HUB_DISABLE_PROGRESS_BARS=1
    export LD_LIBRARY_PATH=${HOME}/.venv/lib/:$LD_LIBRARY_PATH
    export COVERAGE_RCFILE=/auto-round/.azure-pipelines/scripts/ut/coveragerc/xpu.coveragerc

    LOG_DIR=/auto-round/log_dir
    mkdir -p ${LOG_DIR}
    ut_log_name=${LOG_DIR}/ut.log
    SUMMARY_LOG="${LOG_DIR}/results_summary.log"
}

function run_pytest() {
    local test_case=$1
    local ut_log_name=$2

    echo "##[group]Running ${test_case}..."
    # Record the test targets so a retry can rerun exactly these cases.
    printf '%s\n' ${test_case} > "${ut_log_name%.log}.list"
    numactl --physcpubind="${NUMA_CPUSET:-0-27}" --membind="${NUMA_NODE:-0}" \
        pytest -m "not skip_ci" --cov=auto_round --cov-report= --cov-append -vs \
            --junitxml="${ut_log_name%.log}.xml" ${test_case} 2>&1 | tee ${ut_log_name}
    echo "##[endgroup]"
}


function run_common_group() {
    # Run a group of common test files together in a single pytest invocation.
    # $1: group name (used for log file), remaining args: test files
    local group_name=$1
    shift
    local group_tests
    group_tests=$(filter_changed_tests "test" "$*")

    if [ -n "${group_tests}" ]; then
        local ut_log_name="${LOG_DIR}/unittest_common_${group_name}.log"
        run_pytest "${group_tests}" "${ut_log_name}"
    fi
}

function run_common_unit_test() {
    cd /auto-round/test || exit 1
    run_if_retry && return 0
    # common test case for cpu/gpu/xpu
    # Group cases by the first-level folder under unit/common; a single test
    # file placed directly under unit/common (e.g. test_main.py) runs on its own.
    for entry in $(find ./unit/common -mindepth 1 -maxdepth 1 | sort); do
        if [ -d "${entry}" ]; then
            local group_name=$(basename "${entry}")
            run_common_group "${group_name}" "$(find "${entry}" -name "test*.py" | sort)"
        elif [[ "$(basename "${entry}")" == test*.py ]]; then
            local group_name=$(basename "${entry}" .py)
            run_common_group "${group_name}" "${entry}"
        fi
    done
}

function run_unit_test() {
    cd /auto-round/test || exit 1
    run_if_retry && return 0

    local xpu_tests
    xpu_tests=$(filter_changed_tests "test" "$(find ./unit/test_xpu -name "test*.py" | sort)")
    if [ -z "${xpu_tests}" ]; then
        echo "No changed XPU test file, skip."
        return 0
    fi
    
    for test_file in ${xpu_tests}; do
        local test_basename=$(basename ${test_file} .py)
        local ut_log_name="${LOG_DIR}/unittest_${test_basename}.log"
        run_pytest "${test_file}" "${ut_log_name}"
    done
}

function run_unit_test_ark() {
    cd /auto-round/test || exit 1
    run_if_retry && return 0
    
    local ark_tests
    ark_tests=$(filter_changed_tests "test" "$(find ./unit/test_ark -name "test*.py" | sort)")
    if [ -z "${ark_tests}" ]; then
        echo "No changed ARK test file, skip."
        return 0
    fi

    for test_file in ${ark_tests}; do
        local test_basename=$(basename ${test_file} .py)
        local ut_log_name="${LOG_DIR}/unittest_${test_basename}.log"
        run_pytest "${test_file}" "${ut_log_name}"
    done
}

function run_unit_test_llmc() {
    cd /auto-round/test || exit 1
    run_if_retry && return 0

    local llmc_tests
    llmc_tests=$(filter_changed_tests "test" "$(find ./integration/test_xpu -name "test_llmc_integration.py" | sort)")
    if [ -z "${llmc_tests}" ]; then
        echo "No changed XPU LLMC test file, skip."
        return 0
    fi

    echo "##[group]set up llmc UT env..."
    BUILD_TYPE="nightly" uv pip install -r ./integration/test_xpu/requirements_llmc.txt
    uv pip list
    echo "##[endgroup]" 

    for test_file in ${llmc_tests}; do
        local test_basename=$(basename ${test_file} .py)
        local ut_log_name="${LOG_DIR}/unittest_${test_basename}.log"
        run_pytest "${test_file}" "${ut_log_name}"
    done
}

function print_summary() {
    python /auto-round/.azure-pipelines/scripts/ut/print_summary.py --summary-log "${SUMMARY_LOG}"
    exit $?
}

function check_storage_usage() {
    echo "##[group]check storage usage..."
    df -h
    du -sh /auto-round || true
    du -sh /home/hostuser/.cache/huggingface || true
    du -sh /home/hostuser/.cache/huggingface/hub/* || true
    du -sh /home/hostuser/.venv || true
    echo "##[endgroup]"
}

function collect_log() {
    touch "${SUMMARY_LOG}"
    python /auto-round/.azure-pipelines/scripts/ut/collect_result.py \
        --test-type "Unit Tests" --log-pattern "unittest_*.log" --log-dir ${LOG_DIR} \
        --summary-log ${SUMMARY_LOG} --failed-logs-dir "${LOG_DIR}/failed_logs"

    if [ -f .coverage ]; then
        cp .coverage "${LOG_DIR}/.coverage.part${test_part}"
        # Keep .coverage in the failure artifact so a retry can accumulate onto it.
        if [ -d "${LOG_DIR}/failed_logs" ]; then
            cp .coverage "${LOG_DIR}/failed_logs/.coverage"
        fi
    fi
}

function main() {
    setup_environment
    init_changed_tests
    scope_changed_tests "$(cd /auto-round && find test/unit/common test/unit/test_ark test/unit/test_xpu test/integration/test_xpu -name "test_*.py" 2>/dev/null)"
    if [[ "$test_part" == "llmc" ]]; then
        run_unit_test_llmc
    elif [[ "$test_part" == "ark" ]]; then
        run_unit_test_ark
    elif [[ "$test_part" == "common" ]]; then
        run_common_unit_test
    elif [[ "$test_part" == "base" ]]; then
        run_unit_test
    else
        echo "invalid name $test_part"
    fi
    collect_log
    check_storage_usage
    print_summary
}

main "$@"
