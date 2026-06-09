#!/bin/bash
set -e

# Nightly CPU pipeline: runs the slower integration and end-to-end suites that
# are intentionally excluded from the fast PR unit-test pipeline (run_ut.sh).
# Triggered on a daily schedule from .azure-pipelines/nightly-test.yml.

source /auto-round/.azure-pipelines/scripts/change_color.sh

LOG_DIR=/auto-round/log_dir
mkdir -p "${LOG_DIR}"
SUMMARY_LOG="${LOG_DIR}/results_summary.log"

function setup_environment() {
    echo "##[group]set up nightly env..."
    export TZ='Asia/Shanghai'
    export TQDM_MININTERVAL=120
    export HF_HUB_DISABLE_PROGRESS_BARS=1

    uv pip install pytest-cov pytest-html
    uv pip install -U chardet

    # install latest gguf for the conversion / e2e tests
    cd ~ || exit 1
    git clone -b master --quiet --single-branch https://github.com/ggml-org/llama.cpp.git \
        && cd llama.cpp/gguf-py && uv pip install .

    cd /auto-round && uv pip install .

    export LD_LIBRARY_PATH=${HOME}/.venv/lib/:$LD_LIBRARY_PATH
    export FORCE_BF16=1
    export COVERAGE_RCFILE=/auto-round/.azure-pipelines/scripts/ut/.coverage
    echo "##[endgroup]"

    uv pip list
}

function run_pytest_dir() {
    # $1: directory to scan for tests, $2: log prefix
    local test_dir=$1
    local prefix=$2

    if [ ! -d "${test_dir}" ]; then
        echo "##[warning]Directory ${test_dir} not found, skipping."
        return
    fi

    for test_file in $(find "${test_dir}" -name "test*.py" | sort); do
        echo "##[group]Running ${test_file}..."
        local test_basename=$(basename "${test_file}" .py)
        local ut_log_name=${LOG_DIR}/${prefix}_${test_basename}.log

        numactl --physcpubind="${NUMA_CPUSET:-0-15}" --membind="${NUMA_NODE:-0}" \
            python -m pytest --cov=auto_round --cov-report= --html=report.html --self-contained-html \
                --cov-report xml:coverage.xml --cov-append \
                -vs --disable-warnings "${test_file}" 2>&1 | tee "${ut_log_name}"
        echo "##[endgroup]"
    done
}

function run_integration_test() {
    echo "##[group]set up integration deps (INC + LLMC)..."
    INC_PT_ONLY=1 uv pip install -r /auto-round/test/unit/test_cpu/requirements_inc.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu
    BUILD_TYPE="nightly" uv pip install -r /auto-round/test/unit/test_cpu/requirements_llmc.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu
    uv pip uninstall auto-round
    cd /auto-round && uv pip install .
    echo "##[endgroup]"

    cd /auto-round/test || exit 1
    run_pytest_dir ./integration/test_cpu integration
}

function run_e2e_test() {
    cd /auto-round/test || exit 1
    run_pytest_dir ./e2e/test_cpu e2e
}

function print_summary() {
    local status=0
    while IFS= read -r line; do
        if [[ "$line" == *"FAILED"* ]]; then
            $LIGHT_RED && echo "$line" && $RESET
            status=1
        elif [[ "$line" == *"PASSED"* ]]; then
            $LIGHT_GREEN && echo "$line" && $RESET
        elif [[ "$line" == *"NO_TESTS"* ]]; then
            $LIGHT_YELLOW && echo "$line" && $RESET
        else
            echo "$line"
        fi
    done < "${SUMMARY_LOG}"
    exit $status
}

function collect_log() {
    python /auto-round/.azure-pipelines/scripts/ut/collect_result.py \
        --test-type "Nightly Tests" --log-pattern "integration_*.log" --log-dir ${LOG_DIR} --summary-log ${SUMMARY_LOG}
    python /auto-round/.azure-pipelines/scripts/ut/collect_result.py \
        --test-type "Nightly Tests" --log-pattern "e2e_*.log" --log-dir ${LOG_DIR} --summary-log ${SUMMARY_LOG}

    cp .coverage "${LOG_DIR}/.coverage.nightly" 2>/dev/null || true
    cp coverage.xml "${LOG_DIR}/" 2>/dev/null || true
    cp report.html "${LOG_DIR}/" 2>/dev/null || true
}

function main() {
    setup_environment
    run_integration_test
    run_e2e_test
    collect_log
    print_summary
}

main
