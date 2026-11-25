#!/bin/bash
set -e

REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
ARK_DIR=${REPO_DIR}/auto_round_extension/ark

source ${REPO_DIR}/.azure-pipelines/scripts/change_color.sh

function setup_environment() {
    echo "##[group]set up ARK UT env..."
    cd ${ARK_DIR}
    uv pip install -r requirements.txt
    uv pip install ${REPO_DIR}/ark_wheel/*.whl --no-deps
    uv pip install pytest pytest-timeout pandas
    uv pip list
    echo "##[endgroup]"

    # Isolate from the repo-level pytest config.
    echo -e "[pytest]\ntestpaths = .\n" > ./test/pytest.ini

    LOG_DIR=${REPO_DIR}/log_dir
    mkdir -p ${LOG_DIR}
    rm -f ${LOG_DIR}/unittest_ark_*.log
    SUMMARY_LOG="${LOG_DIR}/results_summary.log"
}

function run_unit_test() {
    cd ${ARK_DIR}

    for test_file in $(find ./test -name "test*.py" | sort); do
        local test_basename=$(basename ${test_file} .py)

        echo "##[group]Running ark ${test_file}..."
        local ut_log_name="${LOG_DIR}/unittest_ark_${test_basename}.log"
        COVERAGE_CORE=sysmon pytest --timeout=600 -v ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done
}

function print_summary() {
    local status=0
    for log_file in ${LOG_DIR}/unittest_*.log; do
        if grep -q "FAILED" "${log_file}"; then
            $LIGHT_RED && echo "FAILED: ${log_file}" && $RESET
            status=1
        elif grep -q "ERROR" "${log_file}"; then
            $LIGHT_RED && echo "ERROR: ${log_file}" && $RESET
            status=1
        elif grep -q "passed" "${log_file}"; then
            $LIGHT_GREEN && echo "PASSED: ${log_file}" && $RESET
        fi
    done
    exit $status
}

function main() {
    setup_environment
    run_unit_test
    print_summary
}

main
