#!/bin/bash
set -e

source /auto-round/.azure-pipelines/scripts/change_color.sh
source /auto-round/.azure-pipelines/scripts/ut/detect_changed_tests.sh

TIMEOUT=30
SESSION_TIMEOUT=600

function setup_environment() {
    echo "##[group]set up UT env..."
    uv pip install pytest-cov pytest-timeout
    uv pip list
    echo "##[endgroup]"

    git config --global --add safe.directory /auto-round
    rm -rf /auto-round/auto_round
    cd /auto-round/test || exit 1

    echo "##[group]check xpu env..."
    echo "ZE_AFFINITY_MASK: ${ZE_AFFINITY_MASK}"
    python -c "import torch; print('torch:', torch.__version__); print('xpu available:', torch.xpu.is_available()); print('xpu count:', torch.xpu.device_count())"
    echo "##[endgroup]"

    export TZ='Asia/Shanghai'
    export TQDM_MININTERVAL=60
    export HF_HUB_DISABLE_PROGRESS_BARS=1
    export LD_LIBRARY_PATH=${HOME}/.venv/lib/:$LD_LIBRARY_PATH
    export COVERAGE_RCFILE=/auto-round/.azure-pipelines/scripts/ut/.coveragerc

    LOG_DIR=/auto-round/log_dir
    mkdir -p ${LOG_DIR}
    ut_log_name=${LOG_DIR}/ut.log
    SUMMARY_LOG="${LOG_DIR}/results_summary.log"
}

function run_unit_test() {
    auto_round_path=$(python -c 'import auto_round; print(auto_round.__path__[0])')

    local ark_tests xpu_tests common_tests
    ark_tests=$(filter_changed_tests "test" "$(find ./unit/test_ark -name "test*.py" | sort)")
    xpu_tests=$(filter_changed_tests "test" "$(find ./unit/test_xpu -name "test*.py" | sort)")
    common_tests=$(filter_changed_tests "test" "$(find ./unit/common -name "test*.py" | sort)")

    for test_file in ${common_tests}; do
        local test_basename=$(basename ${test_file} .py)

        echo "##[group]Running common ${test_file}..."
        local ut_log_name="${LOG_DIR}/unittest_common_${test_basename}.log"
        numactl --physcpubind="${NUMA_CPUSET:-0-27}" --membind="${NUMA_NODE:-0}" \
            pytest --timeout=${TIMEOUT} --session-timeout=${SESSION_TIMEOUT} \
                --cov="${auto_round_path}" --cov-report= --cov-append -vs \
                --junitxml="${ut_log_name%.log}.xml" ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done

    for test_file in ${xpu_tests}; do
        local test_basename=$(basename ${test_file} .py)

        echo "##[group]Running xpu ${test_file}..."
        local ut_log_name="${LOG_DIR}/unittest_xpu_${test_basename}.log"
        numactl --physcpubind="${NUMA_CPUSET:-0-27}" --membind="${NUMA_NODE:-0}" \
            pytest --timeout=${TIMEOUT} --session-timeout=${SESSION_TIMEOUT} \
                --cov="${auto_round_path}" --cov-report= --cov-append -vs \
                --junitxml="${ut_log_name%.log}.xml" ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done

    for test_file in ${ark_tests}; do
        local test_basename=$(basename ${test_file} .py)

        echo "##[group]Running ark ${test_file}..."
        local ut_log_name="${LOG_DIR}/unittest_ark_${test_basename}.log"
        numactl --physcpubind="${NUMA_CPUSET:-0-27}" --membind="${NUMA_NODE:-0}" \
            pytest --timeout=${TIMEOUT} --session-timeout=${SESSION_TIMEOUT} \
                --cov="${auto_round_path}" --cov-report= --cov-append -vs \
                --junitxml="${ut_log_name%.log}.xml" ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done

}

function run_unit_test_llmc() {
    local llmc_tests
    llmc_tests=$(filter_changed_tests "test" "$(find ./integration/test_xpu -name "test_llmc_integration.py" | sort)")
    if [ -z "${llmc_tests}" ]; then
        echo "No changed XPU LLMC test file, skip."
        return 0
    fi

    echo "##[group]set up llmc UT env..."
    BUILD_TYPE="nightly" uv pip install -r ./unit/test_xpu/requirements_llmc.txt
    uv pip list
    echo "##[endgroup]" 

    auto_round_path=$(python -c 'import auto_round; print(auto_round.__path__[0])')

    for test_file in ${llmc_tests}; do
        local test_basename=$(basename ${test_file} .py)

        echo "##[group]Running xpu llmc ${test_file}..."
        local ut_log_name="${LOG_DIR}/unittest_xpu_${test_basename}.log"
        numactl --physcpubind="${NUMA_CPUSET:-0-27}" --membind="${NUMA_NODE:-0}" \
            pytest --timeout=${TIMEOUT} --session-timeout=${SESSION_TIMEOUT} \
                --cov="${auto_round_path}" --cov-report= --cov-append -vs \
                --junitxml="${ut_log_name%.log}.xml" ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done
}

function print_summary() {
    python /auto-round/.azure-pipelines/scripts/ut/print_summary.py --summary-log "${SUMMARY_LOG}"
    exit $?
}

function collect_log() {
    touch "${SUMMARY_LOG}"
    python /auto-round/.azure-pipelines/scripts/ut/collect_result.py \
        --test-type "Unit Tests" --log-pattern "unittest_*.log" --log-dir ${LOG_DIR} --summary-log ${SUMMARY_LOG}
    if [ -f .coverage ]; then
        cp .coverage "${LOG_DIR}/.coverage"
        python -m coverage xml -o "${LOG_DIR}/coverage.xml"
        python -m coverage html -d "${LOG_DIR}/htmlcov"
    else
        echo "No coverage data (no test selected), skip coverage report."
        echo "##vso[task.setvariable variable=HAS_COVERAGE]false"
    fi
}

function print_coverage() {
    echo "##[group]overall code coverage..."
    [ -f .coverage ] && python -m coverage report || echo "No coverage data."
    echo "##[endgroup]"
}

function main() {
    setup_environment
    init_changed_tests
    scope_changed_tests "$(cd /auto-round && find test/unit/common test/unit/test_ark test/unit/test_xpu test/integration/test_xpu -name "test_*.py" 2>/dev/null)"
    if [[ "${UT_MODE}" == "llmc" ]]; then
        run_unit_test_llmc
    else
        run_unit_test
    fi
    collect_log
    print_coverage
    print_summary
}

main "$@"
