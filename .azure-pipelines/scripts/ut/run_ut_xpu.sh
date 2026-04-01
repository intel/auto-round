#!/bin/bash
set -e

source /auto-round/.azure-pipelines/scripts/change_color.sh

function setup_environment() {
    echo "##[group]set up UT env..."
    uv pip install pytest-cov pytest-html
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
    export COVERAGE_RCFILE=/auto-round/.azure-pipelines/scripts/ut/.coverage

    LOG_DIR=/auto-round/log_dir
    mkdir -p ${LOG_DIR}
    ut_log_name=${LOG_DIR}/ut.log
    SUMMARY_LOG="${LOG_DIR}/results_summary.log"
}

function run_unit_test() {
    auto_round_path=$(python -c 'import auto_round; print(auto_round.__path__[0])')

    for test_file in $(find ./test_ark -name "test*.py" | sort); do
        local test_basename=$(basename ${test_file} .py)

        echo "##[group]Running ark ${test_file}..."
        local ut_log_name="${LOG_DIR}/unittest_ark_${test_basename}.log"
        numactl --physcpubind="${NUMA_CPUSET:-0-27}" --membind="${NUMA_NODE:-0}" pytest --cov="${auto_round_path}" \
            --cov-report term --html=report.html --self-contained-html \
            --cov-report xml:coverage.xml --cov-append -vs --disable-warnings \
            ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done

    for test_file in $(find ./test_xpu -name "test*.py" | sort); do
        local test_basename=$(basename ${test_file} .py)

        echo "##[group]Running xpu ${test_file}..."
        local ut_log_name="${LOG_DIR}/unittest_xpu_${test_basename}.log"
        numactl --physcpubind="${NUMA_CPUSET:-0-27}" --membind="${NUMA_NODE:-0}" pytest --cov="${auto_round_path}" \
            --cov-report term --html=report.html --self-contained-html \
            --cov-report xml:coverage.xml --cov-append -vs --disable-warnings \
            ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done
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
        --test-type "Unit Tests" --log-pattern "unittest_*.log" --log-dir ${LOG_DIR} --summary-log ${SUMMARY_LOG}
    cp .coverage "${LOG_DIR}/.coverage"
    cp report.html ${LOG_DIR}/
    cp coverage.xml ${LOG_DIR}/
}

function main() {
    setup_environment
    run_unit_test
    collect_log
    print_summary
}

main
