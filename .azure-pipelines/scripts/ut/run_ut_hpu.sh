#!/bin/bash
set -e
source /auto-round/.azure-pipelines/scripts/change_color.sh

function setup_environment() {
    # install requirements
    echo "##[group]set up UT env..."
    export TZ='Asia/Shanghai'
    export TQDM_MININTERVAL=60
    export HF_HUB_DISABLE_PROGRESS_BARS=1
    pip install pytest-cov pytest-html
    pip list
    echo "##[endgroup]"

    rm -rf /auto-round/auto_round
    cd /auto-round/test || exit 1

    export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
    export FORCE_BF16=1
    export COVERAGE_RCFILE=/auto-round/.azure-pipelines/scripts/ut/.coverage

    LOG_DIR=/auto-round/log_dir
    mkdir -p ${LOG_DIR}
    ut_log_name=${LOG_DIR}/ut.log
    SUMMARY_LOG="${LOG_DIR}/results_summary.log"
}

function run_unit_test() {
    auto_round_path=$(python -c 'import auto_round; print(auto_round.__path__[0])')

    for test_file in $(find ./test_hpu -name "test*.py" | sort); do
        local test_basename=$(basename ${test_file} .py)

        echo "##[group]Running ${test_file} in HPU lazy mode..."
        local ut_log_name="${LOG_DIR}/unittest_lazy_${test_basename}.log"
        PT_HPU_LAZY_MODE=1 pytest --cov="${auto_round_path}" \
            --cov-report term --html=report.html --self-contained-html \
            --cov-report xml:coverage.xml --cov-append -vs --disable-warnings \
            ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"

        echo "##[group]Running ${test_file} in HPU compile mode..."
        local ut_log_name="${LOG_DIR}/unittest_compile_${test_basename}.log"
        PT_HPU_LAZY_MODE=0 pytest --mode compile --cov="${auto_round_path}" \
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
    cp report.html ${LOG_DIR}/
    cp coverage.xml ${LOG_DIR}/
    cp .coverage "${LOG_DIR}/.coverage"
}

function main() {
    setup_environment
    run_unit_test
    collect_log
    print_summary
}

main