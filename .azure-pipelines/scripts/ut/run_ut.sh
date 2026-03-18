#!/bin/bash
set -e

test_part=$1

source /auto-round/.azure-pipelines/scripts/change_color.sh

LOG_DIR=/auto-round/log_dir
mkdir -p "${LOG_DIR}"
SUMMARY_LOG="${LOG_DIR}/results_summary.log"

function setup_environment() {
    echo "##[group]set up UT env..."
    export TZ='Asia/Shanghai'
    export TQDM_MININTERVAL=120
    export HF_HUB_DISABLE_PROGRESS_BARS=1

    uv pip install pytest-cov pytest-html
    uv pip list
    # workaround for ark test, remove auto_round_kernel_xpu
    package_path=$(uv pip show auto-round-lib | grep Location:|cut -d: -f2)
    rm -rf $package_path/auto_round_kernel/auto_round_kernel_xpu*

    # install latest gguf for ut test
    cd ~ || exit 1
    git clone -b master --quiet --single-branch https://github.com/ggml-org/llama.cpp.git && cd llama.cpp/gguf-py && uv pip install . sentencepiece

    cd /auto-round && uv pip install .

    rm -rf /auto-round/auto_round
    export LD_LIBRARY_PATH=${HOME}/.venv/lib/:$LD_LIBRARY_PATH
    export FORCE_BF16=1
    export COVERAGE_RCFILE=/auto-round/.azure-pipelines/scripts/ut/.coverage
    echo "##[endgroup]"

    uv pip list
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
            status=1
        else
            echo "$line"
        fi
    done < "${SUMMARY_LOG}"
    exit $status
}

function check_storage_usage() {
    echo "##[group]check storage usage..."
    df -h
    du -sh /auto-round
    du -sh /home/hostuser/.cache/huggingface
    du -sh /home/hostuser/.cache/huggingface/hub/*
    du -sh /home/hostuser/.venv
    echo "##[endgroup]"
}

function run_unit_test() {
    cd /auto-round/test || exit 1
    auto_round_path=$(python -c 'import auto_round; print(auto_round.__path__[0])')

    # Split test files into 5 parts
    find ./test_cpu -name "test*.py" | sort > all_tests.txt
    total_lines=$(wc -l < all_tests.txt)
    NUM_CHUNKS=5
    q=$(( total_lines / NUM_CHUNKS ))
    r=$(( total_lines % NUM_CHUNKS ))
    if [ "$test_part" -le "$r" ]; then
        chunk_size=$(( q + 1 ))
        start_line=$(( (test_part - 1) * chunk_size + 1 ))
    else
        chunk_size=$q
        start_line=$(( r * (q + 1) + (test_part - r - 1) * q + 1 ))
    fi
    end_line=$(( start_line + chunk_size - 1 ))
    selected_files=$(sed -n "${start_line},${end_line}p" all_tests.txt)

    for test_file in ${selected_files}; do
        $LIGHT_PURPLE && echo "##[group]Running ${test_file}..." && $RESET
        local test_basename=$(basename ${test_file} .py)
        local ut_log_name=${LOG_DIR}/unittest_${test_basename}.log

        numactl --physcpubind="${NUMA_CPUSET:-0-15}" --membind="${NUMA_NODE:-0}" \
            python -m pytest --cov="${auto_round_path}" --cov-report term --html=report.html --self-contained-html \
                --cov-report xml:coverage.xml --cov-append \
                -vs --disable-warnings ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done

    python /auto-round/.azure-pipelines/scripts/ut/collect_result.py \
        --test-type "Unit Tests" --log-pattern "unittest_test_*.log" --log-dir ${LOG_DIR} --summary-log ${SUMMARY_LOG}

    # if ut pass, collect the coverage file into artifacts
    cp .coverage "${LOG_DIR}/.coverage.part${test_part}"
}

function main() {
    setup_environment
    run_unit_test
    check_storage_usage
    print_summary
}

main
