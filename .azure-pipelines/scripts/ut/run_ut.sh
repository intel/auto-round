#!/bin/bash
set -e

test_part=$1

source /auto-round/.azure-pipelines/scripts/change_color.sh

LOG_DIR=/auto-round/log_dir
mkdir -p "${LOG_DIR}"
SUMMARY_LOG="${LOG_DIR}/results_summary.log"

function setup_environment() {
    echo "##[group]set up UT env..."
    echo "NUMA_NODE=${NUMA_NODE}"
    echo "NUMA_CPUSET=${NUMA_CPUSET}"
    export TZ='Asia/Shanghai'
    export TQDM_MININTERVAL=120
    export HF_HUB_DISABLE_PROGRESS_BARS=1

    uv pip install pytest-cov
    uv pip install -U chardet
    uv pip list

    # install latest gguf for ut test
    cd ~ || exit 1
    git clone -b master --quiet --single-branch https://github.com/ggml-org/llama.cpp.git && cd llama.cpp/gguf-py && uv pip install .
    
    cd /auto-round && uv pip install .

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
        else
            echo "$line"
        fi
    done < "${SUMMARY_LOG}"
    exit $status
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

function run_unit_test() {
    cd /auto-round/test || exit 1

    # Split test files into 5 parts
    find ./test_cpu -name "test*.py" | grep -Ev "test_llmc|test_inc" | sort > all_tests.txt
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
        echo "##[group]Running ${test_file}..."
        local test_basename=$(basename ${test_file} .py)
        local ut_log_name=${LOG_DIR}/unittest_${test_basename}.log

        numactl --physcpubind="${NUMA_CPUSET:-0-15}" --membind="${NUMA_NODE:-0}" \
            pytest --cov=auto_round --cov-report= --cov-append \
                -vs --disable-warnings ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done
}

function run_inc_unit_test() {
    echo "##[group]set up INC UT env..."
    INC_PT_ONLY=1 uv pip install -r /auto-round/test/test_cpu/requirements_inc.txt --extra-index-url https://download.pytorch.org/whl/cpu
    echo "##[endgroup]"

    cd /auto-round/test || exit 1

    for test_file in $(find ./test_cpu -name "test_inc*.py" | sort); do
        echo "##[group]Running ${test_file}..."
        local test_basename=$(basename ${test_file} .py)
        local ut_log_name=${LOG_DIR}/unittest_${test_basename}.log

        numactl --physcpubind="${NUMA_CPUSET:-0-15}" --membind="${NUMA_NODE:-0}" \
            pytest --cov=auto_round --cov-report= --cov-append \
                -vs --disable-warnings ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done
}

function run_llmc_unit_test() {
    echo "##[group]set up LLMC UT env..."
    BUILD_TYPE="nightly" uv pip install -r /auto-round/test/test_cpu/requirements_llmc.txt --extra-index-url https://download.pytorch.org/whl/cpu
    uv pip uninstall auto-round
    cd /auto-round && uv pip install .
    echo "##[endgroup]"

    cd /auto-round/test || exit 1

    for test_file in $(find ./test_cpu -name "test_llmc*.py" | sort); do
        echo "##[group]Running ${test_file}..."
        local test_basename=$(basename ${test_file} .py)
        local ut_log_name=${LOG_DIR}/unittest_${test_basename}.log

        numactl --physcpubind="${NUMA_CPUSET:-0-15}" --membind="${NUMA_NODE:-0}" \
            pytest --cov=auto_round --cov-report= --cov-append \
                -vs --disable-warnings ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done
}

function collect_log() {
    python /auto-round/.azure-pipelines/scripts/ut/collect_result.py \
        --test-type "Unit Tests" --log-pattern "unittest_test_*.log" --log-dir ${LOG_DIR} --summary-log ${SUMMARY_LOG}

    cp .coverage "${LOG_DIR}/.coverage.part${test_part}"
}

function main() {
    setup_environment
    run_unit_test
    if [ "$test_part" -eq 5 ]; then
        run_inc_unit_test
        run_llmc_unit_test
    fi
    collect_log
    check_storage_usage
    print_summary
}

main
