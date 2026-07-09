#!/bin/bash
set -e

test_part=""
failure_log_context=""
declare -a FAILED_BASE_CASES=()
declare -a FAILED_INC_CASES=()
declare -a FAILED_LLMC_CASES=()

function parse_arguments() {

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --failure-context)
                failure_log_context="$2"
                shift 2
                ;;
            --test-part)
                test_part="$2"
                shift 2
                ;;
            *)
                echo "Unknown argument: $1"
                exit 1
                ;;
        esac
    done

    if [[ -z "${test_part}" ]]; then
        echo "Error: test_part is required"
        echo "Usage: run_ut.sh --test-part <part> [--failure-context <path>] [--failed-test-cases <path>]"
        exit 1
    fi
}

parse_arguments "$@"

source /auto-round/.azure-pipelines/scripts/change_color.sh

LOG_DIR=/auto-round/log_dir
mkdir -p "${LOG_DIR}"
SUMMARY_LOG="${LOG_DIR}/results_summary.log"

function setup_inc_environment() {
    echo "##[group]set up INC UT env..."
    INC_PT_ONLY=1 uv pip install -r /auto-round/test/test_cpu/requirements_inc.txt
    echo "##[endgroup]"
}

function setup_llmc_environment() {
    echo "##[group]set up LLMC UT env..."
    BUILD_TYPE="nightly" uv pip install -r /auto-round/test/test_cpu/requirements_llmc.txt
    uv pip uninstall auto-round
    cd /auto-round && uv pip install .
    echo "##[endgroup]"
}

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

function run_test_cases() {
    local tests=("$@")
    if [[ ${#tests[@]} -eq 0 ]]; then
        return
    fi

    cd /auto-round || exit 1

    for test_case in "${tests[@]}"; do
        if [[ -z "${test_case}" ]]; then
            continue
        fi

        local test_path
        test_path="${test_case%%::*}"
        local test_basename
        test_basename=$(basename "${test_path}" .py)
        local ut_log_name=${LOG_DIR}/unittest_${test_basename}.log

        echo "##[group]Running ${test_case}..."
        numactl --physcpubind="${NUMA_CPUSET:-0-15}" --membind="${NUMA_NODE:-0}" \
            python -m pytest --cov=auto_round --cov-report= --cov-append \
                -vs --disable-warnings "${test_case}" 2>&1 | tee "${ut_log_name}"
        echo "##[endgroup]"
    done
}

function run_unit_test() {
    local -a selected_cases=()

    cd /auto-round || exit 1

    # Split test files into 5 parts
    find ./test/test_cpu/core -name "test*.py" | grep -Ev "test_llmc|test_inc" | sort > all_tests.txt
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
    mapfile -t selected_cases < <(sed -n "${start_line},${end_line}p" all_tests.txt)
    run_test_cases "${selected_cases[@]}"
}

function run_inc_unit_test() {
    local -a inc_cases=("$@")
    if [[ ${#inc_cases[@]} -eq 0 ]]; then
        mapfile -t inc_cases < <(find /auto-round/test/test_cpu -name "test_inc*.py" | sort)
    fi

    setup_inc_environment
    run_test_cases "${inc_cases[@]}"
}

function run_llmc_unit_test() {
    local -a llmc_cases=("$@")
    if [[ ${#llmc_cases[@]} -eq 0 ]]; then
        mapfile -t llmc_cases < <(find /auto-round/test/test_cpu -name "test_llmc*.py" | sort)
    fi

    setup_llmc_environment
    run_test_cases "${llmc_cases[@]}"
}

function collect_log() {
    collect_cmd=(
        python /auto-round/.azure-pipelines/scripts/ut/collect_result.py
        --test-type "Unit Tests"
        --log-pattern "unittest_test_*.log"
        --log-dir "${LOG_DIR}"
        --summary-log "${SUMMARY_LOG}"
        --ci-part "${test_part}"
    )

    if [[ -n "${failure_log_context}" ]]; then
        collect_cmd+=(--failure-context "${failure_log_context}")
    fi

    "${collect_cmd[@]}"

    if [[ -f .coverage ]]; then
        cp .coverage "${LOG_DIR}/.coverage.part${test_part}"
    fi
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
