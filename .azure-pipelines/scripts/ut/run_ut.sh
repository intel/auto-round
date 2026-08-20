#!/bin/bash
set -e

test_part=${UT_MODE}

source /auto-round/.azure-pipelines/scripts/change_color.sh
source /auto-round/.azure-pipelines/scripts/ut/detect_changed_tests.sh

LOG_DIR=/auto-round/log_dir
mkdir -p "${LOG_DIR}"
SUMMARY_LOG="${LOG_DIR}/results_summary.log"

TIMEOUT=30
SESSION_TIMEOUT=600

function setup_environment() {
    echo "##[group]set up UT env..."
    echo "NUMA_NODE=${NUMA_NODE}"
    echo "NUMA_CPUSET=${NUMA_CPUSET}"
    export TZ='Asia/Shanghai'
    export TQDM_MININTERVAL=120
    export HF_HUB_DISABLE_PROGRESS_BARS=1

    # install latest gguf for ut test
    cd ~ || exit 1
    git clone -b master --quiet --single-branch https://github.com/ggml-org/llama.cpp.git && cd llama.cpp/gguf-py && uv pip install .

    # install unit report dependencies
    uv pip install pytest-cov pytest-timeout
    uv pip install -U chardet
    uv pip list

    # install auto-round for unit tests
    cd /auto-round && uv pip install .

    export LD_LIBRARY_PATH=${HOME}/.venv/lib/:$LD_LIBRARY_PATH
    export FORCE_BF16=1
    export COVERAGE_RCFILE=/auto-round/.azure-pipelines/scripts/ut/.coveragerc
    echo "##[endgroup]"

    uv pip list
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

function run_common_unit_test() {
    cd /auto-round/test || exit 1

    # common test case for cpu/gpu/xpu
    local common_tests
    common_tests=$(filter_changed_tests "test" "$(find ./unit/common -name "test*.py" | sort)")

    if [ -n "${common_tests}" ]; then
        echo "##[group]Running common tests..."
        local ut_log_name="${LOG_DIR}/unittest_test_common.log"
        numactl --physcpubind="${NUMA_CPUSET:-0-27}" --membind="${NUMA_NODE:-0}" \
            pytest --timeout=${TIMEOUT} --session-timeout=1200 \
                --cov=auto_round --cov-report= --cov-append -vs \
                --junitxml="${ut_log_name%.log}.xml" ${common_tests} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    fi
}

function run_unit_test() {
    cd /auto-round/test || exit 1

    # Split cpu specific test files into 4 parts.
    # Only fast unit tests run in PR CI; integration (inc/llmc) and e2e suites
    # run in the nightly pipelines (see nightly-test.yml).
    find ./unit/test_cpu -name "test*.py" | sort > all_tests.txt
    total_lines=$(wc -l < all_tests.txt)
    NUM_CHUNKS=4
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
    selected_files=$(filter_changed_tests "test" "${selected_files}")

    if [ -z "${selected_files}" ]; then
        echo "No changed unit test file in part ${test_part}, skip."
        return 0
    fi

    for test_file in ${selected_files}; do
        echo "##[group]Running ${test_file}..."
        local test_basename=$(basename ${test_file} .py)
        local ut_log_name=${LOG_DIR}/unittest_${test_basename}.log

        numactl --physcpubind="${NUMA_CPUSET:-0-15}" --membind="${NUMA_NODE:-0}" \
            pytest -m "not skip_ci" --timeout=${TIMEOUT} --session-timeout=${SESSION_TIMEOUT} \
                --cov=auto_round --cov-report= --cov-append \
                -vs --junitxml="${ut_log_name%.log}.xml" ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done
}

function run_inc_unit_test() {
    local selected_files
    selected_files=$(filter_changed_tests "test/integration" \
        "$(cd /auto-round/test/integration && find ./test_cpu -name "test_inc*.py" | sort)")
    if [ -z "${selected_files}" ]; then
        echo "No changed INC test file, skip."
        return 0
    fi

    echo "##[group]set up INC UT env..."
    INC_PT_ONLY=1 uv pip install -r /auto-round/test/integration/test_cpu/requirements_inc.txt --extra-index-url https://download.pytorch.org/whl/cpu
    echo "##[endgroup]"

    cd /auto-round/test/integration || exit 1

    for test_file in ${selected_files}; do
        echo "##[group]Running ${test_file}..."
        local test_basename=$(basename ${test_file} .py)
        local ut_log_name=${LOG_DIR}/unittest_${test_basename}.log

        numactl --physcpubind="${NUMA_CPUSET:-0-15}" --membind="${NUMA_NODE:-0}" \
            pytest --cov=auto_round --cov-report= --cov-append \
                -vs --junitxml="${ut_log_name%.log}.xml" ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done
}

function run_llmc_unit_test() {
    local selected_files
    selected_files=$(filter_changed_tests "test/integration" \
        "$(cd /auto-round/test/integration && find ./test_cpu -name "test_llmc*.py" | sort)")
    if [ -z "${selected_files}" ]; then
        echo "No changed LLMC test file, skip."
        return 0
    fi

    echo "##[group]set up LLMC UT env..."
    BUILD_TYPE="nightly" uv pip install -r /auto-round/test/integration/test_cpu/requirements_llmc.txt --extra-index-url https://download.pytorch.org/whl/cpu
    uv pip uninstall auto-round
    cd /auto-round && uv pip install .
    echo "##[endgroup]"

    cd /auto-round/test/integration || exit 1

    for test_file in ${selected_files}; do
        echo "##[group]Running ${test_file}..."
        local test_basename=$(basename ${test_file} .py)
        local ut_log_name=${LOG_DIR}/unittest_${test_basename}.log

        numactl --physcpubind="${NUMA_CPUSET:-0-15}" --membind="${NUMA_NODE:-0}" \
            pytest --cov=auto_round --cov-report= --cov-append \
                -vs --junitxml="${ut_log_name%.log}.xml" ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done
}

function collect_log() {
    touch "${SUMMARY_LOG}"
    python /auto-round/.azure-pipelines/scripts/ut/collect_result.py \
        --test-type "Unit Tests" --log-pattern "unittest_test_*.log" --log-dir ${LOG_DIR} --summary-log ${SUMMARY_LOG}

    if [ -f .coverage ]; then
        cp .coverage "${LOG_DIR}/.coverage.part${test_part}"
    fi
}

function main() {
    setup_environment
    init_changed_tests
    scope_changed_tests "$(cd /auto-round && find test/unit/common test/unit/test_cpu test/integration/test_cpu -name "test_*.py" 2>/dev/null)"
    if [ "$test_part" = "inc" ]; then
        run_inc_unit_test
    elif [ "$test_part" = "llmc" ]; then
        run_llmc_unit_test
    elif [ "$test_part" = "0" ]; then
        run_common_unit_test
    else
        run_unit_test
    fi
    collect_log
    check_storage_usage
    print_summary
}

main "$@"
