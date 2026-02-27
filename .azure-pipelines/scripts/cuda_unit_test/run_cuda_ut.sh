#!/bin/bash
set -xe

PATTERN='[-a-zA-Z0-9_]*='

for i in "$@"; do
    case $i in
    --test_case=*)
        test_case=$(echo $i | sed "s/${PATTERN}//")
        ;;
    --test_part=*)
        test_part=$(echo $i | sed "s/${PATTERN}//")
        ;;
    *)
        echo "Parameter $i not recognized."
        exit 1
        ;;
    esac
done

LOG_DIR="${BUILD_SOURCESDIRECTORY}/ut_log_dir"
mkdir -p "${LOG_DIR}"
SUMMARY_LOG="${LOG_DIR}/results_summary.log"

function setup_environment() {
    export TZ='Asia/Shanghai'
    export TQDM_MININTERVAL=120
    export CUDA_VISIBLE_DEVICES=0
    export HF_HUB_DISABLE_PROGRESS_BARS=1
}

function print_test_results_table() {
    echo "##[group]Collect results..."
    local log_pattern=$1
    local test_type=$2

    echo ""
    { printf '=%.0s' {1..120}; echo; } >> "${SUMMARY_LOG}"
    echo "Test Results Summary - ${test_type}" >> "${SUMMARY_LOG}"
    { printf '=%.0s' {1..120}; echo; } >> "${SUMMARY_LOG}"
    printf "%-30s %-10s %-50s\n" "Test Case" "Result" "Log File" >> "${SUMMARY_LOG}"
    printf "%-30s %-10s %-50s\n" "----------" "------" "--------" >> "${SUMMARY_LOG}"
    local total_tests=0
    local passed_tests=0
    local failed_tests=0

    for log_file in ${LOG_DIR}/${log_pattern}; do
        if [ -f "${log_file}" ]; then
            local test_name=$(basename "${log_file}" .log)
            # Remove prefix to get clean test case name
            test_name=${test_name#unittest_cuda_}
            test_name=${test_name#unittest_cuda_vlm_}

            local result="UNKNOWN"
            local failure_count=$(grep -c '== FAILURES ==' "${log_file}" 2>/dev/null || echo 0)
            local error_count=$(grep -c '== ERRORS ==' "${log_file}" 2>/dev/null || echo 0)
            local killed_count=$(grep -c 'Killed' "${log_file}" 2>/dev/null || echo 0)
            local passed_count=$(grep -c ' passed' "${log_file}" 2>/dev/null || echo 0)

            if [ ${failure_count} -gt 0 ] || [ ${error_count} -gt 0 ] || [ ${killed_count} -gt 0 ]; then
                result="FAILED"
                failed_tests=$((failed_tests + 1))
            elif [ ${passed_count} -gt 0 ]; then
                result="PASSED"
                passed_tests=$((passed_tests + 1))
            else
                result="NO_TESTS"
            fi

            total_tests=$((total_tests + 1))
            local log_filename=$(basename "${log_file}")
            printf "%-35s %-10s %-50s\n" "${test_name}" "${result}" "${log_filename}" >> "${SUMMARY_LOG}"
        fi
    done

    { printf '=%.0s' {1..120}; echo; } >> "${SUMMARY_LOG}"
    printf "Total: %d, Passed: %d, Failed: %d\n" ${total_tests} ${passed_tests} ${failed_tests} >> "${SUMMARY_LOG}"
    { printf '=%.0s' {1..120}; echo; } >> "${SUMMARY_LOG}"
    echo "" >> "${SUMMARY_LOG}"
    echo "##[endgroup]"
}

function run_unit_test() {
    # install unit test dependencies
    echo "##[group]set up UT env..."
    cd "${BUILD_SOURCESDIRECTORY}" || exit 1
    uv pip install pytest-cov pytest-html
    uv pip install torch==2.10.0 torchvision
    uv pip install git+https://github.com/casper-hansen/AutoAWQ.git --no-build-isolation
    # uv pip install gptqmodel --no-build-isolation
    # uv pip install -r https://raw.githubusercontent.com/ModelCloud/GPTQModel/refs/heads/main/requirements.txt
    CMAKE_ARGS="-DGGML_CUDA=on -DLLAVA_BUILD=off" uv pip install llama-cpp-python
    uv pip install 'git+https://github.com/ggml-org/llama.cpp.git#subdirectory=gguf-py'
    uv pip install -r test/test_cuda/requirements.txt
    uv pip install -r test/test_cuda/requirements_diffusion.txt
    uv pip install torch==2.10.0 torchvision
    uv pip install -U transformers
    uv pip install .
    echo "##[endgroup]"

    uv pip list
    export COVERAGE_RCFILE="${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/ut/.coverage"
    local auto_round_path=$(python -c 'import auto_round; print(auto_round.__path__[0])')

    cd "${BUILD_SOURCESDIRECTORY}/test" || exit 1

    find ./test_cuda -name "test_*.py" ! -name "test_*vlms.py" ! -name "test_llmc*.py" ! -name "test_*sglang*.py" | sort > all_tests.txt
    total_lines=$(wc -l < all_tests.txt)
    NUM_CHUNKS=3
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
        local ut_log_name=${LOG_DIR}/unittest_cuda_${test_basename}.log

        pytest -m "not skip_ci" --cov="${auto_round_path}" --cov-report term --html=report.html --self-contained-html --cov-report xml:coverage.xml --cov-append -vs --disable-warnings ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done

    mv report.html ${LOG_DIR}/
    mv coverage.xml ${LOG_DIR}/

    # Print test results table and check for failures
    if ! print_test_results_table "unittest_cuda_test_*.log" "CUDA Unit Tests"; then
        echo "Some CUDA unit tests failed. Please check the individual log files for details."
    fi
}

function run_unit_test_llmc() {
    echo "##[group]set up UT env..."
    cd "${BUILD_SOURCESDIRECTORY}" || exit 1
    uv pip install pytest-cov pytest-html
    uv pip install -r test/test_cuda/requirements_llmc.txt
    uv pip install .
    echo "##[endgroup]"
    uv pip list
    cd "${BUILD_SOURCESDIRECTORY}/test" || exit 1

    export COVERAGE_RCFILE="${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/ut/.coverage"
    local auto_round_path=$(python -c 'import auto_round; print(auto_round.__path__[0])')

    for test_file in $(find ./test_cuda -name "test_llmc*.py" | sort); do
        echo "##[group]Running ${test_file}..."
        local test_basename=$(basename ${test_file} .py)
        local ut_log_name=${LOG_DIR}/unittest_cuda_llmc_${test_basename}.log
        pytest -m "not skip_ci" --cov="${auto_round_path}" --cov-report term --html=report_llmc.html --self-contained-html --cov-report xml:coverage_llmc.xml --cov-append -vs --disable-warnings ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done

    mv report_llmc.html ${LOG_DIR}/
    mv coverage_llmc.xml ${LOG_DIR}/
    # Print test results table and check for failures
    if ! print_test_results_table "unittest_cuda_llmc_test_*.log" "CUDA LLMC Tests"; then
        echo "Some CUDA LLMC tests failed. Please check the individual log files for details."
    fi
}

function run_unit_test_sglang() {
    echo "##[group]set up UT env..."
    cd "${BUILD_SOURCESDIRECTORY}" || exit 1
    uv pip install pytest-cov pytest-html
    uv pip install -r test/test_cuda/requirements_sglang.txt
    uv pip install .
    echo "##[endgroup]"

    uv pip list
    cd "${BUILD_SOURCESDIRECTORY}/test" || exit 1
    export COVERAGE_RCFILE="${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/ut/.coverage"
    local auto_round_path=$(python -c 'import auto_round; print(auto_round.__path__[0])')

    for test_file in $(find ./test_cuda -name "test_sglang*.py" | sort); do
        echo "##[group]Running ${test_file}..."
        local test_basename=$(basename ${test_file} .py)
        local ut_log_name=${LOG_DIR}/unittest_cuda_sglang_${test_basename}.log
        pytest -m "not skip_ci" --cov="${auto_round_path}" --cov-report term --html=report_sglang.html --self-contained-html --cov-report xml:coverage_sglang.xml --cov-append -vs --disable-warnings ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done

    mv report_sglang.html ${LOG_DIR}/
    mv coverage_sglang.xml ${LOG_DIR}/
    if ! print_test_results_table "unittest_cuda_sglang_test*.log" "CUDA SGLang Unit Tests"; then
        echo "Some CUDA SGLang unit tests failed. Please check the individual log files for details."
    fi
}


function main() {
    setup_environment
    if [ "${test_case}" == "vlm" ]; then
        run_unit_test_vlm
    elif [ "${test_case}" == "llmc" ]; then
        run_unit_test_llmc
    elif [ "${test_case}" == "sglang" ]; then
        run_unit_test_sglang
    elif [ "${test_case}" == "all" ]; then
        run_unit_test
    else
        echo "Invalid test case specified: ${test_case}. Please use 'vlm', 'llmc', 'sglang', or 'all'."
        exit 1
    fi
    df -h
    du -sh "${BUILD_SOURCESDIRECTORY}"
    du -sh /root/.cache/huggingface
    du -sh /root/.cache/huggingface/hub/*
    du -sh /root/.venv
    cat "${SUMMARY_LOG}"
}

main
