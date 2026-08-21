#!/bin/bash
set -e

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

source ${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/change_color.sh
# Change-based test selection helpers. REPO_DIR points the detector at the
# agent checkout instead of the container default (/auto-round).
REPO_DIR="${BUILD_SOURCESDIRECTORY}"
source ${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/ut/detect_changed_tests.sh

LOG_DIR="${BUILD_SOURCESDIRECTORY}/log_dir"
mkdir -p "${LOG_DIR}"
SUMMARY_LOG="${LOG_DIR}/results_summary.log"
# print_summary reads this file unconditionally; a matrix part that selects no
# test never writes it, so make sure it always exists.
touch "${SUMMARY_LOG}"

function setup_environment() {
    export TZ='Asia/Shanghai'
    export TQDM_MININTERVAL=120
    export CUDA_VISIBLE_DEVICES=0
    export HF_HUB_DISABLE_PROGRESS_BARS=1
}

function print_summary() {
    python ${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/ut/print_summary.py --summary-log "${SUMMARY_LOG}"
    exit $?
}

function check_storage_usage() {
    echo "##[group]check storage usage..."
    df -h
    du -sh "${BUILD_SOURCESDIRECTORY}" || true
    du -sh /root/.cache/huggingface || true
    du -sh /root/.cache/huggingface/hub/* || true
    du -sh /root/.venv || true
    echo "##[endgroup]"
}

function setup_basic_test_env() {
    echo "##[group]Setting up test environment..."

    cd "${BUILD_SOURCESDIRECTORY}" || exit 1
    uv pip install torch==2.13.0 torchvision torchao --index-url https://download.pytorch.org/whl/cu130
    uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu130
    uv pip install 'git+https://github.com/ggml-org/llama.cpp.git#subdirectory=gguf-py'
    uv pip install -r test/unit/test_cuda/requirements.txt
    uv pip install -r test/unit/test_cuda/requirements_diffusion.txt
    uv pip install -U transformers chardet
    uv pip install -U pytest-cov pytest-timeout
    uv pip install kernels==0.15.2 # For sm120: https://github.com/huggingface/transformers/blob/v5.13.1/setup.py#L93
    uv pip uninstall torch torchvision
    uv pip install torch==2.13.0 torchvision torchao --index-url https://download.pytorch.org/whl/cu130
    uv pip install .
    echo "##[endgroup]"

    uv pip list
    export COVERAGE_RCFILE="${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/ut/.coveragerc"

    cd "${BUILD_SOURCESDIRECTORY}/test" || exit 1
}

function run_common_group() {
    # Run a group of common test files together in a single pytest invocation.
    # $1: group name (used for log file), remaining args: test files
    local group_name=$1
    shift
    local group_tests
    group_tests=$(filter_changed_tests "test" "$*")

    if [ -n "${group_tests}" ]; then
        echo "##[group]Running common tests (${group_name})..."
        local ut_log_name="${LOG_DIR}/unittest_common_${group_name}.log"
        pytest -m "not skip_ci" \
            --cov=auto_round --cov-report= --cov-append --timeout=60 --session-timeout=720 \
            -vs --junitxml="${ut_log_name%.log}.xml" \
            ${group_tests} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    fi
}

function run_common_unit_test() {
    # common test case for cpu/gpu/xpu
    # Group cases by the first-level folder under unit/common; a single test
    # file placed directly under unit/common (e.g. test_main.py) runs on its own.
    for entry in $(find ./unit/common -mindepth 1 -maxdepth 1 | sort); do
        if [ -d "${entry}" ]; then
            local group_name=$(basename "${entry}")
            run_common_group "${group_name}" "$(find "${entry}" -name "test*.py" | sort)"
        elif [[ "$(basename "${entry}")" == test*.py ]]; then
            local group_name=$(basename "${entry}" .py)
            run_common_group "${group_name}" "${entry}"
        fi
    done

    python ${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/ut/collect_result.py --test-type "Common Unit Tests" --log-pattern "unittest_common_*.log" --log-dir ${LOG_DIR} --summary-log ${SUMMARY_LOG}
}


function run_unit_test() {
    # run ci cuda ut scope 
    find ./unit/test_cuda -type f -name "test_*.py" | grep -Ev "vlms|llmc|sglang|vllm|multiple_card" | sort > all_tests.txt
    total_lines=$(wc -l < all_tests.txt)
    NUM_CHUNKS=2
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
        echo "No changed CUDA unit test file in part ${test_part}, skip."
        return 0
    fi

    for test_file in ${selected_files}; do
        echo "##[group]Running ${test_file}..."
        local test_basename=$(basename ${test_file} .py)
        local ut_log_name=${LOG_DIR}/unittest_cuda_${test_basename}.log

        pytest -m "not skip_ci" \
            --cov=auto_round --cov-report= --cov-append --timeout=60 --session-timeout=720 \
            -vs --junitxml="${ut_log_name%.log}.xml" \
            ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done
    [ -f .coverage ] && cp .coverage "${LOG_DIR}/.coverage.part${test_part}"

    python ${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/ut/collect_result.py --test-type "CUDA Unit Tests" --log-pattern "unittest_cuda_test_*.log" --log-dir ${LOG_DIR} --summary-log ${SUMMARY_LOG}
}

function run_unit_test_llmc() {
    echo "##[group]set up UT env..."
    cd "${BUILD_SOURCESDIRECTORY}" || exit 1
    rm -rf /root/.venv
    uv venv --python=3.12 /root/.venv
    uv pip install -U pytest-cov pytest-timeout
    BUILD_TYPE="nightly" uv pip install \
        -r test/integration/test_cuda/requirements_llmc.txt \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --index-strategy unsafe-best-match
    uv pip install -U chardet
    uv pip install .
    uv pip list
    echo "##[endgroup]"

    cd "${BUILD_SOURCESDIRECTORY}/test" || exit 1

    export COVERAGE_RCFILE="${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/ut/.coveragerc"

    for test_file in $(find ./integration/test_cuda -name "test_llmc*.py" | sort); do
        echo "##[group]Running ${test_file}..."
        local test_basename=$(basename ${test_file} .py)
        local ut_log_name=${LOG_DIR}/unittest_cuda_llmc_${test_basename}.log
        pytest -m "not skip_ci" \
            --cov=auto_round --cov-report= --cov-append -vs \
            --junitxml="${ut_log_name%.log}.xml" \
            ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done
    [ -f .coverage ] && cp .coverage "${LOG_DIR}/.coverage.llmc"

    python ${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/ut/collect_result.py --test-type "CUDA LLMC Tests" --log-pattern "unittest_cuda_llmc_test_*.log" --log-dir ${LOG_DIR} --summary-log ${SUMMARY_LOG}
}

function run_unit_test_sglang() {
    echo "##[group]set up UT env..."
    cd "${BUILD_SOURCESDIRECTORY}" || exit 1
    rm -rf /root/.venv
    uv venv --python=3.12 /root/.venv
    uv pip install -U pytest-cov pytest-timeout
    uv pip install -r test/integration/test_cuda/requirements_sglang.txt \
        --prerelease=allow \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --index-strategy unsafe-best-match
    local flashinfer_version=$(uv pip show flashinfer-python 2>/dev/null | grep -i "^Version" | awk '{print $2}')
    uv pip install flashinfer-jit-cache==${flashinfer_version} --index-url https://flashinfer.ai/whl/cu130
    uv pip install .
    uv pip list
    echo "##[endgroup]"

    cd "${BUILD_SOURCESDIRECTORY}/test" || exit 1
    export COVERAGE_RCFILE="${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/ut/.coveragerc"

    for test_file in $(find ./integration/test_cuda ./e2e/test_cuda -name "test_sglang*.py" | sort); do
        echo "##[group]Running ${test_file}..."
        local test_basename=$(basename ${test_file} .py)
        local ut_log_name=${LOG_DIR}/unittest_cuda_sglang_${test_basename}.log
        pytest -m "not skip_ci" \
            --cov=auto_round --cov-report= --cov-append -vs \
            --junitxml="${ut_log_name%.log}.xml" \
             ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done
    [ -f .coverage ] && cp .coverage "${LOG_DIR}/.coverage.sglang"

    python ${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/ut/collect_result.py --test-type "CUDA SGLang Tests" --log-pattern "unittest_cuda_sglang_test_*.log" --log-dir ${LOG_DIR} --summary-log ${SUMMARY_LOG}
}

function run_unit_test_vllm() {
    echo "##[group]set up UT env..."
    cd "${BUILD_SOURCESDIRECTORY}" || exit 1
    rm -rf /root/.venv
    uv venv --python=3.12 /root/.venv
    uv pip install -U pytest-cov pytest-timeout
    uv pip install -r test/integration/test_cuda/requirements_vllm.txt \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --index-strategy unsafe-best-match
    local flashinfer_version=$(uv pip show flashinfer-python 2>/dev/null | grep -i "^Version" | awk '{print $2}')
    uv pip install flashinfer-jit-cache==${flashinfer_version} --index-url https://flashinfer.ai/whl/cu130
    uv pip install -U chardet
    uv pip install .
    uv pip list
    echo "##[endgroup]"

    cd "${BUILD_SOURCESDIRECTORY}/test" || exit 1
    export COVERAGE_RCFILE="${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/ut/.coveragerc"

    for test_file in $(find ./integration/test_cuda ./e2e/test_cuda -name "test_vllm*.py" | sort); do
        echo "##[group]Running ${test_file}..."
        local test_basename=$(basename ${test_file} .py)
        local ut_log_name=${LOG_DIR}/unittest_cuda_vllm_${test_basename}.log
        pytest -m "not skip_ci" \
            --cov=auto_round --cov-report= --cov-append -vs \
            --junitxml="${ut_log_name%.log}.xml" \
            ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done
    [ -f .coverage ] && cp .coverage "${LOG_DIR}/.coverage.vllm"

    python ${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/ut/collect_result.py --test-type "CUDA VLLM Tests" --log-pattern "unittest_cuda_vllm_test_*.log" --log-dir ${LOG_DIR} --summary-log ${SUMMARY_LOG}
}

function main() {
    setup_environment
    init_changed_tests
    if [ "${test_case}" == "nightly" ]; then
        run_unit_test_sglang
        run_unit_test_llmc
        run_unit_test_vllm
    elif [ "${test_case}" == "ci" ]; then
        # Mirror the selection below: tests excluded from the ci run (vlms,
        # llmc, sglang, vllm, multiple_card) must not enable filtering.
        scope_changed_tests "$(cd "${BUILD_SOURCESDIRECTORY}" && find test/unit/common test/unit/test_cuda -type f -name "test_*.py" | grep -Ev "vlms|llmc|sglang|vllm|multiple_card")"
        setup_basic_test_env
        if [ "${test_part}" == "0" ]; then
            run_common_unit_test
        else
            run_unit_test
        fi
    else
        echo "##[error]Invalid test case specified: ${test_case}. Please use 'nightly' or 'ci'."
        exit 1
    fi
    check_storage_usage
    print_summary
}

main "$@"
