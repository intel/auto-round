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

LOG_DIR="${BUILD_SOURCESDIRECTORY}/ut_log_dir"
mkdir -p "${LOG_DIR}"
SUMMARY_LOG="${LOG_DIR}/results_summary.log"

function setup_environment() {
    export TZ='Asia/Shanghai'
    export TQDM_MININTERVAL=120
    export CUDA_VISIBLE_DEVICES=0
    export HF_HUB_DISABLE_PROGRESS_BARS=1
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
    du -sh "${BUILD_SOURCESDIRECTORY}"
    du -sh /root/.cache/huggingface
    du -sh /root/.cache/huggingface/hub/*
    du -sh /root/.venv
    echo "##[endgroup]"
}

function run_unit_test() {
    # install unit test dependencies
    echo "##[group]set up UT env..."
    cd "${BUILD_SOURCESDIRECTORY}" || exit 1
    uv pip install torch==2.10.0 torchvision
    uv pip install git+https://github.com/casper-hansen/AutoAWQ.git --no-build-isolation

    # install gptqmodel
    CUDA_VER=$(python -c 'import torch; print(f"cu{torch.version.cuda.replace(".", "")}")')
    PY_VER=$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
    TORCH_VER="torch2.10"
    WHEEL="gptqmodel-5.7.0-${CUDA_VER}${TORCH_VER}-${PY_VER}-${PY_VER}-linux_x86_64.whl"
    URL="https://pkgs.dev.azure.com/lpot-inc/b7121868-d73a-4794-90c1-23135f974d09/_packaging/4728fbab-e069-4cbd-bcca-d35f4d42256b/pypi/download/gptqmodel/5.7/${WHEEL}"
    wget -q "$URL" -O "$WHEEL" || { echo "Download failed. Check CUDA/PyTorch/Python versions match (cu126/cu128/cu130, torch2.10, cp310-cp313)"; exit 1; }
    mv "$WHEEL" "${WHEEL/-${CUDA_VER}${TORCH_VER}-/+${CUDA_VER}.${TORCH_VER}-}"
    uv pip install "./${WHEEL/-${CUDA_VER}${TORCH_VER}-/+${CUDA_VER}.${TORCH_VER}-}" --no-build-isolation
    rm -f "./${WHEEL/-${CUDA_VER}${TORCH_VER}-/+${CUDA_VER}.${TORCH_VER}-}"

    uv pip install gptqmodel --extra-index-url https://pkgs.dev.azure.com/lpot-inc/neural-compressor/_packaging/gptqmodel-wheels/pypi/simple/
    uv pip install -r https://raw.githubusercontent.com/ModelCloud/GPTQModel/refs/tags/v5.7.0/requirements.txt
    uv pip install https://github.com/XuehaoSun/llama-cpp-python/releases/download/v0.3.16/llama_cpp_python-0.3.16-cp312-cp312-linux_x86_64.whl
    uv pip install 'git+https://github.com/ggml-org/llama.cpp.git#subdirectory=gguf-py'
    uv pip install -r test/test_cuda/requirements.txt
    uv pip install -r test/test_cuda/requirements_diffusion.txt
    uv pip install -U transformers
    uv pip install .
    echo "##[endgroup]"

    uv pip list
    export COVERAGE_RCFILE="${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/ut/.coverage"

    cd "${BUILD_SOURCESDIRECTORY}/test" || exit 1

    find ./test_cuda -type f -name "test_*.py" | grep -Ev "vlms|llmc|sglang|vllm|multiple_card" | sort > all_tests.txt
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

    for test_file in ${selected_files}; do
        echo "##[group]Running ${test_file}..."
        local test_basename=$(basename ${test_file} .py)
        local ut_log_name=${LOG_DIR}/unittest_cuda_${test_basename}.log

        pytest -m "not skip_ci" -vs --disable-warnings ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done

    python ${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/ut/collect_result.py --test-type "CUDA Unit Tests" --log-pattern "unittest_cuda_test_*.log" --log-dir ${LOG_DIR} --summary-log ${SUMMARY_LOG}
}

function run_unit_test_llmc() {
    echo "##[group]set up UT env..."
    cd "${BUILD_SOURCESDIRECTORY}" || exit 1
    rm -rf /root/.venv
    uv venv --python=3.12 /root/.venv
    uv pip install -U pytest-cov pytest-html
    uv pip install -r test/test_cuda/requirements_llmc.txt
    uv pip install .
    echo "##[endgroup]"
    uv pip list
    cd "${BUILD_SOURCESDIRECTORY}/test" || exit 1

    export COVERAGE_RCFILE="${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/ut/.coverage"

    for test_file in $(find ./test_cuda -name "test_llmc*.py" | sort); do
        echo "##[group]Running ${test_file}..."
        local test_basename=$(basename ${test_file} .py)
        local ut_log_name=${LOG_DIR}/unittest_cuda_llmc_${test_basename}.log
        pytest -m "not skip_ci" -vs --disable-warnings ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done

    python ${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/ut/collect_result.py --test-type "CUDA LLMC Tests" --log-pattern "unittest_cuda_llmc_test_*.log" --log-dir ${LOG_DIR} --summary-log ${SUMMARY_LOG}
}

function run_unit_test_sglang() {
    echo "##[group]set up UT env..."
    cd "${BUILD_SOURCESDIRECTORY}" || exit 1
    rm -rf /root/.venv
    uv venv --python=3.12 /root/.venv
    uv pip install -U pytest-cov pytest-html
    uv pip install -r test/test_cuda/requirements_sglang.txt
    uv pip install .
    echo "##[endgroup]"

    uv pip list
    cd "${BUILD_SOURCESDIRECTORY}/test" || exit 1
    export COVERAGE_RCFILE="${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/ut/.coverage"

    for test_file in $(find ./test_cuda -name "test_sglang*.py" | sort); do
        echo "##[group]Running ${test_file}..."
        local test_basename=$(basename ${test_file} .py)
        local ut_log_name=${LOG_DIR}/unittest_cuda_sglang_${test_basename}.log
        pytest -m "not skip_ci" -vs --disable-warnings ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done

    python ${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/ut/collect_result.py --test-type "CUDA SGLang Tests" --log-pattern "unittest_cuda_sglang_test_*.log" --log-dir ${LOG_DIR} --summary-log ${SUMMARY_LOG}
}

function run_unit_test_vllm() {
    echo "##[group]set up UT env..."
    cd "${BUILD_SOURCESDIRECTORY}" || exit 1
    rm -rf /root/.venv
    uv venv --python=3.12 /root/.venv
    uv pip install -U pytest-cov pytest-html
    uv pip install -r test/test_cuda/requirements_vllm.txt
    uv pip install .
    echo "##[endgroup]"

    uv pip list
    cd "${BUILD_SOURCESDIRECTORY}/test" || exit 1
    export COVERAGE_RCFILE="${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/ut/.coverage"

    for test_file in $(find ./test_cuda -name "test_vllm*.py" | sort); do
        echo "##[group]Running ${test_file}..."
        local test_basename=$(basename ${test_file} .py)
        local ut_log_name=${LOG_DIR}/unittest_cuda_vllm_${test_basename}.log
        pytest -m "not skip_ci" -vs --disable-warnings ${test_file} 2>&1 | tee ${ut_log_name}
        echo "##[endgroup]"
    done

    python ${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/ut/collect_result.py --test-type "CUDA VLLM Tests" --log-pattern "unittest_cuda_vllm_test_*.log" --log-dir ${LOG_DIR} --summary-log ${SUMMARY_LOG}
}

function main() {
    setup_environment
    if [ "${test_case}" == "vlm" ]; then
        run_unit_test_vlm
    elif [ "${test_case}" == "llmc" ]; then
        run_unit_test_llmc
    elif [ "${test_case}" == "sglang" ]; then
        run_unit_test_sglang
    elif [ "${test_case}" == "vllm" ]; then
        run_unit_test_vllm
    elif [ "${test_case}" == "all" ]; then
        run_unit_test
    else
        echo "##[error]Invalid test case specified: ${test_case}. Please use 'vlm', 'llmc', 'sglang', 'vllm', or 'all'."
        exit 1
    fi
    check_storage_usage
    print_summary
}

main
