#!/bin/bash
set -xe

CONDA_ENV_NAME="unittest_cuda"
PYTHON_VERSION="3.10"
REPO_PATH=$(git rev-parse --show-toplevel)
LOG_DIR=${REPO_PATH}/ut_log_dir
SUMMARY_LOG=${LOG_DIR}/results_summary.log

rm -rf ${LOG_DIR} && mkdir -p ${LOG_DIR}
touch ${SUMMARY_LOG}
[[ -z "$CUDA_VISIBLE_DEVICES" ]] && export CUDA_VISIBLE_DEVICES=0

function create_conda_env() {
    echo "-----[VAL INFO] create conda env -----"
    [[ -d ${HOME}/anaconda3/bin ]] && export PATH=${HOME}/anaconda3/bin/:$PATH
    [[ -d ${HOME}/miniforge3/bin ]] && export PATH=${HOME}/miniforge3/bin/:$PATH
    [[ -d ${HOME}/miniconda3/bin ]] && export PATH=${HOME}/miniconda3/bin/:$PATH

    # create conda env
    source activate base
    if conda info --envs | grep -q "^$CONDA_ENV_NAME\s"; then conda remove -n ${CONDA_ENV_NAME} --all -y; fi
    conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} setuptools -y
    source activate ${CONDA_ENV_NAME}
    conda install -c conda-forge git gxx=11.2.0 gcc=11.2.0 gdb sysroot_linux-64 libgcc uv -y
    export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so.6

    # install AutoRound
    cd ${REPO_PATH}
    uv pip install torch==2.9.1 torchvision
    uv pip install -r requirements.txt
    if [ -d "/proc/driver/nvidia" ]; then
        export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
        export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
    fi
    uv pip install --no-build-isolation .
    uv pip install pytest-cov pytest-html cmake
}

function print_test_results_table() {
    local log_pattern=$1
    local test_type=$2

    echo ""
    { printf '=%.0s' {1..120}; echo; } >> ${SUMMARY_LOG}
    echo "Test Results Summary - ${test_type}" >> ${SUMMARY_LOG}
    { printf '=%.0s' {1..120}; echo; } >> ${SUMMARY_LOG}
    printf "%-30s %-10s %-50s\n" "Test Case" "Result" "Log File" >> ${SUMMARY_LOG}
    printf "%-30s %-10s %-50s\n" "----------" "------" "--------" >> ${SUMMARY_LOG}

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
            printf "%-30s %-10s %-50s\n" "${test_name}" "${result}" "${log_filename}" >> ${SUMMARY_LOG}
        fi
    done

    { printf '=%.0s' {1..120}; echo; } >> ${SUMMARY_LOG}
    printf "Total: %d, Passed: %d, Failed: %d\n" ${total_tests} ${passed_tests} ${failed_tests} >> ${SUMMARY_LOG}
    { printf '=%.0s' {1..120}; echo; } >> ${SUMMARY_LOG}
    echo "" >> ${SUMMARY_LOG}
}

function run_unit_test() {
    # install unit test dependencies
    create_conda_env

    cd ${REPO_PATH}/test
    rm -rf .coverage* *.xml *.html

    uv pip install -v git+https://github.com/casper-hansen/AutoAWQ.git --no-build-isolation
    uv pip install https://github.com/ModelCloud/GPTQModel/releases/download/v5.6.0/gptqmodel-5.6.0+cu126torch2.9-cp310-cp310-linux_x86_64.whl --no-build-isolation
    uv pip install -r https://raw.githubusercontent.com/ModelCloud/GPTQModel/refs/heads/main/requirements.txt
    CMAKE_ARGS="-DGGML_CUDA=on -DLLAVA_BUILD=off" uv pip install llama-cpp-python
    uv pip install 'git+https://github.com/ggml-org/llama.cpp.git#subdirectory=gguf-py'
    uv pip install -r test_cuda/requirements.txt
    uv pip install -r test_cuda/requirements_diffusion.txt
    uv pip install -r test_cuda/requirements_sglang.txt
    uv pip install transformers==4.57.6

    pip list > ${LOG_DIR}/ut_pip_list.txt
    export COVERAGE_RCFILE=${REPO_PATH}/.azure-pipelines/scripts/ut/.coverage
    local auto_round_path=$(python -c 'import auto_round; print(auto_round.__path__[0])')

    # run unit tests individually with separate logs
    for test_file in $(find ./test_cuda -name "test_*.py" ! -name "test_*vlms.py" ! -name "test_llmc*.py" | sort); do
        local test_basename=$(basename ${test_file} .py)
        local ut_log_name=${LOG_DIR}/unittest_cuda_${test_basename}.log
        echo "Running ${test_file}..."

        python -m pytest --cov="${auto_round_path}" --cov-report term --html=report.html --self-contained-html --cov-report xml:coverage.xml --cov-append -vs --disable-warnings ${test_file} 2>&1 | tee ${ut_log_name}
    done

    mv report.html ${LOG_DIR}/
    mv coverage.xml ${LOG_DIR}/

    # Print test results table and check for failures
    if ! print_test_results_table "unittest_cuda_test_*.log" "CUDA Unit Tests"; then
        echo "Some CUDA unit tests failed. Please check the individual log files for details."
    fi
}

function run_unit_test_vlm() {
    # install unit test dependencies
    create_conda_env
    cd ${REPO_PATH}/test
    rm -rf .coverage* *.xml *.html

    uv pip install git+https://github.com/haotian-liu/LLaVA.git@v1.2.2 --no-deps
    local site_path=$(python -c "import site; print(site.getsitepackages()[0])")
    # reference https://github.com/haotian-liu/LLaVA/issues/1448#issuecomment-2119845242
    sed -i '/inputs\[.*image_sizes.*\] = image_sizes/a\        inputs.pop("cache_position")' ${site_path}/llava/model/language_model/llava_llama.py
    uv pip install git+https://github.com/deepseek-ai/DeepSeek-VL2.git timm attrdict --no-deps
    uv pip install -v git+https://github.com/casper-hansen/AutoAWQ.git@v0.2.0 --no-build-isolation
    uv pip install flash-attn==2.7.4.post1 --no-build-isolation
    uv pip install -r test_cuda/requirements_vlm.txt

    pip list > ${LOG_DIR}/vlm_ut_pip_list.txt
    export COVERAGE_RCFILE=${REPO_PATH}/.azure-pipelines/scripts/ut/.coverage
    local auto_round_path=$(python -c 'import auto_round; print(auto_round.__path__[0])')

    # run VLM unit tests individually with separate logs
    for test_file in $(find ./test_cuda -name "test*vlms.py"); do
        local test_basename=$(basename ${test_file} .py)
        local ut_log_name=${LOG_DIR}/unittest_cuda_vlm_${test_basename}.log
        echo "Running ${test_file}..."

        python -m pytest --cov="${auto_round_path}" --cov-report term --html=report_vlms.html --self-contained-html --cov-report xml:coverage_vlms.xml --cov-append -vs --disable-warnings ${test_file} 2>&1 | tee ${ut_log_name}
    done

    mv report_vlms.html ${LOG_DIR}/
    mv coverage_vlms.xml ${LOG_DIR}/

    # Print test results table and check for failures
    if ! print_test_results_table "unittest_cuda_vlm_test*.log" "CUDA VLM Tests"; then
        echo "Some CUDA VLM tests failed. Please check the individual log files for details."
    fi
}

function run_unit_test_llmc() {
    # install unit test dependencies
    create_conda_env

    cd ${REPO_PATH}/test
    rm -rf .coverage* *.xml *.html

    uv pip install -r test_cuda/requirements_llmc.txt

    pip list > ${LOG_DIR}/llmc_ut_pip_list.txt
    export COVERAGE_RCFILE=${REPO_PATH}/.azure-pipelines/scripts/ut/.coverage
    local auto_round_path=$(python -c 'import auto_round; print(auto_round.__path__[0])')

    # run unit tests individually with separate logs
    for test_file in $(find ./test_cuda -name "test_llmc*.py" | sort); do
        local test_basename=$(basename ${test_file} .py)
        local ut_log_name=${LOG_DIR}/unittest_cuda_llmc_${test_basename}.log
        echo "Running ${test_file}..."

        python -m pytest --cov="${auto_round_path}" --cov-report term --html=report_llmc.html --self-contained-html --cov-report xml:coverage_llmc.xml --cov-append -vs --disable-warnings ${test_file} 2>&1 | tee ${ut_log_name}
    done

    mv report_llmc.html ${LOG_DIR}/
    mv coverage_llmc.xml ${LOG_DIR}/
    # Print test results table and check for failures
    if ! print_test_results_table "unittest_cuda_llmc_test_*.log" "CUDA LLMC Tests"; then
        echo "Some CUDA LLMC tests failed. Please check the individual log files for details."
    fi
}

function main() {
    run_unit_test_vlm
    run_unit_test_llmc
    run_unit_test
    cat ${SUMMARY_LOG}
}

main
