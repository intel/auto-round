#!/bin/bash
set -xe

CONDA_ENV_NAME="unittest_cuda"
PYTHON_VERSION="3.10"
REPO_PATH=$(git rev-parse --show-toplevel)
LOG_DIR=${REPO_PATH}/ut_log_dir
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
    pip uninstall auto-round -y
    uv pip install -r requirements.txt
    sed -i '/^torch==/d;/^transformers==/d;/^lm-eval==/d' requirements.txt
    if [ -d "/proc/driver/nvidia" ]; then
        export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
        export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0])")/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
    fi
    uv pip install -v --no-build-isolation .
    uv pip install pytest-cov pytest-html cmake==4.0.2
}

function run_unit_test() {
    # install unit test dependencies
    create_conda_env

    cd ${REPO_PATH}/test/test_cuda
    rm -rf .coverage* *.xml *.html

    uv pip install -v git+https://github.com/casper-hansen/AutoAWQ.git --no-build-isolation
    uv pip install -v git+https://github.com/ModelCloud/GPTQModel.git@v2.2.0 --no-build-isolation
    uv pip install -r https://raw.githubusercontent.com/ModelCloud/GPTQModel/refs/heads/main/requirements.txt
    CMAKE_ARGS="-DGGML_CUDA=on -DLLAVA_BUILD=off" uv pip install llama-cpp-python
    uv pip install 'git+https://github.com/ggml-org/llama.cpp.git#subdirectory=gguf-py'
    uv pip install -r requirements.txt

    uv pip list
    export COVERAGE_RCFILE=${REPO_PATH}/.azure-pipelines/scripts/ut/.coverage
    local auto_round_path=$(python -c 'import auto_round; print(auto_round.__path__[0])')

    # setup test env
    mkdir -p ${LOG_DIR}
    local ut_log_name=${LOG_DIR}/unittest_cuda.log
    find . -name "test_*.py" | sed "s,\.\/,python -m pytest --cov=\"${auto_round_path}\" --cov-report term --html=report.html --self-contained-html  --cov-report xml:coverage.xml --cov-append -vs --disable-warnings ,g" >run.sh
    cat run.sh

    # run unit test
    bash run.sh 2>&1 | tee ${ut_log_name}

    cp report.html ${LOG_DIR}/
    cp coverage.xml ${LOG_DIR}/

    if [ $(grep -c '== FAILURES ==' ${ut_log_name}) != 0 ] || [ $(grep -c '== ERRORS ==' ${ut_log_name}) != 0 ] || [ $(grep -c ' passed' ${ut_log_name}) == 0 ]; then
        echo "Find errors in pytest case, please check the output..."
    fi
}

function run_unit_test_vlm() {
    # install unit test dependencies
    create_conda_env
    cd ${REPO_PATH}/test/test_cuda
    rm -rf .coverage* *.xml *.html

    uv pip install git+https://github.com/haotian-liu/LLaVA.git@v1.2.2 --no-deps
    local site_path=$(python -c "import site; print(site.getsitepackages()[0])")
    # reference https://github.com/haotian-liu/LLaVA/issues/1448#issuecomment-2119845242
    sed -i '/inputs\[.*image_sizes.*\] = image_sizes/a\        inputs.pop("cache_position")' ${site_path}/llava/model/language_model/llava_llama.py
    uv pip install git+https://github.com/deepseek-ai/DeepSeek-VL2.git timm attrdict --no-deps
    uv pip install -v git+https://github.com/casper-hansen/AutoAWQ.git@v0.2.0 --no-build-isolation
    uv pip install flash-attn==2.7.4.post1 --no-build-isolation
    uv pip install -r requirements_vlm.txt

    uv pip list
    export COVERAGE_RCFILE=${REPO_PATH}/.azure-pipelines/scripts/ut/.coverage
    local auto_round_path=$(python -c 'import auto_round; print(auto_round.__path__[0])')

    # setup test env
    mkdir -p ${LOG_DIR}
    local ut_log_name=${LOG_DIR}/unittest_cuda_vlm.log
    find . -name "test*vlms.py" | sed "s,\.\/,python -m pytest --cov=\"${auto_round_path}\" --cov-report term --html=report_vlms.html --self-contained-html  --cov-report xml:coverage_vlms.xml --cov-append -vs --disable-warnings ,g" >run_vlms.sh
    cat run_vlms.sh

    # run unit test
    bash run_vlms.sh 2>&1 | tee ${ut_log_name}

    cp report_vlms.html ${LOG_DIR}/
    cp coverage_vlms.xml ${LOG_DIR}/

    if [ $(grep -c '== FAILURES ==' ${ut_log_name}) != 0 ] || [ $(grep -c '== ERRORS ==' ${ut_log_name}) != 0 ] || [ $(grep -c ' passed' ${ut_log_name}) == 0 ]; then
        echo "Find errors in pytest case, please check the output..."
    fi
}

function main() {
    run_unit_test_vlm
    run_unit_test
}

main
