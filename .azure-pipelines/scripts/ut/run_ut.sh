#!/bin/bash
set -xe

test_part=$1

# install requirements
echo "##[group]set up UT env..."
export TQDM_MININTERVAL=60
uv pip install pytest-cov pytest-html
uv pip install -r /auto-round/test/test_cpu/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# install latest gguf for ut test
cd ~ || exit 1
git clone -b master --quiet --single-branch https://github.com/ggml-org/llama.cpp.git && cd llama.cpp/gguf-py && uv pip install . sentencepiece

cd /auto-round && uv pip install .

echo "##[endgroup]"
uv pip list

cd /auto-round/test || exit 1

export LD_LIBRARY_PATH=${HOME}/.venv/lib/:$LD_LIBRARY_PATH
export FORCE_BF16=1
export COVERAGE_RCFILE=/auto-round/.azure-pipelines/scripts/ut/.coverage
auto_round_path=$(python -c 'import auto_round; print(auto_round.__path__[0])')

LOG_DIR=/auto-round/log_dir
mkdir -p ${LOG_DIR}
ut_log_name=${LOG_DIR}/ut.log

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
printf '%s\n' "${selected_files}" | sed "s,\.\/,python -m pytest --cov=\"${auto_round_path}\" --cov-report term --html=report.html --self-contained-html --cov-report xml:coverage.xml --cov-append -vs --disable-warnings ,g" > run.sh
cat run.sh
bash run.sh 2>&1 | tee "${ut_log_name}"

if [ $(grep -c '== FAILURES ==' ${ut_log_name}) != 0 ] || [ $(grep -c '== ERRORS ==' ${ut_log_name}) != 0 ] || [ $(grep -c ' passed' ${ut_log_name}) == 0 ]; then
    echo "##[error]Find errors in pytest case, please check the output..."
    exit 1
fi

# if ut pass, collect the coverage file into artifacts
cp .coverage "${LOG_DIR}/.coverage.part${test_part}"

echo "UT finished successfully! "
