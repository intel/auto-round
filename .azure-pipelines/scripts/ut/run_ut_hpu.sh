#!/bin/bash
set -xe

# install requirements
echo "set up UT env..."
pip install pytest-cov pytest-html
pip list

cd /auto-round/test/test_cpu || exit 1
find . -type f -exec sed -i '/sys\.path\.insert(0, "\.\.")/d' {} +

export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export FORCE_BF16=1
export COVERAGE_RCFILE=/auto-round/.azure-pipelines/scripts/ut/.coverage
auto_round_path=$(python -c 'import auto_round; print(auto_round.__path__[0])')

LOG_DIR=/auto-round/log_dir
mkdir -p ${LOG_DIR}
ut_log_name=${LOG_DIR}/ut.log

find . -name "test*hpu_only.py" | sed "s,\.\/,python -m pytest --cov=\"${auto_round_path}\" --cov-report term --html=report.html --self-contained-html  --cov-report xml:coverage.xml --cov-append -vs --disable-warnings ,g" > run_lazy.sh
find . -name "test*hpu_only.py" | sed "s,\.\/,python -m pytest --mode compile --cov=\"${auto_round_path}\" --cov-report term --html=report.html --self-contained-html  --cov-report xml:coverage.xml --cov-append -vs --disable-warnings ,g" > run_compile.sh

cat run_lazy.sh
bash run_lazy.sh 2>&1 | tee ${ut_log_name}

cat run_compile.sh
bash run_compile.sh 2>&1 | tee ${ut_log_name}

cp report.html ${LOG_DIR}/
cp coverage.xml ${LOG_DIR}/

if [ $(grep -c '== FAILURES ==' ${ut_log_name}) != 0 ] || [ $(grep -c '== ERRORS ==' ${ut_log_name}) != 0 ] || [ $(grep -c ' passed' ${ut_log_name}) == 0 ]; then
    echo "##[error]Find errors in pytest case, please check the output..."
    exit 1
fi

# if ut pass, collect the coverage file into artifacts
cp .coverage ${LOG_DIR}/.coverage

echo "UT finished successfully! "