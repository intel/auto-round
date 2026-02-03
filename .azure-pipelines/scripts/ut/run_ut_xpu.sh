#!/bin/bash
set -xe

# install requirements
echo "##[group]set up UT env..."
uv pip install pytest-cov pytest-html
uv pip list
echo "##[endgroup]"

# test ark cpu part only before external xpu available
rm -rf /auto-round/auto_round
cd /auto-round/test || exit 1

export ZE_AFFINITY_MASK=2,3 # set xpu affinity
export LD_LIBRARY_PATH=${HOME}/.venv/lib/:$LD_LIBRARY_PATH
export COVERAGE_RCFILE=/auto-round/.azure-pipelines/scripts/ut/.coverage
auto_round_path=$(python -c 'import auto_round; print(auto_round.__path__[0])')

LOG_DIR=/auto-round/log_dir
mkdir -p ${LOG_DIR}
ut_log_name=${LOG_DIR}/ut.log

find ./test_ark -name "test*.py" | sed "s,\.\/,python -m pytest --cov=\"${auto_round_path}\" --cov-report term --html=report.html --self-contained-html --cov-report xml:coverage.xml --cov-append -vs --disable-warnings ,g" > run.sh
cat run.sh
bash run.sh 2>&1 | tee "${ut_log_name}"

cp report.html ${LOG_DIR}/
cp coverage.xml ${LOG_DIR}/

if [ $(grep -c '== FAILURES ==' ${ut_log_name}) != 0 ] || [ $(grep -c '== ERRORS ==' ${ut_log_name}) != 0 ] || [ $(grep -c 'Killed' ${ut_log_name}) != 0 ] || [ $(grep -c ' passed' ${ut_log_name}) == 0 ]; then
    echo "##[error]Find errors in pytest case, please check the output..."
    exit 1
fi

cp .coverage "${LOG_DIR}/.coverage"

echo "UT finished successfully! "
