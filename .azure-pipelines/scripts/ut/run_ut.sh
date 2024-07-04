#!/bin/bash
set -xe

# install requirements
echo "set up UT env..."
pip install pytest-cov pytest-html
pip list

export COVERAGE_RCFILE=/auto-round/.azure-pipelines/scripts/ut/.coverage
aoto_round_path=$(python -c 'import auto_round; print(auto_round.__path__[0])')
cd /auto-round/test || exit 1

LOG_DIR=/auto-round/log_dir
mkdir -p ${LOG_DIR}
ut_log_name=${LOG_DIR}/ut.log
pytest --cov="${aoto_round_path}" -vs --disable-warnings --html=report.html --self-contained-html . 2>&1 | tee -a ${ut_log_name}

cp report.html ${LOG_DIR}/

if [ $(grep -c '== FAILURES ==' ${ut_log_name}) != 0 ] || [ $(grep -c '== ERRORS ==' ${ut_log_name}) != 0 ] || [ $(grep -c ' passed' ${ut_log_name}) == 0 ]; then
    echo "Find errors in pytest case, please check the output..."
    echo "Please search for '== FAILURES ==' or '== ERRORS =='"
    exit 1
fi

# if ut pass, collect the coverage file into artifacts
cp .coverage ${LOG_DIR}/.coverage

echo "UT finished successfully! "