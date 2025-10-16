#!/bin/bash
set -e
pip install coverage
export COVERAGE_RCFILE=/auto-round/.azure-pipelines/scripts/ut/.coverage
coverage_log="/auto-round/log_dir/coverage_log"
cd /auto-round/log_dir

echo "collect coverage for PR branch"
mkdir -p coverage_PR
cp ut-*/.coverage.* ./coverage_PR/
cd coverage_PR
coverage combine --keep --rcfile=${COVERAGE_RCFILE}

cp .coverage /auto-round/
cd /auto-round
coverage report -m --rcfile=${COVERAGE_RCFILE} | tee ${coverage_log}
coverage html -d log_dir/coverage_PR/htmlcov --rcfile=${COVERAGE_RCFILE}
coverage xml -o log_dir/coverage_PR/coverage.xml --rcfile=${COVERAGE_RCFILE}
ls -l log_dir/coverage_PR/htmlcov
