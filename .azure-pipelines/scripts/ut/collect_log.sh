#!/bin/bash
set -e
uv pip install coverage
export COVERAGE_RCFILE=${BUILD_SOURCESDIRECTORY}/auto-round/.azure-pipelines/scripts/ut/.coverage
coverage_log="${BUILD_SOURCESDIRECTORY}/auto-round/log_dir/coverage_log"
cd "${BUILD_SOURCESDIRECTORY}/auto-round/log_dir"

echo "collect coverage for PR branch"
mkdir -p coverage_PR
cp ut-*/.coverage.* ./coverage_PR/
cd coverage_PR
coverage combine --keep --rcfile=${COVERAGE_RCFILE}

cp .coverage "${BUILD_SOURCESDIRECTORY}/auto-round/"
cd "${BUILD_SOURCESDIRECTORY}/auto-round"
coverage report -m --rcfile=${COVERAGE_RCFILE} | tee ${coverage_log}
coverage html -d log_dir/coverage_PR/htmlcov --rcfile=${COVERAGE_RCFILE}
coverage xml -o log_dir/coverage_PR/coverage.xml --rcfile=${COVERAGE_RCFILE}
ls -l log_dir/coverage_PR/htmlcov
