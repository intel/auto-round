#!/bin/bash
set -e
uv pip install coverage

# $1: hardware type used to pick the matching .coveragerc (defaults to cpu).
HW_TYPE=${1:-cpu}
export COVERAGE_RCFILE=${BUILD_SOURCESDIRECTORY}/.azure-pipelines/scripts/ut/coveragerc/${HW_TYPE}.coveragerc
coverage_log="${BUILD_SOURCESDIRECTORY}/log_dir/coverage_log"
cd "${BUILD_SOURCESDIRECTORY}/log_dir"

echo "collect coverage for PR branch"
mkdir -p coverage_PR
# Every matrix part may legitimately select no test (change-based filtering),
# in which case there is nothing to combine.
if ! compgen -G "*_coverage/.coverage.*" > /dev/null; then
    echo "no coverage data found, skip coverage collection."
    exit 0
fi
cp *_coverage/.coverage.* ./coverage_PR/

# "coverage combine" must run from the repo root: the first entry of
# "[paths] source" is the relative path "auto_round", which coverage.py resolves
# against the current directory. Combining from anywhere else leaves the
# recorded install paths unmapped, so parts that measured different prefixes
# (site-packages vs source tree) would count every file twice.
cd "${BUILD_SOURCESDIRECTORY}"
rm -f .coverage
coverage combine --keep --rcfile=${COVERAGE_RCFILE} log_dir/coverage_PR

cp .coverage log_dir/coverage_PR/.coverage
coverage report -m --rcfile=${COVERAGE_RCFILE} | tee ${coverage_log}
coverage html -d log_dir/coverage_PR/htmlcov --rcfile=${COVERAGE_RCFILE}
coverage xml -o log_dir/coverage_PR/coverage.xml --rcfile=${COVERAGE_RCFILE}
ls -l log_dir/coverage_PR/htmlcov
