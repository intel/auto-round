# A retry is any job attempt after the first; failed_tests.list is downloaded
# from the previous attempt before this script runs (clean environment).

SYSTEM_JOBATTEMPT=${SYSTEM_JOBATTEMPT:-1}
LOG_DIR=${LOG_DIR:-/auto-round/log_dir}

function is_retry() {
    [ "${SYSTEM_JOBATTEMPT:-1}" -gt 1 ] && [ -s "${LOG_DIR}/failed_tests.list" ]
}

function retry_selection() {
    is_retry && sort -u "${LOG_DIR}/failed_tests.list"
}

function run_if_retry() {
    is_retry || return 1
    # Seed coverage from the previous attempt so --cov-append accumulates.
    if [ -f "${LOG_DIR}/.coverage" ]; then
        cp "${LOG_DIR}/.coverage" ./.coverage
    fi
    run_pytest "$(retry_selection)" "${LOG_DIR}/unittest_test_job_attempt_${SYSTEM_JOBATTEMPT}.log"
    return 0
}