#!/bin/bash
# Detect the test files modified by the current change set.
#
# Usage:
#   source detect_changed_tests.sh
#   detect_changed_tests            # prints repo-relative test files, one per line
#   returns 0 -> filtering applies (a change set was detected)
#   returns 1 -> filtering does NOT apply, caller should run the full suite
#
# Detection order:
#   1. Filtering is opt-in: unless FILTER_CHANGED_TESTS is truthy the full suite
#      runs. This keeps pipelines that do not request it running everything.
#   2. Only PR builds are filtered. When BUILD_REASON is set and is not
#      "PullRequest" (e.g. a scheduled or manual run) the full suite runs.
#   3. Diff the working tree against BASE_REF (default origin/main, or the PR
#      target branch when available). Filtering only kicks in when the change
#      set is EXCLUSIVELY test files (test/**/test_*.py); if any other file
#      changed (source, helpers, conftest, deletions, ...) the full suite runs.
#
# Only existing files matching test/**/test_*.py are reported; a deleted test
# file counts as a non-test change so the full suite runs.

REPO_DIR="${REPO_DIR:-/auto-round}"
# Prefer the PR target branch reported by Azure DevOps; fall back to origin/main.
_pr_target="${SYSTEM_PULLREQUEST_TARGETBRANCH#refs/heads/}"
BASE_REF="${BASE_REF:-origin/${_pr_target:-main}}"

detect_changed_tests() {
    # Filtering is opt-in; pipelines that do not request it run everything.
    case "${FILTER_CHANGED_TESTS:-}" in
        1 | true | True | TRUE) ;;
        *)
            echo "FILTER_CHANGED_TESTS not enabled, run full suite." >&2
            return 1
            ;;
    esac

    # Only PR builds are filtered; scheduled/manual runs execute everything.
    if [ -n "${BUILD_REASON:-}" ] && [ "${BUILD_REASON}" != "PullRequest" ]; then
        echo "BUILD_REASON=${BUILD_REASON} is not a PR, run full suite." >&2
        return 1
    fi

    if ! command -v git > /dev/null 2>&1; then
        echo "git not available, run full suite." >&2
        return 1
    fi

    cd "${REPO_DIR}" 2>/dev/null || return 1
    git config --global --add safe.directory "${REPO_DIR}" > /dev/null 2>&1 || true

    # Best-effort fetch so BASE_REF resolves; ignore failures (offline runners).
    # CI checkouts are often shallow, which breaks merge-base; unshallow first.
    local base_branch="${BASE_REF#origin/}"
    if [ "$(git rev-parse --is-shallow-repository 2>/dev/null)" = "true" ]; then
        git fetch --quiet --unshallow origin > /dev/null 2>&1 || true
    fi
    git fetch --quiet origin "+refs/heads/${base_branch}:refs/remotes/origin/${base_branch}" > /dev/null 2>&1 || true

    local base_sha
    base_sha="$(git merge-base "${BASE_REF}" HEAD 2>/dev/null)" || base_sha=""
    if [ -z "${base_sha}" ]; then
        echo "cannot resolve base ref '${BASE_REF}', run full suite." >&2
        return 1
    fi

    local all_changed test_changed non_test
    # NOTE: no --diff-filter here. Deletions must stay visible, otherwise a PR
    # that removes a source file and touches a test would look test-only.
    all_changed="$(git diff --name-only "${base_sha}" HEAD 2>/dev/null)"
    if [ -z "${all_changed}" ]; then
        echo "no changed file detected, run full suite." >&2
        return 1
    fi

    # Added/copied/modified/renamed only: a deleted test cannot be executed.
    test_changed="$(git diff --name-only --diff-filter=ACMR "${base_sha}" HEAD -- 'test/**/test_*.py' 2>/dev/null)"
    if [ -z "${test_changed}" ]; then
        echo "no changed test file detected, run full suite." >&2
        return 1
    fi

    # Only filter when the change set is exclusively test files. Any other file
    # (source, test helpers, conftest, requirements, ...) may affect many tests,
    # so fall back to the full suite.
    non_test="$(comm -23 <(printf '%s\n' "${all_changed}" | sort -u) \
                         <(printf '%s\n' "${test_changed}" | sort -u))"
    if [ -n "${non_test}" ]; then
        echo "##[group]non-test files changed, run full suite:" >&2
        printf '  %s\n' "${non_test}" >&2
        echo "##[endgroup]" >&2
        return 1
    fi

    printf '%s\n' "${test_changed}"
    return 0
}

# ---------------------------------------------------------------------------
# Shared helpers for the run_ut*.sh scripts.
# ---------------------------------------------------------------------------

# Populated by init_changed_tests(): whitespace separated repo-relative test
# files touched by this change set. FILTER_TESTS=1 means only run those files.
CHANGED_TEST_FILES=""
FILTER_TESTS=0

# Resolve the change set once and cache it for filter_changed_tests().
init_changed_tests() {
    local changed
    if changed=$(detect_changed_tests); then
        CHANGED_TEST_FILES=$(printf '%s' "${changed}" | tr '\n' ' ')
        FILTER_TESTS=1
        echo "Only running changed test files: ${CHANGED_TEST_FILES}"
    else
        FILTER_TESTS=0
        echo "Running the full test suite."
    fi
}

# Disable filtering when none of the changed tests can be executed by this
# runner. Without this every matrix part could end up with an empty selection
# and the job would produce no results, no summary and no coverage at all.
# $@: repo-relative paths of every test this runner is able to execute.
scope_changed_tests() {
    [ "${FILTER_TESTS}" = "1" ] || return 0

    local scope="$*"
    local changed candidate
    for changed in ${CHANGED_TEST_FILES}; do
        for candidate in ${scope}; do
            if [ "${changed}" = "${candidate#./}" ]; then
                return 0
            fi
        done
    done

    echo "Changed tests are outside this runner's scope, run the full test suite."
    FILTER_TESTS=0
}

# Keep only the entries of a candidate list that were modified by this PR.
# $1: repo-relative directory the candidates are relative to (e.g. "test")
# $2: newline separated candidate test files (e.g. "./unit/test_cpu/test_a.py")
filter_changed_tests() {
    local base_dir=$1
    local candidates=$2

    if [ "${FILTER_TESTS}" != "1" ]; then
        printf '%s' "${candidates}"
        return 0
    fi

    local kept=""
    local candidate normalized changed
    while IFS= read -r candidate; do
        [ -z "${candidate}" ] && continue
        normalized="${base_dir}/${candidate#./}"
        for changed in ${CHANGED_TEST_FILES}; do
            if [ "${changed#./}" = "${normalized}" ]; then
                kept+="${candidate}"$'\n'
                break
            fi
        done
    done <<< "${candidates}"

    printf '%s' "${kept}"
}
