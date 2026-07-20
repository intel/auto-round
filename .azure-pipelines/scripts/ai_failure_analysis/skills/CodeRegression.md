# Code Regression Checklist

Use this to decide whether a CI failure is a **Code Regression** caused by the PR's
own changes. These failures must NOT be rerun blindly; they need a root-cause analysis
and a suggested fix.

## Strong signals
- The failing test file is in `directly_changed_tests` (the PR edited that test), or
  the test imports a source module whose stem is in `changed_source_stems`
  (see `pr_relevance` in the evidence bundle).
- A deterministic Python traceback terminating inside `auto_round/` or
  `auto_round_extension/` source touched by the PR.
- Assertion failures tied to logic the PR changed: `AssertionError`, allclose/shape
  mismatches, changed default values, new/removed function arguments
  (`TypeError: ... unexpected keyword argument`, `missing N required positional`).
- Import errors for symbols the PR renamed/moved (`ImportError`, `AttributeError:
  module ... has no attribute`).

## Counter-signals (probably NOT a regression)
- Strong environment signals (network/disk/OOM/runner) with no in-code assertion.
- The failure reproduces on `main` / is already a tracked known issue.
- The failing area is completely unrelated to the PR's changed files and the error is
  intermittent (consider Flaky Test).

## Root-cause guidance for the analysis step
1. Map each failing test to its source-of-truth module using `pr_relevance`.
2. Read the changed hunks for that module; look for changed signatures, defaults,
   control flow, or removed guards.
3. Confirm the traceback line corresponds to changed code.
4. Propose the minimal fix; do NOT auto-apply — emit it as a reviewable patch only.

## Decision rule
Classify as Code Regression when PR-relevance is high (failing test ∩ PR-changed
source/tests) AND the error is a deterministic in-code failure. Confidence should
scale with how directly the changed code appears in the traceback.
