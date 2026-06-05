# Environment Failure Checklist

Use this to decide whether a CI failure is an **Environment** issue (infra/network/
runner problem) rather than a code defect. Environment failures usually pass on a
clean rerun, so they should route to the automatic rerun path.

## Strong signals (any one is usually decisive)
- Network errors: `Connection reset`, `Connection timed out`, `Temporary failure in
  name resolution`, `Read timed out`, `Max retries exceeded`, TLS/SSL handshake errors.
- Package/index outages: failures contacting `pypi.org`, `download.pytorch.org`,
  `huggingface.co`, `test.pypi.org`; HTTP 429/500/502/503/504 from these hosts.
- Model/dataset download errors: `HfHubHTTPError`, `RepositoryNotFoundError` caused by
  rate limiting, `OSError: ... Connection`, incomplete/`.incomplete` cache files.
- Resource exhaustion: `No space left on device`, `OSError: [Errno 28]`,
  `CUDA out of memory`, `Cannot allocate memory`, `Killed` (OOM killer).
- Runner/agent problems: `The agent lost communication`, `docker: Error response from
  daemon`, container failed to start, disk pressure evictions.
- Rate limiting: `API rate limit exceeded`, HTTP 403 with `rate limit` text.

## Counter-signals (NOT environment)
- A clean Python traceback ending in `AssertionError`, `ValueError`, `TypeError`,
  `KeyError`, etc. inside auto-round source or a test assertion.
- The failing test file or its imported source module was modified by this PR.
- Numerical/accuracy mismatches (e.g. allclose failures) without any infra error text.

## Decision rule
Classify as Environment only when a strong infra signal is present AND there is no
clear in-code assertion/exception attributable to PR changes. If both are present,
prefer Code Regression (a real bug should not be hidden by a rerun).
