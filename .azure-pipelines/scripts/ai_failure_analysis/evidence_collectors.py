"""Deterministic forensic evidence collectors for CI failure classification.

These collectors turn raw failure logs into structured signals so the
classifier (and the Copilot agent) make a *multiple-choice* decision instead of
re-reading megabytes of logs.  The quality of these signals is what actually
improves classification hit-rate; the category labels are only a routing layer.

Each collector returns a JSON-serializable dict.  ``collect_all_evidence``
aggregates them into a single evidence bundle consumed by ``classify.py``.
"""

import re
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment signal patterns
# ---------------------------------------------------------------------------
# Each entry: (signal_key, human_label, compiled_regex).  Patterns are matched
# case-insensitively against the failure excerpt + tail text.
ENVIRONMENT_PATTERNS: list[tuple[str, str, re.Pattern]] = [
    (
        "network_timeout",
        "Network timeout / connection failure",
        re.compile(
            r"(connection\s+timed?\s*out|read\s+timed?\s*out|connection\s+reset|"
            r"failed\s+to\s+establish\s+a\s+new\s+connection|max\s+retries\s+exceeded|"
            r"temporary\s+failure\s+in\s+name\s+resolution|connection\s+aborted|"
            r"ETIMEDOUT|ECONNRESET|ECONNREFUSED)",
            re.IGNORECASE,
        ),
    ),
    (
        "disk_full",
        "Disk space exhausted",
        re.compile(
            r"(no\s+space\s+left\s+on\s+device|disk\s+quota\s+exceeded|ENOSPC|"
            r"cannot\s+write\s+.*:\s+no\s+space)",
            re.IGNORECASE,
        ),
    ),
    (
        "runner_lost",
        "Runner / agent lost or disconnected",
        re.compile(
            r"(the\s+agent\s+(?:lost\s+communication|did\s+not\s+respond)|"
            r"lost\s+communication\s+with\s+the\s+agent|runner\s+has\s+been\s+lost|"
            r"we\s+stopped\s+hearing\s+from\s+agent|received\s+request\s+to\s+deprovision)",
            re.IGNORECASE,
        ),
    ),
    (
        "out_of_memory",
        "Out of memory / OOM killer",
        re.compile(
            r"(out\s+of\s+memory|cannot\s+allocate\s+memory|oom[\s-]?kill|"
            r"killed\s+process|DefaultCPUAllocator:\s+can't\s+allocate|"
            r"std::bad_alloc|MemoryError)",
            re.IGNORECASE,
        ),
    ),
    (
        "hf_download_error",
        "HuggingFace / model hub download failure",
        re.compile(
            r"(HfHubHTTPError|huggingface_hub\.utils.*Error|"
            r"couldn'?t\s+connect\s+to\s+['\"]?https?://huggingface\.co|"
            r"504\s+server\s+error.*huggingface|repository\s+not\s+found.*huggingface|"
            r"we\s+couldn'?t\s+connect\s+to\s+['\"]?https?://huggingface\.co)",
            re.IGNORECASE,
        ),
    ),
    (
        "rate_limited",
        "API rate limit / throttling",
        re.compile(
            r"(rate\s+limit\s+exceeded|too\s+many\s+requests|HTTP\s+429|"
            r"429\s+client\s+error)",
            re.IGNORECASE,
        ),
    ),
]

# Markers used to derive the candidate test file path from a pytest nodeid.
_TEST_NAME_FILE_RE = re.compile(r"^(?P<file>test_[\w/.-]+?\.py)(?:::|$)")


def _failure_text(entry: dict) -> str:
    """Concatenate the searchable text for a single failure entry."""
    return "\n".join(
        part for part in (entry.get("excerpt", ""), entry.get("tail", "")) if part
    )


def _snippet_around(text: str, match: re.Match, radius: int = 120) -> str:
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    snippet = text[start:end].strip().replace("\r", "")
    return re.sub(r"\s+", " ", snippet)


def collect_environment_signals(failures: list[dict]) -> dict:
    """Scan failures for environment-related signals.

    Returns a dict with matched signal keys, supporting evidence snippets, and
    the set of tests in which each signal appeared.
    """
    matched: dict[str, dict] = {}

    for entry in failures:
        text = _failure_text(entry)
        if not text:
            continue
        test_name = entry.get("test_name", "")
        for signal_key, label, pattern in ENVIRONMENT_PATTERNS:
            found = pattern.search(text)
            if not found:
                continue
            record = matched.setdefault(
                signal_key,
                {"label": label, "evidence": [], "tests": []},
            )
            snippet = _snippet_around(text, found)
            if snippet and snippet not in record["evidence"]:
                record["evidence"].append(snippet)
            if test_name and test_name not in record["tests"]:
                record["tests"].append(test_name)

    return {
        "has_environment_signal": bool(matched),
        "signals": matched,
    }


def _run_git(args: list[str], cwd: Path) -> tuple[int, str]:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=False, capture_output=True, text=True
    )
    return result.returncode, (result.stdout or "").strip()


def get_pr_changed_files(project_root: Path, base_ref: str = "main") -> list[str]:
    """Return the list of files changed by the PR relative to base_ref."""
    diff_specs = [
        f"origin/{base_ref}...HEAD",
        f"{base_ref}...HEAD",
        "HEAD~1..HEAD",
    ]
    for spec in diff_specs:
        code, out = _run_git(["diff", "--name-only", spec], cwd=project_root)
        if code == 0:
            files = [line.strip() for line in out.splitlines() if line.strip()]
            if files:
                return sorted(set(files))
    return []


def _candidate_test_file(test_name: str) -> str:
    """Best-effort mapping from a pytest test name/nodeid to a test file path."""
    if not test_name:
        return ""
    match = _TEST_NAME_FILE_RE.match(test_name)
    if match:
        return match.group("file")
    # Bare function/class name like ``test_quantize`` -> ``test_quantize.py``.
    base = test_name.split("::", 1)[0]
    if base.startswith("test_") and not base.endswith(".py"):
        return f"{base}.py"
    return base


def collect_pr_relevance(
    failures: list[dict], project_root: Path, base_ref: str = "main"
) -> dict:
    """Correlate failed tests with PR-changed files.

    This is the strongest signal separating Code Regression from other classes:
    if the PR touched ``foo.py`` and ``test_foo.py`` fails, regression is likely.
    """
    changed_files = get_pr_changed_files(project_root, base_ref)
    changed_set = set(changed_files)
    changed_stems = {Path(f).stem for f in changed_files if f.endswith(".py")}

    # Tests whose own test file was modified by the PR.
    directly_changed_tests: list[str] = []
    # Tests whose corresponding source module (by stem) was modified.
    source_related_tests: list[dict] = []

    for entry in failures:
        test_name = entry.get("test_name", "")
        test_file = _candidate_test_file(test_name)
        test_stem = Path(test_file).stem if test_file else ""

        # Direct: the failing test file itself is in the PR diff.
        for changed in changed_set:
            if test_file and changed.endswith(test_file):
                if test_name not in directly_changed_tests:
                    directly_changed_tests.append(test_name)
                break

        # Indirect: a changed source module matches the test's subject.
        # ``test_quantize`` -> subject stem ``quantize``.
        subject_stem = test_stem[len("test_"):] if test_stem.startswith("test_") else test_stem
        related_modules = sorted(
            f for f in changed_files
            if f.endswith(".py") and subject_stem and Path(f).stem == subject_stem
        )
        if related_modules:
            source_related_tests.append(
                {"test": test_name, "modules": related_modules}
            )

    relevance_score = 0.0
    if failures:
        related_count = len(directly_changed_tests) + len(source_related_tests)
        relevance_score = round(min(1.0, related_count / max(1, len(failures))), 3)

    # Heuristic: PR only touches code under auto_round/ vs only tests/ vs docs.
    touches_source = any(
        f.startswith("auto_round/") or f.startswith("auto_round_extension/")
        for f in changed_files
    )
    touches_tests = any(f.startswith("test/") for f in changed_files)

    return {
        "changed_file_count": len(changed_files),
        "changed_files": changed_files[:100],
        "changed_source_stems": sorted(changed_stems)[:100],
        "directly_changed_tests": directly_changed_tests,
        "source_related_tests": source_related_tests,
        "relevance_score": relevance_score,
        "touches_source": touches_source,
        "touches_tests": touches_tests,
    }


def collect_dependency_changes(project_root: Path) -> dict:
    """Phase 2 stub: compare current dependencies against a success baseline.

    The baseline (``pip freeze`` from the last successful build) will be
    published as a pipeline artifact and diffed here.  Until that artifact
    exists this returns ``available=False`` so the classifier can skip the
    Dependency branch cleanly.
    """
    return {
        "available": False,
        "reason": "dependency baseline artifact not wired yet (Phase 2)",
        "changed_packages": [],
    }


def collect_flaky_signals(failures: list[dict]) -> dict:
    """Phase 2 stub: detect flaky tests via history / rerun comparison.

    A full implementation will consult historical pass/fail rates and the
    Environment-rerun outcome.  For now it returns a neutral, low-confidence
    result so the classifier does not over-trigger the Flaky branch.
    """
    return {
        "available": False,
        "reason": "flaky history source not wired yet (Phase 2)",
        "suspected_flaky_tests": [],
    }


def collect_all_evidence(
    failures: list[dict], project_root: Path, base_ref: str = "main"
) -> dict:
    """Aggregate every collector into a single evidence bundle."""
    return {
        "failure_count": len(failures),
        "environment": collect_environment_signals(failures),
        "pr_relevance": collect_pr_relevance(failures, project_root, base_ref),
        "dependency": collect_dependency_changes(project_root),
        "flaky": collect_flaky_signals(failures),
    }
