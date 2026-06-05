"""Match CI failures against known-issue tracker entries.

Pulls open issues labelled ``CI_known_issue`` from the repository and scores
them against the failure signals.  A high score means the failure is already
tracked, so the pipeline can comment "known issue, safe to skip" instead of
re-analysing it as a regression.
"""

import json
import os
import re
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

DEFAULT_LABEL = "CI_known_issue"
# Tokens shorter than this or in the stop list add noise to the overlap score.
_MIN_TOKEN_LEN = 4
_STOP_TOKENS = {
    "test", "tests", "error", "errors", "failed", "failure", "assert",
    "self", "none", "true", "false", "value", "object", "python", "trace",
    "traceback", "line", "file", "call", "last", "most", "recent",
}
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
# Identifiers that are strong fingerprints: exception classes, error codes.
_EXCEPTION_RE = re.compile(r"\b([A-Z][A-Za-z0-9]+(?:Error|Exception|Warning))\b")


def github_get(url: str, token: str):
    req = Request(
        url,
        method="GET",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urlopen(req) as resp:
        data = resp.read().decode("utf-8")
        return json.loads(data) if data else []


def _tokenize(text: str) -> set[str]:
    tokens = {
        tok.lower()
        for tok in _TOKEN_RE.findall(text or "")
        if len(tok) >= _MIN_TOKEN_LEN
    }
    return tokens - _STOP_TOKENS


def _exceptions(text: str) -> set[str]:
    return {m.lower() for m in _EXCEPTION_RE.findall(text or "")}


def fetch_known_issues(repo_path: str, token: str, label: str = DEFAULT_LABEL) -> list[dict]:
    """Fetch open issues carrying the known-issue label."""
    if not token or not repo_path:
        return []
    url = (
        f"https://api.github.com/repos/{repo_path}/issues"
        f"?state=open&labels={quote(label)}&per_page=100"
    )
    try:
        issues = github_get(url, token)
    except Exception as exc:  # noqa: BLE001 - network best-effort
        print(f"known_issue_matcher: failed to fetch issues: {exc}")
        return []

    result = []
    for issue in issues if isinstance(issues, list) else []:
        # Pull requests are also returned by the issues endpoint; skip them.
        if "pull_request" in issue:
            continue
        result.append(
            {
                "number": issue.get("number"),
                "title": issue.get("title", ""),
                "body": issue.get("body", "") or "",
                "url": issue.get("html_url", ""),
            }
        )
    return result


def _failure_text(failures: list[dict]) -> str:
    parts = []
    for entry in failures:
        parts.append(entry.get("test_name", ""))
        parts.append(entry.get("excerpt", ""))
        parts.append(entry.get("tail", ""))
    return "\n".join(p for p in parts if p)


def score_issue(issue: dict, failure_tokens: set[str], failure_excs: set[str]) -> dict:
    """Score one issue against the aggregated failure fingerprint."""
    issue_text = f"{issue['title']}\n{issue['body']}"
    issue_tokens = _tokenize(issue_text)
    issue_excs = _exceptions(issue_text)

    shared_tokens = failure_tokens & issue_tokens
    shared_excs = failure_excs & issue_excs

    token_score = len(shared_tokens) / max(1, len(issue_tokens)) if issue_tokens else 0.0
    # Exception-class overlap is a much stronger signal than generic tokens.
    exc_score = 1.0 if shared_excs else 0.0

    confidence = round(min(1.0, 0.6 * token_score + 0.4 * exc_score), 3)
    return {
        "number": issue["number"],
        "title": issue["title"],
        "url": issue["url"],
        "confidence": confidence,
        "shared_exceptions": sorted(shared_excs),
        "shared_keywords": sorted(shared_tokens)[:15],
    }


def match_known_issues(
    failures: list[dict],
    repo_path: str,
    token: str,
    label: str = DEFAULT_LABEL,
    min_confidence: float = 0.3,
) -> dict:
    """Return ranked known-issue matches for the given failures."""
    issues = fetch_known_issues(repo_path, token, label)
    if not issues:
        return {"label": label, "checked": 0, "matches": []}

    failure_text = _failure_text(failures)
    failure_tokens = _tokenize(failure_text)
    failure_excs = _exceptions(failure_text)

    scored = [score_issue(issue, failure_tokens, failure_excs) for issue in issues]
    matches = sorted(
        (s for s in scored if s["confidence"] >= min_confidence),
        key=lambda s: s["confidence"],
        reverse=True,
    )
    return {"label": label, "checked": len(issues), "matches": matches}


def _repo_path_from_env() -> str:
    uri = os.environ.get("BUILD_REPOSITORY_URI", "")
    if uri.startswith("https://github.com/"):
        return uri.replace("https://github.com/", "").removesuffix(".git")
    return os.environ.get("REPO_PATH", "")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Match failures against known issues")
    parser.add_argument("--failure-context", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    args = parser.parse_args()

    with open(args.failure_context, "r", encoding="utf-8") as f:
        payload = json.load(f)
    failures = payload.get("failures", [])

    token = (
        os.environ.get("GITHUB_TOKEN", "")
        or os.environ.get("GH_TOKEN", "")
        or os.environ.get("COPILOT_GITHUB_TOKEN", "")
    )
    repo_path = _repo_path_from_env()

    result = match_known_issues(
        failures, repo_path, token, label=args.label, min_confidence=args.min_confidence
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"known_issue_matcher: checked {result['checked']} issues, "
          f"{len(result['matches'])} matches >= {args.min_confidence}")


if __name__ == "__main__":
    main()
