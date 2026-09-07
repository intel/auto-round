"""Post (or update) the AI failure-analysis report as a PR comment.

Idempotent: the comment carries a hidden marker so reruns update the existing
comment instead of spamming new ones. Fails soft — a posting problem never
breaks the pipeline.

With ``--hide`` the script instead minimizes the marker comment as OUTDATED,
used when a rerun passes so the stale failure report collapses automatically.
"""

import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_MARKER = "<!-- ai-failure-analysis -->"
# GitHub caps issue-comment bodies at 65536 chars; leave room for the marker.
_MAX_BODY = 65000


def _build_marker(key: str) -> str:
    """Return the hidden comment marker, scoped by an optional per-workflow key.

    Distinct keys keep separate workflows from overwriting each other's comment.
    """
    key = (key or "").strip()
    return f"<!-- ai-failure-analysis:{key} -->" if key else _MARKER


def _api(url: str, token: str, method: str = "GET", payload: "dict | None" = None) -> "list | dict":
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, headers=headers, method=method)
    with urlopen(req, timeout=20) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body) if body else {}


def _find_existing(repo: str, pr: int, token: str, marker: str) -> "int | None":
    c = _find_existing_comment(repo, pr, token, marker)
    return c.get("id") if c else None


def _find_existing_comment(repo: str, pr: int, token: str, marker: str) -> "dict | None":
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo}/issues/{pr}/comments?per_page=100&page={page}"
        comments = _api(url, token)
        if not isinstance(comments, list) or not comments:
            return None
        for c in comments:
            if marker in (c.get("body") or ""):
                return c
        if len(comments) < 100:
            return None
        page += 1


def _graphql(token: str, query: str, variables: dict) -> dict:
    req = Request(
        "https://api.github.com/graphql",
        data=json.dumps({"query": query, "variables": variables}).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _minimize_comment(node_id: str, token: str, classifier: str = "OUTDATED") -> bool:
    """Hide (minimize) a PR comment via GraphQL, marking it as e.g. OUTDATED."""
    query = (
        "mutation($id: ID!, $reason: ReportedContentClassifiers!) {"
        "  minimizeComment(input: {subjectId: $id, classifier: $reason}) {"
        "    minimizedComment { isMinimized minimizedReason }"
        "  }"
        "}"
    )
    result = _graphql(token, query, {"id": node_id, "reason": classifier})
    if result.get("errors"):
        print(f"Warning: could not hide comment ({result['errors']}); non-blocking.", file=sys.stderr)
        return False
    return bool(result.get("data", {}).get("minimizeComment", {}).get("minimizedComment", {}).get("isMinimized"))


def _hide(repo: str, pr: int, token: str, marker: str) -> None:
    """Find the marker comment and hide it as OUTDATED (used when the run passes)."""
    try:
        existing = _find_existing_comment(repo, pr, token, marker)
    except (HTTPError, URLError, ValueError) as e:
        print(f"Warning: could not look up PR comment ({e}); non-blocking.", file=sys.stderr)
        return
    if not existing:
        print("No existing analysis comment to hide.", file=sys.stderr)
        return
    node_id = existing.get("node_id")
    if not node_id:
        print("Existing comment has no node_id; cannot hide.", file=sys.stderr)
        return
    try:
        if _minimize_comment(node_id, token):
            print(f"Hid PR comment {existing.get('id')} as OUTDATED.", file=sys.stderr)
    except (HTTPError, URLError, ValueError) as e:
        print(f"Warning: could not hide PR comment ({e}); non-blocking.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Post the failure-analysis report to a PR comment")
    parser.add_argument("--repo", default=os.environ.get("REPO_PATH", ""), help="owner/name")
    parser.add_argument("--pr", default=os.environ.get("PR_NUMBER", ""), help="PR number")
    parser.add_argument("--body-file", help="Markdown report to post (required unless --hide)")
    parser.add_argument(
        "--marker-key",
        default=os.environ.get("AI_COMMENT_MARKER_KEY", ""),
        help="Per-workflow key so different pipelines update separate PR comments",
    )
    parser.add_argument(
        "--hide",
        action="store_true",
        help="Hide the existing analysis comment as OUTDATED instead of posting (use when the run passes)",
    )
    parser.add_argument("--token-env", default="AUTO_ROUND_BOT_TOKEN", help="Env var holding the GitHub token")
    args = parser.parse_args()

    marker = _build_marker(args.marker_key)
    token = os.environ.get(args.token_env, "")
    if not args.repo or "/" not in args.repo or not str(args.pr).strip():
        print("No repo/PR number available; skipping PR comment.", file=sys.stderr)
        return
    if not token:
        print(f"No token in ${args.token_env}; skipping PR comment.", file=sys.stderr)
        return

    if args.hide:
        _hide(args.repo, int(args.pr), token, marker)
        return

    if not args.body_file:
        print("No --body-file provided; nothing to post.", file=sys.stderr)
        return
    if not os.path.isfile(args.body_file):
        print(f"Report file not found: {args.body_file}; skipping.", file=sys.stderr)
        return

    with open(args.body_file, encoding="utf-8") as f:
        body = f.read()
    if len(body) > _MAX_BODY:
        body = body[:_MAX_BODY] + "\n\n_...truncated; see the pipeline artifact for the full report._"
    body = f"{marker}\n{body}"

    pr = int(args.pr)
    try:
        existing = _find_existing(args.repo, pr, token, marker)
        if existing is not None:
            _api(
                f"https://api.github.com/repos/{args.repo}/issues/comments/{existing}",
                token,
                method="PATCH",
                payload={"body": body},
            )
            print(f"Updated existing PR comment {existing}.", file=sys.stderr)
        else:
            _api(
                f"https://api.github.com/repos/{args.repo}/issues/{pr}/comments",
                token,
                method="POST",
                payload={"body": body},
            )
            print("Created new PR comment.", file=sys.stderr)
    except (HTTPError, URLError, ValueError) as e:
        print(f"Warning: could not post PR comment ({e}); non-blocking.", file=sys.stderr)


if __name__ == "__main__":
    main()
