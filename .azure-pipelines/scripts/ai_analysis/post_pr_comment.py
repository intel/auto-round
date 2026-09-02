"""Post (or update) the AI failure-analysis report as a PR comment.

Idempotent: the comment carries a hidden marker so reruns update the existing
comment instead of spamming new ones. Fails soft — a posting problem never
breaks the pipeline.
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


def _find_existing(repo: str, pr: int, token: str) -> "int | None":
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo}/issues/{pr}/comments?per_page=100&page={page}"
        comments = _api(url, token)
        if not isinstance(comments, list) or not comments:
            return None
        for c in comments:
            if _MARKER in (c.get("body") or ""):
                return c.get("id")
        if len(comments) < 100:
            return None
        page += 1


def main():
    parser = argparse.ArgumentParser(description="Post the failure-analysis report to a PR comment")
    parser.add_argument("--repo", default=os.environ.get("REPO_PATH", ""), help="owner/name")
    parser.add_argument("--pr", default=os.environ.get("PR_NUMBER", ""), help="PR number")
    parser.add_argument("--body-file", required=True, help="Markdown report to post")
    parser.add_argument("--token-env", default="AUTO_ROUND_BOT_TOKEN", help="Env var holding the GitHub token")
    args = parser.parse_args()

    token = os.environ.get(args.token_env, "")
    if not args.repo or "/" not in args.repo or not str(args.pr).strip():
        print("No repo/PR number available; skipping PR comment.", file=sys.stderr)
        return
    if not token:
        print(f"No token in ${args.token_env}; skipping PR comment.", file=sys.stderr)
        return
    if not os.path.isfile(args.body_file):
        print(f"Report file not found: {args.body_file}; skipping.", file=sys.stderr)
        return

    with open(args.body_file, encoding="utf-8") as f:
        body = f.read()
    if len(body) > _MAX_BODY:
        body = body[:_MAX_BODY] + "\n\n_...truncated; see the pipeline artifact for the full report._"
    body = f"{_MARKER}\n{body}"

    pr = int(args.pr)
    try:
        existing = _find_existing(args.repo, pr, token)
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
