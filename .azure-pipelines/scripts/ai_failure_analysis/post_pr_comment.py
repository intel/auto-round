import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def github_request(method: str, url: str, token: str, payload: dict | None = None):
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    req = Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
    )

    with urlopen(req) as resp:
        data = resp.read().decode("utf-8")
        return json.loads(data) if data else {}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_repo_path() -> str:
    uri = os.environ.get("BUILD_REPOSITORY_URI", "")
    if uri.startswith("https://github.com/"):
        return uri.replace("https://github.com/", "").removesuffix(".git")
    return os.environ.get("REPO_PATH", "")


def find_existing_comment(comments: list[dict], marker: str) -> dict | None:
    for comment in comments:
        body = comment.get("body", "")
        if marker in body:
            return comment
    return None


def build_comment_body(analysis: dict, report_text: str, marker: str) -> str:
    root_cause = analysis.get("root_cause", "N/A")
    confidence = analysis.get("confidence", "unknown")
    patch_path = analysis.get("patch_path", "")
    external_ok = analysis.get("external_copilot_ok", False)

    patch_hint = "Patch is empty in this run."
    if patch_path and Path(patch_path).exists() and Path(patch_path).stat().st_size > 0:
        patch_hint = f"Patch generated at `{patch_path}` and published as pipeline artifact."

    lines = [
        marker,
        "## Copilot CI Failure Analysis",
        "",
        f"- Root cause: {root_cause}",
        f"- Confidence: {confidence}",
        f"- Copilot command executed successfully: {external_ok}",
        f"- {patch_hint}",
        "",
        "### Report",
        report_text[:12000] if report_text else "No detailed report generated.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Post or update PR comment for AI analysis")
    parser.add_argument("--analysis-result", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "")
    pr_number = os.environ.get("SYSTEM_PULLREQUEST_PULLREQUESTNUMBER", "")
    repo_path = parse_repo_path()

    if not token:
        print("GITHUB_TOKEN is missing. Skip posting comment.")
        return
    if not pr_number or not repo_path:
        print("PR context missing. Skip posting comment.")
        return

    analysis = load_json(args.analysis_result)
    report_text = load_text(args.report)
    source_commit = analysis.get("source_commit", "unknown")
    marker = f"<!-- auto-round-ci-copilot-analysis:{source_commit} -->"

    comments_url = f"https://api.github.com/repos/{repo_path}/issues/{pr_number}/comments"

    try:
        comments = github_request("GET", comments_url, token)
        existing = find_existing_comment(comments if isinstance(comments, list) else [], marker)
        body = build_comment_body(analysis, report_text, marker)

        if existing:
            update_url = f"https://api.github.com/repos/{repo_path}/issues/comments/{existing['id']}"
            github_request("PATCH", update_url, token, payload={"body": body})
            print(f"Updated existing analysis comment #{existing['id']}")
        else:
            github_request("POST", comments_url, token, payload={"body": body})
            print("Posted new analysis comment")

    except HTTPError as e:
        print(f"Failed to post PR comment: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
