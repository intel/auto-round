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


def _admin_mention(classification: dict) -> str:
    handle = (classification.get("ci_admin_handle") or "").strip().lstrip("@")
    if classification.get("notify_admin") and handle:
        return f"@{handle} please take a look."
    return ""


def _known_issue_lines(classification: dict) -> list[str]:
    matches = classification.get("known_issue_matches", []) or []
    if not matches:
        return ["No matching known issue was found."]
    lines = ["Matched known issue(s):"]
    for match in matches[:5]:
        number = match.get("number", "?")
        title = match.get("title", "")
        url = match.get("url", "")
        conf = match.get("confidence", 0)
        lines.append(f"- [#{number}]({url}) {title} (confidence {conf})")
    return lines


def _evidence_lines(classification: dict) -> list[str]:
    evidence = classification.get("evidence", []) or []
    if not evidence:
        return []
    lines = ["", "### Evidence"]
    lines.extend(f"- {item}" for item in evidence[:8])
    return lines


def build_category_section(classification: dict, analysis: dict) -> list[str]:
    """Build the category-specific body section of the comment."""
    category = classification.get("classification", "Other")
    confidence = classification.get("confidence", "unknown")
    reasoning = classification.get("reasoning", "")
    admin = _admin_mention(classification)

    header = [
        f"- Classification: **{category}**",
        f"- Confidence: {confidence}",
    ]
    if reasoning:
        header.append(f"- Reasoning: {reasoning}")

    body: list[str] = []
    if category == "Known Issue":
        body.append("This failure matches a tracked known issue and can likely be skipped.")
        body.extend(_known_issue_lines(classification))
        if admin:
            body.append(f"A CI admin can help merge this PR manually. {admin}")
    elif category == "Environment":
        body.append(
            "This looks like an environment/infrastructure issue and is likely "
            "not caused by the PR code changes."
        )
    elif category == "Dependency":
        body.append(
            "This may be caused by a dependency version change. "
            "Dependency baseline diffing is not fully enabled yet (Phase 2); "
            "please review recent dependency updates."
        )
        if admin:
            body.append(admin)
    elif category == "Flaky Test":
        body.append(
            "This test appears unstable (flaky). Consider checking the test for "
            "non-determinism or ordering assumptions."
        )
        if admin:
            body.append(admin)
    elif category == "Code Regression":
        root_cause = analysis.get("root_cause", "N/A")
        suggestion = analysis.get("suggestion", "")
        patch_path = analysis.get("patch_path", "")
        body.append("This failure is likely caused by the PR code changes.")
        body.append(f"- Root cause: {root_cause}")
        if suggestion:
            body.append(f"- Suggested fix: {suggestion}")
        if patch_path and Path(patch_path).exists() and Path(patch_path).stat().st_size > 0:
            body.append(
                f"- A suggested patch was generated (`{patch_path}`) and published "
                "as a pipeline artifact. Review before applying; it is NOT auto-applied."
            )
        else:
            body.append("- No patch was generated in this run.")
    else:  # Other
        body.append(
            "This failure could not be confidently classified and needs manual review."
        )
        if admin:
            body.append(admin)

    return header + [""] + body


def build_comment_body(classification: dict, analysis: dict, report_text: str, marker: str) -> str:
    lines = [
        marker,
        "## Copilot CI Failure Analysis",
        "",
    ]
    lines.extend(build_category_section(classification, analysis))
    lines.extend(_evidence_lines(classification))
    if report_text:
        lines.extend(
            [
                "",
                "<details><summary>Detailed report</summary>",
                "",
                report_text[:12000],
                "",
                "</details>",
            ]
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Post or update PR comment for AI analysis")
    parser.add_argument("--classification-result", required=True, type=Path)
    parser.add_argument("--analysis-result", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
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

    classification = load_json(args.classification_result)
    analysis = load_json(args.analysis_result) if args.analysis_result and args.analysis_result.exists() else {}
    report_text = load_text(args.report) if args.report else ""
    source_commit = classification.get("source_commit") or analysis.get("source_commit", "unknown")
    marker = f"<!-- auto-round-ci-copilot-analysis:{source_commit} -->"

    comments_url = f"https://api.github.com/repos/{repo_path}/issues/{pr_number}/comments"

    try:
        comments = github_request("GET", comments_url, token)
        existing = find_existing_comment(comments if isinstance(comments, list) else [], marker)
        body = build_comment_body(classification, analysis, report_text, marker)

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
