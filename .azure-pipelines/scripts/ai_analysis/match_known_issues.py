"""Match clustered CI failures against known-issue tickets on GitHub.

Fetches open issues labeled ``CI-known-issue`` and fuzzily matches each failure
cluster against the issue title, full body and the failing test names. Clusters
with a confident match are flagged as *known*; the rest stay *unknown* and are
handed to the AI-analysis step.

Issue fetching uses the GitHub REST API with a token read from the environment
(``AUTO_ROUND_BOT_TOKEN`` by default). If no token is available or the request
fails, every cluster is conservatively treated as unknown.
"""

import argparse
import json
import os
import re
import sys
from difflib import SequenceMatcher
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Minimum combined score for a cluster to be considered a known issue.
_MATCH_THRESHOLD = 0.55
# Signature-similarity above which the signature alone counts as a strong signal.
_SIGNATURE_STRONG = 0.6


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"0x[0-9a-f]+", "0xADDR", text)
    text = re.sub(r"\d+", "N", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_known_issues(repo: str, label: str, token: str) -> list[dict]:
    """Return open issues carrying ``label`` (title + body), following pagination."""
    if "/" not in repo:
        raise ValueError(f"repo must be 'owner/name', got: {repo}")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    issues: list[dict] = []
    page = 1
    while True:
        url = f"https://api.github.com/repos/{repo}/issues" f"?labels={label}&state=open&per_page=100&page={page}"
        req = Request(url, headers=headers)
        with urlopen(req, timeout=20) as resp:
            batch = json.loads(resp.read().decode("utf-8"))
        # The issues endpoint also returns PRs; drop them.
        batch = [it for it in batch if "pull_request" not in it]
        issues.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return issues


def _token_overlap(sig_norm: str, issue_norm: str) -> float:
    tokens = [t for t in re.split(r"\W+", sig_norm) if len(t) >= 3]
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in issue_norm)
    return hits / len(tokens)


def score_issue(cluster: dict, issue: dict) -> "tuple[float, list[str]]":
    """Score how well a cluster matches an issue; return (score, reasons)."""
    reasons: list[str] = []
    sig_norm = normalize(cluster.get("signature", ""))
    title_norm = normalize(issue.get("title", ""))
    body_norm = normalize(issue.get("body") or "")
    issue_norm = f"{title_norm}\n{body_norm}"

    # 1) Failing test name mentioned verbatim in the issue — strongest signal.
    test_hit = next((t for t in cluster.get("tests", []) if t and t.lower() in issue_norm), None)
    if test_hit:
        reasons.append(f"test-name:{test_hit}")

    # 2) Error signature similarity / containment against the whole issue.
    sig_ratio = _token_overlap(sig_norm, issue_norm)
    if sig_norm and sig_norm in issue_norm:
        sig_ratio = max(sig_ratio, 1.0)
    if sig_ratio >= _SIGNATURE_STRONG:
        reasons.append(f"signature:{sig_ratio:.2f}")

    # 3) Fuzzy similarity between signature and issue title.
    title_ratio = SequenceMatcher(None, sig_norm, title_norm).ratio() if sig_norm else 0.0
    if title_ratio >= 0.5:
        reasons.append(f"title:{title_ratio:.2f}")

    score = max(
        1.0 if test_hit else 0.0,
        sig_ratio,
        title_ratio,
    )
    return score, reasons


def match_clusters(clusters: list[dict], issues: list[dict]) -> None:
    """Annotate each cluster in place with its best known-issue match (if any)."""
    for cluster in clusters:
        best_score = 0.0
        best_issue = None
        best_reasons: list[str] = []
        for issue in issues:
            score, reasons = score_issue(cluster, issue)
            if reasons and score > best_score:
                best_score = score
                best_issue = issue
                best_reasons = reasons
        if best_issue is not None and best_score >= _MATCH_THRESHOLD:
            cluster["known"] = True
            cluster["issue"] = {
                "number": best_issue.get("number"),
                "title": best_issue.get("title"),
                "url": best_issue.get("html_url"),
                "matched_by": best_reasons,
                "score": round(best_score, 3),
            }
        else:
            cluster["known"] = False
            cluster["issue"] = None


def main():
    parser = argparse.ArgumentParser(description="Match failure clusters against known-issue tickets")
    parser.add_argument("--clusters-json", required=True, help="Clusters JSON from analyze_failures.py")
    parser.add_argument("--output", required=True, help="Annotated clusters JSON output path")
    parser.add_argument("--repo", default=os.environ.get("REPO_PATH", ""), help="owner/name")
    parser.add_argument("--label", default="CI-known-issue", help="Known-issue label")
    parser.add_argument("--token-env", default="AUTO_ROUND_BOT_TOKEN", help="Env var holding the GitHub token")
    args = parser.parse_args()

    with open(args.clusters_json, encoding="utf-8") as f:
        data = json.load(f)
    clusters = data.get("clusters", [])

    token = os.environ.get(args.token_env, "")
    issues: list[dict] = []
    if args.repo:
        try:
            issues = fetch_known_issues(args.repo, args.label, token)
            print(f"Fetched {len(issues)} '{args.label}' issue(s) from {args.repo}", file=sys.stderr)
        except (HTTPError, URLError, ValueError, json.JSONDecodeError) as e:
            print(f"Warning: could not fetch known issues ({e}); treating all as unknown", file=sys.stderr)
    else:
        print("Warning: no repo provided; treating all clusters as unknown", file=sys.stderr)

    match_clusters(clusters, issues)

    known = sum(1 for c in clusters if c.get("known"))
    data["known_count"] = known
    data["unknown_count"] = len(clusters) - known
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Matched {known}/{len(clusters)} cluster(s) to known issues", file=sys.stderr)


if __name__ == "__main__":
    main()
