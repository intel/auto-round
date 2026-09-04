"""Merge clustered failures and AI analyses into one collapsible markdown report.

The report is designed to be pasted as a PR comment after human review. Most
content is wrapped in ``<details>`` blocks to keep it compact; only the header
summary is always visible.
"""

import argparse
import json
import os


def _inline(text: str, limit: int = 160) -> str:
    """Flatten a signature to a single, table-safe line."""
    text = (text or "").replace("|", "\\|").replace("\n", " ").strip()
    return text[:limit]


def _load(path: str) -> dict:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_header(clusters_data: dict, args) -> list[str]:
    total_failed = clusters_data.get("total_failed_tests", 0)
    cluster_count = clusters_data.get("cluster_count", len(clusters_data.get("clusters", [])))
    known = clusters_data.get("known_count", 0)
    unknown = clusters_data.get("unknown_count", cluster_count - known)
    lines = [
        f"# CI Failure Analysis Report ({args.pipeline or 'N/A'})",
        "",
        f"- **Commit:** {args.commit or 'N/A'}",
    ]
    if args.run_url:
        lines.append(f"- **Run:** {args.run_url}")
    lines.append(
        f"- **Totals:** {total_failed} failed test(s), {cluster_count} cluster(s) " f"({known} known / {unknown} new)"
    )
    lines.append("")
    return lines


def build_known_section(clusters: list[dict]) -> list[str]:
    known = [c for c in clusters if c.get("known") and c.get("issue")]
    if not known:
        return []
    lines = [
        f"<details><summary><b>Known issues ({len(known)})</b></summary>",
        "",
        "| Cluster | Issue | Matched by | Occurrences | Sample log |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in known:
        issue = c["issue"]
        num = issue.get("number")
        url = issue.get("url") or ""
        issue_cell = f"[#{num}]({url})" if url else f"#{num}"
        matched_by = ", ".join(issue.get("matched_by", [])) or "-"
        sample_log = ", ".join(c.get("logs", [])) or "-"
        lines.append(
            f"| {c.get('id')} | {issue_cell} | {_inline(matched_by, 80)} | "
            f"{c.get('occurrences')} | {_inline(sample_log, 60)} |"
        )
    lines.extend(["", "</details>", ""])
    return lines


def build_unknown_section(clusters: list[dict], analyses_by_id: dict) -> list[str]:
    unknown = [c for c in clusters if not c.get("known")]
    unknown.sort(key=lambda c: c.get("occurrences", 0), reverse=True)
    if not unknown:
        return []
    lines = [f"## New CI issues ({len(unknown)}) — ranked by frequency", ""]

    for rank, c in enumerate(unknown, start=1):
        sig = _inline(c.get("signature", ""))
        occ = c.get("occurrences", 0)
        lines.append(f"### {rank}. `{sig}` — {occ} occurrence(s)")
        lines.append("")
        lines.append(f"<details><summary>Details for cluster {c.get('id')}</summary>")
        lines.append("")
        lines.append(f"- **Affected tests ({len(c.get('tests', []))}):** {', '.join(c.get('tests', [])) or '-'}")
        lines.append(f"- **Logs:** {', '.join(c.get('logs', [])) or '-'}")
        lines.append(f"- **Matched keywords:** {', '.join(c.get('keywords', [])) or '-'}")

        analysis = analyses_by_id.get(c.get("id"))
        if analysis:
            lines.append("")
            lines.append("**AI analysis**")
            lines.append("")
            lines.append(f"- **Category:** {analysis.get('category', 'Unknown')}")
            lines.append(f"- **Confidence:** {analysis.get('confidence', 'low')}")
            if analysis.get("root_cause"):
                lines.append(f"- **Root cause:** {analysis['root_cause']}")
            if analysis.get("suggested_fix"):
                lines.append(f"- **Suggested fix:** {analysis['suggested_fix']}")
            if analysis.get("patch"):
                lines.append("")
                lines.append("```diff")
                lines.append(analysis["patch"])
                lines.append("```")
            if analysis.get("directions"):
                lines.append(f"- **Investigation directions (low confidence):** {analysis['directions']}")
        else:
            lines.append("")
            lines.append("_Not in the top-N AI-analyzed set._")

        lines.append("")
        lines.append("<details><summary>Sample log excerpt</summary>")
        lines.append("")
        lines.append("```")
        lines.append((c.get("sample", "") or "").strip())
        lines.append("```")
        lines.append("")
        lines.append("</details>")
        lines.append("")
        lines.append("</details>")
        lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser(description="Merge clusters + AI analyses into a markdown report")
    parser.add_argument("--clusters-json", required=True, help="Annotated clusters JSON")
    parser.add_argument("--ai-json", help="AI analysis JSON")
    parser.add_argument("--output", required=True, help="Output markdown path")
    parser.add_argument("--pipeline", default=os.environ.get("BUILD_DEFINITIONNAME", ""))
    parser.add_argument("--pr", default=os.environ.get("PR_NUMBER", ""))
    parser.add_argument("--commit", default=os.environ.get("CURRENT_SHA", ""))
    parser.add_argument("--run-url", default=os.environ.get("RUN_URL", ""))
    args = parser.parse_args()

    clusters_data = _load(args.clusters_json)
    clusters = clusters_data.get("clusters", [])
    ai_data = _load(args.ai_json)
    analyses_by_id = {a.get("cluster_id"): a for a in ai_data.get("analyses", [])}

    lines: list[str] = []
    lines += build_header(clusters_data, args)
    lines += build_known_section(clusters)
    lines += build_unknown_section(clusters, analyses_by_id)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
