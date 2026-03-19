#!/usr/bin/env python3
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fetch and summarize failed unit tests from Azure DevOps CI for a given PR.

Usage:
    python scripts/fetch_ci_failures.py <PR_NUMBER> [--save <output.md>]

Examples:
    python scripts/fetch_ci_failures.py 1570
    python scripts/fetch_ci_failures.py 1570 --save docs/findings/pr1570-failures.md
"""

import argparse
import json
import re
import subprocess
import sys
import textwrap
import urllib.request
from collections import defaultdict
from datetime import datetime

# Azure DevOps project constants
ADO_ORG = "lpot-inc"
ADO_PROJECT_ID = "b7121868-d73a-4794-90c1-23135f974d09"
ADO_API_VERSION = "7.0"
GITHUB_REPO = "intel/auto-round"


def ado_api_url(path):
    """Build an Azure DevOps REST API URL."""
    return f"https://dev.azure.com/{ADO_ORG}/{ADO_PROJECT_ID}/_apis/{path}?api-version={ADO_API_VERSION}"


def fetch_json(url):
    """Fetch a URL and return parsed JSON."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def fetch_text(url):
    """Fetch a URL and return raw text."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", errors="replace")


def get_pr_checks(pr_number):
    """Get CI check results for a PR using gh CLI."""
    result = subprocess.run(
        ["gh", "pr", "checks", str(pr_number), "--repo", GITHUB_REPO, "--json", "name,state,link"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Error: Failed to fetch PR checks: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout)


def get_pr_info(pr_number):
    """Get PR title and URL using gh CLI."""
    result = subprocess.run(
        ["gh", "pr", "view", str(pr_number), "--repo", GITHUB_REPO, "--json", "title,url,headRefName,baseRefName"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"title": f"PR #{pr_number}", "url": f"https://github.com/{GITHUB_REPO}/pull/{pr_number}"}
    return json.loads(result.stdout)


def extract_build_id(checks):
    """Extract Azure DevOps build ID from check links."""
    for check in checks:
        link = check.get("link", "")
        match = re.search(r"buildId=(\d+)", link)
        if match:
            return match.group(1)
    return None


def get_failed_ut_logs(build_id):
    """Get log IDs for failed 'Run UT' tasks from the build timeline.

    Returns a list of dicts: [{job_name, log_id, log_url}, ...]
    """
    timeline_url = ado_api_url(f"build/builds/{build_id}/timeline")
    timeline = fetch_json(timeline_url)

    records = {r["id"]: r for r in timeline.get("records", [])}
    failed_logs = []

    for r in timeline.get("records", []):
        if r.get("name") == "Run UT" and r.get("result") == "failed":
            parent = records.get(r.get("parentId"), {})
            job_name = parent.get("name", "unknown")
            log_info = r.get("log", {})
            log_id = log_info.get("id")
            log_url = log_info.get("url", ado_api_url(f"build/builds/{build_id}/logs/{log_id}"))
            failed_logs.append({"job_name": job_name, "log_id": log_id, "log_url": log_url})

    return failed_logs


def strip_timestamp(line):
    """Remove Azure DevOps timestamp prefix and trailing whitespace from a log line."""
    line = re.sub(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z\s*", "", line)
    return line.rstrip("\r\n")


def strip_ansi(text):
    """Remove ANSI escape codes."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def parse_failures_from_log(log_text):
    """Parse pytest FAILURES sections from raw log text.

    Handles multiple pytest sessions in a single log (CI runs multiple pytest invocations).

    Returns a list of dicts: [{test_name, error_line, traceback}, ...]
    """
    lines = log_text.split("\n")
    failures = []

    in_failures_section = False
    current_test = None
    current_lines = []
    error_lines = []

    def _save_current():
        if current_test:
            failures.append({
                "test_name": current_test,
                "error_line": error_lines[-1] if error_lines else "",
                "traceback": "\n".join(current_lines),
            })

    for raw_line in lines:
        line = strip_timestamp(raw_line)
        line = strip_ansi(line)

        if "= FAILURES =" in line:
            _save_current()
            in_failures_section = True
            current_test = None
            current_lines = []
            error_lines = []
            continue

        if in_failures_section:
            # New test failure header: ___ TestClass.test_method ___
            test_header = re.match(r"^_+ (.+?) _+$", line)
            if test_header:
                test_name = test_header.group(1)
                # Skip non-test headers (e.g. "coverage" report sections)
                if "coverage" in test_name.lower() or "." not in test_name:
                    continue
                _save_current()
                current_test = test_name.replace(" ", "::")
                current_lines = []
                error_lines = []
                continue

            # End of FAILURES section: "short test summary info" or final result line
            if re.match(r"^=+ short test summary info =+$", line) or re.match(
                r"^=+ \d+ failed", line
            ):
                _save_current()
                in_failures_section = False
                current_test = None
                current_lines = []
                error_lines = []
                continue

            # Skip non-failure section headers (e.g. "tests coverage")
            if re.match(r"^=+ .+ =+$", line) and "FAILURES" not in line:
                continue

            if current_test:
                current_lines.append(line)
                if line.startswith("E       ") and not line.startswith("E         "):
                    error_lines.append(line[8:].strip())

    # Handle case where log ends mid-section
    _save_current()

    # Strategy 2: If no FAILURES section found, try "FAILED test_path::..." lines
    if not failures:
        for raw_line in lines:
            line = strip_timestamp(raw_line)
            line = strip_ansi(line)
            match = re.match(r"^FAILED\s+(\S+)", line)
            if match:
                failures.append({
                    "test_name": match.group(1),
                    "error_line": "(no traceback captured)",
                    "traceback": "",
                })

    return failures


def group_failures_by_error(all_failures):
    """Group failures by their root error message for deduplication."""
    groups = defaultdict(list)
    for f in all_failures:
        # Normalize error line for grouping
        error = f["error_line"] or "(unknown error)"
        groups[error].append(f)
    return groups


def format_terminal_output(pr_info, pr_number, checks, all_failures, grouped):
    """Format the summary for terminal output."""
    lines = []
    lines.append("")
    lines.append(f"{'=' * 80}")
    lines.append(f"  CI Failure Summary for PR #{pr_number}")
    lines.append(f"  {pr_info.get('title', '')}")
    lines.append(f"  {pr_info.get('url', '')}")
    lines.append(f"{'=' * 80}")
    lines.append("")

    # CI job overview
    lines.append("CI Job Overview:")
    lines.append(f"  {'Job':<50} {'Status':<10} {'Link'}")
    lines.append(f"  {'-' * 48} {'-' * 8}  {'-' * 20}")
    for check in sorted(checks, key=lambda c: c["name"]):
        name = check["name"]
        state = check["state"]
        marker = "FAIL" if state == "FAILURE" else "PASS" if state == "SUCCESS" else state
        color = "\033[31m" if state == "FAILURE" else "\033[32m" if state == "SUCCESS" else "\033[33m"
        lines.append(f"  {name:<50} {color}{marker:<10}\033[0m")
    lines.append("")

    if not all_failures:
        lines.append("  No failed unit tests found (or unable to parse logs).")
        return "\n".join(lines)

    # Failure summary
    lines.append(f"Failed Unit Tests: {len(all_failures)} total")
    lines.append("")

    for error, tests in grouped.items():
        lines.append(f"  \033[31mError:\033[0m {error}")
        for t in tests:
            lines.append(f"    - [{t['job_name']}] {t['test_name']}")
        lines.append("")

    # Detailed tracebacks
    lines.append(f"{'-' * 80}")
    lines.append("Detailed Tracebacks:")
    lines.append(f"{'-' * 80}")
    for f in all_failures:
        lines.append("")
        lines.append(f"  \033[1m[{f['job_name']}] {f['test_name']}\033[0m")
        if f["traceback"]:
            for tb_line in f["traceback"].split("\n")[:15]:
                lines.append(f"    {tb_line}")
            total_lines = len(f["traceback"].split("\n"))
            if total_lines > 15:
                lines.append(f"    ... ({total_lines - 15} more lines)")
        else:
            lines.append(f"    {f['error_line']}")
    lines.append("")

    return "\n".join(lines)


def format_markdown_output(pr_info, pr_number, checks, all_failures, grouped, build_id):
    """Format the summary as Markdown."""
    lines = []
    lines.append(f"# CI Failure Summary: PR #{pr_number}")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"**PR:** [{pr_info.get('title', f'PR #{pr_number}')}]({pr_info.get('url', '')})")
    lines.append(f"**Branch:** `{pr_info.get('headRefName', '?')}` → `{pr_info.get('baseRefName', '?')}`")
    if build_id:
        lines.append(f"**Build:** [Azure DevOps #{build_id}]"
                      f"(https://dev.azure.com/{ADO_ORG}/{ADO_PROJECT_ID}/_build/results?buildId={build_id})")
    lines.append("")

    # CI job results table
    lines.append("## CI Job Results")
    lines.append("")
    lines.append("| Job | Status |")
    lines.append("|-----|--------|")
    for check in sorted(checks, key=lambda c: c["name"]):
        name = check["name"]
        state = check["state"]
        marker = "**FAIL**" if state == "FAILURE" else "PASS" if state == "SUCCESS" else state
        lines.append(f"| {name} | {marker} |")
    lines.append("")

    if not all_failures:
        lines.append("No failed unit tests found (or unable to parse logs).")
        return "\n".join(lines)

    # Failure summary
    lines.append(f"## Failed Unit Tests ({len(all_failures)} total)")
    lines.append("")

    for error, tests in grouped.items():
        lines.append(f"### `{error}`")
        lines.append("")
        lines.append("| Job | Test |")
        lines.append("|-----|------|")
        for t in tests:
            lines.append(f"| {t['job_name']} | `{t['test_name']}` |")
        lines.append("")

    # Detailed tracebacks
    lines.append("## Detailed Tracebacks")
    lines.append("")
    for f in all_failures:
        lines.append(f"### `{f['test_name']}` ({f['job_name']})")
        lines.append("")
        if f["traceback"]:
            lines.append("```")
            for tb_line in f["traceback"].split("\n")[:30]:
                lines.append(tb_line)
            total_lines = len(f["traceback"].split("\n"))
            if total_lines > 30:
                lines.append(f"... ({total_lines - 30} more lines)")
            lines.append("```")
        else:
            lines.append(f"`{f['error_line']}`")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and summarize failed unit tests from CI for a given PR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              %(prog)s 1570
              %(prog)s 1570 --save docs/findings/pr1570-failures.md
        """),
    )
    parser.add_argument("pr_number", type=int, help="GitHub PR number")
    parser.add_argument("--save", metavar="FILE", help="Save markdown summary to a file")
    parser.add_argument("--no-color", action="store_true", help="Disable colored terminal output")
    args = parser.parse_args()

    pr_number = args.pr_number

    # 1. Get PR info and CI checks
    print(f"Fetching CI status for PR #{pr_number}...")
    pr_info = get_pr_info(pr_number)
    checks = get_pr_checks(pr_number)

    # Filter to Unit-Test checks only
    ut_checks = [c for c in checks if "Unit-Test" in c.get("name", "")]
    failed_checks = [c for c in ut_checks if c["state"] == "FAILURE"]

    if not failed_checks:
        print(f"\nNo failed Unit-Test CI jobs found for PR #{pr_number}.")
        print("All checks:")
        for c in ut_checks:
            print(f"  {c['name']}: {c['state']}")
        return

    # 2. Extract build ID from check links
    build_id = extract_build_id(ut_checks)
    if not build_id:
        print("Error: Could not extract Azure DevOps build ID from check links.", file=sys.stderr)
        sys.exit(1)
    print(f"Azure DevOps build ID: {build_id}")

    # 3. Get failed Run UT task logs
    print("Fetching build timeline...")
    failed_logs = get_failed_ut_logs(build_id)
    if not failed_logs:
        print("Warning: No failed 'Run UT' tasks found in build timeline.", file=sys.stderr)
        print("The failure may be in setup/teardown rather than unit tests.", file=sys.stderr)

    # 4. Parse each failed log for test failures
    all_failures = []
    for fl in failed_logs:
        job_name = fl["job_name"]
        print(f"Parsing logs for {job_name} (logId={fl['log_id']})...")
        log_text = fetch_text(fl["log_url"])
        failures = parse_failures_from_log(log_text)
        for f in failures:
            f["job_name"] = job_name
        all_failures.extend(failures)
        print(f"  Found {len(failures)} failed test(s)")

    # 5. Group by error for deduplication
    grouped = group_failures_by_error(all_failures)

    # 6. Print terminal summary
    if args.no_color:
        term_output = format_terminal_output(pr_info, pr_number, ut_checks, all_failures, grouped)
        # Strip ANSI codes
        term_output = re.sub(r"\033\[[0-9;]*m", "", term_output)
    else:
        term_output = format_terminal_output(pr_info, pr_number, ut_checks, all_failures, grouped)
    print(term_output)

    # 7. Save markdown if requested
    if args.save:
        md_output = format_markdown_output(pr_info, pr_number, ut_checks, all_failures, grouped, build_id)
        import os

        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            f.write(md_output)
        print(f"Markdown summary saved to: {args.save}")


if __name__ == "__main__":
    main()
