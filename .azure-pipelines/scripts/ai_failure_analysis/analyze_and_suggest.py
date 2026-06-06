import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def load_text(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_token_from_env() -> str:
    return (
        os.environ.get("COPILOT_GITHUB_TOKEN", "")
        or os.environ.get("GH_TOKEN", "")
        or os.environ.get("GITHUB_TOKEN", "")
    )


def run_cmd(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> tuple[int, str, str]:
    result = subprocess.run(cmd, cwd=cwd, env=env, check=False, capture_output=True, text=True)
    return result.returncode, result.stdout or "", result.stderr or ""


def truncate_text(value: str, limit: int = 6000) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def all_failures_from_payload(payload: dict) -> list[dict]:
    """Return flattened failure cases from either failures[] or groups[]."""
    if payload.get("failures"):
        return payload.get("failures", [])
    flattened = []
    for group in payload.get("groups", []):
        flattened.extend(group.get("cases", []))
    return flattened


def collect_failed_log_paths(payload: dict, failed_logs_root: Path | None, project_root: Path) -> list[str]:
    failures = all_failures_from_payload(payload)
    requested = [
        (entry.get("log_file") or "").strip()
        for entry in failures
        if (entry.get("log_file") or "").strip()
    ]

    candidate_roots = []
    if failed_logs_root:
        candidate_roots.append(failed_logs_root)

    # Default location used by CI failure packaging.
    candidate_roots.append(project_root / "log_dir" / "failure_logs")

    found: list[str] = []
    seen: set[str] = set()
    for root in candidate_roots:
        if not root.exists():
            continue

        for name in requested:
            direct = root / name
            matches: list[Path] = []
            if direct.exists():
                matches = [direct]
            else:
                matches = list(root.rglob(name))

            for match in matches:
                resolved = str(match.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                found.append(resolved)

    return sorted(found)


def generate_pr_patch(project_root: Path, output_path: Path, base_ref: str = "main") -> tuple[bool, str]:
    diff_specs = [
        f"origin/{base_ref}...HEAD",
        f"{base_ref}...HEAD",
        "HEAD~1..HEAD",
    ]

    for spec in diff_specs:
        code, stdout, _stderr = run_cmd(["git", "diff", "--binary", "--minimal", "--no-color", spec], cwd=project_root)
        if code != 0:
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_text(output_path, stdout)
        return True, f"Generated PR patch using git diff spec: {spec}"

    write_text(output_path, "")
    return False, "Failed to generate PR patch via git diff"


def extract_json_from_response(text: str) -> dict | None:
    if not text.strip():
        return None

    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    bare = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if bare:
        try:
            return json.loads(bare.group(1))
        except json.JSONDecodeError:
            return None

    return None


def build_agent_prompt(
    merged_failure_context: Path,
    failed_log_paths_file: Path,
    pr_patch_file: Path,
    project_root: Path,
) -> str:
    return "\n".join(
        [
            "You are a non-interactive coding agent for CI failure analysis.",
            "Your task MUST follow these steps in order:",
            "1) Error classification: summarize failure types and group failed tests by root symptom.",
            "2) Root-cause analysis: reason about likely root cause with PR changes and project code context.",
            "3) Fix suggestion: provide a minimal and safe code change proposal.",
            "4) Static checks: run at least one static check command and report command and result.",
            "",
            "Use these context sources:",
            f"- merged_failure_context_json: {merged_failure_context}",
            f"- failed_log_paths_list: {failed_log_paths_file}",
            f"- pr_code_changes_patch: {pr_patch_file}",
            f"- project_source_root: {project_root}",
            "",
            "Tooling policy:",
            "- Non-interactive only; do not request user input.",
            "- Prefer read-only inspection of files and git history.",
            "- Prioritize merged failure context first; only scan full failed logs when evidence is insufficient.",
            "- Keep suggestions minimal and low risk.",
            "",
            "Output STRICT JSON only with this schema:",
            "{",
            '  "error_classification": [',
            "    {",
            '      "category": "string",',
            '      "evidence": ["string"],',
            '      "tests": ["string"]',
            "    }",
            "  ],",
            '  "root_cause": "string",',
            '  "confidence": "low|medium|high",',
            '  "suggestion": "string",',
            '  "patch": "unified diff string or empty",',
            '  "static_checks": [',
            "    {",
            '      "command": "string",',
            '      "status": "pass|fail|not_run",',
            '      "output": "string"',
            "    }",
            "  ],",
            '  "selected_model": "string (optional if available)"',
            "}",
        ]
    )


def run_copilot_cli(prompt: str, project_root: Path, github_token: str) -> tuple[bool, str, dict | None, str]:
    env = os.environ.copy()
    if github_token:
        env["COPILOT_GITHUB_TOKEN"] = github_token

    cmd = [
        "copilot",
        "-p",
        prompt,
        "--allow-tool=shell(git:*)",
        "--allow-tool=shell(python:*)",
        "--allow-tool=shell(rg:*)",
        "--allow-tool=read",
        "--allow-tool=write",
        "--no-ask-user",
    ]
    code, stdout, stderr = run_cmd(cmd, cwd=project_root, env=env)
    raw_text = (stdout or "").strip()
    parsed = extract_json_from_response(raw_text)

    message = f"copilot cli exit={code}"
    if stderr.strip():
        message += f"; stderr={truncate_text(stderr.strip(), 1200)}"

    if code == 0 and parsed:
        return True, message, parsed, raw_text
    return False, message, parsed, raw_text


def fallback_analysis(payload: dict) -> dict:
    failures = all_failures_from_payload(payload)
    if not failures:
        return {
            "error_classification": [],
            "root_cause": "No failed cases were found in merged failure context.",
            "confidence": "low",
            "suggestion": "No patch generated.",
            "patch": "",
            "static_checks": [
                {
                    "command": "not_run",
                    "status": "not_run",
                    "output": "Copilot CLI analysis was not available; no agent static checks executed.",
                }
            ],
        }

    first = failures[0]
    excerpt = (first.get("excerpt", "") or first.get("tail", ""))[:1200]
    return {
        "error_classification": [
            {
                "category": "test_failure",
                "evidence": ["first failure excerpt"],
                "tests": [first.get("test_name", "")],
            }
        ],
        "root_cause": "Likely regression in the changed code path exercised by failed unit tests. Inspect the first traceback for precise failure location.",
        "confidence": "medium",
        "suggestion": "Review traceback, apply a minimal fix, and validate with targeted tests.",
        "patch": "",
        "static_checks": [
            {
                "command": "not_run",
                "status": "not_run",
                "output": "Copilot CLI analysis was not available; no agent static checks executed.",
            }
        ],
        "first_failure_excerpt": excerpt,
    }


def run_local_static_checks(project_root: Path, pr_patch_path: Path) -> list[dict]:
    checks: list[dict] = []
    patch_text = load_text(pr_patch_path)
    changed_files = sorted(set(re.findall(r"^\+\+\+ b/(.*\.py)$", patch_text, flags=re.MULTILINE)))

    for rel_path in changed_files[:20]:
        file_path = project_root / rel_path
        if not file_path.exists():
            continue
        code, stdout, stderr = run_cmd([sys.executable, "-m", "py_compile", str(file_path)], cwd=project_root)
        checks.append(
            {
                "command": f"{sys.executable} -m py_compile {rel_path}",
                "status": "pass" if code == 0 else "fail",
                "output": truncate_text((stdout + "\n" + stderr).strip(), 2000),
            }
        )

    if not checks:
        checks.append(
            {
                "command": f"{sys.executable} -m py_compile .azure-pipelines/scripts/ai_failure_analysis/analyze_and_suggest.py",
                "status": "pass",
                "output": "No changed Python files detected in PR patch for local static checks.",
            }
        )
    return checks


def write_skipped_result(
    output_dir: Path,
    classification: str,
    confidence,
    reason: str,
    project_root: Path,
) -> None:
    """Emit a minimal result when deep regression analysis is not applicable.

    Non-regression categories (Known Issue / Environment / Dependency / Flaky /
    Other) are handled by classification + comment routing, so the expensive
    Copilot patch flow is skipped here.
    """
    result_path = output_dir / "analysis_result.json"
    report_path = output_dir / "ai_failure_report.md"
    patch_path = output_dir / "suggested_fix.patch"

    write_text(patch_path, "")
    write_text(
        report_path,
        "\n".join(
            [
                "# Copilot Failure Analysis",
                "",
                f"- Classification: {classification}",
                f"- Confidence: {confidence}",
                "- Deep regression analysis: skipped",
                f"- Reason: {reason}",
            ]
        ),
    )
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "success": True,
                "skipped": True,
                "classification": classification,
                "confidence": confidence,
                "reason": reason,
                "patch_path": str(patch_path),
                "report_path": str(report_path),
                "project_root": str(project_root),
                "source_commit": os.environ.get(
                    "SYSTEM_PULLREQUEST_SOURCECOMMITID", os.environ.get("BUILD_SOURCEVERSION", "")
                ),
                "pr_number": os.environ.get("SYSTEM_PULLREQUEST_PULLREQUESTNUMBER", ""),
            },
            f,
            indent=2,
        )
    print(f"analyze_and_suggest: classification={classification}; skipped deep analysis ({reason}).")


def main():
    parser = argparse.ArgumentParser(description="Analyze merged failure context and generate fix artifacts")
    parser.add_argument("--failure-context", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--failed-logs-root", type=Path, default=None)
    parser.add_argument("--base-ref", default="main")
    parser.add_argument(
        "--classification-result",
        type=Path,
        required=True,
        help="classification_result.json from classify.py; required for regression-group-only analysis.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = load_json(args.failure_context)
    project_root = args.project_root.resolve()
    token = get_token_from_env()

    classification_info = load_json(args.classification_result)
    regression_group_ids: list[str] = []
    per_group = classification_info.get("per_group_results", [])
    regression_group_ids = [
        item.get("group_id", "")
        for item in per_group
        if item.get("classification") == "Code Regression"
    ]

    summary = classification_info.get("summary", {})
    has_regression_group = bool(summary.get("has_code_regression_group", False)) or bool(regression_group_ids)
    if not has_regression_group:
        write_skipped_result(
            args.output_dir,
            "Non-Regression",
            "",
            "no regression groups found in grouped classification result",
            project_root,
        )
        return

    analysis_payload = payload
    groups = payload.get("groups", [])
    selected_groups = [g for g in groups if g.get("group_id", "") in set(regression_group_ids)]
    if not selected_groups:
        write_skipped_result(
            args.output_dir,
            "Non-Regression",
            "",
            "classification lists regression groups, but none were found in failure context",
            project_root,
        )
        return

    analysis_payload = {
        "groups": selected_groups,
        "build": payload.get("build", {}),
        "stats": {
            "group_count": len(selected_groups),
            "failed_cases": sum(len(g.get("cases", [])) for g in selected_groups),
        },
    }
    context_for_prompt = args.output_dir / "regression_groups_context.json"
    with open(context_for_prompt, "w", encoding="utf-8") as f:
        json.dump(analysis_payload, f, indent=2)

    failed_log_paths = collect_failed_log_paths(analysis_payload, args.failed_logs_root, project_root)
    failed_log_paths_file = args.output_dir / "failed_log_paths.txt"
    write_text(failed_log_paths_file, "\n".join(failed_log_paths) + ("\n" if failed_log_paths else ""))

    pr_patch_path = args.output_dir / "pr_code_changes.patch"
    patch_ok, patch_message = generate_pr_patch(project_root, pr_patch_path, base_ref=args.base_ref)

    prompt = build_agent_prompt(
        merged_failure_context=context_for_prompt,
        failed_log_paths_file=failed_log_paths_file,
        pr_patch_file=pr_patch_path,
        project_root=project_root,
    )
    prompt_path = args.output_dir / "copilot_prompt.txt"
    write_text(prompt_path, prompt)

    cli_ok, cli_message, cli_result, cli_raw = run_copilot_cli(prompt, project_root, token)

    fallback = fallback_analysis(analysis_payload)
    analysis = fallback.copy()
    if cli_ok and cli_result:
        analysis.update(
            {
                "error_classification": cli_result.get("error_classification", analysis.get("error_classification", [])),
                "root_cause": cli_result.get("root_cause", analysis.get("root_cause", "")),
                "confidence": cli_result.get("confidence", analysis.get("confidence", "medium")),
                "suggestion": cli_result.get("suggestion", analysis.get("suggestion", "")),
                "patch": cli_result.get("patch", analysis.get("patch", "")),
                "static_checks": cli_result.get("static_checks", analysis.get("static_checks", [])),
            }
        )
        if cli_result.get("selected_model"):
            analysis["selected_model"] = cli_result["selected_model"]
    elif cli_raw:
        analysis["raw_response"] = truncate_text(cli_raw, 6000)

    analysis["local_static_checks"] = run_local_static_checks(project_root, pr_patch_path)

    patch_path = args.output_dir / "suggested_fix.patch"
    report_path = args.output_dir / "ai_failure_report.md"
    result_path = args.output_dir / "analysis_result.json"

    if not patch_path.exists():
        write_text(patch_path, analysis.get("patch", ""))

    report_lines = [
        "# Copilot Failure Analysis",
        "",
        f"- Model: auto (resolved: {analysis.get('selected_model', 'unknown')})",
        f"- Copilot CLI status: {'success' if cli_ok else 'fallback'}",
        f"- Copilot CLI message: {cli_message}",
        "- External Copilot command executed: no",
        "- External command status: disabled",
        f"- PR patch generated: {'yes' if patch_ok else 'no'} ({patch_message})",
        f"- Project source root: {project_root}",
        f"- Failed logs provided: {len(failed_log_paths)}",
        "",
        "## Error Classification",
    ]

    classes = analysis.get("error_classification", [])
    if classes:
        for item in classes:
            category = item.get("category", "unknown")
            tests = ", ".join(item.get("tests", [])[:10])
            evidence = "; ".join(item.get("evidence", [])[:3])
            report_lines.append(f"- {category}: tests=[{tests}] evidence={evidence}")
    else:
        report_lines.append("- No structured classification returned.")

    report_lines.extend([
        "",
        "## Root Cause",
        analysis.get("root_cause", "No root cause available."),
        "",
        "## Suggested Fix",
        analysis.get("suggestion", "No suggestion available."),
        "",
        "## Static Checks (Agent)",
    ])

    for check in analysis.get("static_checks", [])[:20]:
        report_lines.append(f"- [{check.get('status', 'not_run')}] {check.get('command', 'unknown')}")

    report_lines.extend([
        "",
        "## Static Checks (Local)",
    ])
    for check in analysis.get("local_static_checks", [])[:20]:
        report_lines.append(f"- [{check.get('status', 'not_run')}] {check.get('command', 'unknown')}")

    report_lines.append("")

    excerpt = analysis.get("first_failure_excerpt", "")
    if excerpt:
        report_lines.extend([
            "## First Failure Excerpt",
            "```text",
            excerpt,
            "```",
            "",
        ])

    write_text(report_path, "\n".join(report_lines))

    result_payload = {
        "success": cli_ok,
        "skipped": False,
        "classification": classification_info.get("classification", "Code Regression"),
        "summary": classification_info.get("summary", {}),
        "regression_group_ids": regression_group_ids,
        "analyzed_group_count": len(regression_group_ids),
        "copilot_cli_ok": cli_ok,
        "copilot_cli_message": cli_message,
        "external_copilot_ok": False,
        "external_message": "disabled",
        "model": "auto",
        "selected_model": analysis.get("selected_model", ""),
        "error_classification": analysis.get("error_classification", []),
        "root_cause": analysis.get("root_cause", ""),
        "confidence": analysis.get("confidence", ""),
        "suggestion": analysis.get("suggestion", ""),
        "static_checks": analysis.get("static_checks", []),
        "local_static_checks": analysis.get("local_static_checks", []),
        "prompt_path": str(prompt_path),
        "failed_log_paths_file": str(failed_log_paths_file),
        "pr_patch_reference": str(pr_patch_path),
        "report_path": str(report_path),
        "patch_path": str(patch_path),
        "project_root": str(project_root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": os.environ.get("SYSTEM_PULLREQUEST_SOURCECOMMITID", os.environ.get("BUILD_SOURCEVERSION", "")),
        "pr_number": os.environ.get("SYSTEM_PULLREQUEST_PULLREQUESTNUMBER", ""),
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    print(f"Analysis report: {report_path}")
    print(f"Suggested patch: {patch_path}")
    print(f"Analysis result: {result_path}")


if __name__ == "__main__":
    main()
