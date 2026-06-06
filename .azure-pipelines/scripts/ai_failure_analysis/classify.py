"""Classify a CI failure into one of six categories and route handling.

Pipeline position: runs after ``merge_failure_context.py`` and before any
category-specific handling.  It combines deterministic forensic evidence
(``evidence_collectors``) and known-issue matches (``known_issue_matcher``), then
emits Azure DevOps pipeline variables so downstream steps can route handling.

Classification categories (priority order mirrors the triage flow):
  1. Known Issue      2. Environment   3. Dependency
  4. Flaky Test       5. Code Regression   6. Other
"""

import argparse
import json
import os
from pathlib import Path

import evidence_collectors
import known_issue_matcher

# Canonical category labels.
KNOWN_ISSUE = "Known Issue"
ENVIRONMENT = "Environment"
DEPENDENCY = "Dependency"
FLAKY = "Flaky Test"
CODE_REGRESSION = "Code Regression"
OTHER = "Other"

VALID_CATEGORIES = {KNOWN_ISSUE, ENVIRONMENT, DEPENDENCY, FLAKY, CODE_REGRESSION, OTHER}

# Below this confidence the result is forced to ``Other`` and the CI admin is
# pinged, to avoid mis-routing.
CONFIDENCE_THRESHOLD = 0.5

# A known-issue match at or above this confidence short-circuits to Known Issue.
KNOWN_ISSUE_AUTO_THRESHOLD = 0.6

# Handling action keys consumed by post_pr_comment.py / template gating.
HANDLING_BY_CATEGORY = {
    KNOWN_ISSUE: "comment_known_issue",
    ENVIRONMENT: "comment_environment",
    DEPENDENCY: "comment_dependency",
    FLAKY: "comment_flaky",
    CODE_REGRESSION: "analyze_regression",
    OTHER: "comment_other_notify_admin",
}


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def emit_pipeline_variable(name: str, value: str):
    """Emit an Azure DevOps logging command to set a pipeline variable.

    Variable is consumable by later steps in the same job via ``$(NAME)`` /
    ``variables['NAME']``.
    """
    safe = str(value).replace("\r", " ").replace("\n", " ")
    print(f"##vso[task.setvariable variable={name}]{safe}")


def _group_pr_relevance(evidence: dict, group_id: str) -> dict:
    for item in evidence.get("pr_relevance", {}).get("per_group", []):
        if item.get("group_id") == group_id:
            return item
    return {}


def heuristic_classification(group: dict, evidence: dict, known_matches: list[dict]) -> dict:
    """Deterministic group-level classification."""
    if known_matches and known_matches[0].get("confidence", 0.0) >= KNOWN_ISSUE_AUTO_THRESHOLD:
        return {
            "classification": KNOWN_ISSUE,
            "confidence": known_matches[0]["confidence"],
            "evidence": [f"matches known issue #{known_matches[0].get('number')}"],
            "reasoning": "Known-issue matcher returned a high-confidence ticket.",
        }

    if group.get("signature_type") == "env_signal":
        signature = group.get("signature", "")
        signal = signature.split(":", 1)[1] if ":" in signature else signature
        return {
            "classification": ENVIRONMENT,
            "confidence": 0.7,
            "evidence": [f"environment signal: {signal}"],
            "reasoning": "Group signature indicates infrastructure/environment symptoms.",
        }

    pr_group = _group_pr_relevance(evidence, group.get("group_id", ""))
    pr_global = evidence.get("pr_relevance", {})
    if pr_group.get("relevance_score", 0.0) >= 0.5 and pr_global.get("touches_source"):
        return {
            "classification": CODE_REGRESSION,
            "confidence": round(min(0.9, 0.5 + pr_group["relevance_score"] / 2), 3),
            "evidence": [
                f"group_relevance_score={pr_group.get('relevance_score')}",
                f"directly_changed_tests={pr_group.get('directly_changed_tests', [])[:5]}",
            ],
            "reasoning": "This group correlates with PR-changed source or tests.",
        }

    return {
        "classification": OTHER,
        "confidence": 0.3,
        "evidence": ["no strong deterministic signal"],
        "reasoning": "Insufficient evidence for confident classification.",
    }


def finalize_classification(raw_result: dict, known_matches: list[dict], admin_handle: str) -> dict:
    """Apply guard rails: known-issue short-circuit and confidence threshold."""
    classification = str(raw_result.get("classification", "")).strip()
    if classification not in VALID_CATEGORIES:
        classification = OTHER

    try:
        confidence = float(raw_result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    evidence = raw_result.get("evidence", []) or []
    reasoning = raw_result.get("reasoning", "")

    notify_admin = False
    matches = known_matches

    # Guard 1: strong known-issue match always wins.
    if matches and matches[0].get("confidence", 0.0) >= KNOWN_ISSUE_AUTO_THRESHOLD:
        classification = KNOWN_ISSUE
        confidence = max(confidence, matches[0]["confidence"])

    # Guard 2: low-confidence results fall back to Other and ping the admin.
    if classification != KNOWN_ISSUE and confidence < CONFIDENCE_THRESHOLD:
        classification = OTHER
        notify_admin = True
        reasoning = (
            f"Confidence {confidence} below threshold {CONFIDENCE_THRESHOLD}; "
            f"routed to Other for manual review. {reasoning}"
        ).strip()

    if classification == OTHER:
        notify_admin = True

    handling_action = HANDLING_BY_CATEGORY.get(classification, HANDLING_BY_CATEGORY[OTHER])

    return {
        "classification": classification,
        "confidence": round(confidence, 3),
        "evidence": evidence[:10],
        "reasoning": reasoning,
        "handling_action": handling_action,
        "notify_admin": notify_admin,
        "ci_admin_handle": admin_handle,
        "known_issue_matches": matches[:5],
    }


def main():
    parser = argparse.ArgumentParser(description="Classify CI failure and route handling (deterministic)")
    parser.add_argument("--failure-context", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--base-ref", default="main")
    parser.add_argument("--known-issue-label", default=known_issue_matcher.DEFAULT_LABEL)
    parser.add_argument("--admin-handle", default=os.environ.get("CI_ADMIN_HANDLE", "chensuyue"))
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    payload = load_json(args.failure_context)
    groups = payload.get("groups", [])

    evidence = evidence_collectors.collect_all_evidence(groups, project_root, args.base_ref)

    repo_path = known_issue_matcher._repo_path_from_env()
    token = os.environ.get("GITHUB_TOKEN", "")
    known_issues = known_issue_matcher.match_known_issues(
        groups, repo_path, token, label=args.known_issue_label
    )

    known_map = {
        item.get("group_id", ""): item.get("matches", [])
        for item in known_issues.get("per_group_matches", [])
    }

    per_group_results = []
    category_counts = {key: 0 for key in VALID_CATEGORIES}

    for group in groups:
        group_id = group.get("group_id", "")
        matches = known_map.get(group_id, [])
        raw_result = heuristic_classification(group, evidence, matches)
        final_group = finalize_classification(raw_result, matches, args.admin_handle)
        final_group.update(
            {
                "group_id": group_id,
                "group_signature": group.get("signature", ""),
                "group_size": len(group.get("cases", [])),
            }
        )
        per_group_results.append(final_group)
        category_counts[final_group["classification"]] = category_counts.get(final_group["classification"], 0) + 1

    has_regression = any(item.get("classification") == CODE_REGRESSION for item in per_group_results)
    summary = {
        "group_count": len(groups),
        "category_counts": {k: v for k, v in category_counts.items() if v > 0},
        "has_code_regression_group": has_regression,
    }

    failure_count = sum(len(group.get("cases", [])) for group in groups)
    final = {
        "per_group_results": per_group_results,
        "summary": summary,
        "evidence_bundle": evidence,
        "source_commit": payload.get("build", {}).get("source_commit", ""),
        "pr_number": payload.get("build", {}).get("pr_number", ""),
        "group_count": len(groups),
        "failure_count": failure_count,
        "ci_admin_handle": args.admin_handle,
    }

    write_json(args.output, final)

    emit_pipeline_variable("CLASSIFICATION", CODE_REGRESSION if has_regression else "Non-Regression")
    emit_pipeline_variable("HANDLING_ACTION", "grouped_routing")

    print(f"classify: groups={len(groups)} failures={failure_count} "
          f"regression_groups={summary['category_counts'].get(CODE_REGRESSION, 0)}")
    print(f"classify: result written to {args.output}")


if __name__ == "__main__":
    main()
