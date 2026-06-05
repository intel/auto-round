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


def heuristic_classification(evidence: dict, known_issues: dict) -> dict:
    """Deterministic fallback when Copilot is unavailable.

    Mirrors the priority order so behaviour is predictable without the agent.
    """
    matches = known_issues.get("matches", [])
    if matches and matches[0].get("confidence", 0.0) >= KNOWN_ISSUE_AUTO_THRESHOLD:
        return {
            "classification": KNOWN_ISSUE,
            "confidence": matches[0]["confidence"],
            "evidence": [f"matches known issue #{matches[0].get('number')}"],
            "reasoning": "Known-issue matcher returned a high-confidence ticket.",
        }

    env = evidence.get("environment", {})
    if env.get("has_environment_signal"):
        signals = list(env.get("signals", {}).keys())
        return {
            "classification": ENVIRONMENT,
            "confidence": 0.7,
            "evidence": [f"environment signal: {s}" for s in signals][:5],
            "reasoning": "Deterministic environment signals were detected in logs.",
        }

    pr = evidence.get("pr_relevance", {})
    if pr.get("relevance_score", 0.0) >= 0.5 and pr.get("touches_source"):
        return {
            "classification": CODE_REGRESSION,
            "confidence": round(min(0.9, 0.5 + pr["relevance_score"] / 2), 3),
            "evidence": [
                f"relevance_score={pr.get('relevance_score')}",
                f"directly_changed_tests={pr.get('directly_changed_tests', [])[:5]}",
            ],
            "reasoning": "Failed tests correlate with PR-changed source files.",
        }

    return {
        "classification": OTHER,
        "confidence": 0.3,
        "evidence": ["no strong deterministic signal"],
        "reasoning": "Insufficient evidence for confident classification.",
    }


def finalize_classification(raw_result: dict, known_issues: dict, admin_handle: str) -> dict:
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
    matches = known_issues.get("matches", [])

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
    failures = payload.get("failures", [])

    evidence = evidence_collectors.collect_all_evidence(failures, project_root, args.base_ref)

    repo_path = known_issue_matcher._repo_path_from_env()
    token = os.environ.get("GITHUB_TOKEN", "")
    known_issues = known_issue_matcher.match_known_issues(
        failures, repo_path, token, label=args.known_issue_label
    )

    raw_result = heuristic_classification(evidence, known_issues)

    final = finalize_classification(raw_result, known_issues, args.admin_handle)
    final.update(
        {
            "used_copilot": False,
            "evidence_bundle": evidence,
            "source_commit": payload.get("build", {}).get("source_commit", ""),
            "pr_number": payload.get("build", {}).get("pr_number", ""),
            "failure_count": len(failures),
        }
    )

    write_json(args.output, final)

    emit_pipeline_variable("CLASSIFICATION", final["classification"])
    emit_pipeline_variable("HANDLING_ACTION", final["handling_action"])

    print(f"classify: classification={final['classification']} "
          f"confidence={final['confidence']} action={final['handling_action']} "
            f"notify_admin={final['notify_admin']}")
    print(f"classify: result written to {args.output}")


if __name__ == "__main__":
    main()
