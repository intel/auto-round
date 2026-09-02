"""AI analysis of the most frequent unknown CI failures.

Ranks the *unknown* failure clusters by occurrence, takes the top N (default 3)
and asks an AI backend to classify each failure (Code Regression / Environment /
Dependency / Test Case) and propose a fix. When the model is not confident it is
instructed to omit the fix and instead give investigation directions.

The backend is intentionally pluggable behind ``call_backend`` so the current
GitHub Copilot CLI implementation can be swapped later. With ``--backend none``
the script emits the structured prompts only (no model call), which supports the
"generate data, human runs Copilot Chat" fallback.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys

_PROMPT_TEMPLATE = """You are analyzing a failing CI unit test for the AutoRound project (post-training \
quantization for LLMs/VLMs). Determine the failure type and, only when confident, propose a fix.

Respond with ONLY a single JSON object, no prose, using exactly these keys:
{{
  "category": "Code Regression | Environment Issue | Dependency Issue | Test Case Issue | Unknown",
  "confidence": "high | medium | low",
  "root_cause": "concise explanation of the likely cause",
  "suggested_fix": "concrete fix description, or empty string if confidence is low",
  "patch": "a unified diff if you are confident, otherwise empty string",
  "directions": "if confidence is low, list concrete investigation directions; else empty string"
}}

Guidance:
- Use the PR diff to judge whether this is a code regression introduced by the change.
- If the error looks environmental/dependency-related and unrelated to the diff, say so.
- If uncertain, set confidence to "low", leave suggested_fix and patch empty, and fill directions.

## Error signature
{signature}

## Affected tests ({test_count})
{tests}

## Log excerpt
{excerpt}

## PR diff (may be truncated)
{diff}
"""


def call_backend(prompt: str, backend: str, copilot_cmd: str, timeout: int) -> str:
    """Dispatch a prompt to the selected AI backend and return raw text output.

    Swap point: replace the ``copilot`` branch to change the inference provider.
    """
    if backend == "none":
        return ""
    if backend == "copilot":
        return _call_copilot_cli(prompt, copilot_cmd, timeout)
    raise ValueError(f"unknown backend: {backend}")


def _call_copilot_cli(prompt: str, copilot_cmd: str, timeout: int) -> str:
    argv = shlex.split(copilot_cmd) + [prompt]
    env = dict(os.environ)
    # Let the Copilot CLI authenticate with the dedicated token.
    ai_token = os.environ.get("AI_TOKEN", "")
    if ai_token:
        env.setdefault("GITHUB_TOKEN", ai_token)
        env.setdefault("GH_TOKEN", ai_token)
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"Warning: Copilot CLI call failed: {e}", file=sys.stderr)
        return ""
    if proc.returncode != 0:
        print(f"Warning: Copilot CLI exited {proc.returncode}: {proc.stderr.strip()}", file=sys.stderr)
    return proc.stdout.strip()


def parse_model_json(text: str) -> "dict | None":
    """Extract the first JSON object from model output; tolerant of surrounding prose."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def build_prompt(cluster: dict, diff: str, max_excerpt: int, max_diff: int) -> str:
    return _PROMPT_TEMPLATE.format(
        signature=cluster.get("signature", ""),
        test_count=len(cluster.get("tests", [])),
        tests=", ".join(cluster.get("tests", [])) or "(unknown)",
        excerpt=(cluster.get("sample", "") or "")[:max_excerpt],
        diff=(diff or "(no diff available)")[:max_diff],
    )


def analyze(cluster: dict, diff: str, args) -> dict:
    prompt = build_prompt(cluster, diff, args.max_excerpt_chars, args.max_diff_chars)
    raw = call_backend(prompt, args.backend, args.copilot_cmd, args.timeout)
    parsed = parse_model_json(raw)
    result = {
        "cluster_id": cluster.get("id"),
        "signature": cluster.get("signature"),
        "occurrences": cluster.get("occurrences"),
        "tests": cluster.get("tests", []),
    }
    if parsed:
        # Low-confidence answers must not carry a fix/patch (requirement 3.3).
        if str(parsed.get("confidence", "")).lower() == "low":
            parsed["suggested_fix"] = ""
            parsed["patch"] = ""
        result.update(
            {
                "category": parsed.get("category", "Unknown"),
                "confidence": parsed.get("confidence", "low"),
                "root_cause": parsed.get("root_cause", ""),
                "suggested_fix": parsed.get("suggested_fix", ""),
                "patch": parsed.get("patch", ""),
                "directions": parsed.get("directions", ""),
            }
        )
    else:
        result.update(
            {
                "category": "Unknown",
                "confidence": "low",
                "root_cause": "",
                "suggested_fix": "",
                "patch": "",
                "directions": "",
                "prompt": prompt if args.backend == "none" else "",
                "raw": raw,
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description="AI-analyze the top unknown failure clusters")
    parser.add_argument("--clusters-json", required=True, help="Annotated clusters JSON")
    parser.add_argument("--output", required=True, help="AI analysis JSON output path")
    parser.add_argument("--pr-diff-file", help="File containing the PR diff for context")
    parser.add_argument("--top", type=int, default=3, help="Number of top unknown clusters to analyze")
    parser.add_argument("--backend", choices=["copilot", "none"], default="none", help="Inference backend")
    parser.add_argument(
        "--enable-ai-analysis",
        default="true",
        help="Set to a falsy value (false/0/no) to force the no-model 'none' backend",
    )
    parser.add_argument("--copilot-cmd", default="copilot -p", help="Copilot CLI base command")
    parser.add_argument("--timeout", type=int, default=300, help="Per-call timeout in seconds")
    parser.add_argument("--max-excerpt-chars", type=int, default=4000)
    parser.add_argument("--max-diff-chars", type=int, default=12000)
    args = parser.parse_args()

    # The AI switch overrides the backend so a single pipeline step can toggle it.
    if str(args.enable_ai_analysis).strip().lower() not in ("1", "true", "yes", "on"):
        args.backend = "none"

    with open(args.clusters_json, encoding="utf-8") as f:
        data = json.load(f)

    diff = ""
    if args.pr_diff_file and os.path.isfile(args.pr_diff_file):
        with open(args.pr_diff_file, encoding="utf-8", errors="replace") as f:
            diff = f.read()

    unknown = [c for c in data.get("clusters", []) if not c.get("known")]
    unknown.sort(key=lambda c: c.get("occurrences", 0), reverse=True)
    selected = unknown[: args.top]

    analyses = [analyze(cluster, diff, args) for cluster in selected]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"backend": args.backend, "analyses": analyses}, f, indent=2)
    print(f"Analyzed {len(analyses)} unknown cluster(s) with backend '{args.backend}'", file=sys.stderr)


if __name__ == "__main__":
    main()
