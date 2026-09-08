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
import datetime
import json
import os
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
- For an ``AssertionError`` whose log excerpt already contains pytest's ``Full diff`` (the ``-``/``+``
  lines), that diff is usually authoritative: cross-check it against the changed files below and
  conclude directly.
- The ``PR changed files`` section below lists every file this PR touched. Inspect the relevant ones
  yourself with your ``read``/``rg`` tools on the checkout at ``{project_root}`` (and the complete raw
  failure logs under ``{log_path}``) to judge whether the change caused the failure.

## Project source root
{project_root}

## Full failure logs directory
{log_path}

## Error signature
{signature}

## Affected tests ({test_count})
{tests}

## Log excerpt
{excerpt}

## PR changed files
{pr_files}
"""


def call_backend(prompt: str, backend: str, timeout: int, model: str, trace_file: str, cluster_id) -> str:
    """Dispatch a prompt to the selected AI backend and return raw text output.

    Swap point: replace the ``copilot`` branch to change the inference provider.
    """
    if backend == "none":
        return ""
    if backend == "copilot":
        return _call_copilot_cli(prompt, timeout, model, trace_file, cluster_id)
    raise ValueError(f"unknown backend: {backend}")


def _append_trace(trace_file: str, entry: dict) -> None:
    """Append one AI-call record (JSONL) so the analysis process can be audited."""
    if not trace_file:
        return
    try:
        with open(trace_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"Warning: could not write trace file: {e}", file=sys.stderr)


def _parse_copilot_json(stdout: str) -> dict:
    """Parse the Copilot ``--output-format json`` JSONL stream.

    A single stream carries everything we need: the assistant answer, the model
    actually used, and the AI-credit / premium-request usage — so there is no need
    to shell out for stderr credits or dig through debug logs separately.
    """
    answer = ""
    model = ""
    nano_aiu = None
    premium_requests = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = event.get("type")
        data = event.get("data", {})
        if etype == "assistant.message":
            model = data.get("model") or model
            if data.get("content"):
                answer = data["content"]  # keep the last full assistant message
        elif etype == "model.call_start":
            model = data.get("model") or model
        elif etype == "session.usage_checkpoint":
            nano_aiu = data.get("totalNanoAiu", nano_aiu)
            premium_requests = data.get("totalPremiumRequests", premium_requests)
        elif etype == "result":
            usage = event.get("usage", {})
            premium_requests = usage.get("premiumRequests", premium_requests)
    return {
        "answer": answer,
        "model": model,
        "ai_credits": round(nano_aiu / 1e9, 6) if nano_aiu is not None else None,
        "premium_requests": premium_requests,
    }


def _call_copilot_cli(prompt: str, timeout: int, model: str, trace_file: str, cluster_id) -> str:
    argv = [
        "copilot",
        "-p",
        prompt,
        "--allow-tool=shell(python:*)",
        "--allow-tool=shell(rg:*)",
        "--allow-tool=read",
        "--allow-tool=write",
        # The PR diff is pre-computed into the prompt, so no GitHub API access is needed. Dropping the
        # built-in GitHub MCP server trims the tool schema re-sent on every turn (~11k tokens -> far
        # less), which is the dominant per-call cost in this multi-turn loop.
        "--disable-builtin-mcps",
        "--no-ask-user",
        "--output-format",
        "json",
    ]
    if model:
        argv += ["--model", model]
    env = dict(os.environ)
    # Let the Copilot CLI authenticate with the dedicated token.
    ai_token = os.environ.get("AI_TOKEN", "")
    if ai_token:
        env.setdefault("GITHUB_TOKEN", ai_token)
        env.setdefault("GH_TOKEN", ai_token)
    started = datetime.datetime.now(datetime.timezone.utc)
    returncode = None
    stderr = ""
    stdout = ""
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env, check=False)
        returncode = proc.returncode
        stderr = proc.stderr.strip()
        stdout = proc.stdout
        if returncode != 0:
            print(f"Warning: Copilot CLI exited {returncode}: {stderr}", file=sys.stderr)
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"Warning: Copilot CLI call failed: {e}", file=sys.stderr)
        stderr = str(e)
    ended = datetime.datetime.now(datetime.timezone.utc)
    parsed = _parse_copilot_json(stdout)
    raw = parsed["answer"]
    _append_trace(
        trace_file,
        {
            "cluster_id": cluster_id,
            "timestamp": started.isoformat(),
            "duration_s": round((ended - started).total_seconds(), 3),
            "model": parsed["model"] or model or "(unknown)",
            "ai_credits": parsed["ai_credits"],
            "premium_requests": parsed["premium_requests"],
            "returncode": returncode,
            "prompt": prompt,
            "response": raw,
            "transcript": stdout,  # raw JSONL stream: full step-by-step execution trace
        },
    )
    print(
        f"AI call cluster={cluster_id} "
        f"model={parsed['model'] or model or '(unknown)'} "
        f"duration_s={round((ended - started).total_seconds(), 3)} "
        f"ai_credits={parsed['ai_credits']} "
        f"premium_requests={parsed['premium_requests']} "
        f"returncode={returncode}",
        file=sys.stderr,
    )
    return raw


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


def _run_git(project_root: str, args: "list[str]", timeout: int = 120) -> "subprocess.CompletedProcess | None":
    """Run a git command in ``project_root``; return the completed process or None on failure."""
    try:
        return subprocess.run(
            ["git", "-C", project_root or ".", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"Warning: git {' '.join(args)} failed: {e}", file=sys.stderr)
        return None


_unshallowed = False


def _ensure_full_history(project_root: str) -> None:
    """Unshallow the checkout once so merge-base / parent commits are available.

    CI checkouts are usually shallow, which makes ``git diff <sha>^1...<sha>^2`` fail. Doing it here,
    once, keeps that cost out of the analysis loop.
    """
    global _unshallowed
    if _unshallowed:
        return
    _unshallowed = True
    res = _run_git(project_root, ["rev-parse", "--is-shallow-repository"])
    if res is not None and res.stdout.strip() == "true":
        print("Unshallowing repository for PR diff computation...", file=sys.stderr)
        _run_git(project_root, ["fetch", "--quiet", "--unshallow", "origin"], timeout=600)


def compute_pr_changed_files(pr_sha: str, project_root: str, max_chars: int) -> str:
    """Return the ``--name-status`` list of files this PR changed (same for every cluster).

    The three-dot range ``sha^1...sha^2`` is the merge commit's change (merge-base..PR-head). We hand
    the model only the file list and let it read the actual source itself, so the PR context stays
    small and cluster-independent. Returns an empty string when it cannot be produced (e.g. no SHA).
    """
    if not pr_sha:
        return ""
    _ensure_full_history(project_root)
    argv = ["diff", f"{pr_sha}^1...{pr_sha}^2", "--name-status"]
    res = _run_git(project_root, argv)
    if res is None or res.returncode != 0:
        # Retry once after unshallowing in case the first attempt raced a shallow checkout.
        _run_git(project_root, ["fetch", "--quiet", "--unshallow", "origin"], timeout=600)
        res = _run_git(project_root, argv)
    if res is None or res.returncode != 0:
        return ""
    files = res.stdout.strip()
    if len(files) > max_chars:
        files = files[:max_chars] + "\n... (list truncated) ..."
    return files


def _save_pr_changed_files(output_path: str, pr_files: str) -> None:
    """Persist the PR changed-file list next to the analysis output for easy log inspection."""
    if not pr_files:
        return
    out_dir = os.path.dirname(os.path.abspath(output_path)) if output_path else "."
    path = os.path.join(out_dir, "pr_changed_files.txt")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(pr_files + "\n")
    except OSError as e:
        print(f"Warning: could not write PR changed-file list {path}: {e}", file=sys.stderr)


def build_prompt(cluster: dict, pr_files: str, max_excerpt: int, project_root: str, log_path: str) -> str:
    return _PROMPT_TEMPLATE.format(
        signature=cluster.get("signature", ""),
        test_count=len(cluster.get("tests", [])),
        tests=", ".join(cluster.get("tests", [])) or "(unknown)",
        excerpt=(cluster.get("sample", "") or "")[:max_excerpt],
        pr_files=pr_files or "(empty; inspect the checkout with your read/rg tools if needed)",
        project_root=project_root or "(not provided)",
        log_path=log_path or "(not provided)",
    )


def analyze(cluster: dict, pr_files: str, args) -> dict:
    prompt = build_prompt(cluster, pr_files, args.max_excerpt_chars, args.project_root, args.log_dir)
    raw = call_backend(prompt, args.backend, args.timeout, args.model, args.trace_file, cluster.get("id"))
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
    parser.add_argument(
        "--pr-sha",
        default="",
        help="Merge commit SHA of the PR build; used to pre-compute the PR diff embedded in prompts",
    )
    parser.add_argument("--top", type=int, default=3, help="Number of top unknown clusters to analyze")
    parser.add_argument("--backend", choices=["copilot", "none"], default="none", help="Inference backend")
    parser.add_argument(
        "--enable-ai-analysis",
        default="true",
        help="Set to a falsy value (false/0/no) to force the no-model 'none' backend",
    )
    parser.add_argument("--timeout", type=int, default=300, help="Per-call timeout in seconds")
    parser.add_argument("--model", default="", help="Copilot model to use; empty uses the CLI default")
    parser.add_argument("--trace-file", default="", help="JSONL file logging each AI call for auditing")
    parser.add_argument("--project-root", default="", help="Repository checkout root the AI may inspect")
    parser.add_argument("--log-dir", default="", help="Directory holding the full raw failure logs")
    parser.add_argument("--max-excerpt-chars", type=int, default=4000)
    parser.add_argument(
        "--max-diff-chars",
        type=int,
        default=20000,
        help="Max characters of the PR changed-file list embedded in each prompt",
    )
    args = parser.parse_args()

    # The AI switch overrides the backend so a single pipeline step can toggle it.
    if str(args.enable_ai_analysis).strip().lower() not in ("1", "true", "yes", "on"):
        args.backend = "none"

    with open(args.clusters_json, encoding="utf-8") as f:
        data = json.load(f)

    # The PR changed-file list is the same for every cluster, so compute and persist it once; the
    # model reads the actual source itself instead of running git inside a shallow checkout.
    pr_sha = (args.pr_sha or "").strip()
    pr_files = compute_pr_changed_files(pr_sha, args.project_root, args.max_diff_chars)

    unknown = [c for c in data.get("clusters", []) if not c.get("known")]
    unknown.sort(key=lambda c: c.get("occurrences", 0), reverse=True)
    selected = unknown[: args.top]

    analyses = [analyze(cluster, pr_files, args) for cluster in selected]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"backend": args.backend, "analyses": analyses}, f, indent=2)
    print(f"Analyzed {len(analyses)} unknown cluster(s) with backend '  {args.backend}'", file=sys.stderr)


if __name__ == "__main__":
    main()
