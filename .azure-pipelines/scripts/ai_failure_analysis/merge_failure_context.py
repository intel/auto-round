import argparse
import hashlib
import json
import os
import re
from pathlib import Path

ENV_SIGNAL_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "network_timeout",
        re.compile(
            r"(connection\s+timed?\s*out|read\s+timed?\s*out|max\s+retries\s+exceeded|ECONNRESET|ECONNREFUSED|ETIMEDOUT)",
            re.IGNORECASE,
        ),
    ),
    ("disk_full", re.compile(r"(no\s+space\s+left\s+on\s+device|disk\s+quota\s+exceeded|ENOSPC)", re.IGNORECASE)),
    (
        "runner_lost",
        re.compile(
            r"(lost\s+communication\s+with\s+the\s+agent|runner\s+has\s+been\s+lost|did\s+not\s+respond)", re.IGNORECASE
        ),
    ),
    (
        "out_of_memory",
        re.compile(
            r"(out\s+of\s+memory|cannot\s+allocate\s+memory|oom[\s-]?kill|std::bad_alloc|MemoryError)", re.IGNORECASE
        ),
    ),
    (
        "hf_download_error",
        re.compile(
            r"(HfHubHTTPError|huggingface\.co|couldn'?t\s+connect\s+to\s+['\"]?https?://huggingface)", re.IGNORECASE
        ),
    ),
    ("rate_limited", re.compile(r"(rate\s+limit\s+exceeded|too\s+many\s+requests|HTTP\s+429)", re.IGNORECASE)),
]

EXCEPTION_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning))\b")
ASSERT_RE = re.compile(r"(AssertionError[:\s].+)$", re.IGNORECASE | re.MULTILINE)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_failure(entry: dict, source_file: str, artifact: str) -> dict:
    return {
        "source_file": source_file,
        "artifact": artifact,
        "test_name": entry.get("test_name", ""),
        "status": entry.get("status", ""),
        "log_file": entry.get("log_file", ""),
        "duration": entry.get("duration", ""),
        "excerpt": entry.get("excerpt", ""),
        "tail": entry.get("tail", ""),
    }


def _case_text(entry: dict) -> str:
    return "\n".join(part for part in (entry.get("excerpt", ""), entry.get("tail", "")) if part)


def _normalize_text(value: str, max_len: int = 180) -> str:
    text = re.sub(r"\s+", " ", (value or "").strip())
    text = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
    text = re.sub(r"\b\d+\b", "N", text)
    return text[:max_len]


def _detect_env_signal(text: str) -> str:
    for key, pattern in ENV_SIGNAL_PATTERNS:
        if pattern.search(text):
            return key
    return ""


def _extract_exception(text: str) -> str:
    hits = EXCEPTION_RE.findall(text or "")
    return hits[-1] if hits else ""


def _extract_terminal_line(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return ""
    return _normalize_text(lines[-1])


def _extract_assertion(text: str) -> str:
    match = ASSERT_RE.search(text or "")
    if not match:
        return ""
    return _normalize_text(match.group(1))


def build_group_signature(entry: dict) -> tuple[str, str, str, str]:
    """Return (signature_type, signature, summary, evidence)."""
    text = _case_text(entry)
    env_signal = _detect_env_signal(text)
    if env_signal:
        return (
            "env_signal",
            f"env:{env_signal}",
            f"Environment signal: {env_signal}",
            _extract_terminal_line(text) or env_signal,
        )

    exception_name = _extract_exception(text)
    if exception_name:
        terminal = _extract_terminal_line(text)
        signature = f"exception:{exception_name}:{terminal}"
        return (
            "exception",
            signature,
            f"{exception_name} triggered similar failures",
            terminal or exception_name,
        )

    assertion = _extract_assertion(text)
    if assertion:
        return (
            "assertion",
            f"assertion:{assertion}",
            "Assertion-like failures share the same symptom",
            assertion,
        )

    seed = f"{entry.get('test_name', '')}|{_normalize_text(text, max_len=220)}"
    short_hash = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return (
        "fallback",
        f"fallback:{short_hash}",
        "Fallback signature built from normalized failure text",
        _extract_terminal_line(text) or _normalize_text(text, max_len=120),
    )


def get_artifact_name(context_file: Path, input_root: Path) -> str:
    rel = context_file.relative_to(input_root)
    if rel.parts:
        return rel.parts[0]
    return context_file.parent.name


def parse_artifact_attempt(artifact_name: str) -> tuple[str, str, int] | None:
    # Expected artifact format: <prefix>-<part>-<jobAttempt>
    match = re.match(r"^(?P<prefix>.+)-(?P<part>[^-]+)-(?P<attempt>\d+)$", artifact_name)
    if not match:
        return None
    return match.group("prefix"), match.group("part"), int(match.group("attempt"))


def select_latest_context_files(context_files: list[Path], input_root: Path) -> list[Path]:
    latest_by_part: dict[tuple[str, str], tuple[int, Path]] = {}
    passthrough: list[Path] = []

    for context_file in context_files:
        artifact_name = get_artifact_name(context_file, input_root)
        parsed = parse_artifact_attempt(artifact_name)
        if parsed is None:
            passthrough.append(context_file)
            continue

        prefix, part, attempt = parsed
        key = (prefix, part)
        current = latest_by_part.get(key)
        if current is None or attempt > current[0]:
            latest_by_part[key] = (attempt, context_file)

    selected = passthrough + [item[1] for item in latest_by_part.values()]
    return sorted(selected)


def merge_contexts(input_root: Path) -> list[dict]:
    all_context_files = sorted(input_root.rglob("failure_context*.json"))
    context_files = select_latest_context_files(all_context_files, input_root)
    merged_failures: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for context_file in context_files:
        artifact = get_artifact_name(context_file, input_root)
        payload = load_json(context_file)
        for entry in payload.get("failures", []):
            normalized = normalize_failure(entry, str(context_file), artifact)
            key = (normalized.get("test_name", ""), normalized.get("log_file", ""))
            if key in seen:
                continue
            seen.add(key)
            merged_failures.append(normalized)

    groups_by_signature: dict[str, dict] = {}
    for case in merged_failures:
        signature_type, signature, summary, evidence = build_group_signature(case)
        group = groups_by_signature.get(signature)
        if group is None:
            group = {
                "group_id": "",
                "signature_type": signature_type,
                "signature": signature,
                "summary": summary,
                "evidence": [],
                "test_names": [],
                "log_files": [],
                "cases": [],
            }
            groups_by_signature[signature] = group

        group["cases"].append(case)

        test_name = case.get("test_name", "")
        if test_name and test_name not in group["test_names"]:
            group["test_names"].append(test_name)

        log_file = case.get("log_file", "")
        if log_file and log_file not in group["log_files"]:
            group["log_files"].append(log_file)

        if evidence and evidence not in group["evidence"] and len(group["evidence"]) < 5:
            group["evidence"].append(evidence)

    groups = sorted(
        groups_by_signature.values(),
        key=lambda g: (-len(g.get("cases", [])), g.get("signature", "")),
    )
    for index, group in enumerate(groups, start=1):
        group["group_id"] = f"g{index:03d}"

    return groups


def main():
    parser = argparse.ArgumentParser(description="Merge failure context files from all test parts")
    parser.add_argument(
        "--input-root", required=True, type=Path, help="Root folder containing downloaded failure artifacts"
    )
    parser.add_argument("--output", required=True, type=Path, help="Merged failure context JSON path")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    groups = merge_contexts(args.input_root)
    failed_cases = sum(len(group.get("cases", [])) for group in groups)

    payload = {
        "schema_version": "1.0",
        "build": {
            "build_id": os.environ.get("BUILD_BUILDID", ""),
            "source_commit": os.environ.get(
                "SYSTEM_PULLREQUEST_SOURCECOMMITID", os.environ.get("BUILD_SOURCEVERSION", "")
            ),
            "pr_number": os.environ.get("SYSTEM_PULLREQUEST_PULLREQUESTNUMBER", ""),
        },
        "stats": {
            "failed_cases": failed_cases,
            "group_count": len(groups),
        },
        "groups": groups,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Merged {failed_cases} failures into {len(groups)} groups from {args.input_root}")
    print(f"Merged context: {args.output}")


if __name__ == "__main__":
    main()
