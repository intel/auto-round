import argparse
import json
import os
import re
from pathlib import Path


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

    return merged_failures


def main():
    parser = argparse.ArgumentParser(description="Merge failure context files from all test parts")
    parser.add_argument("--input-root", required=True, type=Path, help="Root folder containing downloaded failure artifacts")
    parser.add_argument("--output", required=True, type=Path, help="Merged failure context JSON path")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    failures = merge_contexts(args.input_root)

    payload = {
        "schema_version": "1.0",
        "build": {
            "build_id": os.environ.get("BUILD_BUILDID", ""),
            "source_commit": os.environ.get("SYSTEM_PULLREQUEST_SOURCECOMMITID", os.environ.get("BUILD_SOURCEVERSION", "")),
            "pr_number": os.environ.get("SYSTEM_PULLREQUEST_PULLREQUESTNUMBER", ""),
        },
        "stats": {
            "failed_cases": len(failures),
        },
        "failures": failures,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Merged {len(failures)} failures from {args.input_root}")
    print(f"Merged context: {args.output}")


if __name__ == "__main__":
    main()
