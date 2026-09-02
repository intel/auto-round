"""Extract and cluster CI unit-test failures from downloaded failed logs.

Input is a directory holding the failed ``unittest_*.log`` files together with the
``failures_part*.json`` metadata emitted by ``collect_result.py``. The script scans
each failed log for the well-known failure markers, derives a normalized error
signature per failing test, clusters similar signatures together and writes:

* ``clusters.json``  — structured clusters ranked by occurrence (machine-readable).
* ``decision_log.md`` — an audit trail of how each cluster was formed (matched
  keywords, member tests, merged raw log excerpts). Kept separate from the final
  report on purpose; it is intentionally verbose.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

# Markers we scan for; crash markers are matched case-insensitively.
KEYWORDS = ("== FAILURES ==", "== ERRORS ==", "Traceback", "core dumped", "Killed")
CRASH_MARKERS = ("core dumped", "killed", "segmentation fault", "aborted")

# pytest section header of underscores wrapping a test id, e.g. "____ test_foo ____".
_TEST_HEADER = re.compile(r"^_{5,}\s+(.+?)\s+_{5,}\s*$")
# A trailing exception summary line, e.g. "ValueError: shapes do not match".
_EXC_SUMMARY = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Warning|Failure|Exit)):?\s")
# Similarity threshold above which two normalized signatures share a cluster.
_SIMILARITY = 0.90


@dataclass
class ErrorRecord:
    """A single extracted failure occurrence."""

    test: str
    log: str
    signature: str
    normalized: str
    keywords: list[str]
    excerpt: str


@dataclass
class Cluster:
    """A group of similar error records."""

    id: int
    signature: str
    normalized: str
    keywords: set[str] = field(default_factory=set)
    tests: list[str] = field(default_factory=list)
    logs: set[str] = field(default_factory=set)
    records: list[ErrorRecord] = field(default_factory=list)

    @property
    def occurrences(self) -> int:
        return len(self.records)


def normalize(text: str) -> str:
    """Strip run-specific noise so equivalent errors normalize to the same string."""
    text = text.strip().lower()
    text = re.sub(r"0x[0-9a-f]+", "0xADDR", text)
    text = re.sub(r"/\S+/", "", text)  # drop directory paths, keep basenames
    text = re.sub(r"line \d+", "line N", text)
    text = re.sub(r"\d+", "N", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _matched_keywords(text: str) -> list[str]:
    low = text.lower()
    return [kw for kw in KEYWORDS if kw.lower() in low]


def _signature_from_segment(lines: list[str]) -> "str | None":
    """Pick the most specific error line from a FAILURES/traceback segment."""
    e_lines = [ln[1:].strip() for ln in lines if re.match(r"^E\s+\S", ln)]
    if e_lines:
        # Prefer the exception-summary line over pytest continuation lines
        # (e.g. "+  where 3 = quantize(...)").
        for el in e_lines:
            if _EXC_SUMMARY.match(el) or el.lower().startswith("assert"):
                return el
        return e_lines[0]
    for ln in reversed(lines):
        if _EXC_SUMMARY.match(ln.strip()):
            return ln.strip()
    return None


def _crash_signature(lines: list[str]) -> "str | None":
    for ln in lines:
        low = ln.lower()
        if any(marker in low for marker in CRASH_MARKERS):
            return ln.strip()
    return None


def _excerpt(lines: list[str], limit: int = 15) -> str:
    tail = [ln.rstrip() for ln in lines if ln.strip()][-limit:]
    return "\n".join(tail)


# Trailing pytest noise that should not pollute a test's signature/excerpt.
_SEGMENT_CUT_MARKERS = (
    "captured log",
    "captured stdout",
    "captured stderr",
    "generated xml file",
    "slowest durations",
)


def _trim_segment(lines: list[str]) -> list[str]:
    """Cut a per-test segment at the first trailing-noise marker."""
    for i, ln in enumerate(lines):
        low = ln.lower()
        if any(marker in low for marker in _SEGMENT_CUT_MARKERS):
            return lines[:i]
    return lines


def extract_records(log_path: Path) -> list[ErrorRecord]:
    """Extract error records from a failed log.

    Every log passed here is already known to be a failure (only failed logs are
    staged), so a log that yields no pytest ``FAILURES`` segment still produces a
    single record keyed by the log name (crash/segfault or unrecognized output).
    """
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"Warning: cannot read {log_path} - {e}", file=sys.stderr)
        return []

    lines = content.splitlines()
    # Drop the pytest tail (durations table + short summary); it is noise that
    # would otherwise bleed into the last test's excerpt/signature.
    for i, ln in enumerate(lines):
        if "slowest durations" in ln.lower():
            lines = lines[:i]
            break
    keywords = _matched_keywords(content)
    records: list[ErrorRecord] = []

    # Split the log into per-test segments using pytest FAILURES/ERRORS headers.
    segments: list[tuple[str, list[str]]] = []
    current_name = ""
    current: list[str] = []
    for ln in lines:
        header = _TEST_HEADER.match(ln)
        if header:
            if current:
                segments.append((current_name, current))
            current_name = header.group(1).strip()
            current = []
        else:
            current.append(ln)
    if current:
        segments.append((current_name, current))

    for name, seg in segments:
        if not name:
            continue
        seg = _trim_segment(seg)
        sig = _signature_from_segment(seg)
        if sig is None:
            continue
        records.append(
            ErrorRecord(
                test=name,
                log=log_path.name,
                signature=sig,
                normalized=normalize(sig),
                keywords=_matched_keywords("\n".join(seg)) or keywords,
                excerpt=_excerpt(seg),
            )
        )

    # Crash (segfault/OOM-kill) or unrecognized failures produce no FAILURES
    # section; emit a single record keyed by the log name so it is never dropped.
    if not records:
        crash_sig = _crash_signature(lines)
        if crash_sig is not None:
            signature, normalized = crash_sig, normalize(crash_sig)
        else:
            signature = "Unknown failure (no recognizable error signature)"
            normalized = "unknown failure"
        records.append(
            ErrorRecord(
                test=log_path.stem,
                log=log_path.name,
                signature=signature,
                normalized=normalized,
                keywords=keywords or _matched_keywords(signature),
                excerpt=_excerpt(lines),
            )
        )
    return records


def cluster_records(records: list[ErrorRecord]) -> list[Cluster]:
    """Greedily group records whose normalized signatures are near-identical."""
    clusters: list[Cluster] = []
    for record in records:
        placed = False
        for cluster in clusters:
            if cluster.normalized == record.normalized or (
                SequenceMatcher(None, cluster.normalized, record.normalized).ratio() >= _SIMILARITY
            ):
                cluster.records.append(record)
                cluster.keywords.update(record.keywords)
                cluster.logs.add(record.log)
                if record.test not in cluster.tests:
                    cluster.tests.append(record.test)
                placed = True
                break
        if not placed:
            clusters.append(
                Cluster(
                    id=0,
                    signature=record.signature,
                    normalized=record.normalized,
                    keywords=set(record.keywords),
                    tests=[record.test],
                    logs={record.log},
                    records=[record],
                )
            )
    clusters.sort(key=lambda c: c.occurrences, reverse=True)
    for idx, cluster in enumerate(clusters, start=1):
        cluster.id = idx
    return clusters


def write_clusters_json(clusters: list[Cluster], total_failed: int, output: Path) -> None:
    payload = {
        "total_failed_tests": total_failed,
        "cluster_count": len(clusters),
        "clusters": [
            {
                "id": c.id,
                "signature": c.signature,
                "normalized": c.normalized,
                "keywords": sorted(c.keywords),
                "occurrences": c.occurrences,
                "tests": c.tests,
                "logs": sorted(c.logs),
                "sample": c.records[0].excerpt if c.records else "",
            }
            for c in clusters
        ],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(clusters)} cluster(s) to {output}", file=sys.stderr)


def write_decision_log(clusters: list[Cluster], output: Path) -> None:
    lines = ["# CI Failure Clustering — Decision Log", ""]
    lines.append(f"Total clusters: {len(clusters)}")
    lines.append("")
    for c in clusters:
        lines.append(f"## Cluster {c.id} (occurrences: {c.occurrences})")
        lines.append("")
        lines.append(f"- Signature: `{c.signature}`")
        lines.append(f"- Normalized: `{c.normalized}`")
        lines.append(f"- Matched keywords: {', '.join(sorted(c.keywords)) or '(none)'}")
        lines.append(f"- Logs: {', '.join(sorted(c.logs))}")
        lines.append(f"- Affected tests ({len(c.tests)}): {', '.join(c.tests)}")
        lines.append("")
        lines.append("Merged log excerpts:")
        lines.append("")
        for record in c.records:
            lines.append(f"### {record.test} ({record.log})")
            lines.append("```")
            lines.append(record.excerpt)
            lines.append("```")
            lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote decision log to {output}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Extract and cluster CI test failures")
    parser.add_argument("--input-dir", required=True, type=Path, help="Dir with the failed unittest_*.log files")
    parser.add_argument("--clusters-json", required=True, type=Path, help="Output clusters JSON path")
    parser.add_argument("--decision-log", required=True, type=Path, help="Output decision-log markdown path")
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: input dir not found: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    log_files = sorted(args.input_dir.rglob("unittest_*.log"))

    records: list[ErrorRecord] = []
    seen_logs: set[str] = set()
    for log_file in log_files:
        if log_file.name in seen_logs:  # dedupe job-retry copies of the same log
            continue
        seen_logs.add(log_file.name)
        records.extend(extract_records(log_file))

    total_failed = len(records)
    clusters = cluster_records(records)

    args.clusters_json.parent.mkdir(parents=True, exist_ok=True)
    args.decision_log.parent.mkdir(parents=True, exist_ok=True)
    write_clusters_json(clusters, total_failed, args.clusters_json)
    write_decision_log(clusters, args.decision_log)


if __name__ == "__main__":
    main()
