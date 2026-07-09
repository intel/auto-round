"""Log analyzer for test results with summary generation."""

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


class TestStatus(Enum):
    """Test execution status."""

    PASSED = auto()
    FAILED = auto()
    NO_TESTS = auto()


@dataclass(frozen=True)
class TestResult:
    """Represents a single test result."""

    name: str
    status: TestStatus
    filename: str
    duration: str


class LogAnalyzer:
    """Analyzes test log files and generates summary reports."""

    TIME_PATTERN = re.compile(r"in\s+([\d.]+)s")
    PREFIX_PATTERNS = (
        "unittest_cuda_vlm_",
        "unittest_cuda_vllm_",
        "unittest_cuda_sglang_",
        "unittest_cuda_llmc_",
        "unittest_cuda_",
        "unittest_",
    )

    # pytest test logic failures: test ran but assertion/expectation failed
    FAILURE_MARKERS = (
        "FAILED",
        "== FAILURES ==",
        " failures",
        "AssertionError",
    )

    # runtime/system errors: process crashed, setup/teardown failed, or unhandled exception
    ERROR_MARKERS = (
        "Aborted",
        "Killed",
        "Segmentation fault",
        "core dumped",
        "Error:",
        "ERROR:",
        "== ERRORS ==",
        " errors:",
        "Exception:",
        "Traceback ",
        "Illegal instruction",
    )

    PASS_MARKER = " passed"

    SKIP_MARKERS = (
        " deselected",
        " skipped",
    )

    def __init__(self, log_dir: Path, log_pattern: str = "*.log"):
        self.log_dir = Path(log_dir)
        self.log_pattern = log_pattern

    def analyze_all(self) -> list[TestResult]:
        """Analyze all matching log files and return sorted results."""
        search_path = self.log_dir / self.log_pattern
        print(f"Searching: {search_path}", file=sys.stderr)

        log_files = sorted(self.log_dir.glob(self.log_pattern))
        print(f"Found {len(log_files)} files", file=sys.stderr)

        results = []
        for log_file in log_files:
            if log_file.is_file():
                result = self._analyze_single(log_file)
                results.append(result)
        return results

    def _analyze_single(self, log_file: Path) -> TestResult:
        content = self._read_log(log_file)
        return TestResult(
            name=self._extract_test_name(log_file),
            status=self._determine_status(content),
            filename=log_file.name,
            duration=self._extract_duration(content),
        )

    def _read_log(self, path: Path) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except OSError as e:
            print(f"Warning: Cannot read {path} - {e}", file=sys.stderr)
            return ""

    def _extract_test_name(self, path: Path) -> str:
        name = path.stem
        for prefix in self.PREFIX_PATTERNS:
            if name.startswith(prefix):
                return name[len(prefix) :]
        return name

    def _determine_status(self, content: str) -> TestStatus:
        """Determine test status from log content."""
        if any(marker in content for marker in self.FAILURE_MARKERS):
            return TestStatus.FAILED
        if any(marker in content for marker in self.ERROR_MARKERS):
            return TestStatus.FAILED
        if self.PASS_MARKER in content:
            return TestStatus.PASSED
        if any(marker in content for marker in self.SKIP_MARKERS):
            return TestStatus.NO_TESTS
        return TestStatus.FAILED

    def _extract_duration(self, content: str) -> str:
        last_match = None
        for match in self.TIME_PATTERN.finditer(content):
            last_match = match

        if last_match:
            total_seconds = int(float(last_match.group(1)))
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return "00:00:00"


class ReportGenerator:
    """Generates formatted test summary reports."""

    WIDTHS = {"name": 35, "status": 10, "file": 50, "time": 10}
    SEPARATOR = "=" * 120

    def __init__(self, output_path: Path):
        self.output_path = Path(output_path)

    def generate(self, test_type: str, results: list[TestResult]) -> None:
        """Generate and APPEND summary report to file."""
        if not results:
            print("No results to write", file=sys.stderr)
            return

        stats = self._calculate_stats(results)

        lines = [
            "",
            self.SEPARATOR,
            f"Test Results Summary - {test_type}",
            self.SEPARATOR,
            self._format_header(),
            self._format_subheader(),
            *(self._format_row(r) for r in results),
            self.SEPARATOR,
            f"Total: {stats['total']}, Passed: {stats['passed']}, "
            f"Failed: {stats['failed']}, Skipped: {stats['skipped']}",
            self.SEPARATOR,
            "",
        ]

        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"Summary appended to: {self.output_path.absolute()}", file=sys.stderr)

    def _calculate_stats(self, results: list[TestResult]) -> dict[str, int]:
        total = len(results)
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status == TestStatus.FAILED)
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": total - passed - failed,
        }

    def _format_header(self) -> str:
        return (
            f"{'Test Case':<{self.WIDTHS['name']}} "
            f"{'Result':<{self.WIDTHS['status']}} "
            f"{'Log File':<{self.WIDTHS['file']}} "
            f"{'Time':<{self.WIDTHS['time']}}"
        )

    def _format_subheader(self) -> str:
        return (
            f"{'-' * 10:<{self.WIDTHS['name']}} "
            f"{'-' * 6:<{self.WIDTHS['status']}} "
            f"{'-' * 8:<{self.WIDTHS['file']}} "
            f"{'-' * 4:<{self.WIDTHS['time']}}"
        )

    def _format_row(self, result: TestResult) -> str:
        return (
            f"{result.name:<{self.WIDTHS['name']}} "
            f"{result.status.name:<{self.WIDTHS['status']}} "
            f"{result.filename:<{self.WIDTHS['file']}} "
            f"{result.duration:<{self.WIDTHS['time']}}"
        )


class FailureContextWriter:
    """Extract compact failure context for downstream AI analysis."""

    TRACEBACK_MARKERS = ("Traceback", "== FAILURES ==", "== ERRORS ==", "core dumped", "Killed")

    def __init__(self, log_dir: Path, max_lines: int = 200):
        self.log_dir = Path(log_dir)
        self.max_lines = max_lines

    def write(
        self,
        output_path: Path,
        results: list[TestResult],
        test_type: str,
        summary_log: Path | None = None,
        failure_log_dir: Path | None = None,
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        failed_results = [r for r in results if r.status == TestStatus.FAILED]
        failures = [self._to_failure_entry(result) for result in failed_results]

        payload = {
            "schema_version": "1.0",
            "test_type": test_type,
            "build": {
                "build_id": os.environ.get("BUILD_BUILDID", ""),
                "build_number": os.environ.get("BUILD_BUILDNUMBER", ""),
                "source_commit": os.environ.get(
                    "SYSTEM_PULLREQUEST_SOURCECOMMITID", os.environ.get("BUILD_SOURCEVERSION", "")
                ),
                "pr_number": os.environ.get("SYSTEM_PULLREQUEST_PULLREQUESTNUMBER", ""),
            },
            "stats": {
                "total": len(results),
                "failed": len(failed_results),
            },
            "failures": failures,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"Failure context written to: {output_path.absolute()}", file=sys.stderr)

        target_log_dir = Path(failure_log_dir) if failure_log_dir else output_path.parent / "failure_logs_dir"
        self._collect_failure_logs(target_log_dir, failed_results, output_path, summary_log)

    def _to_failure_entry(self, result: TestResult) -> dict:
        log_path = self.log_dir / result.filename
        content = self._read_file(log_path)
        lines = content.splitlines()
        excerpt = self._extract_excerpt(lines)
        tail = "\n".join(lines[-self.max_lines :]) if lines else ""

        return {
            "test_name": result.name,
            "status": result.status.name,
            "log_file": result.filename,
            "duration": result.duration,
            "excerpt": excerpt,
            "tail": tail,
        }

    def _read_file(self, path: Path) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except OSError:
            return ""

    def _extract_excerpt(self, lines: list[str]) -> str:
        if not lines:
            return ""

        selected = []
        for marker in self.TRACEBACK_MARKERS:
            for idx, line in enumerate(lines):
                if marker in line:
                    if marker == "Traceback":
                        start = max(0, idx - 10)
                        max_end = min(len(lines), idx + 80)
                        failed_end = None
                        for j in range(idx, max_end):
                            if "FAILED" in lines[j]:
                                failed_end = j + 1
                                break
                        end = min(max_end, failed_end) if failed_end is not None else max_end
                    else:
                        start = max(0, idx - 10)
                        end = min(len(lines), idx + 80)
                    selected = lines[start:end]
                    break
            if selected:
                break

        if not selected:
            selected = lines[-self.max_lines :]

        return "\n".join(selected[: self.max_lines])

    def _collect_failure_logs(
        self,
        target_dir: Path,
        failed_results: list[TestResult],
        context_path: Path,
        summary_log: Path | None,
    ) -> None:
        if not failed_results:
            return

        target_dir.mkdir(parents=True, exist_ok=True)

        if summary_log and Path(summary_log).exists():
            shutil.copy2(summary_log, target_dir / Path(summary_log).name)

        for result in failed_results:
            src_log = self.log_dir / result.filename
            if src_log.exists():
                shutil.copy2(src_log, target_dir / result.filename)

        if context_path.exists():
            shutil.copy2(context_path, target_dir / context_path.name)


def main():
    parser = argparse.ArgumentParser(description="Analyze test logs and generate summary")
    parser.add_argument("--test-type", required=True, help="Type of tests")
    parser.add_argument("--log-dir", required=True, type=Path, help="Directory with logs")
    parser.add_argument("--summary-log", required=True, type=Path, help="Output file")
    parser.add_argument("--log-pattern", default="*.log", help="Glob pattern")
    parser.add_argument("--failure-context-file", type=Path, help="Optional output file for failure context JSON")
    parser.add_argument("--failure-context-max-lines", type=int, default=200, help="Max lines per failed case")
    parser.add_argument("--failure-log-dir", type=Path, help="Optional output folder for failed logs package")

    args = parser.parse_args()

    if not args.log_dir.exists():
        print(f"Error: Directory not found: {args.log_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        analyzer = LogAnalyzer(args.log_dir, args.log_pattern)
        results = analyzer.analyze_all()

        reporter = ReportGenerator(args.summary_log)
        reporter.generate(args.test_type, results)

        stats = {
            "total": len(results),
            "passed": sum(1 for r in results if r.status == TestStatus.PASSED),
            "failed": sum(1 for r in results if r.status == TestStatus.FAILED),
        }

        if args.failure_context_file and stats['failed'] > 0:
            context_writer = FailureContextWriter(args.log_dir, max_lines=args.failure_context_max_lines)
            context_writer.write(
                args.failure_context_file,
                results,
                test_type=args.test_type,
                summary_log=args.summary_log,
                failure_log_dir=args.failure_log_dir,
            )

        print(f"Done: {stats['total']} tests, {stats['passed']} passed, {stats['failed']} failed")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
