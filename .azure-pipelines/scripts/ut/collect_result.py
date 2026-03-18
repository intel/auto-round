"""Log analyzer for test results with summary generation."""

import argparse
import re
import sys
from collections import deque
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
    PREFIX_PATTERNS = ("unittest_cuda_vlm_", "unittest_cuda_")

    FAILURE_MARKERS = (
        "FAILED",
        "== FAILURES ==",
        " failures",
        "Killed",
        "AssertionError",
        "Error:",
    )

    ERROR_MARKERS = (
        "ERROR",
        "== ERRORS ==",
        " errors",
        "Exception",
        "Traceback",
    )

    PASS_MARKER = " passed"

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

    def _read_log(self, path: Path, max_lines: int = 100) -> str:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return "".join(deque(f, maxlen=max_lines))
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
        return TestStatus.NO_TESTS

    def _extract_duration(self, content: str) -> str:
        if match := self.TIME_PATTERN.search(content):
            total_seconds = int(float(match.group(1)))
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return "00:00:00"


class ReportGenerator:
    """Generates formatted test summary reports."""

    WIDTHS = {"name": 30, "status": 10, "file": 50, "time": 10}
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


def main():
    parser = argparse.ArgumentParser(description="Analyze test logs and generate summary")
    parser.add_argument("--test-type", required=True, help="Type of tests")
    parser.add_argument("--log-dir", required=True, type=Path, help="Directory with logs")
    parser.add_argument("--summary-log", required=True, type=Path, help="Output file")
    parser.add_argument("--log-pattern", default="*.log", help="Glob pattern")

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
        print(f"Done: {stats['total']} tests, {stats['passed']} passed, {stats['failed']} failed")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
