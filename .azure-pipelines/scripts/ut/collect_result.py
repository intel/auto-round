"""JUnit XML analyzer for test results with summary generation.

Each pytest invocation is expected to emit a JUnit XML report (via
``pytest --junitxml=<name>.xml``) alongside its ``<name>.log`` file. This
script enumerates the log files (which are always produced by ``tee``, even
when the process crashes), locates the sibling XML report and derives the
test status from the structured ``<testsuite>`` attributes instead of parsing
free-form log text.

If the XML report is missing or cannot be parsed (e.g. the process was killed
or segfaulted before pytest could write it), the corresponding test is marked
as FAILED.
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


class TestStatus(Enum):
    """Test execution status."""

    PASSED = auto()
    FAILED = auto()
    NO_TESTS = auto()


@dataclass(frozen=True)
class TestCounts:
    """Aggregated test counts for a single test file."""

    passed: int = 0
    failed: int = 0
    skipped: int = 0
    total: int = 0

    def format(self) -> str:
        return (
            f"passed: {self.passed}, failed: {self.failed}, "
            f"skipped: {self.skipped}, total: {self.total}"
        )


@dataclass(frozen=True)
class TestResult:
    """Represents a single test result."""

    name: str
    status: TestStatus
    filename: str
    duration: str
    counts: TestCounts


class XmlAnalyzer:
    """Analyzes JUnit XML reports and generates summary results."""

    PREFIX_PATTERNS = (
        "unittest_cuda_vlm_",
        "unittest_cuda_vllm_",
        "unittest_cuda_sglang_",
        "unittest_cuda_llmc_",
        "unittest_cuda_",
        "unittest_",
    )

    def __init__(self, log_dir: Path, log_pattern: str = "*.log"):
        self.log_dir = Path(log_dir)
        self.log_pattern = log_pattern

    def analyze_all(self) -> list[TestResult]:
        """Analyze all matching reports and return results sorted by log name."""
        search_path = self.log_dir / self.log_pattern
        print(f"Searching: {search_path}", file=sys.stderr)

        log_files = sorted(self.log_dir.glob(self.log_pattern))
        print(f"Found {len(log_files)} files", file=sys.stderr)

        results = []
        for log_file in log_files:
            if log_file.is_file():
                results.append(self._analyze_single(log_file))
        return results

    def _analyze_single(self, log_file: Path) -> TestResult:
        xml_file = log_file.with_suffix(".xml")
        counts = self._parse_xml(xml_file)
        duration = self._parse_duration(xml_file)
        return TestResult(
            name=self._extract_test_name(log_file),
            status=self._determine_status(counts, xml_file),
            filename=log_file.name,
            duration=duration,
            counts=counts if counts is not None else TestCounts(),
        )

    def _extract_test_name(self, path: Path) -> str:
        name = path.stem
        for prefix in self.PREFIX_PATTERNS:
            if name.startswith(prefix):
                return name[len(prefix) :]
        return name

    def _iter_testsuites(self, xml_file: Path):
        root = ET.parse(xml_file).getroot()
        if root.tag == "testsuite":
            yield root
        else:  # <testsuites> wrapper
            yield from root.iter("testsuite")

    def _parse_xml(self, xml_file: Path) -> "TestCounts | None":
        """Return aggregated counts, or ``None`` if the report is unusable."""
        if not xml_file.is_file():
            return None
        try:
            total = failures = errors = skipped = 0
            for suite in self._iter_testsuites(xml_file):
                total += int(suite.get("tests", 0))
                failures += int(suite.get("failures", 0))
                errors += int(suite.get("errors", 0))
                skipped += int(suite.get("skipped", 0))
        except (ET.ParseError, ValueError, OSError) as e:
            print(f"Warning: cannot parse {xml_file} - {e}", file=sys.stderr)
            return None

        failed = failures + errors
        passed = max(total - failed - skipped, 0)
        return TestCounts(passed=passed, failed=failed, skipped=skipped, total=total)

    def _parse_duration(self, xml_file: Path) -> str:
        if not xml_file.is_file():
            return "00:00:00"
        try:
            seconds = 0.0
            for suite in self._iter_testsuites(xml_file):
                seconds += float(suite.get("time", 0.0))
        except (ET.ParseError, ValueError, OSError):
            return "00:00:00"
        total_seconds = int(seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _determine_status(self, counts: "TestCounts | None", xml_file: Path) -> TestStatus:
        # Missing or unparsable report => the run crashed before finishing.
        if counts is None:
            print(f"Warning: missing/invalid report {xml_file}", file=sys.stderr)
            return TestStatus.FAILED
        if counts.total == 0 or counts.total == counts.skipped:
            return TestStatus.NO_TESTS
        if counts.failed > 0:
            return TestStatus.FAILED
        return TestStatus.PASSED


class ReportGenerator:
    """Generates formatted test summary reports."""

    WIDTHS = {"name": 35, "status": 10, "counts": 45, "file": 50, "time": 10}
    SEPARATOR = "=" * 150

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
            f"{'Counts':<{self.WIDTHS['counts']}} "
            f"{'Log File':<{self.WIDTHS['file']}} "
            f"{'Time':<{self.WIDTHS['time']}}"
        )

    def _format_subheader(self) -> str:
        return (
            f"{'-' * 10:<{self.WIDTHS['name']}} "
            f"{'-' * 6:<{self.WIDTHS['status']}} "
            f"{'-' * 6:<{self.WIDTHS['counts']}} "
            f"{'-' * 8:<{self.WIDTHS['file']}} "
            f"{'-' * 4:<{self.WIDTHS['time']}}"
        )

    def _format_row(self, result: TestResult) -> str:
        return (
            f"{result.name:<{self.WIDTHS['name']}} "
            f"{result.status.name:<{self.WIDTHS['status']}} "
            f"{result.counts.format():<{self.WIDTHS['counts']}} "
            f"{result.filename:<{self.WIDTHS['file']}} "
            f"{result.duration:<{self.WIDTHS['time']}}"
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze JUnit XML reports and generate summary")
    parser.add_argument("--test-type", required=True, help="Type of tests")
    parser.add_argument("--log-dir", required=True, type=Path, help="Directory with logs/reports")
    parser.add_argument("--summary-log", required=True, type=Path, help="Output file")
    parser.add_argument("--log-pattern", default="*.log", help="Glob pattern for log files")

    args = parser.parse_args()

    if not args.log_dir.exists():
        print(f"Error: Directory not found: {args.log_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        analyzer = XmlAnalyzer(args.log_dir, args.log_pattern)
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
