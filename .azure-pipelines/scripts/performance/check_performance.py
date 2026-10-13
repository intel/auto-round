import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

LOG_DIR = Path("/auto-round/log_dir")
OUTPUT_BASE_DIR = Path("/auto-round/.azure-pipelines/scripts/performance")


@dataclass
class QuantMetrics:
    tuning_time_s: float | None = None
    peak_ram_gb: float | None = None
    peak_vram_gb: float | None = None
    output_size_gb: float | None = None


def get_dir_size_gb(path: Path) -> float:
    if not path.exists() or not path.is_dir():
        return 0.0

    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total_bytes / (1024**3), 4)


def parse_log_file(log_file: Path) -> QuantMetrics:
    metrics = QuantMetrics()

    if not log_file.exists():
        logger.warning(f"Log file not found: {log_file}")
        return metrics

    content = log_file.read_text(encoding="utf-8")

    # Use findall to capture all occurrences and take the most recent one.
    time_matches = re.findall(r"tuning time ([0-9]+\.[0-9]+)", content)
    if time_matches:
        metrics.tuning_time_s = round(float(time_matches[-1]), 4)

    return metrics


def parse_peak_memory(metrics_file: Path, metrics: QuantMetrics) -> None:
    """Read high-water marks recorded by measure_peak.py instead of auto-round's sampled log summary."""
    if not metrics_file.exists():
        logger.warning(f"Metrics file not found: {metrics_file}")
        return

    data = json.loads(metrics_file.read_text(encoding="utf-8"))
    metrics.peak_ram_gb = data.get("peak_ram_gb")
    metrics.peak_vram_gb = data.get("peak_vram_gb")


def get_tuning_info() -> dict[str, dict[str, QuantMetrics]]:
    summary = {}
    model_list = ["Qwen/Qwen3-0.6B"]

    for model in model_list:
        summary[model] = {}
        for test_mode in ["current", "baseline"]:
            log_file = LOG_DIR / f"perf_test_{test_mode}.log"
            output_dir = OUTPUT_BASE_DIR / test_mode

            logger.info(f"Processing {log_file}...")

            metrics = parse_log_file(log_file)
            parse_peak_memory(LOG_DIR / f"perf_metrics_{test_mode}.json", metrics)
            metrics.output_size_gb = get_dir_size_gb(output_dir)

            summary[model][test_mode] = metrics

    return summary


def compare_metric(
    metric_name: str,
    current: float | None,
    baseline: float | None,
    tolerance: float = 0.1,
    abs_tolerance: float = 0.0,
) -> bool:
    """Pass when |current - baseline| <= max(tolerance * baseline, abs_tolerance).

    ``abs_tolerance`` is a noise floor for metrics whose jitter is roughly constant in absolute
    terms (e.g. ~0.12GB of glibc heap retention in peak RAM). A purely relative bound shrinks
    together with the baseline and turns that jitter into flaky failures as memory usage drops.
    """
    if current is None or baseline is None:
        logger.error(f"  [-] {metric_name}: Incomplete data (Current: {current}, Baseline: {baseline})")
        return False

    if baseline == 0:
        logger.warning(f"  [!] {metric_name}: Baseline is 0, cannot calculate ratio.")
        return False

    diff = current - baseline
    diff_percent = diff / baseline * 100
    allowed = max(tolerance * abs(baseline), abs_tolerance)

    msg = (
        f"  [*] {metric_name:<20}: Current = {current:<8} | Baseline = {baseline:<8} "
        f"(Diff: {diff:+.4f} / {diff_percent:+.2f}%, Allowed: ±{allowed:.4f})"
    )

    if abs(diff) <= allowed:
        logger.info(f"{msg} -> PASS")
        return True
    else:
        logger.error(f"{msg} -> FAIL")
        return False


def check_performance():
    summary = get_tuning_info()
    all_passed = True

    for model, modes in summary.items():
        logger.info(f"\nEvaluating Model: {model}")
        logger.info("-" * 60)

        current: QuantMetrics = modes.get("current", QuantMetrics())
        baseline: QuantMetrics = modes.get("baseline", QuantMetrics())

        if not compare_metric("Tuning Time (s)", current.tuning_time_s, baseline.tuning_time_s, tolerance=0.05):
            all_passed = False

        if not compare_metric(
            "Peak RAM (GB)", current.peak_ram_gb, baseline.peak_ram_gb, tolerance=0.03, abs_tolerance=0.15
        ):
            all_passed = False

        if not compare_metric(
            "Peak VRAM (GB)", current.peak_vram_gb, baseline.peak_vram_gb, tolerance=0.03, abs_tolerance=0.25
        ):
            all_passed = False

        if not compare_metric("Output Size (GB)", current.output_size_gb, baseline.output_size_gb, tolerance=0.01):
            all_passed = False

    logger.info("=" * 60)
    if all_passed:
        logger.info("✅ Performance check passed: All metrics are within acceptable limits.")
    else:
        logger.error("❌ Performance check failed: Current metrics exceed acceptable limits compared to baseline.")
        sys.exit(1)


def main():
    check_performance()


if __name__ == "__main__":
    main()
