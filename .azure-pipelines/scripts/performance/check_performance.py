import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(message)s")

LOG_DIR = Path("/auto-round/log_dir")
OUTPUT_BASE_DIR = Path("/auto-round/.azure-pipelines/scripts/performance")


@dataclass
class QuantMetrics:
    tuning_time_s: Optional[float] = None
    peak_ram_gb: Optional[float] = None
    peak_vram_gb: Optional[float] = None
    output_size_gb: Optional[float] = None


def get_dir_size_gb(path: Path) -> float:
    if not path.exists() or not path.is_dir():
        return 0.0

    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total_bytes / (1024**3), 4)


def parse_log_file(log_file: Path) -> QuantMetrics:
    metrics = QuantMetrics()

    if not log_file.exists():
        logging.warning(f"Log file not found: {log_file}")
        return metrics

    content = log_file.read_text(encoding="utf-8")

    # Use findall to capture all occurrences and take the most recent one.
    time_matches = re.findall(r"tuning time ([0-9]+\.[0-9]+)", content)
    if time_matches:
        metrics.tuning_time_s = round(float(time_matches[-1]), 4)

    ram_matches = re.findall(r"'peak_ram':\s*([\d.]+)\s*GB,\s*'peak_vram':\s*([\d.]+)\s*GB", content)
    if ram_matches:
        last_ram, last_vram = ram_matches[-1]
        metrics.peak_ram_gb = round(float(last_ram), 4)
        metrics.peak_vram_gb = round(float(last_vram), 4)

    return metrics


def get_tuning_info() -> Dict[str, Dict[str, QuantMetrics]]:
    summary = {}
    model_list = ["Qwen/Qwen3-0.6B"]

    for model in model_list:
        summary[model] = {}
        for test_mode in ["current", "baseline"]:
            log_file = LOG_DIR / f"perf_test_{test_mode}.log"
            output_dir = OUTPUT_BASE_DIR / test_mode

            logging.info(f"Processing {log_file}...")

            metrics = parse_log_file(log_file)
            metrics.output_size_gb = get_dir_size_gb(output_dir)

            summary[model][test_mode] = metrics

    return summary


def compare_metric(
    metric_name: str, current: Optional[float], baseline: Optional[float], tolerance: float = 0.1
) -> bool:
    if current is None or baseline is None:
        logging.error(f"  [-] {metric_name}: Incomplete data (Current: {current}, Baseline: {baseline})")
        return False

    if baseline == 0:
        logging.warning(f"  [!] {metric_name}: Baseline is 0, cannot calculate ratio.")
        return False

    ratio = current / baseline
    diff_percent = (ratio - 1) * 100

    msg = f"  [*] {metric_name:<20}: Current = {current:<8} | Baseline = {baseline:<8} (Diff: {diff_percent:+.2f}%)"

    if 1.0 - tolerance <= ratio <= 1.0 + tolerance:
        logging.info(f"{msg} -> PASS")
        return True
    else:
        logging.error(f"{msg} -> FAIL")
        return False


def check_performance():
    summary = get_tuning_info()
    all_passed = True

    for model, modes in summary.items():
        logging.info(f"\nEvaluating Model: {model}")
        logging.info("-" * 60)

        current: QuantMetrics = modes.get("current", QuantMetrics())
        baseline: QuantMetrics = modes.get("baseline", QuantMetrics())

        if not compare_metric("Tuning Time (s)", current.tuning_time_s, baseline.tuning_time_s, tolerance=0.05):
            all_passed = False

        if not compare_metric("Peak RAM (GB)", current.peak_ram_gb, baseline.peak_ram_gb, tolerance=0.03):
            all_passed = False

        if not compare_metric("Peak VRAM (GB)", current.peak_vram_gb, baseline.peak_vram_gb, tolerance=0.03):
            all_passed = False

        if not compare_metric("Output Size (GB)", current.output_size_gb, baseline.output_size_gb, tolerance=0.01):
            all_passed = False

    logging.info("=" * 60)
    if all_passed:
        logging.info("✅ Performance check passed: All metrics are within acceptable limits.")
    else:
        logging.error("❌ Performance check failed: Current metrics exceed acceptable limits compared to baseline.")
        sys.exit(1)


def main():
    check_performance()


if __name__ == "__main__":
    main()
