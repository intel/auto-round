import re
import sys

LOG_DIR = "/auto-round/log_dir"


def parse_tuning_time(log_file):
    with open(log_file, "r") as f:
        content = f.read()

    pattern = r"tuning time ([0-9]+\.[0-9]+)"
    match = re.search(pattern, content)

    if match:
        elapsed = str_to_float(match.group(1))
        return elapsed
    return None


def str_to_float(value):
    try:
        return round(float(value), 4)
    except ValueError:
        return value


def get_tuning_time():
    summary = {}
    model_list = ["Qwen/Qwen3-0.6B"]
    for model in model_list:
        summary[model] = {}
        for test_mode in ["current", "baseline"]:
            log_file = f"{LOG_DIR}/perf_test_{test_mode}.log"
            print(f"Processing {log_file}...")
            tuning_time = parse_tuning_time(log_file)
            if tuning_time is not None:
                summary[model][test_mode] = tuning_time
            else:
                summary[model][test_mode] = "N/A"

    return summary


def check_performance():
    status = True
    summary = get_tuning_time()
    for model, times in summary.items():
        current_time = times.get("current", "N/A")
        baseline_time = times.get("baseline", "N/A")
        if current_time != "N/A" and baseline_time != "N/A":
            print(f"{model}:\n  Current = {current_time} seconds\n  Baseline = {baseline_time} seconds")
            ratio = current_time / baseline_time
            if ratio < 0.9 or ratio > 1.1:
                status = False
        else:
            print(f"{model}: Tuning time data is incomplete.")
            status = False

    if status:
        print("Performance check passed: Current tuning times are within acceptable limits compared to baseline.")
    else:
        print("Performance check failed: Current tuning times exceed acceptable limits compared to baseline.")
        sys.exit(1)


def main():
    check_performance()


if __name__ == "__main__":
    main()
