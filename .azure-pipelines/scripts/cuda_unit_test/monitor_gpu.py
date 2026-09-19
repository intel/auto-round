# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "nvitop",
#     "plotille",
# ]
# ///

import csv
import math
import os
import sys
import time

import plotille
from nvitop import Device

SIGNAL_FILE = "stop_monitor.flag"
DATA_FILE = "gpu_metrics.csv"


def _to_float(value):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None

    return parsed if math.isfinite(parsed) else None


def _format_metric(value):
    return "N/A" if value is None else f"{value:g}"


def _convert_timestamps(timestamps_sec, max_sec):
    if max_sec < 300:
        return timestamps_sec
    if max_sec < 7200:
        return [round(timestamp / 60.0, 2) for timestamp in timestamps_sec]
    return [round(timestamp / 3600.0, 2) for timestamp in timestamps_sec]


def run_daemon():
    """Background daemon: streams metrics to a local file in real time."""
    if os.path.exists(SIGNAL_FILE):
        os.remove(SIGNAL_FILE)
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)

    print("GPU Monitor Daemon started.")

    try:
        device = Device.cuda(0)
    except Exception as e:
        print(f"CUDA-visible GPU lookup failed: {e}; falling back to NVML GPU 0.")
        try:
            device = Device(0)
        except Exception as e:
            print(f"NVML error: {e}")
            sys.exit(0)

    start_time = time.time()
    print("Daemon is running. Streaming metrics to CSV every 5 seconds...")

    with open(DATA_FILE, "w") as f:
        f.write("elapsed_sec,gpu_util_pct,mem_used_gb\n")

        while not os.path.exists(SIGNAL_FILE):
            try:
                elapsed_sec = int(time.time() - start_time)
                util_val = _to_float(device.gpu_utilization())
                mem_bytes = _to_float(device.memory_used())
                mem_gb = round((mem_bytes / (1024**3)), 2) if mem_bytes is not None else None

                f.write(f"{elapsed_sec},{_format_metric(util_val)},{_format_metric(mem_gb)}\n")
                f.flush()
            except Exception as e:
                print(f"Failed to collect GPU metrics: {e}")

            time.sleep(5.0)

    print("Stop signal received. Daemon exiting.")


def stop_and_plot():
    """Stop monitoring and plot charts with clean axis limits."""
    with open(SIGNAL_FILE, "w") as f:
        f.write("STOP")

    time.sleep(2)

    if not os.path.exists(DATA_FILE):
        print("Error: Could not find GPU metrics data file.")
        sys.exit(1)

    timestamps_sec = []
    gpu_timestamps_sec = []
    gpu_util = []
    mem_timestamps_sec = []
    mem_gb = []
    invalid_rows = 0
    missing_util_rows = 0

    try:
        with open(DATA_FILE, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row_number, parts in enumerate(reader, start=2):
                if len(parts) != 3:
                    invalid_rows += 1
                    print(f"Skipping malformed GPU metrics row {row_number}.")
                    continue

                try:
                    timestamp = int(parts[0])
                except (TypeError, ValueError):
                    invalid_rows += 1
                    print(f"Skipping GPU metrics row {row_number}: invalid timestamp {parts[0]!r}.")
                    continue

                util_value = _to_float(parts[1])
                mem_value = _to_float(parts[2])
                if util_value is None:
                    missing_util_rows += 1
                if util_value is None and mem_value is None:
                    invalid_rows += 1
                    print(f"Skipping GPU metrics row {row_number}: no numeric metrics.")
                    continue

                timestamps_sec.append(timestamp)
                if util_value is not None:
                    gpu_timestamps_sec.append(timestamp)
                    gpu_util.append(util_value)
                if mem_value is not None:
                    mem_timestamps_sec.append(timestamp)
                    mem_gb.append(mem_value)
    except (OSError, csv.Error) as e:
        print(f"Failed to read GPU metrics data: {e}")
        return

    if not timestamps_sec:
        print("No valid data to plot.")
        return

    if invalid_rows:
        print(f"Skipped {invalid_rows} invalid GPU metrics row(s).")
    if missing_util_rows:
        print(f"GPU utilization was unavailable in {missing_util_rows} row(s); omitted from the utilization plot.")

    # --- Dynamic time unit selection ---
    max_sec = max(timestamps_sec) if timestamps_sec else 1
    if max_sec < 300:
        x_label = "Time (Seconds)"
    elif max_sec < 7200:
        x_label = "Time (Minutes)"
    else:
        x_label = "Time (Hours)"

    x_data = _convert_timestamps(timestamps_sec, max_sec)
    gpu_x_data = _convert_timestamps(gpu_timestamps_sec, max_sec)
    mem_x_data = _convert_timestamps(mem_timestamps_sec, max_sec)

    # --- Compute clean axis upper limits ---
    SCALE = 11

    max_x = max(x_data) if x_data and max(x_data) > 0 else 1
    max_mem = max(mem_gb) if mem_gb and max(mem_gb) > 0 else 1

    x_lim = math.ceil(max_x / 8.0) * 8
    y_lim_mem = math.ceil(max_mem / 10.0) * SCALE

    # ==========================================
    # Workaround for Boundary Clipping:
    # Nudge values that sit exactly on the axis limit slightly inward
    # so plotille does not clip them at the canvas edge.
    # ==========================================
    safe_gpu_x_data = [min(x, x_lim - 0.0001) for x in gpu_x_data]
    safe_mem_x_data = [min(x, x_lim - 0.0001) for x in mem_x_data]
    safe_mem_gb = [min(m, y_lim_mem - 0.001) for m in mem_gb]

    if gpu_util:
        print("\n" + "=" * 35 + " GPU Utilization (%) " + "=" * 35)
        try:
            print(
                plotille.plot(
                    safe_gpu_x_data,
                    gpu_util,
                    height=SCALE,
                    width=80,
                    X_label=x_label,
                    Y_label="GPU Util (%)",
                    x_min=0,
                    x_max=x_lim,
                    y_min=0,
                    y_max=SCALE * 10,
                    interp="linear",
                )
            )
        except Exception as e:
            print(f"Failed to plot GPU Utilization: {e}")
    else:
        print("No valid GPU utilization data to plot.")

    if mem_gb:
        print("\n" + "=" * 35 + " GPU Memory Used (GB) " + "=" * 35)
        try:
            print(
                plotille.plot(
                    safe_mem_x_data,
                    safe_mem_gb,
                    height=SCALE,
                    width=80,
                    X_label=x_label,
                    Y_label="Memory (GB)",
                    x_min=0,
                    x_max=x_lim,
                    y_min=0,
                    y_max=y_lim_mem,
                    interp="linear",
                )
            )
        except Exception as e:
            print(f"Failed to plot GPU Memory: {e}")
    else:
        print("No valid GPU memory data to plot.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python monitor_gpu.py [daemon|stop]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "daemon":
        run_daemon()
    elif command == "stop":
        stop_and_plot()
    else:
        print("Unknown command.")
