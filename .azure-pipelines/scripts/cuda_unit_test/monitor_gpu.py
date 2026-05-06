# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "nvitop",
#     "plotille",
# ]
# ///

import math
import os
import sys
import time

import plotille
from nvitop import Device

SIGNAL_FILE = "stop_monitor.flag"
DATA_FILE = "gpu_metrics.csv"


def run_daemon():
    """Background daemon: streams metrics to a local file in real time."""
    if os.path.exists(SIGNAL_FILE):
        os.remove(SIGNAL_FILE)
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)

    print("GPU Monitor Daemon started.")

    try:
        device = Device(0)  # Monitor GPU 0; adjust if multiple GPUs are present
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
                util = device.gpu_utilization()
                util_val = util if util is not None else 0
                mem_bytes = device.memory_used()
                mem_gb = round((mem_bytes / (1024**3)), 2) if mem_bytes is not None else 0.0

                f.write(f"{elapsed_sec},{util_val},{mem_gb}\n")
                f.flush()
            except Exception:
                pass

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
    gpu_util = []
    mem_gb = []

    with open(DATA_FILE, "r") as f:
        lines = f.readlines()[1:]  # skip header
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) == 3:
                timestamps_sec.append(int(parts[0]))
                gpu_util.append(int(parts[1]))
                mem_gb.append(float(parts[2]))

    if not timestamps_sec:
        print("No valid data to plot.")
        return

    # --- Dynamic time unit selection ---
    max_sec = max(timestamps_sec) if timestamps_sec else 1
    if max_sec < 300:
        x_data = timestamps_sec
        x_label = "Time (Seconds)"
    elif max_sec < 7200:
        x_data = [round(t / 60.0, 2) for t in timestamps_sec]
        x_label = "Time (Minutes)"
    else:
        x_data = [round(t / 3600.0, 2) for t in timestamps_sec]
        x_label = "Time (Hours)"

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
    safe_x_data = [min(x, x_lim - 0.0001) for x in x_data]
    safe_gpu_util = gpu_util
    safe_mem_gb = [min(m, y_lim_mem - 0.001) for m in mem_gb]

    print("\n" + "=" * 35 + " GPU Utilization (%) " + "=" * 35)
    try:
        print(
            plotille.plot(
                safe_x_data,
                safe_gpu_util,
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

    print("\n" + "=" * 35 + " GPU Memory Used (GB) " + "=" * 35)
    try:
        print(
            plotille.plot(
                safe_x_data,
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
