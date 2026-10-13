"""Run the auto-round CLI in-process and record its peak memory as JSON.

Usage:
    python measure_peak.py --output metrics.json -- --model_name Qwen/Qwen3-0.6B --scheme W4A16 ...

``peak_ram_gb`` is the kernel-maintained RSS high-water mark (``ru_maxrss``). Unlike the per-block
snapshots logged by auto-round, it does not depend on when memory is sampled or on short-lived child
processes, and both the current and the baseline builds are measured by this same script.

``vram_high_water_gb`` is the allocator high-water mark (``max_memory_allocated``), recorded for
diagnostics only: it covers transient peaks inside a block, but on HPU it also counts tensors whose
frees are still queued in the eager pipeline, so it varies with host/device timing.
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

GB = 1024**3


def peak_ram_gb() -> float:
    # Linux reports ru_maxrss in KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GB


def vram_high_water_gb() -> dict[str, float]:
    import torch

    peaks = {}
    for name in ("hpu", "cuda", "xpu"):
        backend = getattr(torch, name, None)
        try:
            if backend is None or not backend.is_available():
                continue
            if not getattr(backend, "is_initialized", lambda: True)():
                continue
            # Flush asynchronous pipelines (e.g. HPU eager pipeline) so the high-water mark is final.
            backend.synchronize()
            # An HPU process owns a single device and ignores the index argument.
            indices = [backend.current_device()] if name == "hpu" else range(backend.device_count())
            for idx in indices:
                peak = backend.max_memory_allocated(idx)
                if peak > 0:
                    peaks[f"{name}:{idx}"] = peak / GB
        except Exception as error:
            print(f"[measure_peak] Skip {name} peak memory: {error}", file=sys.stderr)
    return peaks


def main() -> None:
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="Path of the JSON metrics file.")
    args = parser.parse_args(argv[:split])

    from auto_round.__main__ import run

    sys.argv = ["auto-round", *argv[split + 1 :]]
    start = time.perf_counter()
    run()
    wall_time_s = time.perf_counter() - start

    vram = vram_high_water_gb()
    metrics = {
        "peak_ram_gb": round(peak_ram_gb(), 4),
        "vram_high_water_gb": round(max(vram.values()), 4) if vram else None,
        "wall_time_s": round(wall_time_s, 2),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[measure_peak] {json.dumps(metrics)}", flush=True)


if __name__ == "__main__":
    main()
