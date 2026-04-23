"""Quick A/B benchmark: old (compressors) vs new (compressors_new) architecture.

Uses AR_DISABLE_NEW_ARCH env-var to toggle.  Runs each configuration in a
subprocess to avoid cross-contamination, with a warmup run to fill OS page cache.
"""

import json
import os
import subprocess
import sys
import time


MODEL = "Qwen/Qwen3-0.6B"
ITERS = "200"
SCHEME = "W4A16"
DEVICE = "cuda:0"

CMD_TEMPLATE = [
    sys.executable, "-m", "auto_round",
    "--model_name", MODEL,
    "--scheme", SCHEME,
    "--iters", ITERS,
    "--device", DEVICE,
]


def run_once(label: str, env_override: dict) -> float:
    env = {**os.environ, **env_override}
    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"  AR_DISABLE_NEW_ARCH={env.get('AR_DISABLE_NEW_ARCH', 'unset')}")
    print(f"{'='*60}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(CMD_TEMPLATE, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"STDERR:\n{proc.stderr[-2000:]}")
        raise RuntimeError(f"{label} failed with rc={proc.returncode}")
    print(f"  {label}: {elapsed:.1f}s")
    return elapsed


def main():
    # Warmup: fill OS page cache & JIT caches
    print("Warmup run (old arch)...")
    run_once("warmup", {"AR_DISABLE_NEW_ARCH": "1"})

    # Interleaved runs to reduce bias
    results = {"old": [], "new": []}
    for trial in range(2):
        if trial % 2 == 0:
            first, second = ("old", "1"), ("new", "0")
        else:
            first, second = ("new", "0"), ("old", "1")
        for label, flag in [first, second]:
            t = run_once(f"{label} (trial {trial+1})", {"AR_DISABLE_NEW_ARCH": flag})
            results[label].append(t)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for arch in ["old", "new"]:
        times = results[arch]
        avg = sum(times) / len(times)
        print(f"  {arch}: {[f'{t:.1f}' for t in times]}  avg={avg:.1f}s")
    old_avg = sum(results["old"]) / len(results["old"])
    new_avg = sum(results["new"]) / len(results["new"])
    diff_pct = (new_avg - old_avg) / old_avg * 100
    print(f"\n  Diff: {diff_pct:+.1f}%  (new vs old)")
    print(f"  {'PASS' if abs(diff_pct) < 5 else 'FAIL'} (threshold: ±5%)")

    json.dump(results, open("benchmark_results/latest.json", "w"), indent=2)


if __name__ == "__main__":
    main()
