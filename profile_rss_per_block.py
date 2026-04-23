"""Granular per-block RSS profiling for peak RAM regression diagnosis.

Instruments both old and new architecture to measure RSS at key points
within the per-block quantization loop.

Usage:
    # New arch:
    python profile_rss_per_block.py
    # Old arch:
    AR_DISABLE_NEW_ARCH=1 python profile_rss_per_block.py
"""
import gc
import os
import resource
import sys
import time


def rss_mb():
    """Get current RSS in MB (no gc.collect - raw measurement)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB -> MB on Linux


def rss_mb_clean():
    """Get current RSS in MB after gc.collect."""
    gc.collect()
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# Use psutil for live RSS (ru_maxrss is peak, not current)
import psutil

_proc = psutil.Process()


def live_rss_mb():
    """Current RSS in MB (not peak)."""
    return _proc.memory_info().rss / (1024*1024)


def live_rss_mb_clean():
    gc.collect()
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass
    return _proc.memory_info().rss / (1024*1024)


arch = os.environ.get("AR_DISABLE_NEW_ARCH", "0")
arch_label = "OLD" if arch == "1" else "NEW"
print(f"\n{'='*70}")
print(f"  {arch_label} Architecture - Granular Per-Block RSS Profiling")
print(f"{'='*70}")
print(f"Before import RSS: {live_rss_mb():.1f} MB")

# Monkey-patch to add instrumentation
if arch != "1":
    # NEW ARCH: patch CalibCompressor._quantize_single_block
    from auto_round.compressors_new import calib as calib_mod
    _orig_quantize_single_block = calib_mod.CalibCompressor._quantize_single_block
    _orig_quantize_blocks = calib_mod.CalibCompressor._quantize_blocks

    _block_rss_log = []

    def _patched_quantize_single_block(self, model, m, input_ids, input_others, q_input):
        block_idx = len(_block_rss_log)
        rss_before = live_rss_mb()

        result = _orig_quantize_single_block(self, model, m, input_ids, input_others, q_input)

        rss_after_return = live_rss_mb()
        gc.collect()
        rss_after_gc = live_rss_mb()
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass
        rss_after_trim = live_rss_mb()

        entry = {
            'block': block_idx,
            'before': rss_before,
            'after_return': rss_after_return,
            'after_gc': rss_after_gc,
            'after_trim': rss_after_trim,
            'delta_return': rss_after_return - rss_before,
            'delta_gc': rss_after_gc - rss_before,
            'delta_trim': rss_after_trim - rss_before,
        }
        _block_rss_log.append(entry)
        print(
            f"  Block {block_idx:2d}: before={rss_before:.1f}  after_ret={rss_after_return:.1f}  "
            f"after_gc={rss_after_gc:.1f}  after_trim={rss_after_trim:.1f}  "
            f"delta_ret={entry['delta_return']:+.1f}  delta_trim={entry['delta_trim']:+.1f} MB",
            flush=True)
        return result

    calib_mod.CalibCompressor._quantize_single_block = _patched_quantize_single_block

else:
    # OLD ARCH: patch LLMCompressor._quantize_block
    from auto_round.compressors import base as base_mod
    _orig_quantize_block = base_mod.LLMCompressor._quantize_block

    _block_rss_log = []

    def _patched_quantize_block(self, block, input_ids, input_others, q_input=None, device="cpu", auto_offload=True):
        block_idx = len(_block_rss_log)
        rss_before = live_rss_mb()

        result = _orig_quantize_block(self, block, input_ids, input_others, q_input, device, auto_offload)

        rss_after_return = live_rss_mb()
        gc.collect()
        rss_after_gc = live_rss_mb()
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass
        rss_after_trim = live_rss_mb()

        entry = {
            'block': block_idx,
            'before': rss_before,
            'after_return': rss_after_return,
            'after_gc': rss_after_gc,
            'after_trim': rss_after_trim,
            'delta_return': rss_after_return - rss_before,
            'delta_gc': rss_after_gc - rss_before,
            'delta_trim': rss_after_trim - rss_before,
        }
        _block_rss_log.append(entry)
        print(
            f"  Block {block_idx:2d}: before={rss_before:.1f}  after_ret={rss_after_return:.1f}  "
            f"after_gc={rss_after_gc:.1f}  after_trim={rss_after_trim:.1f}  "
            f"delta_ret={entry['delta_return']:+.1f}  delta_trim={entry['delta_trim']:+.1f} MB",
            flush=True)
        return result

    base_mod.LLMCompressor._quantize_block = _patched_quantize_block

print(f"After import RSS: {live_rss_mb():.1f} MB")

from auto_round import AutoRound

print(f"After AutoRound import RSS: {live_rss_mb():.1f} MB")

import shutil

save_dir = "/tmp/profile_rss_output"
shutil.rmtree(save_dir, ignore_errors=True)

print(f"\nCreating AutoRound instance...")
ar = AutoRound(
    model="Qwen/Qwen3-0.6B",
    scheme="FP8_STATIC",
    iters=200,
    nsamples=128,
    enable_torch_compile=True,
)
print(f"After init RSS: {live_rss_mb():.1f} MB")
print(f"After init RSS (clean): {live_rss_mb_clean():.1f} MB")

print(f"\nStarting quantize_and_save...\n")
model, folder = ar.quantize_and_save(output_dir=save_dir, format="llm_compressor")

print(f"\n{'='*70}")
print(f"  SUMMARY ({arch_label} Architecture)")
print(f"{'='*70}")
print(f"Final RSS: {live_rss_mb():.1f} MB")
print(f"Final RSS (clean): {live_rss_mb_clean():.1f} MB")
print(f"\nPer-block deltas (after return, after gc+trim):")
for e in _block_rss_log:
    print(
        f"  Block {e['block']:2d}: delta_ret={e['delta_return']:+.1f}  delta_trim={e['delta_trim']:+.1f} MB  "
        f"(abs: {e['after_trim']:.1f} MB)")

# Compute growth rate
if len(_block_rss_log) >= 2:
    first = _block_rss_log[0]['after_trim']
    last = _block_rss_log[-1]['after_trim']
    n = len(_block_rss_log) - 1
    print(f"\nGrowth: {first:.1f} -> {last:.1f} MB over {n} blocks = {(last-first)/n:.1f} MB/block avg")

print(f"\nPeak RSS (ru_maxrss): {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f} MB")

shutil.rmtree(save_dir, ignore_errors=True)
