# Sparse Kernel Benchmark Plan

## Summary

Add a dedicated C++ benchmark executable for sparse prefill kernel tuning, separate from `test_ARK_XPU`.

The first slice focuses on:

- kernel-only sparse prefill timing
- dense `sagev1` prefill baseline
- Wan- and Flux-shaped presets
- explicit sparse row-pattern control for later sparse-prefetch tuning

This benchmark is intended to complement, not replace, the existing Python benchmark harness.

Current implementation status:

- implemented as `bench_ARK_XPU`
- source is `wrapper/test/bench_sparse_kernel.cpp`
- enabled by `-DARK_BENCH=ON`
- validated with model-shaped smoke runs on `wan_self`, `flux_joint`, and `flux_single`

## Key Changes

### 1. New benchmark executable

- Add a dedicated benchmark target in CMake instead of reusing `test_ARK_XPU`.
- Reuse the same XPU/SYCL-TLA build path, generated SDPA sources, and low-level C++ kernel entrypoints already used by the module.
- Keep correctness tests and perf runs separate.

### 2. Model-shaped presets

Add named presets for the current model regimes:

- `wan_self`
- `flux_joint`
- `flux_single`

Preset defaults should follow the current model families:

- Wan: `12` heads, `128` head dim, bf16 V/O, representative long self-attention token count
- Flux: `24` heads, `128` head dim, bf16 V/O
  - `flux_joint`: image+text joint attention shape
  - `flux_single`: image-only stream shape

The first encoded defaults should be reproducible from the current runner assumptions:

- `wan_self`: representative long sequence for the current Wan default run shape
- `flux_joint`: image tokens plus `512` text tokens
- `flux_single`: image-token-only stream for the same image size

### 3. Kernel-only sparse timing

- Build sparse metadata outside the timed region.
- Time only `sdpa_impl_qks8_sparse_pvhalf(...)` for sparse rows.
- Time `sdpa_impl_qks8_pvhalf(...)` for the dense baseline.
- Keep preprocess benchmarking out of this executable.

### 4. Sparse-driving modes

Support both:

- top-k style sweeps through generated block selection counts
- explicit row-pattern presets for tuning

Initial row-pattern presets:

- `all_selected`
- `prefix`
- `stride2`
- `custom_02`
- `custom_035`
- `custom_135`

These are intended to expose both easy and irregular sparse traversal behavior before sparse prefetch is added.

### 5. Output and controls

Expose runtime flags for:

- preset
- warmup
- iters
- topk list
- pattern preset
- optional raw overrides for batch / heads / seq / head_dim
- optional CSV output path

Each benchmark row should include:

- preset
- mode (`dense_sagev1`, `sparse_kernel_only`)
- dtype
- requested topk or pattern
- actual selected ratio / blocks per row
- latency ms
- dense-equivalent TFLOPS
- effective sparse TFLOPS

## Test Plan

- Build the new benchmark target successfully with the existing XPU toolchain.
- Run smoke cases for:
  - `wan_self` all-selected
  - `flux_joint` non-contiguous `custom_035`
  - `flux_single` `prefix`
- Validate:
  - all-selected sparse is close to dense `sagev1`
  - lower selected ratio reduces sparse kernel latency
  - custom non-contiguous rows run cleanly

## Build And Usage

Build:

```bash
source /opt/intel/oneapi/setvars.sh
/home/yiliu7/workspace/venvs/ark/bin/cmake -B xbuild -DARK_BENCH=ON
/home/yiliu7/workspace/venvs/ark/bin/cmake --build xbuild --target bench_ARK_XPU -j 4
```

Usage:

```bash
./xbuild/bench_ARK_XPU --preset flux_single --pattern prefix --topk 1.0,0.5 --warmup 1 --iters 1
```

```bash
./xbuild/bench_ARK_XPU --preset flux_joint --pattern custom_035 --warmup 1 --iters 1
```

```bash
./xbuild/bench_ARK_XPU --preset wan_self --pattern all_selected --warmup 0 --iters 1
```

CLI controls:

- `--preset`: `wan_self`, `flux_joint`, `flux_single`
- `--pattern`: `all_selected`, `prefix`, `stride2`, `custom_02`, `custom_035`, `custom_135`
- `--topk`: comma-separated list
- `--warmup`, `--iters`
- `--csv`
- overrides: `--batch`, `--heads-q`, `--heads-kv`, `--seq-q`, `--seq-kv`, `--head-dim`, `--block-size`

Current smoke results:

- `wan_self`, dense `sagev1`: `14723.477 ms`
- `wan_self`, sparse `all_selected`: `157.020 ms`
- `flux_single`, dense `sagev1`: `32.653 ms`
- `flux_single`, sparse `prefix`, `topk=1.0`: `76.785 ms`
- `flux_single`, sparse `prefix`, `topk=0.5`: `37.250 ms`
- `flux_joint`, sparse `custom_035`: `1.329 ms`

Current caveat:

- the `wan_self` dense single-shot smoke timing looks suspiciously slow and should be rechecked before using that preset as a dense baseline for tuning

## Assumptions

- The benchmark executable is separate from `test_ARK_XPU`.
- V1 is prefill-only and kernel-only.
- The first dense comparison is against `sagev1`, not torch SDPA.
- Model-shaped presets are encoded as reproducible defaults and can be overridden from the command line.
