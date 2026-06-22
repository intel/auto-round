# Sparse Sage Kernel Status

## Current Snapshot

Kernel:

- sparse prefill and cached decode paths exist:
  - `sage_sparse(...)`
  - `sage_sparse_decode(...)`
- kernel contract is stable:
  - `lut + valid_block_num + qscale + kscale + int8 Q/K + fp16/bf16 V`
- `xe_sparse_sagev1_fwd_mainloop.hpp` is currently a copy-paste dense baseline
- this restored performance versus the earlier sparse-native refactor
- true sparse-aware K prefetch is still not implemented
- dense-style K prefetch still only applies when `lut_row == nullptr`

Correctness:

- verified:
  - all-selected sparse = dense parity
  - partial sparse prefill = dense masked reference
  - causal sparse prefill = dense causal+masked reference
  - non-contiguous rows like `[0,3,5]`, `[0,2]`, `[1,3,5]`
  - cached sparse decode smoke
- model-level:
  - Qwen sparse prefill smoke passes
  - Qwen sparse decode smoke passes
  - Wan sparse smoke passes
  - Flux sparse integration exists

Preprocess:

- Triton-XPU is used for:
  - fused `pool + sim + quant`
  - `fill_block_map`
  - `block_map_to_lut`
- remaining routing/math is still partly torch-side
- current main preprocess cost is still routing/pooling related, not LUT conversion

Performance:

- long synthetic benchmark (`B=1, H=40, S=75600, D=128, topk=0.5`):
  - dense `sagev1`: `1172.91 ms`
  - sparse kernel-only: `591.08 ms`
  - sparse e2e: `787.31 ms`
- so:
  - sparse kernel-only: about `1.98x` faster than dense
  - sparse e2e: about `1.49x` faster than dense

Flux block `v2` profile summary:

- dense block: `152.05 ms`
- sparse `topk=0.5`: `122.33 ms`
- sparse `topk=0.1`: `112.96 ms`
- `_triton_bmm_pool_sim_simmean_fuse_quant_xpu`:
  - about `21.3 ms` in both sparse traces
  - largest named Triton preprocess kernel
  - but not the only preprocess bottleneck

Benchmark harness:

- C++ kernel-only benchmark exists:
  - `wrapper/test/bench_sparse_kernel.cpp`
  - target: `bench_ARK_XPU`
- model-shaped presets:
  - `wan_self`
  - `flux_joint`
  - `flux_single`

Sparse prefill vs decode:

| Aspect | Sparse prefill | Sparse decode |
| --- | --- | --- |
| Public API | `sage_sparse(...)` | `sage_sparse_decode(...)` |
| C++ entrypoint | `sdpa_impl_qks8_sparse_pvhalf(...)` | `sdpa_impl_qks8_sparse_decode_pvhalf(...)` |
| Main kernel wrapper | `XeSparseSageFwdKernel` | `XeSparseSageFwdKernel` |
| Mainloop family | `SPARSESAGEV1FwdMainloop` | `SPARSESAGEV1FwdMainloop` with cached-KV path |
| Q shape contract | general prefill, typically `seq_len_q == seq_len_kv` | current v1 requires `seq_len_q == 1` |
| KV source | current `K/V` only | current `K/V` plus `K_cache/V_cache` |
| Sparse block-id meaning | `lut` rows index only the current KV sequence | `lut` rows index the logical concatenation `cache blocks + current blocks` |
| Scale contract | `qscale` and `kscale` over current Q/K blocks | `qscale` plus `kscale` over total visible KV blocks, including cache |
| Extra required inputs | `lut`, `valid_block_num` | `lut`, `valid_block_num`, `K_cache`, `V_cache`, `seq_len_kv_cache` |
| Typical call path | sparse prefill kernel launch | cached sparse decode kernel launch |
| Why separate from dense decode | sparse metadata must be consumed in-kernel | sparse metadata must also resolve cache-vs-current KV blocks |

What is still missing:

- sparse-aware K prefetch
- more dense-vs-sparse hot-path alignment work
- fully Tritonized preprocess routing
- broader decode/GQA coverage
- fp8-V / sparse int8-PV support

## Summary

The ARK sparse Sage attention path is now in a good **prefill-ready** state for integrating Sparge-style preprocess output, and the preprocess side has partial Triton-XPU acceleration.

Current status:

- sparse prefill and cached-decode kernel paths exist and are callable through `sage_sparse(...)` and `sage_sparse_decode(...)`
- kernel contract is `lut + valid_block_num + qscale + kscale + int8 Q/K + fp16/bf16 V`
- sparse traversal is now implemented through a dedicated sparse Sage mainloop in `xe_sparse_sagev1_fwd_mainloop.hpp`
- causal masking is handled in-kernel
- Sparge-style preprocess exists on XPU through `sparge_preprocess_topk(...)` / `sparge_sage2_attn_meansim_topk_xpu(...)`
- the preprocess backend now uses Triton-XPU for:
  - fused `pool + sim + quant`
  - `fill_block_map`
  - `block_map_to_lut`
- remaining preprocess routing/math still uses torch tensor ops
- a dedicated C++ kernel-only benchmark executable now exists as `bench_ARK_XPU` for Wan- and Flux-shaped sparse-kernel tuning

This means the kernel is already usable as the ARK/XPU counterpart of the attention-execution stage of `spas_sage2_attn_meansim_topk_cuda`.

## Dedicated Sparse Mainloop Status

The sparse mainloop split is now partially implemented:

- `xe_sparse_sagev1_fwd_mainloop.hpp` is now a copy-paste baseline from `xe_sagev1_fwd_mainloop.hpp`, not the earlier inheritance-based sparse-native refactor
- this was done because the earlier sparse-native refactor regressed kernel performance by drifting too far from the dense hot path
- the current sparse file is intentionally dense-structured first, so later sparse optimization can start from a performance-stable baseline
- dense-style K prefetch is still guarded by `if (lut_row == nullptr)`, so true LUT-driven sparse rows still do **not** have a real sparse-aware K prefetch path yet

This means:

- performance came back after restoring the dense mainloop structure
- sparse-aware K prefetch is still a separate future optimization
- the current copy-paste sparse mainloop is the baseline that later sparse-prefetch work should optimize from

## What Is Verified

Kernel and API validation completed:

- all-selected sparse rows match dense Sage exactly
- partial-sparse prefill matches dense masked reference
- causal sparse prefill matches dense causal+masked reference
- non-contiguous sparse rows match dense masked reference directly, including patterns such as:
  - `[0, 3, 5]`
  - `[0, 2]`
  - `[1, 3, 5]`
  - `[0, 1, 4]`
- cached sparse decode coverage exists and the model-level all-selected cached path remains healthy
- Python `sage_sparse(...)` smoke coverage exists for:
  - all-selected
  - partial sparse
  - causal partial sparse

Model-level prefill-style smoke coverage completed:

- Qwen no-cache generation test uses the public sparse path
- `all_selected` mode gives exact dense parity for 30 steps
- original `partial_sparse` routing mode is preserved and runs end-to-end
- a preprocess-generated `sparse_preprocess` path is wired into the same smoke harness

Current caveat for preprocess integration:

- preprocess-generated partial sparsity is still an approximation path, so model-level `sparse_preprocess` output can diverge from dense full attention by design
- low-level and tensor-level preprocess replay against the same sparse metadata are validated
- Triton-XPU preprocess metadata matches the torch reference on discrete outputs; small scale-tensor differences remain at floating-point noise level only

Current caveat for the dedicated sparse-mainloop validation:

- the extension build is green
- Python/runtime sparse regressions are green
- the heavyweight `xbuild_ut_icpx --target test_ARK_XPU` rebuild can still stall or get resource-killed on the giant `sdpa.cpp` TU in this shell, so that unit binary is not the primary verification signal for this specific refactor pass
- the previous inheritance-based sparse-native refactor was intentionally replaced by the copy-paste baseline because its hot path underperformed on kernel benchmarks

Current caveat for the new C++ benchmark harness:

- the benchmark target reuses the generated SYCL-TLA SDPA sources, which still include a CUTLASS utility header that expects `oneapi/mkl/rng/device.hpp`
- this node currently does not have a full oneMKL install, so the repo now carries a small local compatibility shim at `compat/include/oneapi/mkl/rng/device.hpp`
- that shim is only present to satisfy the unused CUTLASS utility include during compile; the benchmarked ARK sparse kernel path does not depend on oneMKL RNG behavior

## What The Kernel Already Covers

- prefill only
- prefill with `seq_len_q == seq_len_kv`
- cached decode with `seq_len_q == 1`
- `head_dim = 64` and `128`
- non-causal and causal
- LUT-driven sparse K-block traversal
- int8 `QK`
- fp16/bf16 `V`
- dense-equivalent sparse routing, real partial sparsity, and raw non-contiguous sparse rows

## Current Preprocess Backend

The current preprocess stack is split into:

- Triton-XPU:
  - fused block pooling / similarity / int8 quantization
  - block-map fill
  - LUT generation
- torch tensor ops:
  - pooled `QK` routing score
  - softmax + sort
  - query-tile routing reduction / remap
  - some small mask/index glue

So the preprocess is no longer “torch-only”, but it is not yet fully Tritonized either.

## What Is Still Missing

The following are not complete yet:

- fully Tritonized preprocess routing
- sparse-aware K prefetch for true LUT-driven sparse rows
- `seq_len_q != seq_len_kv` prefill
- fp8-V sparse path
- sparse int8 `PV`
- broader GQA-specific sparse validation
- more decode-side sparse coverage beyond the current cached path

## Performance Status

Current benchmark artifacts:

- `bench_sparse_topk_results_triton_xpu.csv`
- `bench_sparse_topk_results_triton_xpu_75600_w3_i5_v3.csv`

On `B=1, H=40, S=75600, D=128, topk=0.5`:

- `dense_torch_sdpa`: `2119.59 ms`
- `dense_sagev1`: `1172.91 ms`
- `sparse_kernel_only`: `591.08 ms`
- `sparse_e2e`: `787.31 ms`

So at this shape:

- sparse kernel-only is about `1.98x` faster than dense `sagev1`
- sparse end-to-end is about `1.49x` faster than dense `sagev1`

After porting Triton-XPU `fill_block_map` and `block_map_to_lut`, the dominant preprocess stage is now query-tile routing pooling rather than block-map construction.

## C++ Kernel Benchmark Harness

The repo now also has a dedicated C++ kernel-only benchmark executable:

- target: `bench_ARK_XPU`
- source: `wrapper/test/bench_sparse_kernel.cpp`

Current benchmark presets are shaped around the active diffusion integrations:

- `wan_self`
- `flux_joint`
- `flux_single`

This harness is intentionally:

- prefill-only
- kernel-only
- dense baseline = `sagev1`
- sparse mode = `sdpa_impl_qks8_sparse_pvhalf(...)` with metadata built outside the timed region

It supports both:

- top-k style block-count sweeps
- explicit sparse row patterns such as:
  - `all_selected`
  - `prefix`
  - `stride2`
  - `custom_02`
  - `custom_035`
  - `custom_135`

Smoke validation completed on the built executable:

- `wan_self` + `all_selected`
- `flux_single` + `prefix` + `topk=1.0,0.5`
- `flux_joint` + `custom_035`

Observed smoke outputs:

- `wan_self`, dense `sagev1`: `14723.477 ms`
- `wan_self`, sparse `all_selected`: `157.020 ms`
- `flux_single`, dense `sagev1`: `32.653 ms`
- `flux_single`, sparse `prefix`, `topk=1.0`: `76.785 ms`
- `flux_single`, sparse `prefix`, `topk=0.5`: `37.250 ms`
- `flux_joint`, sparse `custom_035`: `1.329 ms`

Current usage contract:

- `--preset` selects `wan_self`, `flux_joint`, or `flux_single`
- `--pattern` selects `all_selected`, `prefix`, `stride2`, `custom_02`, `custom_035`, or `custom_135`
- `--topk` accepts a comma-separated list such as `1.0,0.75,0.5`
- `--warmup` and `--iters` control measurement repeat count
- `--csv` writes machine-readable output
- raw overrides exist for `--batch`, `--heads-q`, `--heads-kv`, `--seq-q`, `--seq-kv`, `--head-dim`, and `--block-size`

Practical caveat from the current smoke runs:

- `wan_self` dense timing is suspiciously slow in the current single-shot smoke run and should be rechecked before using it as a tuning baseline
- the harness itself is working, and the sparse rows execute correctly on all three preset families

## Change Inventory

The sparse-attention bring-up touched these areas:

- public Python APIs
  - `sage_sparse(...)`
  - `sage_sparse_decode(...)`
  - `sparge_preprocess_topk(...)`
  - `sparge_sage2_attn_meansim_topk_xpu(...)`
  - `sparge_preprocess_topk_decode(...)`
  - `sparge_sage2_decode_meansim_topk_xpu(...)`
- pybind / C++ entrypoints
  - sparse prefill and sparse decode bindings in `ark.cpp`
  - sparse SDPA entrypoints in `sdpa.cpp`
  - declarations in `wrapper/include/sycl_tla_common.hpp`
- kernel launch plumbing
  - sparse routing fields (`lut`, `valid_block_num`, `num_q_blocks`, `num_k_blocks`) in `wrapper/include/sycl_tla_sdpa.hpp`
  - sparse launcher selection and config wiring for prefill and cached decode
- kernel implementation
  - sparse row mapping and scale indexing in `wrapper/include/stla/xe_sage_fwd_kernel.hpp`
  - sparse traversal support in `wrapper/include/stla/xe_sagev1_fwd_mainloop.hpp`
  - dedicated sparse-native mainloop implementation in `wrapper/include/stla/xe_sparse_sagev1_fwd_mainloop.hpp`
- preprocess implementation
  - torch reference preprocess in `__init__.py`
  - Triton-XPU preprocess backend in `sparge_preprocess_triton.py`
  - Triton-XPU ports for fused `pool + sim + quant`, `fill_block_map`, and `block_map_to_lut`
- model / application integration
  - Qwen prefill and decode smoke tests
  - Wan self-attention sparse patch in `wan_sparse_patch.py`
  - Wan runner integration in `run_wan.py`
  - Wan top-k sweep helper in `run_spa.sh`
- validation and benchmarking
  - sparse kernel unit coverage in `wrapper/test/test_sdpa.hpp`
  - preprocess e2e checks in `wrapper/test/test_sparge_preprocess_topk_e2e.py`
  - decode preprocess smoke in `wrapper/test/test_sparge_decode_topk_e2e.py`
  - model-level Qwen tests in `../test/test_qwen_sparse_prefill_e2e.py` and `../test/test_qwen_sparse_decode_e2e.py`
  - top-k benchmark harness in `../test/bench_sparse_topk.py`
  - C++ kernel-only benchmark harness in `wrapper/test/bench_sparse_kernel.cpp`
- docs
  - implementation plans, benchmark runbook, Wan note, and this status note under `docs/`

## Practical Conclusion

The kernel side is usable for raw Sparge-style sparse metadata without the old compatibility densification step.

The preprocess side is now good enough for meaningful end-to-end benchmarking, and the next optimization milestone should focus on the remaining routing-heavy torch stages:

- query-tile routing pooling
- pooled routing score / select path
- any remaining torch-side routing glue that prevents preprocess cost from scaling down with sparsity

Those outputs already feed the current `sage_sparse(...)` path directly without changing the sparse kernel contract.

## Build And Test Commands

Commands used for the latest dedicated sparse-mainloop validation:

Build:

```bash
source /opt/intel/oneapi/setvars.sh
/home/yiliu7/workspace/venvs/ark/bin/cmake --build xbuild -j 4
```

Optional heavier unit-target rebuild:

```bash
source /opt/intel/oneapi/setvars.sh
/home/yiliu7/workspace/venvs/ark/bin/cmake --build xbuild_ut_icpx --target test_ARK_XPU -j 1
```

Build the dedicated C++ kernel benchmark:

```bash
source /opt/intel/oneapi/setvars.sh
/home/yiliu7/workspace/venvs/ark/bin/cmake -B xbuild -DARK_BENCH=ON
/home/yiliu7/workspace/venvs/ark/bin/cmake --build xbuild --target bench_ARK_XPU -j 4
```

Sparse runtime regression checks:

```bash
source /opt/intel/oneapi/setvars.sh
/home/yiliu7/workspace/venvs/ark/bin/python wrapper/test/test_sparge_preprocess_topk_e2e.py
```

```bash
source /opt/intel/oneapi/setvars.sh
/home/yiliu7/workspace/venvs/ark/bin/python wrapper/test/test_sage_sparse_prefill_e2e.py
```

```bash
source /opt/intel/oneapi/setvars.sh
/home/yiliu7/workspace/venvs/ark/bin/python -m pytest -q ../test/test_qwen_sparse_prefill_e2e.py -s
```

C++ kernel-benchmark smoke checks:

```bash
source /opt/intel/oneapi/setvars.sh
./xbuild/bench_ARK_XPU --preset flux_single --pattern prefix --topk 1.0,0.5 --warmup 1 --iters 1
```

```bash
source /opt/intel/oneapi/setvars.sh
./xbuild/bench_ARK_XPU --preset flux_joint --pattern custom_035 --warmup 1 --iters 1
```

```bash
source /opt/intel/oneapi/setvars.sh
./xbuild/bench_ARK_XPU --preset wan_self --pattern all_selected --warmup 0 --iters 1
```

Interpretation:

- `xbuild` shared-module build is the primary compile check for this refactor
- the Python sparse replay and Qwen prefill tests are the primary runtime checks
- `test_ARK_XPU` remains useful, but its rebuild path may be limited by local compile-resource pressure on `sdpa.cpp`
