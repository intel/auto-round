Dense vs sparse block benchmark/profile comparison for FLUX joint block 0 at 2048x2048 (`16384` image tokens, `512` text tokens), timestep index `0`.

## V2 Benchmark Table

| Variant | Avg ms | Median ms | Min ms | Max ms | Speedup vs dense | Delta vs dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 152.049 | 149.379 | 146.354 | 160.414 | 1.000x | 0.000 ms |
| Sparse topk=0.5 | 122.331 | 121.073 | 120.493 | 125.428 | 1.243x | -29.718 ms |
| Sparse topk=0.1 | 112.958 | 113.296 | 108.114 | 117.464 | 1.346x | -39.091 ms |

Additional comparison inside the sparse runs:

| Sparse pair | Avg ratio | Delta |
| --- | ---: | ---: |
| topk=0.1 vs topk=0.5 | 1.083x | -9.373 ms |

## V2 Trace Breakdown Table

Device/profile trace files:

- Dense: `profiles/flux_block_joint0_dense_2048x2048_5steps_seed0_t0_v2.json.gz`
- Sparse topk=0.1: `profiles/flux_block_joint0_sparse_topk0.1_2048x2048_5steps_seed0_t0_v2.json.gz`
- Sparse topk=0.5: `profiles/flux_block_joint0_sparse_topk0.5_2048x2048_5steps_seed0_t0_v2.json.gz`

Named sparse-path timings extracted from the `v2` traces:

| Trace | Whole profiled block | `sparge_sage2_attn_meansim_topk_xpu` | `sparge_preprocess_topk` | `_run_triton_xpu_preprocess` | `_triton_bmm_pool_sim_simmean_fuse_quant_xpu` | `sage_sparse` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 155.664 ms | N/A | N/A | N/A | N/A | N/A |
| Sparse topk=0.1 | 119.042 ms | 62.835 ms | 61.691 ms | 60.213 ms | 21.343 ms (3 calls) | 1.101 ms |
| Sparse topk=0.5 | 141.834 ms | 62.672 ms | 61.535 ms | 60.984 ms | 21.214 ms (3 calls) | 1.096 ms |

Smaller Triton preprocess stages from the same traces:

| Trace | `_fill_block_map_triton` | `_triton_fill_block_map_kernel` | `_block_map_lut_triton` | `_triton_block_map_to_lut_kernel` |
| --- | ---: | ---: | ---: | ---: |
| Sparse topk=0.1 | 0.345 ms | 0.056 ms | 0.254 ms | 0.284 ms |
| Sparse topk=0.5 | 0.341 ms | 0.073 ms | 0.272 ms | 0.304 ms |

Key observations from the `v2` traces:

- The fused Triton pool/sim/quant kernel is the largest single named Triton preprocess kernel in both sparse traces.
- Its total device time is almost unchanged between sparse topk=0.1 and topk=0.5:
  - `21.343 ms` vs `21.214 ms`
- The full Triton preprocess path is also almost unchanged:
  - `60.213 ms` vs `60.984 ms`
- `sage_sparse` itself is very small in these traces:
  - about `1.10 ms`
- So the sparse topk=0.1 vs topk=0.5 runtime difference is **not** explained by `_triton_bmm_pool_sim_simmean_fuse_quant_xpu` alone.
- The dominant cost in the profiled sparse block is preprocess, and that preprocess cost is currently only weakly sensitive to topk for this Flux block.

Saved results:
- Benchmark dense: `flux_benchmark_block_joint0_dense_2048x2048_5steps_seed0_t0_profiled.json`
- Benchmark sparse: `flux_benchmark_block_joint0_sparse_2048x2048_5steps_seed0_t0_topk0.1_profiled.json`
- Profile dense: `profiles/flux_block_joint0_dense_2048x2048_5steps_seed0_t0.json.gz`
- Profile sparse: `profiles/flux_block_joint0_sparse_topk0.1_2048x2048_5steps_seed0_t0.json.gz`

Benchmark summary:
- Dense avg: `149.221 ms`
- Sparse topk=0.1 avg: `106.579 ms`
- Delta: `42.642 ms`
- Sparse speedup vs dense: `28.58%`
- Dense / sparse ratio: `1.400x`
- Sparse stats: `sparse_calls=4`, `unsupported_fallbacks=0`, `runtime_fallbacks=0`, `avg_sparsity=0.9015`

Profiler observations:
- Both traces are block-only traces, tagged with user annotation `flux_block`.
- No full-pipeline module hits were found in the trace names.
- Host/device copy overhead is negligible in both traces:
  - Dense `aten::copy_ + aten::to + aten::_to_copy`: about `0.844 ms` total
  - Sparse `aten::copy_ + aten::to + aten::_to_copy`: about `0.909 ms` total
- The profiler does not expose an obvious sparse-kernel symbol name; both traces are still dominated by:
  - one FMHA-style attention kernel
  - `gemm_kernel`
- Top kernel totals observed:
  - Dense:
    - FMHA kernel: `66.752 ms`
    - `gemm_kernel`: `46.710 ms`
  - Sparse topk=0.1:
    - FMHA kernel: `65.428 ms`
    - `gemm_kernel`: `54.616 ms`

Interpretation:
- The block benchmark shows a real sparse speedup at this shape.
- The block profiler confirms the run is isolated to a single block and not dominated by CPU offload copies.
- The profiler kernel names are not sufficient by themselves to explain the sparse speedup, because the sparse path is not surfaced with a distinctive kernel symbol here.
- Use the benchmark result plus sparse runtime stats as the primary evidence that the sparse path is active and faster on this block.
