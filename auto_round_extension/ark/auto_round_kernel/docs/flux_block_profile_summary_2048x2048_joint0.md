# FLUX Block Profile Summary

Setup:
- Scope: single block benchmark/profile
- Model path: `FLUX.1-dev`
- Block: `joint` block `0`
- Resolution: `2048x2048`
- Steps: `5`
- Timestep index: `0`
- Text length: `512`
- Image tokens: `16384`
- Joint-stream tokens: `16896`
- CPU offload: disabled

Artifacts:
- Dense benchmark: `flux_benchmark_block_joint0_dense_2048x2048_5steps_seed0_t0_profiled_v2.json`
- Sparse `topk=0.1` benchmark: `flux_benchmark_block_joint0_sparse_2048x2048_5steps_seed0_t0_topk0.1_profiled_v2.json`
- Sparse `topk=0.5` benchmark: `flux_benchmark_block_joint0_sparse_2048x2048_5steps_seed0_t0_topk0.5_profiled_v2.json`
- Dense trace: `profiles/flux_block_joint0_dense_2048x2048_5steps_seed0_t0_v2.json.gz`
- Sparse `topk=0.1` trace: `profiles/flux_block_joint0_sparse_topk0.1_2048x2048_5steps_seed0_t0_v2.json.gz`
- Sparse `topk=0.5` trace: `profiles/flux_block_joint0_sparse_topk0.5_2048x2048_5steps_seed0_t0_v2.json.gz`

## Benchmark summary

| Mode | Avg latency (ms) | Speedup vs dense |
|---|---:|---:|
| Dense | 152.05 | baseline |
| Sparse `topk=0.1` | 112.96 | 25.71% |
| Sparse `topk=0.5` | 122.33 | 19.55% |

Sparse runtime stats:
- `topk=0.1`: `avg_sparsity=0.9015`, no fallback
- `topk=0.5`: `avg_sparsity=0.5000`, no fallback

## Profiling validation

Dense trace:
- shows the dense attention path with `XeFMHAFwdKernel`
- no sparse preprocess or SAGE kernel symbols

Sparse traces:
- show `flux_sparse_patch.py(96): __call__`
- show `sparge_sage2_attn_meansim_topk_xpu`
- show Triton sparse preprocess functions
- show `XeSageFwdKernel`

Conclusion:
- the sparse `v2` traces are real sparse-attention traces
- the earlier sparse block profile without the patch active during profiler replay should not be used

## 1:1 attention kernel comparison

### Dense

Attention side:
- `XeFMHAFwdKernel`: `68.81 ms`

Non-attention block compute:
- `gemm_kernel`: `49.82 ms`

### Sparse `topk=0.1`

Attention side:
- `_triton_bmm_pool_sim_simmean_fuse_quant_xpu`: `21.34 ms`
- `_triton_fill_block_map_kernel`: `0.056 ms`
- `_triton_block_map_to_lut_kernel`: `0.284 ms`
- sort helper: `0.195 ms`
- reduce helper: `0.279 ms`
- index helper: `0.058 ms`
- `XeSageFwdKernel`: `4.11 ms`

Attention subtotal:
- preprocess kernels: `22.22 ms`
- sparse attention kernel: `4.11 ms`
- total sparse attention side: `26.33 ms`

Non-attention block compute:
- `gemm_kernel`: `49.55 ms`

### Sparse `topk=0.5`

Attention side:
- `_triton_bmm_pool_sim_simmean_fuse_quant_xpu`: `21.21 ms`
- `_triton_fill_block_map_kernel`: `0.073 ms`
- `_triton_block_map_to_lut_kernel`: `0.304 ms`
- sort helper: `0.194 ms`
- reduce helper: `0.286 ms`
- index helper: `0.058 ms`
- `XeSageFwdKernel`: `19.95 ms`

Attention subtotal:
- preprocess kernels: `22.13 ms`
- sparse attention kernel: `19.95 ms`
- total sparse attention side: `42.08 ms`

Non-attention block compute:
- `gemm_kernel`: `56.98 ms`

## Interpretation

Dense attention:
- one large FMHA kernel: `68.81 ms`

Sparse `topk=0.1`:
- preprocess cost is substantial but stable at about `22 ms`
- sparse compute kernel drops to about `4.1 ms`
- total attention side falls to about `26.3 ms`

Sparse `topk=0.5`:
- preprocess stays about the same at `22.1 ms`
- sparse compute kernel grows to about `19.9 ms`
- total attention side rises to about `42.1 ms`

Key takeaways:
- lower `topk` mainly reduces the SAGE kernel time, not the preprocess time
- preprocess is a fixed floor of about `22 ms` at this shape
- `topk=0.1` beats `topk=0.5` because the sparse kernel itself is much smaller
- both sparse modes still beat dense on this block

## Practical conclusion

At `2048x2048` joint block 0:
- dense attention kernel cost is about `68.8 ms`
- sparse `topk=0.1` attention path cost is about `26.3 ms`
- sparse `topk=0.5` attention path cost is about `42.1 ms`

This explains the measured block latencies:
- dense: `152.05 ms`
- sparse `topk=0.1`: `112.96 ms`
- sparse `topk=0.5`: `122.33 ms`
