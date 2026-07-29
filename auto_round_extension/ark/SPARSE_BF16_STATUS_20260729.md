# Sparse BF16 Status 2026-07-29

## Scope

This note captures the current sparse attention status on Wednesday, July 29, 2026 after rebuilding with oneAPI 2025 and rerunning focused accuracy and performance checks on XPU 7.

Environment:

- `ONEAPI_ROOT=/home/yiliu4/intel/oneapi`
- Python: `/home/yiliu4/workspace/auto-round-prefill-clean-sparse-pr/auto_round_extension/ark/ark-torch-212/bin/python`
- device: `XPU_MASK=7`
- native BF16 sparse enabled with `SAGE_ATTN_XPU_SPARSE_BF16_NATIVE=1`
- rebuilt extension: `auto_round_kernel/xbuild_bf16_v2/auto_round_kernel_xpu.cpython-313-x86_64-linux-gnu.so`

Kernel config under test:

- `head_dim=128`
- `q_tile_override=256`
- `sparse_q_block_tokens=256`
- `sparse_k_block_tokens=64`
- layout: `HND`

## Accuracy Status

### Fixed-LUT sparse kernel

`fixed-LUT sparse` means the sparse mask is explicitly provided by `lut` and `valid_block_num`, and the kernel is checked against an exact dense BF16 reference with the same mask.

Target correctness case:

- shape: `B=1, Hq=32, Hkv=8, Sq=512, Skv=768, D=128`
- sparse pattern: first Q tile uses K blocks `[0, 1, 2]`, second Q tile uses `[9, 10, 11]`
- reference: explicit BF16 `Q @ K^T`, add exact mask, softmax, then `P @ V`

Result:

- `bf16_sparse` vs exact BF16 reference:
  - `max_diff=0.003906`
  - `mean_diff=0.000127`

Comparison on the same case:

- `int8_sparse` vs exact BF16 reference:
  - `max_diff=3.826172`
  - `mean_diff=0.344735`
- `int8_sparse` vs `bf16_sparse`:
  - `max_diff=3.826172`
  - `mean_diff=0.344735`

Conclusion:

- `bf16_sparse` is accurate for the exact fixed-LUT sparse kernel path.
- `int8_sparse` is materially less accurate than `bf16_sparse` on the same case.

### Heuristic top-k end-to-end path

The `sparge_sage2_attn_meansim_topk_xpu_bf16` path is heuristic. It should not be treated as an exact correctness reference for a handcrafted LUT mask.

Because of that, the BF16 sparse test now treats:

- exact correctness: `sage_sparse_bf16` vs explicit masked BF16 reference
- heuristic top-k path: benchmark/perf path, not exact reference

## Test Harness Status

File updated:

- `auto_round_kernel/wrapper/test/test_sage_sparse_prefill_e2e.py`

Current test harness behavior:

- force-loads the rebuilt BF16-capable extension from `xbuild_bf16_v2`
- uses explicit masked BF16 math as the BF16 sparse correctness reference
- no longer treats heuristic BF16 top-k output as an exact oracle for the fixed-LUT kernel

Verified run:

```bash
source benchmarks/source_env_xpu67_oneapi2025.sh
export XPU_MASK=7
export SAGE_ATTN_XPU_SPARSE_BF16_NATIVE=1
${ARK_BENCH_PYTHON} auto_round_kernel/wrapper/test/test_sage_sparse_prefill_e2e.py
```

Observed result:

- full script exits successfully under oneAPI 2025
- INT8 sparse cases pass
- BF16 sparse exact-reference cases pass

## Performance Status

### Medium-size exact sparse case

Shape:

- `B=1, Hq=32, Hkv=8, Sq=512, Skv=768, D=128`

Latency:

- dense BF16 with exact mask: `0.105 ms`
- `bf16_sparse`: `0.096 ms`
- `int8_sparse`: `0.090 ms`
- INT8 Q/K quantization only: `0.027 ms`

Speedup vs dense BF16 exact-mask baseline:

- `bf16_sparse`: `1.093x`
- `int8_sparse`: `1.175x`
- `int8_sparse + quant`: `0.902x`

Conclusion:

- kernel-only, `int8_sparse` is slightly faster
- including quantization, this small/medium case does not favor the INT8 path
- `bf16_sparse` is the better accuracy point here

### Long-sequence benchmark

Shape:

- `B=1, Hq=40, Hkv=40, S=75000, D=128`
- layout: `HND`
- `topk=0.5`
- `q_tile_override=256`
- `sparse_q_block_tokens=256`
- `sparse_k_block_tokens=64`

Sparse selection stats:

- `selected_ratio=0.502557`
- `selected_blocks_per_row=588.997`

Sparse timings:

- `preprocess=127.809 ms`
- `int8_sparse_kernel=390.990 ms`
- `int8_sparse_e2e=501.369 ms`
- `bf16_sparse_kernel=804.864 ms`
- `bf16_sparse_e2e=919.896 ms`

Dense BF16 baselines on the same shape:

- `dense_torch_sdpa_bf16=2154.809 ms`
- `dense_ark_sdpa_bf16=2494.633 ms`
- `dense_ark_sagev1_bf16=10848.612 ms`

Speedup vs dense torch BF16 SDPA:

- `int8_sparse_e2e`: about `4.30x`
- `bf16_sparse_e2e`: about `2.34x`

Speedup vs dense ARK BF16 SDPA:

- `int8_sparse_e2e`: about `4.98x`
- `bf16_sparse_e2e`: about `2.71x`

Conclusion:

- at `75k / 40 / 128`, both sparse paths are substantially faster than dense BF16
- `int8_sparse` remains the fastest path
- `bf16_sparse` is slower than `int8_sparse` but still much faster than dense BF16

## Current Summary

- BF16 sparse fixed-LUT kernel correctness is in good shape after rebuilding with oneAPI 2025.
- The previous BF16 mismatch was caused by stale extension loading and an unreliable masked-XPU reference path in the test harness.
- For long sequences, BF16 sparse is a valid accuracy-oriented sparse path with meaningful speedup over dense BF16.
- INT8 sparse is still the fastest sparse option, but it gives up substantial accuracy relative to BF16 sparse on the exact fixed-LUT check.
