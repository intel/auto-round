# Flux Sparse Attention: BF16 vs INT8 Apples-to-Apples Comparison

Date: 2026-08-16

## Scope

This report compares the existing Flux BF16 sparse sweep with the INT8 sparse sweep generated on 2026-08-16.

Both runs use the same effective configuration:

- FLUX.1-dev, 1024×1024 output, 50 denoising steps, guidance scale 3.5, seed 0
- Prompt: `A cat holding a sign that says hello world`
- qtile256, sparse Q block 256, sparse K block 64
- BF16 pipeline tensors; BF16 sparse uses `sparge_sage2_attn_meansim_topk_xpu_sdpa`, while INT8 sparse uses `sparge_sage2_attn_meansim_topk_xpu` with INT8 Q/K quantization
- CPU model offload enabled; one process per XPU

The BF16 runs were collected on 2026-08-14 and the INT8 runs on 2026-08-16. The table is therefore configuration-matched, but not a same-minute controlled performance experiment. The top-k-to-XPU assignment is the same for the common top-k values.

## Common top-k results

`INT8/BF16` is the wall-clock ratio; a value above 1 means INT8 took longer in these runs. `Mean RGB diff` is the mean absolute per-channel difference between the corresponding 1024×1024 PNGs, using the same prompt and seed. It is an image-difference diagnostic, not an image-quality score.

| top-k | attention sparsity | BF16 time (s) | INT8 time (s) | INT8/BF16 | mean RGB diff | BF16 image | INT8 image |
|---:|---:|---:|---:|---:|---:|---|---|
| 1.0 | 0.00% | 119 | 173 | 1.45x | 1.357 | [PNG](flux_bf16_topk_sweep_1024_parallel_20260814_160126/flux_topk1p0.png) | [PNG](flux_int8_topk_sweep_20260816/flux_topk1p0.png) |
| 0.9 | 11.11% | 116 | 173 | 1.49x | 1.150 | [PNG](flux_bf16_topk_sweep_1024_parallel_20260814_160126/flux_topk0p9.png) | [PNG](flux_int8_topk_sweep_20260816/flux_topk0p9.png) |
| 0.8 | 20.83% | 117 | 168 | 1.44x | 6.168 | [PNG](flux_bf16_topk_sweep_1024_parallel_20260814_160126/flux_topk0p8.png) | [PNG](flux_int8_topk_sweep_20260816/flux_topk0p8.png) |
| 0.7 | 30.56% | 116 | 168 | 1.45x | 7.833 | [PNG](flux_bf16_topk_sweep_1024_parallel_20260814_160126/flux_topk0p7.png) | [PNG](flux_int8_topk_sweep_20260816/flux_topk0p7.png) |
| 0.6 | 40.28% | 113 | 169 | 1.50x | 3.121 | [PNG](flux_bf16_topk_sweep_1024_parallel_20260814_160126/flux_topk0p6.png) | [PNG](flux_int8_topk_sweep_20260816/flux_topk0p6.png) |
| 0.5 | 50.00% | 118 | 167 | 1.42x | 1.252 | [PNG](flux_bf16_topk_sweep_1024_parallel_20260814_160126/flux_topk0p5.png) | [PNG](flux_int8_topk_sweep_20260816/flux_topk0p5.png) |
| 0.4 | 61.11% | 114 | 167 | 1.46x | 9.763 | [PNG](flux_bf16_topk_sweep_1024_parallel_20260814_160126/flux_topk0p4.png) | [PNG](flux_int8_topk_sweep_20260816/flux_topk0p4.png) |
| 0.3 | 70.83% | 112 | 165 | 1.47x | 3.150 | [PNG](flux_bf16_topk_sweep_1024_parallel_20260814_160126/flux_topk0p3.png) | [PNG](flux_int8_topk_sweep_20260816/flux_topk0p3.png) |
| 0.2 | 80.56% | 111 | 116 | 1.05x | 10.759 | [PNG](flux_bf16_topk_sweep_1024_parallel_extra_20260814_160550/flux_topk0p2.png) | [PNG](flux_int8_topk_sweep_20260816/flux_topk0p2.png) |
| 0.1 | 90.28% | 111 | 116 | 1.05x | 17.252 | [PNG](flux_bf16_topk_sweep_1024_parallel_extra_20260814_160550/flux_topk0p1.png) | [PNG](flux_int8_topk_sweep_20260816/flux_topk0p1.png) |

The BF16 links for top-k 0.2 and 0.1 are from the extra BF16 sweep directory; the directory name is `flux_bf16_topk_sweep_1024_parallel_extra_20260814_160550`.

## Run integrity

- Every common BF16 and INT8 run completed with exit code 0.
- Every run performed 2,850 sparse attention calls: 1,900 single-stream and 950 joint-stream calls.
- Both kernels reported zero unsupported fallbacks and zero runtime fallbacks.
- All generated files are valid 1024×1024 RGB PNGs.

## BF16-only additional result

BF16 also has a top-k 0.01 run: 111 seconds, 98.61% attention sparsity, with output at [flux_topk0p01.png](flux_bf16_topk_sweep_1024_parallel_extra_20260814_160550/flux_topk0p01.png). There is no matching INT8 top-k 0.01 run in the current sweep.

## Raw records

- [BF16 main summary](flux_bf16_topk_sweep_1024_parallel_20260814_160126/summary.csv)
- [BF16 extra summary](flux_bf16_topk_sweep_1024_parallel_extra_20260814_160550/summary.csv)
- [INT8 summary](flux_int8_topk_sweep_20260816/summary.csv)
