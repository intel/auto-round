# Flux 稀疏注意力：BF16 与 INT8 的一对一对比

日期：2026-08-16

## 范围

本文对比已有的 Flux BF16 稀疏扫描和 2026-08-16 生成的 INT8 稀疏扫描。

两次运行使用相同的有效配置：

- FLUX.1-dev，输出 1024×1024，50 个去噪步，guidance scale 3.5，seed 0
- Prompt：`A cat holding a sign that says hello world`
- qtile256，稀疏 Q block 256，稀疏 K block 64
- Flux pipeline 张量为 BF16；BF16 稀疏使用 `sparge_sage2_attn_meansim_topk_xpu_sdpa`，INT8 稀疏使用带 INT8 Q/K 量化的 `sparge_sage2_attn_meansim_topk_xpu`
- 启用 CPU model offload；每张 XPU 一个进程

BF16 运行于 2026-08-14 收集，INT8 运行于 2026-08-16 收集。因此下表是配置对齐的比较，但不是同一分钟内的严格性能实验。共同 top-k 的 XPU 分配相同。

## 共同 top-k 结果

`INT8/BF16` 是墙钟时间比值；大于 1 表示本次运行中 INT8 用时更长。`Mean RGB diff` 是相同 prompt 和 seed 下，对应 1024×1024 PNG 的逐通道绝对差均值。它是图像差异诊断值，不是图像质量评分。

| top-k | 注意力稀疏率 | BF16 用时（秒） | INT8 用时（秒） | INT8/BF16 | RGB 均值差 | BF16 图像 | INT8 图像 |
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

BF16 的 top-k 0.2 和 0.1 链接来自额外 BF16 扫描目录：`flux_bf16_topk_sweep_1024_parallel_extra_20260814_160550`。

## 运行完整性

- 所有共同 BF16 和 INT8 运行均以 exit code 0 完成。
- 每次运行均执行 2,850 次稀疏注意力：1,900 次 single-stream，950 次 joint-stream。
- 两种 kernel 均报告 unsupported fallback 为 0、runtime fallback 为 0。
- 所有输出文件均为有效的 1024×1024 RGB PNG。

## 仅 BF16 的额外结果

BF16 还运行了 top-k 0.01：111 秒，注意力稀疏率 98.61%，输出为 [flux_topk0p01.png](flux_bf16_topk_sweep_1024_parallel_extra_20260814_160550/flux_topk0p01.png)。当前扫描没有对应的 INT8 top-k 0.01 结果。

## 原始记录

- [BF16 主扫描汇总](flux_bf16_topk_sweep_1024_parallel_20260814_160126/summary.csv)
- [BF16 额外扫描汇总](flux_bf16_topk_sweep_1024_parallel_extra_20260814_160550/summary.csv)
- [INT8 扫描汇总](flux_int8_topk_sweep_20260816/summary.csv)
