# Sparse BF16 状态 2026-07-29

## 范围

本文档记录了 2026 年 7 月 29 日（星期三）在 oneAPI 2025 环境下重新构建并在 XPU 7 上复测后的 sparse attention 当前状态。

环境：

- `ONEAPI_ROOT=/home/yiliu4/intel/oneapi`
- Python: `/home/yiliu4/workspace/auto-round-prefill-clean-sparse-pr/auto_round_extension/ark/ark-torch-212/bin/python`
- 设备：`XPU_MASK=7`
- 原生 BF16 sparse 打开：`SAGE_ATTN_XPU_SPARSE_BF16_NATIVE=1`
- 重建后的扩展：`auto_round_kernel/xbuild_bf16_v2/auto_round_kernel_xpu.cpython-313-x86_64-linux-gnu.so`

本次测试的 kernel 配置：

- `head_dim=128`
- `q_tile_override=256`
- `sparse_q_block_tokens=256`
- `sparse_k_block_tokens=64`
- layout：`HND`

## 精度状态

### 固定 LUT sparse kernel

这里的 `fixed-LUT sparse` 指 sparse mask 由 `lut` 和 `valid_block_num` 显式给出，然后用相同 mask 的精确 dense BF16 参考结果进行校验。

目标正确性用例：

- shape：`B=1, Hq=32, Hkv=8, Sq=512, Skv=768, D=128`
- sparse pattern：第一个 Q tile 使用 K blocks `[0, 1, 2]`，第二个 Q tile 使用 `[9, 10, 11]`
- 参考实现：显式 BF16 `Q @ K^T`，加精确 mask，softmax，然后 `P @ V`

结果：

- `bf16_sparse` 对比精确 BF16 参考：
  - `max_diff=0.003906`
  - `mean_diff=0.000127`

同一用例下的对比：

- `int8_sparse` 对比精确 BF16 参考：
  - `max_diff=3.826172`
  - `mean_diff=0.344735`
- `int8_sparse` 对比 `bf16_sparse`：
  - `max_diff=3.826172`
  - `mean_diff=0.344735`

结论：

- `bf16_sparse` 在精确 fixed-LUT sparse kernel 路径上是准确的。
- 同一用例下，`int8_sparse` 的精度明显差于 `bf16_sparse`。

### 启发式 top-k 端到端路径

`sparge_sage2_attn_meansim_topk_xpu_bf16` 是启发式路径，不应该作为手工构造 LUT mask 的精确正确性参考。

因此当前 BF16 sparse 测试的含义是：

- 精确正确性：`sage_sparse_bf16` 对比显式 masked BF16 参考
- 启发式 top-k 路径：用于 benchmark / perf，不作为精确 oracle

## 测试状态

更新文件：

- `auto_round_kernel/wrapper/test/test_sage_sparse_prefill_e2e.py`

当前测试脚本行为：

- 强制加载 `xbuild_bf16_v2` 中带 BF16 sparse 能力的扩展
- 使用显式 masked BF16 数学结果作为 BF16 sparse 正确性参考
- 不再把启发式 BF16 top-k 输出当成 fixed-LUT kernel 的精确标准答案

验证命令：

```bash
source benchmarks/source_env_xpu67_oneapi2025.sh
export XPU_MASK=7
export SAGE_ATTN_XPU_SPARSE_BF16_NATIVE=1
${ARK_BENCH_PYTHON} auto_round_kernel/wrapper/test/test_sage_sparse_prefill_e2e.py
```

观察结果：

- 整个脚本在 oneAPI 2025 下成功退出
- INT8 sparse 用例通过
- BF16 sparse 精确参考用例通过

## 性能状态

### 中等尺寸精确 sparse 用例

Shape：

- `B=1, Hq=32, Hkv=8, Sq=512, Skv=768, D=128`

时延：

- 带精确 mask 的 dense BF16：`0.105 ms`
- `bf16_sparse`：`0.096 ms`
- `int8_sparse`：`0.090 ms`
- 仅 INT8 Q/K 量化：`0.027 ms`

相对 dense BF16 精确 mask 基线的加速比：

- `bf16_sparse`：`1.093x`
- `int8_sparse`：`1.175x`
- `int8_sparse + quant`：`0.902x`

结论：

- 只看 kernel，`int8_sparse` 略快
- 如果把量化时间算进去，这个中等尺寸用例并不偏向 INT8 路径
- 这里 `bf16_sparse` 是更好的精度点

### 长序列 benchmark

Shape：

- `B=1, Hq=40, Hkv=40, S=75000, D=128`
- layout：`HND`
- `topk=0.5`
- `q_tile_override=256`
- `sparse_q_block_tokens=256`
- `sparse_k_block_tokens=64`

sparse 选择统计：

- `selected_ratio=0.502557`
- `selected_blocks_per_row=588.997`

sparse 时延：

- `preprocess=127.809 ms`
- `int8_sparse_kernel=390.990 ms`
- `int8_sparse_e2e=501.369 ms`
- `bf16_sparse_kernel=804.864 ms`
- `bf16_sparse_e2e=919.896 ms`

同一 shape 下的 dense BF16 基线：

- `dense_torch_sdpa_bf16=2154.809 ms`
- `dense_ark_sdpa_bf16=2494.633 ms`
- `dense_ark_sagev1_bf16=10848.612 ms`

相对 dense torch BF16 SDPA 的加速比：

- `int8_sparse_e2e`：约 `4.30x`
- `bf16_sparse_e2e`：约 `2.34x`

相对 dense ARK BF16 SDPA 的加速比：

- `int8_sparse_e2e`：约 `4.98x`
- `bf16_sparse_e2e`：约 `2.71x`

结论：

- 在 `75k / 40 / 128` 下，两条 sparse 路径都明显快于 dense BF16
- `int8_sparse` 仍然是最快路径
- `bf16_sparse` 虽然慢于 `int8_sparse`，但仍然显著快于 dense BF16

## 当前结论

- 在 oneAPI 2025 重新构建后，BF16 sparse fixed-LUT kernel 的正确性状态是好的。
- 之前 BF16 mismatch 的主要原因是旧扩展被错误加载，以及测试里使用了不可靠的 masked XPU 参考路径。
- 对长序列来说，BF16 sparse 是一个兼顾精度的 sparse 路径，并且相对 dense BF16 有明显加速。
- `int8_sparse` 仍然是最快的 sparse 方案，但在精确 fixed-LUT 检查里，相比 `bf16_sparse` 牺牲了明显精度。

## 2026-07-30 复测摘要

在 2026 年 7 月 30 日（星期四），对 `75k / topk=0.5` 用例进行了重新测试。测试使用空闲的 `XPU 7`、oneAPI 2025，并且显式加载了带 BF16 sparse 能力的扩展 `auto_round_kernel/xbuild_bf16_v2/auto_round_kernel_xpu.cpython-313-x86_64-linux-gnu.so`。

Shape 和配置：

- `B=1, Hq=40, Hkv=40, S=75000, D=128`
- layout：`HND`
- `topk=0.5`
- `q_tile_override=256`
- `sparse_q_block_tokens=256`
- `sparse_k_block_tokens=64`
- warmup `2`，iters `3`

sparse 选择统计：

- `selected_ratio=0.502557`
- `selected_blocks_per_row=588.997`

时延和吞吐摘要：

| Mode | Latency (ms) | Speedup vs Torch BF16 | Speedup vs SageV1 BF16 | Baseline TFLOPS | Effective TFLOPS |
|---|---:|---:|---:|---:|---:|
| `dense_torch_sdpa_bf16` | 1263.297 | 1.000 | 0.595 | 91.190 | 91.190 |
| `dense_sagev1_bf16` | 751.410 | 1.681 | 1.000 | 153.312 | 153.312 |
| `sparse_qtile256_row64k_kernel_only` | 398.381 | 3.305 | 1.944 | 289.171 | 145.325 |
| `sparse_qtile256_row64k_e2e` | 514.271 | 2.560 | 1.506 | 224.007 | 112.576 |
| `sparse_bf16_qtile256_row64k_kernel_only` | 812.091 | 1.556 | 0.925 | 141.856 | 71.291 |
| `sparse_bf16_qtile256_row64k_e2e` | 928.958 | 1.360 | 0.809 | 124.010 | 62.322 |

说明：

- `kernel_only` 表示 preprocess 之后仅 kernel 的时间。
- `e2e` 表示 preprocess 加 kernel 的总时间。
- 这次复测中，BF16 sparse 比 dense torch BF16 SDPA 更快，但仍然慢于 dense `sagev1_bf16`。
