# BF16 稀疏 SDPA K64 vs K32 A/B — 2026-08-14

## 摘要

本次测试将 BF16 稀疏 SDPA qtile256 路径的已选中 block 的 K tile 从 64 token 改为 32 token，
与稠密 `ark.sdpa` 在 head_dim=128 下的 K tile 大小保持一致。这是实验性补丁，尚未整理成最终实现。

结论：**K32 明显更快**。在 topk 0.5 下，BF16 稀疏从慢于稠密 `ark.sdpa` 变为明显快于它。

## 测试设置

- 分支：`feat/sparse-bf16-prefill-v2`
- 设备：Intel Battlemage XPU，`ZE_AFFINITY_MASK=5`
- Benchmark：`benchmarks/bench_sparse_topk.py`
- 形状：`seq_len=75000`，HND，40 头，head_dim 128
- 稀疏配置：`q_tile_override=256`、`sparse_q_block_tokens=256`、`sparse_k_block_tokens=64`
- Benchmark 参数：`--warmup 1 --iters 2`
- 基线 CSV：`benchmarks/results/ab_k64_bf16_20260814_090508.csv`
- K32 CSV：`benchmarks/results/ab_bf16_sdpa_k32_20260814_092404.csv`

## 结果表

下表仅比较 `sparse_sdpa_bf16_kernel_only`。

| topk | K64 延迟 | K32 延迟 | K32 加速 | K64 vs ark | K32 vs ark | effective TFLOPS K64 -> K32 |
|---|---:|---:|---:|---:|---:|---:|
| 1.0 | 2986.014 ms | 1691.257 ms | 1.766x | 0.460x | 0.812x | 38.6 -> 68.1 |
| 0.5 | 1511.047 ms | 862.649 ms | 1.752x | 0.910x | 1.593x | 38.3 -> 67.1 |
| 0.25 | 766.744 ms | 455.005 ms | 1.685x | 1.792x | 3.020x | 38.1 -> 64.2 |

同次运行中稠密 `ark.sdpa` 约为 1374 ms。K64 时，BF16 稀疏 topk 0.5 慢于稠密 ark
（`0.91x`）；K32 时，BF16 稀疏 topk 0.5 变为更快（`1.59x`）。

## 补丁形态

A/B 修改了 `auto_round_kernel/wrapper/include/sycl_tla_sdpa_sparse.hpp` 中的
`launch_sparse_sdpa_prefill_kernel_128`：

```cpp
// K64 原始配置
using ShapeQK = Shape<_256, _64, _32>;
using ShapePV = Shape<_256, _32, _64>;

// K32 实验配置
using ShapeQK = Shape<_256, _32, _32>;
using ShapePV = Shape<_256, _32, _32>;
```

由于路由 LUT 仍以 64-token 稀疏 K block 为单位，实验还修改了
`auto_round_kernel/wrapper/include/stla/xe_sparse_sdpa_fwd_mainloop.hpp`，使每个稀疏路由 block
可以展开为多个物理 K microtile。在 K32 实验中，一个 64-token 路由 block 会执行两次 32-token
mainloop。

## Tile 含义

对于 QK：

```cpp
using ShapeQK = Shape<_256, _32, _32>;
```

含义是：

```text
M = 256   每个 workgroup tile 的 query token 数
N = 32    每个物理 K microtile 的 key token 数
K = 32    head-dim reduction chunk
```

因此 QK 输出的 score tile 是 `256 x 32`。由于 head_dim 是 128，每个物理 tile 仍然需要在特征维上做
4 个 reduction chunk：

```text
head_dim 128 / reduction chunk 32 = 4 个 QK reduction chunk
```

对于 PV：

```cpp
using ShapePV = Shape<_256, _32, _32>;
```

含义是：

```text
M = 256   query 行 / probability 行
N = 32    每个 tile 的 value/output head-dim 列数
K = 32    每个 PV tile reduction 的 selected key/value token 数
```

所以一个 PV microtile 计算 `P[256, 32] x V[32, 32] -> O[256, 32]`。output head_dim 128 会拆成
4 个 32-column output chunk。

一个被选中的 64-token 稀疏路由 block 的伪代码：

```cpp
// K64 原始配置：
// ShapeQK = Shape<_256, _64, _32>
// ShapePV = Shape<_256, _32, _64>
for each selected_route_block_64_tokens {
    scores[256][64] = 0;
    for d = 0; d < 128; d += 32 {
        scores[256][64] += Q[256][d:d+32] * transpose(K[64][d:d+32]);
    }
    softmax_update(scores[256][64]);
    O[256][0:32] += P[256][64] * V[64][0:32];  // 对 output-dim chunk 重复
}

// K32 实验配置：
// ShapeQK = Shape<_256, _32, _32>
// ShapePV = Shape<_256, _32, _32>
for each selected_route_block_64_tokens {
    for micro_tile = 0; micro_tile < 2; ++micro_tile {
        key_start = selected_route_block_64_tokens * 64 + micro_tile * 32;

        scores[256][32] = 0;
        for d = 0; d < 128; d += 32 {
            scores[256][32] += Q[256][d:d+32] * transpose(K[key_start:key_start+32][d:d+32]);
        }
        softmax_update(scores[256][32]);
        O[256][0:32] += P[256][32] * V[key_start:key_start+32][0:32];  // 对 output-dim chunk 重复
    }
}
```

两种配置的 head-dim reduction loop 次数相同（`128 / 32 = 4`）。差别在物理 selected-key 跨度：

```text
K64: 一个 route block -> 一个 256x64 score tile
K32: 一个 route block -> 两个 256x32 score tile
```

K32 对每个 64-token route block 多了一次 microtile iteration，但每个物理 tile 更小。下面的 unitrace
结果显示，更小的 tile 胜出，因为它显著降低了 SEND/SBID 与 memory-write 压力。

## 正确性

重新编译后，`auto_round_kernel/wrapper/test/test_sage_sparse_prefill_e2e.py` 通过。
相关 BF16 SDPA 全选结果：

```text
[sage_sparse_sdpa][torch.bfloat16_all_selected] D=128 max_diff=0.001953 mean_diff=0.000111
```

## Unitrace profile

下面 profile 使用 `topk=0.5`、`seq_len=75000`、HND、head_dim 128、`q_tile_override=256`、
`sparse_q_block_tokens=256`，稀疏路由 block 仍为 64 token。Profiler 会显著拉长墙钟时间，因此下表
使用被 `--include-kernels SPARSESDPAFwdMainloop` 捕获的 kernel GPU metrics，而不是 benchmark 墙钟时间。

文件：

- K32 ComputeBasic：`benchmarks/results/unitrace_k32_compute_20260814/k32_compute.metrics.2889557`
- K32 VectorEngineStalls：`benchmarks/results/unitrace_k32_stalls_20260814/k32_stalls.metrics.2892853`
- K64 ComputeBasic：`benchmarks/results/unitrace_k64_compute_real_20260814/k64_compute_real.metrics.2948755`
- K64 VectorEngineStalls：`benchmarks/results/unitrace_k64_stalls_real_20260814/k64_stalls_real.metrics.2952213`

ComputeBasic，对两个 `SPARSESDPAFwdMainloop` 行聚合：

| metric | K32 | K64 | K64/K32 |
|---|---:|---:|---:|
| GPU time | 1684.6 ms | 3072.4 ms | 1.824x |
| XVE active | 64.718% | 39.934% | 0.617x |
| XVE stall | 35.149% | 59.939% | 1.705x |
| thread occupancy | 99.828% | 99.843% | 1.000x |
| SEND instructions | 18.593G | 81.568G | 4.387x |
| GPU memory read | 43.462 GB | 73.245 GB | 1.685x |
| GPU memory write | 1.633 GB | 8.135 GB | 4.981x |
| LSC byte read | 9653.807 GB | 13389.615 GB | 1.387x |
| LSC byte write | 72.008 MB | 2333.083 GB | 32400.183x |
| SLM read/write | 0 / 0 | 0 / 0 | n/a |

VectorEngineStalls 聚合：

| stall metric | K32 | K64 | delta |
|---|---:|---:|---:|
| SBID | 31.449% | 55.730% | +24.281 pp |
| ALU write | 29.562% | 16.532% | -13.030 pp |
| instruction fetch | 12.329% | 3.638% | -8.691 pp |
| pipe stall | 4.748% | 0.891% | -3.857 pp |
| control | 4.316% | 1.310% | -3.006 pp |
| barrier | 0.240% | 0.531% | +0.291 pp |

## 解读

该 A/B 确认 K64 的已选中 block microtile 是 BF16 稀疏 SDPA 的主要性能问题。这个结果并不能单独证明
存在 register spill；unitrace 也没有直接给出 GRF spill 计数器。但它明确显示 K64 被 memory/SEND
scoreboard 限制：SBID stall 从 31.4% 升到 55.7%，XVE active 从 64.7% 降到 39.9%，SEND 指令增加
4.4x，memory write 增加约 5.0x。SLM 流量仍为 0，因此这些指标没有显示 SLM-backed spill。巨大的 LSC
write 差异是这里能采到的最佳 spill/pressure proxy，符合 K64 带来更高 scratch/memory 压力的判断。

建议：BF16 稀疏 SDPA 的已选中 block 路径保持 K32；除非后续 compiler/offline assembly 数据证明可以
用其它方式消除 write/SBID 压力，否则 K64 应视为性能回退。
