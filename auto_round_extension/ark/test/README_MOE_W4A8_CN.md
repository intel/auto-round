# W4A8 MoE Kernel (int4 权重 / int8 计算) — 性能与精度

## 概览

`test_moe_w4a8_perf.py` 对 **W4A8** ARK XPU MoE kernel 的 **prefill** 与
**decode** 两个阶段进行性能基准测试，并将其数值精度与 fp32 参考实现以及现有的
W4A16 ARK 路径进行对比。

**W4A8** 的含义:

| 组成部分 | 格式 |
|---|---|
| 权重 (checkpoint 中) | int4 对称量化, `group_size = 32` (auto-round 的打包格式) |
| GEMM 主循环中的权重 | **int8** (`group = -1`, 每个输出通道一个 scale) |
| 激活值 | 在 kernel 内部按 token 动态量化 (absmax) 到 **int8** |
| 累加器 | int32 (`s8 × s8 → s32` DPAS) |
| 输出 | fp16 / bf16 |

## 为什么 int8 计算比 int4 weight-only 更快

Xe DPAS 流水线原生支持 `s8 × s8 → s32` 指令。而 weight-only int4 路径必须先把
nibble 展宽到激活值 dtype，再执行 fp16/bf16 matmul；并且由于 int4 的 scale 是按
K 方向每 32 个元素一组的，累加器每 32 个 K 元素就要折叠一次。这种折叠破坏了
DPAS 流水线达到峰值所需的长 K 累加。

ARK 在 dense GEMM 中已经用 **`AUTO_S8`** 选项解决了这个问题：它把 int4
`group=32` 的权重重新缩放成 int8 `group=-1` 的权重：

```
sxt[e][n][j] = max_{g in block j} |s[e][n][g]| * 8 / 127     # 8 = 2^(4-1), int4 的满量程
w8[e][n][k]  = round( w4[e][n][k] * s[e][n][k / group_size] / sxt[e][n][j] )
```

由于 `|w4| <= 8`，重新缩放后的值满足 `|w8| <= 127` — 转换过程永远不会截断。使用
默认的 block (整个 K 轴) 时，主循环变成**一次完整 K 长度的 int32 累加**，尾部只
需一次标量乘法，这是吞吐最高的配置。

本 kernel 把同样的思路应用到 MoE grouped GEMM 上。该转换只在**模型加载时执行一
次**，而不是每次前向都执行。

## 脚本测量的内容

### 精度表

| 列 | 含义 |
|---|---|
| `block` | 解析出的 `AUTO_S8` 重缩放 block 大小 (K 表示每个输出通道一个 scale) |
| `SNR ref(dB)` | W4A8 与由**反量化后**的 int4 权重构建的 fp32 参考实现对比。它隔离出 int8 激活量化 + AUTO_S8 重缩放引入的误差，不包含 int4 权重量化本身的误差。 |
| `cos ref` | 与同一参考实现的余弦相似度 |
| `maxrel ref` | 最大相对误差，用 `max(|ref|, 0.01 · max|ref|)` 归一化，避免接近 0 的输出主导该指标 |
| `SNR w4a16(dB)` / `cos w4a16` | W4A8 与现有 W4A16 ARK kernel 的对比 — 即调用方切换路径时看到的质量差异 |
| `w4a16 SNR ref` | W4A16 与同一 fp32 参考实现的对比，便于两条路径在同一基准上比较 |

pytest 用例断言 `SNR ref >= 20 dB` 且 `cosine >= 0.99`。按 token 做 absmax 的
int8 激活大约损失 7 bit 尾数，正常情况下会明显高于该门限；低于该门限说明存在
*结构性* bug (scale block 错误、layout 转置错误、expert 偏移错误)，而不仅仅是精
度损失。

### 性能表

| 列 | 含义 |
|---|---|
| `torch(ms)` | 在**预先反量化**的权重上按 expert 执行 `A @ W.T` (反量化在计时区间之外) — 纯 matmul 的 PyTorch 上限 |
| `w4a16(ms)` | 同一阶段现有的 ARK int4 kernel (`moe_gemm_decode` / `moe_gemm_prefill`) |
| `w4a8(ms)` | 新的 int8 计算路径 (`ark.moe_gemm_w4a8`) |
| `TFLOPS` | `total_tokens × N × K × 2 / time` |
| `W GB/s` | 被路由 token 实际访问到的专家权重带宽 (`active_experts × N × K × 1 byte / time`) — decode 访存瓶颈的衡量指标 |
| `vs torch` / `vs w4a16` | 加速比 (`other / w4a8`) |
| `prepack(ms)` | 一次性的 int4 → int8 AUTO_S8 转换开销。只在模型加载时支付，**不是**每次前向都支付。 |

## 测试形状

Qwen3-MoE，与 int4 MoE 工作所针对的形状组一致：

```
hidden_size = 2048,  intermediate_size = 768
num_local_experts = 128,  num_experts_per_tok = 8
int4 对称量化权重, group_size = 32

qwen3 up    (gate/up-proj):  N = 2 × 768 = 1536,  K = 2048
qwen3 down  (down-proj)   :  N = 2048,            K =  768
```

被路由的 expert-token 行数为 `batch × top_k`，以 round-robin 方式分布到 128 个专
家上。默认 batch：prefill 为 `128`，decode 为 `1`；`--all-shapes` 会分别扩展为
`{128, 512, 2048, 8192}` 和 `{1, 2, 8, 16}`。

## 如何运行

### 作为 pytest 测试套件

```bash
cd /path/to/auto_round_extension/ark/test

# 全部 (精度 + 性能，两个阶段)，仅最小 batch
pytest -v -s test_moe_w4a8_perf.py

# 完整 batch 扫描
pytest -v -s test_moe_w4a8_perf.py --all-shapes

# 仅精度 / 仅性能
pytest -v -s test_moe_w4a8_perf.py -k accuracy
pytest -v -s test_moe_w4a8_perf.py -k perf

# 单个阶段
pytest -v -s test_moe_w4a8_perf.py -k decode
```

需要加 `-s` 才能看到打印出的表格。

### 作为独立脚本运行 (不依赖 pytest)

```bash
python test_moe_w4a8_perf.py                       # 两个阶段，最小 batch
python test_moe_w4a8_perf.py --all-shapes          # 完整扫描
python test_moe_w4a8_perf.py --phase decode        # 仅 decode
python test_moe_w4a8_perf.py --skip-accuracy       # 仅性能
python test_moe_w4a8_perf.py --dtype fp16          # fp16 激活
python test_moe_w4a8_perf.py --rescale-group-size 256
python test_moe_w4a8_perf.py --warmup 10 --iters 100
```

任何精度门限未通过时，脚本以非 0 状态码退出。

## Python API

```python
import auto_round_kernel as ark

# 1) 模型加载时的一次性转换。
#    weights : [E, N, K // 2] uint8  (打包的 int4 对称量化权重)
#    scales  : [E, N, K // group_size] fp16/bf16
weights_s8, wscales, block = ark.moe_w4a8_prepack(weights, scales, group_size=32)

# 2) 每次前向 (prefill 或 decode)。
out = ark.moe_gemm_w4a8(
    activations,             # [total_tokens, K] fp16/bf16, 按 expert 排序
    weights_s8,              # [E, N, K] int8
    wscales,                 # [E, N, K // block] fp32
    num_tokens_per_expert,   # [E] int32
    rescale_block_size=block,
    phase="auto",            # "auto" | "decode" | "prefill"
)
```

下面的便捷封装会同时完成两步，并按权重/scale 张量的标识缓存转换结果：

```python
out = ark.moe_w4a8(
    activations, weights, num_tokens_per_expert,
    scales=scales, group_size=32, phase="auto",
)

ark.clear_moe_w4a8_prepack_cache()  # 释放缓存的 int8 权重
ark.moe_w4a8_release_scratch()      # 归还设备端 scratch 内存
```

辅助函数：`ark.moe_w4a8_rescale_block_size(K, group_size, rescale_group_size)`
可以在不做任何分配的情况下解析出有效的 block 大小 (即 `wscales` 的形状)。

## 内存开销

预处理后的权重为 `E × N × K` **字节** (int8)，即打包 int4 权重的 **2 倍**：

| 形状 | int4 打包 | int8 预处理后 |
|---|---|---|
| qwen3 up (E=128, N=1536, K=2048) | 201 MB | 402 MB |
| qwen3 down (E=128, N=2048, K=768) | 100 MB | 201 MB |

由于它们会在进程生命周期内一直保留，W4A8 是用内存换计算吞吐。缓存条目同时会持有源
int4 `weights` / `scales` 张量的引用 (缓存 key 基于指针标识，否则被释放后又重新分
配的显存可能与其他层的权重发生地址碰撞)。如果在某个部署场景下这个权衡不划算，可以
在 `ark.moe_w4a8` 上传 `cache_prepack=False` (或调用
`clear_moe_w4a8_prepack_cache()`)。

## 环境变量

| 变量 | 作用 |
|---|---|
| `ARK_MOE_W4A8_AUTO_S8` | 覆盖 AUTO_S8 重缩放 block 大小。未设置 / `-1` 表示每个输出通道一个 scale (最快)。如果取值不是 `group_size` 和 64 的公倍数，或不能整除 K，则静默回退为 K。 |
| `ARK_MOE_W4A8_DECODE_MAX_TOKENS` | `phase="auto"` 时选择 GEMV 的 token 数上限 (默认 `128`)。 |

## 形状约束

kernel 要求：

* `N % 16 == 0` (GEMV 的 N tile 与 DPAS 的 N tile)
* `K % 64 == 0` (DPAS 的 K tile)
* `group_size % 8 == 0` 且 `K % group_size == 0`
* 解析出的重缩放 block 必须是 64 的倍数并且能整除 K

Qwen3-MoE 的两个 GEMM 都满足以上条件 (`K = 2048` 和 `K = 768`)。

## 状态

W4A8 kernel 是新移植的 SYCL/CuTe 实现，在
`auto_round_kernel/wrapper/include/sycl_tla_moe_w4a8.hpp` 中被标记为
`STATUS: NEEDS-HARDWARE-VALIDATION`。本脚本正是为在真实硬件上验证它而设计的：请
先运行精度扫描 (它能立刻暴露 layout / scale 相关的 bug)，再运行性能扫描来调优
tile 阶梯与 decode 阈值。
