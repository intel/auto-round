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

## 性能目标与 roofline

本 kernel 的目标是 **prefill > 100 TFLOPS**、**decode 权重带宽 > 300 GB/s**。
prefill 目标是否*可能*达到，取决于**路由**而不是 kernel 本身：W4A8 grouped GEMM
对每个活跃专家的 int8 权重只读一次，而每读一个权重字节要做 `2 × rows_per_expert`
次浮点运算，因此

```
计算强度   = 2 × rows_per_expert                    [FLOP / byte]
TFLOPS   <= 2 × rows_per_expert × 权重带宽
rows_per_expert = batch × top_k / active_experts
```

其中 `N`、`K` 因子相互抵消 — 只有路由起作用：

| 模型 token 数 | 路由行数 | 每专家行数 | 达到 100 TFLOPS 所需带宽 |
|---|---|---|---|
| 128 (prefill 默认 batch) | 1024 | 8 | 6250 GB/s |
| 512 | 4096 | 32 | 1563 GB/s |
| 2048 | 16384 | 128 | 391 GB/s |
| 4096 (`test_perf_prefill_compute_bound`) | 32768 | 256 | 195 GB/s |
| 8192 (`--all-shapes`) | 65536 | 512 | 98 GB/s |

因此默认 batch 下约 4.5 TFLOPS **并不是 kernel 的缺陷**：在每专家 8 行、kernel 实
测约 285 GB/s 权重带宽的条件下，上限就是 `2 × 8 × 285e9 = 4.56 TFLOPS` — 正好等于
实测值，说明 kernel 已经跑在 DRAM roofline 上。要在该形状上达到 100 TFLOPS 需要
6.25 TB/s，比当前任何 GPU 都高 10 倍以上。在带宽约 285 GB/s 的设备上，该目标最早
在每专家约 176 行 (约 2816 个模型 token) 时才变得可达，这正是
`test_perf_prefill_compute_bound` 使用 4096 个模型 token 的原因。

因此性能表在实测值旁边额外打印 `rows/E` 和 `BW@100T`，并在每次扫描后输出结论：

```
targets [prefill]: prefill compute > 100 TFLOPS
  device copy bandwidth probe: 400 GB/s
  qwen3 up     tokens=1024   rows/E=8.0        4.56 TFLOPS vs 100 -> N/A (bandwidth bound: ...)
  qwen3 down   tokens=32768  rows/E=256.0    102.40 TFLOPS vs 100 -> PASS
```

当设备带宽探测 (每次运行执行一次的大块 device-to-device 拷贝) 表明该路由下目标不
可达时，该行显示 `N/A` 而不是 `FAIL`。该结论默认只用于提示；加上
`--enforce-targets` 可以把它变成硬断言。

### 为什么小 batch 下 `vs w4a16` 小于 1.0

同样的计算强度分析也解释了 `vs w4a16` 这一列。W4A8 需要传输 int4 路径 **2 倍的权
重字节**(每个元素 1 字节 vs 半字节)，换来约 2 倍的 DPAS 峰值，所以只有在 GEMM 变
成计算受限之后才会占优：

```
交叉点 rows/expert ~= int8 峰值 TOPS / (4 × 权重带宽)
```

按约 233 TOPS 的 int8 DPAS 和约 285 GB/s 计算，交叉点约为每专家 200 行 (约 3200 个
模型 token)。decode (每专家 1 行) 和小 batch prefill 都远低于该点，所以
0.55–0.71× 是预期结果：W4A8 是面向大 batch prefill 的优化，在 decode 阶段只能通过
改善**访存**路径来获益。

## Decode：合并访存的 K-split 映射

decode GEMV 最初为**每个输出元素分配一个 work-item**：sub-group 中的第 `l` 号 lane
负责第 `n0 + l` 列，并独自遍历整个 K 轴。这样相邻 lane 读取的地址相差 `K` 字节，一
条 load 指令要触及 16 条不同的 cache line，而每条 line 取回的 64 字节中只用到 16
字节。batch 1 时 kernel 还只启动 `total_tokens × N/16` 个 sub-group (up-proj 为
768 个 SIMD16 work-item)，远不足以掩盖访存延迟。

修复方式与已经让 FP8 decode 达标的 **K-split** 映射相同
(`sycl_tla_moe_decode.hpp` 中的 `launch_fp8_ksplit`)：一个 sub-group 协作处理
`NCOLS` 个输出列，第 `l` 号 lane 负责每个 256 元素步长内位于 `l × 16` 的 16 个连续
K 元素。这样每条 load 覆盖 **256 个连续权重字节**，grid 规模扩大约 16 倍，每个输出
元素再用一次 `sycl::reduce_over_group` 归约各 lane 的部分和。

循环采用 *block 在外* 的结构 (先遍历 AUTO_S8 重缩放 block，再在 block 内遍历 K)，
因此 block scale 被提升为标量，热循环中没有除法；并且与 FP8 版本不同，对 block 大
小没有 2 的幂约束。算术过程保持不变：每个 lane 在每个 block 内累加 int32 部分和，
乘以 block scale，在 sub-group 内求和，再乘以每 token 的激活 scale。差异仅在于浮点
**求和顺序** (先按 lane 累加再跨 lane 归约，而不是由单个 lane 累加所有 block)，因
此两种映射的输出并非逐位相同；`test_decode_ksplit_matches_legacy` 断言两者的 SNR
高于 40 dB、余弦相似度高于 0.9999 — 这远比任何真实的映射错误所能达到的精度更严格。

该映射要求 `N % 16 == 0`、重缩放 block 是 16 的倍数且不小于 256、`K % block == 0`。
不满足时 (例如显式指定 `--rescale-group-size 64`) 会自动回退到原 kernel。

decode 每步还**少启动一个 kernel**：每个 token 的 expert id 改为在激活量化 kernel
内部推导 (它本来就是每个 token 一个 sub-group)，不再单独启动
`fill_expert_id_per_token`。batch 1 时整个 GEMV 只有约 45 µs，省下一次 launch 并非
可忽略的噪声。

## Prefill：访存消息宽度与寄存器压力

在 prefill 规模下，grouped GEMM 旁边还有两项开销，而在本次改动之前它们都是按最坏
情况付出的。

**激活量化。** 把路由后的激活转成 int8 是一个纯流式过程——读两遍 `[T, K]` (先求
absmax，再量化)，写一遍 `[T, K]` 的 int8。在 32768 条路由行、`K = 2048` 时这就是约
200 MB，与 qwen3 up-proj GEMM 为权重流动的约 400 MB 处于同一量级，因此它是整次调用
中实打实的一部分开销，而不是可忽略的前置步骤。原先的映射
(`k = lane; k += SG_SIZE`) 用的是 sub-group 能发出的**最窄**消息：16 个 lane × 1 个
16 位元素是一条 32 字节 load，16 个 lane × 1 个 int8 则是一条 **16 字节 store**——
每条 store 消息只占 cache line 的四分之一。这正是 decode GEMV 在 K-split 重写之前
存在的问题，而在那里修复它带来了 1.09–1.93 倍的收益。

现在每个 lane 负责 `VEC` 个**连续**元素，因此一条消息覆盖 `SG_SIZE × VEC` 个连续元
素：`VEC = 8` 时是 256 字节激活与 128 字节 int8。`VEC` 由 K 决定——`K % 128 == 0`
时取 8 (所有已上线的 MoE 形状均满足：768 / 1536 / 2048 / 3072)，否则取 4，而
`K % 64 == 0` 的形状约束保证后者总能成立——基址未对齐时则回退到标量 kernel。任何会
产生舍入的步骤都没有被重排：每个 lane 的局部归约用的是 `fmax`，它精确且与顺序无
关，因此两种映射送进 sub-group 归约的 absmax 完全相同，每个元素的量化结果也完全相
同。`test_act_quant_vec_matches_scalar` 断言两者**逐位相同**，而
`ARK_MOE_W4A8_ACT_QUANT_VEC=0` 可恢复标量映射以便做 A/B 对比测量。

**在途请求数 (loads in flight)。** 加宽消息解决的是每个**请求**搬运多少字节，并没有
改变一个 work-item 同时挂起多少个请求。这一遍扫描 K 的循环次数是运行期决定的
(`steps = K / (SG_SIZE × VEC)`)，且每个向量都折进同一个 `local_max`，因此循环读起来
就是：发一条 load，停下来等它返回，做一次 `fmax`，再来一遍。Xe core 是顺序执行的，而
`fmax` 在没有 fast-math 时不会被重结合，所以一个线程大约只保持**一条** 256 字节的
load 在途。这是 Little 定律的问题，而不是带宽的问题：640 个并发 sub-group × 256 字节
只有约 160 KB 的在途读取，而一块约 400 GB/s 的设备要在约 1 µs 的访存延迟下保持忙碌需
要约 400 KB。这与 decode GEMV 每次迭代加载两个 chunk 是同一个论证。

现在每次迭代先加载 `UNROLL` 个**互相独立**的向量，然后才开始消费它们，并把它们归约到
`UNROLL` 个各自独立的局部最大值上，使这些 load 也不必串行等待累加器链；量化那一遍同样
按此批量化其 load。`steps % UNROLL` 个向量交给尾循环处理——`K = 768` 时一个 lane 要走
6 个向量，因此在默认 `UNROLL = 4` 下尾循环是真实会执行的代码，而不是形式上的补充。任
何会产生舍入的步骤都没有改变 (`fmax` 精确且与顺序无关，因此各局部最大值合并后逐位相
同)，且 `UNROLL = 1` 与批量化之前的 kernel 逐条指令一致，所以
`ARK_MOE_W4A8_ACT_QUANT_UNROLL=1` 是精确的 A/B 基线。
`test_act_quant_unroll_matches` 会在"K 能整除展开深度"和"K 会留下尾巴"两种情况下断言
逐位相同。

**GEMM 尾声 (epilogue)。** 原先的主循环同时保持两份 C fragment：int32 的 DPAS 累加
器 (每个 AUTO_S8 重缩放 block 清零一次)，以及一份必须跨 block 存活的 float 影子——
因为每个 block 的权重 scale 必须在下一个 block 覆盖累加器之前应用。在**整个**主循
环期间，每个 lane 都要把两者留在 GRF 中：

| tile | sub-group C fragment | 每 lane int32 寄存器 | + 每 lane float 寄存器 |
|---|---|---|---|
| `128x128` | 32 × 32 | 64 | 64 |
| `128x256` | 32 × 64 | 128 | 128 |

在 `grf_size<256>` 下每个 lane 只有 256 个寄存器，因此 `128x128` 时仅 float 影子就
在整个主循环里占掉了四分之一的寄存器堆，而 `128x256` 时两份 fragment 加起来*就是*
整个寄存器堆——留给暂存 A/B tile 的空间为零。这正是 N 方向 256 的 tile 过去在
[实测得到的默认值](#实测得到的默认值)中要多付 35–50% 的原因；去掉 float 影子之后，
整张 tile 表已经收敛到 0–8% 的区间内。而在默认重缩放 block 下它完全是白付的开销：
`blks == 1` (AUTO_S8 `group=-1` 默认值) 根本没有需要跨 block 携带的东西，scale 完全
可以在写出时再折进去。该路径现在不再分配 float fragment，而是一次遍历就应用
`scale_b[col] × scale_a[row]`——与参考的稠密 int8 GEMM 中 `AccumBlock == false`
分支的结构一致。

同一段 epilogue 也不再对越界元素做分支跳过。grouped GEMM 中每个专家的 M 是任意的，
因此 M 边缘的 tile 是部分 tile，**store** 必须保持谓词保护——但 scale 的 **load**
不必：改为把下标钳制到合法范围内，两次读取就都变成了相对于统一基址、编译期偏移的无
条件 load。正是这一点让编译器能把逐元素的读取收敛成一个 sub-group 的 fragment 实际
覆盖的那少数几个不同地址 (同一行组的所有 lane 共享 `scale_a[row]`，而一个 lane 对
它负责的每一行都重复同一个 `scale_b[col]`)；在原先的 `continue` 保护下，每次读取都
位于各自独立的基本块中，这些优化一个也做不了。

**内部 tile。** 上述两类保护——store 谓词与两次下标钳制——只有在 tile 越过专家行数
边界或 N 边界时才是必要的，而是否越界在整个 work-group 内是一致的：`m`、`n` 与 tile
坐标都是 work-group 内统一的量。把这个判断从"每个 fragment 元素一次"提到"每个 tile
一次"，可以为内部 tile 的每个输出元素省下约 4 条指令；而在 `K = 768` (每个 tile 只有
12 个 k-tile) 时，epilogue 在整个 tile 耗时中占据可观的比例。边缘 tile 仍走原来的
保护路径，`ARK_MOE_W4A8_PREFILL_FULL_TILE=0` 也可以把所有 tile 都切回该路径；由于算
术运算及其顺序完全没变，两条路径**逐位相同**，
`test_full_tile_epilogue_matches_predicated` 正是在"每个专家恰好有一个完整 tile 和一
个残缺 tile"的 batch 上断言这一点。

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
| `torch(ms)` | 在**预先反量化**的权重上按 expert 执行 `A @ W.T` (反量化在计时区间之外) — 纯 matmul 的 PyTorch 上限。跳过该基线时显示 `--` (计算受限的行，此时反量化后的 `[E, N, K]` 张量无法与其他数据同时放下) |
| `w4a16(ms)` | 同一阶段现有的 ARK int4 kernel (`moe_gemm_decode` / `moe_gemm_prefill`) |
| `w4a8(ms)` | 新的 int8 计算路径 (`ark.moe_gemm_w4a8`) |
| `rows/E` | 每个**活跃**专家分到的路由 token 数。计算强度为每权重字节 `2 × rows/E` 次浮点运算，因此该数值单独决定了某个形状是否可能成为计算受限 |
| `TFLOPS` | `total_tokens × N × K × 2 / time` |
| `W GB/s` | 被路由 token 实际访问到的专家权重带宽 (`active_experts × N × K × 1 byte / time`) — decode 访存瓶颈的衡量指标 |
| `BW@100T` | 该形状达到 100 TFLOPS 所需的权重带宽。当它超过设备实际能提供的带宽时，`TFLOPS` 就被访存限制，任何 kernel 改动都无法在该形状上达标 |
| `vs torch` / `vs w4a16` | 加速比 (`other / w4a8`) |
| `prepack(ms)` | 一次性的 int4 → int8 AUTO_S8 转换开销。只在模型加载时支付，**不是**每次前向都支付。 |

每次扫描之后都会输出一个 `targets [...]` 段落，给出
[性能目标与 roofline](#性能目标与-roofline) 中描述的 PASS / FAIL / N/A 结论。

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
`test_perf_prefill_compute_bound` 额外增加一个 batch，其大小保证每个专家拿到 256
行 (Qwen3-MoE 为 4096 个模型 token) — 这是 100 TFLOPS 目标不再被权重带宽限制的最
小扫描点。

第二个形状组是 MiniMax-M2，与 `test_moe_prefill_perf.py` 保持一致：

```
hidden_size = 3072,  intermediate_size = 1536
num_local_experts = 192,  num_experts_per_tok = 8

minimax up    :  N = 1536,  K = 3072
minimax down  :  N = 3072,  K = 1536
```

之所以需要它，是因为两个目标都与形状相关：192 个专家会把同样的 batch 摊到 1.5 倍
的专家上 (每专家行数更少，因此相同 batch 下的算力上限*更低*)，而更长的 K 则让
decode GEMV 的顺序访存流更长、也让 prefill 的 tile 每次加载覆盖更多 K。compute-
bound 的 batch 按模型推导，因此 MiniMax 用 6144 个模型 token 达到同样的每专家 256
行。形状组通过 `--models` 选择 (`qwen3` — 默认 —、`minimax`、逗号分隔的列表或
`all`)；MiniMax 的重尾真实路由分布仍在 `test_moe_prefill_perf.py` 中。

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

# 计算受限的 prefill 用例 (4096 个模型 token)，TFLOPS 目标在此可达
pytest -v -s test_moe_w4a8_perf.py -k compute_bound

# 把性能目标从提示信息变成硬断言
pytest -v -s test_moe_w4a8_perf.py -k perf --enforce-targets

# 加入 MiniMax 形状 (--models all 则两组都跑)
pytest -v -s test_moe_w4a8_perf.py -k perf --models minimax

# 扫描 kernel 的各种 dispatch 配置，并打印最快且数值等价的一个
pytest -v -s test_moe_w4a8_perf.py -k sweep
```

`test_perf_decode_config_sweep`、`test_perf_prefill_tile_sweep`、
`test_perf_prefill_act_quant_sweep`、`test_perf_prefill_act_quant_unroll_sweep`
和 `test_perf_prefill_epilogue_sweep` 只构造一
次 workload、只 prepack 一次，然后用同一份数据依次给每种 dispatch 配置计时——decode
的 lane 映射 (legacy GEMV 以及 `CH` × `NCOLS` 的全部组合)、prefill 的 work-group
tile、激活量化的消息宽度与在途请求深度，以及 epilogue 的边界保护。每
种配置都会与第一种配置做数值等价性检查，表格之后还会打印一段 `best
configuration`，按形状给出获胜配置对应的环境变量，因此在硬件上跑一次就能确定这些
调优开关。

需要加 `-s` 才能看到打印出的表格。

### 作为独立脚本运行 (不依赖 pytest)

```bash
python test_moe_w4a8_perf.py                       # 两个阶段，最小 batch
python test_moe_w4a8_perf.py --all-shapes          # 完整扫描
python test_moe_w4a8_perf.py --phase decode        # 仅 decode
python test_moe_w4a8_perf.py --skip-accuracy       # 仅性能
python test_moe_w4a8_perf.py --compute-bound       # 追加 4096 token 的 prefill 用例
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
    activations,  # [total_tokens, K] fp16/bf16, 按 expert 排序
    weights_s8,  # [E, N, K] int8
    wscales,  # [E, N, K // block] fp32
    num_tokens_per_expert,  # [E] int32
    rescale_block_size=block,
    phase="auto",  # "auto" | "decode" | "prefill"
)
```

下面的便捷封装会同时完成两步，并按权重/scale 张量的标识缓存转换结果：

```python
out = ark.moe_w4a8(
    activations,
    weights,
    num_tokens_per_expert,
    scales=scales,
    group_size=32,
    phase="auto",
)

ark.clear_moe_w4a8_prepack_cache()  # 释放缓存的 int8 权重
ark.moe_w4a8_release_scratch()  # 归还设备端 scratch 内存
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

## 实测得到的默认值

下面的默认值来自 BMG 上的一次 `-k sweep` (bf16 激活，decode 为 8 条 routed
行，prefill 为每专家 256 行)。每种配置在计时之前都会先与第一种配置做数值等价性
检查。

### Prefill tile

| 形状 | `128x128` | `256x128` | `128x256` | `256x256` |
|---|---|---|---|---|
| qwen3 up | 2.774 ms | **2.719 ms** | 2.809 ms | 2.808 ms |
| qwen3 down | 1.904 ms | **1.829 ms** | 2.041 ms | 2.058 ms |
| minimax up | 5.584 ms | 5.595 ms | 5.694 ms | **5.477 ms** |
| minimax down | 5.681 ms | 5.605 ms | **5.541 ms** | 5.662 ms |

现在整张表的差距只有 0–8%。它曾经是 35–50%，而那道悬崖的成因并不是 tile 形状：当时
主循环在 int32 累加器之外还全程保持一份 C fragment 的 float 影子，于是 `TileN = 256`
需要每个 SIMD16 lane 128 + 128 个寄存器，即整个 large-GRF 寄存器堆。去掉这份影子之后
(参见 [Prefill：访存消息宽度与寄存器压力](#prefill访存消息宽度与寄存器压力))，N 方向
为 256 的 tile 只落后几个百分点——这正是每个 sub-group 32×64 的 C fragment
(所有 policy 的 `SGLayout` 在 N 方向都是 4) 在 occupancy 而非寄存器溢出上付出的代价。

真正有收益的方向是 `TileM = 256`：它把 `SGLayout` 变成 8×4，从而让每个 lane 的
fragment 仍保持 32×32，同时把专家 B 面板的重复读取次数减半——每个专家的 tile 访存量
为 `M·K·⌈N/TileN⌉ + N·K·⌈M/TileM⌉`。现在它在**全部四个**形状上都不慢于
`128x128`——+2.0% (qwen3 up)、+3.9% (qwen3 down)、+1.3% (minimax down)、
−0.2% (minimax up)。

因此 tile 阶梯为：每专家 `< 16` 行 → `8x128`，`< 128` → `64x128`，`≥ 256` →
**`256x128`**，其余 → `128x128`。六个 policy 全部保留编译，并可通过
`ARK_MOE_W4A8_PREFILL_TILE` 手动选择。

`≥ 256` 这一档过去还附带第二个条件 `N ≥ K`，因为上一次 sweep 中两个 up 投影分别是
+0.2% / −2.4%。这个条件现在已经去掉：本次测量与它的分歧超出了下面所说的噪声下限，说
明它拟合的是一次测量结果，而不是硬件本身的行为。

读这张表时有两点需要注意。"tile auto" 与它实际解析到的那一列跑的是同一个 kernel，二
者却相差最多 9% (qwen3 down 上 2.007 vs 1.829 ms)，因此这里 ~5–9% 以内的差异都在运行
间噪声范围内——`256x256` 领先的那一行也在其中。所以这一档是依据各形状上效果的**符
号**来决定的，而不是依据某一个形状的具体幅度。另外，阶梯里的每专家行数阈值比较的是
`total_tokens / E` 这个**平均值**，因此在路由不均衡时，即便平均为 256 行，仍可能有很
多专家只有残缺的 256 行 tile。

在选中的 tile 下，被扫描的四个形状达到 75.8 / 56.4 / 82.9 / 82.8 TFLOPS
(qwen3 up / down、minimax up / down)——相比去掉 float 影子和引入向量化激活量化之前的
61.1 / 39.4 / 68.8 / 65.3 有明显提升，但 qwen3 down 距离 100 TFLOPS 的目标仍有差距：
`K = 768` 时一个 tile 只有 12 个 k-tile，epilogue 与 prologue 因此占了相当大的比例。

### Prefill 激活量化

| 形状 | 标量 | 向量化 (默认) | 加速比 |
|---|---|---|---|
| qwen3 up | 3.040 ms | **2.699 ms** | 1.13× |
| qwen3 down | 2.088 ms | **1.824 ms** | 1.14× |
| minimax up | 6.269 ms | **5.623 ms** | 1.12× |
| minimax down | 5.965 ms | **5.716 ms** | 1.04× |

量化 routed 激活只是对 `[T, K]` 的一次流式遍历，而与它并行的 GEMM 本身就要搬运约
400 MB；仅仅把 32 字节 load / 16 字节 store 换成 256 字节 / 128 字节，就能带来整次调
用 4–14% 的收益。`ARK_MOE_W4A8_ACT_QUANT_VEC=0` 可恢复标量映射。

一个 work-item 能让多少条这样的宽 load 同时**在途**，则由另一个开关
`ARK_MOE_W4A8_ACT_QUANT_UNROLL` 控制 (1 = 上表所测的映射、2、4 = 默认值)；
`test_perf_prefill_act_quant_unroll_sweep` 会为它计时，上表就是其 `unroll 1` 基线。

### Prefill epilogue 边界保护

| 形状 | 带保护 | 内部 tile 快速路径 (默认) | 加速比 |
|---|---|---|---|
| qwen3 up | 2.774 ms | **2.634 ms** | 1.05× |
| qwen3 down | 2.214 ms | **1.924 ms** | 1.15× |
| minimax up | 5.884 ms | **5.785 ms** | 1.02× |
| minimax down | 5.968 ms | **5.700 ms** | 1.05× |

两列的主循环完全相同，只有 store 不同，因此这就是每个输出元素约 4 条指令的代价。它在
主循环最短的形状上占比最大——`K = 768` 的 qwen3 down 每个 tile 只跑 12 个 k-tile——正
是当初按指令数推理所预期的那个形状。`ARK_MOE_W4A8_PREFILL_FULL_TILE=0` 可切回带保护
的 epilogue；两者逐位相同。

### Decode 的 chunk 宽度与列分块

| 形状 | 数值等价配置中最快的一个 | 默认值 (`CH=16`、`NCOLS=2`) | 相同 `NCOLS` 下的 `CH=32` |
|---|---|---|---|
| qwen3 up | ch16 ncols2 — **284.0 GB/s** | 284.0 GB/s | 278.9 GB/s |
| qwen3 down | ch16 ncols4 — **285.7 GB/s** | 280.1 GB/s | 244.4 GB/s |
| minimax up | ch16 ncols1 — **271.0 GB/s** | 268.1 GB/s | 259.9 GB/s |
| minimax down | ch16 ncols2 — **315.5 GB/s** | 315.5 GB/s | 308.7 GB/s |

`CH = 32` 从未取胜，最多还慢 13%，因此默认值保持 `16`。`NCOLS = 2` 在四个形状中的
两个上最快，在另外两个上也与最优值相差不到 2%；而 `1` 在 qwen3 up 上慢 47%、`4` 在
minimax up 上慢 14%，因此 `2` 同样保持为默认值。在这组默认值下，K-split 映射相对
legacy GEMV 的收益为 1.09–1.93×。

## 环境变量

| 变量 | 作用 |
|---|---|
| `ARK_MOE_W4A8_AUTO_S8` | 覆盖 AUTO_S8 重缩放 block 大小。未设置 / `-1` 表示每个输出通道一个 scale (最快)。如果取值不是 `group_size` 和 64 的公倍数，或不能整除 K，则静默回退为 K。 |
| `ARK_MOE_W4A8_DECODE_MAX_TOKENS` | `phase="auto"` 时选择 GEMV 的 token 数上限 (默认 `128`)。 |
| `ARK_MOE_W4A8_DECODE_KSPLIT` | 合并访存的 K-split decode 映射，**默认开启**。设为 `0` 可回退到原来每个输出一个 work-item 的 GEMV (便于 A/B 对比)。形状不满足条件时该开关无效。 |
| `ARK_MOE_W4A8_DECODE_KSPLIT_NCOLS` | K-split 映射中每个 sub-group 处理的输出列数：`1`、`2` (默认) 或 `4`。取值越大，激活数据的加载可以摊到更多列上，但要求 `N % (16 × NCOLS) == 0`。默认值 `2` 来自实测，参见[实测得到的默认值](#实测得到的默认值)。 |
| `ARK_MOE_W4A8_DECODE_KSPLIT_CH` | 每个 lane 每次加载的 K 元素数 (即字节数)：`16` (默认) 或 `32`。`32` 可以把访存指令数减半、并让每个线程同时在途的字节数翻倍，代价是更多 GRF；它要求 re-scale block 至少为 512，否则会自动回退到 `16`。实测中它在所有形状上都慢于 `16`，因此只作为扫描项而非推荐值。 |
| `ARK_MOE_W4A8_PREFILL_TILE` | 强制指定 prefill 的 work-group tile：`8x128`、`64x128`、`128x128`、`128x256`、`256x128`、`256x256`。不设置 (默认) 时按 tile 阶梯自动选择：每专家 ≥ 256 行时取 `256x128`，否则取 `128x128` (参见[实测得到的默认值](#实测得到的默认值))。现在整张表的差距已收敛到 0–8%，因此这个开关是重新调优用的旋钮，而不再对应一道性能悬崖。 |
| `ARK_MOE_W4A8_ACT_QUANT_VEC` | 向量化的每 token 激活量化 (每个 lane 负责 4 或 8 个连续的 K 元素，而不是按 sub-group 宽度跨步)；**默认开启**，在被扫描的形状上带来 1.04–1.14× 的收益。设为 `0` 可强制使用标量映射以便做 A/B 测量。当 K 或缓冲区对齐不满足条件时该开关被忽略，此时本就会运行标量 kernel。 |
| `ARK_MOE_W4A8_ACT_QUANT_UNROLL` | 激活量化 kernel 在开始消费之前先加载的向量个数：`1`、`2` 或 `4` (默认)。取值越大，一个 work-item 保持在途的字节越多——这一遍在每线程仅一条在途 load 时受限于延迟而非带宽——代价是 GRF 占用。`1` 即批量化之前的 kernel，可作为 A/B 基线；所有取值逐位相同。不在 `{1, 2, 4}` 中的取值会回退到默认值。只对向量化映射生效。 |
| `ARK_MOE_W4A8_PREFILL_FULL_TILE` | 对既不触及 M 边界也不触及 N 边界的 tile，跳过 epilogue 中的 store 谓词与 scale 下标钳制；**默认开启**，在被扫描的形状上带来 1.02–1.15× 的收益。该判断在 work-group 内是一致的，因此代价是每个 tile 一次比较，而不是每个输出元素若干次。设为 `0` 可强制所有 tile 都走带保护的 epilogue (两者必须逐位相同)。 |

## 形状约束

kernel 要求：

* `N % 16 == 0` (GEMV 的 N tile 与 DPAS 的 N tile)
* `K % 64 == 0` (DPAS 的 K tile)
* `group_size % 8 == 0` 且 `K % group_size == 0`
* 解析出的重缩放 block 必须是 64 的倍数并且能整除 K

Qwen3-MoE 的两个 GEMM 都满足以上条件 (`K = 2048` 和 `K = 768`)。

decode 的 K-split 映射还额外要求重缩放 block 不小于 256 且是 16 的倍数；不满足的形
状会退回到原 GEMV，而不是报错。

## 状态

W4A8 kernel 是新移植的 SYCL/CuTe 实现，在
`auto_round_kernel/wrapper/include/sycl_tla_moe_w4a8.hpp` 中被标记为
`STATUS: PARTIALLY HARDWARE-VALIDATED`。各项性能扫描都已在 BMG 上跑过——tile 阶梯、
激活量化的默认值、内部 tile 的 epilogue 以及 decode 的 `CH` / `NCOLS` 默认值正是来自
那些运行 (参见[实测得到的默认值](#实测得到的默认值))，并且所有被扫描的配置都通过了配
置间的数值等价性检查。`test_act_quant_vec_matches_scalar`、
`test_full_tile_epilogue_matches_predicated` 与 `test_decode_ksplit_matches_legacy`
也已在设备上通过，因此向量化激活量化、内部 tile 的 epilogue 与 K-split decode 映射既
有计时数据，也都与各自的前身做过比对。

仍需在设备上运行的部分：与 fp32 参考实现对比的精度扫描，它能立刻暴露 layout /
scale 相关的 bug。

激活量化的批量 load 是唯一一项只经过推导、既未实测计时也尚未在设备上运行的改动：先加
载 `UNROLL` 个向量再开始消费，使一个 work-item 同时挂起这么多请求，而不是只有一个。它
在设备上的检查是 `test_act_quant_unroll_matches` (在"K 能整除展开深度"和"K 会留下尾
巴"两种情况下的逐位一致性) 与 `test_perf_prefill_act_quant_unroll_sweep`；设置
`ARK_MOE_W4A8_ACT_QUANT_UNROLL=1` 即可在不重新编译的情况下回退。

这一项落地之后，prefill 的下一步是写出本身：epilogue 目前仍然通过标量指针、按谓词逐
元素写 D，而同类的 int8 与 fp8 kernel 对 D 用的是 block-2D copy。在卡在 56 TFLOPS 的
`K = 768` 形状上，一个 tile 只有 12 个 k-tile，写出因此占据了其中相当大的一部分时间。
与上面几项不同，它并不是纯 C++ 的改动：Xe DPAS 的 C fragment 给每个 lane 分配的是一
*列*，因此同一个 lane 的相邻数值在内存中相隔 N，只有硬件的 2D block-store 消息才能把
它们合并成宽消息。移植它需要用到 `partition_sg_fragment_C` /
`partition_sg_fragment_S` 以及经由 float 中间 fragment 的 `reorder` (累加器是 int32，
必须先按行、再按列施加 scale)，而同类 kernel 中没有任何一个是对**带 scale 的** int32
累加器做 2D 写出的——因此这项工作需要一个具备 SYCL 编译器和设备的环境，而不是一个开
关。
