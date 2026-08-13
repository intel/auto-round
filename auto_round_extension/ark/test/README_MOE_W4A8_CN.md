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

### 本文所有实测数据所用的设备

一块 **Intel Arc Pro B60** (Battlemage，`BMG-G21` — 也就是默认的 AOT 目标
`intel_gpu_bmg_g21`)：20 个 Xe2 core / 160 个 XVE，频率约 2.4 GHz，24 GB GDDR6，
192-bit 位宽。它给出的各条上限：

| 上限 | 数值 |
|---|---|
| int8 XMX (DPAS) | 160 XVE × 512 int8 ops/clk × 2.4 GHz ≈ **197 TOPS** |
| bf16 / fp16 XMX | ≈ **98 TFLOPS**，是 int8 速率的一半 — 这正是 W4A8 存在的理由 |
| DRAM 引脚带宽 | **456 GB/s**；测试脚本的 device-copy 探测读到约 400 GB/s (引脚带宽的 88%) |
| 占用率上限 | 160 XVE × 8 个线程槽 = **1280 个并发 SIMD16 sub-group** |

因此下面两个目标分别是 int8 峰值的 51% 和引脚带宽的 66%。`Arc Pro B60 Dual` 卡会
暴露两个这样的设备，kernel 只看到其中一个。

本 kernel 的目标是 **prefill > 100 TFLOPS**、**decode 权重带宽 > 300 GB/s**。
prefill 目标是否*可能*达到，取决于**路由**而不是 kernel 本身。

### 权重并不是唯一的数据流

本文早期版本用**权重这一条流**来建立 prefill 的 roofline：W4A8 grouped GEMM 对每个
活跃专家的 int8 权重只读一次，每读一个权重字节要做 `2 × rows_per_expert` 次浮点运算，
因此

```
计算强度   = 2 × rows_per_expert                    [FLOP / byte]
TFLOPS   <= 2 × rows_per_expert × 权重带宽
rows_per_expert = batch × top_k / active_experts
```

其中 `N`、`K` 因子相互抵消。这个式子对**权重流**是对的，对**总流量**是错的。一次
`moe_gemm_w4a8` 调用要搬运的是五条流，而不是一条 (`T = batch × top_k` 为路由行数)：

| 数据流 | 字节数 | 随谁增长 |
|---|---|---|
| 量化 kernel 读入的激活 | `T × K × sizeof(act)` | `T` |
| 写出的 int8 副本 | `T × K` | `T` |
| GEMM 再读回的同一份 int8 副本 | `T × K` | `T` |
| 所有活跃专家的权重 | `E_active × N × K` | `E_active` |
| 输出 | `T × N × sizeof(out)` | `T` |

只有第四行是 `W GB/s` 统计的、也是旧模型唯一计入的部分。因为它是唯一**不**随 token
数增长的一条，所以它在小 batch 下占主导 (那里旧公式几乎是精确的)，而恰恰在用于衡量
目标的计算受限区间里变成了**少数**。在每专家 256 行 (本测试脚本此前使用的 batch) 时：

| 形状 | 权重 | 总流量 | 权重占比 | 达到 100 TFLOPS 所需带宽 (旧模型) | 400 GB/s 下的上限 |
|---|---|---|---|---|---|
| qwen3 up (N=1536, K=2048) | 403 MB | 772 MB | 52% | **374** GB/s (195) | 107 TFLOPS |
| qwen3 down (N=2048, K=768) | 201 MB | 436 MB | 46% | **423** GB/s (195) | **94 TFLOPS** |
| minimax up (N=1536, K=3072) | 906 MB | 1661 MB | 55% | **358** GB/s (195) | 112 TFLOPS |
| minimax down (N=3072, K=1536) | 906 MB | 1510 MB | 60% | **326** GB/s (195) | 123 TFLOPS |

旧模型给这四个形状打印的都是 195 GB/s。真实需求是它的 1.7–2.2 倍 — 而对 qwen3
down-projection 来说，这个需求**超过了设备能提供的带宽**：需要 423 GB/s，而引脚带宽
456 GB/s 的实测拷贝只有约 400 GB/s。该路由下它的上限是 94 TFLOPS，也就是说
**无论 kernel 怎么改，在此前测量所用的 batch 上这个形状都不可能达到 100 TFLOPS**。
它也正是历次扫描中离目标最远的形状 (50–56 TFLOPS)，这并非巧合：K 最小意味着非权重
流量占比最大。

因此 `_PREFILL_TARGET_ROWS_PER_EXPERT` 从 256 提高到 **384** — 这是能让四个形状的
上限全部越过 100 TFLOPS 的最小整数路由 (在 400 GB/s 探测值下分别为 112 / 130 /
137 / 154 TFLOPS；其中 qwen3 down 单独要求 ≥ 290 行/专家)。换算成模型 token 数，
Qwen3-MoE 为 6144，MiniMax 为 9216。

小 batch 的结论不变，因为那里权重项占主导：

| 模型 token 数 | 路由行数 | 每专家行数 | 达到 100 TFLOPS 所需带宽 |
|---|---|---|---|
| 128 (prefill 默认 batch) | 1024 | 8 | 约 6300 GB/s |
| 512 | 4096 | 32 | 约 1600 GB/s |
| 2048 | 16384 | 128 | 约 440 GB/s |
| 6144 (`test_perf_prefill_compute_bound`) | 49152 | 384 | 约 310 GB/s |

默认 batch 下约 4.5 TFLOPS **并不是 kernel 的缺陷**：在每专家 8 行、kernel 实测约
285 GB/s 的条件下，上限就是 `2 × 8 × 285e9 = 4.56 TFLOPS` — 正好等于实测值，说明
kernel 已经跑在 DRAM roofline 上。要在该形状上达到 100 TFLOPS 需要 6 TB/s 以上，
是 B60 那 456 GB/s 的 13 倍以上。

因此性能表在实测值旁边额外打印 `rows/E`、`DRAM GB/s` (五条流的总和) 和 `BW@100T`，
并在每次扫描后输出结论：

```
targets [prefill]: prefill compute > 100 TFLOPS
  device copy bandwidth probe: 390 GB/s
  qwen3 up     tokens=1024   rows/E=8.0        4.56 TFLOPS vs 100 -> N/A (bandwidth bound: ...)
  qwen3 down   tokens=49152  rows/E=384.0     66.77 TFLOPS vs 100 -> FAIL (61% of the 109 TFLOPS bandwidth ceiling)
  minimax down tokens=73728  rows/E=384.0    104.77 TFLOPS vs 100 -> PASS (70% of the 150 TFLOPS bandwidth ceiling)
```

当设备带宽探测 (每次运行执行一次的大块 device-to-device 拷贝) 表明该路由下目标不
可达时，该行显示 `N/A` 而不是 `FAIL`；可达的行还会额外打印它达到了自身上限的百分之
多少 — 这才是 kernel 改动能够撬动的部分。该结论默认只用于提示；加上
`--enforce-targets` 可以把它变成硬断言。

### 8K 提示词用例：提示词长度不等于每专家行数目标

`_PREFILL_TARGET_ROWS_PER_EXPERT` 是**按模型推导**出来的，所以两组形状都落在同样的
每专家 384 行上 (Qwen3-MoE 是 6144 个模型 token，MiniMax 是 9216)。真实的 prefill
恰好相反：提示词长度是固定的，由专家数去除它。因此 `test_perf_prefill_long_seq` 跑
的是一条 **8K token 的提示词** — 8192 个模型 token、65536 条路由行，与
`test_moe_prefill_perf.py` 扫描的 8K 组相同 — 而两个模型会落在**不同**的区间：

| 形状 | 每专家行数 | 权重 | 总访存量 | 达到 100 TFLOPS 所需带宽 | 400 GB/s 下的上限 | 每专家 384 行时的上限 |
|---|---|---|---|---|---|---|
| qwen3 up (N=1536, K=2048) | 512 | 403 MB | 1141 MB | 277 GB/s | **145 TFLOPS** | 129 |
| qwen3 down (N=2048, K=768) | 512 | 201 MB | 671 MB | 326 GB/s | **123 TFLOPS** | 112 |
| minimax up (N=1536, K=3072) | 341 | 906 MB | 1913 MB | 309 GB/s | **129 TFLOPS** | 137 |
| minimax down (N=3072, K=1536) | 341 | 906 MB | 1711 MB | 277 GB/s | **145 TFLOPS** | 154 |

对 Qwen3-MoE 的 128 个专家来说，8K 提示词是每专家 512 行，比计算受限 batch 多三分之
一，所以上限提高 10–12%、100 TFLOPS 目标的余量更大 — 这里应当出现整个套件中最高的
prefill TFLOPS。而对 MiniMax 的 192 个专家来说，同一条提示词只有每专家 341 行，**低
于**计算受限 batch，上限反而下降 6%。也就是说同一个 kernel 在同样的提示词长度下，
在一个模型上更快、在另一个模型上更慢 — 这正是两个模型都要测的原因：决定吞吐的是路
由，而不是序列长度。

实测结果正是如此。相对计算受限 batch 的 93.8 / 66.8 / 101.0 / 104.8 TFLOPS，8K 提示
词读到 98.2 / 70.5 / 93.5 / 95.4 — 两个 qwen3 形状提高约 5%，两个 minimax 形状下降
7–9%，方向与各自路由的变化一致。而以**占上限的比例**看几乎没有变化 (70 / 59 / 74 /
68% 对 74 / 61 / 76 / 70%)，这才是有用的读法：变的是可达的上限，而不是 kernel 与上限
的距离。

每专家 512 行同时还会改变 **tile 阶梯**，这也是这个点要单独跑一遍 tile 扫描的原因。
256 行 tile 实际调度 `⌈M/256⌉·256` 行，所以在每专家 384 行时只能带着三分之一算力花在
padding 上被测量；512 是 256 的整数倍，是套件里唯一满足
`⌈M/256⌉·256 == ⌈M/128⌉·128` 的路由，也就是唯一能公平评判 `TileM = 256` 的地方。
`test_perf_prefill_tile_sweep_long_seq` 给出了结论 — qwen3 up 上持平、qwen3 down 上落
后 5.4% — 阶梯中的 256 行档因此被移除 (见 [Prefill tile](#prefill-tile))。

### 为什么小 batch 下 `vs w4a16` 小于 1.0

同样的计算强度分析也解释了 `vs w4a16` 这一列。W4A8 需要传输 int4 路径 **2 倍的权
重字节**(每个元素 1 字节 vs 半字节)，换来约 2 倍的 DPAS 峰值，所以只有在 GEMM 变
成计算受限之后才会占优：

```
交叉点 rows/expert ~= int8 峰值 TOPS / (4 × 权重带宽)
```

按 B60 的约 197 TOPS int8 DPAS 和 kernel 实测约 285 GB/s 计算，交叉点约为每专家
173 行 (约 2760 个模型 token) — 这与带宽 roofline 首次允许 100 TFLOPS 的路由 (上面
的约 176 行) 基本重合，也就是说在这块卡上两个交叉点落在同一处。decode (每专家 1 行)
和小 batch prefill 都远低于该点，所以
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
load 在途。这是 Little 定律的问题，而不是带宽的问题：1280 个并发 sub-group (B60 的
占用率上限 — 160 个 XVE × 8 个线程槽) × 256 字节只有约 320 KB 的在途读取，而一块
456 GB/s 的设备要在约 1 µs 的访存延迟下保持忙碌需要约 456 KB，何况实际 launch 很少
能填满每一个线程槽。这与 decode GEMV 每次迭代加载两个 chunk 是同一个论证。

现在每次迭代先加载 `UNROLL` 个**互相独立**的向量，然后才开始消费它们，并把它们归约到
`UNROLL` 个各自独立的局部最大值上，使这些 load 也不必串行等待累加器链；量化那一遍同样
按此批量化其 load。在默认 `UNROLL = 4` 下每个线程持有 1 KB，因此远在填满线程槽之前
就已经越过了那 456 KB 的门槛。`steps % UNROLL` 个向量交给尾循环处理——`K = 768` 时一
个 lane 要走 6 个向量，因此在默认 `UNROLL = 4` 下尾循环是真实会执行的代码，而不是形
式上的补充。任
何会产生舍入的步骤都没有改变 (`fmax` 精确且与顺序无关，因此各局部最大值合并后逐位相
同)，且 `UNROLL = 1` 与批量化之前的 kernel 逐条指令一致，所以
`ARK_MOE_W4A8_ACT_QUANT_UNROLL=1` 是精确的 A/B 基线。
`test_act_quant_unroll_matches` 会在"K 能整除展开深度"和"K 会留下尾巴"两种情况下断言
逐位相同。

**只读一遍这一行。** 批量化只改变了加载的方式，没有减少加载的次数。absmax 必须先看
完整行才能量化第一个元素，因此这一遍要先读一次 `[T, K]`、做归约、再读一次 `[T, K]`。
只要行还在缓存里，第二次读就由 L2 提供；但一个 work-group 后量化的那些行，会在这一遍
结束之前把它先量化的那些行挤出去——在 8 MB 的 L2 和 `K = 2048` 时每行 4 KB 的条件下，
*在没有任何其它数据驻留*的前提下也只装得下约 2000 行，而紧接着 GEMM 的权重还要争抢同
一块缓存。

一行数据小到足以放进寄存器：一个 lane 拥有 `K / 16` 个元素，`K = 2048` 时是 256 字节
——占量化 kernel 每 lane 128 个 dword 预算中的 64 个 (与 GEMM 不同，它没有用
`grf_size<256>` 启动)。单遍 kernel 把整行一次性读入、做归约、再直接从寄存器里量化写
出；第二次读消失了，而且所有 load 都在第一次被消费之前就发出，这等于把 `UNROLL` 想做
的事情一并做了，而不是与它冲突。

`MAX_STEPS` 是让这段 fragment 落在寄存器而不是 scratch 上的编译期上界：循环是对
`MAX_STEPS` 的 `#pragma unroll`，配合 `if (s < steps)` 保护，因此所有下标都是常量，
SROA 可以把数组提升为标量。共实例化两档——8 个向量 (`VEC = 8` 时 `K ≤ 1024`，32 个
dword) 与 16 个向量 (`K ≤ 2048`，64 个 dword)——更长的行仍走两遍 kernel，这也是
minimax 的 `K = 3072` up-projection 依旧走旧路径的原因。局部最大值仍然是 4 个累加器，
因此归约的代价和结果都没有变化。

这是一次寄存器压力上的赌博：如果 64 个 dword 的行数据加上寻址导致溢出，这一遍会变得
**更慢**。`ARK_MOE_W4A8_ACT_QUANT_SINGLE_PASS=0` 可以精确地切回两遍 kernel，
`test_perf_prefill_act_quant_single_pass_sweep` 负责给两者计时，
`test_act_quant_single_pass_matches` 则在"恰好填满一档的 K (2048)"和"填不满的
K (768)"两种情况下断言二者逐位相同。

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
[实测得到的默认值](#实测得到的默认值)中要多付 35–50% 的原因；去掉 float 影子、随后又去
掉 epilogue 的标量 store 之后，256 宽的 tile 现在已经成为阶梯的默认选择。而在默认重缩放
block 下它完全是白付的开销：
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

**store 本身。** 把 store 周围的指令削减之后，剩下的就是 store 本身。Xe DPAS 的 C
fragment 让一个 lane 持有每个 8×16 atom 的一*列*，因此一个 sub-group 的 16 个 lane
持有的是*同一行的 16 个连续列*：对 16 位的 `ElementD` 来说，一次标量
`c[row * n + col] = ...` 只是一条 32 字节的消息——半条 cache line——而一个 32×32 的
sub-group fragment 要发 **64** 条。通过硬件 2D block store，同样的字节只需要少数几条
消息，这也正是所有同族 prefill kernel 写 D 时采用的方式
(`sycl_tla_moe_prefill_{fp8,int,s4}_dpas.hpp`)，以及 `sycl_tla_dense_gemm.hpp` 中的
稠密 GEMM 在完全相同的累加器形状上采用的方式。

D 才是这件事在 prefill 尺寸下值得做、而不只是"顺手整理"的原因：每专家 384 行时，
qwen3 down-projection 每个专家要写 1.5 MB 的 fp16——与它读入的 int8 权重字节数恰好相
等，因为那里的 N (2048) 比 K (768) 更大，占该专家流量的三分之一以上；而它恰好又是主
循环最短的形状，等于把 epilogue 的代价付了两遍。

移植参考的是 `dense_gemm_detail::gemm_device_impl`，而不是同族的 MoE kernel。后者用
`reorder(tCrC, tCrC_out)` 把 MMA fragment 搬进显式选定的 `XE_STORE_2D` atom 的
fragment，而 `reorder` 搬的是**寄存器**：在 `float` 累加器下这是免费的，但本 kernel
用 `int32` 累加 (`XE_DPAS_TT<8, int32_t, int8_t, int8_t>` 的 `FrgTypeC`)，必须先缩放
并做数值转换，而 `reorder` 并不做转换。`make_block_2d_copy_D` 的布局直接来自 MMA 自身
的 C 划分，因此缩放后的 `ElementD` fragment——`make_tensor_like<ElementD>(tCrC)`，用
标量路径所用的同一组 `tCgC(i)` 坐标填充——可以直接交给
`copy(copy_d, tCrD, tCgC)`，中间不需要任何 `reorder`。

它还**去掉**了 store 谓词，而不只是跳过它：2D block 消息会裁剪到 D tensor 所描述的表
面，因此 M 边缘的部分 tile 由硬件丢弃越界的行——同族 grouped GEMM 处理残缺专家时依赖
的正是这一点。只有 scale 的 load 仍需钳制下标，而且只在边缘 tile 上。

该描述符要求基址 64 字节对齐、行 pitch 是 16 字节的倍数。D 的每专家基址是
`outputs + pre_rows × N`，其中 `pre_rows` 是运行期的路由值，因此 dispatcher 检查的是
`N × sizeof(ElementD) % 64 == 0`——在 tensor 本身对齐的前提下，这能保证*每一个*专家的
基址都对齐，同时也覆盖了 pitch——外加 tensor 基址本身。所有已支持的 N (16 位 D 下的
1536 / 2048 / 3072) 都满足；不满足的形状继续走标量 store。
`ARK_MOE_W4A8_PREFILL_STORE_2D=0` 同样会切回标量 store 以便 A/B 测量，
`test_prefill_2d_store_matches_scalar` 则在"每个专家恰好有一个完整 tile 和一个残缺
tile"的 batch 上断言两者写出的比特完全一致——这正是一个不做裁剪的 store 会污染下一个
专家行数据的场景。

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
| `DRAM GB/s` | 本次调用搬运的**全部**流量：读入的 fp16 激活、写出并再读回的 int8 副本、专家权重，以及输出——参见[权重并不是唯一的数据流](#权重并不是唯一的数据流)。这才是应当与设备 456 GB/s 相比较的数值 |
| `BW@100T` | 该形状达到 100 TFLOPS 所需的 DRAM 带宽 (计入全部五条数据流)。当它超过设备实际能提供的带宽时，`TFLOPS` 就被访存限制，任何 kernel 改动都无法在该形状上达标 |
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
`test_perf_prefill_compute_bound` 额外增加一个 batch，其大小保证每个专家拿到 384 行
(Qwen3-MoE 为 6144 个模型 token) — 这是在计入
[全部五条数据流](#权重并不是唯一的数据流)而不只是权重之后，100 TFLOPS 目标在*所有*已
支持形状上都低于设备带宽天花板的最小整值扫描点。`test_perf_prefill_long_seq` 则补上
另一类 prefill 采样点：一条固定长度的 **8K token 提示词** (8192 个模型 token、65536
条路由行)，对 Qwen3-MoE 是每专家 512 行、对 MiniMax 是 341 行 — 详见
[8K 提示词用例](#8k-提示词用例提示词长度不等于每专家行数目标)。

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
bound 的 batch 按模型推导，因此 MiniMax 用 9216 个模型 token 达到同样的每专家 384
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

# 计算受限的 prefill 用例 (6144 个模型 token)，TFLOPS 目标在此可达
pytest -v -s test_moe_w4a8_perf.py -k compute_bound

# 8K 提示词的 prefill 用例 (8192 个模型 token) 及其 tile 扫描
pytest -v -s test_moe_w4a8_perf.py -k long_seq

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

`test_perf_prefill_tile_sweep_long_seq` 是同一个 tile 扫描在 8K 提示词路由下的版本，
那里阶梯选中的档位不同 (Qwen3-MoE 是每专家 512 行而不是 384 行)；同一形状若在多个
batch 上被扫描，`best configuration` 会为每个 batch 各打印一行 — 因为最优配置既取决
于形状，也同样取决于路由。

需要加 `-s` 才能看到打印出的表格。

### 作为独立脚本运行 (不依赖 pytest)

```bash
python test_moe_w4a8_perf.py                       # 两个阶段，最小 batch
python test_moe_w4a8_perf.py --all-shapes          # 完整扫描
python test_moe_w4a8_perf.py --phase decode        # 仅 decode
python test_moe_w4a8_perf.py --skip-accuracy       # 仅性能
python test_moe_w4a8_perf.py --compute-bound       # 追加 6144 token 的 prefill 用例
python test_moe_w4a8_perf.py --long-seq            # 追加 8K 提示词的 prefill 用例
python test_moe_w4a8_perf.py --dtype fp16          # fp16 激活
python test_moe_w4a8_perf.py --rescale-group-size 256
python test_moe_w4a8_perf.py --warmup 10 --iters 100
```

`--long-seq` 与 `--sweep-configs` 一起使用时，还会在 8K 提示词下重跑一遍 prefill 的
tile 扫描。

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

在 24 GB 的 B60 上这个权衡有一条硬上限：上表两个 GEMM 每个 MoE 层约需 0.6 GB int8，
外加它所持有的约 0.3 GB int4，因此 48 层的 Qwen3-MoE 需要约 29 GB 预处理权重，放不
下。要缓存整个模型就得用多卡或更大显存的配置；在单块 B60 上，只缓存受 prefill 支配
的那些层，其余传 `cache_prepack=False`。

## 实测得到的默认值

下面的每一项默认值都来自上文那块 Arc Pro B60 上的 `-k sweep` 运行 (bf16 激活，decode 为
8 条 routed 行，prefill 为**每专家 384 行**，即测试套件所用的计算受限 batch)；tile 阶梯另
外还有一次在 8K 提示词路由 (每专家 512 / 341 行) 下的扫描。每种配置在计时之前都会先与第
一种配置做数值等价性检查。

prefill 路径上已经没有任何未实测的默认值了：单遍激活量化与 2D block store 过去仅凭推导
就默认开启，现在各自都有下面的实测表格。

**读表须知——噪声下限为 2–7%，而且是实测出来的。** 有两个 sweep 自带对照组。unroll
sweep 中有三个形状的 K 使其走到单遍量化 kernel，而 `UNROLL` 在那条路径上是死代码，因此
那些行是三组**完全相同**的 kernel：本次运行中它们的离散度分别为 3.3%、4.5%、7.3% (上一
次为 0.4%、2.1%、3.9%)。tile sweep 则每行都含一对重复测量，因为 `auto` 实际启动的就是其
中某一个显式 tile，这些重复对的离散度为 0.2–1.9%。凡是小于同一张表内对照组离散度的差
异，都应视为运行间波动；也不要把一张表里的数字与另一张表里的数字直接比较。

### Prefill tile

计算受限 batch 下 (两个模型都是每专家 384 行)：

| 形状 | `auto` | `128x128` | `128x256` | `256x128` | `256x256` |
|---|---|---|---|---|---|
| qwen3 up | 3.540 ms | **3.518 ms** | 3.585 ms | 4.404 ms | 3.970 ms |
| qwen3 down | 2.472 ms | 2.473 ms | **2.432 ms** | 2.696 ms | 2.547 ms |
| minimax up | 6.976 ms | **6.823 ms** | 6.878 ms | 8.899 ms | 8.019 ms |
| minimax down | **6.749 ms** | 7.227 ms | 6.874 ms | 9.096 ms | 7.823 ms |

8K 提示词下 (Qwen3-MoE 每专家 512 行，MiniMax 341 行)：

| 形状 | 每专家行数 | `auto` | `128x128` | `128x256` | `256x128` | `256x256` |
|---|---|---|---|---|---|---|
| qwen3 up | 512 | **4.371 ms** | 4.382 ms | 4.393 ms | 4.468 ms | 4.394 ms |
| qwen3 down | 512 | 3.075 ms | 3.030 ms | **2.903 ms** | 3.025 ms | 3.059 ms |
| minimax up | 341 | 6.564 ms | 6.673 ms | **6.449 ms** | 9.057 ms | 7.744 ms |
| minimax down | 341 | 6.466 ms | 6.725 ms | **6.450 ms** | 9.373 ms | 7.307 ms |

**`auto` 是对照组，不是候选项。** 它启动的就是阶梯选中的那个显式 tile — 上面除第二张表
中两个 qwen3 行 (当时阶梯还有 256 行档、选中 `256x256`) 之外都是 `128x256` — 因此每一行
里都含一对重复测量，即同一个 kernel 被测了两次。八对重复的两次读数相差 0.2–1.9%，这就是
读这两张表时应参照的运行间波动下限；它比 unroll sweep 中相同 kernel 的行 (本次 3.3–7.3%)
更严格，因为它就测在它所限定的那次扫描内部。

**M 方向的关键是 padding、不是寄存器——而一旦去掉 padding，它也就没有优势了。** 一个专
家会启动 `ceil(M / TileM)` 个**完整** tile，因此在每专家 384 行时，256 行的 tile 要为
384 行数据调度 512 行 (三分之一的 MAC 被浪费掉)，而 128 行的 tile 恰好是三个；在每专家
341 行时更差 (512/341 = 1.5)。所有存在 padding 的行中 `256x*` 那 1.05–1.45× 的落后全部
由此而来：在两个长 K 形状上 (主循环占主导)，同口径的比值 (qwen3 up 1.25×、minimax up
1.30×，均取 `TileN = 128`) 在噪声范围内**就等于** padding 比 512/384 = 1.33。

8K 提示词为 Qwen3-MoE 消去了这一项 — 每专家 512 行正好是 256 的整数倍，两种 tile 调度的
行数相同 — 而且这是整个套件中唯一这样的路由。在 `TileN` 相同的口径下，256 行 tile 在
qwen3 up 上读到 −2.0% / 0.0% (`TileN` 为 128 / 256)、在 qwen3 down 上读到 +0.2% /
−5.4%：最好也只是持平，而在主循环最短的那个形状上落后 5.4%。把每个 M tile 重复读取 B 的
次数减半确实省了访存，但它所需的 512 线程 work-group 又在调度粒度上把这点收益还了回去。

所以这一档是被**移除**，而不是加条件保留。唯一曾对它有利的读数来自更早那次每专家 256 行
的运行 (领先 1.3–3.9%，本身就在噪声下限之内)；它从未被实测赢过；而且阶梯只能看到
`total_tokens / E` 这个**平均**每专家行数，因此即便平均值能整除，路由不均衡时各个专家仍
会掉回 padding 悬崖。两个 256 行的 policy 仍然保留编译，可用
`ARK_MOE_W4A8_PREFILL_TILE` 选择，以便在寄存器预算不同的设备上重新扫描。

**N 方向没有代价。** 只要表格能对比，256 宽的 tile 都不落后：在每专家 384 行时它以
1.05× 拿下 minimax down、以 1.02× 拿下 qwen3 down，另外两个落后 0.8–1.9% (在下限之
内)；在 8K 提示词下它以 3.5–4.4% 拿下四个中的三个，并与 qwen3 up 持平。第一次 sweep 在
256 宽 N tile 上看到的 35–50% 悬崖来自 float C 影子 (参见
[Prefill：访存消息宽度与寄存器压力](#prefill访存消息宽度与寄存器压力))；第二次 sweep 中
残留的部分——三个形状上落后 0–8%——是在 epilogue 仍用**标量** store 时测得的，而当一个
32×64 的 fragment 改用少数几条 block 消息 (而非 128 条标量消息) 送出之后，它就消失了。
因此只要 `N % 256 == 0` (所有已发布的 N：1536 / 2048 / 3072) 阶梯就取 256 宽，否则取 128
宽——在那些形状上更宽的 tile 只会带来 padding。

于是完整的阶梯为：每专家 `< 16` 行 → `8x128`，`< 128` → `64x128`，其余情况下
`N % 256 == 0` 时取 `128x256`、否则取 `128x128`。六个 policy 全部保留编译，并可通过
`ARK_MOE_W4A8_PREFILL_TILE` 手动选择。

在阶梯选中的 tile 下，被扫描的四个形状在计算受限 batch 上达到 3.585 / 2.432 / 6.878 /
6.874 ms，即 86.3 / 63.6 / 101.2 / 101.2 TFLOPS；而两个版本之前的阶梯为
69.9 / 52.7 / 76.9 / 75.5 (1.23× / 1.21× / 1.31× / 1.34×)，并且现在每个形状都在最快
tile 的 1.9% 以内。四个形状中有两个达到 100 TFLOPS 的目标；qwen3 down 仍是例外，只有
64 TFLOPS——`K = 768` 时一个 tile 只有 12 个 k-tile，epilogue 与 prologue 因此占了相当大
的比例。在 8K 提示词下阶梯落在 4.393 / 2.903 / 6.449 / 6.450 ms，两个 qwen3 形状上的变
化正是移除 256 行档所带来的：up 投影持平，down 投影快 1.06×。

sweep 的各行只能**互相**比较，不能与上文的性能表比较：同样的形状、batch 与配置，在
`test_perf_prefill_compute_bound` 中读到 3.296 ms，而在 tile sweep 中读到
3.518–3.585 ms——因为性能测试会在同一负载上紧接着 W4A16 之后计时 W4A8，而 sweep 每个形状
都是从新建的用例开始。本节的所有结论都取自同一次扫描内部的差异。

有一点没有变：阶梯比较的是 `total_tokens / E` 这个**平均**每专家行数，因此在路由不均衡
时，即便平均为 384 行，各个专家的 tile 数仍可能相差很大。

### Prefill 激活量化

| 形状 | 标量 | 向量化 (默认) | 加速比 |
|---|---|---|---|
| qwen3 up | 3.633 ms | **3.255 ms** | 1.12× |
| qwen3 down | 2.437 ms | **2.221 ms** | 1.10× |
| minimax up | 7.704 ms | **6.702 ms** | 1.15× |
| minimax down | 7.111 ms | **6.637 ms** | 1.07× |

量化 routed 激活只是对 `[T, K]` 的一次流式遍历，而与它并行的 GEMM 本身就要搬运约
400 MB；仅仅把 32 字节 load / 16 字节 store 换成 256 字节 / 128 字节，就能带来整次调用
7–15% 的收益。`ARK_MOE_W4A8_ACT_QUANT_VEC=0` 可恢复标量映射。

一个 work-item 能让多少条这样的宽 load 同时**在途**，则由另一个开关
`ARK_MOE_W4A8_ACT_QUANT_UNROLL` 控制 (1、2、4 = 默认值)。只有 minimax up 是它真正的
A/B——另外三个形状走的是下面的单遍 kernel，`UNROLL` 在那里是死代码——在它上面三档分别读
到 1 时 6.982 ms、2 时 6.795 ms、4 时 6.837 ms。把 load 批量发出相对 `UNROLL = 1` 值
1.02–1.03×；而 2 与 4 之间 0.6% 的差距，远小于同一次扫描中那些死代码行 3.3–7.3% 的离散
度，因此默认值仍保持为 4 (上一次运行中它是 4 时 8.959 ms、2 时 8.967 ms、1 时
9.139 ms)。

### Prefill 单遍激活量化

| 形状 | K | 两遍 | 单遍 (默认) | 加速比 |
|---|---|---|---|---|
| qwen3 up | 2048 | 3.401 ms | **3.269 ms** | 1.04× |
| qwen3 down | 768 | 2.399 ms | **2.262 ms** | 1.06× |
| minimax down | 1536 | 6.752 ms | **6.645 ms** | 1.02× |
| minimax up | 3072 | 6.989 ms | 6.948 ms | —（不适用该路径） |

这是所有改动中唯一存在真实下行风险的一项：激活行在 absmax 与量化两遍之间保存在寄存器里，
一旦溢出，这一遍就会变慢而不是变快。实测没有溢出。minimax up 的 `K = 3072` 超过了 16 个
向量的门限，因此它那两行跑的是同一个两遍 kernel，0.6% 的差异正是这次扫描自带的噪声对照。

### Prefill store

| 形状 | 标量 store | 2D block store (默认) | 加速比 |
|---|---|---|---|
| qwen3 up | 3.884 ms | **3.395 ms** | 1.14× |
| qwen3 down | 2.850 ms | **2.349 ms** | 1.21× |
| minimax up | 7.808 ms | **7.133 ms** | 1.09× |
| minimax down | 7.814 ms | **6.721 ms** | 1.16× |

这是这组改动中 prefill 收益最大的一项，而且它出在 epilogue 而不是主循环：一个 32×32 的
sub-group fragment 由少数几条 block 消息送出，取代了 64 条只有半条 cache line 的标量消
息。名次也符合推理——qwen3 down 的主循环只有 12 个 k-tile、对 epilogue 的摊薄最少，且它
的 D 与权重一样大，因此收益最大。(上一次运行读到 1.16 / 1.35 / 1.12 / 1.20×，名次相同。)

### Prefill epilogue 边界保护

| 形状 | 带保护 | 内部 tile 快速路径 (默认) | 加速比 |
|---|---|---|---|
| qwen3 up | 3.573 ms | **3.473 ms** | 1.03× |
| qwen3 down | 2.434 ms | **2.341 ms** | 1.04× |
| minimax up | 6.729 ms | **6.703 ms** | 1.00× |
| minimax down | 6.635 ms | **6.569 ms** | 1.01× |

两列的主循环完全相同，只有 store 不同，因此这就是每个输出元素约 4 条指令的代价。它在主
循环最短的形状上占比最大——`K = 768` 的 qwen3 down 每个 tile 只跑 12 个 k-tile——正是当
初按指令数推理所预期的那个形状；而两个 minimax 形状读到持平，这也正是一项只改动 epilogue
的优化在最能摊薄它的形状上应有的表现。`ARK_MOE_W4A8_PREFILL_FULL_TILE=0` 可切回带保护的
epilogue；两者逐位相同。

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

这些读数相当于 B60 那 456 GB/s 引脚带宽的 59–69% (若以 device-copy 探测实际达到的
带宽为基准则是 68–79%)，因此只有 minimax down 越过了 300 GB/s 的目标。decode 每做
一次乘加就要读一个权重字节、别无其他，所以剩下的差距在访存消息效率，而不在算力。

## 环境变量

| 变量 | 作用 |
|---|---|
| `ARK_MOE_W4A8_AUTO_S8` | 覆盖 AUTO_S8 重缩放 block 大小。未设置 / `-1` 表示每个输出通道一个 scale (最快)。如果取值不是 `group_size` 和 64 的公倍数，或不能整除 K，则静默回退为 K。 |
| `ARK_MOE_W4A8_DECODE_MAX_TOKENS` | `phase="auto"` 时选择 GEMV 的 token 数上限 (默认 `128`)。 |
| `ARK_MOE_W4A8_DECODE_KSPLIT` | 合并访存的 K-split decode 映射，**默认开启**。设为 `0` 可回退到原来每个输出一个 work-item 的 GEMV (便于 A/B 对比)。形状不满足条件时该开关无效。 |
| `ARK_MOE_W4A8_DECODE_KSPLIT_NCOLS` | K-split 映射中每个 sub-group 处理的输出列数：`1`、`2` (默认) 或 `4`。取值越大，激活数据的加载可以摊到更多列上，但要求 `N % (16 × NCOLS) == 0`。默认值 `2` 来自实测，参见[实测得到的默认值](#实测得到的默认值)。 |
| `ARK_MOE_W4A8_DECODE_KSPLIT_CH` | 每个 lane 每次加载的 K 元素数 (即字节数)：`16` (默认) 或 `32`。`32` 可以把访存指令数减半、并让每个线程同时在途的字节数翻倍，代价是更多 GRF；它要求 re-scale block 至少为 512，否则会自动回退到 `16`。实测中它在所有形状上都慢于 `16`，因此只作为扫描项而非推荐值。 |
| `ARK_MOE_W4A8_PREFILL_TILE` | 强制指定 prefill 的 work-group tile：`8x128`、`64x128`、`128x128`、`128x256`、`256x128`、`256x256`。不设置 (默认) 时按 tile 阶梯自动选择：每专家 `< 16` 行 → `8x128`，`< 128` → `64x128`，其余情况取 128 行 tile，且只要 `N % 256 == 0` 就取 256 宽 (参见[实测得到的默认值](#实测得到的默认值))。两个 256 行的 tile 仍保留编译，但阶梯已不会选中它们；强制指定最多会慢 1.45×。 |
| `ARK_MOE_W4A8_ACT_QUANT_VEC` | 向量化的每 token 激活量化 (每个 lane 负责 4 或 8 个连续的 K 元素，而不是按 sub-group 宽度跨步)；**默认开启**，在被扫描的形状上带来 1.04–1.13× 的收益。设为 `0` 可强制使用标量映射以便做 A/B 测量。当 K 或缓冲区对齐不满足条件时该开关被忽略，此时本就会运行标量 kernel。 |
| `ARK_MOE_W4A8_ACT_QUANT_UNROLL` | 激活量化 kernel 在开始消费之前先加载的向量个数：`1`、`2` 或 `4` (默认，实测最快)。取值越大，一个 work-item 保持在途的字节越多——这一遍在每线程仅一条在途 load 时受限于延迟而非带宽——代价是 GRF 占用。`1` 即批量化之前的 kernel，可作为 A/B 基线；所有取值逐位相同。不在 `{1, 2, 4}` 中的取值会回退到默认值。只对向量化的**两遍**映射生效：下面的单遍 kernel 一次性发出整行，会忽略这个开关。 |
| `ARK_MOE_W4A8_ACT_QUANT_SINGLE_PASS` | 在 absmax 与量化两步之间把激活行留在寄存器里，而不是把 `[T, K]` 读两遍；在行放得下时**默认开启** (`VEC = 8` 下 `K ≤ 2048`，占每 lane 128 个 dword 中的 64 个)，在满足条件的形状上带来 1.00–1.05× 的收益。设为 `0` 可强制走两遍 kernel——更长的行本来也走它。两者逐位相同。 |
| `ARK_MOE_W4A8_PREFILL_FULL_TILE` | 对既不触及 M 边界也不触及 N 边界的 tile，跳过 epilogue 中的 store 谓词与 scale 下标钳制；**默认开启**，在被扫描的形状上最多带来 1.08× 的收益 (落后时也不超过 0.9%)。该判断在 work-group 内是一致的，因此代价是每个 tile 一次比较，而不是每个输出元素若干次。设为 `0` 可强制所有 tile 都走带保护的 epilogue (两者必须逐位相同)。 |
| `ARK_MOE_W4A8_PREFILL_STORE_2D` | 用硬件 2D block store 写 D，而不是每个 fragment 元素发一条 32 字节的标量消息；在输出满足对齐条件 (`N × sizeof(ElementD) % 64 == 0`，所有已支持形状均满足) 时**默认开启**，是这组改动中 prefill 收益最大的一项，达 1.12–1.35×。设为 `0` 可强制使用标量 store——不满足对齐门限的形状本来也走它。两者逐位相同。 |

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
`STATUS: PARTIALLY HARDWARE-VALIDATED`。各项性能扫描都已在一块 Intel Arc Pro B60 上
跑过，并且**每一个** dispatch 默认值现在都来自那些运行——tile 阶梯、激活量化的访存消息
宽度、unroll 深度与单遍那一档、内部 tile 的 epilogue、2D block store，以及 decode 的
`CH` / `NCOLS` (参见[实测得到的默认值](#实测得到的默认值))。所有被扫描的配置也都通过了
配置间的数值等价性检查；六项逐位一致性测试——`test_act_quant_vec_matches_scalar`、
`test_act_quant_unroll_matches`、`test_act_quant_single_pass_matches`、
`test_full_tile_epilogue_matches_predicated`、
`test_prefill_2d_store_matches_scalar` 与 `test_decode_ksplit_matches_legacy`——也都已
在设备上通过，因此每一项优化既有计时数据，也都与各自的前身做过比对。

仍需在设备上运行的部分：与 fp32 参考实现对比的精度扫描，它能立刻暴露 layout /
scale 相关的 bug。两个 8K 提示词的 prefill 用例
(`test_perf_prefill_long_seq`、`test_perf_prefill_tile_sweep_long_seq`) 现已跑过，并且
解决了阶梯中仅剩的那个未决问题：在每专家 512 行 — 唯一一种 256 行 tile 的 padding 不多
于 128 行 tile 的路由 — 上，256 行 tile 并不占优，因此这一档是被移除、而不是加条件保留。

本节此前列有三项"只经过推导、既未实测计时也尚未在设备上运行"的 prefill 改动，因为编写
它们的环境既没有 XPU 也没有 SYCL 编译器。这三项现在都已实测两次，并且都保持了原有默认
值：

| 改动 | 回退方式 | 实测结果 |
|---|---|---|
| 激活量化的批量 load——同时挂起 `UNROLL` 个请求而不是一个 | `ARK_MOE_W4A8_ACT_QUANT_UNROLL=1` | 在唯一真正走这条路径的形状上，`UNROLL = 2` 或 `4` 快 1.02–1.03×；2 与 4 之间的差异在噪声内 |
| 单遍激活量化——行数据留在寄存器中，`[T, K]` 只读一次而不是两次 | `ARK_MOE_W4A8_ACT_QUANT_SINGLE_PASS=0` | 1.00–1.06×；留在寄存器里的行并未溢出 |
| D 的 2D block store——每个 sub-group fragment 由少数几条 block 消息取代 64 条 32 字节的标量消息 | `ARK_MOE_W4A8_PREFILL_STORE_2D=0` | 1.09–1.35×，prefill 单项收益最大 |

本节此前把 2D store 列为"需要设备而不是一个开关"的工作，理由是同类 MoE kernel 都经由
`partition_sg_fragment_S` + `reorder` 抵达它，而其中没有任何一个是对**带 scale 的**
int32 累加器做 2D 写出的。事实证明那个参考对象选错了：`reorder` 搬的是寄存器、并不做
数值转换，因此它本来就承载不了 int32→fp16 的 epilogue。而同一个编译单元里的
`sycl_tla_dense_gemm.hpp` 早就在编译真正可用的那套序列 (`make_block_2d_copy_D(mma, D)`
+ `make_tensor_like<ElementD>(tCrC)` + `copy(copy_d, tCrD, tCgC)`，且正是 32 位累加器
配 16 位输出)，所以这项移植终究是纯 C++ 的改动。

单遍量化 kernel 是其中唯一存在真实下行风险的一项——留在寄存器里的行数据一旦溢出，这一
遍就会变慢而不是变快——而扫描结果在所有走这条路径的形状上都支持保留它。

这些扫描该拿什么作为参照，也已经改变了。本文档中 prefill 的 roofline 此前只统计了权重
字节数，把这些形状真正需要的带宽低估了 1.7–2.2×，让一个天花板只有 94 TFLOPS 的形状看
起来像是 kernel 的缺陷 (参见 [roofline](#权重并不是唯一的数据流))。把所有数据流都计入
之后，四个受算力约束的形状实际上跑在各自真实天花板的 60–74%，而受算力约束的 batch 也
从每专家 256 行提高到 384 行，好让 100 TFLOPS 在所有形状上都是可达的。剩下的差距在访
存而不是算术：目前仍摆在桌面上的最大一项收益，是把激活量化融合进 GEMM 的 A-tile 加载
中，这将同时消掉 int8 副本的写与读——5 条数据流中的 2 条，视 K 而定约占 14–22% 的流
量——但那是主循环的改动，需要在有设备的环境里开发。

### prefill 还剩下多少空间

在受算力约束的 batch 上，四个形状为 93.8 / 66.8 / 101.0 / 104.8 TFLOPS，即各自带宽天花
板的 61–76%，因此剩余空间分成两部分：这次调用仍在搬的流量，以及路由所决定的天花板。

| 方向 | 会改变什么 | 体现在哪里 |
|---|---|---|
| 把激活量化融合进 GEMM 的 A-tile 加载 | 消掉 5 条数据流中的 2 条 (int8 副本的写、以及随后的读回) — `K = 768` 时占 14%、`K = 2048` 时 21%、`K = 3072` 时 22% | 所有形状；这是仍未做的最大一项 |
| 让每个专家分到更多行 | kernel 里什么都不用改 — 它*抬高*的是天花板，因为只有权重这一条流不随 token 数增长 | 8K 提示词对 Qwen3-MoE 正是这个实验：每专家 512 行把天花板从 129 / 112 抬到 145 / 123 TFLOPS，实测的 98.2 / 70.5 TFLOPS 也随之上移 |
| `K = 3072` 的单遍激活量化 | 省掉对 `[T, K]` 的第二次读，在受算力约束的 batch 下约 450 MB | 仅 minimax up；它的一行是每 lane 96 个 dword，超过了 16 向量那一档 |

还有第四个方向已经收敛：256 行的 tile 能把每个 M tile 重复读 B 的次数减半，但 8K 提示词
的扫描已经在唯一一种它不比 128 行 tile 多 padding 的路由上测过它，结果最好也只是持平
(参见 [Prefill tile](#prefill-tile))，因此阶梯不再选用它。

`qwen3 down` (`N = 2048, K = 768`) 仍是那个异常值，只有约 64–70 TFLOPS：每个 tile 只有
12 个 k-tile，是四者中最短的主循环，其输出与权重一样大，而它在任何路由下的天花板也都是
四者中最低的。它同时也是上面这些访存侧改动收益最大的形状。
