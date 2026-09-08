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

> **这些上限属于 B60，不能套用到 B70。** 下文引用的 B70 数据里，W4A16 基线在
> qwen3 up 上跑到了 104 TFLOPS (bf16)，*高于*上表中 98 TFLOPS 的 bf16 峰值——说明
> 那是一块更大的芯片，因而基于上表推导出的所有上限、`BW@100T` 以及 PASS/FAIL 判定
> 在那里都是错的。测试脚本会探测自己所在的设备 (`_device_bandwidth_gbps`，在每张
> 性能表顶部打印为 `device copy bandwidth probe:`)；请以那一行为准来读判定结果，
> 并在把本节用于非 B60 芯片之前重新测量。

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

### 为什么大 batch 下 `vs w4a16` 只有约 1.2–1.7x

越过临界点之后，这个比值就不再取决于临界点，而是变成了一个流量比。B70 上以单条 8K
提示词测得的已发布契约数据：

| 形状 | E | N | K | rows/E | w4a16 | w4a8 | TFLOPS | vs w4a16 |
|---|---|---|---|---|---|---|---|---|
| qwen3 up | 128 | 1536 | 2048 | 512 | 3.964 ms | 2.845 ms | 144.92 | 1.39x |
| qwen3 down | 128 | 2048 | 768 | 512 | 2.413 ms | 2.062 ms | 99.98 | 1.17x |
| minimax up | 192 | 1536 | 3072 | 341 | 6.828 ms | 4.358 ms | 141.91 | 1.57x |
| minimax down | 192 | 3072 | 1536 | 341 | 7.023 ms | 4.260 ms | 145.18 | 1.65x |

这四行并不是对同一个内核质量的四次测量。把两条路径都代入 `_traffic_bytes`，W4A8
搬运的字节数是 int4 路径的 **1.43–1.81 倍**——翻倍的权重，加上 int4 路径根本不用付
的激活往返。int4 路径仍然受限于反量化，只有 151–195 GB/s；W4A8 跑在 325–439 GB/s，
也就是已经撞上了访存墙。以最快一行实测到的 439 GB/s 为准：

| 形状 | 上限 | 实测 | 占 roofline |
|---|---|---|---|
| qwen3 up | 158.7 | 144.92 | 91% |
| qwen3 down | 134.9 | 99.98 | **74%** |
| minimax up | 142.0 | 141.91 | 100% |
| minimax down | 158.7 | 145.18 | 92% |

也就是说四个形状里有三个已经在设备可达带宽的 8–9% 以内，而 `vs w4a16` 的差异跟随的
是算术强度 (K)，不是内核质量：qwen3 落后于 minimax 是因为它的 K 更小，用来摊薄激活
数据流的权重项也就更小。tile、store、epilogue 或调度上的任何改动都撼动不了这一点——
这些配置搬运的字节数完全相同。

唯一有效的杠杆是[契约 1](#契约-1--调用方直接提供-int8-激活)，而且在 B70 上它的价值超过
本文档里所有调优旋钮的总和：

| 形状 | w4a16 | 已发布 | 契约 1 | TFLOPS | vs w4a16 |
|---|---|---|---|---|---|
| qwen3 up | 3.679 ms | 3.034 ms | **2.090 ms** | 135.91 → **197.31** | 1.30x → **1.76x** |
| qwen3 down | 2.596 ms | 2.107 ms | **1.796 ms** | 97.83 → **114.78** | 1.16x → **1.45x** |

两个形状都越过了 100 TFLOPS 目标，而 `qwen3 up` 的 197 TFLOPS 是*已发布* W4A8 路径的
1.45 倍。差距的全部就在这里：不是 mainloop，也不是 tile——而是那次激活往返，只要让调用方
把它本来就持有的 int8 传进来就能删掉。另一个契约的方向则相反，见[还剩下什么](#还剩下什么)。

### int8 权重副本，以及树内反对它的先例

另一个流量杠杆是 prepack 本身。让 int4 留在 DRAM 里、在寄存器内扩宽成 int8，可以省掉
`E × N × K / 2` 字节——占 qwen3 up 流量的 18%、minimax up 的 24%——同时还能把
[prepack 显存开销](#内存开销)减半，而这正是单卡跑 30B 模型会爆显存的原因。

树内存在一个直接的先例，而且它的结论与当前设计相反。`sycl_tla_moe_prefill_s4_dpas.hpp`
之所以存在，就是因为 s4 prefill 路径曾经做过与本 kernel 的 AUTO_S8 prepack 完全相同的
事——通过一块 workspace 把 int4 物化成 `[E, N, K]` int8
(`launch_upcast_int4_sym_to_int8`)，再把这块缓冲交给 int8 DPAS mainloop。该头文件记录了
结果：

> upcast pass 写出 `E * N * K` 字节，mainloop 再经由 L2 把它们读回来——相比直接读取
> packed nibble，这基本上让 B 侧的全局内存流量翻了一倍。在 BMG 上，这次 workspace
> 往返使 DPAS 路径在 auto-round prefill 扫描的*每一个*形状上都退化到了传统
> bf16-dequant + 标准 GEMM 回退路径*之下*。

那条路径后来被改写为直接读取 packed 的 `[E, N, K/2]` nibble，并通过本 kernel 已经在用的
同一套 `cute::reorder(tBrB, tCrB)` 机制、借助
`NumericArrayConverter<ElementA, cutlass::int4b_t, N>` 在寄存器内解包，同时在
`k_tile * tile_k % group_size == 0` 边界上做延迟的按组折算。所以模板是存在的；本文档早前
声称它不存在 (当时指向的是 `sycl_tla_s8_gemm.hpp`，那个接口接收 `const int8_t* b`)，那是
看错了头文件。

尚未确定的是它在这里是否划算，因为 W4A8 并不等同于 s4 路径。AUTO_S8 换来的是一次全宽
int32 累加 (`blks == 1`)；改读 int4 就把按组折算重新放回 K 循环，于是在 `group_size = 32`
下一次累加会变成 64 次部分累加。这是用带宽换计算——在 kernel 受带宽限制时方向是对的，而
一旦不再受带宽限制，方向就是错的。契约 1 正是决定这一点的关键：它在不触碰 mainloop 的前提下
削减 qwen3 up 约 37% 的流量，而一个在启用契约*之后*已经受计算限制的 kernel，再增加折算去省
它本来就不必等待的字节，只会更慢而不是更快。契约 2 已经实测过，不进入这笔账——它是退化的，
所以本文早先草稿在这里用的约 45% 这个数字，统计的是实际上无法以正收益删掉的字节。

关于这个先例该采信到什么程度，有一点需要说明：它报告的那次退化是实测结果，但取代它的那个
头文件自身标注着 `STATUS: NEEDS-HARDWARE-VALIDATION -- untested single-pass port`，并由
env 开关控制、以便在运行时直接屏蔽。它提供的是 mainloop 结构的模板，而不是"这个结构在这里
更快"的证据。

因此顺序是：先在 B70 上实测契约 1，然后再决定 packed-nibble mainloop 值不值得做。如果在它
启用之后仍有形状不达标，那它就是下一项改动，而 s4 头文件就是可以照抄的模板——但上面那条
`STATUS` 提醒意味着它必须在硬件上验证过，而不只是移植完成。

## 削减 prefill 流量：两个可选的调用契约

把 roofline 反过来读。计算受限 batch 下四个形状跑在各自天花板的 61–76%，也就是说
把 kernel 做到完美最多也只有 1.3–1.6× — 但 `qwen3 down` 在每专家 384 行时的天花板
本身只有 109 TFLOPS，而且**当每专家行数趋于无穷时也只有 105 TFLOPS**：那时权重流
已经被摊薄掉，剩下的每行 `2 × K` 字节激活流量和 `N × 2` 字节输出完全不会被摊薄。
所以无论 batch 多大、mainloop 怎么调，这个形状都到不了 100 TFLOPS。必须减它的流
量。每专家 384 行时它搬运的那 554 MB 具体去了哪里：

| 数据流 | 字节 | 占比 |
|---|---|---|
| 权重 (唯一不随 token 数增长的一条) | 201 MB | 36% |
| 输出 `[T, N]` fp16 | 201 MB | 36% |
| 激活量化的来回 (读 fp16、写 int8、再读回 int8) | 151 MB | 27% |

权重那三分之一动不了。另外三分之二并不是 kernel 写得不好 — 它们是**接口层面**的
冗余，而且只有站在调用之外才看得见：

* fp16 激活是**上一个** kernel 写出来的 (down-proj 对应的是 SiLU/gate 的 elementwise
  pass)，它本来就可以直接写 int8；
* `D` 的每一行紧接着就会被 top-k 加权规约吃掉，`top_k` 行合成 1 行。

因此 `moe_gemm_w4a8` 增加了两个可选契约，让同时掌握两端的调用方把这些流删掉。两者
都是 opt-in、默认关闭，原有调用方式完全不变。

### 契约 1 — 调用方直接提供 int8 激活

```python
out = ark.moe_gemm_w4a8(
    qact,  # [T, K] int8，按专家排序的行
    weights_s8,
    wscales,
    num_tokens_per_expert,
    activation_scale=ascale,  # [T] fp32，每行一个反量化 scale
    out_dtype=torch.bfloat16,  # 期望的 fp16/bf16 输出类型
)
```

`qact[r, k] × ascale[r]` 必须能还原出原来的 fp16 行，这正是 kernel 自带量化器产生
的结果 (`round(x × 127 / absmax)`、`absmax / 127`)。生产者在它本来就持有整行的那批
寄存器里就能算出 absmax，所以上游是零成本；在本次调用里则删掉了五条流中的三条 —
fp16 读、int8 写、int8 读回 — 外加一次 kernel launch。GEMM 本身没有任何改动，因此
传入同样的 int8 就会得到同样的结果，差距不超过行 scale 那次除法本身被允许的**一个
输出格式步长** (`test_prequantized_activations_match_internal` 及其 decode 版本断言
的就是这个界)。

这个契约**不**保证的是：调用方从同一份 fp16 重新推导出的 int8 一定与 kernel 自己算
出来的字节相同。两边都在对 `x × 127 / absmax` 做舍入，但 SPIR-V 允许除法带若干 ulp
的误差，而 16 位激活的取值网格又粗到经常出现精确的舍入分界，于是少数元素会朝另一侧
舍入。这样的元素随后就是它参与的每一个点积的另一个**输入** — 是绝对量的扰动，只要
某个累加器抵消到接近零，用 ULP 衡量就没有上界，尽管其能量微不足道。这个差异属于调用
方的量化器，而不属于本次调用：把生产者实际算出的字节直接传进来，这个问题就不存在。
上面那两个测试则是从构造上回避它 — 量化的行落在 int8 网格上 (`a = q × 2^-e`、
`|q| ≤ 127`)，每个乘积都是整数，两边的量化器都没有分界要打破。

### 上游没有 int8 时如何用上契约 1：去重量化

上面的契约假设生产者能直接输出 int8。多数调用方并没有这样的算子——上一个算子输出
bf16，动态量化总得有人做。但这并不意味着契约 1 用不上，关键不在于量化做不做，而在于
它在**哪里**做。

量化器是逐行 absmax：`a.abs().amax(dim=1)`，然后 `round(x × 127 / absmax)`。其中
没有任何一项依赖专家。所以一个 token 的 int8 字节和 scale 是**这个 token 自身**的属性，
在它被路由到的每一行上都完全相同。

再数一下行数。up/gate 投影拿到的是按专家排序的 `[T, K]`，其中 `T = batch × top_k`
——而这些行正是 `batch` 个不同 token 的 `top_k` 份**副本**。在实际使用的
`top_k = 8` 下，调用内的量化因此把每个 token 的行读了 8 遍、absmax 算了 8 遍、
写出 8 行完全相同的 int8。其中八分之七是冗余的。

把同一个量化提到 permute 之上，这部分就消失了。调用方量化自己手里真正拥有的
`batch` 行，然后 permute int8 而不是 bf16：

```python
qact, ascale = quantize_rows(hidden_states)  # [batch, K] -> int8 + [batch] fp32
out = ark.moe_gemm_w4a8(
    qact.index_select(0, row_to_token),  # [T, K] int8，按专家排序
    weights_s8,
    wscales,
    num_tokens_per_expert,
    activation_scale=ascale.index_select(0, row_to_token),
    out_dtype=torch.bfloat16,
)
```

**这里必须诚实地做一次核对。**契约 1 单独测出来的数字之所以亮眼，有一部分原因是它把
量化**挪出**了计时区间。如果调用方自己去量化已经 permute 过的 `[T, K]`，收益会被
原样还回去——字节数没变，只是换了个计时的人。真正让工作量变小的是去重；而且它同时把
permute 减半，因为 permute 现在每个元素只搬 1 字节而不是 2 字节。把两侧一起算，
以 qwen3 up 投影的形状为例：

| | kernel | 调用方的 permute | 端到端 |
|---|---|---|---|
| 调用内量化 | 1141 MB | 268 MB（bf16） | **1409 MB** |
| 去重后 | 738 MB | 185 MB（量化 8192 行 + int8 permute） | **923 MB** |

端到端流量降低 1.53x，而 kernel 那一半逐字节就是测得 2.090 ms 的那次调用。
`test_perf_prefill_dedup_quant_long_seq` 会把两条路径**连同 permute 一起**计时
——只有这样的对比才能区分真正的节省和被挪走的开销——并断言两者输出一致。

#### 去重那部分由谁来量化，决定了它是赢还是输

行数变少并不自动等于时间变短，第一次实测就是这么说的：**0.95x**，是倒退。permute
如预期减半（1.011 → 0.468 ms），但量化 8192 行反而比在算子内量化 65536 行还贵。

原因在量化器，不在去重。`_quantize_rows` 是 eager-torch 参考实现——它先升到 fp32，
然后每个算子走一遍张量，一共约七遍，每遍都物化一个全尺寸中间结果。而调用内路径用的是
融合的 SYCL 量化器，每行只读一次、absmax 一直留在寄存器里。按每行算大约便宜十一倍，
这足以把 8 倍的行数削减吃掉。三个点在 B70 上、qwen3 up 投影形状、8K prompt 的实测：

| 路径 | permute | quant | GEMM | 合计 | vs 调用内 |
|---|---|---|---|---|---|
| 调用内量化 | 1.011（bf16） | 0.818（融合，65536 行） | 2.033 | 3.862 | — |
| 去重 + torch 量化 | 0.468（int8） | 1.558（torch，8192 行） | 2.033 | 4.059 | **0.95x** |
| 去重 + 融合量化 | 0.468（int8） | 0.136（融合，8192 行） | 2.033 | **2.637** | **1.46x** |

上面的流量模型预测 1.53x，设备返回 1.46x，说明推动这个结果的确实是字节数。以同样的
端到端口径对比 W4A16（其 GEMM 实测 3.679 ms，且它 permute 的是 16-bit），去重路径是
**1.78x**。

融合量化器没有独立的 Python 入口，所以 benchmark 不去假设它的开销：它在同一形状、
同一份权重上分别计时 16-bit 输入和 int8 输入的同一个 GEMM，取差值——两者唯一不同的
工作就是对这些行做的算子内量化。在 `T` 行和 `batch` 行上各做一次，还能交叉验证这个
开销确实与行数成正比（流式 pass 必然如此）；测试会断言这个比值落在 `top_k` 的 2 倍
以内，这样一个其实只是测量噪声的差值就不会悄悄变成一个醒目的结论数字。实测是
**8 倍行数对应 6.0x**——略低于线性，方向正是固定的每次 launch 开销所预期的（较小规模
下每行贵 33%），这也是为什么这个检查用的是一个区间而不是相等。

实际结论是：**不要用 eager-torch 量化器去做去重。**值得上线的版本是把量化折进产出
`hidden_states` 的那个算子的 epilogue（MoE 之前的 norm），那里行本来就在寄存器里、
absmax 是免费的——也就是上面契约里说的“上游做这件事是免费的”。退一步，用一个融合的
量化 kernel 处理 `[batch, K]` 的 hidden states，就是 0.136 ms 那一列所代表的；而
1.558 ms 那一列是改用 eager torch 的代价。

有两条边界需要讲清楚：

* **只适用于 up/gate。**down 投影的 `T` 行是 SiLU 的输出：每一条路由行对应一行不同的
  数据，没有可去重的东西。它走向同一契约的路径是把量化折进 SiLU 的 epilogue——那里
  本来就在写这个张量、行也本来就在寄存器里，属于契约 1 所说的“免费”，实测 1.45x。
* **`[T, K]` 的 int8 仍然会被物化。**要把它也去掉，就得在 mainloop 内部 gather A，
  直接读去重后的 `[batch, K]` int8（16.8 MB，小到可以常驻 cache）。那会再省下
  134 MB 的写以及大部分读，但也会把 A 侧的 2D block load 变成逐行 gather——正是让
  契约 2 失利的那一类改动。这是 kernel 改动而非调用方改动，没有硬件实测就不该动。

### 契约 2 — 把 top-k 规约折进 epilogue

```python
out = ark.moe_gemm_w4a8(
    activations,
    weights_s8,
    wscales,
    num_tokens_per_expert,
    row_to_token=row_to_token,  # [T] int32，路由行 -> 模型 token
    routing_weights=routing_weights,  # [T] fp32，该行的门控权重
    output_rows=batch,  # -> [batch, N] fp32，需预先清零
)
```

epilogue 不再写出 `[T, N]` 让调用方自己规约，而是把每个元素乘上该行的 routing
weight 之后 `atomic_add` 到 `out[row_to_token[r]]`。`T × N × 2` 的写变成
`batch × N × 4` 的读改写 — 在 `top_k = 8` 时即使按双向计费也只有原来的四分之一 —
而且那个独立的规约 kernel (又一次 `T × N` 读加 `batch × N` 写，本文的流量模型从来
没算过它，因为那是另一次调用) 整个消失。

它只适用于**第二个**投影：up/gate 投影的输出要按路由行进入 SiLU，必须保持未规约状
态。启用前还有两点需要知道：

* fp32 atomic 的累加顺序不确定，因此结果**不是逐位相同**的 — 两次运行之间也不是。
  它是用 harness 的 SNR/余弦门槛而不是相等断言来对照无融合路径验证的
  (`test_fused_reduce_matches_unfused`)。
* 输出缓冲区必须由调用方清零；当分配由 Python 包装层负责时，它会分配一个已清零的。

### 这两项值多少

保持每个形状**实测**的有效带宽不变 — 即假设 kernel 一点没变好，只是搬得更少 — 并
使用 harness 自己的流量模型：

| 形状 | 路由 | 实测 | + int8 输入 | + 融合规约 | 两者 |
|---|---|---|---|---|---|
| qwen3 up (N=1536, K=2048) | 384 行/专家 | 93.8 | **137** | 不适用 | 不适用 |
| qwen3 down (N=2048, K=768) | 384 行/专家 | 66.8 | 84.0 | 81.6 | **109** |
| minimax up (N=1536, K=3072) | 384 行/专家 | 101.0 | **152** | 不适用 | 不适用 |
| minimax down (N=3072, K=1536) | 384 行/专家 | 104.8 | 129 | 120 | **152** |
| qwen3 up | 8K 提示词 | 98.2 | **152** | 不适用 | 不适用 |
| qwen3 down | 8K 提示词 | 70.5 | 91.0 | 88.1 | **123** |
| minimax up | 8K 提示词 | 93.5 | **137** | 不适用 | 不适用 |
| minimax down | 8K 提示词 | 95.4 | 116 | 108 | **135** |

(「不适用」= 融合规约不适用于 up/gate 投影，所以这些形状能达到的数字看
`+ int8 输入` 那一列。)

B70 现在已经实测了 `+ int8 输入` 一列中 8K 提示词的那两行，模型算得相当接近：它推算
`qwen3 up` 为 **152**、`qwen3 down` 为 **91.0**，而设备实际给出 **197.31** 和
**114.78**。两者都超出了推算——模型是按整轮扫描的平均带宽给已发布路径的激活往返定价的，
而删掉它同时还改善了剩余部分的局部性，这是单纯数字节看不到的。两个形状都**只靠契约 1**
就越过了 100 TFLOPS，而这张表原本认为只有 `qwen3 up` 能做到。

相比之下，`融合规约` 与 `两者` 这两列现在已知在另一个方向上是错的。B70 实测契约 2 是
**退化的**，因为这个模型按 fused epilogue 删掉的字节给它定价，而它实际上是按新增的约
1.34 亿次 device 作用域原子操作定价的 ([还剩下什么](#还剩下什么))。请把 `+ int8 输入`
一列当作已在 8K 路由下得到验证，把两个融合列当作实测未能达到的上界。

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

# 削减流量的调用契约：A/B 扫描、8K 提示词、等价性测试
pytest -v -s test_moe_w4a8_perf.py -k contract
pytest -v -s test_moe_w4a8_perf.py -k "prequantized or fused_reduce"
```

`test_perf_prefill_contract_sweep` 和 `test_perf_prefill_contracts_long_seq` 会在
两种 prefill 路由下，对两个[调用契约](#削减-prefill-流量两个可选的调用契约)的四种
组合分别计时，而 `test_perf_prefill_prequant_long_seq` 则在 8K 路由下单独测量契约 1——
它是推理框架无需改动层代码就能采纳的那一种配置，也是先前"两个契约一起开"的测试无法
看穿的那一种。由于每个契约都会改变这次调用搬运的内容，每一行的 `DRAM GB/s`、
`BW@100T` 和天花板都按**该行自己的**流量模型计算，因此各列在不同契约之间仍然可比。
融合规约的行是与规范化后的基线 (由 harness 对无融合输出做规约) 在 SNR 门槛下比较
的，而不是其它扫描使用的逐位相同。

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
python test_moe_w4a8_perf.py --contracts           # 追加 int8 输入 + 融合规约的 prefill 用例
python test_moe_w4a8_perf.py --dtype fp16          # fp16 激活
python test_moe_w4a8_perf.py --rescale-group-size 256
python test_moe_w4a8_perf.py --warmup 10 --iters 100
python test_moe_w4a8_perf.py --rounds 5             # drift 偏大时增加轮转次数
```

`--rounds` 控制扫描在各配置之间轮转的次数 (默认 3)。每一轮计时
`ITERS // rounds` 次迭代，因此调大它几乎没有额外开销；当 `drift` 一列与想要区分的
差距相当时，应当调大它。

`--long-seq` 与 `--sweep-configs` 一起使用时，还会在 8K 提示词下重跑一遍 prefill 的
tile 扫描；`--contracts` 以同样方式追加契约的 A/B 扫描。

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

两个可选的 prefill 契约可以削减本次调用搬运的流量 (见
[削减 prefill 流量](#削减-prefill-流量两个可选的调用契约))，两者默认关闭：

```python
# 调用方已经持有 int8 激活以及每行一个的反量化 scale。
out = ark.moe_gemm_w4a8(
    qact,  # [total_tokens, K] int8
    weights_s8,
    wscales,
    num_tokens_per_expert,
    activation_scale=ascale,  # [total_tokens] fp32
    out_dtype=torch.bfloat16,  # 返回的 fp16/bf16 输出类型
    rescale_block_size=block,
)

# 仅第二个投影：把 top-k 加权规约折进 epilogue。
out = ark.moe_gemm_w4a8(  # -> [batch, N] fp32
    activations,
    weights_s8,
    wscales,
    num_tokens_per_expert,
    row_to_token=row_to_token,  # [total_tokens] int32
    routing_weights=routing_weights,  # [total_tokens] fp32
    output_rows=batch,
    rescale_block_size=block,
    phase="prefill",
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

### 预取深度与 K —— 已实测两轮，结论是否定的

`moe_w4a8_prefill_prefetch_dist` 对所有形状都返回同一个常量 (`3`)。为质疑它而
搭建的扫描已经在 B70 上跑过两轮，结论是这个常量维持不变。

值得质疑的理由在于：`3` 在各个已发布形状上所占 mainloop 的比例差别很大。在 64
元素的 k-tile 下，qwen3 down (`K = 768`) 的 mainloop 只有 12 次迭代，qwen3 up
(`K = 2048`) 是 32 次，minimax up (`K = 3072`) 是 48 次，因此 prologue 在第一个
循环里占四分之一，在最后一个里只占十六分之一。qwen3 down 也正是还有余量的那个
形状——它只达到 DRAM 上限的 74%，而另外三个是 91–100%。如果 prologue 真是原因，
深度就应该把这些形状区分开。

但它没有，而真正给出结论的是第二轮。B70，`TFLOPS`，第 1 轮 / 第 2 轮：

| 深度 | up @384 | down @384 | up @512 | down @512 |
|---|---|---|---|---|
| 1 | 123.29 / 123.72 | 85.34 / 85.32 | 135.99 / **143.08** | 89.45 / 89.86 |
| 2 | 122.82 / **124.65** | **85.37** / **85.54** | 136.67 / **144.61** | 92.93 / 92.52 |
| 3 *(默认)* | **130.09** / 122.65 | 85.28 / 85.14 | 141.96 / 136.44 | 93.32 / **93.16** |
| 4 | 129.87 / 121.34 | 84.82 / 84.81 | **143.85** / 135.26 | **93.33** / 92.97 |
| 6 | 127.78 / 121.02 | 84.59 / 84.79 | 141.28 / 135.14 | 90.33 / 90.63 |
| 8 | 119.76 / 120.14 | 84.65 / 85.10 | 130.49 / 133.05 | 90.34 / 90.77 |

要横向比较两轮，而不是纵向读某一列。在 qwen3 up 上排名**发生了反转**：第 1 轮的
峰值在深度 3–4，第 2 轮的峰值在深度 1–2；而在*同一个深度*上两轮之间的差异
(@384 最大 7.0%，@512 最大 6.4%) 比这次扫描想要区分的 4–10% 差距还要大。任何排名
都撑不过这种情况。把每一轮按各自的最优值归一化，就能看出实际被测量的是什么：

| 扫描位次 | 第 1 | 第 2 | 第 3 | 第 4 | 第 5 | 第 6 |
|---|---|---|---|---|---|---|
| up @512 第 1 轮 | 94.5 | 95.0 | 98.7 | **100.0** | 98.2 | 90.7 |
| up @512 第 2 轮 | 98.9 | **100.0** | 94.4 | 93.5 | 93.5 | 92.0 |

第 2 轮从第二个位次起单调下降，第 1 轮先热身再下降，而**最后测量的那个配置在两轮
里都是最慢的**。这是扫描过程中的频率下降 (clock droop)，而不是深度本身的性质：
qwen3 up 跑在 144 TFLOPS 的 int8 负载上，把芯片加热的速度快过扫描测量它的速度，
于是谁被排在前面测，谁就赢。

qwen3 down 没有这个现象——它在两轮中每个深度都能复现到 0.5% 以内，因为它只有
93 TFLOPS 且 48% 是写入，功耗低得多。而它也是唯一有真实信号的形状：深度 1 确实慢
4%，深度 2/3/4 构成一个平台，6/8 又退回约 3%。两轮结论一致，而已发布的默认值正好
落在这个平台的中间。

所以：不需要 K 相关的启发式，而且这里也没有任何东西能解释 qwen3 down 的
74%——参见[还剩下什么](#还剩下什么)。

被修的是测试工具而不是 kernel。`_sweep_timings` 现在会把各个配置在
`SWEEP_ROUNDS` 轮之间轮转 (round-robin)，而不是把每个配置一次跑完，取每个配置
受降频影响最小的那一轮，并按配置打印轮次间的 `drift`；当优胜者领先幅度落在该
drift 之内时，"best configuration" 那一行会被抑制。每轮的迭代次数是
`ITERS // SWEEP_ROUNDS`，因此扫描并不会比以前更慢。`run_perf` 也做了同样的轮转
改造——它此前先测 w4a8、最后测 w4a16，把headline `vs w4a16` 比值的分子与分母放在
了降频曲线的两端，从而使该比值*偏高*。

```bash
pytest test_moe_w4a8_perf.py -k "prefetch_sweep" -v
python test_moe_w4a8_perf.py --skip-accuracy --prefetch --long-seq --rounds 5
```

### 常驻 kernel 何时去领下一个 tile

Prefill GEMM 是一个**常驻 (persistent)** kernel：grid 是按设备规模来定的
(`sm_count` × 单个 Xe core 能容纳的 work-group 数)，而不是按问题规模，所以一个
work-group 并不只负责一个 tile——它会循环取活，下一个 tile 的编号来自一次
device-scope 的 `atomicAdd`，落在同一个 dword 上，所有常驻 work-group 都会撞上它。

这次领取过去是发在它所对应的那个 tile **之后**的：

```
   [ GEMM tile ]  -> atomicAdd -> 等待 -> [ GEMM tile ]  -> atomicAdd -> 等待 ...
                    \___________________/
                     这一段没有任何东西可以重叠
```

一次打在竞争 dword 上的 L2 往返，而且恰好落在循环中 work-group 手里没有任何
在途工作可以掩盖它的位置。现在它发在**之前**：

```
   atomicAdd -> [ GEMM tile ] -> 使用结果 -> atomicAdd -> [ GEMM tile ] -> ...
                \____________/
                 这条消息在整个 mainloop 期间都在途
```

有两个细节决定了这是真的隐藏了延迟，而不只是把停顿挪了个位置：

* **结果先留在私有寄存器里，而不是直接写进 SLM。** 紧跟 atomic 之后的 SLM store
  会让 lane 0 当场等这个结果，进而让整个 work-group 卡在 mainloop 的第一个
  barrier 上。把 store 放到 GEMM 之后才真正推迟了这次等待；而 atomic 本身不可能
  被下沉到 mainloop 的 barrier 之后，所以它会留在写下的位置。
* **SLM 槽位在两个 dword 之间乒乓。** 只用一个 dword 时，lane 0 为 tile `i + 1`
  写入的值可能抢在某个较慢的 sub-group 读 tile `i` 的值之前，那就需要每个 tile
  再加一个 barrier 来防止；用两个槽位则让写和读落在不同地址上。

计算哪些 tile 完全没变——每个 tile 一次领取、同样的编号、同样的消费顺序——所以两种
顺序是**逐位相同**的，`test_prefill_claim_early_matches` 用 ragged batch 上的
`torch.equal` 来断言这一点 (每专家 300 行，于是每个专家同时有完整 tile 和不完整
tile，tile 遍历必须遵守的专家边界也就进入了比较范围)。

这项改动同时把工作计数器的清零挪到了主机侧。它过去是由 group 0 / lane 0 在 kernel
入口处在设备上清零的，而这与其他所有 work-group 打在同一个 dword 上的 `atomicAdd`
之间没有任何顺序保证——这个竞态之所以一直成立，只是因为一个 work-group 的首次领取
要等一整个 GEMM tile 之后、也就是几微秒之后才会发生。提前领取把这个窗口压缩到了
几条指令，所以 `MoEGEMMLauncher_w4a8` 现在用 `queue::memset` 填这个 dword，并让
kernel `depends_on` 这次填充。一个 dword、一条额外命令，没有任何同步。

这项改动值多少，上界取决于一个 tile 有多少工作可以用来掩盖这次停顿，所以它应该在
**down** 投影上显现、而在 up 投影上几乎看不到：在同样的 tile 形状、同一个 kernel 下，
`K = 768` 的一个 tile 只有 12 个 k-tile，而 `K = 2048` 有 32 个。qwen3 down 同时也是
距离自身带宽上限最远的那个形状。如果所有形状都打平，那也是一个真实的结论——它会说明
这次领取从来就不是瓶颈，down 投影的差距在 prologue、epilogue 或者 D 的写上，参见
[还剩下什么](#还剩下什么)。

```bash
pytest test_moe_w4a8_perf.py -k "claim_early" -v
python test_moe_w4a8_perf.py --skip-accuracy --claim-early --long-seq --rounds 5
```

### 还剩下什么

在 prologue 已被实测排除、tile 阶梯也已扫描过之后，剩下的差距在于流量，而且它的
分布并不均匀：

| 形状 | 读 | 写 | 写占比 |
|---|---|---|---|
| qwen3 up | 805 MB | 336 MB | 29% |
| qwen3 down | 352 MB | 319 MB | **48%** |

qwen3 down 未规约的 `[T, N]` 输出单独就有 268 MB——占整个调用搬运量的 40%，是它最大
的单条数据流，比权重还大。这就是那 74% 的全部原因：它是整个测试集中写入占比最高的
形状，只跑出 325 GB/s，而 qwen3 up 能跑到 401 GB/s——这两个数正是测试脚本自己在
`DRAM GB/s` 一列打印出来的值，所以上面的拆分是对实测结果的分解，而不是另一套模型。
任何预取深度、tile、store 模式或调度改动都撼动不了一个受写入限制的形状——而这正是
各轮扫描反复给出的结果。

这条数据流不是靠重新调度、而是靠 fused 规约契约*直接消除*的：它把 `[T, N]` 的写入
换成 `[batch, N]` 的累加。这是唯一还能改变字节数的杠杆——而在 B70 上它**输了**：

| 形状 | 已发布契约 | 两个契约都开 | TFLOPS | `DRAM GB/s` |
|---|---|---|---|---|
| qwen3 up | 3.034 ms | 3.254 ms (**0.93x**) | 135.91 → 126.72 | 376.1 → 195.9 |
| qwen3 down | 2.107 ms | 3.198 ms (**0.66x**) | 97.83 → 64.47 | 318.4 → 120.7 |

字节更少，时间更长。这个组合本身就是全部结论：有效带宽一列直接腰斩，而一个搬运
量少了 44% 的受带宽限制 kernel 不可能出现这种情况。fused 路径的开销，无论是什么，
都不是花在 DRAM 上的。

它花在 epilogue 上。`store_fused` 对**每一个输出元素**都发一次 device 作用域的
`atomic_add_f32`——qwen3 up 是 1 亿次，qwen3 down 是 1.34 亿次——而且由于 scatter
无法使用 block store，开启它同时也放弃了实测价值 1.12–1.35x 的 2D store。把两行
数据分别解出原子操作速率，结果是一致的：up 多花约 1.3 ms、down 多花约 1.6 ms，都是
约 800 亿次原子操作/秒。两个形状、两种路由、同一个常数——代价模型就是原子操作次数，
而它由 `T * N` 决定，任何调优参数都碰不到。

所以在这个硬件上，契约 2 不是一个值得采纳的契约。它删掉的那次 `[T, N]` 写入确实存在，
但一次合并的 268 MB store 胜过 1.34 亿次分散的读改写，其优势远超字节数所暗示的程度。

**契约 1 单独测下来，就是答案。** 此前 B70 上的两轮测试都是把两个契约一起打开的，
于是契约 2 那约 1.5 ms 的开销淹没了激活往返所节省的一切，那两轮数据对它什么都说明不了。
单独隔离出来之后：

| 形状 | w4a16 | 已发布 | 契约 1 | TFLOPS | vs w4a16 |
|---|---|---|---|---|---|
| qwen3 up | 3.679 ms | 3.034 ms | **2.090 ms** | 135.91 → **197.31** | 1.30x → **1.76x** |
| qwen3 down | 2.596 ms | 2.107 ms | **1.796 ms** | 97.83 → **114.78** | 1.16x → **1.45x** |

它删掉 `3 * T * K` 字节——qwen3 up 全部 1141 MB 中的 402 MB——而此前推算的约 1.97 ms
与实测的 2.090 ms 相差在 6% 以内。与契约 2 不同，它保持结果逐位一致。

一个显而易见的反驳是：它需要上游能产出 int8，而多数流水线并没有——上一个算子输出 bf16，
动态量化总得有人做。但它依然是可达的，因为逐行 absmax 不依赖专家，所以在 up/gate 投影上，
调用内的那一遍量化处理的是每个 token 的 `top_k` 份相同副本。把 `batch` 个不同的行量化
一次、再 permute int8，就能到达同一个调用，实测**端到端 1.46x**（已包含调用方的
permute）——完整过程见[去重量化](#上游没有-int8-时如何用上契约-1去重量化)，其中也包括：
如果去重后的行交给 eager-torch 量化器来做，它反而是**倒退**。

```bash
pytest test_moe_w4a8_perf.py -k prequant_long_seq -v   # 只开契约 1
pytest test_moe_w4a8_perf.py -k dedup_quant -v         # 上游没有 int8 时如何达到它
pytest test_moe_w4a8_perf.py -k contracts_long_seq -v  # 两个契约都开，用于对照
```

后面这个扫描此前的对比方式并不公平，而且偏差的方向恰好不利于该契约：它把每个配置都只按
一次裸 GEMM 计时，于是 fused 那一行要在自己的 epilogue 里承担规约开销，而其余各行留下
一个 `[T, N]` 张量、把规约丢给调用方——那部分工作从来没有被计时。现在扫描会把每个未融合
配置本应承担的规约计入，并单列出一个 `+reduce` 列，因此 `vs default` 比较的是*产出已规约
输出*的代价，而不是"从 GEMM 返回"的代价。值得强调的是：这次修正把计费方式改成了对契约 2
更有利的方向，而契约 2 依然输了。

这两列应当作为上下界来读，因为单看任何一列都不是答案：

* 只看 `ms` (也就是此前的行为) 是该契约价值的**下界**——它默认基线可以完全跳过规约。
* `ms + reduce` 是**上界**——规约是用 torch 的 `index_add_` 计时的，它会产生 fp32 临时
  张量，而手写 epilogue 并不需要。

对 `qwen3 down` 来说，这两个界之间的差距并非细节：它未规约的 `[T, N]` 输出有 268 MB，
因此它交还给调用方的那次规约要读这 268 MB、再多写 67 MB。这是整个调用中最大的一条数据
流，而旧的计费方式一个字节都没算。

这笔开销只会记在 down 投影那几行上，这个契约本身也是如此。MoE 层只对第二个 GEMM 的输出
做规约；up/gate 的结果仍然是展开状态、每个路由 token 一行，直接进入 SiLU。在那里做融合
并不是调用方能够采纳的契约，因此 `run_perf` 不再把它应用到那几行——先前表格中 `up` 的
退化，测的是一个谁都无法上线的配置。契约 2 是一个只属于 down 投影的契约。

在真实的 MoE 层里契约 1 是免费的：`up`/`gate` 共享激活，因此 int8 副本只需做一次就能
同时喂给两者。应当把它当作调用约定，而不是一项优化。契约 2 的前提——`down` 的下游本来
就是 epilogue 可以顺手完成的 unpermute + 加权求和——本身是成立的，但在这个硬件上，
epilogue 做这件事比单独一遍做得更差。

### 带宽探针在说谎，而且这是有后果的

契约 1 那一轮打印出了 `118% of the 167 TFLOPS bandwidth ceiling`，而这是 roofline
不可能出现的情况。天花板来自一个设备拷贝探针，而该探针在同一套测试的三次连续运行中分别
报出 **439、373 和 299 GB/s**——对一个被各处判定当作硬件常数的数字来说，这是 47% 的摆幅。
在 299 GB/s 时，它甚至*低于* kernel 自身正在跑出的 353 GB/s。

这从来就不只是显示问题。`_assert_targets` 会对任何天花板低于目标的行免除目标校验，理由是
再改 kernel 也达不到——于是一个测低了的探针，会把"受带宽限制、不可达"这个免死金牌发给
其实只是慢的行，而 `--enforce-targets` 也就不再真正生效。

两处修复。探针现在取多轮中的最优值，而不是单次突发的中位数，理由和本文其它所有测量都做
最小值过滤是同一个：最快的那次拷贝受降频污染最小。另外，如果某一行搬运自身流量的速率高于
探针，那就是设备至少能维持该速率的直接证据，因此天花板会按它重新标定——探针永远只是一个
下界。这个修正只会抬高天花板，也就是只会让判定更严格。

对同一轮数据重新解读之后，结论就变了：

| 形状 | 原打印 | 修正后天花板 | 修正后 |
|---|---|---|---|
| qwen3 up | 167 的 118% | 197.3 | **100%** — 已在 roofline 上 |
| qwen3 down | 119 的 97% | 140.6 | **82%** — 还有 18% 余量 |

所以 `qwen3 down` **并没有**像那个坏探针的"97%"所暗示的那样已经到头。在契约 1 之下，
它是唯一还留有内核余量的形状，而上面关于写入占比的分析正是原因所在。

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
| `ARK_MOE_W4A8_PREFILL_STORE_2D` | 用硬件 2D block store 写 D，而不是每个 fragment 元素发一条 32 字节的标量消息；在输出满足对齐条件 (`N × sizeof(ElementD) % 64 == 0`，所有已支持形状均满足) 时**默认开启**，是这组改动中 prefill 收益最大的一项，达 1.12–1.35×。设为 `0` 可强制使用标量 store——不满足对齐门限的形状本来也走它。两者逐位相同。使用融合 top-k 规约时会自动关闭，因为那是 scatter，用不了 block store。 |
| `ARK_MOE_W4A8_PREFILL_PREFETCH` | prefill mainloop 预取 A/B 的 k-tile 深度：`1`–`8`，默认 `3`。预取越深越能掩盖 DRAM 延迟，代价是 GRF 和更长的 prologue——对短 mainloop 影响最大 (`qwen3 down` 每个 tile 只有 12 个 k-tile)。所有取值逐位相同；`test_perf_prefill_prefetch_sweep` (compute-bound batch) 与 `test_perf_prefill_prefetch_sweep_long_seq` (8K 提示词路由) 会对完整的 `1 / 2 / 3 / 4 / 6 / 8` 区间计时。超出 `1`–`8` 的取值回退到默认值。扫描结果显示排名平坦，因此默认值保持不变——参见[预取深度与 K](#预取深度与-k--已实测两轮结论是否定的)。 |
| `ARK_MOE_W4A8_PREFILL_CLAIM_EARLY` | 常驻 prefill kernel 何时从 device-scope 工作计数器领取下一个 tile：**`1` (默认)** 把 `atomicAdd` 发在 GEMM 之前，于是这次 L2 往返在整个 mainloop 期间都在途；`0` 恢复旧顺序，即它完全暴露在两个 tile 之间的空档里。计算哪些 tile 完全没变，因此两者逐位相同 (`test_prefill_claim_early_matches` 断言 `torch.equal`)；`test_perf_prefill_claim_early_sweep{,_long_seq}` 会对这一对计时。tile 本身可用于掩盖停顿的工作越少收益越大——参见[常驻 kernel 何时去领下一个 tile](#常驻-kernel-何时去领下一个-tile)。 |

## 形状约束

kernel 要求：

* `N % 16 == 0` (GEMV 的 N tile 与 DPAS 的 N tile)
* `K % 64 == 0` (DPAS 的 K tile)
* `group_size % 8 == 0` 且 `K % group_size == 0`
* 解析出的重缩放 block 必须是 64 的倍数并且能整除 K

Qwen3-MoE 的两个 GEMM 都满足以上条件 (`K = 2048` 和 `K = 768`)。

decode 的 K-split 映射还额外要求重缩放 block 不小于 256 且是 16 的倍数；不满足的形
状会退回到原 GEMV，而不是报错。

## 源码结构

整条路径按对 cutlass（以及 bestla）的依赖拆成四个头文件，使得任何一个翻译单元都不会
实例化过多 kernel：

| 头文件 | 内容 | 需要 CuTe |
| --- | --- | --- |
| `sycl_tla_moe_w4a8_scratch.hpp` / `.cpp` | 设备 scratch 显存（`DeviceMemoryPool`），头文件只放声明，实现放在 `.cpp` | 否 |
| `sycl_tla_moe_w4a8_helpers.hpp` | host 辅助函数、prefill tile 阶梯、四个对外入口 | 否 |
| `sycl_tla_moe_w4a8_kernels.hpp` | 激活量化、AUTO_S8 prepack、decode GEMV 及其 K-split 变体 | 否 |
| `sycl_tla_moe_w4a8.hpp` | DPAS tile policy、分组 prefill GEMM 及其 launcher | 是 |

`sycl_tla_generation.cmake` 据此生成 19 个翻译单元而不是一个：一个只看到 helpers 的
dispatcher，十二个 prefill 翻译单元（每个 dtype x tile 一个，各含一个 DPAS kernel），
以及六个完全不依赖 cutlass 的翻译单元，分别对应 decode、激活量化和 prepack（每个
dtype 一个）。拆分之前，单个翻译单元要实例化全部 52 个 kernel，编译器 RSS 峰值约
4.2 GB；这里的拆法与 `sycl_tla_moe_prefill_s4_*.cpp` 拆分 S4 prefill 的方式一致。
运行时 API 与各项 dispatch 决策均不受影响。

scratch 池按照 `sycl_tla_moe_decode_scratch.{hpp,cpp}` 的同样方式单独拆出：它需要
`utils.hpp` 里的 `DeviceMemoryPool`，而该头文件会带入 bestla 的 AVX512/xbyak JIT
头文件；如果从头文件包含，每个不依赖 cutlass 的 W4A8 翻译单元的头文件行数会从约
3.7k 膨胀到约 44k。

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
| tile 的领取改到 GEMM 之前而不是之后，让工作计数器的 device-scope atomic 与 mainloop 重叠 | `ARK_MOE_W4A8_PREFILL_CLAIM_EARLY=0` | 尚未在 B70 上实测——上界取决于每个 tile 的停顿，因此预期在 12 个 k-tile 的 down 投影上显现、在 up 投影上接近于零 |

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
存而不是算术 — 但真正值得删掉的流量原来在调用边界上，而不在主循环里，这正是那两个
[调用契约](#削减-prefill-流量两个可选的调用契约)所做的事。

这两个契约以及 `ARK_MOE_W4A8_PREFILL_PREFETCH` 这个扫描点，是当前的
`NEEDS-HARDWARE-VALIDATION` 项：它们经过了推导、通过了 lint，但既没有编译过也没有计
过时，因为撰写环境既没有 XPU 也没有 SYCL 编译器。需要在设备上按顺序验证的是：

1. `test_prequantized_activations_match_internal` 及其 decode 版本 — int8 输入路径与
   kernel 自带量化器的对比。两个用例都构造在 int8 网格上 (`a = q × 2^-e`、
   `|q| ≤ 127`)：行 absmax 恰好是 `127 × 2^-e`，`127 / absmax` 与 `absmax / 127` 都是
   2 的幂，因此没有任何乘积靠近舍入分界，两条路径必然产生完全相同的 int8。剩下的只有
   设备自身那次求行 scale 的除法，最多差一个 ulp — 而它是整行的乘性因子，所以无论输出
   落在哪里，都最多移动一个输出格式的步长。因此一旦失败，就说明契约确有问题 — scale
   转置、行错位、把 scale 当成其倒数 — 因为这些都不止一个步长。断言会打印最大 ULP
   距离、有多少元素发生了变化以及 SNR，可以区分末位差异与结构性错误。
2. `test_fused_reduce_matches_unfused` — 与无融合路径的 SNR/余弦对比。预期约 54 dB
   (无融合那一侧的 bf16 舍入误差主导，超过 fp32 atomic 的重结合误差)，门槛是
   20 dB / 0.99。
3. `test_perf_prefill_contract_sweep` 与 `test_perf_prefill_contracts_long_seq` — 实测
   加速比是否跟得上流量模型。如果 int8 输入路径的收益明显**超过**模型预测，说明原来
   fp16 的 A 读没有命中 L2，主循环侧还有 blocking 可做；明显低于预测，则说明量化那一
   遍原本与 GEMM 的重叠程度好于按字节数的估计。
4. `test_perf_prefill_prefetch_sweep` — 纯 kernel 侧的 A/B，若有收益，在短主循环的形状
   上约为 3–8%。

### prefill 还剩下多少空间

在受算力约束的 batch 上，四个形状为 93.8 / 66.8 / 101.0 / 104.8 TFLOPS，即各自带宽天花
板的 61–76%，因此剩余空间分成两部分：这次调用仍在搬的流量，以及路由所决定的天花板。

| 方向 | 会改变什么 | 体现在哪里 |
|---|---|---|
| 调用方直接提供 int8 激活 ([契约 1](#契约-1--调用方直接提供-int8-激活)) | 消掉 5 条数据流中的 3 条 — `K = 768` 时占 27%、`K = 2048` 时 37%、`K = 3072` 时 44% | 所有形状；这是最大的一项，也是唯一能单独把 `qwen3 up` 送过 100 的一项 |
| 融合 top-k 规约 ([契约 2](#契约-2--把-top-k-规约折进-epilogue)) | 把 `T × N` 的 fp16 写变成 `batch × N` 的 fp32 读改写，并删掉独立的规约 kernel | 仅第二个投影 — 而且在 B70 上**实测更慢**，因为那次读改写是每个元素一次 device 作用域原子操作，还放弃了 2D block store |
| 让每个专家分到更多行 | kernel 里什么都不用改 — 它*抬高*的是天花板，因为只有权重这一条流不随 token 数增长 | 有效但有上限：每专家行数趋于无穷时 `qwen3 down` 的天花板收敛到 105 TFLOPS，因此单靠这一项在该形状上永远到不了目标 |
| 预取深度、调度器 tile 顺序、非临时 (non-temporal) 的 D store | 纯主循环/epilogue 侧的工作，对手是四个形状当前 239–296 GB/s 的实际带宽 | `qwen3 down` 是四者中最低的 (239 GB/s)：它的 D 是**写**，而 12 个 k-tile 是最短的主循环，因此 prologue/epilogue 摊得最差 |
| `K = 3072` 的单遍激活量化 | 省掉对 `[T, K]` 的第二次读，在受算力约束的 batch 下约 450 MB | 仅 minimax up；它的一行是每 lane 96 个 dword，超过了 16 向量那一档 — 而且在契约 1 之下已无意义，因为那一遍整个被删掉了 |

有两个方向是靠分析而不是靠实测收敛的：

**256 行的 tile** 能把每个 M tile 重复读 B 的次数减半，但 8K 提示词的扫描已经在唯一一种
它不比 128 行 tile 多 padding 的路由上测过它，结果最好也只是持平 (参见
[Prefill tile](#prefill-tile))，因此阶梯不再选用它。

**为 up/gate 投影去重 A。** 它的 `[T, K]` 输入把每个 token 重复了 `top_k = 8` 次，所以
改成接收 `[batch, K]` 加 `sorted_token_ids` 再按索引取行，可以把激活相关的数据流减少
8× — vLLM 的 `fused_moe` 就是这么做的。但它搬不过来：A tile 是通过 Xe 的 **2D block
描述符**加载的，那个描述符描述的是一块规则表面 (基址、pitch、高度) 上的矩形，无法做
gather，因此 gather 版的 A 会退化成 tile 每行一条单行 block load。只在**量化器**里做
gather 同样省不到：一个 token 的 8 份拷贝散布在整个按专家排序的区间里，命不中 L2。契约
1 在不动加载路径的前提下，为同样的形状删掉了同样的字节，所以最后实现的是它。

**把激活量化融合进 GEMM 的 A-tile 加载** — 本文档早先版本把它列为仍未做的最大一项收益、
占流量的 14–22% — 在算术上同样站不住，那个判断是错的。它只有在量化后的 A panel 能常驻
SLM **且**每行的 absmax 仍然可得时才划算：

* 每行一个 absmax 意味着必须先看完整行才能量化其中任何一个元素，所以在 A-tile 加载时做
  量化的主循环只能按 **fp16** (每元素 2 字节) 读 A — 这恰好抵消掉省下来的 int8 写 + 读
  回 (1 + 1 字节/元素)。净收益为零。
* 若改为让 int8 panel 常驻，`128 × 768` 的 int8 panel 是 96 KB，塞不进一个 work-group 的
  SLM 预算；即使 `TileM = 64` 也要 48 KB，超出了该 tile 配置允许的范围。
* 改用按 k-block 的激活 scale 可以绕开 absmax，但需要把 fp32 shadow accumulator 加回来 —
  那正是当初为了让 `128x256` tile 放得下而删掉的寄存器开销。

真正可删的那条流是**生产者写出的**那条，而不是 GEMM 读进来的那条，也就是契约 1。

`qwen3 down` (`N = 2048, K = 768`) 仍是那个异常值，只有约 64–70 TFLOPS：每个 tile 只有
12 个 k-tile，是四者中最短的主循环，其输出与权重一样大，而它在任何路由下的天花板也都是
四者中最低的。它同时也是这两个契约收益最大的形状，也是唯一两个契约都需要的形状。
