# MoE Prefill 性能测试

## 概览

`test_moe_prefill_perf.py` 文件提供了 MoE (混合专家) prefill 操作的全面性能基准测试,并附带 TFLOPS (每秒万亿次浮点运算) 计算。

## 什么是 MoE Prefill?

**Prefill** 是 LLM 推理过程中同时处理许多 token (例如整个 prompt 或一批序列) 的阶段。在 MoE 模型中,token 会被路由到不同的专家,每个专家可能接收多个 token。这与 **decode** (token 生成) 不同,后者通常每次每个专家只处理一个 token。

## 特性

### 1. **全面的数据类型支持**
- FP16 (float16)
- BF16 (bfloat16)
- INT8 (对称与非对称量化)
- INT4 (对称与非对称量化)
- INT2 (对称与非对称量化)
- FP8 (float8_e4m3fn 与 float8_e5m2)

### 2. **TFLOPS 计算**
测试对每种配置按下式计算 TFLOPS:
```
FLOPs = total_tokens × K × N × 2
TFLOPS = FLOPs / (time_in_seconds) / 1e12
```

其中:
- `total_tokens`: 所有专家的 token 总数
- `K`: 输入特征维度
- `N`: 输出特征维度
- `×2`: 每次乘加操作计为 2 FLOPs

### 3. **多种 MoE 配置**
测试覆盖多种真实的 MoE 场景:
- **小模型** (8 专家,Mixtral 风格): 4096×4096, 4096×14336, 14336×4096
- **中等模型** (8 专家): 各种 token 分布
- **大模型** (16, 32, 64 专家,DeepSeek 风格): 2048×2048
- **不均匀分布**: 模拟真实路由模式

### 4. **基线对比**
每项测试将 ARK MoE kernel 与 PyTorch 基线实现进行对比:
- **Baseline**: 单个 `torch.bmm`,输入为 `[E, M_max, K]` padding 后的激活缓冲区 (每个专家的 token 切片 padding 到全局最大 tokens-per-expert)。对量化测试,权重会被预先反量化,因此 `baseline(ms)` 列只测量 matmul 开销。
- **ARK Kernel**: 带融合操作的优化 `ark.moe_gemm`。
- **Speedup**: 报告 `baseline / ark` — 融合 kernel 相对 matmul-only 基线的加速比。

## 如何运行

### 运行全部测试:
```bash
cd /path/to/auto_round_extension/ark/test
pytest -v -s test_moe_prefill_perf.py
```

### 运行特定数据类型:
```bash
# 仅 FP16 测试
pytest -v -s test_moe_prefill_perf.py::TestMoEGemmPrefillPerf::test_perf_fp
```

## 代码结构

```
test_moe_prefill_perf.py
├── 计时工具 (_xpu_time_ms)
│   └── 使用 XPU 事件获取精确 GPU 计时
├── FLOPS 计算 (_compute_moe_flops)
│   └── 计算理论 FLOPs 用于 TFLOPS 指标
├── 基线实现 (_default_moe_prefill, _build_bmm_pad_layout)
│   └── 单个 `torch.bmm`,输入 [E, M_max, K] padding 后的激活
├── 测试形状 (PREFILL_SHAPES)
│   └── 多种真实 MoE 配置
└── 测试用例 (TestMoEGemmPrefillPerf)
    ├── test_perf_fp (FP16/BF16)
    ├── test_perf_int4 (INT4 sym/asym)
    ├── test_perf_int8 (INT8 sym/asym)
    ├── test_perf_int2 (INT2 sym/asym)
    └── test_perf_fp8 (FP8 e4m3fn/e5m2)
```

## 关键指标

1. **TFLOPS**: 越大越好 — 表示计算吞吐
2. **Speedup**: 越大越好 — 表示相对基线的性能提升
3. **Latency (ms)**: 越小越好 — 实际 kernel 执行时间

## 相关文件

- `test_moe.py`: MoE GEMM 的正确性测试
- `test_moe_decode_perf.py`: MoE decode (每个专家单 token) 性能测试
- `test_bench_bmg.py`: SDPA 性能基准测试及 TFLOPS

## FP8 Prefill 路径 (env 开关)

FP8 prefill 基准 (`test_perf_fp8`) 在 `ark(ms)` 列测量默认 ARK 路径,在
同一批形状上,`native(ms)` / `native TFLOPS` 列测量融合 **原生 FP8** 路径,
`dpas(ms)` / `dpas TFLOPS` 列测量混合输入 **DPAS FP8** 路径。四种底层
kernel 由三个独立的环境变量选择 — 首次调用时读取并缓存 — 优先级如下:

| 优先级       | 环境变量                                                              | Kernel                                                                                                                              |
| ------------ | -------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 1 (最高)     | `ARK_MOE_PREFILL_DPAS_FP8` 未设置或为真值 (**默认开启**)              | **混合输入 DPAS FP8 grouped GEMM (Variant B)**。移植自 `vllm-project/vllm-xpu-kernels` 的 `xe_gemm_4bits` — FP8 字节通过 CuTe `reorder` 在寄存器中上转到 `act_dtype`,然后使用内联 `apply_scale` (IGA asm) 应用 per-K-group scale。XMX 受限;预期为标量 native 路径的 ~2-2.5×。与 auto-round 校准输出使用相同的 `[E, N, K/group_size]` scale 布局 — 无需重新量化。实现于 `sycl_tla_moe_prefill_fp8_dpas.hpp`。**状态:NEEDS-HARDWARE-VALIDATION** (未经测试的移植)。 |
| 2            | `ARK_MOE_PREFILL_NATIVE_FP8=1`                                       | 标量原生 FP8 融合 GEMM。无 `[E, K, N]` bf16/fp16 工作区。FP8 字节在 GEMM kernel 内的寄存器中上转到 `act_dtype`,per-K-group scale 融合到累加器中。仅写回最终输出行。实现于 `sycl_tla_moe_prefill_fp8_native.hpp`。 |
| 3            | `ARK_MOE_PREFILL_FUSED_FP8=1`                                        | SLM 转置反量化 kernel (`sycl_tla_moe_prefill_fused.hpp`),后接标准 bf16/fp16 grouped GEMM。仍向 DRAM 写入 `[E, K, N]` 工作区。仅 FP8-E4M3。                                                    |
| 4 (默认)     | 以上都未设置                                                          | v1 反量化 kernel (`sycl_tla_moe_mixed.hpp::launch_dequant_fp8`),后接标准 bf16/fp16 grouped GEMM。FP8-E4M3 与 FP8-E5M2 均支持。                                                                 |

**DPAS 路径形状前置条件** — 任何条件不满足时,`moe_gemm_prefill`
dispatcher 会静默回退到优先级 2/3/4:

- `N % 64 == 0` (BN)
- `K % 32 == 0` (BK)
- `K % group_size == 0`
- `group_size ∈ {32, 64, 128, 256}`
- `asym == False` (FP8 量化仅支持对称)

**Native 路径形状前置条件** — 相同的回退语义:

- `N % 16 == 0` (BN = SG_SIZE = 16)
- `K % 32 == 0` (BK)
- `K % group_size == 0`
- `group_size % 32 == 0` (保证每 tile 沿 K 的 scale 恒定)
- `asym == False` (FP8 量化仅支持对称)

Native 与 DPAS 都支持 **E4M3** 与 **E5M2**,同时支持 **F16** 与 **BF16**
激活,覆盖与默认列相同的 `PREFILL_SHAPES` 矩阵。

### Variant A — per-tensor FP8 DPAS (独立入口)

此移植还通过独立的 Python 入口暴露 **Variant A** per-tensor FP8 DPAS
grouped GEMM:

```python
outputs = ark.moe_gemm_prefill(
    activations,  # [total_tokens, K], f16/bf16
    weights,  # [E, K, N] 行主 FP8 (vllm 布局)
    num_tokens_per_expert,  # [E] int32
    scales=scales,  # [E] fp32,每专家一个 per-tensor scale
    scale_scheme="per_tensor",
)
```

这逐字节镜像了 vllm-xpu-kernels 的 `cutlass_grouped_gemm_xe2_impl` FP8
分支。它需要 **重新量化的 checkpoint** (每专家一个 FP32 标量,权重
转置到 `[E, K, N]`),因此更适合作为对延迟敏感的 decode 路径的可选项,
而不是现有 auto-round FP8 checkpoint 的即插即用 — 后者优先使用 Variant B。

**状态:NEEDS-HARDWARE-VALIDATION** (未经测试的移植)。

在测试运行时通过 env 启用:

```bash
# 默认 (DPAS Variant B) — auto-round 原生校准方案。
pytest -v -s test_moe_prefill_perf.py::TestMoEGemmPrefillPerf::test_perf_fp8

# 仅强制标量 native 路径 (关闭 DPAS)。
ARK_MOE_PREFILL_DPAS_FP8=0 ARK_MOE_PREFILL_NATIVE_FP8=1 pytest -v -s test_moe_prefill_perf.py::TestMoEGemmPrefillPerf::test_perf_fp8

# 强制融合反量化路径。
ARK_MOE_PREFILL_DPAS_FP8=0 ARK_MOE_PREFILL_FUSED_FP8=1 pytest -v -s test_moe_prefill_perf.py::TestMoEGemmPrefillPerf::test_perf_fp8

# 性能测试内部逐行切换 env,因此 `ark(ms)`、`native(ms)`、`dpas(ms)`
# 各列均测量特定路径,与外部 env 设置无关。
```

关于精度对齐,`test_moe_prefill_accuracy.py::test_accuracy_fp8` 覆盖
dequant/native 路径,`test_accuracy_fp8_dpas_per_group` /
`test_accuracy_fp8_per_tensor_dpas` 覆盖 DPAS Variant B / A,均在相同
生产形状下测试;所有路径共享容差 `rtol=atol=1e-1` (E4M3) / `1e-1`
(E5M2)。

## INT8 Prefill 路径(可选 env 开关)

INT8 sym prefill 基准(`test_perf_int8`,`asym=False`)也带一列混合输入
**DPAS INT8**(`dpas(ms)` / `dpas TFLOPS`),对应 FP8 per-K-group Variant B
路径。`ark(ms)` 列强制 `ARK_MOE_PREFILL_DPAS_INT8=0`,测量传统的
dequant + GEMM 路径;`dpas(ms)` 列重新启用该开关,在同一批形状上测量新的
混合输入路径。

| 优先级       | Env 开关                                                     | Kernel                                                                                                                                                                                                                              |
| ------------ | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 (最高)     | `ARK_MOE_PREFILL_DPAS_INT8` 未设置或为真值(**默认开启**)   | **混合输入 DPAS INT8 grouped GEMM (Variant B)**。INT8 字节通过 CuTe `reorder` 在寄存器中上转到 `act_dtype`,然后通过组边界延迟折叠应用 per-K-group scale(与 FP8 per-group 路径完全一致)。scale 使用与 auto-round INT8 校准输出相同的 `[E, N, K/group_size]` 布局 —— 无需重新量化。同时支持 sym 与 asym:asym 通过一次 per-M 行 per-K-group 的激活行和预计算,把折叠改写为 `Σ_g s · (Σ w·a − z · Σ a)`。实现于 `sycl_tla_moe_prefill_int_dpas.hpp`。**状态:NEEDS-HARDWARE-VALIDATION**(未经测试的移植)。 |
| 2 (默认回退) | `ARK_MOE_PREFILL_DPAS_INT8=0`                                | v1 dequant kernel(`sycl_tla_moe_mixed.hpp::launch_dequant_int8`)后接标准 bf16/fp16 grouped GEMM。同时支持 sym 与 asym。                                                                                                              |

**DPAS 路径形状前置条件** — 任何条件不满足时,`moe_gemm_prefill`
分发器会静默回退到优先级 2(与 FP8 per-group 谓词一致):

- `N % 64 == 0` (BN)
- `K % 32 == 0` (BK)
- `K % group_size == 0`
- `group_size ∈ {32, 64, 128, 256}`
- `asym`:sym 与 asym 均支持(asym 会额外做一次 `Σ a` 预计算)

精度对齐由
`test_moe_prefill_accuracy.py::test_accuracy_int8_dpas_per_group`
在与 `test_accuracy_int8` 相同的生产形状下覆盖,使用标准 INT8 容差
(`rtol=atol=1e-1`)。

## INT4-sym Prefill 路径(opt-in env 开关)

INT4 sym prefill 性能测试(`test_perf_int4`,`asym=False`)带一个
混合输入 **DPAS S4** 列(`dpas(ms)` / `dpas TFLOPS`)。`test_perf_int4`
在 `ark(ms)` 列强制 `ARK_MOE_PREFILL_DPAS_S4=0` 与
`ARK_MOE_PREFILL_DPAS_INT8=0`(传统 dequant + GEMM 路径),在
`dpas(ms)` 列启用 `ARK_MOE_PREFILL_DPAS_S4=1`(单遍 packed-nibble
mainloop)。

S4-sym 有两条独立的 DPAS 路径;asym S4 始终回退到 dequant 路径。

| 优先级       | Env 开关                                                                        | Kernel                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------ | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1 (最高)     | `ARK_MOE_PREFILL_DPAS_S4` 未设置或为真值(**默认开启**)                        | **S4-sym 单遍 DPAS 混合输入 mainloop**。直接读取 packed `[E, N, K/2]` `uint8_t` nibble,通过 CuTe `reorder(tBrB, tCrB)`(依赖 `NumericArrayConverter<ElementA, cutlass::int4b_t, N>`)在寄存器中把 S4 上转到 `act_dtype`。B 侧 global 带宽正好是 S8 路径的一半。Per-K-group scale 使用与 INT8 相同的组边界延迟折叠。实现于 `sycl_tla_moe_prefill_s4_dpas.hpp`。**状态:NEEDS-HARDWARE-VALIDATION**(未经测试的移植)。 |
| 2 (回退)     | `ARK_MOE_PREFILL_DPAS_S4=0` 且 `ARK_MOE_PREFILL_DPAS_INT8` 为真值(**默认开启**) | **S4→S8 上转 + 共享 INT8 DPAS mainloop**。两遍:`launch_upcast_int4_sym_to_int8` 把权重写成 `[E, N, K]` `int8_t`(复用 dequant workspace),再由标准 INT8 per-group DPAS mainloop 消费。相较路径 1 需要付出 ~E·N·K 字节的 workspace 往返。实现于 `sycl_tla_moe_mixed.hpp` + `sycl_tla_moe_prefill_int_dpas.hpp`。 |
| 3 (默认回退) | `ARK_MOE_PREFILL_DPAS_S4=0` 且 `ARK_MOE_PREFILL_DPAS_INT8=0`                    | v1 dequant kernel(`sycl_tla_moe_mixed.hpp::launch_dequant_int4`)后接标准 bf16/fp16 grouped GEMM。同时支持 sym 与 asym。                                                                                                                                                                                                                                                                                                                                              |

**S4 DPAS 路径形状前置条件** — 任何条件不满足时,`moe_gemm_prefill`
分发器会静默回退到优先级 2(再回退到 3):

- `N % 64 == 0` (BN)
- `K % 32 == 0` (BK)
- `K % group_size == 0`
- `group_size % 2 == 0`(nibble 对不会跨越组边界)
- `group_size ∈ {32, 64, 128, 256}`
- `asym == false`(asym S4 不在两条 DPAS 路径的支持范围内)

**S4 DPAS tile 策略** — 单遍 mainloop(优先级 1)现在按每专家平均
token 数(`A_avg_M = total_tokens / E`)选择专用的 4-bit tile 策略,
与参考实现 `vllm-project/vllm-xpu-kernels`
(`grouped_gemm_xe2_interface.hpp`)的 `w4a16` 分派一致。由于 packed-
nibble 的 B 流字节量是 INT8 路径的一半,大 M tile 加宽到 `128×256×32`
(相比 INT8 的 `128×128×16`),以更充分利用 DPAS 累加器与减半的 B 侧
带宽:

| `A_avg_M` 分档 | WG tile (M×N×K) | 策略(`sycl_tla_moe_prefill_fp8_dpas.hpp`) |
| -------------- | --------------- | ------------------------------------------ |
| `≤ 4`          | `8×64×32`       | `dpas_w4a16_policy_m_8`                     |
| `≤ 8`          | `16×64×32`      | `dpas_w4a16_policy_m_16`(= `w8a16_m_16`)   |
| `≤ 128`        | `32×64×32`      | `dpas_w4a16_policy_m_32`(= `w8a16_m_32`)   |
| `> 128`        | `128×256×32`    | `dpas_w4a16_policy`                         |

中等大小的 `32×64` tile 现在覆盖 `A_avg_M` 至 128(此前在 33 就跳到大
tile),避免了常见 chunked-prefill batch 大小下的 padding 浪费。

**S4 DPAS decode 路径** — decode(生成)阶段(`sycl_tla_moe_decode.hpp`,
int4-sym / `S4_CLIP`,`!asym`,`ARK_MOE_DECODE_DPAS_S4` 默认开启)拥有
独立的 dispatch `moe_decode_s4_dpas_per_group_dispatch`,对齐
vLLM-xpu-kernels 的 `w4a16` decode dispatch。它与 prefill 使用相同的
`A_avg_M` 阶梯选择 DPAS tile(`_m_8` → `_m_16` → `_m_32` → 大 tile):
仅在极小 batch 尾部(`A_avg_M ≤ 4`)使用 8 行 tile,一旦平均每个专家
路由超过 4 个 token,M tile 就随之增大。早先的版本直接钉死 8 行的
`dpas_w4a16_policy_m_8` tile,假设 decode 阶段每个专家只见到少量 token,
但在较大的 decode batch(序列多、top-k 高或专家少)下,这会导致 M
维度欠填充,并把(受带宽约束的)打包权重重复流式加载 2–4 次,使吞吐
大约只有参考实现的一半。它复用共享的 per-group mainloop 的 2D VNNI 块
加载(`get_block_2d_copy_A/B` + `make_block_2d_prefetch`)与寄存器驻留
的 per-N scale(`sg_scale[]`,每个 K-group 折叠一次),读取相同的
`[E, N, K/2]` 打包权重 + `[E, N, K/group]` scale,无需重新打包。
`ARK_MOE_DECODE_S4_DPAS_M8=1` 会强制使用旧的钉死 8 行 tile 以便 A/B
对比(数值完全相同,仅 tile 形状不同)。**状态:NEEDS-HARDWARE-VALIDATION**
(未经测试的移植)。

**占用率门控 — decode 规模的 batch 直接复用 int4-asym 的实现。** 即使是
最小的 DPAS tile 也要处理每个专家 8 行 token,因此平均每个专家不足 8 个
token 的 batch 会为几乎全是 padding 的行付出完整的权重流式加载代价。
decode 正好处于这一区间:MiniMax-M2(192 个专家)bs1 只有 8 个 token,
bs32 只有 256 个 token,即平均每个专家 0.04–1.3 个 token;实测同样形状下
int4-sym(DPAS)为 0.31–0.34 ms / 1.55 ms,而 int4-asym(标量 GEMV)为
0.12 ms / 1.45 ms。因此除非 batch 平均每个专家至少有 8 个 token,int4-sym
的 decode 会被路由到与 int4-asym *完全相同* 的标量 GEMV kernel
(`launch_int4` 及其 coalesced 变体,`Asym=false`)。`ARK_MOE_DECODE_DPAS_S4_MIN_TPE` 可覆盖该
"每专家 token 数" 阈值;设为 `0` 则关闭门控(只要形状门控通过就走 DPAS),
精度测试与 DPAS/标量 对比性能测试即使用该设置。

**基于 32 位字的 nibble 解码；sym 恢复真正的有符号 nibble。** 在两者都走
标量 GEMV 之后，int4-sym 在 *同一个* kernel 里仍比 int4-asym 慢，尽管 sym
的浮点运算严格更少。差异在于 nibble 解码，而第一次尝试的修复（符号翻转
恒等式 `signed == (unsigned ^ 8) - 8`，即 `^ 0x88`）并没有弥合差距：它让
sym 仍然停留在 8 位类型的运算上 —— 一次 `sycl::vec<uint8_t,N>` 的 XOR 加上
逐字节的 掩码/移位 —— 而 Xe 会把这类窄类型运算展开，无法直接跑在原生
32 位数据通路上；同时它还迫使 sym 携带一个恒为 8 的 zero-point（见下面的
"激活求和"）。

现在两种模式都通过共享的 `decode_int4_octet` 原语解码：它接收一个打包的
*32 位字* 中的 8 个 nibble，每个 nibble 只用一对 DWORD 移位/掩码（asym）
或一对 DWORD 左移 + 算术右移（sym）即可取出。没有 8 位类型的向量，没有
XOR，没有窄化转换，并且每 8 个 K 元素只需一次 32 位加载而不是一次字节
向量加载。在两种模式下，对全部 2^32 种输入字，逐 nibble 的结果都与
`decode_int4_pair` 逐位相同（已穷举验证），因此这纯粹是指令选择层面的改动。
它应用于 `launch_int4`、`launch_int4_coalesced`，并且由于该原语是共享的，
prefill 的混合精度路径同样受益。

由于 sym 重新恢复了 *真正的有符号* nibble，它的每组折叠退化为
`acc += scale * Σ a·q`，完全没有 zero-point 项；asym 则仍是
`acc += scale * (Σ a·q − zero · Σ a)`。

**按 4 字节分块的 coalesced repack。** coalesced 回退路径
(`launch_int4_coalesced`，`ARK_MOE_DECODE_COALESCE_INT4` 默认开启)会在设备端把
`[E, N, K/2]` 权重重排，使 sub-group 的加载连续。原先的重排布局
`[E, N/16, K/2, 16]` 每个 lane 每步只放一个字节，因此虽然 16 个 lane 合起来覆盖
一条 cache line，每个 lane 仍然发出的是*字节*加载。现在布局改为
`[E, N/16, ceil(K/8), 16, 4]`：一个 chunk 为
tile 内 16 列中的每一列存放 4 个连续的打包字节，按 lane 主序排列，因此 lane `l`
在 chunk 偏移 `l*4` 处读取自己的 4 个字节，sub-group 整体仍然覆盖 64 个连续字节。
一个 lane 的这 4 个字节是连续的，因此恰好构成一个小端 32 位字：lane 只需发出
一次 DWORD 加载(权重加载指令数降为 1/4)，并直接交给 `decode_int4_octet`，
两种模式下 8 个 nibble 都用原生 32 位运算取出。group_size 为 8 的倍数时
(16/32/64/128/256，即全部已发布的量化配置)每个 K 组都从 chunk 边界开始，向量
阶段覆盖整个组；其他偶数 group_size 则通过标量前导/收尾循环在同一布局上处理。
对外的 `[E, N, K/2]` 权重约定保持不变。

**提取激活求和(仅 asym)。** asym 的 int4 GEMV 按
`scale * (Σ a·q − zero · Σ a)` 折叠每组的 scale/zero。`Σ a` 只依赖激活行与 K 组，
与输出列无关，但此前它是在内层循环里重复计算的 —— 每个 sub-group lane 算一遍
(16 倍冗余)，每个 N-tile work-group 再算一遍 —— 每个 K 元素多付出一次浮点加法。
现在它被预先计算成一张 `[total_tokens, K/group_size]` 的 fp32 表
(`launch_act_group_sums`)，GEMV 内层循环只累加 `Σ a·q`，每组读取一个 float。
求和顺序的变化仅带来几个 fp32 ULP 的差异，远在 kernel 现有的量化容差之内。

**sym 完全跳过这一前置 pass。** `launch_act_group_sums` 是一个独立的
`parallel_for`，在 in-order queue 上会完全串行地排在 GEMV 之前。对 decode
规模而言这笔交易并不划算：它在一个本就受访存带宽限制的循环里省下每个 K
元素一次浮点加法，却给一次 GEMV 仅几十微秒(bs1)的调用额外增加了一整次
kernel 派发 —— 这正是让 sym 走"有偏置无符号解码"反而变*慢*的原因。现在
sym 解码出真正的有符号 nibble，不含 zero-point 项，因此只有在 `Asym` 为真
时才会计算该表(并派发该 kernel)。

**用 scratch 池替代每次调用的 `malloc_device`。** repack 缓冲区原本是临时的 USM
分配，每次 decode 调用都必须在一次阻塞的 `queue::wait()` 之后释放 —— 而 decode
每生成一个 token 就调用一次，因此这次分配加同步的开销已经与 GEMV 本身同量级。
现在 repack 缓冲区与激活求和表都取自按 queue 持有、按需增长的常驻 slab
(`DeviceScratchPool`)，稳态 decode 不再有任何分配，也不引入主机侧同步；生产者
kernel 与 GEMV 之间的顺序由 in-order queue 保证。
`ark.moe_decode_release_scratch()`(pybind `moe_decode_release_scratch`)可将内存
归还。

repack *kernel* 默认仍每次调用都执行。设置
`ARK_MOE_DECODE_INT4_REPACK_CACHE=1` 可在权重缓冲区地址与形状不变时复用上一次的
repack 结果 —— 这对权重固定的真实推理循环是成立的。它**默认关闭**，因为其 tag
是指针身份：被释放后重新分配的权重张量可能落在同一地址(torch 的缓存分配器在
测试循环中很容易出现这种情况)，此时陈旧的 repack 会静默产生错误结果。启用它的
调用方必须在丢弃权重张量之前调用 `ark.moe_decode_release_scratch()`。

| 环境变量 | 默认值 | 作用 |
| -------- | ------ | ---- |
| `ARK_MOE_DECODE_COALESCE_INT4` | 开启 | int4 标量回退使用按 4 字节分块的 coalesced repack GEMV；设为 `0` 则强制使用按 lane 跨步的旧版 `launch_int4`。 |
| `ARK_MOE_DECODE_COALESCE_MIN_TOKENS` | `num_experts * TOKEN_BLOCK` | coalesced kernel 值回其 repack 开销所需的最小总 token 数；设为 `0` 关闭该门控(一致性/A-B 测试即如此设置)。 |
| `ARK_MOE_DECODE_INT4_REPACK_CACHE` | 关闭 | 在同一权重缓冲区上跨调用复用 repack 结果。仅当调用方掌握权重生命周期时才安全。 |

coalesced 路径的性能 A/B 见
`test_moe_decode_perf.py::test_perf_int4_coalesced_vs_strided`(在相同形状上切换
`ARK_MOE_DECODE_COALESCE_INT4` 0/1)。正确性由
`test_moe.py::test_decode_int4_coalesced_matches_scalar`、
`::test_decode_int4_coalesced_token_blocking`、
`::test_decode_int4_coalesced_unaligned_group_size`(非 8 的倍数的 group_size，
覆盖标量前导/收尾路径)以及 `::test_decode_int4_repack_cache` 覆盖。

**占用率门控阈值扫描。** `ARK_MOE_DECODE_DPAS_S4_MIN_TPE` 的默认值 8 来自
`dpas_w4a16_policy_m_8` 的 tile 行数，而非实测结果。定位真实交叉点的扫描用例是
`test_moe_decode_perf.py::test_perf_int4_sym_dpas_vs_scalar_threshold`；它默认的
token 数(16–128)都远低于该门控(8 × 192 个专家 == 1536 个 token)，因此需要传入
`--all-shapes` 把扫描扩展到 256/512/1024/1536/3072 个 token，从两侧夹住门控。
在拿到硬件数据之前，默认值仍保持为 8。

精度对齐由
`test_moe_prefill_accuracy.py::test_accuracy_int4_dpas_per_group`
覆盖,该用例强制 `ARK_MOE_PREFILL_DPAS_S4=1` +
`ARK_MOE_PREFILL_DPAS_INT8=0`,专门验证单遍 mainloop 路径,形状矩阵与
`test_accuracy_int4` 一致,容差 `rtol=atol=1e-1`。

## FP8 Decode 路径 (`sycl_tla_moe_decode.hpp`)

int4-sym decode 的性能已经达标,把它推到达标的两个手段同样适用于 FP8:
让 dequant 离开按字节的数据通路,以及不要在每次 decode 调用里重复付出
启动开销。在此之上,还把 vllm-xpu-kernels 的 FP8 MoE dispatch 镜像成一个
decode 专用入口。

**Word-native FP8 解码 (`ARK_FP8_DECODE_MODE`, 默认 `word`)。** decode
GEMV 每读一个权重字节大约只做一次乘加,所以 dequant *就是* kernel 本身。
两条旧解码路径每个字节都要付出真实开销:`lut` 每个权重元素都要向 128 项
幅值表发一次访存再做一次符号选择,`bits` 则要跑一串带分支的 `ldexp`。
两者还都索引了 8-bit 类型的 `sycl::vec<uint8_t, 16>`,而 Xe 的 ALU 通道是
32-bit 的、无法直接寻址它,于是 IGC 只能展开成窄类型 regioning ——
正是 `decode_int4_octet` 为 nibble 解决过的那个问题。

这些工作其实都不必要:FP8 字节本身就是一个 IEEE 风格的浮点数,而 fp16 是
两种 FP8 格式的*超集*,整个转换就是一次位域搬移。

| 格式 | fp16 位模式 | 精确性 |
| ---- | ----------- | ------ |
| E5M2 | `byte << 8` | 对全部 256 种编码逐位精确 —— 符号位位置相同、5 位指数相同、bias 同为 15。次正规数仍是次正规数,`exp==31` 仍是 Inf/NaN。 |
| E4M3 | `(byte + (byte & 0x80)) << 7` | 对全部 254 种有限编码(正规数、次正规数、两个零)逐位精确,得到真值 × `2^-8`。 |

E4M3 的 4 位指数 bias 为 7,而 fp16 的 bias 是 15,所以位域搬移会留下一个
常数因子 `2^-8`;`fp8_word_scale_bias<IsE4M3>()`(`256.0f`)被折叠进
per-K-group 的 scale,是一个精确的 2 的幂、每组只乘一次,因此对单个元素而言
零开销。把符号位加到它自身上,恰好会把它再进位一格,这就是符号搬移与幅值
搬移能合并成一次加法加一次移位的原因。

kernel 以 `sycl::vec<uint32_t, 4>` 读取权重 —— 与它替换掉的字节向量是同一次
16 字节访存、同样的 16 字节对齐要求 —— 再由 `decode_fp8_quad_half_bits` 用
少量原生 DWORD 运算把每个 32 位字变成四个 fp16 位模式(SWAR,不会跨 lane
进位)。两个部分累加器打断 fp32 依赖链,与 `int4_decode_chunk` 的做法一致。
两个原语都放在 `sycl_tla_moe_dequant.hpp`,并已对两种格式的全部 256 个字节
值做过穷举验证。

**E4M3 NaN 注意事项。** E4M3 的两个 NaN 编码(`0x7F` / `0xFF`;
`torch.float8_e4m3fn` 没有 Inf)会解码成 ±480 而不是 NaN,因为纯位域搬移
到不了 fp16 的任何 NaN 模式。auto-round 的 FP8 checkpoint 是按
`finfo(float8_e4m3fn).max == 448` 缩放并 clamp 得到的,所以这两个编码不可能
出现。需要 NaN 传播的调用方可以选择 `ARK_FP8_DECODE_MODE=lut` 或 `=bits`。

**K-split lane 映射(`ARK_MOE_DECODE_FP8_KSPLIT`,默认 ON)。** 当 dequant
只剩几条 DWORD 运算之后,scalar GEMV 就是一个纯粹的带宽问题:每个权重字节
大约只做一次乘加,所以它最快只能跑到专家 tile 的搬运速度。原来的映射把一个
输出元素交给一个 *work-item*,于是一个 lane 要独自走完整条 `[n, K]` 权重行。
由此带来两笔开销:

* **权重访存不合并。** 同一 sub-group 中 lane `l` 与 lane `l+1` 读到的字节
  相距 `K`,因此每条 16 字节的 load 指令都会被拆成 16 个 cache line 请求。
  DRAM 字节并没有浪费(每个 lane 会沿着自己的行把这些 line 用完),但内存
  控制器看到的是每个 sub-group 16 条互相独立的数据流 —— 这正是 DRAM row
  buffer 最不擅长的访问模式。
* **线程太少。** grid 只有 `total_tokens × N / 16` 个 sub-group ——
  MiniMax-M2 batch-1 一步(8 个 token,N=1536)只有 768 个 SIMD16 线程,
  低于 BMG 级 GPU 的线程槽数量,飞行中的 load 永远不足以掩盖 DRAM 延迟。

`launch_fp8_ksplit` 把映射转置过来:一个 *sub-group* 负责一个输出元素,由它
的 16 个 lane 切分 K。lane `l` 在每个 256 元素的步长内拥有起点为 `l*16` 的
16 个连续 K 元素,于是一条指令覆盖 256 字节**连续**权重(四条完整 cache
line)和 512 字节连续激活,每个线程只走一条顺序数据流,线程数则提升 16×
(上述 batch-1 场景为 12288 个 sub-group)。代价是每个输出元素一次
`reduce_over_group` —— 相对 `K` 次乘加只是几条 shuffle —— 以及 16× 的 L1
激活流量,而在这样的计算密度下 L1 有充足余量。

int4 的回退路径解决的是同一个问题,办法是把权重 repack 成 N-tiled 布局
(`ARK_MOE_DECODE_COALESCE_INT4`),那需要额外完整扫一遍权重张量并占用
scratch 显存。FP8 权重每元素一个字节、本来就是 K 连续的,所以只切分 lane
映射就能拿到同样的合并访存,无需 repack、无需 scratch、也不多一次 kernel
启动。

该 kernel 用移位来索引 scale 数组,因此形状门控要求 `group_size` 是 ≥ 16 的
2 的幂(已发布的 FP8 配置 —— 32 / 64 / 128 / 256 —— 全部满足),另外还要
`N%16==0`、`K%group_size==0` 以及 `K ≥ 256`(保证 sub-group 的每个 lane 至少
分到一个 chunk);其余情况继续走老的 GEMV,它支持任意 group size。三种 `ARK_FP8_DECODE_MODE` 解码器在两种映射下都能运行,所以
decode mode 的 A/B 依然是同口径对比。
**状态:NEEDS-HARDWARE-VALIDATION。**

**K-split kernel 内的 N 分块(`ARK_MOE_DECODE_FP8_KSPLIT_NCOLS`,默认 2)。**
当一个 sub-group 只负责一个输出列时,热循环中每读一个 16 字节权重 chunk,
既要发一条权重访存,又要发一条 32 字节的激活访存 —— 线程请求的数据里有一半
是激活行,而该 token 的每一列都会重复读它 —— 并且飞行中的权重 load 始终只
有两条。让一个 sub-group 负责 `NCOLS` 个连续列,激活 chunk 只需读一次就能
被所有列复用:

| | `NCOLS=1` | `NCOLS=n` |
| --- | --- | --- |
| 每个权重 chunk 的激活访存条数 | 1 | 1/n |
| 飞行中的独立权重 load | 2 | 2n |

前者降低请求队列压力,后者提升 memory-level parallelism —— 对于一个远低于
DRAM 峰值带宽的流式 GEMV,后者才是真正的瓶颈。代价是活跃的权重向量与
累加器变成 `n` 倍,超过某个点 kernel 就会 spill,所以只提供 1、2、4 这个
很短的阶梯,并且默认值取得保守。

一个 work-group 仍然是 16 个 sub-group,因此它现在覆盖 `16 * NCOLS` 列;
若 `N` 无法按所请求的因子切分,host 侧会回退到最大的、合法的更小 2 的幂
(`N=1536` 与 `N=3072` 在所有因子下都能整除)。lane → K chunk 的映射、
每个 chunk 的 scale 折叠以及最后的 `reduce_over_group` 都没有改动,因此
单个输出元素的算术完全不变,`NCOLS=1` 与改动前的 kernel 完全一致。
`test_perf_fp8_ksplit_ncols_sweep` 会逐形状打印全部三个因子的耗时,便于用
实测数据确定默认值。
**状态:NEEDS-HARDWARE-VALIDATION。**

**路由表校验(`ARK_MOE_VALIDATE_ROUTING`,默认 OFF)。** Python 入口原先
在每次调用时都会检查 `sum(num_tokens_per_expert) == total_tokens`。当路由表
本身就在设备上时,这个求和意味着一次 reduction kernel 外加一次**阻塞式**的
device-to-host 拷贝,也就是一次完整的流水线 flush —— 而 decode 一步的 kernel
本身只有约 150 µs,并且每生成一个 token 就要付一次。它同样落在 decode
benchmark 的计时区间内,因为记录计时 event 时队列正好是空的。
现在这个求和关系是调用方契约(C++ 侧本来就不需要 host 上的值:它直接使用
设备指针,并在设备上推导 `expert_id_per_token`,且会 clamp 到
`num_experts - 1`);调试 router 时可设置 `ARK_MOE_VALIDATE_ROUTING=1` 恢复
即时校验。位于 host(CPU)上的路由表仍然始终校验,因为对它们求和是免费的。

**FP8 DPAS decode dispatch。** `moe_decode_fp8_dpas_per_group_dispatch`
(`sycl_tla_moe_prefill_fp8_dpas.hpp`,`ARK_MOE_DECODE_DPAS_FP8` 默认 ON)
是 S4 decode dispatch 的 FP8 对应物:同一套 mainloop、同样的 `[E, N, K]`
FP8 字节 + `[E, N, K/group]` scale、无需 repack。它与 prefill dispatch 有
两点 decode 专属的差异。

*更细的 small-M 阶梯。* vllm-xpu-kernels 的参考 `w8a16` dispatch 最小只到
16 行 tile,而它的 `w4a16` dispatch 多一个 8 行档位。decode 的 `A_avg_M`
远低于 16,缺这一档意味着每个 M tile 有一半是 padding,而受带宽约束的 FP8
权重要为这些毫无贡献的行反复搬运。`dpas_w4a16_policy_m_8` 不含任何 4-bit
专用类型 —— 它纯粹是一个 `8×64×32` 的 `WGTile` / `SGLayout` 形状 ——
所以 FP8 mainloop 可以原样复用它,补上这一档:

| `A_avg_M` 档位 | WG tile (M×N×K) | Policy |
| -------------- | --------------- | ------ |
| `≤ 4`          | `8×64×32`       | `dpas_w4a16_policy_m_8` |
| `≤ 8`          | `16×64×32`      | `dpas_w8a16_policy_m_16` |
| `≤ 128`        | `32×64×32`      | `dpas_w8a16_policy_m_32` |
| `> 128`        | `128×128×16`    | `dpas_w8a16_policy` |

上面几档对齐的是 S4 的 *decode* 阶梯,而不是 FP8 prefill 的那条 ——
后者的 `≤ 512 → m_32` 档是按 prefill 规模的 batch 调过的。

*常驻 atomic 计数器。* prefill dispatch 每次调用都用 `sycl::malloc_device`
分配 work-group 计数器、再用 `sycl::free` 释放,这两个操作各会强制一次队列
同步。在 prefill 规模下这只是噪声,但在 decode 规模下 —— GEMM 本身只有几十
微秒、且每生成一个 token 就要发一次调用 —— 它占总时间的比例相当可观。
decode dispatch 改用每队列常驻的一个 slot(`get_persistent_atomic_buffer`,
现已与 S4 头文件共享,两条路径共用一份 cache)。走上这条快路径时还会跳过
`fill_expert_id_per_token` 前置 pass,因为 DPAS dispatch 直接消费
`num_tokens_per_expert` —— decode 时间线上少一次 kernel 启动。
**状态:NEEDS-HARDWARE-VALIDATION**(该头文件是未经硬件验证的移植)。

**占用率门控 —— 真实 decode batch 仍走 scalar GEMV。** 理由与 int4-sym
相同:decode 阶梯能选到的最小 tile 每个专家处理 8 行 token,所以平均每专家
不足 8 个 token 时,tile 大部分是 padding。这正是 decode 的场景(MiniMax-M2,
192 个专家:每专家 0.04–1.3 个 token),因此除非 batch 平均每专家至少提供
8 个 token,FP8 decode 一律走 scalar GEMV。
`ARK_MOE_DECODE_DPAS_FP8_MIN_TPE` 可覆盖该阈值;`0` 关闭门控,这也是对齐
用例与 A/B 性能用例所设置的值。未通过 per-group 形状门控
(`N%64==0`、`K%32==0`、`K%group_size==0`、
`group_size ∈ {32,64,128,256}`)的形状始终回退到 scalar GEMV。

| Env 变量 | 默认值 | 作用 |
| -------- | ------ | ---- |
| `ARK_FP8_DECODE_MODE` | `word` | scalar GEMV 的 FP8 解码实现:`word`(位域搬移 + 折叠 scale bias)、`lut`(128 项幅值表)、`bits`(内联位运算)。 |
| `ARK_FP8_DECODE_USE_LUT` | 未设置 | 旧的选择开关;当它被显式设置、且 `ARK_FP8_DECODE_MODE` 未设置或取值无法识别时仍然生效:truthy → `lut`,falsy → `bits`。它同时仍然驱动 mixed-input prefill 路径。 |
| `ARK_MOE_DECODE_DPAS_FP8` | ON | 形状与占用率门控都通过时,把 FP8 decode 路由到 per-group DPAS grouped GEMM;`0` 强制走 scalar GEMV。 |
| `ARK_MOE_DECODE_DPAS_FP8_MIN_TPE` | `8` | 走 DPAS 路径所需的最小每专家 token 数;`0` 关闭门控(对齐/A-B 用例所设)。 |
| `ARK_MOE_DECODE_FP8_KSPLIT` | ON | scalar GEMV 的 lane 映射:一个 sub-group 负责一个输出元素、由 lane 切分 K(访存合并,线程数 ×16);`0` 强制走老的「一个 work-item 一个输出元素」GEMV。未通过门控(`group_size` 为 ≥ 16 的 2 的幂、`N%16==0`、`K%group_size==0`、`K ≥ 256`)的形状始终使用老映射。 |
| `ARK_MOE_DECODE_FP8_KSPLIT_NCOLS` | `2` | K-split GEMV 中一个 sub-group 负责的输出列数(1、2 或 4)。取值越大,一次激活 load 被复用的列越多、飞行中的权重 load 越多,代价是活跃寄存器更多。若 `16 * NCOLS` 无法整除 `N`,会回退到最大的、合法的更小 2 的幂。 |
| `ARK_MOE_VALIDATE_ROUTING` | OFF | 即时校验 `sum(num_tokens_per_expert) == activations.shape[0]`(针对位于设备上的路由表)。该校验每次调用都要付一次阻塞式 device-to-host 同步,因此改为按需开启;位于 CPU 上的路由表始终校验。 |

性能 A/B 行是 `test_moe_decode_perf.py::test_perf_fp8_word_vs_lut`
(`speedup` 为 `lut / word`)、`::test_perf_fp8_ksplit_vs_strided`
(`speedup` 为 `strided / ksplit`)、`::test_perf_fp8_ksplit_ncols_sweep`
(`speedup` 为 `NCOLS=1 / 最优 NCOLS`,并打印全部因子)与
`::test_perf_fp8_dpas_vs_scalar`(`speedup` 为 `scalar / dpas`)。正确性由
`test_moe.py::test_decode_fp8_modes_match`(三种解码器互相一致,且各自都
对齐 dequant 参考)、`::test_decode_fp8_ksplit_matches_strided`(两种 lane
映射一致,并覆盖非 2 的幂 `group_size` 的回退)、
`::test_decode_fp8_ksplit_ncols_match`(各分块因子结果一致,并覆盖 `N`
无法整除时的回退)与
`::test_decode_fp8_dpas_matches_scalar` 覆盖。

## FP8 per-expert (per-tensor) 性能测试

`test_perf_fp8_per_tensor` 提供 Variant A DPAS 路径的性能表格,对应
**每专家一个 FP32 scale** 的量化方案(`scales.shape == [E]`,权重
`[E, K, N]` 行主 FP8 — vllm 布局)。参数化覆盖所有 dtype 组合
(fp16/bf16 × E4M3/E5M2),形状矩阵与 `test_perf_fp8` 相同。

```bash
# Prefill: 通过 scale_scheme="per_tensor" 分发到
# moe_gemm_prefill_fp8_dpas (Variant A)。构建缺少该 pybind 符号时静默跳过。
pytest -v -s test_moe_prefill_perf.py::TestMoEGemmPrefillPerf::test_perf_fp8_per_tensor
```

`test_moe_decode_perf.py::test_perf_fp8_per_tensor` 补充 decode 侧的
相同量化方案。由于 C++ decode kernel 目前没有原生的 `[E]` per-tensor
入口(只接受 per-K-group `[E, N, K/group_size]` scales),该测试通过
把每专家标量 **广播** 到 K-group 维度来喂给现有 kernel — 语义上等价
于 per-tensor 量化 checkpoint,与 `test_perf_fp8` 走同一条代码路径,
用于验证该量化方案在现有 decode kernel 上的运行成本。

```bash
pytest -v -s test_moe_decode_perf.py::TestMoEGemmDecodePerf::test_perf_fp8_per_tensor
```

## INT8 per-expert (per-tensor) 性能测试

`test_perf_int8_per_tensor` 是 FP8 Variant A DPAS 路径的 **INT8** 对应
入口:权重以每元素 1 字节的形式存放为 `[E, K, N]` 行主序
`torch.int8`,scale 为每专家一个 FP32 标量(`scales.shape == [E]`)。
kernel 内部的 DPAS 指令仍按 `bf16`/`fp16` 运行(与 FP8 Variant A 完全
一致),在寄存器内先把 `int8` 上采样到激活 dtype 再送入 DPAS,因此峰
值性能与 FP8 相同、但 weight footprint 更小。

```python
outputs = ark.moe_gemm_prefill(
    activations,  # [total_tokens, K],f16/bf16
    weights,  # [E, K, N] 行主序 torch.int8(vllm 布局)
    num_tokens_per_expert,  # [E] int32
    scales=scales,  # [E] fp32,每专家一个 per-tensor scale
    scale_scheme="per_tensor",
)
```

该分支会调用 `moe_gemm_prefill_int_dpas`(Variant A INT8) ——
`per_tensor` 方案现在按 `weights.dtype` 分派(FP8 走原有 FP8 DPAS
入口;`torch.int8` 走新的 INT8 DPAS 入口)。构建时未链接该 pybind
符号则测试自动跳过。

```bash
pytest -v -s test_moe_prefill_perf.py::TestMoEGemmPrefillPerf::test_perf_int8_per_tensor
```

精度对齐通过
`test_moe_prefill_accuracy.py::test_accuracy_int8_per_tensor_dpas`
在同一份生产形状上覆盖,使用标准 INT8 容差
(`rtol=atol=1e-1`)。

**状态:NEEDS-HARDWARE-VALIDATION**(未在硬件上验证过的移植;Phase 1
仅支持 sym。per-group / asym 的 INT4 / INT2 DPAS 是后续阶段,将复用
同一份 mainloop 骨架,只在其中追加 unpack 步骤)。
