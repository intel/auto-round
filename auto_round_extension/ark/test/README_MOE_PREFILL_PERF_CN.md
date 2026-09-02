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

精度对齐由
`test_moe_prefill_accuracy.py::test_accuracy_int4_dpas_per_group`
覆盖,该用例强制 `ARK_MOE_PREFILL_DPAS_S4=1` +
`ARK_MOE_PREFILL_DPAS_INT8=0`,专门验证单遍 mainloop 路径,形状矩阵与
`test_accuracy_int4` 一致,容差 `rtol=atol=1e-1`。

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
