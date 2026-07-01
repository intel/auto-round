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
- **Base+Deq**: 对量化测试,反量化步骤单独计时,`base+deq(ms)` 列报告 `baseline + deq` — 一个"权重量化存储但复用标准 matmul 基线"的流水线每步付出的端到端开销。对于 FP 行,此值等于 `baseline(ms)`。
- **ARK Kernel**: 带融合操作的优化 `ark.moe_gemm`。
- **Speedup**: 报告 `(base+deq) / ark` — 与我们融合 kernel 的现实对比点。

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
    activations,          # [total_tokens, K], f16/bf16
    weights,              # [E, K, N] 行主 FP8 (vllm 布局)
    num_tokens_per_expert,# [E] int32
    scales=scales,        # [E] fp32,每专家一个 per-tensor scale
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
生产形状下测试;所有路径共享容差 `rtol=atol=7e-2` (E4M3) / `1e-1`
(E5M2)。

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
