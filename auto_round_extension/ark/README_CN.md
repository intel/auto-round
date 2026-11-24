## 什么是 AutoRound Kernel（ARK）？

AutoRound Kernel（ARK）是面向 Intel 平台的低比特加速库，为大语言模型推理提供三类优化算子。

| 算子类别 | CPU | XPU（Battlemage） |
|:--|:--:|:--:|
| **仅权重量化线性层**（INT4/INT8/FP8/FP4） | ✅ | ✅ |
| **MoE 分组 GEMM** | ❌ | ✅ |
| **SageAttention**（SDPA / SAGE） | ❌ | ✅ |

**已验证 CPU：** Intel Xeon Scalable（Sapphire Rapids / Emerald Rapids）、Intel Xeon 6（Sierra Forest / Granite Rapids）<br>
**已验证 GPU：** Intel Arc B 系列 / Arc Pro B 系列（Battlemage）

### 生态集成

ARK 算子已集成到以下项目：

| 项目 | 集成 | 说明 |
|--|--|--|
| [vllm](https://github.com/vllm-project/vllm) | [`inc_wna16_linear.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_linear.py#L379-L470) | `INCXPUARKLinearMethod` 通过 `auto_round_kernel.qlinear.QuantLinear` 在 XPU 上提供仅权重量化线性层。 |
| [vllm-omni](https://github.com/vllm-project/vllm-omni) | [`sage_attn.py`](https://github.com/vllm-project/vllm-omni/blob/f8340d078e4e9c3b793bd92d55d891b29703f0a8/vllm_omni/diffusion/attention/backends/sage_attn.py#L27) | `SageAttentionBackend` 通过 `ARK.sagev1` 在 XPU 上加速扩散模型注意力。 |
| [Transformers](https://github.com/huggingface/transformers)（通过 [auto-round](https://github.com/intel/auto-round)） | [`backend.py`](https://github.com/intel/auto-round/blob/main/auto_round/inference/backend.py#L531) | AutoRound 量化的模型默认在 CPU/XPU 上使用 ARK，无需额外配置。 |

---

## 1. 线性层（仅权重量化 GEMM）

面向大语言模型推理的低比特仅权重量化线性层，支持 CPU 和 XPU。

### API

| API | 说明 | 平台 |
|--|--|--|
| `QuantLinear` | 统一的 PyTorch 模块（GPTQ/AWQ/原始量化检查点） | CPU / XPU |
| `QuantLinearGPTQ` | GPTQ 格式检查点加载器 | CPU / XPU |
| `QuantLinearAWQ` | AWQ 格式检查点加载器 | CPU / XPU |
| `QuantLinearFP8` | FP8 仅权重量化线性层 | CPU / XPU |
| `woqgemm` | 低层仅权重量化 GEMM（打包格式） | CPU / XPU |
| `woqgemm_s8` | 带 scale 的低层 INT8 权重 GEMM | CPU / XPU |
| `_repack_quantized_weight` | 将原始 qweight/qzero/scale 重新打包为 ARK 格式 | CPU / XPU |
| `_unpack_weight` | 将 ARK 格式权重解包回全精度 | CPU / XPU |

### 关键特性

> **W4A8 / W2A8 重缩放（QQQ 风格）**：XPU 上支持将低比特权重（INT2/INT4）重缩放为 INT8，并通过 INT8 GEMM 计算，避免 FP16 反量化。该功能通过环境变量 `ARK_AUTO_S8` 自动启用，详见 [xpu_wrapper.hpp](auto_round_kernel/wrapper/include/xpu_wrapper.hpp)。

**XPU 上的稠密 W4A16（SYCL TLA）**

设置 `ARK_WOQ_S4_DPAS_SPLIT_K4_M1=1` 后，`M=1, K=4096, 8192 <= N <= 16384` 且 `N` 能被 128 整除的调用可以走 K64 split-K4。这是唯一的单行 DPAS 规则，不依赖设备 ID 或形状表。

该路由要求对称 INT4 权重、FP16 内核输入输出、group-major FP16 scale、group size 为 32、64 或 128，以及未处于图录制状态的有序队列。每个队列复用 256 KiB 的 FP32 临时空间；四份 FP32 partial 求和后只加一次可选 bias，再转为 FP16。Split-K 改变累加顺序，输出不保证与不切分 DPAS 逐 bit 相等。

默认关闭，规则未命中的单行调用仍走 GEMV。`M>1` 的分流、权重打包和 BF16 转换不变。启用前请在目标设备测量整次调用延迟；FP16 和 BF16 共用规则，BF16 在下界附近不保证有收益。

### M=1 W4A16 基准测试

[bench_woq_s4.py](test/bench_woq_s4.py) 复用 [test_woq_s4.py](test/test_woq_s4.py) 的测试夹具，在包括 `4096x16384` 的十种 K/N 形状上对照 GEMV 和 K64 split-K4，覆盖 group32/128、有无 bias 以及 FP16/BF16 输入。宽 N 规则内的所有配置走生产路由。规则外的 split-K4 使用独立探针，标为 `path=probe`，不代表生产路由。

在 ARK 目录下运行，使用可导入新编译 ARK XPU 运行时的 Python 3.11 或更新版本。用 `ZE_AFFINITY_MASK` 选择一张空闲卡。以下命令只检查生产内核和数值，无需编译探针，也不计时：

```bash
PYTHONPATH="$PWD/test${PYTHONPATH:+:$PYTHONPATH}" \
    python -P test/bench_woq_s4.py --routes-only --shape 4096x8192 --shape 4096x16384
```

全量扫描时，将 `SYCL_TLA_SRC`、`PYBIND11_SRC` 指向编译 ARK 时使用的依赖源码根目录，并为目标设备编译独立探针。以下以 CRI 为例，B70 使用 `bmg-g31`：

```bash
source /opt/intel/oneapi/setvars.sh --force
icpx --version
kernel="$PWD/auto_round_kernel"
probe_dir=$(mktemp -d)
icpx -std=c++17 -O3 -DNDEBUG -shared -fPIC -fsycl -fno-sycl-instrument-device-code \
    -fsycl-targets=spir64 -Xs "-device cri" \
    -Xspirv-translator "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate" \
    -DARK_XPU=1 -DARK_SYCL_TLA=1 -DBTLA_SYCL -DCUTLASS_ENABLE_SYCL=1 -DSYCL_INTEL_TARGET=1 \
    -I"$kernel/wrapper/include" -I"$kernel" -I"$kernel/bestla" \
    -isystem "$SYCL_TLA_SRC/include" -isystem "$SYCL_TLA_SRC/tools/util/include" \
    -isystem "$SYCL_TLA_SRC/examples/common" -isystem "$PYBIND11_SRC/include" \
    -isystem "$(python -c 'import sysconfig; print(sysconfig.get_path("include"))')" \
    test/ark_2673_dpas_probe.cpp -o "$probe_dir/ark_2673_dpas_probe.so"
ARK_VERBOSE=2 PYTHONPATH="$probe_dir:$PWD/test${PYTHONPATH:+:$PYTHONPATH}" \
    python -P test/bench_woq_s4.py
```

可重复指定 `--shape KxN` 缩小范围，使用 `--routes-only` 只检查数值和实际内核而不计时，或用 `--io-buffer-ab=split --shape 4096x8192` 复跑 BF16 输入/输出缓冲区实验。脚本会打印加载库的哈希并验证设备，也可用 `--expect-sha256`、`--expect-probe-sha256` 和 `--device-uuid` 固定复跑环境。计时要求空闲显存大于 90%，且包含分配/清零、dtype 转换与归约，除非 A/B 明确改变该环节。

### 支持的数据类型

#### CPU

| 权重类型 | 计算类型 | Scale 类型 | 算法 |
|--|:--:|:--:|:--:|
| INT1–INT8 | INT8<sup>[1]</sup> / BF16 / FP32 | BF16 / FP32 | 对称 / 非对称 |
| FP8（E4M3、E5M2） | BF16 / FP32 | FP32 / FP8（E8M0） | 不适用 |
| FP4（E2M1） | BF16 / FP32 | BF16 / FP32 | 不适用 |

#### XPU

| 权重类型 | 计算类型 | Scale 类型 | 算法 |
|--|:--:|:--:|:--:|
| INT4、INT8 | INT8 / FP16 | FP16 | 对称 |
| FP8（E4M3、E5M2） | FP16 | FP16 / FP8（E8M0） | 不适用 |

<sup>[1]</sup> INT8 计算包含动态激活量化，结果会反量化为浮点类型。

### 示例

```python
import auto_round_kernel as ark

# 准备量化权重：qweight [K, N] int4/int2，scale [K/G, N] fp16/fp32，zp [K/G, N] int4/int2
packw = ark.repack_quantized_weight(
    qweight,
    scale,
    zp,
    blocksize=128,
    compute_type="fp16",
    weight_type="int4",
    scale_type="fp16",
    asym=False,
)

# 运行仅权重量化 GEMM：activation [M, K] -> output [M, N]
output = ark.woqgemm(
    activation,
    packw,
    bias,
    n,
    k,
    groupsize=128,
    compute_type="fp16",
    weight_type="int4",
    scale_type="fp16",
    asym=False,
)

# 解包回全精度以便验证
_decompressed = ark.unpack_weight(
    packw,
    dtype=torch.float16,
    n=n,
    k=k,
    groupsize=128,
    compute_type="fp16",
    weight_type="int4",
    scale_type="fp16",
    asym=False,
)
```

详见 [test_weightonly.py](test/test_weightonly.py)，其中包含 CPU 和 XPU 上的权重重新打包、验证和 `woqgemm` 运行示例。

---

## 2. MoE（混合专家分组 GEMM）

面向 MoE 层的分组 GEMM，不同专家可以处理不同数量的 token。

### API

| 函数 | 说明 | 平台 | 激活类型 | 权重类型 |
|--|--|--|:--:|:--:|
| `ark.moe_gemm(...)` | 专家分组 GEMM | XPU | FP16 / BF16 | FP16 / BF16 |
| `ark.moe_gemm(...)`（开发中） | INT4 权重分组 GEMM | XPU | FP16 / BF16 | INT4 🚧 |
| `ark.moe_gemm(...)`（开发中） | INT2 权重分组 GEMM | XPU | FP16 / BF16 | INT2 🚧 |
| `ark.moe_gemm(...)`（开发中） | INT8 权重分组 GEMM | XPU | FP16 / BF16 | INT8 🚧 |

> 🚧 INT2 / INT4 / INT8 权重支持仍在开发中，详见 [#PR](https://github.com/intel/auto-round/pull)。

### 细节

| 参数 | 形状 | 类型 |
|--|--|--|
| activations | `[total_tokens, K]` | FP16 / BF16 |
| weights | `[num_experts, K, N]`（行主序） | FP16 / BF16 |
| num_tokens_per_expert | `[num_experts]` | INT32 |
| scales（可选） | `[num_experts, N]` | FP16 / BF16 |
| **output** | `[total_tokens, N]` | 与激活相同 |

### 示例

```python
# FP16/BF16 MoE
output = ark.moe_gemm(activations, weights, num_tokens_per_expert)

# INT4 MoE（即将支持）
# output = ark.moe_gemm(activations, q4_weights, num_tokens_per_expert, scales=scales)
```

构建要求：`ARK_SYCL_TLA=ON`。详见 [test_moe.py](test/test_moe.py)。

---

## 3. SageAttention（XPU SDPA 加速）

ARK 在 XPU 上提供完整的缩放点积注意力内核族，从原生 FP16 SDPA 到 INT8 量化 SageAttention 变体均可使用。

### API 概览

| 函数 | 说明 | Q/K/V 输入 | PV 精度 | Head Dim |
|--|--|--|:--:|:--:|
| `ark.sdpa` | FP16/BF16 SDPA（flash attention） | FP16 / BF16 | FP16 | 64、96、128、192 |
| `ark.sage` | 低层 INT8 SAGE（预量化 Q/K） | INT8（Q/K）、FP16（V） | FP16 | 64、128 |
| `ark.sage_pvi8` | 低层 INT8 SAGE（预量化 Q/K/V） | INT8 | INT8 | 64、128 |
| `ark.sagev1` | 高层 FP16 -> 内部 Q/K 量化 -> SAGE | FP16 / BF16 | FP16 | 64、128 |
| `ark.sagev1_pvi8` | 高层 FP16 -> 内部 Q/K/V 量化 -> SAGE PV INT8 | FP16 / BF16 | INT8 | 64、128 |
| `ark.sageattn` | SageAttention 兼容 API 的分发器 | FP16 / BF16 | FP16 / INT8 | 64、128 |
| `ark.sage_dynquant` | 动态 INT8 分块 Q/K 量化 -> SAGE（可替换 SDPA） | FP16 / BF16 | FP16 | 64、128 |

### 约束

| 约束 | `sdpa` | `sagev1` / `sagev1_pvi8` / `sage_dynquant` |
|--|:--:|:--:|
| Q/K/V 类型 | FP16、BF16 | FP16、BF16 |
| Head dim | 64、96、128、192 | 64、128 |
| `dropout_p` | 必须为 0.0 | 必须为 0.0 |
| 布尔 mask | 回退到 PyTorch | 回退到 PyTorch |
| 加性 mask 形状 | `[B, 1, Sq, Skv]` FP32 | `[B, 1, Sq, Skv]` FP32 |
| `quant_block_size` | 不适用 | 1（逐 token）或 >=32 |

---

## 安装

### 通过 pip 安装

```bash
pip install auto-round-lib
```

### 从源码安装

```bash
pip install . --no-build-isolation
# 或
python setup.py bdist_wheel; pip install dist/*
```

启用 MoE / SageAttention 支持需要 `ARK_SYCL_TLA=ON`。

对于 oneAPI/SYCL-TLA 构建，每个编译进程的模板编译可能需要数 GB 内存。如果构建主机内存不足，可以通过 `ARK_SYCL_TLA_JOBS` 限制并发的 XPU 编译数量：

```bash
ARK_SYCL_TLA_JOBS=1 pip install . --no-build-isolation
```

在配备 oneAPI 的构建主机上，建议启用编译命令并测量每个翻译单元，再选择更高的并发数：

```bash
python tools/measure_sycl_tla_compile_memory.py \
  --build-dir /path/to/ark/xbuild \
  --output /tmp/ark-sycl-tla-rss.json
```

该工具会串行运行编译器命令，并报告峰值 RSS、直接分发数量和可达模板声明数量。要使测量结果反映 SYCL-TLA 构建，当前环境必须提供 `icx` 以及生成的 `compile_commands.json`。

---

## 测试

| 测试 | 说明 |
|--|--|
| [test_weightonly.py](test/test_weightonly.py) | WOQ GEMM 打包、解包和运行（CPU/XPU） |
| [test_moe.py](test/test_moe.py) | MoE 分组 GEMM |
| [test_flash_attn.py](test/test_flash_attn.py) | SDPA（flash attention）prefill |
| [test_sdpa.py](test/test_sdpa.py) | SDPA 基准测试 |
| [test_sdpa_parity.py](test/test_sdpa_parity.py) | SDPA 与 PyTorch 的一致性测试 |
| [test_sage_dynquant.py](test/test_sage_dynquant.py) | SageAttention 动态 INT8 量化基准测试 |
| [test_bench_bmg.py](test/test_bench_bmg.py) | BMG SDPA / SageAttention 基准测试 |
| [test_matmul.py](test/test_matmul.py) | 低层 matmul |
| [test_packq.py](test/test_packq.py) | 权重打包工具 |
