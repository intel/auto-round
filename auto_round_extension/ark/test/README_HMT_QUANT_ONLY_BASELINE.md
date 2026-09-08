# HMT+Quant 融合 kernel：Quant-only Baseline 带宽测试计划

- **日期**：2026-09-08
- **分支**：`xpu-hmt-mxfp4`
- **状态**：改动落地完成、正确性回归 166 passed；`--wan` 首轮预览已跑（正式复测待做）
- **任务来源**：[# CRI-WAN-HMT]（同事提供的 WAN 逐算子表格）

## 1. 背景与目标

被测对象是激活侧融合 kernel：**32-point Hadamard (HMT) + MXFP4 量化**，输入 fp16/bf16 激活，
输出 packed FP4 codes + E8M0 scale。该 kernel 纯 memory-bound（读 `2 B/elem`，写 `≈0.53 B/elem`），
HMT 的蝶形运算在寄存器内完成、理论应被内存延迟掩盖。

**目标**：用 **Quant-only（去掉 HMT，只做 MXFP4 量化）** 作为 baseline，验证命题——

> **加上 HMT 后有效带宽几乎不掉**（即融合是"免费"的），
> 并顺带用 streaming-copy roofline 兜底 Quant-only 的绝对带宽是否合理。

**口径（已与同事确认）**
- 指标：**有效带宽 BW（GB/s）**，不是 latency / TFLOPS。
- 范围：**只测 HMT+Quant 激活量化部分，不包含 GEMM**。
- Baseline 方案：**选项 1 —— 从自己的融合 kernel 中剥离 HMT，得到 quant-only**。

## 2. 为什么选选项 1（受控消融）

| 方案 | 说明 | 结论 |
|---|---|---|
| **1. 自剥离 quant-only** | 同一个 kernel，只跳过 HMT（norm + 5 级蝶形） | ✅ 首选：load/store/向量化/打包/e8m0/输出 layout 全部一致，是唯一干净的 ablation |
| 2. 仓库现成量化器 | 多为 Triton/torch（CUDA 取向）、输出契约可能不同 | 可作绝对 sanity check，但字节数未必一致 |
| 3. vllm-xpu-ops 量化器 | 需确认是 MXFP4+group32+e8m0+packed 才可比 | 同上，仅当字节契约完全一致才值得 |

**关键前提**：带宽 = 字节数 / 时间。**只有两个 kernel 移动的字节数完全一致，BW 才可比。**
选项 1 天然满足：它读同样的激活、写同样的 codes/scale，唯一差别是跳过寄存器内的蝶形。

## 3. 字节流量契约（任何实现都必须满足）

对形状 `[M, K]`、dtype 元素大小 `sizeof(T)`：

```
bytes = M*K*sizeof(T)   // 读激活一遍
      + M*K/2           // 写 packed codes（2 个 FP4 值/字节）
      + M*K/32          // 写 e8m0 scale（每 32 元素 1 字节）
```

- 输入：fp16 / bf16，连续激活，`K % 32 == 0`；
- 输出契约与融合 kernel 完全一致：group=32、E8M0、packed FP4（偶数元素低 nibble）、canonical zero。
- Quant-only 路径的数值 = **直接量化原始 x**（不乘 norm、不跑蝶形），仅此与融合版不同。

## 4. 被测 shape（来自 WAN 表格的输入激活 `[M, K]`）

同事表格每行 `[M, K]` = 该 GEMM 的输入激活 = 要做 HMT+Quant 的张量（HMT dim=32，K 均可整除 32）。

| 分组 | 激活 shape [M, K] | 每块出现位置 | 组数 M·K/32 |
|---|---|---|---|
| A（主力） | `[75600, 5120]` | qkv / attn-out / cross-q / cross-out / FFN-up | 12.1 M |
| B | `[75600, 13824]` | FFN-down（K 不同！） | 32.7 M |
| C（小） | `[512, 5120]` | cross-kv（text 侧） | 81.9 K |

> 划掉的 text projection / text-encoder FFN：**已确认也要跑**（见 §9）。
> `[M, K]` 是否按 DP 再切分 → **已确认：用表格里的全量 M**（BW 口径与 M 无关，DRAM-bound 下 M 不变；latency 口径才需按 DP 换算）。

## 5. 测量口径与公式

沿用 `benchmarks/bench_mxfp4_hadamard.py` 的既有协议：

- `bench()`：warmup + 多轮取平均，`torch.xpu.synchronize()` 包住计时窗口；
- `BW = bytes / mean_latency`；
- 报告三组数字：
  1. `BW(quant-only)`
  2. `BW(HMT+quant)`
  3. `ratio = BW(HMT+quant) / BW(quant-only)` ← 主验收指标（目标 ≈ 1.0）
- 兜底 sanity：`BW(quant-only) / BW(streaming-copy)`（复用 copy roofline 基线），
  确认 quant-only 的绝对带宽接近机器可达带宽，而不是"两个都慢的自我安慰"；
- 沿用 cache-residency 判定（`CACHE_TOLERANCE`）：cache 内配置不参与 gate。

## 6. 软件改动计划

### 6.1 C++ kernel（`wrapper/include/xpu_mxfp4_hadamard.hpp`）
- 在 FWHT per-item 路径（`fwht_quant_per_item`）增加 **quant-only 开关**：
  - 方案 A（推荐）：给 `fwht_quant_per_item` 加 `bool quant_only` 参数，为真时**跳过 norm 乘法与 5 级蝶形**，直接量化原始 load 值；
  - 方案 B：用模板/`if constexpr` 派生出独立 `quant_per_item`，二者共存（代码略多但互不影响热路径）；
  - 其余代码（load、absmax、e8m0、编码、canonical zero、打包、store）一行不动。
- `mxfp4_hadamard_quant(...)` dispatch 增加透传参数。

### 6.2 绑定（`auto_round_kernel/ark.cpp`）
- 静态入口 `mxfp4_hadamard_quant(...)` 增加 `bool use_quant_only` 参数并校验/透传；
- `m.def(...)` 增加对应 `pybind11::arg("use_quant_only")`。

### 6.3 Python wrapper（`auto_round_kernel/mxfp4_hadamard.py`）
- `mxfp4_hadamard_quant(...)` 增加私有 kwarg `_quant_only: bool = False`（风格对齐 `_force_xmx`），
  为真时强制走 quant-only（FWHT 路径剥离 HMT）；
- 新增 **quant-only 参考**（纯 torch）：对原始 `x` 复用现有 `_e8m0_and_quantized` + `_encode_fp4` + `pack_codes`，
  即 `quant_reference(x) -> (codes, scale)`（不经过 `transform_reference`）。

### 6.4 正确性测试（`test/test_mxfp4_hadamard.py`）
- quant-only 路径：codes/scale 与 `quant_reference` **bit-exact**；
- 覆盖 fp16/bf16 × 上述三档 shape（小 shape 即可，如 `(64,5120)/(64,13824)/(16,5120)`）；
- 边界：全 0 输入（scale=0、codes=0）、canonical zero。

### 6.5 Benchmark（`benchmarks/bench_mxfp4_hadamard.py` 或新增脚本）
- 加入 WAN 三档 shape 与 `quant_only` 计时，输出：
  `shape / dtype / path / BW(quant-only) / BW(HMT+quant) / ratio / BW_copy / cached?`；
- 分别支持 FWHT 默认路径（Sylvester）与 `--xmx`（可选）。

## 7. 验收标准（provisional，跑完首轮再定稿）

- ✅ 正确性：quant-only 输出 bit-exact；
- ✅ 主指标：`BW(HMT+quant)/BW(quant-only) ≥ 0.9`（预期接近 1.0；若显著低于 0.9 说明 HMT 未完全隐藏，需排查）；
- ✅ sanity：`BW(quant-only)` 显著接近 streaming-copy 可达带宽（体现为合理 ratio，非 cache-resident 配置）；
- ✅ 数据可复现：同一 shape/dtype 多轮波动在可接受范围。

## 8. 交付物

- 逐 shape 表格：`quant-only GB/s | HMT+quant GB/s | ratio`，分 dtype/路径；
- （可选）CSV 导出；
- 结论一句话：HMT 融合带来的带宽开销 ~X%。

## 9. 与同事确认结果：

- 路径：只测默认 FWHT（bit-exact，Sylvester H）
- dtype：bf16（kernel 正确性测试仍覆盖 fp16/bf16；bench 的 `--wan` 口径只跑 bf16）
- M 就用表格里的全量 M
- 划掉的两行（text projection / text-encoder FFN）也要跑
- 期望交付：只交单算子 BW（不需要按 block 重复次数汇总整模型）

## 10. TODOs

### Phase 1：改动落地（已完成）
- [x] C++：`fwht_quant_per_item` 加 `quant_only` 开关（方案 A：bool 参数，跳 norm 乘法语义与 5 级蝶形）
- [x] C++：`ark.cpp` 绑定与 `m.def` 透传 `use_quant_only`
- [x] Python：`mxfp4_hadamard_quant` 增加 `_quant_only`
- [x] Python：新增 quant-only 纯 torch 参考 `mxfp4_quant_reference`（+ `__init__` 导出）

### Phase 2：正确性（已完成）
- [x] `test_mxfp4_hadamard.py` 增加 quant-only 用例（CPU 参考 + XPU kernel bit-exact）
- [x] CPU 参考用例跑通：`TestReferenceContract` 28 passed（含 quant-only reference 与 identity 等价性）
- [x] XPU kernel 用例跑通：完整文件 **166 passed**（fp16/bf16 + 边界 + `(4,13824)`/`(2,5120)` 大 K shape；quant-only 与 `mxfp4_quant_reference` bit-exact）

### Phase 3：Benchmark
- [x] 扩展 `bench_mxfp4_hadamard.py`：`--wan` 四档 WAN 激活 shape、quant_only 计时、`f/q` ratio 列与 gate、双正确性校验
- [x] 首轮预览（Arc Pro B60，warmup5/iters30，共享设备、噪声大，非正式）：

      M       K dtype   ok  okQ  BW_fused  BW_quant    f/q   BW_copy   f/cp
      75600  5120 bf16  pass pass   375.0    390.3  0.961    398.0  0.942
      75600 13824 bf16  pass pass   380.4    391.1  0.973    399.0  0.953
        512  5120 bf16  pass pass   353.7    378.2  0.935   2085.5  (cached)
        512  4096 bf16  pass pass   305.2    337.2  0.905   2076.1  (cached)

      → 小 shape 落在 cache-resident，被 gate 排除；DRAM-bound 两档 f/q≈0.96–0.97（HMT 开销 ~3–4%），达标（≥0.9）。
- [ ] 正式复测（加 iters / 独占设备后重跑，供交付）
- [ ] （可选）XMX 路径复测
- [ ] 记录 streaming-copy roofline，做 quant-only 绝对带宽 sanity

### Phase 4：分析与交付
- [ ] 生成逐 shape 结果表 + 结论
- [ ] 交付逐算子 BW（只交单算子，不汇总整模型）
