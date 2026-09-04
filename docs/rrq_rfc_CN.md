# RFC: 递归残差量化（RRQ）INT2+2+2+2

- 状态：草案
- 作者：AutoRound contributors
- 目标版本：待定
- 评审人：AutoRound maintainers 和 runtime owners

## 摘要

本 RFC 提议实现实验性 RRQ 算法。RRQ 将每个符合条件的权重张量编码为一个 INT2 基础平面（base）和三个 INT2 残差平面（residual）。推理时通过选择 base + 0~3 个 residual 平面，可以按 2、4、6 或 8 个等效比特重建权重。

**分步交付策略**：
1. **Phase 1**：RTN 量化，base model（INT2）和 residual model（3×INT2）**分开存储**，各自独立可用。
2. **Phase 2**：支持从已有 base model + 原始 FP 权重生成 residual model。
3. **Phase 3**：加入 AutoRound sign-SGD 调优（OPT），使每轮量化真正最优。

第一版严格只支持 `INT2+INT2+INT2+INT2`、weight-only 线性层和 eager PyTorch 推理。现有的 AutoRound、AutoGPTQ、AWQ、GGUF、vLLM、Triton、CUDA、XPU、HPU、MLX 及激活量化后端将拒绝 RRQ，不能静默丢弃残差平面。

## 动机

目前同时部署 W2A16 和 W4A16 需要保留两个独立量化模型。RRQ 保留 W2 表示，并用后续三个 W2 平面编码重构误差。最大表示的大小约为四个 INT2 平面及四套量化元数据，同时精度成为加载和运行时选择。

RRQ 并非 bit-shift 表示。每个平面独立拟合 scale 和 zero point，因此相较直接固定比特量化可能降低重构误差。本 RFC 不预设无条件的精度或性能结论，二者均应由验收测试量化。

## 目标（分阶段）

### Phase 1：RTN 量化 + 分开存储

- 增加 `RRQConfig` 和 `rrq` 算法别名。
- 对每个符合层，用 RTN（`disable_opt_rtn=True`）做 4 轮顺序量化：`W → QDQ_W2 → E₁ → QDQ_W2 → E₂ → QDQ_W2 → E₃ → QDQ_W2`。
- **分开存储**：
  - Base model：标准 INT2 量化模型（与现有 W2A16 格式完全兼容，可直接被现有 runtime 加载）。
  - Residual model：独立的 3×INT2 平面产物（新的 `auto_round:rrq` 格式）。
- 在 eager PyTorch 中支持组合加载：base + 0~3 个 residual → 2/4/6/8 bit 推理。
- 对不支持的格式和选项尽早报出可操作错误。

### Phase 2：从已有 base model 生成 residual

- 支持从已导出的 base model（INT2 checkpoint）+ 原始 FP 权重，生成 residual model。
- 无需重新量化 base，只需计算 `E₁ = W_fp − W_dequant_base` 并对残差做 3 轮 RTN INT2 量化。
- 适用于已有 INT2 量化模型的用户增量升级场景。

### Phase 3：AutoRound 调优（OPT）

- 将每轮 RTN 替换为完整 AutoRound sign-SGD 优化（`iters > 0`）。
- 每轮以当前残差为目标，独立调优到该轮最优，前缀冻结进入下一轮。
- 调优后的 base + residual 组合精度优于纯 RTN 分解。

## 非目标

- 任意平面数、位宽、数据类型、每平面 group size 或混合平面组合。
- 融合 GPU kernel，或宣称与固定 INT4/INT8 kernel 性能相当。
- 将 base 和 residual 打包为单一 checkpoint（第一版分开存储）。
- 第一版不支持 activation quantization、LoRA 训练、diffusion、MLLM、embedding、卷积和 Conv1D。
- Phase 1 不追求调优精度提升（纯 RTN），调优留给 Phase 3。

## 建议 API

### Phase 1：完整 RRQ 量化（base + residual 分开导出）

```python
from auto_round import AutoRound, RRQConfig

config = RRQConfig(group_size=128, sym=False)
autoround = AutoRound(model, tokenizer, alg_configs=config)
autoround.quantize()
# base model：标准 INT2，可直接用于现有 W2A16 runtime
autoround.save_quantized("./model-rrq-base", format="auto_round")
# residual model：3 个 INT2 平面
autoround.save_quantized("./model-rrq-residual", format="auto_round:rrq")
```

### Phase 2：从已有 base 生成 residual

```python
from auto_round import generate_rrq_residual

# base 已量化 checkpoint + 原始 FP 权重
generate_rrq_residual(
    base_model="./model-rrq-base",   # 已导出的 INT2 base
    raw_model="./Qwen3-8B",          # 原始 FP 权重
    output_dir="./model-rrq-residual",
    group_size=128,
    sym=False,
)
```

### Phase 3：OPT 调优

```python
config = RRQConfig(group_size=128, sym=False, iters=200, lr=1e-3)
# 其余同 Phase 1
```

`RRQConfig` 固定以下值；调用者覆盖时抛出 `ValueError`：

| 字段 | 值 |
| --- | --- |
| `bits` | `2` |
| `data_type` | `int` |
| `act_bits` | `16` |
| 残差平面数 | `3` |
| 平面总数 | `4` |

`group_size` 延用已有标量契约（`-1`、`0` 或正整数）。第一版每个平面使用相同的 `sym` 值。默认值需要评审；此 RFC 建议 `sym=False`，因为非对称 INT2 更容易处理非零中心的残差分布。已有 `check_to_quantized` 规则排除的层保持浮点。

加载后的 RRQ 模型提供 `set_rrq_bits(bits: Literal[2, 4, 6, 8])`，原子地修改全部 RRQ 层的活跃平面数。按请求混合精度、按层策略和部分磁盘加载留给后续 RFC。

## 算法

RRQ 采用**分 4 次顺序最优化**：每一轮以"当前残差"作为该轮优化目标，独立 sign-SGD 调优到该轮最优，前缀累加冻结进入下一轮。整体即"2-bit 优化到最优 → 残差再 2-bit 优化到最优 → …"，共 4 轮。

设目标权重为 $W$，平面数 $K=4$。设已收敛前缀重构 $A_0=0$，当前残差 $E_0\equiv W$。第 $k$ 轮 ($k=0,1,2,3$) 独立做一遍完整 AutoRound 调优：

$$
\hat E_k,\, s_k,\, z_k \;\triangleq\; \arg\min_{v_k, m_k, c_k}\; \mathcal{L}\!\big(\, x\, \big(F\big(x,\, A_k + \mathrm{QDQ}_{\text{INT2}}(E_k;\, v_k, m_k, c_k)\big) + b\big)\, \big),\qquad
A_{k+1}\;\triangleq\; A_k + \hat E_k,\qquad
E_{k+1}\;\triangleq\; W - A_{k+1}.
$$

每轮调优收敛后，把该平面的 $(\hat E_k, s_k, z_k)$ 冻结，进入下一轮；下一轮的优化目标就是新的残差 $E_{k+1}$。$\mathrm{QDQ}$ 内部 `round` 走 straight-through（STE），保证该轮 $v_k, m_k, c_k$ 可微。

关键点：残差 $E_k$ 始终在**原始浮点权重域**计算（$E_k = W - \sum_{j < k}\hat E_j$），而不是在整数 code 域；每轮独立持有 $s_k, z_k$。每轮都是完整的可微 AutoRound 优化，基于真实残差做 scale/量化。

各轮语义：

- 第 0 轮（base）：以 $W$ 为目标，完整 AutoRound 调优 $v_0$ + 组 min/max scale（$m_0, c_0$），得到最优 2-bit 表示 $\hat E_0$。
- 第 1 轮（residual, 2+2 → 4-bit）：以 $E_1 = W - \hat E_0$ 为目标，再跑一遍 AutoRound 调优 $v_1, m_1, c_1$；收敛结果与第 0 轮的 $(\hat E_0, s_0, z_0)$ 一起构成 4-bit 前缀最优。
- 第 2 / 3 轮同理：以 $E_k$ 为目标独立调优，使 $(\hat E_0 + \cdots + \hat E_k)$ 在该轮 loss 意义下最优。

即：2-bit 最优 + 4-bit 前缀最优 + 6-bit 前缀最优 + 8-bit 前缀最优，一次训练全部得到。

实现上每轮复用 AutoRound 现有单层调优循环：`wrapper_block` → 校准 → sign-SGD 迭代 → `collect_best_params` → `unwrapper` 把该轮平面冻结回模块；下一轮在冻结前缀的基础上重新 wrap、重新调优。4 轮全部收敛后写回 K 个平面的 code/scale/zp，得到可加载的 `RRQLinear`；推理时按需前缀累加（1/2/3/4 平面 → 2/4/6/8 bit）。

调优期开销：4 轮调优，第 $k$ 轮 forward 需做 $(k+1)$ 次 weight QDQ，总 QDQ 次数 ≈ $1+2+3+4=10$ 次（单层 base 仅 1 次），均为 mem-bound 逐元素运算，远小于 matmul；对正确性优先的 MVP 可接受，后续可融合 kernel 优化。

## 模型和 Checkpoint ABI

RRQ 采用**分开存储**策略：base model 和 residual model 是两个独立 artifact，各自独立可加载。

### Base Model（Phase 1 产出）

Base model 是标准 INT2 量化模型，使用现有 `auto_round` 导出格式：

- `quantization_config` 中的 `quant_method` 为现有值（如 `"auto_round"` / `"gptq"`）。
- 包含 `qweight`（packed INT2）、`scales`、`qzeros`（如不对称）。
- **完全兼容现有 runtime**（AutoGPTQ、vLLM、llama.cpp 等），无需任何修改。
- 用户可直接用 base model 做 2-bit 推理，无需 RRQ 支持。

### Residual Model（Phase 1 产出 / Phase 2 单独生成）

Residual model 是新的 `auto_round:rrq` 格式，包含 3 个 INT2 残差平面：

| buffer/attribute | 形状 | 含义 |
| --- | --- | --- |
| `qweight_1` | packed INT2 | 第 1 残差平面（2+2→4-bit 的增量） |
| `qweight_2` | packed INT2 | 第 2 残差平面（4+2→6-bit 的增量） |
| `qweight_3` | packed INT2 | 第 3 残差平面（6+2→8-bit 的增量） |
| `scales_1` ... `scales_3` | 现有 scale 布局 | 每残差平面 group scale |
| `qzeros_1` ... `qzeros_3` | 现有 zero point 布局 | 非对称量化时的每平面 zero point |
| `rrq_format_version` | 字符串元数据 | ABI 版本，初始为 `"1"` |

注意：residual model **不包含** `qweight_0`/`scales_0`（base 平面），因为 base 由独立的 base model 提供。

导出的 `quantization_config`（residual model）应包含：

```json
{
  "quant_method": "auto-round-rrq",
  "format_version": 1,
  "base_bits": 2,
  "residual_planes": [2, 2, 2],
  "supported_effective_bits": [4, 6, 8],
  "group_size": 128,
  "sym": false
}
```

### 推理模块 `RRQLinear`

组合加载时（base + residual），引入 eager 推理模块 `RRQLinear`：

| buffer/attribute | 来源 | 含义 |
| --- | --- | --- |
| `qweight_0`, `scales_0`, `qzeros_0` | base model | base 平面 |
| `qweight_1` ... `qweight_3` | residual model | 残差平面 |
| `scales_1` ... `scales_3`, `qzeros_1` ... `qzeros_3` | residual model | 残差平面元数据 |
| `rrq_active_planes` | 运行时属性，非持久化 | 当前活跃平面数（1~4） |

`RRQLinear.forward(x)` 反量化并累加选定前缀平面，随后调用 `torch.nn.functional.linear`。这是正确性参考实现，不是性能 kernel。

### 存储和加载规则

- Base model 和 residual model 各自使用标准 safetensors shard 存储。
- 组合加载时，loader 验证：(1) base model 的 `bits=2, group_size, sym` 与 residual model 匹配；(2) residual 平面数 = 3；(3) shape 一致。任何不匹配均 fail fast。
- 仅加载 base model 时（不使用 residual），走现有标准 INT2 推理路径，不触发 RRQ 逻辑。
- 非 RRQ 后端（GGUF、MLX 等）遇到 `quant_method="auto-round-rrq"` 的 residual model 时必须显式报错。

## 集成步骤（分阶段）

### Phase 1：RTN 量化 + 分开存储

1. 添加 `auto_round.algorithms.quantization.rrq.config.RRQConfig`，注册 alias `rrq`，并从 `auto_round.__init__` 导出。`RRQConfig` 继承 `QuantizationConfig`，固定 `bits=2, data_type="int", act_bits=16`，新增字段 `num_residual_planes: int = 3`（第一版固定为 3，不可修改）。
2. 实现 `RRQRTNQuantizer(BaseQuantizer)`：
   - `quantize_block` 对每个 eligible linear 层执行 4 轮 RTN：
     - 轮 0：对原始权重 $W$ 做 INT2 RTN → 保存为 base（`layer.weight` 替换为 QDQ 结果，`layer.scale`/`layer.zp` 为 base 的 scale/zp）。
     - 轮 1~3：计算残差 $E_k = W_{\text{orig}} - A_k$，对 $E_k$ 做 INT2 RTN，将结果保存为 `layer.rrq_residual_k`（k=1,2,3），包含 `qweight_k`, `scales_k`, `qzeros_k`。
   - 注意：RTN 阶段不需要 calibration data（`need_calib = False`），不需要 `WrapperLinear`，直接对 weight tensor 做 QDQ 即可。
3. 导出：
   - `save_quantized(format="auto_round")` 输出 base model（标准 INT2，与现有 W2A16 完全相同）。
   - `save_quantized(format="auto_round:rrq")` 输出 residual model（仅含 `qweight_1/2/3`, `scales_1/2/3`, `qzeros_1/2/3`）。
   - 两个目录可独立存在，residual model 目录不包含 base 数据。
4. 添加 `RRQLinear` 推理模块 + 可复用 INT2 pack/unpack helper。`RRQLinear` 接受 base + residual 的 tensor 组合，`forward(x, active_planes)` 反量化并累加前缀平面后调用 `F.linear`。
5. 添加组合加载入口：用户指定 base model 路径 + residual model 路径，loader 将二者合并为 `RRQLinear` 模块。
6. 非 RRQ 后端（GGUF、MLX、AutoGPTQ 等）遇到 `quant_method="auto-round-rrq"` 时显式报错。

### Phase 2：从已有 base 生成 residual

1. 添加 `generate_rrq_residual(base_model_path, raw_model_path, output_dir, ...)` 工具函数。
2. 流程：加载 base model 的量化 tensor（`qweight_0`, `scales_0`, `qzeros_0`）→ 反量化得到 $\hat W_0$ → 加载原始 FP 权重 $W$ → 计算 $E_1 = W - \hat W_0$ → 对 $E_1$ 做 3 轮 RTN INT2 → 导出 residual model。
3. 不需要 calibration data，纯 RTN。
4. 校验：base model 的 `bits=2, group_size, sym` 必须与生成的 residual 一致。

### Phase 3：AutoRound OPT 调优

1. 实现 `RRQSignRoundQuantizer(BaseQuantizer)`：
   - 每轮复用 `sign_round` 的单层 sign-SGD 调优循环（`WrapperLinear` + `iters` 次迭代）。
   - 轮 0：以 $W$ 为目标，调优 $v_0, m_0, c_0$ → 冻结 base 平面。
   - 轮 $k$ (k=1,2,3)：以 $E_k = W - A_k$ 为目标（$A_k$ 为已冻结前缀的 dequant 结果），调优 $v_k, m_k, c_k$ → 冻结。
   - `forward` 中：$F.linear(x, A_k + QDQ_{\text{INT2}}(E_k; v_k, m_k, c_k), b)$，其中 $A_k$ 是已冻结常张量（不参与梯度），只有当前轮的 QDQ 参数可微。
2. 参数命名：`value_k`, `min_scale_k`, `max_scale_k`（k=0..3），复用现有 per-layer lr 分组。
3. 收敛后 `unwrapper` 写回 K 个平面的 code/scale/zp，导出方式同 Phase 1。
4. `RRQConfig` 新增 `iters`、`lr`、`minmax_lr` 等字段（默认值为 RTN 即 `iters=0`），Phase 1 用户不设这些字段等价于 RTN。

## 验证与验收标准

### Phase 1 验收

在确定性小 tensor 和 tiny linear module 上，单元测试必须证明：

- 配置拒绝所有非 `bits=2, data_type="int"` 的组合和 activation quantization。
- 4 轮 RTN 后，base model 的 `qweight`/`scales`/`qzeros` 与直接 `RTN W2A16` 输出一致（bit-exact）。
- Residual model 包含 3 个平面的 tensor，shape 与 packed code 范围正确。
- 对每个前缀 $p$（1~4），$W_{\text{deq}}$（base + 前 $p-1$ 个 residual 的 dequant 累加）在 dtype 容差内等于显式反量化结果。
- 残差范数单调递减：$\|E_1\|_2 \geq \|E_2\|_2 \geq \|E_3\|_2 \geq \|E_4\|_2$。
- Base model 可独立被现有 W2A16 runtime 加载和推理（不需要 RRQ 代码）。
- Residual model 独立加载时缺少 base 必须报错。
- `RRQLinear` 在 2/4/6/8 活跃比特下的输出，在 dtype 容差内等于 `F.linear(x, W_reconstructed_p, bias)`。
- 非 RRQ backend 对 `quant_method="auto-round-rrq"` 给出明确拒绝。

### Phase 2 验收

- `generate_rrq_residual` 从已有 base + FP 权重生成的 residual，与 Phase 1 完整流程产出的 residual 一致（bit-exact）。
- 生成的 residual model 可正确组合加载并推理。
- base model 的 `group_size`/`sym` 不匹配时 fail fast 报错。

### Phase 3 验收

- 顺序 4 轮 sign-SGD 调优确实生效：每轮迭代中该轮平面参数收到非零梯度。
- 各前缀重构范数 $\|W - A_k\|_2$ 单调递减。
- 4-bit 前缀（$A_2$）结果与"单平面直接 4-bit AutoRound"不同，证明残差分解确实带来收益。
- LM-eval 精度：RRQ 2+2 前缀 ≥ RTN W4（同 group_size/sym）；RRQ 2+2+2 前缀 ≥ RTN W6；RRQ 2+2+2+2 ≥ RTN W8。
- 初始 CUDA benchmark：eager RRQ 4-bit 与 W4A16 的延迟/内存对比报告。融合 kernel 出现前不设性能 SLA。

## 兼容性和失败行为

RRQ 为 opt-in，不改变既有算法的输出。checkpoint 通过 `quant_method` 和 `format_version` 自描述，通用 W2/W4 实现不能加载。加载不支持的硬件或请求不支持的位宽必须抛错，不能退化到任意精度。

第一版 `set_rrq_bits` 是模型级的，避免层意外使用不同精度。它只接受 2、4、6、8，并映射为 1、2、3、4 个活跃平面；该选择仅为运行时状态，不写入 checkpoint。

## 考虑过的替代方案

**四份独立模型文件。** 可直接使用现有 runtime，但用户需分别下载/存储 4 个模型，且 base 权重在每份中重复存储。RRQ 的 base + residual 分离设计在部署灵活性（可选加载）和存储效率（base 只存一次）之间取得平衡。

**将四个平面打包为单个 synthetic INT8 code。** 这会丢失独立 scale/zero-point 语义，且 2/4/6-bit 前缀选择需要 runtime kernel 做 bit-masking，增加实现复杂度，因此不采用。

**复用 SVDQuant residual 支持。** SVDQuant 表示低秩 FP 分支 + 一个残差分支（`residual_linear` + `lora_up/down`），不是多层可累加的量化平面，也不具备 RRQ 的 checkpoint/inference 契约。

**先实现融合 kernel。** 这会推迟正确性验证，并在语义尚未验证时固定 ABI。参考模块先建立正确性，runtime owner 可在相同 ABI 下添加融合实现。

**联合优化（单个 optimizer 同时优化 4 个平面）。** 理论上可做全局最优，但 4 个平面互相耦合导致 sign-SGD 的 per-element 搜索空间爆炸，收敛性难以保证，且工程复杂度显著更高。分轮顺序优化（每轮独立调优到最优）更稳定且与 AutoRound 现有机制天然兼容。

## 待评审问题

1. 第一版应采用建议的非对称（`sym=False`），还是对称（`sym=True`）以更贴近既有 W2A16 部署路径？
2. `auto_round:rrq` 是否为可接受的 residual 格式名，`auto-round-rrq` 是否为期望的 checkpoint `quant_method` 值？
3. 组合加载时是否允许 per-layer 选择不同 active planes（如某些 layer 用 8-bit、某些用 2-bit），还是 v1 只支持全局统一？
4. Residual model 中是否应保存 `base_model_hash` 字段以便验证 base 版本一致性？
5. Phase 2 的 `generate_rrq_residual` 命令是否应作为 CLI 子命令（`auto-round generate-residual ...`）还是独立 Python 脚本？