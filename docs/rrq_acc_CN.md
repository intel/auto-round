# RRQ 在 Qwen3.8-27B 上的精度与成本（实验性）

> **注意**：RRQ 为实验性功能，API、格式和行为可能在未来版本中发生变化。

递归残差量化（RRQ）将权重打包为 K 个顺序的 INT2 平面——1 个基础平面加 K-1
个残差平面。因此单个 checkpoint 即可支持 2/4/6/8-bit 仅权重量化精度，在加载时
切换而无需重新量化，并额外支持逐层混合精度。

本文报告 **Qwen3.8-27B** 按 **2+2+2+2**（1 个基础 + 3 个残差，共 4 个 2-bit
平面）量化后的实测精度、存储占用与量化成本。

## 配置

| 项目 | 取值 |
|---|---|
| 模型 | `Qwen/Qwen3-27B`（Qwen3.8，64 个解码层，400 个打包线性模块） |
| 量化方案 | `W2A16`（仅权重 INT2，激活 16-bit） |
| 平面数 | 4 × 2-bit（1 个基础 + 3 个残差） |
| Group size / sym | 128 / 对称 |
| 算法 | RRQ **RTN**（`--iters 0`，逐平面零样本） |
| 校准 | `nsamples=128`，`seqlen=2048`，AutoRound 默认数据集 |
| 排除层 | `mtp.*` 保持 FP16（`--ignore_layers mtp`）；MTP 层不支持 2-bit |
| 硬件 | 量化用 1 张 GPU（80 GB 级）；评测每个位宽各用 1 张 GPU |
| 评测 | lm-eval，完整测试集，bf16 forward |

### 量化命令

```bash
auto_round --model Qwen/Qwen3-27B --scheme W2A16 \
    --format auto_round:rrq --alg_configs rrq --iters 0 \
    --ignore_layers mtp \
    --nsamples 128 --seqlen 2048 --low_gpu_mem_usage \
    --output_dir ./rrq_out
```

该命令产出两个产物：

```
rrq_out/<model>/base/       # 标准 INT2 W2A16 基础平面（平面 0）
rrq_out/<model>/residual/   # auto_round:rrq 产物（残差平面 1..3）
```

### 评测命令

```python
from transformers import AutoTokenizer
from auto_round.inference import load_rrq_model
from auto_round.eval.evaluation import simple_evaluate_user_model

model = load_rrq_model("./rrq_out/<model>/base", "./rrq_out/<model>/residual", active_bits=4, device="cuda")
tokenizer = AutoTokenizer.from_pretrained("./rrq_out/<model>/base")

result = simple_evaluate_user_model(
    model,
    tokenizer,
    batch_size=8,
    tasks=["piqa", "winogrande", "hellaswag", "arc_easy", "arc_challenge"],
)
```

## 量化成本

| 项目 | 取值 |
|---|---|
| 总耗时 | **26 分 21 秒** |
| 单层耗时 | 约 24.7 秒 × 64 层 |
| 峰值显存 | 10.74 GB |
| 峰值内存 | 78.35 GB |

作为对比，相同配置改用符号梯度下降调优（`--iters 200`）时每层约需 11 分钟
（合计约 11 小时），即比本文使用的 RTN 路径慢约 **27 倍**。

## 存储占用

| 产物 | 大小 |
|---|---|
| `base/`（平面 0） | 13.44 GB |
| `residual/`（平面 1–3） | 19.53 GB |
| **完整 checkpoint（全部 4 个平面）** | **32.97 GB** |

实际只需加载目标精度所对应的平面：

| 精度 | 加载平面数 | 占用 | 相对 FP16（约 52 GiB） |
|---|---|---|---|
| 2-bit | 1（基础） | 13.44 GB | 24.1% |
| 4-bit | 2 | 19.95 GB | 35.7% |
| 6-bit | 3 | 26.46 GB | 47.4% |
| 8-bit | 4 | 32.97 GB | 59.0% |

核心优势是单个 checkpoint 覆盖全部四个工作点：2-bit 仅需 13.4 GB，比 FP16
减少 76%，同时更高精度仍可从同一产物中获得。

## 精度

lm-eval，完整测试集，bf16 forward。

| 精度 | 平面数 | PIQA | WinoGrande | HellaSwag | ARC-easy | ARC-challenge | **平均** |
|---|---|---|---|---|---|---|---|
| 2-bit | 1 | 74.59 | 68.59 | 50.69 | 78.45 | 49.06 | **64.28** |
| 50% 混合<sup>†</sup> | 1.5（平均） | 77.37 | 71.74 | 58.24 | 81.57 | 51.28 | **68.04** |
| 4-bit | 2 | 80.09 | 75.93 | 63.13 | 82.53 | 56.66 | **71.67** |
| 6-bit | 3 | 81.39 | 76.48 | 63.69 | 82.49 | 56.91 | **72.19** |
| 8-bit | 4 | 80.85 | 75.61 | 63.58 | 82.66 | 56.57 | **71.85** |

<sup>†</sup> 400 层中 200 层为 4-bit（基础 + 1 个残差），另外 200 层为 2-bit
（仅基础）。通过 `residual_fraction=0.5`、`residual_seed=0` 选定，可复现。
有效平均精度约 3-bit。

### 观察结论

1. **2 → 4-bit 是决定性跃升**（平均 +7.39）。单个 INT2 平面携带的信息量远不
   够，第一个残差平面的增益在所有平面中最大。HellaSwag 提升最多（+12.45），
   其次是 ARC-challenge（+7.59）。
2. **4/6/8-bit 实际等价**（差距小于 0.6 分，属评测噪声）。在 RTN 下第 3、第 4
   个残差平面已饱和：6-bit 比 4-bit 高 0.52，而 8-bit 反而比 6-bit 低 0.34，
   因此趋势**并非单调**。
3. **混合精度落在线性插值上**。2-bit（64.28）与 4-bit（71.67）的理论中点为
   67.98，实测 68.04。这说明各层的精度贡献近似独立且均匀，因此
   `residual_fraction` 是一个可预测的调节旋钮：存储与精度都随比例线性变化。

## 混合精度

`load_rrq_model` 通过三个参数支持逐层随机精度，并附带一个确定性辅助函数：

```python
from auto_round.inference import load_rrq_model
from auto_round.inference.rrq_linear import set_rrq_bits, set_rrq_random_residual

# 统一精度
model = load_rrq_model(base_dir, residual_dir, active_bits=4)

# 随机混合：一半层 4-bit，一半层 2-bit（可复现）
model = load_rrq_model(
    base_dir,
    residual_dir,
    residual_fraction=0.5,
    residual_seed=0,
    residual_high_bits=4,
    residual_low_bits=2,
)

# 或加载后再切换
set_rrq_bits(model, 6)
set_rrq_random_residual(model, fraction=0.25, seed=1, high_bits=4, low_bits=2)
```

## 推理性能

`RRQLinear` 常驻内存的是打包后的 INT2 平面，并在每次 forward 时反量化。这正是单个
checkpoint 能在 2/4/6/8-bit 之间切换的原因，但代价是慢：每层都要为每个 token 重建
权重张量。

调用一次 `materialize_rrq(model)` 即可把每个 RRQ 层转换为稠密 `nn.Linear`，把反量化
成本只付一次，之后便是普通的稠密矩阵乘。代价是内存与灵活性：稠密副本需要约每参数
2 字节（27B 模型 bf16 约 54 GB），且精度会冻结在调用时生效的那一档。

```python
from auto_round import load_rrq_model, materialize_rrq

model = load_rrq_model(base_dir, residual_dir, active_bits=4, device="cuda")
# ... 需要时先验证 2/4/6/8-bit 的精度 ...
materialize_rrq(model)  # 冻结当前精度，转为稠密推理
```

在 Qwen3.8-27B 上实测（8-bit，单张 80 GB 卡）：

| 路径 | 速度 | 峰值显存 |
|---|---|---|
| 仅 `load_rrq_model`（每次 forward 反量化） | 0.31 tok/s | 30.9 GB |
| 调用 `materialize_rrq` 后 | **15.08 tok/s** | 51.1 GB |

即 **48 倍**加速，且数值等价（同样以 float32 累加，再舍入到层的 dtype）。

此外，当整个模型能放下时，`load_rrq_model` 会自动缓存反量化后的权重：仅当**每一层**
的稠密副本能装进设备空闲显存时才开启缓存，否则会打印所需的 GiB 数。部分缓存没有
帮助，因为每次 forward 都会经过所有层，未缓存的层会成为瓶颈。

## 说明与限制

- **MTP 层。** Qwen3.8 带有 Multi-Token Prediction 头层（`mtp.*`）。RRQ 仅
  支持 2-bit 平面，而这些层在 2-bit 下不受支持，因此通过
  `--ignore_layers mtp` 排除并保持 FP16。若不排除，base 产物会包含打包的
  `mtp.*` 层，而 residual 产物不包含，`load_rrq_model` 会正确拒绝加载这对
  不匹配的产物。
- **bf16 forward。** RRQ 以 float32 累加各平面，再执行 bf16 矩阵乘。以
  float32 加载 base 时 27B 模型需要约 108 GB，单张 80 GB 卡会 OOM，因此 base
  以 bfloat16 加载（仅在 bf16 不可用时退化为 float16）。
- **全精度基线。** 上表未包含 FP16 参考行，仅报告 RRQ 各工作点之间的相对
  比较；相对 FP 的精度保留率将在测出后补充。
- **单一模型。** 上述数据来自一个 27B 模型和一套校准配置。绝对值会因模型和
  校准预算而异；可预期的是定性的形态（第一个残差平面增益最大、之后饱和）。
