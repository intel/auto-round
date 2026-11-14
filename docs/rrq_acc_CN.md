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

### 为已有 INT2 基础模型生成残差

如果已经有量化好的 INT2 模型，`generate_rrq_residual` 可以在不重新量化基础
平面的前提下生成残差平面：

```python
from auto_round.export import generate_rrq_residual

generate_rrq_residual(
    "./rrq_out/<model>/base",
    raw_model="Qwen/Qwen3-27B",
    output_dir="./rrq_out/<model>/residual",
    group_size=128,
    sym=True,
    device="cuda",
    num_workers=32,
)
```

各层之间完全独立，因此可以完美并行。`num_workers` 控制展开到多少个进程，
`device` 决定量化运算在何处执行；单独写 `"cuda"` 会展开为全部可见 GPU，任务
按轮转方式分配。在 Qwen3.8-27B（400 层、8 卡、`num_workers=32`）上耗时
**5 分 10 秒**，而单进程 CPU 串行需要数小时。每个 worker 会绑定到一张 GPU；
打包（pack）仍在 CPU 上完成，因为打包 kernel 只支持 CPU。

```bash
# 不写 Python 脚本的多卡用法
python -c "
from auto_round.export import generate_rrq_residual
generate_rrq_residual('base/', raw_model='Qwen/Qwen3-27B',
                      output_dir='residual/', group_size=128, sym=True,
                      device='cuda', num_workers=32)"
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

## 基础平面质量：RTN vs. AutoRound

上面的数据全部使用 RTN（`--iters 0`）生成每一个平面。基础平面是其中最重要的
一个，因此值得只对它投入 sign-SGD 调优，再增量生成残差平面：

1. 用完整的 AutoRound 调优量化基础平面（`--iters 1000`），可跨 GPU 分片执行。
2. 把这个 INT2 基础模型交给 `generate_rrq_residual` 生成残差平面。

这正是 `generate_rrq_residual` 的用途。需要注意残差平面仍然使用 RTN（与一次性
量化路径相同的优化 RTN 缩放搜索），所以两种配置的唯一差异是基础平面的产生方式。

| 配置 | 基础平面 | 残差平面 | 基础成本 | 残差成本 |
|---|---|---|---|---|
| **A** | RTN（`iters=0`） | RTN | 含在总时间内 | — |
| **B** | AutoRound（`iters=1000`） | RTN | 6 小时 09 分（4 卡） | 5 分 10 秒（8 卡） |

配置 A 总耗时 26 分 21 秒；配置 B 因为要对基础平面做 sign-SGD 调优，额外多花 6
小时。

### 精度对比

lm-eval，完整测试集，bf16 前向，评测流程完全一致。

| 精度 | A：全部 RTN | B：AutoRound 基础 + RTN 残差 | **Δ** |
|---|---|---|---|
| 2-bit | 64.28 | **68.02** | **+3.74** |
| 4-bit | **71.67** | 72.02 | +0.35 |
| 6-bit | **72.19** | 72.03 | −0.16 |
| 8-bit | **71.85** | 71.79 | −0.06 |

配置 B 的详细数据：

| 精度 | PIQA | WinoGrande | HellaSwag | ARC-easy | ARC-challenge |
|---|---|---|---|---|---|
| 2-bit | 77.97 | 71.35 | 55.60 | 81.86 | 53.33 |
| 4-bit | 80.47 | 76.16 | 63.38 | 82.91 | 57.17 |
| 6-bit | 81.18 | 74.90 | 63.67 | 83.16 | 57.25 |
| 8-bit | 81.07 | 75.37 | 63.75 | 82.20 | 56.57 |

逐任务差异（B − A）：

| 精度 | PIQA | WinoGrande | HellaSwag | ARC-easy | ARC-challenge | 平均 |
|---|---|---|---|---|---|---|
| 2-bit | +3.38 | +2.76 | **+4.91** | +3.41 | +4.27 | **+3.74** |
| 4-bit | +0.38 | +0.23 | +0.25 | +0.38 | +0.51 | +0.35 |
| 6-bit | −0.21 | −1.58 | −0.02 | +0.67 | +0.34 | −0.16 |
| 8-bit | +0.22 | −0.24 | +0.17 | −0.46 | −0.00 | −0.06 |

2-bit 时增益覆盖了全部任务，且最弱的两个任务提升最多（HellaSwag +4.91、
ARC-challenge +4.27）。4-bit 以上差异很小且符号不一致，属于噪声。

配置 B 的精度阶梯：

| 区间 | 平均增益 |
|---|---|
| 2 → 4-bit | **+4.00**（决定性） |
| 4 → 6-bit | +0.01（已饱和） |
| 6 → 8-bit | −0.24（非单调） |

B 比 A 提前一级饱和：基础平面更好意味着第一个残差平面就补齐了几乎全部差距，
因此第 3、第 4 个平面已是纯噪声。

### 观察结论

1. **调优基础平面只在 2-bit 有回报**（+3.74）：多花 6 小时量化时间换约 3.7 个
   精度点。HellaSwag 提升最多（+4.91），其次是 ARC-challenge（+4.27）。
2. **4-bit 及以上两种配置等价**（差距 ≤ 0.35，即评测噪声）。只要有一个残差
   平面存在，基础平面就不再是瓶颈，昂贵的调优被浪费掉了：额外的残差平面会
   吸收基础平面留下的误差。
3. **调优后的 2-bit 基础可与 3-bit 混合精度 checkpoint 竞争。** 对比 B 的
   2-bit（68.02，13.44 GB）与 A 的 `residual_fraction=0.5`（68.04，
   19.95 GB）：平均分相同，而 B 少用 6.5 GB，代价是能力分布不同——
   HellaSwag 低 2.64，但 ARC-challenge 高 2.05。
4. **实践建议。** 如果目标部署精度是 2-bit，就用 AutoRound 调优基础平面再增量
   生成残差。如果目标是 4-bit 或更高，或者一份 checkpoint 需要服务多个精度，
   直接用 RTN（`--iters 0`）量化全部平面，半小时即可拿到全部四个工作点。

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

## 分阶段精度：prefill 用 6-bit，decode 用 2-bit

prefill 与 decode 的瓶颈正好相反，因此它们的最优权重位宽也不同：

- **prefill** 把整段 prompt 压过每一份权重，属于计算受限：矩阵乘占主导，权重
  宽一点几乎不增加成本。
- **decode** 每步只产出一个 token，属于带宽受限：每个权重只服务一个 token，
  权重位宽**就是**速度。

`set_rrq_phase_bits` 利用这一点，按输入长度选择平面数量，无需在模型中额外传参——
decode 喂入 `(batch, 1, hidden)`，prefill 喂入 `(batch, seq, hidden)`：

```python
from auto_round.inference import load_rrq_model, set_rrq_phase_bits

model = load_rrq_model(base_dir, residual_dir, active_bits=6, device="cuda")
set_rrq_phase_bits(model, prefill_bits=6, decode_bits=2)

model.generate(**inputs)  # prefill 走 6-bit，随后每个生成的 token 走 2-bit
```

在 Qwen3.8-27B 上实测（105 token 的 prompt，单张 80 GB 卡，参考 kernel）：

| 配置 | Prefill | Decode | Prefill 相对 6-bit | Decode 相对 2-bit |
|---|---|---|---|---|
| 2-bit | 1690 ms | 783 ms/tok | — | — |
| 6-bit | 2580 ms | 2397 ms/tok | 1.00× | — |
| 8-bit | 3369 ms | 3200 ms/tok | — | — |
| **prefill 6 / decode 2** | **2412 ms** | **779 ms/tok** | **0.94×** | **1.00×** |

prefill 阶段运行的就是与 uniform 6-bit 相同的三平面精度，因此 2412 ms 对
2580 ms 属于运行波动，不是真实收益。真正的收益在 decode：**779 ms/tok，与
uniform 2-bit 完全一致**，而 uniform 6-bit 是 2397 ms/tok。以生成 8 个 token
计，端到端**快 3.1 倍**，且 token 流完全一致——首个 token 与 8-bit 运行相同，
后续所有 token 与 2-bit 运行相同。

有两处细节让这个方案是精确的而非近似的：

- **按序列长度而非总 token 数判断。** 批量 decode 的形状是
  `(batch, 1, hidden)`，其总元素数等于 batch size，若按总数判断会被误判成
  prefill。因此阶段由 `x.shape[-2]` 决定。
- **缓存按平面数量分槽。** 两个阶段在每次请求中交替出现，单槽缓存会把下一个
  阶段要用的权重驱逐掉，导致两个阶段每步都重建。每层为每个平面数量保留一个槽，
  `required_rrq_cache_bytes` 也按两倍计算，使全有或全无的缓存判断依然准确。

需要注意这是参考实现，按需反量化，因此加速受打包权重带宽限制。真正完全发挥这个
方案的是融合 kernel：它能按序列长度选择平面数量而不物化权重，此时 decode 只读取
一个平面而不是三个。

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
