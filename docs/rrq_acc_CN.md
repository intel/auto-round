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

配置 A 就是 [量化命令](#量化命令) 那一节的单次命令，它会同时写出 `base/` 和
`residual/`：

```bash
auto_round --model Qwen/Qwen3.8-27B --scheme W2A16 \
    --format auto_round:rrq --alg_configs rrq --iters 0 \
    --ignore_layers mtp \
    --nsamples 128 --seqlen 2048 --low_gpu_mem_usage \
    --output_dir ./rrq_out/A
```

配置 B 分两步：先用 sign-SGD 调优基础平面，再由它派生残差平面。注意这里基础平面
用的是 `--format auto_round` 而不是 `auto_round:rrq`——RRQ 导出器会在一次流程中
同时写基础与残差平面，并且残差也用 RTN 生成，这不是 B 想要的结果。

```bash
# 第一步：用完整 AutoRound 调优生成基础平面（4 卡约 6 小时 09 分）
auto_round --model Qwen/Qwen3.8-27B --scheme W2A16 \
    --format auto_round --alg_configs auto_round --iters 1000 \
    --ignore_layers mtp \
    --nsamples 512 --seqlen 2048 --low_gpu_mem_usage \
    --output_dir ./rrq_out/B_base
```

```python
# 第二步：由该基础模型生成残差平面（8 卡约 5 分 10 秒）
from auto_round.export import generate_rrq_residual

generate_rrq_residual(
    "./rrq_out/B_base",
    raw_model="Qwen/Qwen3.8-27B",
    output_dir="./rrq_out/B_residual",
    group_size=128,
    sym=True,
    device="cuda",
    num_workers=32,
)
```

两种配置的加载方式：

```python
from auto_round import load_rrq_model

# A：基础与残差由同一次导出产生，是同级目录
model = load_rrq_model("./rrq_out/A/base", "./rrq_out/A/residual", active_bits=4)

# B：残差是单独生成的，路径与基础模型无关
model = load_rrq_model("./rrq_out/B_base", "./rrq_out/B_residual", active_bits=4)
```

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

### 推理基准上的验证

五项任务的平均分是一个浅层探针：2-bit 下 +3.74 的收益确实存在但幅度有限，而且
这些任务都不需要持续的多步推理。两个更难的基准在 prefill6/decode2 下做了同样
的对比，此时基础平面是唯一承载模型的部件：

| 基准 | A：全部 RTN | B：AutoRound 基础平面 | 差值 | 显著性 |
|---|---|---|---|---|
| GSM8K（n=1319） | 78.32% | **85.97%** | **+7.66** | p = 2.5e-07 |
| MATH-500（n=500） | 36.00% | **60.60%** | **+24.60** | p = 7.9e-26 |

在 MATH-500 上配对比较极为悬殊：B 解出了 130 道 A 解不出的题，而 A 只解出 7 道
B 解不出的题（McNemar z = +10.51）。剔除被截断的生成后，B 为 72.49%，A 为
46.04%。

收益随任务难度增长——小学算术上 +7.66，竞赛数学上 +24.60——这正是「基础平面设定
了一个残差平面只能部分弥补的上限」所预期的形态。五项浅层任务的平均值把这一点
分别在 GSM8K 上低估了约 2 倍、在 MATH 上低估得更多。

第二个效应体现在生成长度上。在 decode-2 下，调优后的基础平面在 MATH-500 上平均
生成 **2503** token，而 RTN 基础平面是 **4119**，截断率从 21.8% 降到 16.4%。
因此调优基础平面不仅在这个位宽下提升正确性，还能阻止模型退化成更长、更低效的
推理——在固定输出预算下，这与精度收益同样重要。

**关于 4-bit 结论的注意事项。** 下文「从 4-bit 起两种配置收敛」的结论只在五
项浅层任务、均匀位宽下测过。B 组在 GSM8K 与 MATH-500 上 prefill6/decode4 与
uniform 4-bit 的运行在产出结果前被中止，因此 **4-bit 等价性在推理基准上尚未
验证**。相比之下，2-bit 的结论现在已由三个独立基准支撑。

### 观察结论

1. **调优基础平面只在 2-bit 有回报**（+3.74）：多花 6 小时量化时间换约 3.7 个
   精度点。HellaSwag 提升最多（+4.91），其次是 ARC-challenge（+4.27）。在推理
   基准上方向相同且强得多：GSM8K +7.66，MATH-500 +24.60。
2. **4-bit 及以上两种配置等价**（差距 ≤ 0.35，即评测噪声）*在五项浅层任务、
   均匀位宽下*。只要有一个残差平面存在，基础平面就不再是瓶颈，昂贵的调优被
   浪费掉了：额外的残差平面会吸收基础平面留下的误差。这一点尚未在 GSM8K 或
   MATH-500 上确认，因此 4-bit 等价性应视为合理推测而非既成结论。
3. **调优后的 2-bit 基础可与 3-bit 混合精度 checkpoint 竞争。** 对比 B 的
   2-bit（68.02，13.44 GB）与 A 的 `residual_fraction=0.5`（68.04，
   19.95 GB）：平均分相同，而 B 少用 6.5 GB，代价是能力分布不同——
   HellaSwag 低 2.64，但 ARC-challenge 高 2.05。
4. **实践建议。** 如果目标部署精度是 2-bit，就用 AutoRound 调优基础平面再增量
   生成残差。如果目标是 4-bit 或更高，或者一份 checkpoint 需要服务多个精度，
   直接用 RTN（`--iters 0`）量化全部平面是更好的权衡，可省 6 小时，但需留意
   上述 4-bit 注意事项。

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

各拆分方式的 decode 吞吐（融合 kernel、batch 16、每条序列生成 96 token、单张
80 GB 卡）：

| 配置 | Decode | 相对 2-bit |
|---|---|---|
| 2-bit | **121.2 tok/s** | 1.00× |
| prefill 6 / decode 2 | **110.1 tok/s** | 0.91× |
| **prefill 6 / decode 2+2×50%，仅 MLP（96/400）** | **92.6 tok/s** | **0.76×** |
| prefill 6 / decode 2+2×50%，全部层（200/400） | 83.8 tok/s | 0.69× |
| prefill 6 / decode 2+2×50%，MLP 100%（192/400） | 78.7 tok/s | 0.65× |
| 4-bit | 68.8 tok/s | 0.57× |
| prefill 6 / decode 4 | 68.2 tok/s | 0.56× |
| 6-bit | 48.8 tok/s | 0.40× |

batch 必须不超过 `_RRQ_FUSED_MAX_BATCH`（16）。一旦超过，`RRQLinear` 会完全跳过融合
kernel 并退化为 dequant + matmul：2-bit decode 从 batch 16 的 113 tok/s 掉到 batch
24 的 **30 tok/s**。超过这个阈值的基准测到的是参考路径，而不是 kernel。

### GSM8K 精度

这个配置要回答的精度问题是：*6-bit prefill 能否挽回 2-bit decode 损失的精度？*
GSM8K 是一个合适的探针，因为推理过程发生在生成阶段，而生成阶段走的是 decode 的
位宽。

完整测试集（1319 题），贪心解码，按最终答案精确匹配。生成配置：chat template
且**关闭 thinking**、**`max_new_tokens=2048`**，因此每条思维链都能自然结束。
这对推理模型很关键——若 thinking 开启且输出预算较短，模型会把整个预算消耗在
`thinking` 块里而永远不输出答案，从而让所有配置的数字系统性偏低（见下文的
*生成配置的敏感性*）。

| 配置 | 正确数 | GSM8K 精度 | 95% 置信区间 | 截断数 |
|---|---|---|---|---|
| 全部 2-bit | 955 / 1319 | 72.40% | ±2.41 | 39 |
| **prefill 6 / decode 2** | **1033 / 1319** | **78.32%** | ±2.22 | 73 |
| prefill 6 / decode 2+2×50%，仅 MLP（96/400） | 1206 / 1319 | 91.43% | ±1.51 | 35 |
| **prefill 6 / decode 2+2×50%，全部层（200/400）** | **1213 / 1319** | **91.96%** | ±1.47 | 38 |
| prefill 6 / decode 2+2×50%，MLP 100%（192/400） | 1213 / 1319 | 91.96% | ±1.47 | 43 |
| 全部 4-bit | 1245 / 1319 | 94.39% | ±1.24 | 15 |
| **prefill 6 / decode 4** | **1261 / 1319** | **95.60%** | ±1.11 | 10 |
| 全部 6-bit | 1247 / 1319 | 94.54% | ±1.23 | 10 |
| bf16（未量化） | 1249 / 1319 | 94.69% | ±1.21 | — |

以未量化行为参照，所有 decode 达到 4-bit 及以上的配置在统计上都无法与它区分，
而比它更窄的配置则都可以：

| 对比 | 差值 | 是否显著 |
|---|---|---|
| prefill 6 / decode 4 - bf16 | +0.91 | 否（p = 0.28） |
| 全部 6-bit - bf16 | -0.15 | 否（p = 0.86） |
| 全部 4-bit - bf16 | -0.30 | 否（p = 0.73） |
| MLP 100%（2.96-bit）- bf16 | -2.73 | 是（p = 0.005） |
| 仅 MLP（2.48-bit）- bf16 | -3.26 | 是（p = 0.001） |

因此无损阈值是 **decode 4-bit** 这一档，而不是某个特定配置——prefill6/decode4、
uniform 4-bit 与 uniform 6-bit 都落在未量化的分数上。三者之中 prefill6/decode4
仍然是推荐工作点，理由纯粹是速度（同等精度下 68.2 tok/s 对 uniform 6-bit 的
48.8 tok/s），这使得 uniform 6-bit 被完全支配。高于 bf16 的那 +0.91 属于噪声；
在 N = 1319 下可分辨的最小差异约为 1.7 个点。

decode 位宽是主导因素。从 2-bit 到 4-bit decode 值 **+21.99 分**；从 4-bit 到
6-bit decode 只值 **+0.15**，等于没有。GSM8K 的算术几乎全部发生在 decode 阶段，
因此精度在 decode 达到 4-bit 后就饱和了。

混合 decode 的几个工作点是本轮的重点，而且 **残差平面花在哪里，比花在多少层上更
重要**。下面每一行都把单层从 2-bit 加宽到 4-bit，区别只在于如何分配：

| 配置 | 4-bit 层数 | 有效 decode | GSM8K | Decode |
|---|---|---|---|---|
| prefill 6 / decode 2 | 0 | 2.00-bit | 78.32% | 0.91× |
| **2+2×50%，仅 MLP** | **96** | **2.48-bit** | **91.43%** | **0.76×** |
| 2+2×50%，全部层 | 200 | 2.96-bit | 91.96% | 0.69× |
| 2+2×50%，MLP 100% | 192 | 2.96-bit | 91.96% | 0.65× |
| 全部 4-bit | 400 | 4.00-bit | 94.39% | 0.57× |

两个结论：

- **仅 MLP 是本轮曲线上的最佳权衡。** 把残差平面只花在 96 个 MLP 投影上就有
  91.43%，只比 200 层方案低 **0.53 分**，但 decode 快 **10%**（0.76× 对 0.69×）。
  attention 投影（`linear_attn.*`、`self_attn.*`）正是带宽预算最紧张的部分，
  让它们保持 2-bit 几乎是免费的。
- **等预算对照进一步确认了这个排序。** 把全部 192 个 MLP 层都升级（2.96-bit，与
  200 层随机抽样占用相同的平面数）得到完全相同的 91.96%，但更慢（0.65× 对
  0.69×）。跨模块族随机抽样的速度优势，正是因为它升级的 attention 层更少。

与两侧的均匀配置相比，仅 MLP 这一点用 **0.15× 速度换来 +13.11 分**，而再爬到
全部 4-bit 还要再花 0.19× 却只多 **+2.96 分**。这正是曲线的拐点。

| 对比 | 差值 | 是否显著 |
|---|---|---|
| 全部 4-bit − 全部 2-bit | +21.99 | 是 |
| 仅 MLP 混合 − prefill6/decode2 | +13.11 | 是 |
| 全部 4-bit − 仅 MLP 混合 | +2.96 | 是 |
| 全部层混合 − 仅 MLP 混合 | +0.53 | 否 |
| 全部 6-bit − 全部 4-bit | +0.15 | 否 |
| prefill6/decode4 − 全部 4-bit | +1.21 | 否（p ≈ 0.15） |

**实践建议——按 decode 预算选工作点：**

- decode 必须走 2-bit：`set_rrq_phase_bits(model, 6, 2)`，0.91× 换 +5.91 分，
  无额外存储。
- 还能再让出约 0.15× decode 带宽：**用仅 MLP 的混合 decode**，
  `set_rrq_random_residual(model, fraction=0.5, seed=0, high_bits=4, low_bits=2,
  prefill_bits=6, only="mlp.")`。它以 0.76× 达到 91.43%，是本表中单位 decode
  带宽精度最高的一点，且所有 attention 投影仍只读一个 INT2 平面。
- 需要最后那约 3 分：全部 4-bit（94.39%，0.57×）。它与 prefill6/decode4
  （95.60%）在统计上相同、decode 速度相同，而 prefill 更便宜，因此那里无需拆分。
- 全部 6-bit 对 GSM8K 不值得：94.54% 落在全部 4-bit 的噪声区间内，而 decode
  速度只有 2-bit 的 0.40×。

机制上，prefill 只影响第一个 token，而 GSM8K 的大部分计算发生在 decode 阶段——
算术链是一个 token 一个 token 生成的。更好的 prefill 能让模型有个正确的起点，这
就是那 +5.91 分的来源；但它无法补回低精度 decode 权重在后续丢掉的信息。这也解释
了为什么 decode 一旦到 4-bit，prefill 加宽就失去作用：decode 权重本身已经不再是
瓶颈。

#### 生成配置的敏感性

本文档此前报告了低得多的数字（42.00 / 53.45 / 63.23%）——那是同一模型、同一测试
集，但生成配置有缺陷。早期测量存在两个问题：没有走 chat 模板（裸 prompt），且输出
预算只有 320 token。在模型 reasoning 模式开启时，仅 ` thinking` 就吃掉了预算：2-bit
的 660 条完成中有 22 条完全解析不出数字，其余也在解题中途被截断。两个改动缺一不可
——关闭 thinking，以及 2048 token 的预算。对照如下：

| 版本 | thinking | max_new_tokens | 2-bit | p6d2 | 6-bit |
|---|---|---|---|---|---|
| 早期（无效） | 开 | 320 | 42.00% | 53.45% | 63.23% |
| **当前** | **关** | **2048** | **72.40%** | **78.32%** | **94.54%** |

残余截断仍有一点影响，主要发生在模型更容易啰嗦的 2-bit：剔除截断样本后分别为
73.98% / 82.58% / 95.19%。排序与各差距的大小都不变，因此实践结论不是输出预算
造成的假象。

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

### MATH-500 精度

GSM8K 在 decode 达到 4-bit 后就饱和了，因此它无法说明 prefill 精度是否起了作用。
要区分这两个阶段，需要一个更难的基准，而且这个基准的 *output_type* 必须真的会走
decode。这就排除了 MMLU：lm-eval 把它当作 `loglikelihood` 评分，每次 forward 都是
完整的 context 加 continuation，`seq_len > prefill_threshold`。在本方案的分阶段
设定下，MMLU 只会在 **prefill** 位宽下运行，完全无法用于研究 decode 精度。

MATH-500 属于 `generate_until`：prompt 只 prefill 一次，之后每步喂入一个 token。
在 MATH-500 的 prompt 上对 `RRQLinear._resolve_forward_planes` 打点统计，确认这个
划分是真实的而非名义上的——在 prefill 6 / decode 4 下，**98.4% 的 forward 是单
token 的 decode**（25200 次 decode 走 2 个平面，400 次 prefill 走 3 个平面）。

完整测试集（500 题），贪心解码，8192 token 预算，对最终 `\boxed{}` 做轻度归一化
后精确匹配。预算很关键：在 4096 token 下有 7.8% 的题目在解题中途被截断，且它们
全部计为错误，这把分数低估了 5 个点。被截断的题目随后以更大的预算重跑，并按题目
id 合并。

| 配置 | 正确数 | MATH-500 精度 | 95% CI | 截断 |
|---|---|---|---|---|
| bf16（未量化） | 439 / 500 | 87.80% | ±2.87 | 3 |
| **prefill 6 / decode 4** | **436 / 500** | **87.20%** | ±2.93 | 6 |
| uniform 4-bit（prefill 4 / decode 4） | 424 / 500 | 84.80% | ±3.15 | 13 |
| prefill 6 / decode 2 | 180 / 500 | 36.00% | ±4.21 | 109 |

两个结论，而且性质不同。

**decode 用 4-bit 在这里同样是无损的。** prefill6/decode4 比未量化模型低 0.60 个
点，McNemar p = 0.44——无法区分。这 **不是** 因为基准太弱看不出差异：
prefill6/decode2 直接崩到 36.00%，掉了 51.8 个点，p < 0.0001，其中 decode-4 解出
而 decode-2 解不出的题有 256 道，反方向为零。MATH-500 对 decode 精度的区分力非常
强，而 4-bit 依然落在未量化的分数上。

**prefill 用 6-bit 确实有收益，但不是推理质量。** 在 decode 固定为 4-bit 的前提下，
把 prefill 从 4-bit 提到 6-bit 值 **+2.40 个点，McNemar z = +2.68，p = 0.0073**。
拆解这个增益可以看到它来自哪里：

| 来源 | 贡献 |
|---|---|
| 两种配置都跑完的 484 道题 | **+0.83** 点（z = 1.26，p = 0.21） |
| 4-bit prefill 多截断了 7 道题 | **+1.4** 点 |
| 合计 | +2.40 点 |

在两种配置都能完整跑完的题上，两者在统计上是一样的。可测量的效应落在 **输出长度**
上，而不是答案正确性：4-bit prefill 会让模型变啰嗦，因而更容易撞上输出上限。

| 配置 | 平均 token | p90 token |
|---|---|---|
| bf16 | 1292 | 2802 |
| prefill 6 / decode 4 | 1400 | 3084 |
| prefill 4 / decode 4 | 1530 | 3733 |

因此对 prefill 精度的诚实表述是 **更短、更可靠的输出**——需要服务的 token 更少、
因预算而被丢弃的答案也更少——而不是更强的算术能力。这仍然是真实的端到端收益，因为
服务延迟和截断损失都改善了，但不应被描述成推理能力的提升。

按难度拆解，prefill-4 到 prefill-6 的全部差异都集中在最难的一档，较容易的档位则
完全持平：

| 难度 | n | prefill 6 / decode 4 | prefill 4 / decode 4 | 差值 |
|---|---|---|---|---|
| 1 | 42 | 92.86% | 92.86% | +0.00 |
| 2 | 90 | 90.00% | 88.89% | +1.11 |
| 3 | 104 | 88.46% | 88.46% | +0.00 |
| 4 | 124 | 88.71% | 88.71% | +0.00 |
| 5 | 124 | 84.68% | 82.26% | +2.42 |

复现方式：针对 `HuggingFaceH4/MATH-500` 使用 `--mode p6d4` / `p4d4` / `p6d2`；
基线行以 bfloat16 加载未量化 checkpoint，并使用相同的 prompt、判分器和预算。

## 融合 decode kernel

decode 阶段每个序列只喂一个 token，因此每次 forward 都是矩阵-向量乘：要搬运整个
权重，却几乎没有算术。通用路径先把打包的 INT2 反量化成 bf16，再跑 GEMM，写回并
重读的字节数比打包权重多 8 倍（2 字节/参数 对 0.25）。在单 token 场景下，这次
物化占了**单层时间的 65%**。

`auto_round/inference/rrq_triton.py` 提供了一个不物化权重的 Triton kernel：每个
program 负责一块输出特征，在寄存器里解包 INT2 并直接累加 `x @ W`。它在 CUDA 上
且 `seq_len == 1` 时自动生效，其他情况回退到反量化路径，因此 prefill 和非 CUDA
平台仍然正常工作。

在 Qwen3.8-27B 上实测（16 token 贪心解码，单张 80 GB 卡）：

| 配置 | 反量化路径 | 融合 kernel | 加速比 |
|---|---|---|---|
| 全部 2-bit | 1708 ms/tok | **253 ms/tok** | 6.74× |
| 全部 6-bit | 5242 ms/tok | **359 ms/tok** | 14.63× |
| prefill 6 / decode 2 | 1708 ms/tok | **254 ms/tok** | 6.74× |

6-bit 收益更大，因为它原本要物化三个平面。注意分阶段配置的 decode 速度与全部
2-bit 相同（254 对 253 ms/tok），这正是它的意义所在：decode 支付 2-bit 带宽，
同时保留 6-bit prefill。

数值行为，以下为对 fp32 真值（保留 float32 的反量化权重做 `x @ W`）的最大相对
误差，在 24 层上实测：

| 平面数 | kernel 与 torch 路径 | **kernel 与 fp32 真值** | **torch 路径与 fp32 真值** |
|---|---|---|---|
| 2-bit (1) | 1.98e-04 | **2.51e-03** | 2.51e-03 |
| 4-bit (2) | 4.36e-03 | **2.44e-03** | 2.88e-03 |
| 6-bit (3) | 4.47e-03 | **2.44e-03** | 3.16e-03 |

也就是说，kernel **与反量化路径精度相当，在 2 个及以上平面时还略微更准**。两个
效应解释了这些数字：

- 反量化路径在 GEMM 前把累加的权重舍入成 bf16，相对 fp32 有约 2.5e-03 的代价。
  kernel 全程保持 float32，因此在 2、3 平面时避开了这次舍入，更接近真值。
- kernel 与 torch 之间的差异（最高 4.5e-03）主要来自 torch 一侧的这次 bf16
  舍入，而非 kernel 误差。在 2-bit（不涉及累加顺序）时，两条路径与 fp32 真值的
  距离完全相同。

这也意味着融合路径不会以不同于参考路径的方式改变模型精度：两者都是对同一
float32 结果的 bf16 级近似。调用 `set_rrq_fused_decode(model, False)` 可强制走
反量化路径，便于在观察到差异时做二分定位。

## 说明与限制

- **MTP 层。** Qwen3.8 带有 Multi-Token Prediction 头层（`mtp.*`）。RRQ 仅
  支持 2-bit 平面，而这些层在 2-bit 下不受支持，因此通过
  `--ignore_layers mtp` 排除并保持 FP16。若不排除，base 产物会包含打包的
  `mtp.*` 层，而 residual 产物不包含，`load_rrq_model` 会正确拒绝加载这对
  不匹配的产物。
- **bf16 forward。** RRQ 以 float32 累加各平面，再执行 bf16 矩阵乘。以
  float32 加载 base 时 27B 模型需要约 108 GB，单张 80 GB 卡会 OOM，因此 base
  以 bfloat16 加载（仅在 bf16 不可用时退化为 float16）。
- **全精度基线。** bf16 参考线已在 MATH-500 上测出（87.80%，见 *MATH-500 精度*），
  但尚未在 GSM8K 上测，因此上面的 GSM8K 表格仍只报告 RRQ 各工作点之间的相对比较。
  确立「decode 4-bit 无损」这一结论的是 MATH-500 的结果；GSM8K 的排序与之一致，
  但它本身并没有锚定到 bf16 行。
- **单一模型。** 上述数据来自一个 27B 模型和一套校准配置。绝对值会因模型和
  校准预算而异；可预期的是定性的形态（第一个残差平面增益最大、之后饱和）。
