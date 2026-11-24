# W4A4 量化方案评测报告

**模型：** Llama-3.1-8B-Instruct
**评测日期：** 2026-11-19
**评测状态：** 12 种方案全部完成

## 摘要

本实验评估了 12 种 group size 为 16 的 W4A4 量化方案，任务包括 GSM8K、MMLU、HellaSwag 和 PIQA。`EXP-UINT4-ZP-BF16` 的四项平均分最高，为 **75.14%**；同时在 GSM8K flexible/strict（75.97% / 75.44%）、MMLU（66.00%）以及 PIQA acc/acc_norm（79.60% / 80.58%）上领先。`EXP-UINT4-ZP-FP16` 的 HellaSwag acc 最高，为 59.16%；`EXP-UINT4-ZP-UE5M3` 的 HellaSwag acc_norm 最高，为 78.72%。

四项平均分适合作为综合参考，不能替代针对具体任务的方案选择。所有结果均来自单次评测，应视为点估计。

## 实验方法

- **模型：** Llama-3.1-8B-Instruct，模型快照 `0e9e39f249a16976918f6564b8830bc894c89659`。
- **量化：** 权重 4 bit、激活 4 bit（W4A4），group size 16，round-to-nearest（RTN），以 fake quant 方式评测。
- **校准：** RTN 路径不使用校准数据。命令保留了 128 个样本的数据集参数，但该 RTN 配置不会用它拟合量化参数。
- **随机种子：** 42。
- **似然任务：** MMLU acc、HellaSwag acc 与 acc_norm、PIQA acc 与 acc_norm；评测 dtype 为 BF16。前 8 个方案的 batch size 为 32，后 4 个方案为 16。
- **生成任务：** GSM8K，5-shot greedy 解码，BF16 评测 dtype，batch size 8，生成长度上限 1024 tokens。
- **结果口径：** 下表均为百分比。四项平均为 GSM8K strict-match、MMLU acc、HellaSwag acc_norm 和 PIQA acc_norm 的算术平均。

## Scheme 定义

下列方案均以 4 bit 量化权重和激活，group size 为 16；各方案通过码值表示和 scale 类型区分。

| Scheme                 | 权重/激活码值表示                                          | 每组 scale 与偏移                                       |
| ---------------------- | ---------------------------------------------------------- | ------------------------------------------------------- |
| `EXP-FP4-UE5M3`      | 有符号 E2M1 FP4；可表示的幅值为 0、0.5、1、1.5、2、3、4、6 | 非负 UE5M3 scale；无 zero-point                         |
| `EXP-FP4-BF16`       | 有符号 E2M1 FP4                                            | 非负 BF16 scale；无 zero-point                          |
| `EXP-FP4-FP16`       | 有符号 E2M1 FP4                                            | 非负 FP16 scale；无 zero-point                          |
| `EXP-INT4-E4M3`      | 有符号整数码 `[-8, 7]`                                   | 有符号 E4M3 scale；scale 符号决定哪一侧使用较大幅值端点 |
| `EXP-INT4-UE5M3`     | 对称有符号整数码 `[-7, 7]`                               | 非负 UE5M3 scale                                        |
| `EXP-INT4-UBF16`     | 对称有符号整数码 `[-7, 7]`                               | 非负 BF16 scale                                         |
| `EXP-INT4-UFP16`     | 对称有符号整数码 `[-7, 7]`                               | 非负 FP16 scale                                         |
| `EXP-INT4-BF16`      | 有符号整数码 `[-8, 7]`                                   | 有符号 BF16 scale；scale 符号决定哪一侧使用较大幅值端点 |
| `EXP-INT4-FP16`      | 有符号整数码 `[-8, 7]`                                   | 有符号 FP16 scale；scale 符号决定哪一侧使用较大幅值端点 |
| `EXP-UINT4-ZP-UE5M3` | 无符号整数码 `[0, 15]`                                   | 非负 UE5M3 scale 与每组 zero-point                      |
| `EXP-UINT4-ZP-BF16`  | 无符号整数码 `[0, 15]`                                   | 非负 BF16 scale 与每组 zero-point                       |
| `EXP-UINT4-ZP-FP16`  | 无符号整数码 `[0, 15]`                                   | 非负 FP16 scale 与每组 zero-point                       |

带 zero-point 的方案以无符号存储码围绕每组 zero-point 表示数值，反量化后的值仍可为负数。`UE5M3`、`UBF16`、`UFP16` 中的 `U` 表示 scale 非负，不表示浮点编码为无符号格式。

## 精度结果

| Scheme                 | GSM8K flexible / strict | MMLU acc | HellaSwag acc / acc_norm | PIQA acc / acc_norm |        四项平均 |
| ---------------------- | ----------------------: | -------: | -----------------------: | ------------------: | --------------: |
| `EXP-FP4-UE5M3`      |           70.36 / 63.84 |    63.67 |            58.43 / 77.49 |       77.86 / 79.00 |           71.00 |
| `EXP-INT4-E4M3`      |           65.20 / 65.13 |    62.04 |            57.66 / 76.71 |       77.75 / 78.56 |           70.61 |
| `EXP-INT4-UE5M3`     |           67.40 / 64.97 |    64.39 |            58.56 / 78.32 |       78.45 / 79.27 |           71.74 |
| `EXP-UINT4-ZP-UE5M3` |           73.31 / 71.57 |    65.48 |            58.80 / 78.72 |       79.22 / 80.20 |           73.99 |
| `EXP-FP4-BF16`       |           69.37 / 63.46 |    64.00 |            58.50 / 78.22 |       78.29 / 78.89 |           71.14 |
| `EXP-FP4-FP16`       |           69.67 / 62.32 |    63.78 |            58.46 / 78.05 |       78.35 / 79.76 |           70.98 |
| `EXP-INT4-UBF16`     |           68.99 / 67.10 |    64.76 |            58.54 / 78.34 |       78.62 / 79.38 |           72.40 |
| `EXP-INT4-UFP16`     |           70.28 / 68.16 |    64.29 |            58.49 / 78.47 |       78.56 / 79.54 |           72.62 |
| `EXP-UINT4-ZP-BF16`  |           75.97 / 75.44 |    66.00 |            58.82 / 78.52 |       79.60 / 80.58 | **75.14** |
| `EXP-UINT4-ZP-FP16`  |           74.75 / 73.54 |    65.95 |  **59.16** / 78.54 |       78.89 / 79.92 |           74.49 |
| `EXP-INT4-BF16`      |           72.25 / 70.05 |    65.04 |            58.78 / 78.52 |       78.78 / 79.38 |           73.25 |
| `EXP-INT4-FP16`      |           70.96 / 68.08 |    65.38 |            58.74 / 78.54 |       78.73 / 79.65 |           72.91 |

## 结果分析与结论

1. `EXP-UINT4-ZP-BF16` 按本报告定义的四项平均排名第一（75.14%），并在 GSM8K flexible/strict、MMLU acc、PIQA acc/acc_norm 上取得最高分。
2. 带 zero-point 的三种方案整体表现较强，四项平均在 73.99% 至 75.14% 之间，高于本表中不带 zero-point 的 FP4 和 INT4 方案。
3. scale dtype 的效果依任务而异，BF16 与 FP16 不存在对所有任务都占优的统一排序。例如，`EXP-UINT4-ZP-FP16` 的 HellaSwag acc 最高，而 `EXP-UINT4-ZP-UE5M3` 的 HellaSwag acc_norm 最高。
4. 方案选择应结合目标工作负载。四项平均可用于整体比较；若部署目标明确，应优先参考对应任务指标。

以上结论仅适用于本模型、本实验配置和单次评测结果，不构成统计显著性结论，也不能保证其他模型或推理后端上的排序一致。
