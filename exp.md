# FP4/INT4 RTN 量化精度对比

## 结论

本实验比较 4 个模型在 BF16 基线和 group size 16 量化方案下的表现。所有量化结果均使用 RTN，**不使用校准数据**。带 zero-point 的 `EXP-UINT4-ZP-BF16/FP16` 结果来自早期矩阵实验；当时实验目录和日志使用旧名 `EXP-INT4-ZP-BF16S/FP16S`，现按实际无符号码和 zero-point 语义统一命名。旧名仍可用于 CLI，并映射到新名称。

按 GSM8K strict-match、MMLU acc、HellaSwag acc_norm 和 PIQA acc_norm 四项指标等权平均，带 zero-point 的 `EXP-UINT4-ZP-UE5M3` 是本次量化方案中平均分最高的方案（80.09），接近 BF16（80.44）。但各模型的相对表现不同，不能据此认为它在所有模型和任务上都更优。

| 方案                     | 权重 / 激活 | 四模型平均分 |
| ------------------------ | ----------- | -----------: |
| BF16                     | BF16 / BF16 |        80.44 |
| `EXP-UINT4-ZP-UE5M3`† | W4 / A16    |        80.09 |
| `EXP-UINT4-ZP-BF16`†  | W4 / A16    |        79.92 |
| `EXP-INT4-BF16`†      | W4 / A16    |        79.80 |
| `EXP-UINT4-ZP-FP16`†  | W4 / A16    |        79.75 |
| `EXP-INT4-FP16`†      | W4 / A16    |        79.72 |
| `EXP-INT4-UBF16`†     | W4 / A16    |        79.33 |
| `EXP-INT4-UFP16`†     | W4 / A16    |        79.25 |
| `EXP-INT4-E4M3`†      | W4 / A16    |        79.13 |
| `EXP-INT4-UE5M3`†‡   | W4 / A16    |        78.91 |
| `EXP-FP4-UE5M3`        | W4 / A4     |        78.72 |
| `EXP-FP4-BF16`†       | W4 / A4     |        78.35 |
| `EXP-FP4-FP16`†       | W4 / A4     |        78.35 |

注意：上表为历史结果。当时 FP4 方案是 W4A4，INT4 方案只量化权重（W4A16），两类分数不是同等条件下的对比。带 `†` 的结果与当前 preset 的激活配置不同，带 `‡` 的结果还使用了已修复的 `[-8, 7]` 截断逻辑，详见「受影响的历史结果」；修复后的当前定义结果见「完整结果 · W4A4」。

BF16 与 FP16 scale 的平均结果几乎相同，没有显示出某一种 dtype 的全面优势。所有结果均来自单次评测，小幅差异不应解读为统计显著。

### 受影响的历史结果

已修复两个问题：INT4 方案此前只量化权重（`act_bits=16`），FP4 的 BF16/FP16 scale 变体的激活使用 `nvfp4_v2` 而非与权重相同的格式；`EXP-INT4-UE5M3` 此前使用 `[-8, 7]` 并把正侧最大值截到 7。现在所有 EXP 方案的激活都与权重同格式量化（W4A4），`EXP-INT4-UE5M3` 恢复对称 `[-7, 7]`。本文历史表格中的标记含义如下：

| 方案                                                                                                                    | 历史结果的实际配置                        | 当前定义                                    | 标记 |
| ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------- | ---- |
| `EXP-INT4-E4M3`、`EXP-UINT4-ZP-UE5M3`、`EXP-INT4-UBF16/UFP16`、`EXP-UINT4-ZP-BF16/FP16`、`EXP-INT4-BF16/FP16` | W4A16                                     | W4A4，激活与权重同格式                      | †   |
| `EXP-INT4-UE5M3`                                                                                                      | W4A16；码值 `[-8, 7]`，正侧最大值截到 7 | W4A4；码值 `[-7, 7]`                      | †‡ |
| `EXP-FP4-BF16/FP16`                                                                                                   | W4A4；激活为 `nvfp4_v2`（UE5M3 scale）  | W4A4；激活与权重同为 BF16/FP16 scale 的 FP4 | †   |
| `EXP-FP4-UE5M3`                                                                                                       | W4A4；激活为 `nvfp4_v2`                 | 不变                                        | 无   |

带标记的历史分数不代表当前 preset 的精度，仅保留用于追溯；如需按当前定义比较，请使用「完整结果 · W4A4」或重新评测。

### 平均精度

平均精度取 GSM8K strict-match、MMLU acc、HellaSwag acc_norm、PIQA acc_norm 四项的算术平均。先对各模型分别计算，再对四个模型等权平均。该均值用于汇总比较，不代表任务难度或统计显著性相同。

#### 按模型

| 模型                  |  BF16 | `EXP-FP4-UE5M3` | `EXP-INT4-E4M3`† | `EXP-UINT4-ZP-UE5M3`† | `EXP-INT4-UE5M3`†‡ | `EXP-FP4-BF16`† | `EXP-FP4-FP16`† | `EXP-UINT4-ZP-BF16`† | `EXP-UINT4-ZP-FP16`† | `EXP-INT4-UBF16`† | `EXP-INT4-UFP16`† | `EXP-INT4-BF16`† | `EXP-INT4-FP16`† |
| --------------------- | ----: | ----------------: | ------------------: | -----------------------: | ---------------------: | -----------------: | -----------------: | ----------------------: | ----------------------: | -------------------: | -------------------: | ------------------: | ------------------: |
| Llama-3.1-8B-Instruct | 76.24 |             72.00 |               72.78 |                    75.51 |                  72.88 |              71.30 |              71.08 |                   75.43 |                   75.12 |                74.53 |                74.66 |               74.59 |               74.51 |
| Phi-4                 | 82.67 |             81.91 |               81.98 |                    82.33 |                  81.68 |              81.60 |              81.55 |                   82.25 |                   82.21 |                82.04 |                81.98 |               82.21 |               82.10 |
| Qwen2.5-7B-Instruct   | 78.53 |             77.66 |               78.09 |                    78.99 |                  77.48 |              76.96 |              77.34 |                   78.30 |                   78.03 |                77.96 |                77.74 |               79.18 |               79.09 |
| Qwen3-32B             | 84.32 |             83.31 |               83.69 |                    83.54 |                  83.61 |              83.55 |              83.43 |                   83.69 |                   83.64 |                82.77 |                82.63 |               83.22 |               83.19 |

#### 按方案

| Scheme                   | Llama-3.1-8B-Instruct | Phi-4 | Qwen2.5-7B-Instruct | Qwen3-32B | 四模型平均 |
| ------------------------ | --------------------: | ----: | ------------------: | --------: | ---------: |
| BF16                     |                 76.24 | 82.67 |               78.53 |     84.32 |      80.44 |
| `EXP-FP4-UE5M3`        |                 72.00 | 81.91 |               77.66 |     83.31 |      78.72 |
| `EXP-INT4-E4M3`†      |                 72.78 | 81.98 |               78.09 |     83.69 |      79.13 |
| `EXP-UINT4-ZP-UE5M3`† |                 75.51 | 82.33 |               78.99 |     83.54 |      80.09 |
| `EXP-INT4-UE5M3`†‡   |                 72.88 | 81.68 |               77.48 |     83.61 |      78.91 |
| `EXP-FP4-BF16`†       |                 71.30 | 81.60 |               76.96 |     83.55 |      78.35 |
| `EXP-FP4-FP16`†       |                 71.08 | 81.55 |               77.34 |     83.43 |      78.35 |
| `EXP-UINT4-ZP-BF16`†  |                 75.43 | 82.25 |               78.30 |     83.69 |      79.92 |
| `EXP-UINT4-ZP-FP16`†  |                 75.12 | 82.21 |               78.03 |     83.64 |      79.75 |
| `EXP-INT4-UBF16`†     |                 74.53 | 82.04 |               77.96 |     82.77 |      79.33 |
| `EXP-INT4-UFP16`†     |                 74.66 | 81.98 |               77.74 |     82.63 |      79.25 |
| `EXP-INT4-BF16`†      |                 74.59 | 82.21 |               79.18 |     83.22 |      79.80 |
| `EXP-INT4-FP16`†      |                 74.51 | 82.10 |               79.09 |     83.19 |      79.72 |

## Details

### 实验设置

比较对象为 Llama-3.1-8B-Instruct、Phi-4、Qwen2.5-7B-Instruct 和 Qwen3-32B；评测任务为 GSM8K、MMLU、HellaSwag 和 PIQA。GSM8K 使用 5-shot greedy、生成上限 1024 tokens；评测 dtype 为 BF16。全文结果统一采用 GSM8K strict-match、MMLU acc、HellaSwag acc_norm 和 PIQA acc_norm。

| CLI scheme                                    | 格式     | 量化方式（当前定义；激活与权重同格式，W4A4）                                      | 理论权重 BPW |
| --------------------------------------------- | -------- | --------------------------------------------------------------------------------- | -----------: |
| BF16                                          | BF16     | 未量化基线                                                                        |           16 |
| `EXP-FP4-UE5M3`                             | FP4      | E2M1、逐组 UE5M3 scale、无 global scale                                           |          4.5 |
| `EXP-INT4-E4M3`                             | INT4     | 码值 `[-8, 7]`、逐组 signed E4M3 scale；scale 可取负，绝对值较大的一侧映射到 -8 |          4.5 |
| `EXP-INT4-UE5M3`                            | INT4     | 码值 `[-7, 7]`、逐组 UE5M3 scale；scale 非负，对称使用 ±7                      |          4.5 |
| `EXP-UINT4-ZP-UE5M3`                        | UINT4-ZP | 无符号码 `[0, 15]`、逐组 UE5M3 scale 和 zero-point                              |        4.75* |
| `EXP-FP4-BF16` / `EXP-FP4-FP16`           | FP4      | E2M1、逐组 BF16 / FP16 scale                                                      |          5.0 |
| `EXP-INT4-UBF16` / `EXP-INT4-UFP16`       | INT4     | 码值 `[-7, 7]`、逐组非负 BF16 / FP16 scale                                      |          5.0 |
| `EXP-UINT4-ZP-BF16` / `EXP-UINT4-ZP-FP16` | UINT4-ZP | 无符号码 `[0, 15]`、逐组 BF16 / FP16 scale 和 zero-point                        |        5.25* |
| `EXP-INT4-BF16` / `EXP-INT4-FP16`         | INT4     | 码值 `[-8, 7]`、逐组带符号 BF16 / FP16 scale；绝对值较大的一侧映射到 -8         |          5.0 |

BPW 按 group size 16 和紧凑元数据计算：`BPW = weight_bits + scale_bits / 16 + zero_point_bits / 16`。E4M3/UE5M3 scale 按 8 bit、BF16/FP16 scale 按 16 bit；带 `*` 的方案假设 zero-point 在 `[0, 15]` 内紧凑存为 4 bit。表中未计入对齐、填充等容器开销。这里是理论值，fake 格式产物的实际存储大小可能不同。

表中 `EXP-UINT4-ZP-UE5M3` 使用无符号码 `[0, 15]` 和 zero-point，反量化后的权重仍可为负。`EXP-FP4-UE5M3` 即 `NVFP4_E5M3` preset，不同于使用 E4M3 scale 和 FP32 global scale 的 `NVFP4`。当前所有 EXP 方案的激活都用与权重相同的格式量化（W4A4，group size 16）；历史结果与当前定义的差异见「受影响的历史结果」。BF16/FP16 表示 scale 的浮点 dtype；`UBF16/UFP16` 中的 `U` 表示该对称 INT4 方案的 scale 为非负值，不表示 BF16/FP16 是无符号浮点 dtype。所有新名称仅更改 CLI 标识，不改变内部量化实现或序列化 dtype。

旧名映射如下：`EXP-FP4` → `EXP-FP4-UE5M3`，`EXP-NVSINT4` → `EXP-INT4-E4M3`，`EXP-NVINT4` → `EXP-INT4-UE5M3`，`EXP-NVINT4-ZP` → `EXP-UINT4-ZP-UE5M3`；`EXP-FP4-BF16S/FP16S` → `EXP-FP4-BF16/FP16`；scale INT4 变体中，`EXP-SINT4-BF16S/FP16S` → `EXP-INT4-BF16/FP16`，对称 `[-7, 7]` 的 `EXP-INT4-BF16S/FP16S` → `EXP-INT4-UBF16/UFP16`，带 zero-point 的 `EXP-INT4-ZP-BF16S/FP16S` → `EXP-UINT4-ZP-BF16/FP16`。历史日志不会重写，查看旧目录时应按此映射理解。

所有量化均为 RTN（`--iters 0 --disable_opt_rtn`），不读取或使用校准数据。部分复现脚本保留了 `--dataset`、`--nsamples` 等参数，但 RTN 路径不消费这些数据。随机种子为 42。

### 复现方法

设置 `MODEL_PATH` 为待评测模型目录。BF16 基线直接将 `EVAL_MODEL` 指向原模型；量化方案先运行量化命令，再将其指向量化产物。`SCHEME` 可取上表任一量化方案。

```bash
MODEL_PATH=/path/to/model
SCHEME=EXP-INT4-UE5M3
OUTPUT_DIR=/path/to/output/${SCHEME}

python -m auto_round "$MODEL_PATH" \
  --scheme "$SCHEME" \
  --format fake \
  --nsamples 128 \
  --seqlen 2048 \
  --batch_size 8 \
  --dataset NeelNanda/pile-10k \
  --seed 42 \
  --device_map 0 \
  --enable_deterministic_algorithms \
  --disable_opt_rtn \
  --iters 0 \
  --output_dir "$OUTPUT_DIR"
EVAL_MODEL="$OUTPUT_DIR"
EVAL_BS=8
Llama-3.1-8B-Instruct 使用 EVAL_BS=32。
auto-round eval "$EVAL_MODEL" \
  --tasks mmlu,hellaswag,piqa \
  --device_map 0 \
  --seed 42 \
  --eval_bs "$EVAL_BS" \
  --eval_model_dtype bfloat16 \
  --eval_task_by_task
auto-round eval "$EVAL_MODEL" \
  --tasks gsm8k \
  --device_map 0 \
  --seed 42 \
  --eval_bs 8 \
  --num_fewshot 5 \
  --eval_model_dtype bfloat16 \
  --eval_gen_kwargs max_gen_toks=1024,do_sample=False
```

### 完整结果

分数为百分比。结果列统一为 GSM8K strict-match、MMLU acc、HellaSwag acc_norm 和 PIQA acc_norm。两张表的 BF16 基线相同。所有结果均为单次评测。

#### WOQ（W4A16）

INT4 方案的历史结果，只量化权重（`act_bits=16`），对应上方「结论」和「平均精度」中带 `†` 的 INT4 分数。`‡` 表示还使用了已修复的 `[-8, 7]` 截断逻辑。

| 模型 | Scheme | GSM8K strict-match | MMLU acc | HellaSwag acc_norm | PIQA acc_norm |
| --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B-Instruct | BF16 | 76.72 | 68.07 | 79.17 | 81.01 |
| Llama-3.1-8B-Instruct | `EXP-INT4-E4M3` | 68.99 | 64.19 | 77.78 | 80.14 |
| Llama-3.1-8B-Instruct | `EXP-UINT4-ZP-UE5M3` | 75.82 | 66.42 | 79.00 | 80.79 |
| Llama-3.1-8B-Instruct | `EXP-INT4-UE5M3`‡ | 68.31 | 64.88 | 78.09 | 80.25 |
| Llama-3.1-8B-Instruct | `EXP-INT4-UBF16` | 71.49 | 66.56 | 79.39 | 80.69 |
| Llama-3.1-8B-Instruct | `EXP-INT4-UFP16` | 71.72 | 66.74 | 79.29 | 80.90 |
| Llama-3.1-8B-Instruct | `EXP-UINT4-ZP-BF16` | 74.91 | 67.11 | 79.08 | 80.63 |
| Llama-3.1-8B-Instruct | `EXP-UINT4-ZP-FP16` | 74.15 | 67.17 | 78.97 | 80.20 |
| Llama-3.1-8B-Instruct | `EXP-INT4-BF16` | 71.72 | 66.91 | 78.97 | 80.74 |
| Llama-3.1-8B-Instruct | `EXP-INT4-FP16` | 71.49 | 66.85 | 79.00 | 80.69 |
| Phi-4 | BF16 | 90.67 | 76.86 | 81.98 | 81.18 |
| Phi-4 | `EXP-INT4-E4M3` | 89.39 | 75.89 | 81.57 | 81.07 |
| Phi-4 | `EXP-UINT4-ZP-UE5M3` | 89.99 | 75.98 | 81.59 | 81.77 |
| Phi-4 | `EXP-INT4-UE5M3`‡ | 88.63 | 75.67 | 81.36 | 81.07 |
| Phi-4 | `EXP-INT4-UBF16` | 89.23 | 76.46 | 81.47 | 81.01 |
| Phi-4 | `EXP-INT4-UFP16` | 88.86 | 76.46 | 81.47 | 81.12 |
| Phi-4 | `EXP-UINT4-ZP-BF16` | 89.54 | 76.24 | 81.61 | 81.61 |
| Phi-4 | `EXP-UINT4-ZP-FP16` | 89.92 | 76.10 | 81.54 | 81.28 |
| Phi-4 | `EXP-INT4-BF16` | 89.69 | 76.79 | 81.49 | 80.85 |
| Phi-4 | `EXP-INT4-FP16` | 89.31 | 76.73 | 81.45 | 80.90 |
| Qwen2.5-7B-Instruct | BF16 | 81.58 | 71.64 | 80.52 | 80.36 |
| Qwen2.5-7B-Instruct | `EXP-INT4-E4M3` | 83.17 | 70.85 | 79.22 | 79.11 |
| Qwen2.5-7B-Instruct | `EXP-UINT4-ZP-UE5M3` | 84.91 | 71.26 | 80.03 | 79.76 |
| Qwen2.5-7B-Instruct | `EXP-INT4-UE5M3`‡ | 81.05 | 70.44 | 79.54 | 78.89 |
| Qwen2.5-7B-Instruct | `EXP-INT4-UBF16` | 81.65 | 71.15 | 79.82 | 79.22 |
| Qwen2.5-7B-Instruct | `EXP-INT4-UFP16` | 80.89 | 71.09 | 79.92 | 79.05 |
| Qwen2.5-7B-Instruct | `EXP-UINT4-ZP-BF16` | 81.73 | 71.66 | 80.27 | 79.54 |
| Qwen2.5-7B-Instruct | `EXP-UINT4-ZP-FP16` | 80.59 | 71.82 | 80.22 | 79.49 |
| Qwen2.5-7B-Instruct | `EXP-INT4-BF16` | 86.35 | 71.36 | 80.10 | 78.89 |
| Qwen2.5-7B-Instruct | `EXP-INT4-FP16` | 86.20 | 71.29 | 80.08 | 78.78 |
| Qwen3-32B | BF16 | 91.89 | 80.79 | 82.60 | 82.00 |
| Qwen3-32B | `EXP-INT4-E4M3` | 90.67 | 80.35 | 82.28 | 81.45 |
| Qwen3-32B | `EXP-UINT4-ZP-UE5M3` | 90.37 | 80.24 | 82.27 | 81.28 |
| Qwen3-32B | `EXP-INT4-UE5M3`‡ | 90.90 | 80.01 | 82.29 | 81.23 |
| Qwen3-32B | `EXP-INT4-UBF16` | 88.48 | 80.42 | 81.43 | 80.74 |
| Qwen3-32B | `EXP-INT4-UFP16` | 87.79 | 80.36 | 81.45 | 80.90 |
| Qwen3-32B | `EXP-UINT4-ZP-BF16` | 90.60 | 80.63 | 82.07 | 81.45 |
| Qwen3-32B | `EXP-UINT4-ZP-FP16` | 90.07 | 80.54 | 82.18 | 81.77 |
| Qwen3-32B | `EXP-INT4-BF16` | 89.01 | 80.56 | 82.12 | 81.18 |
| Qwen3-32B | `EXP-INT4-FP16` | 88.78 | 80.58 | 82.15 | 81.23 |

#### W4A4

当前 preset 下的全方案结果：12 个方案的激活均与权重同格式量化（W4A4、group size 16、RTN、seed 42、fake 量化）。Qwen3-32B 的 `EXP-FP4-UE5M3`、`EXP-INT4-UE5M3`、`EXP-UINT4-ZP-UE5M3` 三项 GSM8K 使用推理阶段编译的激活 QDQ，已验证与 eager 结果逐位一致。

| 模型 | Scheme | GSM8K strict-match | MMLU acc | HellaSwag acc_norm | PIQA acc_norm |
| --- | --- | --- | --- | --- | --- |
| Llama-3.1-8B-Instruct | BF16 | 76.72 | 68.07 | 79.17 | 81.01 |
| Llama-3.1-8B-Instruct | `EXP-FP4-UE5M3` | 63.84 | 63.67 | 77.49 | 79.00 |
| Llama-3.1-8B-Instruct | `EXP-INT4-E4M3` | 65.13 | 62.04 | 76.71 | 78.56 |
| Llama-3.1-8B-Instruct | `EXP-INT4-UE5M3` | 64.97 | 64.39 | 78.32 | 79.27 |
| Llama-3.1-8B-Instruct | `EXP-UINT4-ZP-UE5M3` | 71.57 | 65.48 | 78.72 | 80.20 |
| Llama-3.1-8B-Instruct | `EXP-FP4-BF16` | 63.46 | 64.00 | 78.22 | 78.89 |
| Llama-3.1-8B-Instruct | `EXP-FP4-FP16` | 62.32 | 63.78 | 78.05 | 79.76 |
| Llama-3.1-8B-Instruct | `EXP-INT4-UBF16` | 67.10 | 64.76 | 78.34 | 79.38 |
| Llama-3.1-8B-Instruct | `EXP-INT4-UFP16` | 68.16 | 64.29 | 78.47 | 79.54 |
| Llama-3.1-8B-Instruct | `EXP-UINT4-ZP-BF16` | 75.44 | 66.00 | 78.52 | 80.58 |
| Llama-3.1-8B-Instruct | `EXP-UINT4-ZP-FP16` | 73.54 | 65.95 | 78.54 | 79.92 |
| Llama-3.1-8B-Instruct | `EXP-INT4-BF16` | 70.05 | 65.04 | 78.52 | 79.38 |
| Llama-3.1-8B-Instruct | `EXP-INT4-FP16` | 68.08 | 65.38 | 78.54 | 79.65 |
| Phi-4 | BF16 | 90.67 | 76.86 | 81.98 | 81.18 |
| Phi-4 | `EXP-FP4-UE5M3` | 88.86 | 75.67 | 81.00 | 80.52 |
| Phi-4 | `EXP-INT4-E4M3` | 88.55 | 75.42 | 81.14 | 80.79 |
| Phi-4 | `EXP-INT4-UE5M3` | 85.82 | 76.11 | 81.00 | 81.18 |
| Phi-4 | `EXP-UINT4-ZP-UE5M3` | 89.23 | 75.99 | 81.26 | 81.28 |
| Phi-4 | `EXP-FP4-BF16` | 88.48 | 75.92 | 81.02 | 81.01 |
| Phi-4 | `EXP-FP4-FP16` | 88.63 | 76.06 | 80.96 | 80.36 |
| Phi-4 | `EXP-INT4-UBF16` | 88.02 | 75.77 | 81.03 | 80.69 |
| Phi-4 | `EXP-INT4-UFP16` | 89.08 | 76.09 | 81.06 | 80.58 |
| Phi-4 | `EXP-UINT4-ZP-BF16` | 90.22 | 75.92 | 81.36 | 81.18 |
| Phi-4 | `EXP-UINT4-ZP-FP16` | 89.76 | 75.78 | 81.30 | 81.07 |
| Phi-4 | `EXP-INT4-BF16` | 88.78 | 76.47 | 81.20 | 80.79 |
| Phi-4 | `EXP-INT4-FP16` | 88.78 | 76.47 | 81.20 | 80.79 |
| Qwen2.5-7B-Instruct | BF16 | 81.58 | 71.64 | 80.52 | 80.36 |
| Qwen2.5-7B-Instruct | `EXP-FP4-UE5M3` | 83.24 | 70.57 | 79.04 | 78.07 |
| Qwen2.5-7B-Instruct | `EXP-INT4-E4M3` | 81.96 | 69.69 | 78.44 | 77.64 |
| Qwen2.5-7B-Instruct | `EXP-INT4-UE5M3` | 75.59 | 70.11 | 78.55 | 78.84 |
| Qwen2.5-7B-Instruct | `EXP-UINT4-ZP-UE5M3` | 81.20 | 70.67 | 79.24 | 79.05 |
| Qwen2.5-7B-Instruct | `EXP-FP4-BF16` | 83.17 | 70.25 | 79.07 | 78.51 |
| Qwen2.5-7B-Instruct | `EXP-FP4-FP16` | 82.64 | 70.35 | 79.20 | 77.91 |
| Qwen2.5-7B-Instruct | `EXP-INT4-UBF16` | 76.88 | 70.05 | 78.86 | 77.64 |
| Qwen2.5-7B-Instruct | `EXP-INT4-UFP16` | 77.03 | 70.11 | 78.76 | 78.29 |
| Qwen2.5-7B-Instruct | `EXP-UINT4-ZP-BF16` | 81.27 | 71.41 | 79.57 | 79.49 |
| Qwen2.5-7B-Instruct | `EXP-UINT4-ZP-FP16` | 80.44 | 71.14 | 79.70 | 78.13 |
| Qwen2.5-7B-Instruct | `EXP-INT4-BF16` | 83.55 | 70.36 | 79.11 | 78.40 |
| Qwen2.5-7B-Instruct | `EXP-INT4-FP16` | 83.55 | 70.36 | 79.11 | 78.40 |
| Qwen3-32B | BF16 | 91.89 | 80.79 | 82.60 | 82.00 |
| Qwen3-32B | `EXP-FP4-UE5M3` | 91.96 | 79.57 | 81.34 | 81.18 |
| Qwen3-32B | `EXP-INT4-E4M3` | 90.67 | 79.48 | 81.76 | 80.14 |
| Qwen3-32B | `EXP-INT4-UE5M3` | 87.19 | 79.28 | 80.50 | 80.14 |
| Qwen3-32B | `EXP-UINT4-ZP-UE5M3` | 91.05 | 79.60 | 82.05 | 81.01 |
| Qwen3-32B | `EXP-FP4-BF16` | 90.90 | 79.66 | 81.74 | 80.36 |
| Qwen3-32B | `EXP-FP4-FP16` | 91.96 | 79.42 | 81.63 | 81.23 |
| Qwen3-32B | `EXP-INT4-UBF16` | 86.88 | 79.69 | 80.27 | 79.76 |
| Qwen3-32B | `EXP-INT4-UFP16` | 87.87 | 79.01 | 80.64 | 80.20 |
| Qwen3-32B | `EXP-UINT4-ZP-BF16` | 91.96 | 79.86 | 81.83 | 80.96 |
| Qwen3-32B | `EXP-UINT4-ZP-FP16` | 90.37 | 79.78 | 82.06 | 80.90 |
| Qwen3-32B | `EXP-INT4-BF16` | 89.23 | 79.70 | 81.60 | 80.09 |
| Qwen3-32B | `EXP-INT4-FP16` | 89.23 | 79.69 | 81.62 | 80.09 |
