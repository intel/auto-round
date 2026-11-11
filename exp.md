# FP4/INT4 RTN 量化精度对比

## 结论

本实验比较 4 个模型在 BF16 基线和 group size 16 量化方案下的表现。所有量化结果均使用 RTN，**不使用校准数据**。

按 GSM8K strict-match、MMLU acc、HellaSwag acc_norm 和 PIQA acc_norm 四项指标等权平均，带 zero-point 的 `exp-nvint4-zp` 是本次量化方案中平均分最高的方案（80.09），接近 BF16（80.44）。但各模型的相对表现不同，不能据此认为它在所有模型和任务上都更优。

| 方案 | 四模型平均分 |
| --- | ---: |
| BF16 | 80.44 |
| `exp-nvint4-zp` | 80.09 |
| `EXP-INT4-BF16S` | 79.33 |
| `EXP-INT4-FP16S` | 79.25 |

BF16S 与 FP16S scale 的平均结果几乎相同，没有显示出某一种 dtype 的全面优势。所有结果均来自单次评测，小幅差异不应解读为统计显著。

### 平均精度

平均精度取 GSM8K strict-match、MMLU acc、HellaSwag acc_norm、PIQA acc_norm 四项的算术平均。先对各模型分别计算，再对四个模型等权平均。该均值用于汇总比较，不代表任务难度或统计显著性相同。

#### 按模型

| 模型 | BF16 | `exp-fp4` | `exp-nvsint4` | `exp-nvint4-zp` | `exp-nvint4` | `EXP-INT4-BF16S` | `EXP-INT4-FP16S` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.1-8B-Instruct | 76.24 | 72.00 | 72.78 | 75.51 | 72.88 | 74.53 | 74.66 |
| Phi-4 | 82.67 | 81.91 | 81.98 | 82.33 | 81.68 | 82.04 | 81.98 |
| Qwen2.5-7B-Instruct | 78.53 | 77.66 | 78.09 | 78.99 | 77.48 | 77.96 | 77.74 |
| Qwen3-32B | 84.32 | 83.31 | 83.69 | 83.54 | 83.61 | 82.77 | 82.63 |

#### 按方案

| Scheme | Llama-3.1-8B-Instruct | Phi-4 | Qwen2.5-7B-Instruct | Qwen3-32B | 四模型平均 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 76.24 | 82.67 | 78.53 | 84.32 | 80.44 |
| `exp-fp4` | 72.00 | 81.91 | 77.66 | 83.31 | 78.72 |
| `exp-nvsint4` | 72.78 | 81.98 | 78.09 | 83.69 | 79.13 |
| `exp-nvint4-zp` | 75.51 | 82.33 | 78.99 | 83.54 | 80.09 |
| `exp-nvint4` | 72.88 | 81.68 | 77.48 | 83.61 | 78.91 |
| `EXP-INT4-BF16S` | 74.53 | 82.04 | 77.96 | 82.77 | 79.33 |
| `EXP-INT4-FP16S` | 74.66 | 81.98 | 77.74 | 82.63 | 79.25 |


## Details

### 实验设置
比较对象为 Llama-3.1-8B-Instruct、Phi-4、Qwen2.5-7B-Instruct 和 Qwen3-32B；评测任务为 GSM8K、MMLU、HellaSwag 和 PIQA。GSM8K 使用 5-shot greedy、生成上限 1024 tokens；评测 dtype 为 BF16。GSM8K 按 flexible-extract / strict-match 展示，HellaSwag 和 PIQA 按 acc / acc_norm 展示。

| CLI scheme | 格式 | 量化方式 | 理论权重 BPW |
| --- | --- | --- | ---: |
| BF16 | BF16 | 未量化基线 | 16 |
| `exp-fp4` | FP4 | E2M1 权重、逐组 E4M3 scale、FP32 global scale | `4.5 + 32/N` |
| `exp-nvsint4` | INT4 | 有符号 INT4、逐组 signed E4M3 scale | 4.5 |
| `exp-nvint4` | INT4 | 有符号 INT4（`[-8, 7]`）、逐组 E5M3 scale、无 zero-point | 4.5 |
| `exp-nvint4-zp` | INT4-ZP | 非对称 INT4、逐组 E5M3 scale 和 zero-point | 4.75* |
| `EXP-FP4-BF16S` / `EXP-FP4-FP16S` | FP4 | E2M1 权重、逐组 BF16 / FP16 scale | 5.0 |
| `EXP-INT4-BF16S` / `EXP-INT4-FP16S` | INT4 | 对称 INT4（`[-7, 7]`）、无 zero-point、逐组 BF16 / FP16 scale | 5.0 |
| `EXP-INT4-ZP-BF16S` / `EXP-INT4-ZP-FP16S` | INT4-ZP | 非对称 INT4、逐组 BF16 / FP16 scale 和 zero-point | 5.25* |
| `EXP-SINT4-BF16S` / `EXP-SINT4-FP16S` | INT4 | 有符号 INT4（`[-8, 7]`）、无 zero-point、逐组 BF16 / FP16 scale | 5.0 |

BPW 按 group size 16 和紧凑元数据计算：`BPW = weight_bits + scale_bits / 16 + zero_point_bits / 16`。E4M3/E5M3 scale 按 8 bit、BF16/FP16 scale 按 16 bit；带 `*` 的方案假设 zero-point 在 `[0, 15]` 内紧凑存为 4 bit。`N` 为共享 FP32 global scale 的权重张量元素数；表中未计入对齐、填充等容器开销。这里是理论值，fake 格式产物的实际存储大小可能不同。

表中 `exp-nvint4-zp` 是带 zero-point 的非对称方案；`exp-nvint4` 是无 zero-point 的有符号 INT4（`[-8, 7]`），而 `EXP-INT4` 是对称 INT4（`[-7, 7]`）。BF16S/FP16S 表示 scale 的数据类型；同位宽权重下两者理论 BPW 相同。结果章节仅列出本次实验实际评测的方案。

所有量化均为 RTN（`--iters 0 --disable_opt_rtn`），不读取或使用校准数据。部分复现脚本保留了 `--dataset`、`--nsamples` 等参数，但 RTN 路径不消费这些数据。随机种子为 42。

### 复现方法

设置 `MODEL_PATH` 为待评测模型目录。BF16 基线直接将 `EVAL_MODEL` 指向原模型；量化方案先运行量化命令，再将其指向量化产物。`SCHEME` 可取上表任一量化方案。
```bash
MODEL_PATH=/path/to/model
SCHEME=exp-nvint4
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

分数为百分比。GSM8K 按 flexible-extract / strict-match 展示，HellaSwag 和 PIQA 按 acc / acc_norm 展示。

| 模型 | Scheme | GSM8K flexible / strict | MMLU acc | HellaSwag acc / acc_norm | PIQA acc / acc_norm |
| --- | --- | ---: | ---: | ---: | ---: |
| Llama-3.1-8B-Instruct | BF16 | 78.54 / 76.72 | 68.07 | 59.07 / 79.17 | 79.76 / 81.01 |
| Llama-3.1-8B-Instruct | `exp-fp4` | 68.76 / 67.93 | 63.32 | 58.29 / 77.69 | 77.91 / 79.05 |
| Llama-3.1-8B-Instruct | `exp-nvsint4` | 69.14 / 68.99 | 64.19 | 58.44 / 77.78 | 79.11 / 80.14 |
| Llama-3.1-8B-Instruct | `exp-nvint4-zp` | 76.27 / 75.82 | 66.42 | 59.32 / 79.00 | 79.60 / 80.79 |
| Llama-3.1-8B-Instruct | `exp-nvint4` | 70.58 / 68.31 | 64.88 | 59.03 / 78.09 | 79.43 / 80.25 |
| Llama-3.1-8B-Instruct | `EXP-INT4-BF16S` | 74.37 / 71.49 | 66.56 | 59.42 / 79.39 | 79.98 / 80.69 |
| Llama-3.1-8B-Instruct | `EXP-INT4-FP16S` | 74.75 / 71.72 | 66.74 | 59.39 / 79.29 | 79.82 / 80.90 |
| Phi-4 | BF16 | 90.60 / 90.67 | 76.86 | 63.17 / 81.98 | 80.96 / 81.18 |
| Phi-4 | `exp-fp4` | 90.37 / 90.45 | 75.67 | 62.16 / 81.00 | 80.03 / 80.52 |
| Phi-4 | `exp-nvsint4` | 89.31 / 89.39 | 75.89 | 62.59 / 81.57 | 80.85 / 81.07 |
| Phi-4 | `exp-nvint4-zp` | 89.92 / 89.99 | 75.98 | 62.88 / 81.59 | 80.96 / 81.77 |
| Phi-4 | `exp-nvint4` | 88.55 / 88.63 | 75.67 | 62.40 / 81.36 | 81.18 / 81.07 |
| Phi-4 | `EXP-INT4-BF16S` | 89.16 / 89.23 | 76.46 | 62.62 / 81.47 | 80.74 / 81.01 |
| Phi-4 | `EXP-INT4-FP16S` | 88.78 / 88.86 | 76.46 | 62.65 / 81.47 | 80.96 / 81.12 |
| Qwen2.5-7B-Instruct | BF16 | 88.17 / 81.58 | 71.64 | 62.02 / 80.52 | 79.71 / 80.36 |
| Qwen2.5-7B-Instruct | `exp-fp4` | 84.76 / 82.94 | 70.57 | 60.88 / 79.04 | 77.58 / 78.07 |
| Qwen2.5-7B-Instruct | `exp-nvsint4` | 86.13 / 83.17 | 70.85 | 60.77 / 79.22 | 77.48 / 79.11 |
| Qwen2.5-7B-Instruct | `exp-nvint4-zp` | 86.13 / 84.91 | 71.26 | 61.85 / 80.03 | 78.84 / 79.76 |
| Qwen2.5-7B-Instruct | `exp-nvint4` | 87.04 / 81.05 | 70.44 | 61.15 / 79.54 | 78.73 / 78.89 |
| Qwen2.5-7B-Instruct | `EXP-INT4-BF16S` | 85.97 / 81.65 | 71.15 | 61.33 / 79.82 | 78.73 / 79.22 |
| Qwen2.5-7B-Instruct | `EXP-INT4-FP16S` | 85.90 / 80.89 | 71.09 | 61.35 / 79.92 | 78.56 / 79.05 |
| Qwen3-32B | BF16 | 90.67 / 91.89 | 80.79 | 63.90 / 82.60 | 81.01 / 82.00 |
| Qwen3-32B | `exp-fp4` | 90.67 / 91.28 | 79.28 | 63.13 / 81.57 | 80.25 / 81.12 |
| Qwen3-32B | `exp-nvsint4` | 89.54 / 90.67 | 80.35 | 63.50 / 82.28 | 80.47 / 81.45 |
| Qwen3-32B | `exp-nvint4-zp` | 89.54 / 90.37 | 80.24 | 63.60 / 82.27 | 80.58 / 81.28 |
| Qwen3-32B | `exp-nvint4` | 90.14 / 90.90 | 80.01 | 63.43 / 82.29 | 79.60 / 81.23 |
| Qwen3-32B | `EXP-INT4-BF16S` | 87.04 / 88.48 | 80.42 | 61.93 / 81.43 | 79.65 / 80.74 |
| Qwen3-32B | `EXP-INT4-FP16S` | 86.28 / 87.79 | 80.36 | 61.93 / 81.45 | 79.54 / 80.90 |

