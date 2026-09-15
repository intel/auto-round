# AutoScheme 比特分配求解器

AutoScheme 通过求解一个背包问题来为每一层分配量化方案：每层有一组候选选项，每个选项带有
*比特代价* 和 *预测损失代价*（Delta-Loss 分数），求解器在平均比特预算的约束下为每层选择一个
选项，使总损失最小。

可通过 `AutoScheme.solver`（或命令行 `--auto_scheme_solver`）选择两种求解器：

| 求解器 | 说明 |
|:---|:---|
| `dp`（默认） | 背包动态规划。在离散化的比特网格上是精确的，但状态空间随比特预算增长。 |
| `lagrangian` | 通过拉格朗日对偶求解同一个背包问题。给定价格 `lam`（每比特的损失），每层独立选择 `argmin_s loss_s + lam * bits_s`；总比特数随 `lam` 单调递减，因此二分搜索可将解驱动到预算上。可精确命中 *小数* avg_bits 目标，且无需离散化状态空间。 |

由于对偶解只会落在每层 (bits, loss) 曲线的凸包上，需要两次原始修复来弥合整数间隙：贪心修复用
掉剩余预算，成对交换局部搜索用一次降级来资助一次升级。实践中修复后的对偶解恰好用满预算，且与
DP 的分配结果一致。

## 用法

```python
from auto_round import AutoRound
from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

scheme = AutoScheme(
    avg_bits=4.5,
    options="MXFP4,MXFP8",
    solver="lagrangian",  # 默认为 "dp"
)
ar = AutoRound(model_name, scheme=scheme)
model, layer_config = ar.quantize()
```

命令行：

```bash
auto_round Qwen/Qwen3-8B --avg_bits 4.5 --options "MXFP4,MXFP8" --solver lagrangian
```

## 精度

所有实验均使用 RTN（`iters=0`）、相同的校准数据（128 条样本，seqlen 512）、相同的 options 和
相同的 `avg_bits`。唯一的变量是求解器，因此任何差异都可归因于比特分配本身。`lm_head` 不量化。

使用 lm-eval 在 `lambada_openai`、`hellaswag`、`piqa`、`winogrande`、`truthfulqa_mc1`、
`openbookqa`、`boolq`、`arc_easy`、`arc_challenge`、`mmlu` 上评测。
表中的 **Avg** 是所有评测任务（*包含* MMLU 各子任务）的平均值，因此不等于所列十列的平均。

### 汇总

| 实验 | 模型 | `dp` | `lagrangian` | 差值 |
|:---|:---|:---:|:---:|:---:|
| INT, avg_bits 3.5 | Qwen3-8B | 0.6990 | **0.7045** | **+0.55pp** |
| INT, avg_bits 3.5 | Llama-3.1-8B-Instruct | 0.6386 | **0.6388** | +0.02pp |
| INT, avg_bits 3.0 | Qwen3-8B | 0.4643 | 0.4643 | 0.00pp |
| INT, avg_bits 3.0 | Llama-3.1-8B-Instruct | 0.5794 | **0.5813** | **+0.19pp** |
| MXFP4/8, avg_bits 4.5 | Qwen3-8B | 0.6942 | 0.6942 | 0.00pp |
| MXFP4/8, avg_bits 4.5 | Llama-3.1-8B-Instruct | 0.6221 | 0.6221 | 0.00pp |

在所有测试配置中拉格朗日求解器从未落后：6 个场景中 3 个与 DP 持平、3 个胜出。

### 表 1 — INT W2A16/W4A16/W8A16, avg_bits 3.5

**Qwen3-8B**

| 求解器 | 比特直方图 | lambada | hellaswag | piqa | winogrande | truthfulqa | openbookqa | boolq | arc_easy | arc_chal | mmlu | Avg |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `dp` | int2 65, int4 187 | 0.3276 | 0.6763 | 0.7144 | 0.6409 | 0.3599 | 0.3880 | 0.8425 | 0.6515 | 0.4343 | 0.6954 | 0.6990 |
| `lagrangian` | int2 63, int4 189 | 0.3088 | 0.6783 | 0.7111 | 0.6527 | 0.3660 | 0.3740 | 0.8410 | 0.6515 | 0.4292 | 0.7008 | **0.7045** |

**Llama-3.1-8B-Instruct**

| 求解器 | 比特直方图 | lambada | hellaswag | piqa | winogrande | truthfulqa | openbookqa | boolq | arc_easy | arc_chal | mmlu | Avg |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `dp` | int2 55, int4 169 | 0.6103 | 0.7549 | 0.7797 | 0.7261 | 0.3378 | 0.4200 | 0.8343 | 0.7433 | 0.4949 | 0.6281 | 0.6386 |
| `lagrangian` | int2 52, int4 172 | 0.5973 | 0.7547 | 0.7840 | 0.7214 | 0.3366 | 0.4180 | 0.8336 | 0.7483 | 0.5051 | 0.6248 | **0.6388** |

### 表 2 — INT W2A16/W4A16/W8A16, avg_bits 3.0

**Qwen3-8B** —— 两种求解器产生 *完全相同* 的分配，因此分数逐位一致。

| 求解器 | 比特直方图 | lambada | hellaswag | piqa | winogrande | truthfulqa | openbookqa | boolq | arc_easy | arc_chal | mmlu | Avg |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `dp` | int2 129, int4 123 | 0.1091 | 0.5721 | 0.6730 | 0.5572 | 0.3305 | 0.3320 | 0.5636 | 0.5215 | 0.3456 | 0.4466 | 0.4643 |
| `lagrangian` | int2 129, int4 123 | 0.1091 | 0.5721 | 0.6730 | 0.5572 | 0.3305 | 0.3320 | 0.5636 | 0.5215 | 0.3456 | 0.4466 | 0.4643 |

**Llama-3.1-8B-Instruct**

| 求解器 | 比特直方图 | lambada | hellaswag | piqa | winogrande | truthfulqa | openbookqa | boolq | arc_easy | arc_chal | mmlu | Avg |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `dp` | int2 109, int4 115 | 0.5700 | 0.5870 | 0.7013 | 0.6338 | 0.3219 | 0.3880 | 0.6939 | 0.6077 | 0.4232 | 0.5649 | 0.5794 |
| `lagrangian` | int2 106, int4 118 | 0.5750 | 0.5875 | 0.7002 | 0.6511 | 0.3182 | 0.3800 | 0.6985 | 0.6149 | 0.4258 | 0.5669 | **0.5813** |

### 表 3 — MXFP4/MXFP8, avg_bits 4.5

两个模型在两种求解器下都产生 *完全相同* 的分配，因此每个任务的分数都精确一致。

**Qwen3-8B**

| 求解器 | 比特直方图 | lambada | hellaswag | piqa | winogrande | truthfulqa | openbookqa | boolq | arc_easy | arc_chal | mmlu | Avg |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `dp` | mx_fp4 173, mx_fp8 79 | 0.6072 | 0.6943 | 0.7519 | 0.6551 | 0.3317 | 0.4060 | 0.8627 | 0.7597 | 0.5188 | 0.6795 | 0.6942 |
| `lagrangian` | mx_fp4 173, mx_fp8 79 | 0.6072 | 0.6943 | 0.7519 | 0.6551 | 0.3317 | 0.4060 | 0.8627 | 0.7597 | 0.5188 | 0.6795 | 0.6942 |

**Llama-3.1-8B-Instruct**

| 求解器 | 比特直方图 | lambada | hellaswag | piqa | winogrande | truthfulqa | openbookqa | boolq | arc_easy | arc_chal | mmlu | Avg |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `dp` | mx_fp4 159, mx_fp8 65 | 0.6375 | 0.7557 | 0.7884 | 0.7040 | 0.3097 | 0.4060 | 0.8156 | 0.7525 | 0.5000 | 0.6122 | 0.6221 |
| `lagrangian` | mx_fp4 159, mx_fp8 65 | 0.6375 | 0.7557 | 0.7884 | 0.7040 | 0.3097 | 0.4060 | 0.8156 | 0.7525 | 0.5000 | 0.6122 | 0.6221 |

### 结果解读

两种求解器在同一张分数表上优化同一个目标，因此当问题存在明确最优解时它们会收敛到 *同一个*
分配 —— MXFP4/8 4.5 比特以及 Qwen3-8B 的 INT 3.0 比特就是这种情况，其比特直方图和所有任务分数
都完全一致。

差异只出现在背包问题存在接近平局的场景：DP 工作在离散化的比特网格上，而拉格朗日对偶可精确处理
小数预算，因此可能落在略有不同的权衡点上（例如 `int2 63 / int4 189` 而非
`int2 65 / int4 187`）。在所有这类情况下，拉格朗日的选择都不劣于 DP。

需要注意的是，即使平均值提升，单个任务的分数仍会双向波动 —— 在 Qwen3-8B 3.5 比特上，拉格朗日
分配在 `lambada_openai` 上低了 1.9pp，但在 `winogrande`、`truthfulqa_mc1`、`hellaswag` 和
`mmlu` 上都有提升。评判分配质量应看总体平均，而非任何单一任务。

### 复现

```bash
sh run_autoscheme_staged_ab.sh mxfp4
```

A/B 驱动脚本见 `autoscheme_staged_ab.py`，各实验的具体设置见 `run_autoscheme_staged_ab.sh`。

