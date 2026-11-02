# 实验性 FP4/INT4 BKM

本文记录 group size 为 16 的实验 scheme 在 Llama-3.1-8B-Instruct 上进行 HellaSwag 对比的可复现方法和结果。

## Scheme

| CLI scheme | 格式 | 说明 |
| --- | --- | --- |
| `exp-fp4` | FP4 | 使用 `NVFP4_E5M3` preset |
| `exp-nvsint4` | INT4 | 有符号 INT4，scale 使用 E4M3 |
| `exp-nvint4` | INT4 | 无符号 INT4，scale 使用 E5M3 |

本实验使用上述 `exp-*` 名称；它们不同于早期笔记中的 `exp-sint4` 和 `exp-int4`。

## 运行

将 `SCHEME` 设为上表中的一个 scheme，将 `MODE` 设为 `rtn`、`opt-rtn` 或 `tuned`。选择 0-6 中一张空闲 GPU；GPU 7 保留禁用。设置 `CUDA_VISIBLE_DEVICES` 后，`--device_map 0` 表示所选的可见 GPU。

```bash
MODEL_PATH=/models/Llama-3.1-8B-Instruct
GPU_ID=0
SCHEME=exp-nvint4
MODE=tuned
RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUTPUT_ROOT="/data0/xinhe/autoround-exp-hellaswag/runs/${RUN_ID}"

case "${MODE}" in
  rtn)
    MODE_ARGS=(--iters 0 --disable_opt_rtn)
    ;;
  opt-rtn)
    MODE_ARGS=(--iters 0)
    ;;
  tuned)
    MODE_ARGS=(--iters 200 --enable_minmax_tuning)
    ;;
  *)
    printf 'Unknown MODE: %s\n' "${MODE}" >&2
    exit 2
    ;;
esac

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m auto_round "${MODEL_PATH}" \
  --scheme "${SCHEME}" \
  --format fake \
  --tasks hellaswag \
  --eval_bs 32 \
  --nsamples 128 \
  --seqlen 2048 \
  --dataset NeelNanda/pile-10k \
  --seed 42 \
  --device_map 0 \
  --enable_deterministic_algorithms \
  --output_dir "${OUTPUT_ROOT}/${SCHEME}-${MODE}/model" \
  "${MODE_ARGS[@]}"
```

三种模式的区别：

| 模式 | 参数 | 行为 |
| --- | --- | --- |
| RTN | `--iters 0 --disable_opt_rtn` | Round-to-nearest 基线，并禁用优化 RTN pass |
| opt-RTN | `--iters 0` | 使用优化 RTN pass，不进行迭代调优 |
| tuned | `--iters 200 --enable_minmax_tuning` | 运行 200 次调优迭代，并启用 min/max tuning |

记录的实验使用 `NeelNanda/pile-10k` 作为校准数据（128 个样本，序列长度 2048），HellaSwag 评测 batch size 为 32，随机种子为 42。下表每项依次为 `acc` 和 `acc_norm`；评测日志中对应的近似 95% 置信区间分别为 ±0.49 和 ±0.41 个百分点。

## 结果

模型：Llama-3.1-8B-Instruct；评测集：HellaSwag。

| Scheme | RTN acc / acc_norm | opt-RTN acc / acc_norm | tuned acc / acc_norm |
| --- | ---: | ---: | ---: |
| `exp-fp4` | 58.48% / 77.67% | 58.70% / 78.13% | 58.11% / 77.33% |
| `exp-nvsint4` | 58.40% / 77.79% | 58.44% / 77.56% | 58.33% / 77.81% |
| `exp-nvint4` | 59.34% / 78.99% | 59.31% / 78.90% | 59.23% / 78.92% |

在这次实验中，tuned 没有稳定优于 RTN 或 opt-RTN。考虑到评测误差，这些小幅差异不应过度解读；仅凭单次实验不足以确定可靠的排序。

## 建议验证矩阵

建议用以下模型和数据集扩大验证范围。Llama-3.1-8B-Instruct/HellaSwag 已由上方结果覆盖；其余组合是后续测试建议，不代表已有实测结果。

| 模型 | GSM8K | MMLU | HellaSwag | PIQA |
| --- | --- | --- | --- | --- |
| Phi-4 | 建议测试 | 建议测试 | 建议测试 | 建议测试 |
| Llama-3.1-8B-Instruct | 建议测试 | 建议测试 | 已完成，见上表 | 建议测试 |
| Qwen2.5-7B-Instruct | 建议测试 | 建议测试 | 建议测试 | 建议测试 |
| Qwen3-32B | 建议测试 | 建议测试 | 建议测试 | 建议测试 |

`--tasks` 使用对应 lm-eval task 名称（`gsm8k`、`mmlu`、`hellaswag`、`piqa`）；具体任务名应以当前安装的 lm-eval 版本支持情况为准。运行完整矩阵前，先确认任务可用，并根据所选 GPU 调整评测 batch size。比较不同 scheme 时，应固定模型 revision、任务配置、量化模式、校准数据和随机种子。

## 参考

BKM 代码：[intel/auto-round 的 exp-int4 分支](https://github.com/intel/auto-round/tree/exp-int4)