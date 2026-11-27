# AutoScheme Tuning-Aware Layer Config 实验

## 机器适用范围

本文中的 `/models/Qwen3-8B` 模型路径，以及“GPU 7 禁止使用”的约束，**仅适用于 hostname 为 `mlp-dgx-01` 的当前机器**。每次启动实验前先检查：

```bash
hostname -f
```

若输出不是 `mlp-dgx-01`，立即停止；不要使用本文的模型路径，也不要把 GPU 7 的本机限制套用到其他机器。先在目标机器确认模型实际位置和 GPU 资源策略，再更新本节配置后执行。不得仅因 GPU 7 在本机禁用，就推断它在其他机器也禁用。

## 目标与假设

目标是在 `/models/Qwen3-8B` 上，以 MXFP4+MXFP8、`avg_bits=4.5` 为主实验，检验 tuning-aware AutoScheme 是否能在相同平均 bit 下提高精度。MXFP 是权重和 activation 联合量化：MXFP4 通常是 W4A4，MXFP8 是 W8A8。此前只讨论 weight rounding 会漏掉 activation 误差；本实验必须分开记录 weight、activation 和合计影响。

待验证假设：现有 AutoScheme 的 RTN 分数会把“RTN 误差大但可被 tuning 修复”的层判成高敏感；使用 tuning 后、同一 FP 参考输出下测得的残差重新估计 layer config，可以把 bit 预算从可修复层转给不可修复层。若 activation 项占主导，真正有价值的变化可能是 act rounding / activation scale，而非 weight rounding。

层间耦合也要实测：闭环 tuning 使用上一层量化输出，开环 tuning 使用 FP 输入；两种测量不可混为一谈。全 MXFP4 的 tuning residual 只用作候选指标，最终成败由完整混合配置在统一评估集上的结果决定。

## 指标定义

对每个 layer、scheme、固定 calibration batch 及相同 FP 上游输入，采集：

- `S_act`: activation QDQ 残差对反向梯度的绝对一阶代理，按当前 AutoScheme 的 `sum(abs(grad * (x - qdq_x)))` 定义。
- `S_weight`: weight QDQ 残差的一阶代理，按 `sum(abs(grad * (W - qdq_W)))` 定义。
- `S_combined = S_act + S_weight`：当前 AutoScheme allocator 使用的总分。
- `S2_act`, `S2_weight`, `S2_joint`: 同输入/输出损失下的二阶或直接 forward loss 残差；能分离时分别记录，joint 为主决策验证量。不要默认一阶代理等价于 tuning 目标。
- `r_act = S_act_tuned / max(S_act_RTN, eps)`，`r_weight` 同理；候选 combined 分数为 `r_act*S_act_RTN + r_weight*S_weight_RTN`。分母过小的层标记为 low-signal，不直接用比值放大噪声。
- 配置预算使用 AutoScheme 实际 parameter-bit accounting；`ignore_scale_zp_bits=True` 仅为对齐历史表格，另存真实 bits 与 scale/zp bits。

必须输出每层 act/weight 占比、RTN 与 tuned residual、上游输入漂移、闭环/开环差异。激活和权重误差可能相消或交互，因此决策最终需要通过 joint loss / PPL 验证，不能只把两个分数机械相加后宣称因果。

## 实验矩阵

所有实验固定模型 checkpoint、tokenizer、calibration 样本及顺序、seed、seqlen、batch size、AutoRound 迭代数、评估脚本版本。禁用 lm_head 量化以对齐历史 AutoScheme 表。仅在 `mlp-dgx-01` 上使用 `/models/Qwen3-8B` 并排除 GPU 7；其他主机须先完成上面的机器检查和本机配置确认。每个 GPU 任务都显式设置 `CUDA_VISIBLE_DEVICES`。

| ID | 配置/流程 | 目的 | 状态 |
|---|---|---|---|
| A0 | BF16 | 质量上界 | 待运行/确认口径 |
| A1 | 全 MXFP4 + tuning | 低 bit 下界、采集 tuned act/weight 分量 | 待运行 |
| A2 | AutoScheme MXFP4/MXFP8, 4.5 bit, iters=0 | 复现 RTN allocator 与历史口径 | 待运行 |
| A3 | 当前 AutoScheme MXFP4/MXFP8, 4.5 bit, iters=200 | 主基线；两个 seed 测 tuning 方差 | 待运行 |
| B-act | RTN 分数乘 tuning 后 `r_act`；weight multiplier=1 | 单独检验 activation 可修复性 | 待运行 |
| B-weight | RTN 分数乘 tuning 后 `r_weight`；act multiplier=1 | 单独检验 weight rounding 可修复性 | 待运行 |
| B-joint | act/weight 分量分别乘实测 residual ratio | 主实验：分量感知 allocator | 待运行 |
| B-open | 从开环 residual ratio 生成 joint override | 测量上游量化引起的偏差 | 待运行 |
| C-pass2 | B-joint 混合配置 tuning 后重测、重分配一次 | 测混合配置改变输入分布后的稳定性 | 仅当 B-joint 有提升时运行 |
| C-joint | 对 allocator 边界层做逐层 MXFP4/MXFP8 替换，测真实 joint CE/PPL 增量 | 检查分数非加性和交互 | 仅测边界 Top-K |

至少对 A3 与最优 B arm 跑 2 个 tuning seed；首轮其余 arm 固定 seed。记录 mean/std，不以单次波动下结论。

## 执行步骤

1. 运行小模型 CPU 单测，验证 cache 兼容以及 act/weight override 算式；检查 pre-commit。
2. 先验证 hostname 是 `mlp-dgx-01`；仅在该机器确认 `/models/Qwen3-8B`、校准数据和评估数据可用，并选空闲 GPU（排除 7）。hostname 不匹配时停止，不使用这个模型路径或 GPU 规则。若当前没有符合条件的空闲卡，不启动 GPU 任务。
3. 跑 A2 快速校准，保存 layer config、每层 RTN `S_act/S_weight`、score cache、log、实际平均 bits；确认 act_bits=4/8 的方案确实被评分。
4. 跑 A1 tuning（优先闭环），在固定 reference/gradient 定义下采集 tuned residual；然后补开环测量，区分本层修复与对上游误差的补偿。确认当前工具能以相同 objective 取得 component residual；若不能，先补 instrumentation，不用异口径结果造 override。
5. 生成三个独立 override：activation-only、weight-only、joint。对小比例分母使用 shrinkage/阈值策略；输出 ratio 分布、clip/low-signal 层及排名稳定性。
6. 分别执行 B arms，保留 cache 与 override 原件，核验 override 命中层数、DP 输出 bits、avg bits、按层 act_bits/weight bits。
7. 首筛统一 wikitext2 PPL；不劣于 A3 或结构变化显著的 arm 进入 lm-eval，报告 lambada_openai、hellaswag、piqa、winogrande、mmlu 各项及宏平均。文档表中历史 4.5-bit Qwen3-8B accuracy=0.6550、纯 4-bit=0.6316、5-bit=0.6594、BF16=0.6764；只有相同任务、tokenizer、fake/real backend 与 lm-head 口径一致时才横向比较。
8. B-joint 有提升时跑 C-pass2 和边界 C-joint；否则依据失败诊断调整指标，不扩模型规模。

## 判定标准

主成功标准：最佳 B arm 的 lm-eval 五任务宏平均比 A3 高至少 0.003（绝对准确率），提升大于 A3 两个 seed 的波动；至少 3/5 任务提升，wikitext2 PPL 不恶化。A2 是机制/历史 sanity check，不替代 tuned A3 主基线。

次要证据：B-joint 比 B-act、B-weight 更稳定或更好；收益集中于之前 RTN 高分但 tuned residual 低的层；闭环与开环差异可解释；C-pass2 不推翻大部分分配。

## 失败预判与后续分析

- **组件分数缺失或同一层两次测量不在同一输入/梯度上**：不能生成有效 ratio。补充 score instrumentation 或用同批冻结 activation/gradient 重放；不要把两个独立 AutoScheme run 的归一化数值直接相除。
- **Activation 项占比极大但 B-act 无提升**：优先检查 act QDQ 的 scale/amax 是否实际由 tuning 优化、quantizer 是否 per-token/per-group、采样是否覆盖长尾；尝试直接对 activation scale 做小网格搜索，再用 joint CE 验证。仅调 weight rounding 不会解决纯 activation clipping/underflow。
- **A1 tuned residual 不降或 ratio 约为 1**：可能当前 tuning 目标未覆盖 MXFP act，或误差来自不可修复的动态范围/scale；检查 per-channel/group saturation、E8M0 scale、饱和率和零化率，区分可修复 rounding 与 clipping。
- **B config 有变化但精度持平/下降**：检查 layer interaction、上游漂移补偿和 calibration/eval 分布差异；比较闭环/开环与 C-pass2；对 DP 边界层测真实 ΔCE，避免单层分数可加性假设。
- **ratio 极端或大面积 >1**：小分母噪声、objective 不一致或 tuning 随机性。按 baseline 分数做 shrinkage，阈值屏蔽 low-signal 层，跑第二 seed，并用直接 joint loss 复核。
- **PPL 改善而任务精度下降（或反之）**：说明代理指标与下游任务不一致；以预注册的 lm-eval 宏平均为主判据，分任务报告，不挑选单项结果。
- **提升小于 seed 波动**：结论记为 inconclusive；增加 seed 或扩大 calibration samples，再决定是否继续。

## 结果记录

每完成一个实验，在下表追加一行，并链接日志、config、score cache、评估 JSON。记录 commit、模型标识、seed、GPU index、命令、real/nominal avg bits 和运行状态。

| 日期 | ID | commit | seed/GPU | Avg bits (real) | PPL | lm-eval 五项/均值 | 关键发现 | artifacts |
|---|---|---|---|---:|---:|---|---|---|
| 2026-11-27 | Infra | working tree | CPU | n/a | n/a | n/a | act/weight score components 与 override pure tests 通过；GPU 实验待检查设备和模型 | 当前工作区 |

## 实验实施记录

### 2026-11-27：分量感知基础设施

- AutoScheme score cache 将继续兼容旧 `[bits, loss]`，新分数额外保存 `S_act`、`S_weight`；后续报告同时保留 component 与 combined。
- 新增 `AR_AUTO_SCHEME_SCORE_OVERRIDE`，JSON 按 scheme index、layer、`act`/`weight` multiplier 调整分量；默认未设置时分数路径不变。
- 已通过：`pytest test/unit/common/schemes/test_auto_scheme.py -k 'autoscheme_score_override' -q`（2 passed）。
- 完整 cache 集成与 override/DP 回归：4 passed；pre-commit 全部通过。首次回归曾因旧断言按二元组解包而失败，已修为前两项，并加入旧 cache override fail-fast 测试。
- 新 cache 条目保留 `[bits, mix_score, act_score, weight_score]`；旧 `[bits, loss]` 仍可无 override 正常读取。override 只作用于内存副本，不改 cache 文件；缺组件、未知 layer 或非法 multiplier 会明确报错。
- 本机 `hostname -f` 为 `mlp-dgx-01`；GPU 状态检查：0/1/2/4/5/6 号卡繁忙；3 号卡显存使用 1.2 GiB，不满足空闲阈值；7 号卡仅在本机禁用。因此本轮没有启动任何模型 GPU 实验。`/models/Qwen3-8B` 仅确认在本机存在；仓库内未找到本地 calibration/Wikitext 数据文件。
- 重要方法约束：不能直接用“MXFP4 run 的梯度分数 / MXFP8 run 的梯度分数”构造 ratio；要用固定参考、冻结输入/梯度或直接 objective 的共同口径测 tuned residual。当前已完成分量保存与 override 注入，不声称 ratio 估计/精度提升已验证。
- 下一步：补齐共同输入的 RTN/tuned act/weight residual 采集脚本；等有空闲 GPU 后先跑 A2 sanity 与 A1 residual，再决定是否跑完整评估矩阵。
