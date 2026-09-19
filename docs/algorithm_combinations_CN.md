# 算法组合

AutoRound 可以在量化之前（或量化过程中）与多种算法结合使用。
本页对每种组合进行汇总，并从两个维度进行评级：

- **精度收益（Accuracy Gain）** —— 相比原始 AutoRound，该变换是否能提升量化模型的精度？
- **可部署性（Deployment）** —— 得到的模型当前是否能够实际部署/服务
  （内核支持、导出路径、真实推理引擎）？

## 图例

| 信号灯 | 含义                        |
|:---:|:--------------------------|
| 🟢  | 良好 —— 收益明显 / 可直接部署        |
| 🟡  | 部分 —— 收益有条件 / 支持有限或处于实验阶段 |
| 🔴  | 较差 —— 无明显收益 / 暂时无法部署      |

> **说明：** 「精度收益」被评为部分/较差可能有两方面原因：
> （1）当前AutoRound实现尚存在局限；（2）来自我们内部的主观评估。
> 二者都会随着实现完善与更多评测数据的补充而变化。

## 矩阵

| 组合                                 | 精度收益 | 可部署性 | 详情                                      | 命令行用法                            | 备注                                   | 参考文献                                                 |
|:-----------------------------------|:----:|:----:|:----------------------------------------|:---------------------------------|:-------------------------------------|:-----------------------------------------------------|
| AutoRound + AWQ（激活感知缩放）            |  🟢  |  🟢  | [awq_details](awq_details.md)           | `--algorithm awq,signround`      | 在量化激活（如 W4A4）场景下推荐使用。                | [arXiv:2306.00978](https://arxiv.org/abs/2306.00978) |
| AutoRound + Hadamard 旋转            |  🟢  |  🔴  | [rotation_details](rotation_details.md) | `--algorithm hadamard,signround` | 尤其适用于 INT4（W4A4）及部分 MXFP4 场景；暂无生产内核。 | [arXiv:2404.00456](https://arxiv.org/abs/2404.00456) |
| AutoRound + SpinQuant              |  🟡  |  🔴  | [rotation_details](rotation_details.md) | 仅 Python API                     | 需学习旋转矩阵，精度更高但有额外训练开销；暂无生产内核。         | [arXiv:2405.16406](https://arxiv.org/abs/2405.16406) |
| AutoRound + LFQ（logit 感知的末层块量化）    |  🔴  |  🟢  | [lfq_acc](lfq_acc.md)                   | `--enable_lfq`                   | 优化末层块以提升低比特生成质量。                     | [arXiv:2605.29756](https://arxiv.org/abs/2605.29756) |
| AutoRound + MX Attention（MXFP4 变体） |  🟡  |  🔴  | [mxnv_acc](mxnv_acc.md)                 | `--data_type mx_fp4_rceil_v2`    | 采用 7.25 作为 scale 计算的分母。              | [arXiv:2607.24377](https://arxiv.org/abs/2607.24377) |
| AutoRound + SVDQuant（低秩离群值吸收）      |  🟡  |  🟡  | [svdquant_details](svdquant_details.md) | `--algorithm svdquant,signround` | 推荐用于扩散模型；目前仅支持 FLUX。                 | [arXiv:2411.05007](https://arxiv.org/abs/2411.05007) |

