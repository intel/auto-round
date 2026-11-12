# AWQ 算法结果

本文总结了原生 AutoRound 流程与 AWQ 组合流程的实验对比。目标是评估在 `W4A16`、`MXFP4` 和 `INT8`（`W8A8`）方案下，在 RTN 或 AutoRound 之前加入 AWQ smoothing 是否能够提升准确率。

所有表格结果均基于 AutoRound 0.15.0 release，使用 `--format fake` 测得，并在 `mmlu,gsm8k,piqa,hellaswag,winogrande` 上评估。这五个常用 LLM 基准覆盖互补的误差类型：MMLU 侧重事实知识与综合理解，GSM8K 侧重数学推理，PIQA 侧重物理常识，HellaSwag 侧重常识句子补全，Winogrande 侧重代词与指代消解。报告该任务集合的平均分，可以降低单个任务波动对结论的影响，并更稳定地反映量化引入的精度变化。除非特别说明，其余量化参数均采用 AutoRound 默认设置。

## AWQ 算法特点

AWQ 是一种 activation-aware 的权重平滑方法。它根据校准激活搜索逐通道 scaling factor，通过缩放降低对量化更敏感的权重通道，然后再把变换后的模型交给最终量化算法处理。

因此，AWQ 与 RTN、AutoRound 是正交关系：AWQ 改变的是量化前的权重分布，而 RTN、AutoRound、AutoRound2 决定最终量化值如何生成。这样的设计允许 AWQ 与不同终端量化算法组合，而不需要改变这些算法本身的核心优化逻辑。

从资源角度看，AWQ 不会给最终量化模型引入持续性的 RAM 或 VRAM 负担。在校准阶段，`AWQ_RTN` 的主机 RAM 与其他流程基本处于同一水平，VRAM 明显低于迭代式 AutoRound 流程。AWQ 后接 AutoRound 或 AutoRound2 时，峰值 VRAM 主要由后续 AutoRound 优化决定，而不是 AWQ 本身带来的固定开销。

AWQ 的主要额外成本是校准时间。对于 PTQ 来说，这属于一次性的离线成本；当 AWQ 带来明确准确率收益时，尤其是在 MXFP4 和部分 W4A16 场景中，这个时间增加是可以接受的。

表中的配置对应以下 CLI 用法：

```bash
# 原始 RTN 基线
auto-round-rtn --model <model> --scheme <scheme> --format fake \
  --tasks "mmlu,gsm8k,piqa,hellaswag,winogrande"

# 优化版 RTN 基线
auto-round-opt-rtn --model <model> --scheme <scheme> --format fake \
  --tasks "mmlu,gsm8k,piqa,hellaswag,winogrande"

# 原生 AutoRound / 启用算法扩展的 AutoRound
auto-round --model <model> --scheme <scheme> --format fake \
  --algorithm auto_round \
  --tasks "mmlu,gsm8k,piqa,hellaswag,winogrande"

auto-round --model <model> --scheme <scheme> --format fake \
  --algorithm auto_round_v2 \
  --tasks "mmlu,gsm8k,piqa,hellaswag,winogrande"

# 历史 AWQ + 普通 RTN / AWQ + AutoRound / AWQ + 启用算法扩展的 AutoRound
auto-round --model <model> --scheme <scheme> --format fake \
  --algorithm awq,rtn --disable_opt_rtn \
  --tasks "mmlu,gsm8k,piqa,hellaswag,winogrande"

auto-round --model <model> --scheme <scheme> --format fake \
  --algorithm awq,auto_round \
  --tasks "mmlu,gsm8k,piqa,hellaswag,winogrande"

auto-round --model <model> --scheme <scheme> --format fake \
  --algorithm awq,auto_round_v2 \
  --tasks "mmlu,gsm8k,piqa,hellaswag,winogrande"
```

早期 `AWQ_RTN` 结果使用当时 AWQ 默认的普通 RTN，可显式表示为 `--algorithm awq,rtn --disable_opt_rtn`。最新 `main` 默认使用 AWQ + opt-RTN，可通过 `--algorithm awq` 或 `--algorithm awq,opt_rtn` 选择。明确标为 `AWQ + opt-RTN` 的结果使用的是优化 RTN，而不是旧的普通 RTN 默认配置。所有历史测量结果保持不变。

## 标定数据选择

AWQ 为激活统计以及 smoothing、clipping 搜索维护独立的校准预算。这些设置只影响 AWQ 的预处理；下游 RTN 或 AutoRound 量化算法仍使用完整的全局校准缓存，并自行执行最终的 scale 搜索。

| CLI 选项 | `AWQConfig` 字段 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `--awq-nsamples` | `awq_nsamples` | `128` | AWQ 激活统计、smoothing 和 clipping 使用的最大校准样本数。 |
| `--awq_seqlen` | `awq_seqlen` | `512` | AWQ 校准使用的最大序列长度；`<= 0` 表示关闭截断。 |
| `--awq_smooth_batch_size` | `smooth_batch_size` | `None` | AWQ scale 搜索中重放 parent 模块的 micro-batch 大小；较小的正数可降低峰值显存，`<= 0` 表示关闭 micro-batching。 |
| `--awq-v2-scale-search` | `v2_scale_search` | `auto` | 终端量化算法为 AutoRoundV2 时，AWQ 内部 QDQ 使用的 scale 搜索策略。 |

`--awq-nsamples` 独立于全局 `--nsamples`。AWQ 会从已缓存的校准输入中确定性地随机抽取一个子集（随机种子固定为 42），并在所有 block 中复用同一子集。全局缓存不会被修改，因此下游 RTN/AutoRound 量化算法和 FP reference forward 仍使用全部已缓存样本。将 `--awq-nsamples` 设置为大于等于 `--nsamples` 即可使用全部样本。

`--awq-v2-scale-search` 仅在终端量化算法为 AutoRoundV2 时生效，即选择 `--algorithm auto_round_v2`（或 `awq,auto_round_v2`）。终端量化算法为 RTN 时，该选项会被忽略。此策略只改变用于给 AWQ 候选打分的 QDQ；终端 AutoRoundV2 量化算法始终会执行自己的初始化搜索。

- `auto`（默认）：INT scheme 使用 `off`，MXFP、NVFP 等非 INT scheme 使用 `full`。
- `off`：AWQ 候选损失使用普通 RTN。速度最快，也是 INT 的推荐默认值。
- `full`：对每个 AWQ 候选都复现终端 V2 的优化 init-scale 搜索。速度最慢，但最贴近终端量化。
- `reuse`：每个 balance layer 只做一次优化 init-scale 搜索，并在所有 AWQ 候选中复用。速度介于 `off` 和 `full` 之间。

基于 WnA16 数据，最终决定 INT 数据类型使用 `off`：AWQ 搜索的平均时间开销降低约 30%，并且精度几乎没有损失。

当 smoothing 或 hard clipping 修改了某个 block 时，AWQ 会在量化前重新执行一次 FP reference forward，避免使用过期的 reference 作为量化目标或传递给下一个 block。

```bash
# INT：auto 解析为 off；AWQ 最多使用 128 个样本，终端量化使用 256 个。
auto-round --model <model> --scheme W4A16 --format fake \
  --algorithm awq,auto_round_v2 \
  --nsamples 256 --awq-nsamples 128

# MXFP4：auto 解析为 full；使用 reuse 可缩短 AWQ 校准时间。
auto-round --model <model> --scheme MXFP4 --format fake \
  --algorithm awq,auto_round_v2 \
  --awq-v2-scale-search reuse

# INT：显式对每个 AWQ 候选执行完整的优化搜索。
auto-round --model <model> --scheme W4A16 --format fake \
  --algorithm awq,auto_round_v2 \
  --awq-v2-scale-search full
```

这些参数也可以通过 `AWQConfig` 设置：

```python
from auto_round import AutoRound, AWQConfig, SignRoundV2Config

ar = AutoRound(
    model="<model>",
    scheme="W4A16",
    alg_configs=[
        AWQConfig(awq_nsamples=128, awq_seqlen=512, v2_scale_search="auto"),
        SignRoundV2Config(iters=200),
    ],
)
```

表中 `AR2` 表示启用 `--algorithm auto_round_v2`（原 `--enable_alg_ext`）的 AutoRound，`AWQ_AR2` 表示 AWQ 后接启用同一算法扩展的 AutoRound。准确率列以百分制展示，例如原始值 `0.7058` 写作 `70.58`。

AWQ 相关行使用默认 AWQ smoothing 流程。除非显式设置 `--awq_apply_clip`，否则不包含 AWQ weight clipping。

观察表中，`AVG Win Rate` 表示在当前 scheme 下所有可比较模型中，AWQ 组合方法 AVG 更高的比例；`Max AVG Delta Gain` 表示最大的 AVG 绝对提升（百分点）；`Max AVG Rel. Gain` 表示最大的 AVG 相对提升。对于 `AWQ_RTN`，当 `RTN` 和 `opt_rtn` 都存在时，会同时纳入这两个基线进行比较。

## MXFP4

<table border="1">
  <tr>
    <td>Model</td>
    <td>Config</td>
    <td>GSM8K (%)</td>
    <td>HellaSwag (%)</td>
    <td>MMLU (%)</td>
    <td>PIQA (%)</td>
    <td>WinoGrande (%)</td>
    <td>AVG (%)</td>
    <td>Time cost (s)</td>
    <td>RAM</td>
    <td>VRAM</td>
  </tr>
  <tr>
    <td rowspan="8">Llama-3.1-8B-I</td>
    <td>BF16</td>
    <td>70.51</td>
    <td>59.71</td>
    <td>68.37</td>
    <td>80.09</td>
    <td>73.40</td>
    <td>70.42</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr><td>RTN</td><td>50.11</td><td>55.54</td><td>57.34</td><td>76.01</td><td>70.01</td><td>61.80</td><td>88.00</td><td>9.83GB</td><td>2.99GB</td></tr>
  <tr><td>opt_rtn</td><td>52.08</td><td>55.82</td><td>57.86</td><td>76.88</td><td>71.11</td><td>62.75</td><td>100.00</td><td>10.02GB</td><td>3.84GB</td></tr>
  <tr><td>AR</td><td>60.73</td><td>55.29</td><td>61.34</td><td>77.53</td><td>70.01</td><td>64.98</td><td>976.75</td><td>24.33GB</td><td>16.86GB</td></tr>
  <tr><td>AR2</td><td>61.87</td><td>56.59</td><td>61.60</td><td>77.97</td><td>71.11</td><td>65.83</td><td>1151.81</td><td>24.38GB</td><td>17.97GB</td></tr>
  <tr><td>AWQ_RTN</td><td>50.27</td><td>55.51</td><td>58.55</td><td>76.93</td><td>70.56</td><td>62.36</td><td>993.51</td><td>23.36GB</td><td>7.91GB</td></tr>
  <tr><td>AWQ_AR</td><td>60.05</td><td>56.00</td><td>61.96</td><td>78.07</td><td>72.14</td><td>65.64</td><td>2042.67</td><td>24.35GB</td><td>18.80GB</td></tr>
  <tr><td><b>AWQ_AR2</b></td><td><b>62.02</b></td><td><b>56.40</b></td><td><b>61.84</b></td><td><b>78.62</b></td><td><b>72.14</b></td><td><b>66.20</b></td><td><b>2164.04</b></td><td><b>24.40GB</b></td><td><b>17.94GB</b></td></tr>
  <tr>
    <td rowspan="8">Qwen3-8B</td>
    <td>BF16</td>
    <td>87.41</td>
    <td>57.16</td>
    <td>72.92</td>
    <td>76.71</td>
    <td>67.72</td>
    <td>72.38</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr><td>RTN</td><td>81.27</td><td>52.24</td><td>66.13</td><td>73.01</td><td>63.14</td><td>67.16</td><td>119.00</td><td>7.81GB</td><td>2.66GB</td></tr>
  <tr><td>opt_rtn</td><td>83.55</td><td>52.60</td><td>66.10</td><td>73.50</td><td>64.40</td><td>68.03</td><td>125.00</td><td>7.99GB</td><td>3.42GB</td></tr>
  <tr><td>AR</td><td>80.82</td><td>52.64</td><td>67.98</td><td>74.81</td><td>66.30</td><td>68.51</td><td>999.52</td><td>22.31GB</td><td>15.43GB</td></tr>
  <tr><td>AR2</td><td>83.70</td><td>53.41</td><td>68.51</td><td>75.57</td><td>67.56</td><td>69.75</td><td>1240.32</td><td>22.34GB</td><td>16.18GB</td></tr>
  <tr><td>AWQ_RTN</td><td>82.79</td><td>53.54</td><td>67.48</td><td>75.19</td><td>67.17</td><td>69.23</td><td>1073.95</td><td>21.33GB</td><td>7.14GB</td></tr>
  <tr><td>AWQ_AR</td><td>79.91</td><td>53.07</td><td>68.99</td><td>75.35</td><td>67.32</td><td>68.93</td><td>2095.77</td><td>22.32GB</td><td>16.72GB</td></tr>
  <tr><td><b>AWQ_AR2</b></td><td><b>83.70</b></td><td><b>53.89</b></td><td><b>69.54</b></td><td><b>75.57</b></td><td><b>68.19</b></td><td><b>70.18</b></td><td><b>2244.87</b></td><td><b>22.37GB</b></td><td><b>16.03GB</b></td></tr>
</table>

### MXFP4 观察

| Comparison | AVG Win Rate | Max AVG Delta Gain | Max AVG Rel. Gain |
| --- | ---: | ---: | ---: |
| `AWQ_RTN` vs `RTN` / `opt_rtn` | 3/4 (75%) | +2.07 pts | +3.08% |
| `AWQ_AR` vs `AR` | 2/2 (100%) | +0.66 pts | +1.02% |
| `AWQ_AR2` vs `AR2` | 2/2 (100%) | +0.43 pts | +0.62% |

- 在这些实验中，MXFP4 仍然是 AWQ 组合收益最清晰的场景。AWQ 后接 AutoRound 或 AutoRound2 时，两个模型的 AVG 都有提升。
- `AWQ_AR2` 在两个模型上都取得 MXFP4 最高 AVG：Llama 为 66.20，Qwen 为 70.18。
- `AWQ_RTN` 在两个模型上都高于原始 `RTN`。这说明 AWQ smoothing 有价值，同时优化版 RTN 仍是低成本 MXFP4 强基线。
- 主要代价是一次性的运行时间。`AWQ_AR` 和 `AWQ_AR2` 约为对应原生 AutoRound 流程的 1.8x-2.1x，峰值 RAM/VRAM 仍主要由后续 AutoRound 优化器决定。

## W4A16

<table border="1">
  <tr>
    <td>Model</td>
    <td>Config</td>
    <td>GSM8K (%)</td>
    <td>HellaSwag (%)</td>
    <td>MMLU (%)</td>
    <td>PIQA (%)</td>
    <td>WinoGrande (%)</td>
    <td>AVG (%)</td>
    <td>Time cost (s)</td>
    <td>RAM</td>
    <td>VRAM</td>
  </tr>
  <tr>
    <td rowspan="8">Llama-3.1-8B-I</td>
    <td>BF16</td>
    <td>70.51</td>
    <td>59.71</td>
    <td>68.37</td>
    <td>80.09</td>
    <td>73.40</td>
    <td>70.42</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr><td>RTN</td><td>71.80</td><td>59.14</td><td>65.46</td><td>79.43</td><td>72.77</td><td>69.72</td><td>60.00</td><td>25.08GB</td><td>0.56GB</td></tr>
  <tr><td>opt_rtn</td><td>67.63</td><td>58.99</td><td>65.67</td><td>78.94</td><td>73.95</td><td>69.04</td><td>162.30</td><td>28.31GB</td><td>7.90GB</td></tr>
  <tr><td>AR</td><td>67.40</td><td>58.93</td><td>66.81</td><td>79.87</td><td>73.72</td><td>69.35</td><td>769.27</td><td>24.29GB</td><td>12.32GB</td></tr>
  <tr><td>AR2</td><td>69.60</td><td>59.13</td><td>66.81</td><td>79.87</td><td>73.09</td><td>69.70</td><td>918.59</td><td>24.04GB</td><td>13.26GB</td></tr>
  <tr><td>AWQ_RTN</td><td>70.58</td><td>59.09</td><td>65.89</td><td>79.87</td><td>72.85</td><td>69.66</td><td>1075.44</td><td>24.07GB</td><td>7.68GB</td></tr>
  <tr><td><b>AWQ_AR</b></td><td><b>73.24</b></td><td><b>59.11</b></td><td><b>66.97</b></td><td><b>80.14</b></td><td><b>72.93</b></td><td><b>70.48</b></td><td><b>1813.51</b></td><td><b>24.07GB</b></td><td><b>13.32GB</b></td></tr>
  <tr><td>AWQ_AR2</td><td>70.13</td><td>59.09</td><td>67.11</td><td>79.87</td><td>73.56</td><td>69.95</td><td>3533.26</td><td>24.07GB</td><td>13.17GB</td></tr>
  <tr>
    <td rowspan="8">Qwen3-8B</td>
    <td>BF16</td>
    <td>87.41</td>
    <td>57.16</td>
    <td>72.92</td>
    <td>76.71</td>
    <td>67.72</td>
    <td>72.38</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr><td>RTN</td><td>86.43</td><td>56.18</td><td>70.84</td><td>75.52</td><td>67.17</td><td>71.23</td><td>97.00</td><td>23.10GB</td><td>0.50GB</td></tr>
  <tr><td>opt_rtn</td><td>87.26</td><td>55.96</td><td>71.58</td><td>76.33</td><td>68.43</td><td>71.91</td><td>180.17</td><td>27.17GB</td><td>7.38GB</td></tr>
  <tr><td>AR</td><td>88.02</td><td>56.03</td><td>72.13</td><td>76.22</td><td>67.72</td><td>72.02</td><td>779.03</td><td>21.73GB</td><td>11.41GB</td></tr>
  <tr><td>AR2</td><td>86.96</td><td>55.95</td><td>72.38</td><td>76.71</td><td>68.27</td><td>72.05</td><td>1028.79</td><td>22.00GB</td><td>12.48GB</td></tr>
  <tr><td>AWQ_RTN</td><td>86.13</td><td>56.04</td><td>71.91</td><td>75.63</td><td>67.56</td><td>71.45</td><td>1108.32</td><td>22.02GB</td><td>6.93GB</td></tr>
  <tr><td>AWQ_AR</td><td>86.73</td><td>56.13</td><td>71.51</td><td>76.22</td><td>69.06</td><td>71.93</td><td>1957.19</td><td>21.94GB</td><td>12.07GB</td></tr>
  <tr><td><b>AWQ_AR2</b></td><td><b>86.50</b></td><td><b>56.07</b></td><td><b>72.23</b></td><td><b>76.55</b></td><td><b>69.14</b></td><td><b>72.10</b></td><td><b>3673.00</b></td><td><b>22.03GB</b></td><td><b>12.79GB</b></td></tr>
</table>

### W4A16 观察

| Comparison | AVG Win Rate | Max AVG Delta Gain | Max AVG Rel. Gain |
| --- | ---: | ---: | ---: |
| `AWQ_RTN` vs `RTN` / `opt_rtn` | 2/4 (50%) | +0.62 pts | +0.90% |
| `AWQ_AR` vs `AR` | 1/2 (50%) | +1.13 pts | +1.63% |
| `AWQ_AR2` vs `AR2` | 2/2 (100%) | +0.25 pts | +0.36% |

- W4A16 的 AWQ 收益依赖模型和后续量化方法。最明确的收益是 Llama 上 `AWQ_AR` 相比 `AR` 提升 +1.13 平均分，AVG 相对提升 +1.63%。
- `AWQ_RTN` 不适合作为 W4A16 的强默认项。它相对 `RTN` 和 `opt_rtn` 的 AVG 胜率为 2/4，但收益较小且不稳定。
- `AWQ_AR2` 在两个模型上都提升 AVG，但增益较小（+0.25 分和 +0.05 分），需要结合额外耗时判断是否值得启用。

## 8B 模型上的补充 W2/W3/W4 结果

以下为使用 `--format fake` 测得的 Qwen3-8B 与 Llama-3.1-8B-I 结果，覆盖 W2A16G64、W3A16 和 W4A16。若某个配置评测了多轮，则各项指标取多轮结果的算术平均；ARC_C 仅对记录了该指标的轮次取平均。ARC_C 完整列出，但不计入 AVG；AVG 仍为 GSM8K、HellaSwag、MMLU、PIQA、WinoGrande 五项平均。

### Qwen3-8B

| Scheme | Config | ARC_C | GSM8K | HellaSwag | MMLU | PIQA | WinoGrande | AVG |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W2A16G64 | awq_ar | 42.24 | 52.95 | 44.91 | 56.47 | 71.71 | 62.95 | 57.80 |
| W2A16G64 | arv2 | 43.26 | 58.68 | 45.21 | 57.89 | 71.82 | 66.69 | 60.06 |
| W2A16G64 | awq_arv2_full | 42.58 | 59.63 | 46.06 | 58.52 | 71.44 | 65.43 | 60.22 |
| W2A16G64 | awq_arv2_off | 43.43 | 62.17 | 45.73 | 58.70 | 72.88 | 66.45 | 61.19 |
| W2A16G64 | awq_arv2_reuse | 43.52 | 57.47 | 45.25 | 56.81 | 71.66 | 66.03 | 59.44 |
| W3A16 | awq_ar | 51.62 | 84.50 | 53.85 | 69.40 | 75.74 | 67.88 | 70.27 |
| W3A16 | awq_arv2_full | 51.62 | 83.06 | 53.80 | 69.56 | 76.22 | 68.62 | 70.25 |
| W3A16 | awq_arv2_grid9 | - | 84.46 | 53.77 | 69.36 | 76.39 | 68.43 | 70.48 |
| W3A16 | awq_arv2_off | 52.90 | 85.90 | 54.20 | 70.33 | 76.25 | 68.31 | 71.00 |
| W3A16 | awq_arv2_reuse | 52.90 | 86.24 | 53.52 | 70.59 | 76.22 | 68.94 | 71.11 |
| W3A16 | arv2 | 52.56 | 86.05 | 53.75 | 70.04 | 75.90 | 69.06 | 70.96 |
| W4A16 | awq_ar | 54.18 | 86.35 | 56.19 | 71.98 | 75.79 | 68.35 | 71.73 |
| W4A16 | arv2 | 54.18 | 86.66 | 56.05 | 72.12 | 76.55 | 69.69 | 72.21 |
| W4A16 | awq_arv2_full | 55.12 | 86.58 | 55.94 | 72.30 | 76.28 | 69.14 | 72.05 |
| W4A16 | awq_arv2_grid9 | 52.99 | 85.67 | 55.97 | 71.70 | 76.82 | 68.43 | 71.72 |
| W4A16 | awq_arv2_off | 53.75 | 86.43 | 56.11 | 72.06 | 76.28 | 69.93 | 72.16 |
| W4A16 | awq_arv2_reuse | 54.35 | 87.04 | 55.95 | 72.10 | 76.33 | 69.46 | 72.18 |

### Llama-3.1-8B-I

| Scheme | Config | ARC_C | GSM8K | HellaSwag | MMLU | PIQA | WinoGrande | AVG |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| W2A16G64 | awq_ar | 36.77 | 17.29 | 46.55 | 46.06 | 72.58 | 64.48 | 49.39 |
| W2A16G64 | arv2 | 38.74 | 22.21 | 47.08 | 48.39 | 73.34 | 66.85 | 51.57 |
| W2A16G64 | awq_arv2_full | 38.14 | 20.77 | 47.60 | 47.35 | 73.61 | 67.96 | 51.46 |
| W2A16G64 | awq_arv2_off | 38.40 | 24.11 | 47.49 | 50.44 | 73.88 | 68.19 | 52.82 |
| W2A16G64 | awq_arv2_reuse | 38.31 | 22.29 | 47.51 | 48.51 | 72.91 | 67.72 | 51.79 |
| W3A16 | awq_ar | 48.21 | 63.76 | 56.75 | 62.42 | 77.97 | 72.53 | 66.69 |
| W3A16 | arv2 | 47.95 | 64.75 | 57.00 | 63.26 | 78.84 | 73.01 | 67.37 |
| W3A16 | awq_arv2_full | 49.62 | 65.81 | 56.85 | 63.34 | 78.81 | 72.69 | 67.50 |
| W3A16 | awq_arv2_off | 49.53 | 68.50 | 57.00 | 63.03 | 79.27 | 72.10 | 67.98 |
| W3A16 | awq_arv2_reuse | 49.14 | 67.97 | 57.03 | 63.53 | 78.53 | 72.77 | 67.97 |
| W4A16 | awq_ar | 52.05 | 71.72 | 58.91 | 67.38 | 80.14 | 73.32 | 70.29 |
| W4A16 | arv2 | 52.65 | 71.72 | 58.98 | 67.37 | 80.09 | 73.95 | 70.42 |
| W4A16 | awq_arv2_full | 52.99 | 64.90 | 59.21 | 66.81 | 79.65 | 73.48 | 68.81 |
| W4A16 | awq_arv2_off | 51.96 | 73.09 | 58.94 | 67.38 | 79.92 | 73.24 | 70.51 |
| W4A16 | awq_arv2_reuse | 52.13 | 72.02 | 59.06 | 67.08 | 79.92 | 73.72 | 70.36 |

### W2/W3/W4 观察

- 在 W2A16G64 上，`AWQ + AR2` 明显优于 `AWQ + AR` 和 native `AR2`。`awq_arv2_off` 在 Qwen3-8B 上达到 61.19 AVG（`arv2` 为 60.06，`awq_ar` 为 57.80），在 Llama-3.1-8B-I 上达到 52.82（分别为 51.57 和 49.39）。
- 在 W3A16 上，`off` 与 `reuse` 仍是最强的 AWQ 策略：Qwen3-8B 上 AVG 为 71.00 / 71.11，Llama-3.1-8B-I 上为 67.98 / 67.97。不同 AWQ 配置之间的差距收窄到约 1 分。
- 在 W4A16 上，AWQ 的增益明显减弱，native `arv2` 已具备竞争力。Qwen3-8B 上 native `arv2` 最高（72.21，`awq_arv2_reuse` 为 72.18），Llama-3.1-8B-I 上 `awq_arv2_off` 最高（70.51，native `arv2` 为 70.42）。
- 对于WOQ INT scheme, 在这些 8B 对比中，`awq_arv2_full` 从未成为最佳策略；增加每个候选的搜索开销并未带来更高的 AVG。

## Qwen3.8-27B 案例：W4A16 与 MXFP4 W4A4

本节独立于上方的 `--format fake` 表格，使用导出 checkpoint 和 vLLM 在 B200 上评测，评测环境为 `lm-eval` 0.4.12、vLLM 0.27.1。HellaSwag、MMLU、PIQA、Winogrande 使用 `acc`，GSM8K 使用 `exact_match,strict-match`。

### 实验设置

- 模型：Qwen3.8-27B，Qwen3.5 hybrid-attention 架构，共 64 个 block。
- 校准：128 个样本，序列长度 2048；AWQ 最多使用 128 个样本和 512 token 校准序列。
- W4A16：INT W4A16，`--format auto_round`，并开启确定性算法。
- MXFP4：W4A4，`--format llm_compressor`；评测使用 `FlashInferMxFp4LinearKernel`。

各 recipe 对应的配置如下：

| 表中名称 | 算法参数 |
| --- | --- |
| AWQ + AR2 full / off / reuse | `--algorithm awq,auto_round_v2 --awq-v2-scale-search POLICY --iters 200` |
| AWQ + opt-RTN | `--algorithm awq,opt_rtn --iters 0` |
| Native AR2 full | `--algorithm auto_round_v2 --iters 200` |
| Native opt-RTN | `--algorithm opt-rtn --iters 0` |
| Native plain RTN | `--algorithm rtn --disable_opt_rtn --iters 0 --disable_model_free` |
| Model-free RTN | `--algorithm rtn --disable_opt_rtn --iters 0 --model_free` |

将 `POLICY` 替换为 `full`、`off` 或 `reuse`。

与前面的早期 `AWQ_RTN` 结果不同，下方 `AWQ + opt-RTN` 行使用的是优化 RTN。

### W4A16 INT

| 配置 | GSM8K (%) | HellaSwag (%) | MMLU (%) | PIQA (%) | WinoGrande (%) | AVG (%) | 量化时间 (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 96.97 | 63.68 | 83.43 | 80.90 | 75.45 | 80.09 | - |
| AWQ + AR2 full | 96.13 | 63.32 | 83.02 | 81.61 | 76.64 | 80.14 | 103 |
| AWQ + AR2 off | 96.66 | 63.39 | 82.42 | 81.39 | 76.40 | 80.06 | 76 |
| AWQ + AR2 reuse | 97.19 | 63.11 | 82.46 | 81.01 | 76.72 | 80.10 | 69 |
| AWQ + opt-RTN | 96.66 | 63.59 | 82.70 | 81.07 | 76.64 | 80.13 | 60 |

量化时间为参考值；本次运行中，`off` 相比 `full` 约节省26.3%的量化时间，AVG 仅下降 0.09 分。`reuse` 是 AWQ+AR2 中最省时的配置，AWQ + opt-RTN 整体最快，且 AVG 基本持平。

### MXFP4 W4A4

| 配置 | GSM8K (%) | HellaSwag (%) | MMLU (%) | PIQA (%) | WinoGrande (%) | AVG (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 96.97 | 63.68 | 83.43 | 80.90 | 75.45 | 80.09 |
| Native plain RTN | 94.84 | 61.12 | 80.08 | 80.30 | 74.35 | 78.14 |
| Model-free RTN | 94.54 | 61.12 | 80.08 | 80.30 | 74.35 | 78.08 |
| Native opt-RTN | 95.22 | 61.10 | 77.97 | 79.65 | 71.35 | 77.06 |
| Native AR2 full | 93.10 | 61.29 | 80.62 | 80.41 | 74.35 | 77.96 |
| AWQ + AR2 reuse | 94.01 | 61.18 | 80.37 | 80.47 | 73.95 | 78.00 |
| AWQ + AR2 full | 94.84 | 61.55 | 81.24 | 80.69 | 73.80 | 78.42 |
| AWQ + opt-RTN | 95.30 | 61.73 | 80.35 | 79.00 | 75.30 | 78.34 |

最佳 AWQ W4A4 结果是 AWQ + AR2 full，AVG 为 **78.42**，仅比 AWQ + opt-RTN 高 0.08 分。相对 native AR2，AWQ + AR2 full 提升 0.46 分；相对 native opt-RTN，AWQ + opt-RTN 提升 1.28 分。Native plain RTN 和 model-free RTN 分别达到 78.14 和 78.08，是强基线。

## INT8 / W8A8

<table border="1">
  <tr>
    <td>Model</td>
    <td>Config</td>
    <td>GSM8K (%)</td>
    <td>HellaSwag (%)</td>
    <td>MMLU (%)</td>
    <td>PIQA (%)</td>
    <td>WinoGrande (%)</td>
    <td>AVG (%)</td>
    <td>Time cost (s)</td>
    <td>RAM</td>
    <td>VRAM</td>
  </tr>
  <tr>
    <td rowspan="5">Llama-3.1-8B-I</td>
    <td>BF16</td>
    <td>70.51</td>
    <td>59.71</td>
    <td>68.37</td>
    <td>80.09</td>
    <td>73.40</td>
    <td>70.42</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr><td><b>RTN</b></td><td><b>72.25</b></td><td><b>59.73</b></td><td><b>68.10</b></td><td><b>79.71</b></td><td><b>74.82</b></td><td><b>70.92</b></td><td><b>98.00</b></td><td><b>9.53GB</b></td><td><b>0.85GB</b></td></tr>
  <tr><td>AR</td><td>68.99</td><td>59.63</td><td>67.87</td><td>79.71</td><td>74.11</td><td>70.06</td><td>988.81</td><td>23.87GB</td><td>13.38GB</td></tr>
  <tr><td>AWQ_RTN</td><td>72.63</td><td>59.54</td><td>68.13</td><td>79.82</td><td>73.72</td><td>70.77</td><td>1001.96</td><td>22.98GB</td><td>7.55GB</td></tr>
  <tr><td>AWQ_AR</td><td>70.05</td><td>59.65</td><td>68.20</td><td>79.87</td><td>74.11</td><td>70.38</td><td>1879.53</td><td>23.95GB</td><td>14.21GB</td></tr>
  <tr>
    <td rowspan="5">Qwen3-8B</td>
    <td>BF16</td>
    <td>87.41</td>
    <td>57.16</td>
    <td>72.92</td>
    <td>76.71</td>
    <td>67.72</td>
    <td>72.38</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr><td>RTN</td><td>87.49</td><td>56.93</td><td>72.53</td><td>76.01</td><td>63.95</td><td>71.38</td><td>88.00</td><td>7.58GB</td><td>0.75GB</td></tr>
  <tr><td><b>AR</b></td><td><b>88.10</b></td><td><b>56.74</b></td><td><b>72.46</b></td><td><b>76.06</b></td><td><b>68.11</b></td><td><b>72.29</b></td><td><b>905.88</b></td><td><b>21.83GB</b></td><td><b>12.37GB</b></td></tr>
  <tr><td>AWQ_RTN</td><td>86.58</td><td>57.05</td><td>72.57</td><td>76.77</td><td>68.35</td><td>72.26</td><td>1046.69</td><td>20.93GB</td><td>6.89GB</td></tr>
  <tr><td>AWQ_AR</td><td>86.81</td><td>56.77</td><td>72.58</td><td>76.61</td><td>68.11</td><td>72.18</td><td>2277.47</td><td>21.92GB</td><td>13.30GB</td></tr>
</table>

### INT8 观察

| Comparison | AVG Win Rate | Max AVG Delta Gain | Max AVG Rel. Gain |
| --- | ---: | ---: | ---: |
| `AWQ_RTN` vs `RTN` | 1/2 (50%) | +0.88 pts | +1.23% |
| `AWQ_AR` vs `AR` | 1/2 (50%) | +0.32 pts | +0.46% |

- INT8/W8A8 在这些实验中本身已经是高精度设置。大多数 INT8 recipe 的结果接近 BF16 参考水平，部分 recipe 的 AVG 还略高于 BF16。
- 对 AWQ 组合的 INT8 路径而言，AVG 变化幅度较小且依赖模型。`AWQ_RTN` 在 Qwen 上高于 `RTN`，但在 Llama 上略低于 `RTN`；`AWQ_AR` 在 Llama 上高于 `AR`，但在 Qwen 上略低于 `AR`。
- 考虑到 INT8 的原生基线已经较强，且 AWQ 带来的 AVG 变化较小，AWQ 更适合作为 INT8/W8A8 的可选探索路径，而不是首选默认路径。

## 总结

AWQ 更适合量化误差对 activation-aware weight smoothing 敏感的 scheme，例如 MXFP4。它可以提升 scheme 准确率，尤其是与启用算法扩展（`--algorithm auto_round_v2`）的 AutoRound 组合时。

AWQ 不会给最终量化模型增加持续性内存开销，并且与终端量化算法正交，因此可以与 RTN、AutoRound 或 AutoRound2 组合，而不改变这些算法本身的核心量化逻辑。

在 Qwen3.8-27B 案例中，W4A16 INT 对 AWQ 的 QDQ 策略不敏感：`off` 相比 `full` 降低了量化时间，AVG 仅变化 0.09 分；`reuse` 和 opt-RTN 也很有竞争力。对于 MXFP4 W4A4，AWQ + AR2 的 `full` 和 `reuse` 给出最强的 AWQ 结果。
