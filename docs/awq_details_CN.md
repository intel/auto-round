# AWQ 算法结果

本文总结了原生 AutoRound 流程与 AWQ 组合流程的实验对比。目标是评估在 `W4A16`、`MXFP4` 和 `INT8`（`W8A8`）方案下，在 RTN 或 AutoRound 之前加入 AWQ smoothing 是否能够提升准确率。

所有结果均使用 `--format fake` 测得，并在 `mmlu,gsm8k,piqa,hellaswag,winogrande` 上评估。这五个常用 LLM 基准覆盖互补的误差类型：MMLU 侧重事实知识与综合理解，GSM8K 侧重数学推理，PIQA 侧重物理常识，HellaSwag 侧重常识句子补全，Winogrande 侧重代词与指代消解。报告该任务集合的平均分，可以降低单个任务波动对结论的影响，并更稳定地反映量化引入的精度变化。除非特别说明，其余量化参数均采用 AutoRound 默认设置。

## AWQ 算法特点

AWQ 是一种 activation-aware 的权重平滑方法。它根据校准激活搜索逐通道 scaling factor，通过缩放降低对量化更敏感的权重通道，然后再把变换后的模型交给最终量化算法处理。

因此，AWQ 与 RTN、AutoRound 是正交关系：AWQ 改变的是量化前的权重分布，而 RTN、AutoRound、AutoRound2 决定最终量化值如何生成。这样的设计允许 AWQ 与不同终端量化算法组合，而不需要改变这些算法本身的核心优化逻辑。

从资源角度看，AWQ 不会给最终量化模型引入持续性的 RAM 或 VRAM 负担。在校准阶段，`AWQ_RTN` 的主机 RAM 与其他流程基本处于同一水平，VRAM 明显低于迭代式 AutoRound 流程。AWQ 后接 AutoRound 或 AutoRound2 时，峰值 VRAM 主要由后续 AutoRound 优化决定，而不是 AWQ 本身带来的固定开销。

AWQ 的主要额外成本是校准时间。对于 PTQ 来说，这属于一次性的离线成本；当 AWQ 带来明确准确率收益时，尤其是在 MXFP4 和部分 W4A16 场景中，这个时间增加是可以接受的。

表中的配置对应以下 CLI 用法：

```bash
# 原始 RTN 基线
auto-round --model <model> --scheme <scheme> --format fake \
  --iters 0 --disable_opt_rtn --disable_model_free \
  --tasks "mmlu,gsm8k,piqa,hellaswag,winogrande"

# 优化版 RTN 基线
auto-round --model <model> --scheme <scheme> --format fake \
  --iters 0 --enable_opt_rtn --disable_model_free \
  --tasks "mmlu,gsm8k,piqa,hellaswag,winogrande"

# 原生 AutoRound / 启用算法扩展的 AutoRound
auto-round --model <model> --scheme <scheme> --format fake \
  --algorithm auto_round \
  --tasks "mmlu,gsm8k,piqa,hellaswag,winogrande"

auto-round --model <model> --scheme <scheme> --format fake \
  --enable_alg_ext --algorithm auto_round \
  --tasks "mmlu,gsm8k,piqa,hellaswag,winogrande"

# AWQ + RTN / AWQ + AutoRound / AWQ + 启用算法扩展的 AutoRound
auto-round --model <model> --scheme <scheme> --format fake \
  --algorithm awq \
  --tasks "mmlu,gsm8k,piqa,hellaswag,winogrande"

auto-round --model <model> --scheme <scheme> --format fake \
  --algorithm awq,auto_round \
  --tasks "mmlu,gsm8k,piqa,hellaswag,winogrande"

auto-round --model <model> --scheme <scheme> --format fake \
  --enable_alg_ext --algorithm awq,auto_round \
  --tasks "mmlu,gsm8k,piqa,hellaswag,winogrande"
```

表中 `AR2` 表示启用 `--enable_alg_ext` 的 AutoRound，`AWQ_AR2` 表示 AWQ 后接启用 `--enable_alg_ext` 的 AutoRound。准确率列以百分制展示，例如原始值 `0.7058` 写作 `70.58`。BF16 作为参考基线放在每个模型分组的第一行。每个模型在每个 scheme 下平均分最高的量化行使用粗体标出。原始表格中有一行 Llama MXFP4 标为 `WQ_AR2`，本文将其视为 `AWQ_AR2`。

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
    <td>Timecost</td>
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
  <tr><td>RTN</td><td>59.06</td><td>58.88</td><td>64.63</td><td>78.94</td><td>72.77</td><td>66.86</td><td>72.30</td><td>25.52GB</td><td>0.83GB</td></tr>
  <tr><td>opt_rtn</td><td>65.20</td><td>58.91</td><td>65.04</td><td>79.05</td><td>73.48</td><td>68.34</td><td>62.00</td><td>25.65GB</td><td>1.65GB</td></tr>
  <!-- TODO: MXFP4 Llama-3.1 的 AR 数据异常（GSM8K=0.00，AVG=43.13），暂时视为初步结果，后续复核。 -->
  <tr><td>AR</td><td>0.00</td><td>31.20</td><td>48.50</td><td>75.24</td><td>60.69</td><td>43.13</td><td>959.21</td><td>25.39GB</td><td>16.30GB</td></tr>
  <tr><td>AR2</td><td>67.32</td><td>57.94</td><td>64.11</td><td>80.25</td><td>73.80</td><td>68.68</td><td>1104.91</td><td>24.38GB</td><td>17.78GB</td></tr>
  <tr><td>AWQ_RTN</td><td>66.26</td><td>58.36</td><td>66.12</td><td>79.00</td><td>73.95</td><td>68.74</td><td>1132.04</td><td>24.36GB</td><td>7.60GB</td></tr>
  <tr><td>AWQ_AR</td><td>69.75</td><td>57.74</td><td>65.71</td><td>79.27</td><td>72.69</td><td>69.03</td><td>2136.06</td><td>24.47GB</td><td>19.24GB</td></tr>
  <tr><td><b>AWQ_AR2</b></td><td><b>71.27</b></td><td><b>58.29</b></td><td><b>66.07</b></td><td><b>80.25</b></td><td><b>73.88</b></td><td><b>69.95</b></td><td><b>2274.64</b></td><td><b>24.40GB</b></td><td><b>18.12GB</b></td></tr>
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
  <tr><td>RTN</td><td>86.05</td><td>55.07</td><td>70.30</td><td>75.41</td><td>66.93</td><td>70.75</td><td>61.00</td><td>23.23GB</td><td>0.74GB</td></tr>
  <tr><td>opt_rtn</td><td>86.50</td><td>55.53</td><td>70.64</td><td>75.79</td><td>66.77</td><td>71.05</td><td>66.00</td><td>23.55GB</td><td>1.40GB</td></tr>
  <tr><td>AR</td><td>86.73</td><td>54.90</td><td>71.56</td><td>76.44</td><td>70.09</td><td>71.94</td><td>968.45</td><td>22.32GB</td><td>15.68GB</td></tr>
  <tr><td>AR2</td><td>86.58</td><td>55.07</td><td>71.86</td><td>76.61</td><td>69.77</td><td>71.98</td><td>1195.57</td><td>22.35GB</td><td>16.00GB</td></tr>
  <tr><td>AWQ_RTN</td><td>85.90</td><td>55.79</td><td>71.54</td><td>76.93</td><td>67.80</td><td>71.59</td><td>1088.34</td><td>22.34GB</td><td>6.93GB</td></tr>
  <tr><td>AWQ_AR</td><td>85.37</td><td>54.41</td><td>71.47</td><td>76.55</td><td>70.32</td><td>71.62</td><td>2346.84</td><td>22.45GB</td><td>16.75GB</td></tr>
  <tr><td><b>AWQ_AR2</b></td><td><b>86.66</b></td><td><b>55.19</b></td><td><b>71.99</b></td><td><b>77.37</b></td><td><b>70.80</b></td><td><b>72.40</b></td><td><b>2266.50</b></td><td><b>22.37GB</b></td><td><b>16.16GB</b></td></tr>
</table>

### MXFP4 观察

| Comparison | AVG Win Rate | Max AVG Delta Gain | Max AVG Rel. Gain |
| --- | ---: | ---: | ---: |
| `AWQ_RTN` vs `RTN` / `opt_rtn` | 4/4 (100%) | +1.88 pts | +2.81% |
| `AWQ_AR` vs `AR` | 1/2 (50%) | +25.90 pts | +60.05%* |
| `AWQ_AR2` vs `AR2` | 2/2 (100%) | +1.27 pts | +1.85% |

*Llama 的 `AWQ_AR` vs `AR` 结果受到原生 `AR` 基线异常影响（`GSM8K=0.00`，`AVG=43.13`），因此相对提升偏高。

- MXFP4 是 AWQ 组合收益最清晰的场景。`AWQ_RTN` 在两个模型上都高于 `RTN` 和 `opt_rtn`，`AWQ_AR2` 在两个模型上都高于 `AR2`。
- `AWQ_AR2` 是 MXFP4 中更稳定的组合：两个模型 AVG 均提升，最高带来 +1.27 分、+1.85% 的 AVG 相对提升。
- `AWQ_AR` 并不稳定。Llama 的大幅收益主要来自原生 `AR` 基线异常，而 Qwen 上 `AWQ_AR` 的 AVG 低于原生 `AR`。
- 代价仍然是耗时和显存。`AWQ_RTN` 约为 `RTN` 耗时的 18 倍；在 MXFP4 上，`AWQ_AR` / `AWQ_AR2` 约为原生 `AR` / `AR2` 的 2 倍。

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
    <td>Timecost</td>
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
    <td>Timecost</td>
    <td>RAM</td>
    <td>VRAM</td>
  </tr>
  <tr>
    <td rowspan="6">Llama-3.1-8B-I</td>
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
  <tr><td>RTN</td><td>71.49</td><td>59.64</td><td>68.37</td><td>80.30</td><td>74.27</td><td>70.81</td><td>58.00</td><td>25.00GB</td><td>0.56GB</td></tr>
  <tr><td>AR</td><td>69.90</td><td>59.62</td><td>68.36</td><td>80.36</td><td>73.64</td><td>70.38</td><td>836.48</td><td>24.90GB</td><td>13.63GB</td></tr>
  <tr><td>AWQ_RTN</td><td>73.24</td><td>59.66</td><td>68.30</td><td>80.09</td><td>73.32</td><td>70.92</td><td>1052.19</td><td>23.98GB</td><td>7.67GB</td></tr>
  <tr><td>AWQ_AR</td><td>68.99</td><td>59.77</td><td>68.12</td><td>80.30</td><td>73.09</td><td>70.05</td><td>1228.53</td><td>24.10GB</td><td>14.29GB</td></tr>
  <tr><td><b>RTN_smooth2048_clip</b></td><td><b>72.93</b></td><td><b>59.89</b></td><td><b>68.27</b></td><td><b>80.14</b></td><td><b>73.95</b></td><td><b>71.04</b></td><td><b>1668.43</b></td><td><b>24.50GB</b></td><td><b>18.46GB</b></td></tr>
  <tr>
    <td rowspan="6">Qwen3-8B</td>
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
  <tr><td>RTN</td><td>87.79</td><td>57.09</td><td>72.90</td><td>76.77</td><td>67.80</td><td>72.47</td><td>84.00</td><td>23.09GB</td><td>0.50GB</td></tr>
  <tr><td>AR</td><td>86.96</td><td>56.57</td><td>72.58</td><td>77.20</td><td>68.11</td><td>72.28</td><td>881.10</td><td>23.66GB</td><td>12.58GB</td></tr>
  <tr><td>AWQ_RTN</td><td>86.96</td><td>57.22</td><td>72.85</td><td>76.39</td><td>68.27</td><td>72.34</td><td>1143.95</td><td>21.93GB</td><td>6.93GB</td></tr>
  <tr><td>AWQ_AR</td><td>87.41</td><td>56.80</td><td>72.65</td><td>76.71</td><td>67.64</td><td>72.24</td><td>1324.51</td><td>22.85GB</td><td>13.40GB</td></tr>
  <tr><td><b>RTN_smooth2048_clip</b></td><td><b>87.87</b></td><td><b>57.14</b></td><td><b>72.90</b></td><td><b>76.66</b></td><td><b>67.96</b></td><td><b>72.51</b></td><td><b>1796.55</b></td><td><b>22.47GB</b></td><td><b>15.77GB</b></td></tr>
</table>

### INT8 观察

| Comparison | AVG Win Rate | Max AVG Delta Gain | Max AVG Rel. Gain |
| --- | ---: | ---: | ---: |
| `AWQ_RTN` vs `RTN` | 1/2 (50%) | +0.11 pts | +0.16% |
| `AWQ_AR` vs `AR` | 0/2 (0%) | -0.04 pts | -0.06% |

- INT8/W8A8 在这些实验中本身已经是高精度设置。大多数 INT8 recipe 的结果接近 BF16 参考水平，部分 recipe 的 AVG 还略高于 BF16。
- 实验性的 `RTN_smooth2048_clip` 行在两个模型的 INT8 表中都取得最高 AVG：Llama 为 71.04，Qwen 为 72.51。
- 对 AWQ 组合的 INT8 路径而言，AVG 变化幅度较小。Llama `AWQ_RTN` 相比 `RTN` 提升 +0.11 分（AVG 相对提升 +0.16%），其他 AWQ 组合结果则略低于对应的原生量化路径。
- 考虑到 INT8 的原生基线已经较强，AWQ 更适合作为 INT8/W8A8 的可选探索路径，而不是首选默认路径。

## 总结

### 优势

- AWQ 可以提升 MXFP4 准确率，尤其是与启用 `--enable_alg_ext` 的 AutoRound 组合时。
- 在本次 Llama W4A16 实验中，AWQ 有明显收益：`AWQ_AR` 平均分达到 70.48，略高于 BF16 参考结果。
- AWQ 不会给最终量化模型增加持续性内存开销。在校准阶段，`AWQ_RTN` 的 RAM 与其他流程接近，VRAM 明显低于迭代式 AutoRound 流程。
- AWQ 与终端量化算法正交，可以与 RTN、AutoRound 或 AutoRound2 组合，而不改变这些算法本身的核心量化逻辑。
- AWQ 更适合量化误差对 activation-aware weight smoothing 敏感的 scheme，例如 MXFP4。

### 劣势

- AWQ 会增加一次性的校准时间。即使是 `AWQ_RTN`，这些 8B 模型也需要约 18 分钟，而原始 `RTN` 约 1 分钟；当准确率收益明确时，这个离线时间成本是可以接受的。
- AWQ 对 W4A16 的提升不稳定；Qwen W4A16 的收益很小或接近中性。
- 对 INT8/W8A8，AWQ 带来的额外 AVG 收益有限，因为原生 INT8 基线已经接近 BF16。
- `AWQ_AR` 与 `AWQ_AR2` 会继承迭代式 AutoRound 优化的 GPU 内存特征。这些行里的较高 VRAM 应理解为后续优化器的成本，而不是 AWQ 给最终模型增加的持续性内存开销。
- AWQ 的收益依赖模型和 scheme，因此更适合作为可选组合，而不是通用默认配置。

## 使用建议

- 当目标是 MXFP4，或某个 W4A16 模型已验证存在收益时，可以将 AWQ 作为可选预处理算法使用。
- 当目标 scheme 显示出明确准确率收益时，可以把 AWQ 的额外校准时间视为可接受的离线成本。
- AWQ 内部 QDQ 默认使用原始 RTN 是合理的。这些结果没有显示出在 AWQ 校准期间使用优化版 RTN 有明确的准确率/成本收益。
- 对不含 AWQ 的终端 RTN，仍应保留优化版 RTN，尤其是 MXFP4，因为它能以较低成本提升准确率。
- 对 INT8/W8A8，除非具体模型显示出明确收益，否则优先使用原生 RTN。
