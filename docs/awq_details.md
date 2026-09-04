# AWQ Algorithm Results

This document summarizes an experimental comparison between native AutoRound flows and AWQ-composed flows. The goal is to evaluate whether applying AWQ smoothing before RTN or AutoRound improves accuracy for `W4A16`, `MXFP4`, and `INT8` (`W8A8`) schemes.

All table results are based on the AutoRound 0.15.0 release, measured with `--format fake`, and evaluated on `mmlu,gsm8k,piqa,hellaswag,winogrande`. These five widely used LLM benchmarks cover complementary error modes: MMLU for factual knowledge and broad understanding, GSM8K for mathematical reasoning, PIQA for physical commonsense, HellaSwag for commonsense sentence completion, and Winogrande for pronoun and coreference resolution. Reporting the average score across this task set reduces sensitivity to single-task variance and provides a more stable view of quantization-induced accuracy changes. The results exclude earlier fake-format measurements affected by in-place model mutation during evaluation. Unless noted otherwise, the remaining quantization settings follow the AutoRound defaults.

## AWQ Algorithm Characteristics

AWQ is an activation-aware weight smoothing method. It searches for per-channel scaling factors from calibration activations, applies the scaling to reduce quantization-sensitive weight channels, and then hands the transformed model to the terminal quantization algorithm.

AWQ changes the pre-quantization weight distribution, while RTN, AutoRound, and AutoRound2 decide how the final quantized values are produced. In this design, AWQ can be composed with different quantizers without changing their core optimization logic.

From the resource perspective, AWQ does not introduce a persistent RAM or VRAM burden in the quantized model. During calibration, `AWQ_RTN` stays around the same host RAM level as the other flows and uses much less VRAM than iterative AutoRound flows. When AWQ is followed by AutoRound or AutoRound2, the peak VRAM is mainly determined by the downstream AutoRound optimization rather than by AWQ itself.

The main added cost is calibration runtime. This is a one-time offline cost and is acceptable when AWQ provides measurable accuracy gains, especially for MXFP4 and selected W4A16 cases.

The configurations in the tables map to the following CLI usage:

```bash
# Pure RTN baseline
auto-round-rtn --model <model> --scheme <scheme> --format fake \
  --tasks "mmlu,gsm8k,piqa,hellaswag,winogrande"

# Optimized RTN baseline
auto-round-opt-rtn --model <model> --scheme <scheme> --format fake \
  --tasks "mmlu,gsm8k,piqa,hellaswag,winogrande"

# Native AutoRound / AutoRound with algorithm extension
auto-round --model <model> --scheme <scheme> --format fake \
  --algorithm auto_round \
  --tasks "mmlu,gsm8k,piqa,hellaswag,winogrande"

auto-round --model <model> --scheme <scheme> --format fake \
  --enable_alg_ext --algorithm auto_round \
  --tasks "mmlu,gsm8k,piqa,hellaswag,winogrande"

# AWQ + RTN / AWQ + AutoRound / AWQ + AutoRound with algorithm extension
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

In the result tables, `AR2` denotes AutoRound with `--enable_alg_ext`, and `AWQ_AR2` denotes AWQ followed by AutoRound with `--enable_alg_ext`. Accuracy columns are reported as percentages, so a raw score such as `0.7058` is shown as `70.58`. BF16 is included as the reference baseline at the top of each model group. The quantized row with the highest model-level AVG in each scheme is highlighted in bold.

The AWQ rows use the default AWQ smoothing flow. They do not include AWQ weight clipping unless `--awq_apply_clip` is explicitly enabled.

In the observation tables, `AVG Win Rate` counts how often the AWQ-composed method improves model-level AVG accuracy for that scheme, `Max AVG Delta Gain` is the largest absolute AVG improvement in percentage points, and `Max AVG Rel. Gain` is the largest relative AVG improvement. For `AWQ_RTN`, the comparison includes both `RTN` and `opt_rtn` when both baselines are available.

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
  <tr>
    <td>RTN</td>
    <td>50.11</td>
    <td>55.54</td>
    <td>57.34</td>
    <td>76.01</td>
    <td>70.01</td>
    <td>61.80</td>
    <td>88.00</td>
    <td>9.83GB</td>
    <td>2.99GB</td>
  </tr>
  <tr>
    <td>opt_rtn</td>
    <td>52.08</td>
    <td>55.82</td>
    <td>57.86</td>
    <td>76.88</td>
    <td>71.11</td>
    <td>62.75</td>
    <td>100.00</td>
    <td>10.02GB</td>
    <td>3.84GB</td>
  </tr>
  <tr>
    <td>AR</td>
    <td>60.73</td>
    <td>55.29</td>
    <td>61.34</td>
    <td>77.53</td>
    <td>70.01</td>
    <td>64.98</td>
    <td>976.75</td>
    <td>24.33GB</td>
    <td>16.86GB</td>
  </tr>
  <tr>
    <td>AR2</td>
    <td>61.87</td>
    <td>56.59</td>
    <td>61.60</td>
    <td>77.97</td>
    <td>71.11</td>
    <td>65.83</td>
    <td>1151.81</td>
    <td>24.38GB</td>
    <td>17.97GB</td>
  </tr>
  <tr>
    <td>AWQ_RTN</td>
    <td>50.27</td>
    <td>55.51</td>
    <td>58.55</td>
    <td>76.93</td>
    <td>70.56</td>
    <td>62.36</td>
    <td>993.51</td>
    <td>23.36GB</td>
    <td>7.91GB</td>
  </tr>
  <tr>
    <td>AWQ_AR</td>
    <td>60.05</td>
    <td>56.00</td>
    <td>61.96</td>
    <td>78.07</td>
    <td>72.14</td>
    <td>65.64</td>
    <td>2042.67</td>
    <td>24.35GB</td>
    <td>18.80GB</td>
  </tr>
  <tr>
    <td><b>AWQ_AR2</b></td>
    <td><b>62.02</b></td>
    <td><b>56.40</b></td>
    <td><b>61.84</b></td>
    <td><b>78.62</b></td>
    <td><b>72.14</b></td>
    <td><b>66.20</b></td>
    <td><b>2164.04</b></td>
    <td><b>24.40GB</b></td>
    <td><b>17.94GB</b></td>
  </tr>
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
  <tr>
    <td>RTN</td>
    <td>81.27</td>
    <td>52.24</td>
    <td>66.13</td>
    <td>73.01</td>
    <td>63.14</td>
    <td>67.16</td>
    <td>119.00</td>
    <td>7.81GB</td>
    <td>2.66GB</td>
  </tr>
  <tr>
    <td>opt_rtn</td>
    <td>83.55</td>
    <td>52.60</td>
    <td>66.10</td>
    <td>73.50</td>
    <td>64.40</td>
    <td>68.03</td>
    <td>125.00</td>
    <td>7.99GB</td>
    <td>3.42GB</td>
  </tr>
  <tr>
    <td>AR</td>
    <td>80.82</td>
    <td>52.64</td>
    <td>67.98</td>
    <td>74.81</td>
    <td>66.30</td>
    <td>68.51</td>
    <td>999.52</td>
    <td>22.31GB</td>
    <td>15.43GB</td>
  </tr>
  <tr>
    <td>AR2</td>
    <td>83.70</td>
    <td>53.41</td>
    <td>68.51</td>
    <td>75.57</td>
    <td>67.56</td>
    <td>69.75</td>
    <td>1240.32</td>
    <td>22.34GB</td>
    <td>16.18GB</td>
  </tr>
  <tr>
    <td>AWQ_RTN</td>
    <td>82.79</td>
    <td>53.54</td>
    <td>67.48</td>
    <td>75.19</td>
    <td>67.17</td>
    <td>69.23</td>
    <td>1073.95</td>
    <td>21.33GB</td>
    <td>7.14GB</td>
  </tr>
  <tr>
    <td>AWQ_AR</td>
    <td>79.91</td>
    <td>53.07</td>
    <td>68.99</td>
    <td>75.35</td>
    <td>67.32</td>
    <td>68.93</td>
    <td>2095.77</td>
    <td>22.32GB</td>
    <td>16.72GB</td>
  </tr>
  <tr>
    <td><b>AWQ_AR2</b></td>
    <td><b>83.70</b></td>
    <td><b>53.89</b></td>
    <td><b>69.54</b></td>
    <td><b>75.57</b></td>
    <td><b>68.19</b></td>
    <td><b>70.18</b></td>
    <td><b>2244.87</b></td>
    <td><b>22.37GB</b></td>
    <td><b>16.03GB</b></td>
  </tr>
</table>

### MXFP4 Observations

| Comparison | AVG Win Rate | Max AVG Delta Gain | Max AVG Rel. Gain |
| --- | ---: | ---: | ---: |
| `AWQ_RTN` vs `RTN` / `opt_rtn` | 3/4 (75%) | +2.07 pts | +3.08% |
| `AWQ_AR` vs `AR` | 2/2 (100%) | +0.66 pts | +1.02% |
| `AWQ_AR2` vs `AR2` | 2/2 (100%) | +0.43 pts | +0.62% |

- MXFP4 remains the clearest case for AWQ composition in these experiments. AWQ improves AVG for both AutoRound and AutoRound2 on both models.
- `AWQ_AR2` gives the highest MXFP4 AVG for both models, reaching 66.20 on Llama and 70.18 on Qwen.
- `AWQ_RTN` improves over plain `RTN` on both models. This suggests AWQ smoothing is useful, while optimized RTN remains a strong low-cost MXFP4 baseline.
- The main tradeoff is one-time runtime. `AWQ_AR` and `AWQ_AR2` take roughly 1.8x-2.1x the corresponding native AutoRound runtime, while peak RAM/VRAM stays in the same range as the downstream AutoRound optimizer.

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
  <tr>
    <td>RTN</td>
    <td>71.80</td>
    <td>59.14</td>
    <td>65.46</td>
    <td>79.43</td>
    <td>72.77</td>
    <td>69.72</td>
    <td>60.00</td>
    <td>25.08GB</td>
    <td>0.56GB</td>
  </tr>
  <tr>
    <td>opt_rtn</td>
    <td>67.63</td>
    <td>58.99</td>
    <td>65.67</td>
    <td>78.94</td>
    <td>73.95</td>
    <td>69.04</td>
    <td>162.30</td>
    <td>28.31GB</td>
    <td>7.90GB</td>
  </tr>
  <tr>
    <td>AR</td>
    <td>67.40</td>
    <td>58.93</td>
    <td>66.81</td>
    <td>79.87</td>
    <td>73.72</td>
    <td>69.35</td>
    <td>769.27</td>
    <td>24.29GB</td>
    <td>12.32GB</td>
  </tr>
  <tr>
    <td>AR2</td>
    <td>69.60</td>
    <td>59.13</td>
    <td>66.81</td>
    <td>79.87</td>
    <td>73.09</td>
    <td>69.70</td>
    <td>918.59</td>
    <td>24.04GB</td>
    <td>13.26GB</td>
  </tr>
  <tr>
    <td>AWQ_RTN</td>
    <td>70.58</td>
    <td>59.09</td>
    <td>65.89</td>
    <td>79.87</td>
    <td>72.85</td>
    <td>69.66</td>
    <td>1075.44</td>
    <td>24.07GB</td>
    <td>7.68GB</td>
  </tr>
  <tr>
    <td><b>AWQ_AR</b></td>
    <td><b>73.24</b></td>
    <td><b>59.11</b></td>
    <td><b>66.97</b></td>
    <td><b>80.14</b></td>
    <td><b>72.93</b></td>
    <td><b>70.48</b></td>
    <td><b>1813.51</b></td>
    <td><b>24.07GB</b></td>
    <td><b>13.32GB</b></td>
  </tr>
  <tr>
    <td>AWQ_AR2</td>
    <td>70.13</td>
    <td>59.09</td>
    <td>67.11</td>
    <td>79.87</td>
    <td>73.56</td>
    <td>69.95</td>
    <td>3533.26</td>
    <td>24.07GB</td>
    <td>13.17GB</td>
  </tr>
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
  <tr>
    <td>RTN</td>
    <td>86.43</td>
    <td>56.18</td>
    <td>70.84</td>
    <td>75.52</td>
    <td>67.17</td>
    <td>71.23</td>
    <td>97.00</td>
    <td>23.10GB</td>
    <td>0.50GB</td>
  </tr>
  <tr>
    <td>opt_rtn</td>
    <td>87.26</td>
    <td>55.96</td>
    <td>71.58</td>
    <td>76.33</td>
    <td>68.43</td>
    <td>71.91</td>
    <td>180.17</td>
    <td>27.17GB</td>
    <td>7.38GB</td>
  </tr>
  <tr>
    <td>AR</td>
    <td>88.02</td>
    <td>56.03</td>
    <td>72.13</td>
    <td>76.22</td>
    <td>67.72</td>
    <td>72.02</td>
    <td>779.03</td>
    <td>21.73GB</td>
    <td>11.41GB</td>
  </tr>
  <tr>
    <td>AR2</td>
    <td>86.96</td>
    <td>55.95</td>
    <td>72.38</td>
    <td>76.71</td>
    <td>68.27</td>
    <td>72.05</td>
    <td>1028.79</td>
    <td>22.00GB</td>
    <td>12.48GB</td>
  </tr>
  <tr>
    <td>AWQ_RTN</td>
    <td>86.13</td>
    <td>56.04</td>
    <td>71.91</td>
    <td>75.63</td>
    <td>67.56</td>
    <td>71.45</td>
    <td>1108.32</td>
    <td>22.02GB</td>
    <td>6.93GB</td>
  </tr>
  <tr>
    <td>AWQ_AR</td>
    <td>86.73</td>
    <td>56.13</td>
    <td>71.51</td>
    <td>76.22</td>
    <td>69.06</td>
    <td>71.93</td>
    <td>1957.19</td>
    <td>21.94GB</td>
    <td>12.07GB</td>
  </tr>
  <tr>
    <td><b>AWQ_AR2</b></td>
    <td><b>86.50</b></td>
    <td><b>56.07</b></td>
    <td><b>72.23</b></td>
    <td><b>76.55</b></td>
    <td><b>69.14</b></td>
    <td><b>72.10</b></td>
    <td><b>3673.00</b></td>
    <td><b>22.03GB</b></td>
    <td><b>12.79GB</b></td>
  </tr>
</table>

### W4A16 Observations

| Comparison | AVG Win Rate | Max AVG Delta Gain | Max AVG Rel. Gain |
| --- | ---: | ---: | ---: |
| `AWQ_RTN` vs `RTN` / `opt_rtn` | 2/4 (50%) | +0.62 pts | +0.90% |
| `AWQ_AR` vs `AR` | 1/2 (50%) | +1.13 pts | +1.63% |
| `AWQ_AR2` vs `AR2` | 2/2 (100%) | +0.25 pts | +0.36% |

- W4A16 shows model-dependent behavior. The clearest gain is Llama `AWQ_AR` over `AR`, with +1.13 average points and +1.63% relative AVG gain.
- `AWQ_RTN` is not a strong W4A16 default. It wins 2/4 AVG comparisons against `RTN` and `opt_rtn`.
- `AWQ_AR2` improves AVG on both models, but the gain is small compared with the added runtime.

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
  <tr>
    <td>RTN</td>
    <td><b>72.25</b></td>
    <td><b>59.73</b></td>
    <td><b>68.10</b></td>
    <td><b>79.71</b></td>
    <td><b>74.82</b></td>
    <td><b>70.92</b></td>
    <td><b>98.00</b></td>
    <td><b>9.53GB</b></td>
    <td><b>0.85GB</b></td>
  </tr>
  <tr>
    <td>AR</td>
    <td>68.99</td>
    <td>59.63</td>
    <td>67.87</td>
    <td>79.71</td>
    <td>74.11</td>
    <td>70.06</td>
    <td>988.81</td>
    <td>23.87GB</td>
    <td>13.38GB</td>
  </tr>
  <tr>
    <td>AWQ_RTN</td>
    <td>72.63</td>
    <td>59.54</td>
    <td>68.13</td>
    <td>79.82</td>
    <td>73.72</td>
    <td>70.77</td>
    <td>1001.96</td>
    <td>22.98GB</td>
    <td>7.55GB</td>
  </tr>
  <tr>
    <td>AWQ_AR</td>
    <td>70.05</td>
    <td>59.65</td>
    <td>68.20</td>
    <td>79.87</td>
    <td>74.11</td>
    <td>70.38</td>
    <td>1879.53</td>
    <td>23.95GB</td>
    <td>14.21GB</td>
  </tr>
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
  <tr>
    <td>RTN</td>
    <td>87.49</td>
    <td>56.93</td>
    <td>72.53</td>
    <td>76.01</td>
    <td>63.95</td>
    <td>71.38</td>
    <td>88.00</td>
    <td>7.58GB</td>
    <td>0.75GB</td>
  </tr>
  <tr>
    <td><b>AR</b></td>
    <td><b>88.10</b></td>
    <td><b>56.74</b></td>
    <td><b>72.46</b></td>
    <td><b>76.06</b></td>
    <td><b>68.11</b></td>
    <td><b>72.29</b></td>
    <td><b>905.88</b></td>
    <td><b>21.83GB</b></td>
    <td><b>12.37GB</b></td>
  </tr>
  <tr>
    <td>AWQ_RTN</td>
    <td>86.58</td>
    <td>57.05</td>
    <td>72.57</td>
    <td>76.77</td>
    <td>68.35</td>
    <td>72.26</td>
    <td>1046.69</td>
    <td>20.93GB</td>
    <td>6.89GB</td>
  </tr>
  <tr>
    <td>AWQ_AR</td>
    <td>86.81</td>
    <td>56.77</td>
    <td>72.58</td>
    <td>76.61</td>
    <td>68.11</td>
    <td>72.18</td>
    <td>2277.47</td>
    <td>21.92GB</td>
    <td>13.30GB</td>
  </tr>
</table>

### INT8 Observations

| Comparison | AVG Win Rate | Max AVG Delta Gain | Max AVG Rel. Gain |
| --- | ---: | ---: | ---: |
| `AWQ_RTN` vs `RTN` | 1/2 (50%) | +0.88 pts | +1.23% |
| `AWQ_AR` vs `AR` | 1/2 (50%) | +0.32 pts | +0.46% |

- INT8/W8A8 is already a high-accuracy setting in these experiments. Most INT8 recipes are close to the BF16 reference, and some recipes slightly exceed BF16 AVG.
- For AWQ-composed INT8, the AVG movement is small and model-dependent. `AWQ_RTN` improves over `RTN` on Qwen but is slightly below `RTN` on Llama; `AWQ_AR` improves over `AR` on Llama but is slightly below `AR` on Qwen.
- Given the strong INT8 baseline and small AWQ deltas, AWQ is better positioned as an optional exploration path for INT8/W8A8 rather than a primary default.

## Summary

AWQ is most useful for schemes where quantization is sensitive to activation-aware weight smoothing, such as MXFP4. It can improve scheme accuracy, especially when combined with AutoRound plus `--enable_alg_ext`.

AWQ does not add persistent memory overhead to the quantized model and is orthogonal to terminal quantization algorithms, so it can be composed with RTN, AutoRound, or AutoRound2 without changing their core quantization logic.
