# AWQ Algorithm Results

This document summarizes an experimental comparison between native AutoRound flows and AWQ-composed flows. The goal is to evaluate whether applying AWQ smoothing before RTN or AutoRound improves accuracy for `W4A16`, `MXFP4`, and `INT8` (`W8A8`) schemes.

All results were measured with `--format fake` and evaluated on `mmlu,gsm8k,piqa,hellaswag,winogrande`. These five widely used LLM benchmarks cover complementary error modes: MMLU for factual knowledge and broad understanding, GSM8K for mathematical reasoning, PIQA for physical commonsense, HellaSwag for commonsense sentence completion, and Winogrande for pronoun and coreference resolution. Reporting the average score across this task set reduces sensitivity to single-task variance and provides a more stable view of quantization-induced accuracy changes. Unless noted otherwise, the remaining quantization settings follow the AutoRound defaults.

## AWQ Algorithm Characteristics

AWQ is an activation-aware weight smoothing method. It searches for per-channel scaling factors from calibration activations, applies the scaling to reduce quantization-sensitive weight channels, and then hands the transformed model to the terminal quantization algorithm.

AWQ changes the pre-quantization weight distribution, while RTN, AutoRound, and AutoRound2 decide how the final quantized values are produced. In this design, AWQ can be composed with different quantizers without changing their core optimization logic.

From the resource perspective, AWQ does not introduce a persistent RAM or VRAM burden in the quantized model. During calibration, `AWQ_RTN` stays around the same host RAM level as the other flows and uses much less VRAM than iterative AutoRound flows. When AWQ is followed by AutoRound or AutoRound2, the peak VRAM is mainly determined by the downstream AutoRound optimization rather than by AWQ itself.

The main added cost is timecost and the increase is acceptable when AWQ provides measurable accuracy gains, especially for MXFP4 and selected W4A16 cases.

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

In the result tables, `AR2` denotes AutoRound with `--enable_alg_ext`, and `AWQ_AR2` denotes AWQ followed by AutoRound with `--enable_alg_ext`.

In the observation tables, `AVG Win Rate` counts how often the AWQ-composed method improves model-level AVG accuracy for that scheme, `Max AVG Delta Gain` is the largest absolute AVG improvement in percentage points, and `Max AVG Rel. Gain` is the largest relative AVG improvement. For `AWQ_RTN`, the comparison includes both `RTN` and `opt_rtn` when both baselines are available.

## MXFP4

**Note: For Llama-3.1-Instruct, we recommend removing `@use_kernel_forward_from_hub("RMSNorm")` in [modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L52C1-L52C40); the tests below did not remove it, which can make the accuracy results inaccurate.**

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
  <tr>
    <td>RTN</td>
    <td>59.06</td>
    <td>58.88</td>
    <td>64.63</td>
    <td>78.94</td>
    <td>72.77</td>
    <td>66.86</td>
    <td>72.30</td>
    <td>25.52GB</td>
    <td>0.83GB</td>
  </tr>
  <tr>
    <td>opt_rtn</td>
    <td>65.20</td>
    <td>58.91</td>
    <td>65.04</td>
    <td>79.05</td>
    <td>73.48</td>
    <td>68.34</td>
    <td>62.00</td>
    <td>25.65GB</td>
    <td>1.65GB</td>
  </tr>
  <!-- TODO: MXFP4 Llama-3.1 AR data is anomalous (GSM8K=0.00, AVG=43.13); treat as preliminary and revisit for now. -->
  <tr>
    <td>AR</td>
    <td>0.00</td>
    <td>31.20</td>
    <td>48.50</td>
    <td>75.24</td>
    <td>60.69</td>
    <td>43.13</td>
    <td>959.21</td>
    <td>25.39GB</td>
    <td>16.30GB</td>
  </tr>
  <tr>
    <td>AR2</td>
    <td>67.32</td>
    <td>57.94</td>
    <td>64.11</td>
    <td>80.25</td>
    <td>73.80</td>
    <td>68.68</td>
    <td>1104.91</td>
    <td>24.38GB</td>
    <td>17.78GB</td>
  </tr>
  <tr>
    <td>AWQ_RTN</td>
    <td>66.26</td>
    <td>58.36</td>
    <td>66.12</td>
    <td>79.00</td>
    <td>73.95</td>
    <td>68.74</td>
    <td>1132.04</td>
    <td>24.36GB</td>
    <td>7.60GB</td>
  </tr>
  <tr>
    <td>AWQ_AR</td>
    <td>69.75</td>
    <td>57.74</td>
    <td>65.71</td>
    <td>79.27</td>
    <td>72.69</td>
    <td>69.03</td>
    <td>2136.06</td>
    <td>24.47GB</td>
    <td>19.24GB</td>
  </tr>
  <tr>
    <td><b>AWQ_AR2</b></td>
    <td><b>71.27</b></td>
    <td><b>58.29</b></td>
    <td><b>66.07</b></td>
    <td><b>80.25</b></td>
    <td><b>73.88</b></td>
    <td><b>69.95</b></td>
    <td><b>2274.64</b></td>
    <td><b>24.40GB</b></td>
    <td><b>18.12GB</b></td>
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
    <td>86.05</td>
    <td>55.07</td>
    <td>70.30</td>
    <td>75.41</td>
    <td>66.93</td>
    <td>70.75</td>
    <td>61.00</td>
    <td>23.23GB</td>
    <td>0.74GB</td>
  </tr>
  <tr>
    <td>opt_rtn</td>
    <td>86.50</td>
    <td>55.53</td>
    <td>70.64</td>
    <td>75.79</td>
    <td>66.77</td>
    <td>71.05</td>
    <td>66.00</td>
    <td>23.55GB</td>
    <td>1.40GB</td>
  </tr>
  <tr>
    <td>AR</td>
    <td>86.73</td>
    <td>54.90</td>
    <td>71.56</td>
    <td>76.44</td>
    <td>70.09</td>
    <td>71.94</td>
    <td>968.45</td>
    <td>22.32GB</td>
    <td>15.68GB</td>
  </tr>
  <tr>
    <td>AR2</td>
    <td>86.58</td>
    <td>55.07</td>
    <td>71.86</td>
    <td>76.61</td>
    <td>69.77</td>
    <td>71.98</td>
    <td>1195.57</td>
    <td>22.35GB</td>
    <td>16.00GB</td>
  </tr>
  <tr>
    <td>AWQ_RTN</td>
    <td>85.90</td>
    <td>55.79</td>
    <td>71.54</td>
    <td>76.93</td>
    <td>67.80</td>
    <td>71.59</td>
    <td>1088.34</td>
    <td>22.34GB</td>
    <td>6.93GB</td>
  </tr>
  <tr>
    <td>AWQ_AR</td>
    <td>85.37</td>
    <td>54.41</td>
    <td>71.47</td>
    <td>76.55</td>
    <td>70.32</td>
    <td>71.62</td>
    <td>2346.84</td>
    <td>22.45GB</td>
    <td>16.75GB</td>
  </tr>
  <tr>
    <td><b>AWQ_AR2</b></td>
    <td><b>86.66</b></td>
    <td><b>55.19</b></td>
    <td><b>71.99</b></td>
    <td><b>77.37</b></td>
    <td><b>70.80</b></td>
    <td><b>72.40</b></td>
    <td><b>2266.50</b></td>
    <td><b>22.37GB</b></td>
    <td><b>16.16GB</b></td>
  </tr>
</table>

### MXFP4 Observations

| Comparison | AVG Win Rate | Max AVG Delta Gain | Max AVG Rel. Gain |
| --- | ---: | ---: | ---: |
| `AWQ_RTN` vs `RTN` / `opt_rtn` | 4/4 (100%) | +1.88 pts | +2.81% |
| `AWQ_AR` vs `AR` | 1/2 (50%) | +25.90 pts | +60.05%* |
| `AWQ_AR2` vs `AR2` | 2/2 (100%) | +1.27 pts | +1.85% |

*The Llama `AWQ_AR` vs `AR` result is inflated by the unstable native `AR` baseline (`GSM8K=0.00`, `AVG=43.13`).

- MXFP4 is the strongest case for AWQ composition. `AWQ_RTN` improves over both `RTN` and `opt_rtn` on both models, and `AWQ_AR2` improves over `AR2` on both models.
- `AWQ_AR2` gives the most consistent MXFP4 profile: AVG improves on both models, with up to +1.27 points and +1.85% relative AVG gain.
- `AWQ_AR` is not uniformly reliable. The large Llama gain is driven by an unstable native `AR` baseline, while Qwen `AWQ_AR` is below native `AR` on AVG.
- The tradeoff remains runtime and VRAM. `AWQ_RTN` costs around 18x the `RTN` runtime, while `AWQ_AR` / `AWQ_AR2` cost about 2x native `AR` / `AR2` on MXFP4.

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
  <tr>
    <td>RTN</td>
    <td>71.49</td>
    <td>59.64</td>
    <td>68.37</td>
    <td>80.30</td>
    <td>74.27</td>
    <td>70.81</td>
    <td>58.00</td>
    <td>25.00GB</td>
    <td>0.56GB</td>
  </tr>
  <tr>
    <td>AR</td>
    <td>69.90</td>
    <td>59.62</td>
    <td>68.36</td>
    <td>80.36</td>
    <td>73.64</td>
    <td>70.38</td>
    <td>836.48</td>
    <td>24.90GB</td>
    <td>13.63GB</td>
  </tr>
  <tr>
    <td>AWQ_RTN</td>
    <td>73.24</td>
    <td>59.66</td>
    <td>68.30</td>
    <td>80.09</td>
    <td>73.32</td>
    <td>70.92</td>
    <td>1052.19</td>
    <td>23.98GB</td>
    <td>7.67GB</td>
  </tr>
  <tr>
    <td>AWQ_AR</td>
    <td>68.99</td>
    <td>59.77</td>
    <td>68.12</td>
    <td>80.30</td>
    <td>73.09</td>
    <td>70.05</td>
    <td>1228.53</td>
    <td>24.10GB</td>
    <td>14.29GB</td>
  </tr>
  <tr>
    <td><b>RTN_smooth2048_clip</b></td>
    <td><b>72.93</b></td>
    <td><b>59.89</b></td>
    <td><b>68.27</b></td>
    <td><b>80.14</b></td>
    <td><b>73.95</b></td>
    <td><b>71.04</b></td>
    <td><b>1668.43</b></td>
    <td><b>24.50GB</b></td>
    <td><b>18.46GB</b></td>
  </tr>
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
  <tr>
    <td>RTN</td>
    <td>87.79</td>
    <td>57.09</td>
    <td>72.90</td>
    <td>76.77</td>
    <td>67.80</td>
    <td>72.47</td>
    <td>84.00</td>
    <td>23.09GB</td>
    <td>0.50GB</td>
  </tr>
  <tr>
    <td>AR</td>
    <td>86.96</td>
    <td>56.57</td>
    <td>72.58</td>
    <td>77.20</td>
    <td>68.11</td>
    <td>72.28</td>
    <td>881.10</td>
    <td>23.66GB</td>
    <td>12.58GB</td>
  </tr>
  <tr>
    <td>AWQ_RTN</td>
    <td>86.96</td>
    <td>57.22</td>
    <td>72.85</td>
    <td>76.39</td>
    <td>68.27</td>
    <td>72.34</td>
    <td>1143.95</td>
    <td>21.93GB</td>
    <td>6.93GB</td>
  </tr>
  <tr>
    <td>AWQ_AR</td>
    <td>87.41</td>
    <td>56.80</td>
    <td>72.65</td>
    <td>76.71</td>
    <td>67.64</td>
    <td>72.24</td>
    <td>1324.51</td>
    <td>22.85GB</td>
    <td>13.40GB</td>
  </tr>
  <tr>
    <td><b>RTN_smooth2048_clip</b></td>
    <td><b>87.87</b></td>
    <td><b>57.14</b></td>
    <td><b>72.90</b></td>
    <td><b>76.66</b></td>
    <td><b>67.96</b></td>
    <td><b>72.51</b></td>
    <td><b>1796.55</b></td>
    <td><b>22.47GB</b></td>
    <td><b>15.77GB</b></td>
  </tr>
</table>

### INT8 Observations

| Comparison | AVG Win Rate | Max AVG Delta Gain | Max AVG Rel. Gain |
| --- | ---: | ---: | ---: |
| `AWQ_RTN` vs `RTN` | 1/2 (50%) | +0.11 pts | +0.16% |
| `AWQ_AR` vs `AR` | 0/2 (0%) | -0.04 pts | -0.06% |

- INT8/W8A8 is already a high-accuracy setting in these experiments. Most INT8 recipes are close to the BF16 reference, and some recipes slightly exceed BF16 AVG.
- The experimental `RTN_smooth2048_clip` rows give the best INT8 AVG for both models in this table: 71.04 on Llama and 72.51 on Qwen.
- In `RTN_smooth2048_clip`, `smooth2048` denotes AWQ calibration with sequence length 2048 (`awq_seqlen=2048`), and `clip` denotes AWQ weight clipping (`apply_clip=True`) enabled during the AWQ search.
- For AWQ-composed INT8, the AVG movement is small. Llama `AWQ_RTN` improves over `RTN` by +0.11 points (+0.16% relative), while the other AWQ-composed comparisons are slightly below their native counterparts.
- Given the strong INT8 baseline, AWQ is better positioned as an optional exploration path for INT8/W8A8 rather than a primary default.

## Summary

AWQ is most useful for schemes where quantization is sensitive to activation-aware weight smoothing, such as MXFP4. It can improve scheme accuracy, especially when combined with AutoRound plus `--enable_alg_ext`.
AWQ does not add persistent memory overhead to the quantized model and is orthogonal to terminal quantization algorithms, so it can be composed with RTN, AutoRound, or AutoRound2 without changing their core quantization logic.
