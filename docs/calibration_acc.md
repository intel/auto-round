## Calibration Dataset Study (WIP)

**Current Recommendation**: For 4-bit quantization, current results suggest **75% UltraChat + 25% OpenCode (Mix2 recipe)**. With the default `nsamples=128`, this recipe can be applied using `--dataset 'HuggingFaceH4/ultrachat_200k:apply_chat_template=true:num=96,nvidia/OpenCodeInstruct:num=32'`.

**Reading the tables:** Bold numbers are the currently reported summary scores; when repeated evaluations are available, the smaller line shows the original runs in their original order. A dash indicates a missing or unreported run. The displayed summary scores and raw results are preserved as supplied; not all summaries are based on five completed runs.

### Qwen3.8-27B-W4A16

<table>
<thead><tr><th>Calibration Dataset</th><th>AIME26</th><th>GPQA Diamond</th><th>IFBench Strict</th><th>MMLU-Pro</th><th>LiveCodeBench</th><th>SWE-mini</th><th>Terminal_Bench</th></tr></thead>
<tbody>
<tr><td>cci3hq</td><td align="center"><b>96.67</b><br><sub>96.67 / 90 / 100 / 100 / 96.67</sub></td><td align="center"><b>90.40</b><br><sub>91.92 / 90.4 / 88.89 / 91.41 / 89.39</sub></td><td align="center"><b>74.92</b><br><sub>73.67 / 76.16</sub></td><td align="center">85.03</td><td align="center">91.18</td><td align="center">–</td><td align="center">–</td></tr>
<tr><td>OpenCode</td><td align="center"><b>98.00</b><br><sub>96.67 / 96.67 / 100 / 100 / 96.67</sub></td><td align="center"><b>89.27</b><br><sub>89.39 / 87.88 / 90.4 / 89.39 / –</sub></td><td align="center"><b>75.23</b><br><sub>74 / 76.45</sub></td><td align="center">85.26</td><td align="center">92.51</td><td align="center">58.00</td><td align="center">49.40</td></tr>
<tr><td>pile10k</td><td align="center"><b>97.33</b><br><sub>100 / 93.33 / 100 / 93.33 / 100</sub></td><td align="center"><b>89.09</b><br><sub>87.88 / 88.38 / 88.89 / 89.9 / 90.4</sub></td><td align="center"><b>73.67</b><br><sub>72.33 / 75</sub></td><td align="center">85.00</td><td align="center">91.37</td><td align="center">–</td><td align="center">–</td></tr>
<tr><td>UltraChat</td><td align="center"><b>98.00</b><br><sub>100 / 93.33 / 100 / 100 / 96.67</sub></td><td align="center"><b>90.78</b><br><sub>90.91 / 91.41 / 90.4 / 90.4 / –</sub></td><td align="center"><b>74.75</b><br><sub>73.33 / 76.16</sub></td><td align="center">85.20</td><td align="center">91.75</td><td align="center">56.00</td><td align="center">41.60</td></tr>
<tr><td>UltraFineWeb</td><td align="center"><b>97.33</b><br><sub>96.67 / 100 / 100 / 96.67 / 93.33</sub></td><td align="center"><b>89.90</b><br><sub>89.39 / 90.4 / 90.4 / 90.4 / 88.89</sub></td><td align="center"><b>71.63</b><br><sub>70 / 73.26</sub></td><td align="center">84.96</td><td align="center">91.75</td><td align="center">–</td><td align="center">–</td></tr>
<tr><td>mix_a<br><sub>UltraChat 50% + OpenCode 25% + cci3hq 12.5% + pile10k 12.5%</sub></td><td align="center"><b>99.34</b><br><sub>100 / 100 / 100 / 96.7 / 100</sub></td><td align="center"><b>90.10</b><br><sub>90.9 / 89.4 / 90.9 / 90.4 / 88.9</sub></td><td align="center"><b>73.65</b><br><sub>72.3 / 75</sub></td><td align="center">85.40</td><td align="center"><b>92.70</b></td><td align="center"><b>60.00</b></td><td align="center">–</td></tr>
<tr><td>mix_b<br><sub>UltraChat 75% + OpenCode 25%</sub></td><td align="center"><b>98.02</b><br><sub>96.7 / 100 / 96.7 / 100 / 96.7</sub></td><td align="center"><b>90.4</b><br><sub>87.9 / 90.9 / 89.9 / 91.9 / 91.4</sub></td><td align="center"><b>76.00</b><br><sub>74.7 / 77.3</sub></td><td align="center"><b>85.50</b></td><td align="center">92.20</td><td align="center">56.90</td><td align="center">–</td></tr>
<tr><td>mix_b (chat template)<br><sub>UltraChat 75% (apply_chat_template=true) + OpenCode 25%</sub></td><td align="center"><b>99.18</b><br><sub>100 / 100 / 96.7 / 100 / –</sub></td><td align="center"><b>89.53</b><br><sub>87.9 / 90.4 / 89.4 / 90.4 / –</sub></td><td align="center">–</td><td align="center">–</td><td align="center">–</td><td align="center">–</td><td align="center">–</td></tr>
<tr><td>mix_c<br><sub>UltraChat 87.5% + OpenCode 12.5%</sub></td><td align="center"><b>96.67</b><br><sub>93.3 / 100 / 96.7 / – / –</sub></td><td align="center"><b>88.40</b><br><sub>88.4 / 90.4 / 86.4 / – / –</sub></td><td align="center"><b>73.00</b><br><sub>71.3 / 74.7</sub></td><td align="center">–</td><td align="center">–</td><td align="center">–</td><td align="center">–</td></tr>
</tbody>
</table>

### Gemma-4-12B-it-W4A16

<table>
<thead><tr><th>Calibration Dataset</th><th>AIME26</th><th>GPQA Diamond</th><th>IFBench Strict</th></tr></thead>
<tbody>
<tr><td>cci3hq</td><td align="center"><b>49.34</b><br><sub>46.7 / 46.7 / 50 / 50 / 53.3</sub></td><td align="center"><b>38.70</b><br><sub>42.4 / 35.4 / 36.9 / 40.4 / 38.4</sub></td><td align="center"><b>58.70</b><br><sub>59 / 58.4</sub></td></tr>
<tr><td>OpenCode</td><td align="center"><b>66.00</b><br><sub>63.3 / 76.7 / 60 / 60 / 70</sub></td><td align="center"><b>52.00</b><br><sub>51.5 / 50.5 / 54 / – / –</sub></td><td align="center"><b>64.25</b><br><sub>63.7 / 64.8</sub></td></tr>
<tr><td>pile10k</td><td align="center"><b>58.66</b><br><sub>53.3 / 53.3 / 66.7 / 56.7 / 63.3</sub></td><td align="center">–</td><td align="center"><b>65.95</b><br><sub>65.3 / 66.6</sub></td></tr>
<tr><td>UltraChat</td><td align="center"><b>68.00</b><br><sub>66.7 / 63.3 / 66.7 / 70 / 73.3</sub></td><td align="center"><b>62.60</b><br><sub>61.6 / 61.1 / 65.1 / – / –</sub></td><td align="center"><b>68.75</b><br><sub>67.7 / 69.8</sub></td></tr>
<tr><td>UltraFineWeb</td><td align="center"><b>57.34</b><br><sub>56.7 / 60 / 60 / 56.7 / 53.3</sub></td><td align="center">–</td><td align="center"><b>67.80</b><br><sub>66.7 / 68.9</sub></td></tr>
<tr><td>mix_a<br><sub>UltraChat 50% + OpenCode 25% + cci3hq 12.5% + pile10k 12.5%</sub></td><td align="center">56.70</td><td align="center">53.50</td><td align="center"><b>67.15</b><br><sub>66.3 / 68</sub></td></tr>
<tr><td>mix_b<br><sub>UltraChat 75% + OpenCode 25%</sub></td><td align="center"><b>76.70</b></td><td align="center">58.60</td><td align="center">67.95</td></tr>
<tr><td>mix_b (chat template)<br><sub>UltraChat 75% (apply_chat_template=true) + OpenCode 25%</sub></td><td align="center"><b>70.00</b><br><sub>73.3 / 66.7 / 76.7 / 80 / 53.3</sub></td><td align="center"><b>64.32</b><br><sub>65.1 / 63.6 / 65.7 / 62.1 / 65.1</sub></td><td align="center"><b>69.05</b><br><sub>68 / 70.1</sub></td></tr>
<tr><td>mix_c<br><sub>UltraChat 87.5% + OpenCode 12.5%</sub></td><td align="center"><b>66.68</b><br><sub>66.7 / 66.7 / 66.7 / 63.3 / 70</sub></td><td align="center"><b>62.82</b><br><sub>60.1 / 61.6 / 66.7 / 64.6 / 61.1</sub></td><td align="center"><b>68.75</b><br><sub>67.7 / 69.8</sub></td></tr>
</tbody>
</table>

### Reproduction Setup

**(1) Environment**
- Quantization & vLLM: Python 3.11.17, AutoRound 0.16.0, PyTorch 2.13.0+cu130, vLLM 0.29.0
- Standard Evaluation: Python 3.11.17, EvalScope 1.12.0
- Agent Evaluation: Python 3.12.15, EvalScope 1.12.0

**(2) Model Configurations**

**Qwen**
- Hardware: H200 / B200
- Quantization: W4A16g128
- Generation: temperature=1, top_p=1, top_k=20, min_p=0
- Max tokens: 131,072
- Evaluation: batch_size=32, timeout=14,400s, seed=42
- Thinking: enabled, preserve_thinking=true
- Streaming: disabled

**Gemma**
- Model: /models/gemma-4-12B-it
- Served model name: gemma4_bf16
- Hardware: A100
- Dtype: bfloat16
- Thinking: enabled
- Reasoning parser: gemma4
- Generation: temperature=1, top_p=0.95, top_k=64, min_p=0
- Repetition penalty: 1
- Max tokens: 32,768
- Evaluation: batch_size=16, timeout=7,200s
- Enforce eager: enabled
- Max model length: not explicitly set

**Benchmark Datasets**

| Benchmark | Samples |
|---|---:|
| AIME26 | 30 |
| GPQA Diamond | 198 |
| IFBench (Prompts & Instructions; Strict Prompt Strict, Instruction Strict = avg) | 300 & 344 |
| LiveCodeBench (release_v6) | 1,055 |
| MMLU-Pro | 12,032 |
| SWE-bench Verified Mini | 50 |
