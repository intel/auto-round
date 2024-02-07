<div align="center">

AutoRound
===========================
<h3> Advanced Weight-Only Quantization Algorithm for LLMs</h3>

[![python](https://img.shields.io/badge/python-3.8%2B-blue)](https://github.com/intel/auto-round)
[![version](https://img.shields.io/badge/release-0.1-green)](https://github.com/intel/auto-round)
[![license](https://img.shields.io/badge/license-Apache%202-blue)](https://github.com/intel/auto-round/blob/main/LICENSE)
---
<div align="left">

AutoRound is an advanced weight-only quantization algorithm for low-bits LLM inference. It's tailored for a wide range of models and consistently delivers noticeable improvements, often significantly outperforming SignRound with the cost of more tuning time for quantization.

## Prerequisites
- Python 3.9 or higher

## Installation
### Build from Source
```bash
pip install -r requirements.txt
python setup.py install
```
## Usage




### On CPU/GPU

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=True, torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
bits, group_size, scheme = 4, 128, "asym"
autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, scheme=scheme)
autoround.quantize()

# Intel CPU Inference, For now only support llama, mistral and gpt-j.
# then follow ITREX(https://github.com/intel/intel-extension-for-transformers/tree/main/intel_extension_for_transformers/llm/runtime/neural_speed) to load the model and do inference
# currently please install neural-speed (https://github.com/intel/neural-speed) from source
output_dir = "./tmp_autoround"
autoround.export(output_dir)

from intel_extension_for_transformers.transformers import AutoModelForCausalLM, WeightOnlyQuantConfig

woq_config = WeightOnlyQuantConfig(group_size=group_size, scheme=scheme, use_autoround=True)  ##only supports 4 bits currently
prompt = "Once upon a time, a little girl"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=woq_config, trust_remote_code=True,device="cpu")
outputs = model.generate(inputs, max_new_tokens=30)
```


### Tuning on Intel Gaudi2

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=True, torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
bits, group_size, scheme = 4, 128, "asym"

# need to load model first, then import
from auto_round import AutoRound
autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, scheme=scheme,
                      device="hpu", scale_dtype="bf16", amp=False)
autoround.quantize()

# Intel CPU Inference, Currently, llama, bloom, and mistral are supported.
output_dir = "/path/to/quantized_model"
autoround.export(output_dir)
# then follow ITREX(https://github.com/intel/intel-extension-for-transformers/tree/main/intel_extension_for_transformers/llm/runtime/neural_speed) to load the model and do inference
```


<details>
  <summary>Detailed Hyperparameters</summary>

- `model`: The PyTorch model to be quantized.
            
- `tokenizer`: An optional tokenizer for processing input data. If none is provided, a dataloader must be supplied.
  
- `bits (int)`: Number of bits for quantization (default is 4).
  
- `group_size (int)`: Size of the quantization group (default is 128).

- `scheme (str)`: The quantization scheme (sym/asym) to be used (default is "asym").
  
- `use_quant_input (bool)`: Whether to use the output of the previous quantized block as the input for the current block (default is True).
  
- `enable_minmax_tuning (bool)`: Whether to enable weight min-max tuning (default is True).
  
- `iters (int)`: Number of tuning iterations (default is 200).
  
- `lr (float)`: The learning rate for rounding value (default is None, it will be set to 1.0/iters automatically).
  
- `minmax_lr (float)`: The learning rate for min-max tuning (default is None, it will be set to lr automatically).
  
- `n_samples (int)`: Number of samples for tuning (default is 512).
  
- `seqlen (int)`: Data length of the sequence for tuning (default is 2048).
  
- `bs (int)`: Batch size for training (default is 8).
  
- `amp (bool)`: Whether to use automatic mixed precision (default is True).
  
- `n_blocks (int)`: Packing several blocks as one for tuning together (default is 1).
  
- `gradient_accumulate_steps (int)`: Number of gradient accumulation steps (default is 1).
  
- `low_gpu_mem_usage (bool)`: Whether to save GPU memory at the cost of a little tuning time (default is True).
  
- `dataset_name (str)`: The default dataset name for tuning (default is "NeelNanda/pile-10k").
  
- `dataset_split (str)`: The split of the dataset to be used for tuning (default is "train").
  
- `dataloader`: The dataloader for tuning data.
  
- `weight_config (dict)`: Configuration for weight quantization (default is an empty dictionary), mainly for mixed bits or mixed precision.
  
- `device`: The device to be used for tuning (default is "cuda:0").

</details>


## Validated Models
For wikitext2/ptb-new/c4-new ppl, we follow the code of gptq and set the sequence length to 2048. For lm-eval wikitext ppl, we adopt lm-eval. The quantization configure is W4G128.

<table border="1">
  <tr>
    <th>Model</th>
    <th>Method </th>
    <th>Acc AVG.</th>
    <th>MMLU</th>
    <th>Lamb.</th>
    <th>Hella.</th>
    <th>Wino.</th>
    <th>Piqa</th>
    <th>Truth.</th>
    <th>Open.</th>
    <th>Boolq</th>
    <th>RTE</th>
    <th>ARC-e</th>
    <th>ARC-c.</th>
    <th>wikitext2 ppl
    <th>ptb_new ppl</th>
    <th>c4_new ppl</th>
    <th>lm_eval wikitext ppl</th>
   
  </tr>

  <tr>
    <td rowspan="3">Intel/neural-chat-7b-v3 </td>
    <th>FP16</th>
    <td>67.92</td> <! acc avg -->
    <td>61.13</td> <! MMLU -->
    <td>73.03</td> <! Lambada_openai -->
    <td>66.39</td> <! Hellsaswag -->
    <td>76.40</td> <! Winogrande -->
    <td>81.01</td> <! Piqa -->
    <td>47.37</td> <! Truthfulqa -->
    <td>38.8</td> <! Openbookqa -->
    <td>86.97</td> <! Boolq -->
    <td>75.81</td> <! RTE -->
    <td>82.66</td> <! Arc easy -->
    <td>57.51</td> <! Arc Challenge  -->
    <td>6.00</td>  <! wikitext2 ppl  -->
    <td>48.96</td> <! ptb_new ppl  -->
    <td>9.65</td>    <! c4_new ppl  -->
    <td>-</td> <! lm-eval wikitext ppl  -->
  </tr>

  </tr>
    <th>Ours</th>
    <td>66.90</td> <! acc avg -->
    <td>60.56</td> <! MMLU -->
    <td>72.19</td> <! Lambada_openai -->
    <td>65.28</td> <! Hellsaswag -->
    <td>75.37</td> <! Winogrande -->
    <td>81.18</td> <! Piqa -->
    <td>46.76</td> <! Truthfulqa -->
    <td>36.0</td> <! Openbookqa -->
    <td>86.91</td> <! Boolq -->
    <td>73.29</td> <! RTE -->
    <td>81.73</td> <! Arc easy -->
    <td>56.66</td> <! Arc Challenge  -->
    <td>6.21</td>  <! wikitext2 ppl  -->
    <td>59.78</td> <! ptb_new ppl  -->
    <td>10.01</td>    <! c4_new ppl  -->
    <td>-</td> <! lm-eval wikitext ppl  -->
  </tr>

  </tr>
    <th>Ours iters=1K,use_quant_input=False, minmax_lr=0.002</th>
    <td>67.70</td> <! acc avg -->
    <td>60.57</td> <! MMLU -->
    <td>73.74</td> <! Lambada_openai -->
    <td>65.62</td> <! Hellsaswag -->
    <td>77.43</td> <! Winogrande -->
    <td>80.85</td> <! Piqa -->
    <td>47.61</td> <! Truthfulqa -->
    <td>36.8</td> <! Openbookqa -->
    <td>86.94</td> <! Boolq -->
    <td>75.09</td> <! RTE -->
    <td>82.66</td> <! Arc easy -->
    <td>57.34</td> <! Arc Challenge  -->
    <td>6.17</td>  <! wikitext2 ppl  -->
    <td>59.12</td> <! ptb_new ppl  -->
    <td>9.83</td>    <! c4_new ppl  -->
    <td>-</td> <! lm-eval wikitext ppl  -->
  </tr>


  <tr>
    <td rowspan="3">mistralai/Mixtral-8x7B-v0.1 </td>
    <th>BF16</th>
   <td>67.16</td>
    <td>69.83</td>
    <td>78.44</td>
    <td>64.89</td>
    <td>76.40</td>
    <td>82.43</td>
    <td>34.15</td>
    <td>35.40</td>
    <td>84.98</td>
    <td>71.12</td>
    <td>84.22</td>
    <td>56.91</td>
    <td>3.84</td>
    <td>19.22</td>
    <td>7.41</td>
    <td>-</td>
 
  </tr>
  <tr>
    <th>Ours</th>
    <td>65.98</td>
    <td>68.90</td>
    <td>78.11</td>
    <td>64.31</td>
    <td>74.27</td>
    <td>82.10</td>
    <td>30.97</td>
    <td>34.20</td>
    <td>84.57</td>
    <td>67.87</td>
    <td>83.96</td>
    <td>56.57</td>
    <td>4.08</td>
    <td>354</td>
    <td>7.56</td>
    <td>-</td>
  </tr>
  <tr>
    <th>Ours iters=1K,use_quant_input=False 
    <td>66.78</td>
    <td>68.68</td>
    <td>78.61</td>
    <td>64.40</td>
    <td>76.56</td>
    <td>81.99</td>
    <td>32.56</td>
    <td>34.80</td>
    <td>85.96</td>
    <td>70.76</td>
    <td>83.96</td>
    <td>56.31</td>
    <td>3.99</td>
    <td>17.65</td>
    <td>7.52</td>
    <td>-</td>
 
  </tr>
  <tr>
    <td rowspan="3">microsoft/phi-2 </td>
    <th>FP16</th>
    <td>61.80</td>
    <td>56.40</td>
    <td>62.78</td>
    <td>55.83</td>
    <td>75.77</td>
    <td>78.67</td>
    <td>31.21</td>
    <td>40.40</td>
    <td>83.36</td>
    <td>62.45</td>
    <td>80.05</td>
    <td>52.90</td>
    <td>9.71</td>
    <td>18.16</td>
    <td>14.12</td>
    <td>11.05</td>

  </tr>
  <tr>
    <th>Ours</th>
    <td>61.67</td>
    <td>54.57</td>
    <td>61.32</td>
    <td>55.04</td>
    <td>76.48</td>
    <td>78.89</td>
    <td>29.74</td>
    <td>40.60</td>
    <td>83.24</td>
    <td>66.43</td>
    <td>79.76</td>
    <td>52.30</td>
    <td>9.98</td>
    <td>18.67</td>
    <td>14.39</td>
    <td>11.37</td>

  </tr>

  </tr>
    <th>Ours iters=1K,use_quant_input=False </th>
    <td>61.47</td> <! acc avg -->
    <td>55.41</td> <! MMLU -->
    <td>61.77</td> <! Lambada_openai -->
    <td>54.92</td> <! Hellsaswag -->
    <td>76.40</td> <! Winogrande -->
    <td>78.29</td> <! Piqa -->
    <td>31.09</td> <! Truthfulqa -->
    <td>40.0</td> <! Openbookqa -->
    <td>83.24</td> <! Boolq -->
    <td>63.54</td> <! RTE -->
    <td>79.29</td> <! Arc easy -->
    <td>52.22</td> <! Arc Challenge  -->
    <td>9.97</td>  <! wikitext2 ppl  -->
    <td>18.63</td> <! ptb_new ppl  -->
    <td>14.37</td>    <! c4_new ppl  -->
    <td>11.35</td> <! lm-eval wikitext ppl  -->
  </tr>
</table>

We provide a [comprehensive analysis](docs/README.md) with other methods in our accuracy data section. Notably, our approach has outperformed GPTQ with a score of 30/32 and AWQ with a score of 27/32 across llamv1/llamav2/mistral-7b on W4G-1, W4G128, W3G128, W2G128.  And the tuning costs are comparable.

## Tips
1 Consider increasing tuning steps to achieve better results, albeit with increased tuning time. Additionally, setting 'use_quant_input' to False or adjusting 'minmax_lr' to 2.0/iters has been observed to occasionally yield improved results.

2 Leverage AutoGPTQ to run the model on GPU

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_round import AutoRound

model_name = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(
            model_name, low_cpu_mem_usage=True, torch_dtype="auto", trust_remote_code=True
        )
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

autoround = AutoRound(model, tokenizer, bits=4, group_size=128, scheme="asym")
autoround.quantize()

## export to autogptq
# please install auto-gptq https://github.com/AutoGPTQ/
output_dir = "/path/to/quantized_model"
autoround.export(output_dir, target="auto_gptq", use_triton=True)
# then follow auto-gptq to load the model and inference
```
  
## Examples
Quantization has been enabled for various large language models. Please refer to the [example readme](examples/README.md) for details.


## Reference
If you find SignRound useful for your research, please cite our paper:
```bash
@article{cheng2023optimize,
  title={Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs},
  author={Cheng, Wenhua and Zhang, Weiwei and Shen, Haihao and Cai, Yiyang and He, Xin and Lv, Kaokao},
  journal={arXiv preprint arXiv:2309.05516},
  year={2023}
}
```



