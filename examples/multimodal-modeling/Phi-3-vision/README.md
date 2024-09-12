Step-by-Step
============
transformers>=4.41.0
This document presents step-by-step instructions for auto-round.
# Run Quantization on Phi-3-vision Models

In this example, we introduce an straight-forward way to execute quantization on some popular multimodal models such as Phi-3-vision. 

## Download the calibration data

Our calibration process resembles the official visual instruction tuning process.

Please download the annotation of the final mixture our instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:

COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip), and unzip the image folder to any directory you desire.


## 2. Run Examples
PyTorch 1.8 or higher version is needed

Enter into the examples folder and install lm-eval to run the evaluation
```bash
pip install -r requirements.txt
```

- **Default Settings:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name microsoft/Phi-3-vision-128k-instruct  --bits 4 --group_size 128
```

- **Speedup the tuning:**

disable_low_gpu_mem_usage(more gpu memory)

reduce the seqlen to 512(potential large accuracy drop)

or combine them

- **Enable quantized lm-head:**

Currently only support in Intel xpu and AutoRound format, however, we found the fake tuning could improve the accuracy is some scenarios. Disable --low_gpu_mem_usage is strongly recommended if the whole model could be loaded to the device, otherwise it will be quite slow to cache the inputs of lm-head. Another way is reducing nsamples,e.g. 128, to alleviate the issue.
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name microsoft/Phi-3-vision-128k-instruct  --bits 4 --group_size 128 --quant_lm_head
```

- **Utilizing the AdamW Optimizer:**

Include the flag `--adam`. Note that AdamW is less effective than sign gradient descent in many scenarios we tested.

- **Running on Intel Gaudi2**
```bash
bash run_autoround.sh
```


## 3. Run Inference

```python
from PIL import Image
import requests
import io
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from auto_round import AutoRoundConfig
quantized_model_path = "./tmp_autoround"
model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2') # use _attn_implementation='eager' to disable flash attention

processor = AutoProcessor.from_pretrained(quantized_model_path, trust_remote_code=True)

messages = [ \
    {"role": "user", "content": "<|image_1|>\nWhat is shown in this image?"}, \
    {"role": "assistant", "content": "The chart displays the percentage of respondents who agree with various statements about their preparedness for meetings. It shows five categories: 'Having clear and pre-defined goals for meetings', 'Knowing where to find the information I need for a meeting', 'Understanding my exact role and responsibilities when I'm invited', 'Having tools to manage admin tasks like note-taking or summarization', and 'Having more focus time to sufficiently prepare for meetings'. Each category has an associated bar indicating the level of agreement, measured on a scale from 0% to 100%."}, \
    {"role": "user", "content": "Provide insightful questions to spark discussion."}]

url = "https://assets-c4akfrf5b4d3f4b7.z01.azurefd.net/assets/2024/04/BMDataViz_661fb89f3845e.png" 
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open(io.BytesIO(requests.get(url, stream=True).content))

prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")

generation_args = {
    "max_new_tokens": 50,
    "temperature": 0.0,
    "do_sample": False,
}

generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

# remove input tokens 
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

print(response)
# 1. How does the level of agreement on each statement reflect the overall preparedness of respondents for meetings?
# 2. What are the most and least agreed-upon statements, and why might that be the case?
# 3.
```
<!-- 

## 4. Results
Using [COCO 2017](https://cocodataset.org/) and [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) datasets for quantization calibration, and lm_eval dataset for evaluation. please follow the [recipe](./run_autoround.sh) and [evaluate script](./run_eval.sh). The results for Phi-3-vision-128k-instruct are as follows:
| Metric         | bf16   | INT4   |
|----------------|--------|--------|
| avg            | 0.6014 | 0.5940 |
| mmlu           | 0.6369 | 0.6310 |
| lambada_openai | 0.6487 | 0.6406 |
| hellaswag      | 0.5585 | 0.5483 |
| winogrande     | 0.7395 | 0.7451 |
| piqa           | 0.7954 | 0.7889 |
| truthfulqa_mc1 | 0.3084 | 0.2987 |
| openbookqa     | 0.3580 | 0.3600 |
| boolq          | 0.8532 | 0.8557 |
| arc_easy       | 0.8371 | 0.8346 |
| arc_challenge  | 0.5572 | 0.5469 |
| cmmlu          | 0.4074 | 0.3950 |
| ceval          | 0.4027 | 0.4012 |
| gsm8k          | 0.7157 | 0.6755 | -->



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








