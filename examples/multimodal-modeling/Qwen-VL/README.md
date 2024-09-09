Step-by-Step
============

This document presents step-by-step instructions for auto-round.
# Run Quantization on Qwen-VL Models

In this example, we introduce an straight-forward way to execute quantization on some popular multimodal models such as Qwen-VL and Qwen2-VL-Instruct. 

## Download the calibration data

Our calibration process resembles the official visual instruction tuning process.

Please download the annotation of the final mixture our instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:

COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip), and unzip the image folder to any directory you desire.

You can also refer to the official Qwen-VL finetuning requirements to create a [custom dataset](https://github.com/QwenLM/Qwen-VL/blob/master/README.md#data-preparation)

## Download the evaluation data

Please refer to [Qwen-VL evaluation](https://github.com/cognitedata/Qwen-VL-finetune/blob/master/eval_mm/EVALUATION.md)
<details>
<summary>TextVQA Data Preparation</summary>

```bash
mkdir -p data/textvqa && cd data/textvqa

# download images
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip && unzip train_val_images.zip

# download annotations and questions
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train_questions.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val_questions.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val.jsonl

cd ../..

```
</details>

<br />

<details>
<summary>ScienceQA Data Preparation</summary>

```bash
mkdir -p data/scienceqa/images && cd data/scienceqa/images

# download images
wget https://scienceqa.s3.us-west-1.amazonaws.com/images/test.zip && unzip test.zip

cd ..

# download original questions
wget https://github.com/lupantech/ScienceQA/blob/main/data/scienceqa/problems.json

# download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/scienceqa/scienceqa_test_img.jsonl

cd ../..

```
</details>
<br />

## 2. Run Examples
Enter into the examples folder and install requirements
```bash
pip install -r requirements.txt
```

- **Default Settings:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name Qwen/Qwen-VL  --bits 4 --group_size 128
```

- **Speedup the tuning:**

reduce the seqlen to 512(potential large accuracy drop)

or combine them

- **Enable quantized lm-head:**

Currently only support in Intel xpu and AutoRound format,however, we found the fake tuning could improve the accuracy is some scenarios. low_gpu_mem_usage=False is strongly recommended if the whole model could be loaded to the device, otherwise it will be quite slow to cache the inputs of lm-head. Another way is reducing nsamples,e.g. 128, to alleviate the issue.
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name Qwen/Qwen-VL  --bits 4 --group_size 128 --quant_lm_head
```

- **Utilizing the AdamW Optimizer:**

Include the flag `--adam`. Note that AdamW is less effective than sign gradient descent in many scenarios we tested.

- **Running on Intel Gaudi2**
```bash
bash run_autoround.sh
```

## 3. run inference

```python
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from transformers.generation import GenerationConfig
  import torch
  from transformers import set_seed
  set_seed(1234)
  from auto_round.auto_quantizer import AutoHfQuantizer
  quantized_model_path = "./tmp_autoround"
  tokenizer = AutoTokenizer.from_pretrained(quantized_model_path, trust_remote_code=True)
  # use bf16
  model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto", trust_remote_code=True, bf16=True).eval()
  # use fp16
  # model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto", trust_remote_code=True, fp16=True).eval()
  # use cpu only
  # model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cpu", trust_remote_code=True).eval()
  # use cuda device
  # model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cuda", trust_remote_code=True).eval()
  query = tokenizer.from_list_format([{'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'}, \
      {'text': 'Generate the caption in English with grounding:'}, \
  ])
  inputs = tokenizer(query, return_tensors='pt')
  inputs = inputs.to(model.device)
  with torch.cuda.amp.autocast(): 
      pred = model.generate(**inputs)
  response = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False)
  print(response)
  # <img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>Generate the caption in English with grounding:<ref> Woman</ref><box>(451,379),(731,806)</box> and<ref> her dog</ref><box>(219,424),(576,896)</box> playing on the beach<|endoftext|>
  image = tokenizer.draw_bbox_on_latest_picture(response)
  if image:
    image.save('2.jpg')
  else:
    print("no box")

```


## 4. Results
Using [COCO 2017](https://cocodataset.org/) and [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) datasets for quantization calibration, and TextVQA dataset for evaluation. please follow the [recipe](./run_autoround.sh) and [evaluate script](./run_eval.sh). The results for Qwen-VL are as follows:
| Metric         | bf16   | INT4   |
|:----------------|:--------|:--------|
| avg            | 0.5628 | 0.5589 |
| paper-avg      | 0.5603 | 0.5611 |
| mmlu           | 0.4828 | 0.4639 |
| lambada_openai | 0.6782 | 0.6664 |
| hellaswag      | 0.5593 | 0.5487 |
| winogrande     | 0.6827 | 0.6875 |
| piqa           | 0.7786 | 0.7748 |
| truthfulqa_mc1 | 0.2876 | 0.2901 |
| openbookqa     | 0.2880 | 0.2940 |
| boolq          | 0.7012 | 0.7318 |
| arc_easy       | 0.7201 | 0.7327 |
| arc_challenge  | 0.4249 | 0.4206 |
| cmmlu          | 0.4798 | 0.4618 |
| ceval          | 0.4814 | 0.4569 |
| textVQA        | 0.6402 | 0.6379 |
| scienceVQA     | 0.6748 | 0.6574 |



## 5. Environment

PyTorch 1.8 or higher version is needed


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










