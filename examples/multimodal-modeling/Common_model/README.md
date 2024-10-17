Step-by-Step
============
This document presents step-by-step instructions for auto-round.

# Run Quantization on Qwen-VL/Qwen2-VL-Instruct Models

In this example, we introduce an straight-forward way to execute quantization on some popular multimodal models such as Qwen-VL and Qwen2-VL-Instruct. 

## Download the calibration data

Our calibration process resembles the official visual instruction tuning process.

Please download the annotation of the final mixture our instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:

COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip), and unzip the image folder to any directory you desire.

You can also refer to the official Qwen-VL finetuning requirements to create a [custom dataset](https://github.com/QwenLM/Qwen-VL/blob/master/README.md#data-preparation)

## Download the evaluation data
Please refer to [Qwen-VL evaluation](https://github.com/cognitedata/Qwen-VL-finetune/blob/master/eval_mm/EVALUATION.md)

Note: The quantized Qwen2-VL model evaluation is currently not supported.
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

For Qwen2-VL quantization/Inference, please follow [Qwen2 official requirements](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) and install the latest version of transformers by **building from source**.

- **Default Settings:**
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --model_name Qwen/Qwen2-VL-7B-Instruct  --bits 4 --group_size 128
```

- **Speedup the tuning:**

reduce the seqlen to 512(potential large accuracy drop)

or combine them

- **Running on Intel Gaudi2**
```bash
bash run_autoround.sh
```

## 3. run inference

- Qwen-VL inference

```python
  from transformers import AutoModelForCausalLM, AutoTokenizer
  from transformers.generation import GenerationConfig
  import torch
  from transformers import set_seed
  set_seed(1234)
  from auto_round import AutoRoundConfig
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


- Qwen2-VL-7B-Instruct inference

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from auto_round import AutoRoundConfig
quantized_model_path="./tmp_autoround"
model = Qwen2VLForConditionalGeneration.from_pretrained(quantized_model_path, device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
messages = [{
    "role": "user",
    "content": [
        {
            "type": "image",
            "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        },
        {"type": "text", "text": "Describe this image."},]
}]
# Preparation for inference
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")
 
# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=50)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
# The image depicts a serene beach scene at sunset. A woman is sitting on the sand, facing a large dog that appears to be a Labrador Retriever. The dog is wearing a harness and is extending its paw towards the woman's hand, possibly

# messages = [{
#     "role": "user",
#     "content": [
#         {
#             "type": "image",
#             "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg",
#         },
#         {"type": "text", "text": "What does the label 15 represent? (1) lava (2) core (3) tunnel (4) ash cloud"},]
# }]

# The label 15 represents an ash cloud. In the context of a volcano, an ash cloud is formed when volcanic ash is ejected into the atmosphere during an eruption. Therefore, the correct answer is:\n\n(4) ash cloud

```


- Llama-3.2-11B-Vision-Instruct inference

```python
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from auto_round.auto_quantizer import AutoHfQuantizer
quantized_model_path="./tmp_autoround"
model = MllamaForConditionalGeneration.from_pretrained(
    quantized_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
processor = AutoProcessor.from_pretrained(quantized_model_path)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

prompt = "<|image|><|begin_of_text|>If I had to write a haiku for this one"
inputs = processor(image, prompt, return_tensors="pt", truncation=True).to(model.device)

output = model.generate(**inputs, max_new_tokens=30)
print(processor.decode(output[0]))

# <|begin_of_text|><|image|><|begin_of_text|>If I had to write a haiku for this one, it would be:

# Rabbit in a coat
# Dressed up in style for the day
# Country charm abounds

# The image depicts a rabbit
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












