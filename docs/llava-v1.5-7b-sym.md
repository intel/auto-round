
## Model Details

This model is an int4 model with group_size 128 and symmetric quantization of [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b). Load the model with revision="8ab8ff" to use AutoGPTQ format.

## How To Use

### Requirements

1. Clone this repository and navigate to LLaVA folder
```shell
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
```

2. Refine LLaVA repo
```
vi llava/model/language_model/llava_llama.py
# add 'cache_position = None,' to line 71.
```
3. Install Package
```
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

### INT4 Inference
```python
from auto_round import AutoRoundConfig  ## must import for auto-round format
import requests
import torch
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.train.train import preprocess, preprocess_multimodal, DataCollatorForSupervisedDataset


class DataArgs:
    is_multimodal = True
    mm_use_im_start_end = False


quantized_model_path = "Intel/llava-v1.5-7b-inc-private"

tokenizer, model, image_processor, _ = load_pretrained_model(
    quantized_model_path,
    model_base=None,
    model_name=quantized_model_path,
    torch_dtype="auto",
    device_map="auto",
    ##revision="8ab8ff" ##AutoGPTQ format
)
image_url = "http://images.cocodataset.org/train2017/000000116003.jpg"
messages = [{"from": "human", "value": "What is the tennis player doing in the image?\n<image>"}]

# Preparation for inference
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
image_input = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].to(model.device)
input_data = preprocess_multimodal([messages], DataArgs())
inputs = preprocess(input_data, tokenizer, has_image=(image_input is not None))

output = model.generate(inputs["input_ids"].to(model.device), images=image_input.unsqueeze(0).half(), max_new_tokens=50)
print(tokenizer.batch_decode(output))

##INT4: The tennis player is celebrating a victory, raising his arms in the air, and holding his tennis racket.

##BF16: The tennis player is celebrating a victory, raising his arms in the air, and holding a tennis racket.

image_url = "http://images.cocodataset.org/train2017/000000411975.jpg"
messages = [{"from": "human", "value": "How many people are on the baseball field in the picture?\n<image>"}]

##INT4: There are three people on the baseball field in the picture.

##BF16: There are three people on the baseball field in the picture.


image_url = "http://images.cocodataset.org/train2017/000000093025.jpg"
messages = [{"from": "human", "value": "How many people and animals are there in the image?\n<image>"}]

##INT4: There are two people and one animal in the image.

##BF16: There are two people and one animal in the image.
```

## Evaluation the model
pip3 install lmms_eval. The evaluation process may encounter errors that require changing model backend or evaluation code. Detailed instructions will be provided in a future update
```bash
auto-round-mllm --lmms --model Intel/llava-v1.5-7b-inc-private --tasks pope,textvqa_val,scienceqa,mmbench_en  --output_dir "./eval_result" --device cuda:0 
```
|Metric             |16bits|Pile Calib INT4  | Llava Calib INT4  |
|:-------------------|:------|:------|:--------------|
|avg                |65.40 |65.91 | 65.79 |
|MMBench_DEV_EN_V11 |64.09 |64.43 |64.43 |
|ScienceQA_VAL      |64.87 |67.20 |66.80 |
|TextVQA_VAL        |45.56 |45.71 |45.81 |
|POPE               |87.09 |86.31 |86.12 |

### Generate the model
Here is the sample command to reproduce the model.
```bash
pip install auto-round
auto-round-mllm \
--model liuhaotian/llava-v1.5-7b \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--nsample 512 \
--seqlen 2048 \
--format 'auto_gptq,auto_round' \
--output_dir "./tmp_autoround"
```

## Ethical Considerations and Limitations

The model can produce factually incorrect output, and should not be relied on to produce factually accurate information. Because of the limitations of the pretrained model and the finetuning datasets, it is possible that this model could generate lewd, biased or otherwise offensive outputs.

Therefore, before deploying any applications of the model, developers should perform safety testing.

## Caveats and Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model.

Here are a couple of useful links to learn more about Intel's AI software:

- Intel Neural Compressor [link](https://github.com/intel/neural-compressor)

## Disclaimer

The license on this model does not constitute legal advice. We are not responsible for the actions of third parties who use this model. Please consult an attorney before using this model for commercial purposes.

## Cite

@article{cheng2023optimize, title={Optimize weight rounding via signed gradient descent for the quantization of llms}, author={Cheng, Wenhua and Zhang, Weiwei and Shen, Haihao and Cai, Yiyang and He, Xin and Lv, Kaokao and Liu, Yi}, journal={arXiv preprint arXiv:2309.05516}, year={2023} }

[arxiv](https://arxiv.org/abs/2309.05516) [github](https://github.com/intel/auto-round)
