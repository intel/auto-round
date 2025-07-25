
## Model Details

This model is an int4 model with group_size 128 and symmetric quantization of [meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct). Load the model with revision="f036ca" to use AutoGPTQ format.

## How To Use

### Requirements
Please use Transformers version 4.45.0 or later
AutoRound version >= 0.4.1

### INT4 Inference
```python
from auto_round import AutoRoundConfig  ## must import for auto-round format
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

quantized_model_path = "Intel/Llama-3.2-11B-Vision-Instruct-inc-private"

model = MllamaForConditionalGeneration.from_pretrained(
    quantized_model_path,
    torch_dtype="auto",
    device_map="auto",
    ##revision="f036ca" ##AutoGPTQ format
)
processor = AutoProcessor.from_pretrained(quantized_model_path)
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
messages = [
    {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": "Please write a haiku for this one, it would be: "}],
    }
]

# Preparation for inference
image = Image.open(requests.get(image_url, stream=True).raw)
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(output[0]))

##INT4:
##  Here is a haiku for the rabbit:

##  Whiskers twitching bright
##  Ears perked up, alert and keen
##  Spring's gentle delight<|eot_id|>


##BF16:
## Here is a haiku for the rabbit:

## Whiskers twitching fast
## In a coat of blue and brown
## Hoppy little soul<|eot_id|>

image_url = "http://images.cocodataset.org/train2017/000000411975.jpg"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "How many people are on the baseball field in the picture?"},
        ],
    }
]
##INT4: There are five people on the baseball field in the picture.
##

##BF16: There are five people on the baseball field in the picture.
##

image_url = "https://intelcorp.scene7.com/is/image/intelcorp/processor-overview-framed-badge:1920-1080?wid=480&hei=270"
messages = [
    {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": "Which company does this picture represent?"}],
    }
]
##INT4: This picture represents Intel.
##

##BF16: This image represents Intel, a multinational semiconductor corporation headquartered in Santa Clara, California.
##
```

## Evaluation the model
pip3 install git+https://github.com/open-compass/VLMEvalKit.git@7de2dcb. The evaluation process may encounter errors that require changing model backend or evaluation code. Detailed instructions will be provided in a future update.
```bash
auto-round-mllm --eval --model Intel/Llama-3.2-11B-Vision-Instruct-inc-private --tasks MMBench_DEV_EN_V11,ScienceQA_VAL,TextVQA_VAL,POPE --output_dir "./eval_result"
```
|Metric             |16bits|Pile Calib INT4  |Llava Calib INT4|
|:-------------------|:------|:------|:------|
|avg                |66.05 |67.81 |66.02 |
|MMBench_DEV_EN_V11 |52.86 |53.48 |52.17 |
|ScienceQA_VAL      |68.86 |70.39 |69.15 |
|TextVQA_VAL        |54.49 |59.62 |55.07 |
|POPE               |88.00 |87.76 |87.71 |

### Generate the model
Here is the sample command to reproduce the model.
```bash
pip install auto-round
auto-round-mllm \
--model meta-llama/Llama-3.2-11B-Vision-Instruct \
--device 0 \
--group_size 128 \
--bits 4 \
--iters 1000 \
--nsample 512 \
--seqlen 512 \
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
