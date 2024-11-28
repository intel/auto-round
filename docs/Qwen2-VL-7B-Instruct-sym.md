
## Model Details

This model is an int4 model with group_size 128 and symmetric quantization of [Qwen/Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct). Load the model with revision="a7269c6" to use AutoGPTQ format.

## How To Use


### Requirements
Please use Transformers version 4.45.0 or later, or you might encounter the following error:
```
KeyError: 'qwen2_vl'
```

### INT4 Inference
```python
from auto_round import AutoRoundConfig ## must import for auto-round format
import requests
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Intel/Qwen2-VL-7B-Instruct-inc-private",
    torch_dtype="auto",
    device_map="auto"，
    ##revision="a7269c6" ##AutoGPTQ format
)
processor = AutoProcessor.from_pretrained("Intel/Qwen2-VL-7B-Instruct-inc-private")
image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_url,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs = Image.open(requests.get(image_url, stream=True).raw)
inputs = processor(
    text=[text],
    images=image_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])

##INT4:
## 'The image depicts a serene beach scene with a woman and her dog. The woman is sitting on the sand, facing the ocean, and appears to be engaging in a playful interaction with her dog. The dog, which is wearing a harness, is sitting beside her and has its front paw raised, seemingly giving a high-five to the woman. The woman is smiling and seems to be enjoying the moment. The beach is relatively empty, with gentle waves in the background, and the lighting suggests it is either early morning or late afternoon, creating a warm and peaceful atmosphere.'

##BF16:
## "The image depicts a serene beach scene with a woman and her dog enjoying a moment together. The woman is sitting on the sandy beach, facing the ocean, and appears to be engaging in a playful activity with her dog. She is wearing a plaid shirt and dark pants, and her hair is long and dark. The dog, which is a large breed, possibly a Labrador Retriever, is sitting in front of her, wearing a harness. The dog is extending its front paw towards the woman's hand, as if it is giving her a high-five. The woman is smiling and seems to be enjoying the interaction.\n\nThe beach is"

image_url = "http://images.cocodataset.org/train2017/000000411975.jpg"
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_url,
            },
            {"type": "text", "text": "图片中的棒球场上有多少人？"},
        ],
    }
]
##INT4:
## 图片中的棒球场上有五个人。

##BF16:
## 图片中的棒球场上有三个人。

image_url = "https://intelcorp.scene7.com/is/image/intelcorp/processor-overview-framed-badge:1920-1080?wid=480&hei=270"
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_url,
            },
            {"type": "text", "text": "这张图片代表哪家公司？"},
        ],
    }
]
##INT4:
## 这张图片代表英特尔公司（Intel）。英特尔是全球领先的半导体公司，主要生产中央处理器（CPU）和其他计算机硬件。

##BF16:
## 这张图片代表英特尔公司（Intel）。图片中的标志是英特尔的标志，标志下方的文字“Intel Inside”表明这是英特尔的宣传标志，用于表明该产品使用了英特尔的处理器或其他技术。

```

## Evaluation the model
pip3 install git+https://github.com/open-compass/VLMEvalKit.git@7de2dcb. The evaluation process may encounter errors that require changing model backend or evaluation code. Detailed instructions will be provided in a future update.
```bash
auto-round-mllm --eval --model Intel/Qwen2-VL-7B-Instruct-inc-private --tasks MMBench_DEV_EN_V11,ScienceQA_VAL,TextVQA_VAL,POPE --output_dir "./eval_result"
```
|Metric             |16bits|Pile Calib INT4  | Llava Calib INT4  |
|:-------------------|:------|:------|:------|
|avg                |83.92 |83.82 |83.42 |
|MMBench_DEV_EN_V11 |80.50 |79.64 |80.42 |
|ScienceQA_VAL      |84.69 |83.88 |83.26 |
|TextVQA_VAL        |84.36 |84.28 |84.11 |
|POPE               |86.13 |87.57 |85.89 |

### Generate the model
Here is the sample command to reproduce the model.
```bash
pip install auto_round
auto-round-mllm \
--model Qwen/Qwen2-VL-7B-Instruct \
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
