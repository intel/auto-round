
## Model Details

This model is an int4 model with group_size 128 and symmetric quantization of [microsoft/Phi-3.5-vision-instruct](https://huggingface.co/microsoft/Phi-3.5-vision-instruct). Load the model with revision="13b4c3d" to use AutoGPTQ format.
## How To Use


### Requirements

The current `transformers` version can be verified with: `pip list | grep transformers`.

Examples of required packages:
```
flash_attn==2.5.8
numpy==1.24.4
Pillow==10.3.0
Requests==2.31.0
torch==2.3.0
torchvision==0.18.0
transformers==4.43.0
accelerate==0.30.0
```


### INT4 Inference
```python
from auto_round import AutoRoundConfig  ##must import for auto-round format
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

model_id = "Intel/Phi-3.5-vision-instruct-inc-private"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype="auto",
    ##revision="13b4c3d" ##AutoGPTQ format
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=4)

image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
content = "Describe this image."
messages = [
    {"role": "user", "content": "<|image_1|>\n" + content},
]

prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs = Image.open(requests.get(image_url, stream=True).raw)
inputs = processor(prompt, image_inputs, return_tensors="pt").to(model.device)

generation_args = {
    "max_new_tokens": 1000,
    "temperature": 0.0,
    "do_sample": False,
}

generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

# remove input tokens
generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(response)
##INT4:
## The image captures a serene beach scene at sunset with a person and a dog. The person is seated on the sand, reading a book, while the dog, wearing a harness, sits attentively beside them. The sun is low on the horizon, casting a warm glow and long shadows on the sand. The ocean is calm, and the sky is clear, suggesting a peaceful end to the day.

##BF16:
## The image shows a person sitting on a sandy beach with a dog. The person is wearing a plaid shirt and is holding a book, while the dog is sitting next to them, looking at the book. The beach is near the ocean, and the sun is low in the sky, suggesting it is either sunrise or sunset. The sky is clear, and the overall atmosphere is calm and serene.


image_url = "http://images.cocodataset.org/train2017/000000411975.jpg"
content = "How many people are there on the baseball field in the image?"
##INT4:
## There are three people on the baseball field in the image.

##BF16:
## There are three people on the baseball field in the image.


image_url = "https://intelcorp.scene7.com/is/image/intelcorp/processor-overview-framed-badge:1920-1080?wid=480&hei=270"
content = "This image represents which company?"
##INT4:
## The image represents the company Intel, as indicated by the text 'intel INSIDE'.

##BF16:
## The image represents the company Intel, as indicated by the logo and the text 'INSIDE'.
```


## Evaluation the model
pip3 install git+https://github.com/open-compass/VLMEvalKit.git@7de2dcb. The evaluation process may encounter errors that require changing model backend or evaluation code. Detailed instructions will be provided in a future update
```bash
auto-round-mllm --eval --model Intel/Phi-3.5-vision-instruct-inc-private --tasks MMBench_DEV_EN_V11,ScienceQA_VAL,TextVQA_VAL,POPE --output_dir "./eval_result"
```
|Metric             |16bits|Pile Calib INT4  | Llava Calib INT4  |
|-------------------|:------|:------|:------|
|avg                |77.64 |77.14 |76.87|
|MMBench_DEV_EN_V11 |71.83 |71.36 |70.90|
|ScienceQA_VAL      |90.56 |89.75 |89.13|
|TextVQA_VAL        |65.36 |64.77 |64.66|
|POPE               |82.82 |82.67 |82.80|

### Generate the model
Here is the sample command to reproduce the model.
```bash
pip install auto-round
auto-round-mllm \
--model microsoft/Phi-3.5-vision-instruct \
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
