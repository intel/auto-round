
## Model Details

This model is an int4 model with group_size 128 and symmetric quantization of [THUDM/cogvlm2-llama3-chat-19B](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B). 
## How To Use
### INT4 Inference
```python
import torch
from PIL import Image
from auto_round import AutoRoundConfig  ##must import for auto-round format
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests

MODEL_PATH = "Intel/cogvlm2-llama3-chat-19B-inc-private"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = (
    AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype="auto", trust_remote_code=True, device_map=DEVICE)
    .to(DEVICE)
    .eval()
)

image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
content = "Describe this image."

text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"
query = text_only_template.format(content)

image = Image.open(requests.get(image_url, stream=True).raw)
input_by_model = model.build_conversation_input_ids(tokenizer, query=query, images=[image], template_version="chat")
inputs = {
    "input_ids": input_by_model["input_ids"].unsqueeze(0).to(DEVICE),
    "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(DEVICE),
    "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(DEVICE),
    "images": [[input_by_model["images"][0].to(DEVICE).to(model.dtype)]] if image is not None else None,
}
gen_kwargs = {
    "max_new_tokens": 2048,
    "pad_token_id": 128002,
}

with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(outputs[0])
    response = response.split("<|end_of_text|>")[0]
    print(response)
##INT4:
## The image depicts a serene beach scene during what appears to be the golden hour, just before sunset. A woman is seated on the sandy shore, facing the vast expanse of the ocean. The waves are gently crashing on the beach, and the sky is painted with hues of orange and blue. The woman seems to be in a relaxed posture, possibly enjoying the tranquility of the moment. Beside her is a large, golden-colored dog, possibly a Labrador, wearing a colorful harness. The dog is sitting upright, looking at the woman, and they seem to share a bond of trust and affection. The overall mood of the image is peaceful, reflective, and heartwarming.

##BF16:
## The image showcases a serene beach setting during what appears to be either sunrise or sunset. In the foreground, a woman sits on the sandy beach, dressed in casual attire, including a checkered shirt and jeans. She is engaged in a moment of connection with a golden retriever dog, which is seated beside her. The dog wears a colorful harness and is looking up at the woman, possibly in anticipation of a treat or a playful gesture. The vast expanse of the ocean can be seen in the background, with gentle waves crashing onto the shore. The sky is clear, and the warm hues of the setting or rising sun cast a soft glow over the scene, creating a tranquil and heartwarming atmosphere.

image_url = "http://images.cocodataset.org/train2017/000000411975.jpg"
content = "图片中的棒球场上有多少人？"
##INT4:
## In the image provided, there are four individuals on the baseball field. There are two children in the foreground, presumably engaged in some playful activity or perhaps participating in a game. One child is bending over, possibly picking something up off the ground, while the other stands beside them. Additionally, there is an adult standing further back, holding a microphone, suggesting they might be an announcer or commentator for the game. Lastly, there is another adult, seen from the back, sitting in the stands. So, in total, there are four people on the baseball field.

##BF16:
## In the image provided, there are five people visible on the baseball field.

image_url = "https://intelcorp.scene7.com/is/image/intelcorp/processor-overview-framed-badge:1920-1080?wid=480&hei=270"
content = "这张图片代表哪家公司？"
##INT4:
## The image represents the company Intel. The logo in the image is the Intel Inside logo, which is commonly used by Intel to signify the presence of their processors in various electronic devices.

##BF16:
## The image represents the company Intel.
```



### Generate the model
Here is the sample command to reproduce the model.
```bash
pip install auto-round
auto-round-mllm \
--model THUDM/cogvlm2-llama3-chat-19B \
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
