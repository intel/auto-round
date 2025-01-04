import sys
sys.path.insert(0, './')
from auto_round import AutoRoundConfig  ##must import for autoround format
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

quantized_model_dir = "/dataset/int4_models/DeepSeek-V3-bf16-w4g128-auto-awq"

quantization_config = AutoRoundConfig(
    backend="cpu"
)

model = AutoModelForCausalLM.from_pretrained(
    quantized_model_dir,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="cpu",
    # revision="8fe0735",  ##use autoround format, the only difference is config.json
    quantization_config=quantization_config,  ##cpu only machine could not set this
    # cache_dir='/home/sdp/disks/nvme1n1p1'

)

for n,m in model.named_modules():
    m.name = n

tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir, trust_remote_code=True)
prompts = [
    "9.11和9.8哪个数字大",
    "strawberry中有几个r?",
    # "How many r in strawberry.",
    # "There is a girl who likes adventure,",
    # "Please give a brief introduction of DeepSeek company.",
    # "hello"

]

texts=[]
for prompt in prompts:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    texts.append(text)
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

outputs = model.generate(
    input_ids=inputs["input_ids"].to(model.device),
    attention_mask=inputs["attention_mask"].to(model.device),
    max_length=50,  # 设置生成的最大长度
    num_return_sequences=1,  # 每个 prompt 返回的结果数量
    do_sample=False
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs["input_ids"], outputs)
]

decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

# 打印结果
for i, prompt in enumerate(prompts):
    input_id = inputs
    print(f"Prompt: {prompt}")
    print(f"Generated: {decoded_outputs[i]}")
    print("-" * 50)

