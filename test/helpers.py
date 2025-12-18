import os

import pytest

# Automatic choose local path or model name.
opt_name_or_path = "/tf_dataset/auto_round/models/facebook/opt-125m"
if not os.path.exists(opt_name_or_path):
    opt_name_or_path = "facebook/opt-125m"

qwen_name_or_path = "/tf_dataset/auto_round/models/Qwen/Qwen3-0.6B"
if not os.path.exists(qwen_name_or_path):
    qwen_name_or_path = "Qwen/Qwen3-0.6B"

lamini_name_or_path = "/tf_dataset/auto_round/models/MBZUAI/LaMini-GPT-124M"
if not os.path.exists(lamini_name_or_path):
    lamini_name_or_path = "MBZUAI/LaMini-GPT-124M"

gptj_name_or_path = "/tf_dataset/auto_round/models/hf-internal-testing/tiny-random-GPTJForCausalLM"
if not os.path.exists(gptj_name_or_path):
    gptj_name_or_path = "hf-internal-testing/tiny-random-GPTJForCausalLM"


# HPU mode checking
def is_pytest_mode_compile():
    return pytest.mode == "compile"


def is_pytest_mode_lazy():
    return pytest.mode == "lazy"


# General model inference code
def model_infer(model, tokenizer, apply_chat_template=False):
    prompts = [
        "Hello,my name is",
        # "The president of the United States is",
        # "The capital of France is",
        # "The future of AI is",
    ]
    if apply_chat_template:
        texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)
        prompts = texts

    inputs = tokenizer(prompts, return_tensors="pt", padding=False, truncation=True)

    outputs = model.generate(
        input_ids=inputs["input_ids"].to(model.device),
        attention_mask=inputs["attention_mask"].to(model.device),
        do_sample=False,  ## change this to follow official usage
        max_new_tokens=5,
    )
    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs["input_ids"], outputs)]

    decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    for i, prompt in enumerate(prompts):
        print(f"Prompt: {prompt}")
        print(f"Generated: {decoded_outputs[i]}")
        print("-" * 50)
    return decoded_outputs[0]
