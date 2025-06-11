# Copied from https://huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ#you-can-then-use-the-following-code

# ==------------------------------------------------------------------------------------------==
# Set the model name or path
# CUDA_VISIBLE_DEVICES=None OMP_NUM_THREADS=56 numactl -l -C 0-55  python  test_load.py
# ==------------------------------------------------------------------------------------------==
model_name_or_path = "./Llama-3.2-1B-Early-w4g32-auto_round_gptq_hf_format"
# model_name_or_path = "./tmp_autoround/Llama-3.2-3B-Instruct-w4g128-auto_awq"
model_name_or_path = "./tmp_autoround/Llama-3.2-3B-Instruct-w4g128-auto_round-gptq-hf"
model_name_or_path = "./tmp_autoround/Llama-3.2-3B-Instruct-w4g128-auto_gptq/"
model_name_or_path = "./llama-3b-ins-2/Llama-3.2-1B-Instruct-w4g128-auto_gptq/"
# model_name_or_path = "./Llama-3.2-3B-Early-w4g64-auto_round_gptq_hf_format"


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
with torch.no_grad():

    # To use a different branch, change revision
    # For example: revision="gptq-4bit-64g-actorder_True"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens = 5 if device == "cpu" else 100
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)


    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

    prompt = "Tell me about AI"
    if device == "cuda":
        prompt = f"""[INST] <<SYS>>
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
        <</SYS>>
        {prompt}[/INST]

        """

    print("\n\n*** Generate:")


    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output = model.generate(
        inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=max_new_tokens
    )
    print(tokenizer.decode(output[0]))
