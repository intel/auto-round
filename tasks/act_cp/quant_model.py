import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/storage/yiliu7/Qwen/Qwen3-30B-A3B"
# model_name = "/storage/yiliu7/Qwen/Qwen3-30B-A3B-L2"

# tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_name,device_map="cpu", torch_dtype="auto",trust_remote_code=True)


from auto_round import AutoRound
from auto_round import schemes as ar_schemes

scheme = "MXFP4"
scheme = "FP8_STATIC"
scheme = "MXFP8"

# autoround = AutoRound(
#     # model,
#     model_name,
#     # tokenizer,
#     scheme=scheme,
#     iters=5,
#     low_gpu_mem_usage=True,
#         # low_gpu_mem_usage=True,
#         enable_activation_checkpointing=True,
# )
SAVE_DIR = model_name.rstrip("/").split("/")[-1] + f"-{scheme}-AR-REAL-TEST"
# res = autoround.quantize_and_save(format="auto_round", output_dir=SAVE_DIR)


TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a Python function to check if a number is prime.",
    "Translate 'Hello, how are you?' to Chinese.",
]


def test_generation():
    """Reload the quantized model and test generation quality."""
    print(f"\n{'='*80}")
    print("  STEP 2: Reload and test generation")
    print(f"  Loading from: {SAVE_DIR}")
    print(f"{'='*80}\n")

    tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        SAVE_DIR,
        # device_map=DEVICE,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"  Model loaded: {model.__class__.__name__}")
    print(f"  Device: {next(model.parameters()).device}")
    print()

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"  --- Prompt {i+1}: {prompt}")

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                temperature=1.0,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        print(f"  Response: {generated[:300]}")
        print()

    del model, tokenizer
    torch.cuda.empty_cache()
    # gc.collect()


test_generation()
