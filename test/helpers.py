import copy
import os

import pytest
import torch
import transformers

from auto_round.utils import get_attr, llm_load_model, mllm_load_model, set_attr


# Automatic choose local path or model name.
def get_model_path(model_name: str) -> str:
    ut_path = f"/tf_dataset/auto_round/models/{model_name}"
    local_path = f"/models/{model_name.split('/')[-1]}"

    if os.path.exists(ut_path):
        return ut_path
    elif os.path.exists(local_path):
        return local_path
    else:
        return model_name


opt_name_or_path = get_model_path("facebook/opt-125m")
qwen_name_or_path = get_model_path("Qwen/Qwen3-0.6B")
lamini_name_or_path = get_model_path("MBZUAI/LaMini-GPT-124M")
gptj_name_or_path = get_model_path("hf-internal-testing/tiny-random-GPTJForCausalLM")
phi2_name_or_path = get_model_path("microsoft/phi-2")
deepseek_v2_name_or_path = get_model_path("deepseek-ai/DeepSeek-V2-Lite")
qwen_moe_name_or_path = get_model_path("Qwen/Qwen1.5-MoE-A2.7B")
qwen_vl_name_or_path = get_model_path("Qwen/Qwen2-VL-2B-Instruct")
qwen_2_5_vl_name_or_path = get_model_path("Qwen/Qwen2.5-VL-3B-Instruct")
gemma_name_or_path = get_model_path("benzart/gemma-2b-it-fine-tuning-for-code-test")


# Slice model into tiny model for speedup
def get_tiny_model(model_name_or_path, num_layers=2, is_mllm=False, **kwargs):
    """Generate a tiny model by slicing layers from the original model."""

    def slice_layers(module):
        """slice layers in the model."""
        for name, child in module.named_children():
            if isinstance(child, torch.nn.ModuleList) and len(child) > num_layers:
                new_layers = torch.nn.ModuleList(child[:num_layers])
                setattr(module, name, new_layers)
                return True
            if slice_layers(child):
                return True
        return False

    kwargs["dtype"] = "auto" if "auto" not in kwargs else kwargs["dtype"]
    kwargs["trust_remote_code"] = True if "trust_remote_code" not in kwargs else kwargs["trust_remote_code"]
    if is_mllm:
        model, processor, tokenizer, image_processor = mllm_load_model(model_name_or_path, **kwargs)
        if hasattr(model.config, "vision_config"):
            if hasattr(model.config.vision_config, "num_hidden_layers"):  # mistral, etc.
                model.config.num_hidden_layers = num_layers
            elif hasattr(model.config.vision_config, "depth"):  # qwen vl
                model.config.vision_config.depth = num_layers
    else:
        model, tokenizer = llm_load_model(model_name_or_path, **kwargs)
    slice_layers(model)

    if hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = num_layers
    if hasattr(model.config, "layer_types"):
        model.config.layer_types = model.config.layer_types[:num_layers]

    return model


# for fixture usage only
def save_tiny_model(model_name_or_path, tiny_model_path, num_layers=2, is_mllm=False, force_untie=False, **kwargs):
    """Generate  a tiny model and save to the specified path."""
    model = get_tiny_model(model_name_or_path, num_layers=num_layers, is_mllm=is_mllm, **kwargs)
    if force_untie:
        if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
            model.config.tie_word_embeddings = False
            for key in model._tied_weights_keys:
                weight = get_attr(model, key)
                set_attr(model, key, copy.deepcopy(weight))
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    test_path = os.path.dirname(__file__)
    tiny_model_path = os.path.join(test_path, tiny_model_path.removeprefix("./"))
    model.save_pretrained(tiny_model_path)
    tokenizer.save_pretrained(tiny_model_path)
    if is_mllm:
        processor = transformers.AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        image_processor = transformers.AutoImageProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        processor.save_pretrained(tiny_model_path)
        image_processor.save_pretrained(tiny_model_path)
    print(f"[Fixture]: built tiny model path:{tiny_model_path} for testing in session")
    return tiny_model_path


# HPU mode checking
def is_pytest_mode_compile():
    return pytest.mode == "compile"


def is_pytest_mode_lazy():
    return pytest.mode == "lazy"


# General model inference code
def model_infer(model, tokenizer, apply_chat_template=False):
    """Run model inference and print generated outputs."""
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


# Dummy dataloader for testing
class DataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


fixed_input = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)


def get_output(model_name_or_path):
    """Get model output for fixed input."""
    try:
        model, tokenizer = llm_load_model(model_name_or_path)
    except:
        model, processor, tokenizer, image_processor = mllm_load_model(model_name_or_path)
    outputs = model(fixed_input)[0]
    return outputs.detach().cpu()


def is_model_outputs_similar(model_path_1, model_path_2, metric="cosine_similarity", threshold=0.98, k=5, verbose=True):
    """
    Compare outputs from two models using specified metric and return pass/fail.

    Args:
        model_path_1: Path to first model
        model_path_2: Path to second model
        metric: Metric to use - "mse", "cosine_similarity"/"cos_sim", or "topk"
        threshold: Threshold value for pass/fail
        k: K value for top-k metric (only used when metric="topk")
        verbose: Whether to print detailed results

    Returns:
        bool: True if metric passes threshold, False otherwise
    """
    if verbose:
        print(f"\n{'='*70}")
        print("Comparing Model Outputs")
        print(f"{'='*70}")
        print(f"Model 1: {model_path_1}")
        print(f"Model 2: {model_path_2}")
        print(f"Metric:  {metric} | Threshold: {threshold}" + (f" | K: {k}" if "top" in metric.lower() else ""))
        print(f"{'='*70}\n")

    output_1 = get_output(model_path_1)
    output_2 = get_output(model_path_2)
    metric = metric.lower().replace("-", "_")

    # Calculate metric and check threshold
    if metric == "mse":
        value = torch.mean((output_1.float() - output_2.float()) ** 2).item()
        passed = value <= threshold
        if verbose:
            print(f"MSE: {value:.6f} | Threshold: <= {threshold} | {'✓ PASS' if passed else '✗ FAIL'}\n")

    elif metric in ["cosine_similarity", "cos_sim", "cosine"]:
        out1 = output_1.float().flatten()
        out2 = output_2.float().flatten()
        value = torch.nn.functional.cosine_similarity(out1.unsqueeze(0), out2.unsqueeze(0)).item()
        passed = value >= threshold
        if verbose:
            print(f"Cosine Similarity: {value:.6f} | Threshold: >= {threshold} | {'✓ PASS' if passed else '✗ FAIL'}\n")

    elif metric in ["topk", "top_k"]:
        _, topk_1 = torch.topk(output_1, k=min(k, output_1.size(-1)), dim=-1)
        _, topk_2 = torch.topk(output_2, k=min(k, output_2.size(-1)), dim=-1)

        total_agreement = 0
        total_positions = topk_1.numel() // topk_1.size(-1)

        for i in range(topk_1.size(0)):
            for j in range(topk_1.size(1)):
                set1 = set(topk_1[i, j].tolist())
                set2 = set(topk_2[i, j].tolist())
                total_agreement += len(set1 & set2) / k

        value = total_agreement / total_positions
        passed = value >= threshold
        if verbose:
            print(
                f"Top-{k} Agreement: {value:.4%} | Threshold: >= {threshold:.4%} | {'✓ PASS' if passed else '✗ FAIL'}\n"
            )

    else:
        raise ValueError(f"Unknown metric: {metric}. Choose from: 'mse', 'cosine_similarity', 'topk'")

    return passed
