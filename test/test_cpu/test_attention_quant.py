import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round.experimental.attention import attention_quant_ctx

model_name = "Kimi-K2-Instruct-BF16"
model_name = "/models/Qwen3-30B-A3B"
model_name = "facebook/opt-125m"

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model_name = "/models/DeepSeek-V2-Lite-Chat/"
model_name = "/data5/yliu7/HF_HOME/unsloth/gpt-oss-20b-BF16"
model_name = "/data4/yliu/unsloth/gpt-oss-120b-BF16"
model_name = "/data5/yliu7/HF_HOME/Qwen3-30B-A3B"
model_name = "/models/Qwen3-15B-A2B-Base"
model_name = "/data5/yliu7/HF_HOME/unsloth/gpt-oss-20b-BF16/"
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model_name = "/data5/yliu7/HF_HOME/Qwen/Qwen2.5-0.5B-Instruct/"
model_name = "/data6/GLM-4.5-Air"
model_name = "/data5/yliu7/HF_HOME/meta-llama/Llama-3.2-1B-Instruct/"


def check_param(model):
    for name, param in model.named_parameters():
        if "scale" in name:
            print(
                f"{name}: shape={param.shape}, dtype={param.dtype}, device={param.device}, min={param.min().item()}, max={param.max().item()}"
            )


import shutil
import tempfile

import pytest

from auto_round import AutoRound
from auto_round import schemes as ar_schemes
from auto_round.experimental import qmodules as ar_qmodules
from auto_round.export.export_to_autoround import AutoRoundFormat
from auto_round.export.export_to_autoround import qlinear_fp as ar_qlinear_fp


def has_param(quantized_model_path, target_param_name: str) -> bool:
    """Check if a parameter with the given name exists in the model."""
    # Load all safetensors in the directory
    import os

    from safetensors import safe_open

    for file_name in os.listdir(quantized_model_path):
        if file_name.endswith(".safetensors"):
            file_path = os.path.join(quantized_model_path, file_name)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for param_name in f.keys():
                    if target_param_name in param_name:
                        return True
    return False


def test_quant_attention():

    with tempfile.TemporaryDirectory() as temp_dir:
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        model_name = "/data5/yliu7/HF_HOME/meta-llama/Llama-3.2-1B-Instruct/"

        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype="auto",
        )

        scheme = "FP8_STATIC"
        # Initialize AutoRound for quantization
        autoround = AutoRound(
            model,
            tokenizer,
            scheme=scheme,
            iters=0,
            nsamples=2,
            static_attention_dtype="fp8",
        )

        # Quantize and save the model to the temporary directory
        quantized_model_path = f"{temp_dir}/tmp_autoround_{scheme}"
        autoround.quantize_and_save(format="auto_round", output_dir=quantized_model_path)

        assert has_param(quantized_model_path, "q_scale"), "Quantization parameter not found in the model."
        assert has_param(quantized_model_path, "k_scale"), "Quantization parameter not found in the model."
        assert has_param(quantized_model_path, "v_scale"), "Quantization parameter not found in the model."

        # Perform inference with the quantized model
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path,
            torch_dtype="auto",
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        prompt = "Ai is "

        # Tokenize the input prompt
        encode = tokenizer.encode(prompt, return_tensors="pt")

        # Generate output tokens
        output_tokens = model.generate(
            encode,
            max_length=30,
        )
        output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # Print and validate the output
        print(f"Prompt: {prompt}")
        print(f"Output: {output}")
        assert output is not None, "Output should not be None"


# with torch.device("cuda"):
#     tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
#     config = transformers.AutoConfig.from_pretrained(model_name,trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto",trust_remote_code=True)
#     # from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

#     with attention_quant_ctx(model):
#         print(model)
#         prompt = "The future of AI is "
#         input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
#         output_tokens = model.generate(
#             input_ids,
#             max_length=30,
#         )
#         output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
#         print(f"Prompt: {prompt}")
#         print(f"Output: {output}")
#         check_param(model)
