import os
import torch
import tqdm
from loguru import logger
import logging
import safetensors
from safetensors import safe_open
from safetensors.torch import save_file
import json

logging.basicConfig(level=logging.DEBUG)
torch.set_grad_enabled(False)

# CONSTANTS
SAFETENSORS = "safetensors"
WEIGHT_SCALE_NAME = "weight_scale"
INPUT_SCALE_NAME = "scale_input"
SCALE_DTYPE = torch.bfloat16
SCALE_FILE_NAME = f"scales.{SAFETENSORS}"
FULL_RANGE = torch.finfo(torch.float8_e4m3fn).max
WEIGHT_BACKOFF = 1.0
QUANT_MODULE_TYPES = (torch.nn.Linear,)
SKIP_WEIGHT_LST = {
    "model.norm",
    "layernorm",
    "e_score_correction_bias",
    # "lm_head.weight",
    "embed_tokens",
    "mlp.gate.weight",  # mlp.gate is not linear
}

MODEL_STATE_DICT_MAPPING_FILENAME = "model.safetensors.index.json"


seed = 0
import random

random.seed(seed)
import torch

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import numpy as np

np.random.seed(seed)


# torch.use_deterministic_algorithms(True)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)




def pre_dequantize(model):
    """
    Pre-dequantize all FP8QDQLinear layers in the model.
    """
    for name, module in model.named_modules():
        if module.__class__.__name__ == "FP8QDQLinear":
            logger.info(f"Pre-dequantizing {name}")
            module.pre_dequantize()
        else:
            logger.debug(f"Skipping {name} as it is not FP8QDQLinear")


def qdq_eval(model_path, not_patch_lin=False):
    import transformers
    from transformers.modeling_utils import no_init_weights


    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    logger.info(f"Patched model: {model}")
    model.eval()
    model.to("cuda")
    import torch

    model = torch.compile(model)
    # pre_dequantize(model)
    with torch.device("cuda"):
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        prompt = "Hi, who"
        encode = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            output_tokens = model.generate(encode, max_length=100)
            output = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Output: {output}")

    # from auto_round.script.llm import eval_task_by_task

    # eval_task_by_task(
    #     model=model,
    #     device="cuda",
    #     tasks="gsm8k",
    #     batch_size=32,
    #     limit=128,
    #     # trust_remote_code=not args.disable_trust_remote_code,
    #     # eval_model_dtype=args.eval_model_dtype
    # )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--qmodel_path", type=str, required=True)
    parser.add_argument(
        "--not_patch_lin", action="store_true", help="Measure float model"
    )
    args = parser.parse_args()
    qdq_eval(args.qmodel_path, not_patch_lin=args.not_patch_lin)


"""
p load_w8a8.py --qmodel_path  /data5/yliu7/HF_HOME/Qwen3-32B-w8afp8
Running generate_until requests:  76%|███ | 97/128 [11:45<03:
Running generate_until requests: 100%|███| 128/128 [11:45<00:00,  5.51s/it]
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7422|±  |0.0388|
|     |       |strict-match    |     5|exact_match|↑  |0.6797|±  |0.0414|

total eval time: 742.8823928833008
"""