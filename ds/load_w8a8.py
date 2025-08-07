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


def skip_weight(weight_name):
    return any([skip_name in weight_name for skip_name in SKIP_WEIGHT_LST])


def get_cpu_mem_size_in_gb():
    import psutil

    mem = psutil.virtual_memory()
    return mem.available


from torch import nn


# from _fp8_quant/_core/fp_utils.py
def calc_maxabs_scale(xmaxabs, fullscale, backoff=1):
    scale = xmaxabs / (fullscale * backoff)
    return scale


def quant_tensor(tensor):
    # Note:
    #  1. Check the scale dtype
    #  2. Check the scale shape
    amax = tensor.abs().max()
    scale = calc_maxabs_scale(amax, FULL_RANGE, WEIGHT_BACKOFF)
    scale = scale.to(SCALE_DTYPE)
    qtensor = tensor / scale
    cliped_qtensor = torch.clamp(qtensor, -FULL_RANGE, FULL_RANGE)
    cliped_qtensor_fp8 = cliped_qtensor.to(torch.float8_e4m3fn)
    return scale, cliped_qtensor_fp8


def quant_tensor_with_scale(tensor, scale):
    # Note:
    #  1. Check the scale dtype
    #  2. Check the scale shape
    qtensor = tensor / scale
    cliped_qtensor = torch.clamp(qtensor, -FULL_RANGE, FULL_RANGE)
    cliped_qtensor_fp8 = cliped_qtensor.to(torch.float8_e4m3fn)
    return scale, cliped_qtensor_fp8


# Adapted from https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/1d044fd82b15f1cedb197a288e50cc96a2c27205/inference/model.py#L91-L108
class FP8QDQLinear(torch.nn.Linear):
    dtype = torch.bfloat16
    fp8_dtype = torch.float8_e4m3fn

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, device=None
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=FP8QDQLinear.fp8_dtype),
            requires_grad=True,
        )
        self.weight_scale = nn.Parameter(
            torch.zeros((out_features, 1), dtype=FP8QDQLinear.dtype),
            requires_grad=False,
        )
        self.input_scale = nn.Parameter(
            torch.zeros((1, 1), dtype=FP8QDQLinear.dtype), requires_grad=False
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.pre_dequantized = False

    def dequant_weight_online(self):
        if self.pre_dequantized:
            return self.weight
        fp8_weight = self.weight
        qdq_weight = fp8_weight.to(FP8QDQLinear.dtype) * self.weight_scale
        return qdq_weight

    def pre_dequantize(self):
        if self.pre_dequantized:
            return
        dequant_weight = self.dequant_weight_online()
        del self.weight
        del self.weight_scale
        self.weight = nn.Parameter(dequant_weight, requires_grad=False)
        self.pre_dequantized = True

    def qdq_input(self, bf16_input: torch.Tensor):
        input_scale, input_fp8 = quant_tensor_with_scale(
            bf16_input, self.input_scale.data
        )
        qdq_input_bf16 = input_fp8.to(FP8QDQLinear.dtype) * input_scale
        return qdq_input_bf16

    @classmethod
    def create_from_linear(cls, linear: nn.Linear):
        qdq_linear = cls(linear.in_features, linear.out_features)
        qdq_linear.weight.data = linear.weight.data
        if linear.bias is not None:
            qdq_linear.bias = linear.bias
        return qdq_linear

    def forward(self, bf16_input: torch.Tensor) -> torch.Tensor:
        qdq_input = self.qdq_input(bf16_input)
        qdq_weight = self.dequant_weight_online()
        out = torch.nn.functional.linear(qdq_input, qdq_weight, self.bias)
        return out


def patch_lin():
    logger.warning("Patching torch.nn.Linear to FP8QDQLinear")
    torch.nn.Linear = FP8QDQLinear


def pre_dequantize(model):
    """
    Pre-dequantize all FP8QDQLinear layers in the model.
    """
    for name, module in model.named_modules():
        if isinstance(module, FP8QDQLinear):
            logger.info(f"Pre-dequantizing {name}")
            module.pre_dequantize()
        else:
            logger.debug(f"Skipping {name} as it is not FP8QDQLinear")


def qdq_eval(model_path, not_patch_lin=False):
    import transformers
    from transformers.modeling_utils import no_init_weights

    if not not_patch_lin:
        patch_lin()

    def _patch__initialize_weights(self, module):
        module._is_hf_initialized = True

    transformers.modeling_utils.PreTrainedModel._initialize_weights = (
        _patch__initialize_weights
    )
    # patch_transformers()
    with no_init_weights():
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

    from auto_round.script.llm import eval_task_by_task

    eval_task_by_task(
        model=model,
        device="cuda",
        tasks="gsm8k",
        batch_size=32,
        limit=128,
        # trust_remote_code=not args.disable_trust_remote_code,
        # eval_model_dtype=args.eval_model_dtype
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--qmodel_path", type=str, required=True)
    parser.add_argument(
        "--not_patch_lin", action="store_true", help="Measure float model"
    )
    args = parser.parse_args()
    qdq_eval(args.qmodel_path, not_patch_lin=args.not_patch_lin)
