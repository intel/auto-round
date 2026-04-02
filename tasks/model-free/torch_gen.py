import torch
import torch.nn as nn

# ============================================================================
# Out-of-tree patches for FineGrainedFP8 to support static quantization
# (per-tensor weight_scale + input_scale) used by Hunyuan-A13B-Instruct-FP8
# ============================================================================

# === Patch 1: Allow static activation scheme in FineGrainedFP8Config ===
from transformers.utils.quantization_config import FineGrainedFP8Config

_orig_post_init = FineGrainedFP8Config.post_init


def _patched_post_init(self):
    self.activation_scheme = self.activation_scheme.lower()
    if self.activation_scheme not in ["dynamic", "static"]:
        raise ValueError(f"Activation scheme {self.activation_scheme} not supported")
    if not hasattr(self, "weight_block_size"):
        self.weight_block_size = None  # per-tensor, no block size
    elif self.weight_block_size is not None:
        if len(self.weight_block_size) != 2:
            raise ValueError("weight_block_size must be a tuple of two integers")
        if self.weight_block_size[0] <= 0 or self.weight_block_size[1] <= 0:
            raise ValueError("weight_block_size must be a tuple of two positive integers")


FineGrainedFP8Config.post_init = _patched_post_init


# === Patch 2: Static FP8 Linear module ===
_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MAX = torch.finfo(_FP8_DTYPE).max


class StaticFP8Linear(nn.Module):
    """Per-tensor static FP8 linear layer.

    Weights stored in fp8_e4m3fn with scalar weight_scale.
    Input quantized at inference using pre-calibrated scalar input_scale.
    Dequant formula: dequantized = fp8_value * scale
    """

    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=_FP8_DTYPE, device=device))
        self.weight_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=device))
        self.register_buffer("input_scale", torch.tensor(1.0, dtype=torch.bfloat16, device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        original_dtype = x.dtype
        # Dequantize weight: fp8 -> compute dtype, then scale
        w = self.weight.to(original_dtype) * self.weight_scale.to(original_dtype)
        # Static activation quantization (QDQ): quantize input then dequantize
        input_scale_f = self.input_scale.to(torch.float32)
        x_scaled = x.to(torch.float32) / input_scale_f
        x_clamped = torch.clamp(x_scaled, -_FP8_MAX, _FP8_MAX)
        x_fp8 = x_clamped.to(_FP8_DTYPE)
        x_qdq = x_fp8.to(original_dtype) * input_scale_f.to(original_dtype)
        return nn.functional.linear(x_qdq, w, self.bias)


# === Patch 3: Patch replace_with_fp8_linear for static scheme ===
import transformers.integrations.finegrained_fp8 as _tfp8

_orig_replace = _tfp8.replace_with_fp8_linear


def _get_quantized_layer_names(model_name_or_path):
    """Read the safetensors index to find which layers have weight_scale (are quantized)."""
    import json, os

    index_path = os.path.join(model_name_or_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return None  # Can't determine, fall back to replacing all
    with open(index_path) as f:
        idx = json.load(f)
    quantized = set()
    for key in idx["weight_map"]:
        if key.endswith(".weight_scale"):
            base = key.rsplit(".weight_scale", 1)[0]
            quantized.add(base)
    return quantized


# Pre-read which layers are quantized from the model checkpoint
_quantized_layers = _get_quantized_layer_names("/mnt/disk1/yiliu7/tencent/Hunyuan-A13B-Instruct-FP8/")


def _patched_replace_with_fp8_linear(model, modules_to_not_convert=None, quantization_config=None):
    if getattr(quantization_config, "activation_scheme", "dynamic") == "static":
        modules_to_not_convert = modules_to_not_convert or []
        if quantization_config.modules_to_not_convert:
            modules_to_not_convert.extend(quantization_config.modules_to_not_convert)
        modules_to_not_convert = list(set(modules_to_not_convert))

        has_been_replaced = False
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear):
                if any(skip in name for skip in modules_to_not_convert):
                    continue
                # Only replace layers that are actually quantized in checkpoint
                if _quantized_layers is not None and name not in _quantized_layers:
                    continue
                with torch.device("meta"):
                    new_mod = StaticFP8Linear(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                    )
                model.set_submodule(name, new_mod)
                has_been_replaced = True

        if not has_been_replaced:
            print(
                "WARNING: No linear modules were replaced with StaticFP8Linear. "
                "Please double check your model architecture."
            )
        return model
    else:
        return _orig_replace(model, modules_to_not_convert, quantization_config)


_tfp8.replace_with_fp8_linear = _patched_replace_with_fp8_linear


# === Patch 4: Fix quantized param check and missing keys for StaticFP8Linear ===
from transformers.quantizers.quantizer_finegrained_fp8 import (
    FineGrainedFP8HfQuantizer,
)

_orig_check_quantized = FineGrainedFP8HfQuantizer.check_quantized_param


def _patched_check_quantized_param(self, model, param_value, param_name, state_dict, **kwargs):
    from transformers.modeling_utils import get_module_from_name

    module, tensor_name = get_module_from_name(model, param_name)
    if isinstance(module, StaticFP8Linear):
        return False  # All params are pre-quantized, load directly
    return _orig_check_quantized(self, model, param_value, param_name, state_dict, **kwargs)


FineGrainedFP8HfQuantizer.check_quantized_param = _patched_check_quantized_param

_orig_update_missing = FineGrainedFP8HfQuantizer.update_missing_keys


def _patched_update_missing_keys(self, model, missing_keys, prefix):
    not_missing = []
    for name, module in model.named_modules():
        if isinstance(module, StaticFP8Linear):
            for mk in missing_keys:
                if (name in mk or name in f"{prefix}.{mk}") and not mk.endswith(".weight"):
                    not_missing.append(mk)
    return [k for k in missing_keys if k not in not_missing]


FineGrainedFP8HfQuantizer.update_missing_keys = _patched_update_missing_keys


# === Patch 5: GPU capability check bypass ===
_orig_validate = FineGrainedFP8HfQuantizer.validate_environment


def _patched_validate(self, *args, **kwargs):
    try:
        _orig_validate(self, *args, **kwargs)
    except (RuntimeError, ValueError):
        pass  # Skip GPU capability check for unsupported GPUs


FineGrainedFP8HfQuantizer.validate_environment = _patched_validate

# ============================================================================
# End of patches — original script below
# ============================================================================

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/mnt/disk1/yiliu7/tencent/Hunyuan-A13B-Instruct-FP8/"
# model_name = "/mnt/disk1/yiliu7/HY3.0-FP8-Testing"
# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True)

# prepare the model input
prompt = "The capital of France is"
model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Remove token_type_ids if present (not used by this model)
if "token_type_ids" in model_inputs:
    del model_inputs["token_type_ids"]

# conduct text completion
generated_ids = model.generate(**model_inputs, max_new_tokens=20)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
content = tokenizer.decode(output_ids).strip("\n")
print("content:", content)
