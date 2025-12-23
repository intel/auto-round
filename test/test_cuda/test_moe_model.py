import shutil

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, Llama4ForConditionalGeneration
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3ForCausalLM, Qwen3MLP

from auto_round import AutoRound
from auto_round.modelling.replace_modules import ReplacementModuleBase
from auto_round.utils import LazyImport, logger, unsupported_meta_device


@pytest.fixture
def setup_gpt_oss():
    """Fixture to set up the GPT-OSS model and tokenizer."""
    model_name = "/models/gpt-oss-20b-BF16"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.num_hidden_layers = 1  # Reduce layers for testing
    model = GptOssForCausalLM(config)
    output_dir = "test_quantized_gpt_oss"
    return model, tokenizer, output_dir, config


@pytest.fixture
def setup_llama4():
    """Fixture to set up the llama4 model and tokenizer."""
    model_name = "/dataset/Llama-4-Scout-17B-16E-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.vision_config.num_hidden_layers = 2  # Reduce layers for testing
    config.text_config.num_hidden_layers = 2
    model = Llama4ForConditionalGeneration(config)
    output_dir = "test_quantized_llama4"
    return model, tokenizer, output_dir, config


def quantize_model(model, tokenizer, output_dir, scheme, iters=0):
    """Helper function to quantize the model with the given scheme."""
    autoround = AutoRound(
        model,
        tokenizer,
        scheme=scheme,
        nsamples=2,
        iters=iters,
        fp_layers="self_attn,router,lm_head,mlp.gate",
    )
    quantized_model, save_folder = autoround.quantize_and_save(format="auto_round", output_dir=output_dir)
    return quantized_model


def test_gptoss(setup_gpt_oss):
    model, tokenizer, output_dir, config = setup_gpt_oss

    # Below parameter is set to be same as the full model
    # Remove it to avoid mismatch during quantized model loading
    delattr(model.config, "layer_types")

    quantized_model = quantize_model(model, tokenizer, output_dir, "MXFP4")

    # Ensure the quantized model is not None
    assert quantized_model is not None, "Quantized model should not be None."

    loaded_model = GptOssForCausalLM.from_pretrained(output_dir)
    quantized_model.to("cuda")
    loaded_model.to("cuda")
    for n, m in quantized_model.named_modules():
        if m.__class__.__name__ == "QuantLinear":
            loaded_m = loaded_model.get_submodule(n)
            assert (loaded_m.weight_packed == m.weight_packed).all()

    inp = torch.randint(0, 100, (1, 64)).to("cuda")
    with torch.inference_mode():
        loaded_out = loaded_model(inp)

    # clean the output directory after test
    shutil.rmtree(output_dir, ignore_errors=True)


def test_llama4(setup_llama4):
    model, tokenizer, output_dir, config = setup_llama4

    # Below parameters are set to be same as the full model
    # Remove them to avoid mismatch during quantized model loading
    model.config.text_config.no_rope_layers = []
    delattr(model.config.text_config, "moe_layers")
    delattr(model.config.text_config, "layer_types")

    quantized_model = quantize_model(model, tokenizer, output_dir, "MXFP4")

    # Ensure the quantized model is not None
    assert quantized_model is not None, "Quantized model should not be None."

    loaded_model = Llama4ForConditionalGeneration.from_pretrained(output_dir)
    quantized_model.to("cuda")
    loaded_model.to("cuda")
    for n, m in quantized_model.named_modules():
        if m.__class__.__name__ == "QuantLinear":
            loaded_m = loaded_model.get_submodule(n)
            assert (loaded_m.weight_packed == m.weight_packed).all()

    inp = torch.randint(0, 100, (1, 64)).to("cuda")
    with torch.inference_mode():
        loaded_out = loaded_model(inp)

    # clean the output directory after test
    shutil.rmtree(output_dir, ignore_errors=True)


class NewQwen3MLP(ReplacementModuleBase):
    def __init__(self, original: Qwen3MLP, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.new_gate_proj = original.gate_proj
        self.new_up_proj = original.up_proj
        self.new_down_proj = original.down_proj
        self.act_fn = nn.GELU()

    def forward(self, x):
        gate = self.new_gate_proj(x)
        up = self.new_up_proj(x)
        gate_act = self.act_fn(gate)
        act = gate_act * up
        out = self.new_down_proj(act)
        return out

    @classmethod
    def original_module_class(cls) -> str:
        return "Qwen3MLP"

    @classmethod
    def from_original(cls, original: Qwen3MLP, config: Qwen3Config):
        return cls(original, config)


@pytest.fixture
def setup_qwen3():
    """Fixture to set up the qwen3 model and tokenizer."""
    model_name = "/models/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    config.num_hidden_layers = 2  # Reduce layers for testing
    config.layer_types = config.layer_types[:2]  # Reduce layers for testing
    model = Qwen3ForCausalLM(config)
    output_dir = "test_quantized_qwen3"
    return model, tokenizer, output_dir, config


def has_module(model: torch.nn.Module, module_name: str) -> bool:
    for n, m in model.named_modules():
        if module_name in n:
            return True
    return False


def test_register_module_out_of_tree_base():
    from auto_round.modelling.replace_modules import ReplacementModuleBase

    for name, subclass in ReplacementModuleBase._replacement_registry.items():
        if name == "Qwen3MLP":
            assert subclass == NewQwen3MLP, "Qwen3MLP not registered correctly."


def test_register_module_out_of_tree_model(setup_qwen3):
    model, tokenizer, output_dir, config = setup_qwen3
    quantized_model = quantize_model(model, tokenizer, output_dir, "MXFP4")
    # Ensure the quantized model is not None
    assert quantized_model is not None, "Quantized model should not be None."
    check_module_names = ["new_gate_proj", "new_up_proj", "new_down_proj"]
    for name in check_module_names:
        assert has_module(quantized_model, name), f"Module {name} not found in quantized model."
    loaded_model = Qwen3ForCausalLM.from_pretrained(output_dir)
    quantized_model.to("cuda")

    for name in check_module_names:
        assert has_module(loaded_model, name), f"Module {name} not found in loaded model."
    loaded_model.to("cuda")
    for n, m in quantized_model.named_modules():
        if m.__class__.__name__ == "QuantLinear":
            loaded_m = loaded_model.get_submodule(n)
            assert (loaded_m.weight_packed == m.weight_packed).all()

    inp = torch.randint(0, 100, (1, 64)).to("cuda")
    with torch.inference_mode():
        loaded_out = loaded_model(inp)

    # clean the output directory after test
    shutil.rmtree(output_dir, ignore_errors=True)
