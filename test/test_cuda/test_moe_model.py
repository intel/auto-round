import shutil

import pytest
import torch
from transformers import AutoConfig, AutoTokenizer, Llama4ForConditionalGeneration
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

from auto_round import AutoRound


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
