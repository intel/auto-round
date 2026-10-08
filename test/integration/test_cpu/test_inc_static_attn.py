import shutil
from test.helpers import get_model_path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig

from auto_round import AutoRound
from auto_round.experimental.attention import HOOKED_ATTENTION_NAME, QUERY_MAX_NAME, attention_quant_ctx

deepseekv3_model_name = get_model_path("tflsxyy/DeepSeek-V3-bf16-4layers")


def assert_valid_q_scale(model):
    q_scale = model.model.layers[0].self_attn.q_scale
    assert torch.isfinite(q_scale).all(), f"q_scale must be finite, got {q_scale}"
    assert (q_scale > 0).all(), f"q_scale must be positive, got {q_scale}"


def test_attention_context_uses_module_config_and_restores_it():
    class TestAttention(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.layer_idx = 0
            self.k_proj = torch.nn.Linear(4, 4, bias=False)

    model_config = PretrainedConfig()
    model_config._attn_implementation = "eager"
    attention_config = PretrainedConfig()
    attention_config._attn_implementation = "eager"
    attention = TestAttention(attention_config)
    model = torch.nn.Module()
    model.config = model_config
    model.attention = attention

    with attention_quant_ctx(model):
        assert model_config._attn_implementation == "eager"
        assert attention_config._attn_implementation == HOOKED_ATTENTION_NAME
        assert hasattr(attention, "impl")
        assert hasattr(attention, "q_scale")
        assert hasattr(attention, QUERY_MAX_NAME)

    assert model_config._attn_implementation == "eager"
    assert attention_config._attn_implementation == "eager"
    assert hasattr(attention, "q_scale")
    assert not hasattr(attention_config, "_auto_round_original_attn_impl")
    assert not hasattr(attention, QUERY_MAX_NAME)


def test_deepseek_v2(tiny_deepseek_v2_model_path):
    model_name = tiny_deepseek_v2_model_path
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    output_dir = "./tmp/test_quantized_deepseekv2"
    autoround = AutoRound(
        model,
        tokenizer,
        scheme="FP8_STATIC",
        static_attention_dtype="fp8",
        iters=0,
        seqlen=2,
        trust_remote_code=False,
    )
    quantized_model, save_folder = autoround.quantize_and_save(format="llm_compressor", output_dir=output_dir)
    assert quantized_model is not None, "Expected quantized_model to be not None"
    assert_valid_q_scale(quantized_model)

    # clean the output directory after test
    shutil.rmtree(output_dir, ignore_errors=True)


@pytest.fixture
def setup_deepseekv3():
    """Fixture to set up model and tokenizer."""
    from transformers import AutoConfig

    model_name = deepseekv3_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
    config.num_hidden_layers = 1  # Reduce layers for testing
    model = AutoModelForCausalLM.from_config(config)
    model.config.name_or_path = None
    output_dir = "./tmp/test_quantized_deepseekv3"
    return model, tokenizer, output_dir, config


def test_deepseek_v3(setup_deepseekv3):
    model, tokenizer, output_dir, config = setup_deepseekv3
    autoround = AutoRound(
        model,
        tokenizer,
        scheme="FP8_STATIC",
        static_attention_dtype="fp8",
        iters=0,
        seqlen=2,
        trust_remote_code=False,
    )
    quantized_model, save_folder = autoround.quantize_and_save(format="llm_compressor", output_dir=output_dir)
    assert quantized_model is not None, "Expected quantized_model to be not None"
    assert_valid_q_scale(quantized_model)

    # clean the output directory after test
    shutil.rmtree(output_dir, ignore_errors=True)
