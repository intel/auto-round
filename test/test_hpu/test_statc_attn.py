import shutil

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.utils import is_hpex_available

from ..helpers import get_model_path, is_pytest_mode_lazy

deepseekv2_model_name = get_model_path("deepseek-ai/DeepSeek-V2-Lite-Chat")


@pytest.fixture
def setup_deepseekv2():
    """Fixture to set up model and tokenizer."""
    model_name = deepseekv2_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
    config.num_hidden_layers = 1  # Reduce layers for testing
    model = AutoModelForCausalLM.from_config(config)
    output_dir = "./tmp/test_quantized_deepseekv2"
    return model, tokenizer, output_dir, config


@pytest.mark.skipif(not is_hpex_available(), reason="HPU is not supported")
@pytest.mark.skipif(not is_pytest_mode_lazy(), reason="Only for lazy mode")
def test_deepseek_v2_on_hpu(setup_deepseekv2):
    model, tokenizer, output_dir, config = setup_deepseekv2
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
    device = quantized_model.model.layers[0].self_attn.q_scale.device
    assert not all(
        quantized_model.model.layers[0].self_attn.q_scale == torch.tensor([0.0], device=device)
    ), "q_scale is not collected"

    # clean the output directory after test
    shutil.rmtree(output_dir, ignore_errors=True)
