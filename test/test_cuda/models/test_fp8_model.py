import shutil

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from auto_round import AutoRound


@pytest.fixture
def setup_qwen_fp8():
    """Fixture to set up the GPT-OSS model and tokenizer."""
    model_name = "INC4AI/Qwen3-30B-A3B-Instruct-2507-FP8-2Layers"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    output_dir = "test_quantized_qwen3_fp8_moe_mxfp"
    return tokenizer, output_dir, config, model_name


def test_qwen3_fp8_moe_mxfp(setup_qwen_fp8):
    tokenizer, output_dir, config, model_name = setup_qwen_fp8
    autoround = AutoRound(
        model_name,
        scheme="MXFP4",
        nsamples=2,
        seqlen=32,
        iters=0,
    )
    quantized_model, _ = autoround.quantize_and_save(format="auto_round", output_dir=output_dir)
    assert quantized_model is not None, "Quantized model should not be None."
    loaded_model = AutoModelForCausalLM.from_pretrained(output_dir)
    loaded_model.to("cuda")
    quantized_model.to("cuda")
    for n, m in quantized_model.named_modules():
        if m.__class__.__name__ == "QuantLinear":
            loaded_m = loaded_model.get_submodule(n)
            assert (loaded_m.weight_packed == m.weight_packed).all()
    # Expect all linear in experts are quantized
    for n, m in quantized_model.named_modules():
        if "experts" in m.__class__.__name__.lower():
            for sub_n, sub_m in m.named_modules():
                assert sub_m.__class__.__name__ == "QuantLinear", f"Module {n}.{sub_n} is not quantized."
    inp = torch.randint(0, 100, (1, 64)).to("cuda")
    with torch.inference_mode():
        loaded_out = loaded_model(inp)

    # test generation
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    text = "There is a girl who likes adventure,"
    inputs = tokenizer(text, return_tensors="pt").to(device=loaded_model.device)
    print(tokenizer.decode(loaded_model.generate(**inputs, max_new_tokens=50)[0]))
    # clean the output directory after test
    shutil.rmtree(output_dir, ignore_errors=True)
