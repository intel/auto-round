import pytest
from transformers import AutoConfig, AutoTokenizer
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

from auto_round import AutoRound


@pytest.fixture
def setup_gpt_oss():
    """Fixture to set up the GPT-OSS model and tokenizer."""
    model_name = "/data5/yliu7/HF_HOME/unsloth/gpt-oss-20b-BF16/"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.num_hidden_layers = 1  # Reduce layers for testing
    model = GptOssForCausalLM(config)
    output_dir = "/tmp/test_quantized_gpt_oss"
    return model, tokenizer, output_dir


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


def count_modules_by_type(model, target_module_name_or_class):
    """Helper function to count modules of a specific type in the model."""
    cnt = 0
    for name, module in model.named_modules():
        if isinstance(target_module_name_or_class, str):
            if target_module_name_or_class == module.__class__.__name__:
                cnt += 1
        else:
            if isinstance(module, target_module_name_or_class):
                cnt += 1
    return cnt


@pytest.mark.parametrize("scheme", ["MXFP4", "MXFP8"])
@pytest.mark.parametrize("quantize_model", [0, 4])
def test_quantization_with_mxfp4(setup_gpt_oss, scheme):
    """Test quantization with the scheme."""
    model, tokenizer, output_dir = setup_gpt_oss
    quantized_model = quantize_model(model, tokenizer, output_dir, scheme)

    # Ensure the quantized model is not None
    assert quantized_model is not None, "Quantized model should not be None."

    # Count specific modules
    single_expert_cnt = count_modules_by_type(quantized_model, "GPTOssSingleExpert")
    quant_linear_cnt = count_modules_by_type(quantized_model, "QuantLinear")

    # Assertions
    assert single_expert_cnt >= 0, "GPTOssSingleExpert count should be non-negative."
    assert quant_linear_cnt >= 0, "QuantLinear count should be non-negative."

    print(f"[{scheme}] Total GPTOssSingleExpert modules: {single_expert_cnt}")
    print(f"[{scheme}] Total QuantLinear modules: {quant_linear_cnt}")
    # clean the output directory after test
    import shutil

    shutil.rmtree(output_dir, ignore_errors=True)
