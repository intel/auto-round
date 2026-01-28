import shutil

import pytest
from packaging import version
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, Llama4ForConditionalGeneration
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration

from auto_round import AutoRound

from ...helpers import get_model_path, transformers_version

gpt_oss_name_or_path = get_model_path("unsloth/gpt-oss-20b-BF16")
llama4_name_or_path = get_model_path("meta-llama/Llama-4-Scout-17B-16E-Instruct")
qwen3_vl_moe_name_or_path = get_model_path("Qwen/Qwen3-VL-30B-A3B-Instruct")
# local path for debug
# llama4_name_or_path = get_model_path("/dataset/Llama-4-Scout-17B-16E-Instruct")


@pytest.fixture
def setup_gpt_oss():
    """Fixture to set up the GPT-OSS model and tokenizer."""
    model_name = gpt_oss_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.num_hidden_layers = 1  # Reduce layers for testing
    model = GptOssForCausalLM(config)
    output_dir = "./tmp/test_quantized_gpt_oss"
    return model, tokenizer, output_dir, config


@pytest.fixture
def setup_llama4():
    """Fixture to set up the llama4 model and tokenizer."""
    model_name = llama4_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.vision_config.num_hidden_layers = 1  # Reduce layers for testing
    config.text_config.num_hidden_layers = 1
    # config.vision_config.rope_theta = config.vision_config.rope_parameters["rope_theta"] # for transformers >= 5.0
    model = Llama4ForConditionalGeneration(config)
    output_dir = "./tmp/test_quantized_llama4"
    return model, tokenizer, output_dir, config


@pytest.fixture
def setup_qwen3_vl_moe():
    """Fixture to set up the qwen3_vl_moe model and tokenizer."""
    model_name = qwen3_vl_moe_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    config.vision_config.num_hidden_layers = 1
    config.text_config.num_hidden_layers = 1
    config.num_hidden_layers = 1  # Reduce layers for testing
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen3VLMoeForConditionalGeneration(config)
    output_dir = "/tmp/test_quantized_qwen3_vl_moe"
    return model, tokenizer, processor, output_dir, config


def quantize_model(model, tokenizer, output_dir, scheme, iters=0):
    """Helper function to quantize the model with the given scheme."""
    autoround = AutoRound(
        model,
        tokenizer,
        scheme=scheme,
        nsamples=2,
        iters=iters,
        seqlen=32,
        ignore_layers="self_attn,router,lm_head,mlp.gate",
    )
    quantized_model, save_folder = autoround.quantize_and_save(format="auto_round", output_dir=output_dir)
    return quantized_model, save_folder


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
def test_gptoss(setup_gpt_oss, scheme):
    model, tokenizer, output_dir, config = setup_gpt_oss

    # Below parameter is set to be same as the full model
    # Remove it to avoid mismatch during quantized model loading
    delattr(model.config, "layer_types")

    quantized_model, output_dir = quantize_model(model, tokenizer, output_dir, scheme)

    # Ensure the quantized model is not None
    assert quantized_model is not None, "Quantized model should not be None."
    from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear
    from auto_round.modelling.gpt_oss import GPTOssSingleExpert

    single_expert_cnt = count_modules_by_type(quantized_model, GPTOssSingleExpert)
    quant_linear_cnt = count_modules_by_type(quantized_model, QuantLinear)
    assert (
        single_expert_cnt == config.num_local_experts
    ), f"Expected {config.num_local_experts} GPTOssSingleExpert modules, found {single_expert_cnt}."
    assert (
        quant_linear_cnt == config.num_hidden_layers * 3 * config.num_local_experts
    ), f"Expected {config.num_hidden_layers * 3 * config.num_local_experts} QuantLinear modules, found {quant_linear_cnt}."

    print(f"[{scheme}] Total {GPTOssSingleExpert.__name__} modules: {single_expert_cnt}")
    print(f"[{scheme}] Total {QuantLinear.__name__} modules: {quant_linear_cnt}")

    if scheme == "MXFP4":
        loaded_model = GptOssForCausalLM.from_pretrained(output_dir)
        for n, m in quantized_model.named_modules():
            if m.__class__.__name__ == "QuantLinear":
                loaded_m = loaded_model.get_submodule(n)
                assert (loaded_m.weight_packed.to("cpu") == m.weight_packed.to("cpu")).all()
    # clean the output directory after test
    shutil.rmtree(output_dir, ignore_errors=True)


@pytest.mark.skipif(
    transformers_version >= version.parse("5.0.0"),
    reason="transformers v5 'Llama4VisionConfig' object has no attribute 'rope_theta'",
)
def test_llama4(setup_llama4):
    model, tokenizer, output_dir, config = setup_llama4

    # Below parameters are set to be same as the full model
    # Remove them to avoid mismatch during quantized model loading
    model.config.text_config.no_rope_layers = []
    delattr(model.config.text_config, "moe_layers")
    delattr(model.config.text_config, "layer_types")

    quantized_model, output_dir = quantize_model(model, tokenizer, output_dir, "MXFP4")

    # Ensure the quantized model is not None
    assert quantized_model is not None, "Quantized model should not be None."

    loaded_model = Llama4ForConditionalGeneration.from_pretrained(output_dir)
    for n, m in quantized_model.named_modules():
        if m.__class__.__name__ == "QuantLinear":
            loaded_m = loaded_model.get_submodule(n)
            assert (loaded_m.weight_packed.to("cpu") == m.weight_packed.to("cpu")).all()
    # clean the output directory after test
    shutil.rmtree(output_dir, ignore_errors=True)


def test_qwen3_vl_moe_mxfp(setup_qwen3_vl_moe):
    model, tokenizer, processor, output_dir, config = setup_qwen3_vl_moe
    autoround = AutoRound(
        model,
        tokenizer=tokenizer,
        processor=processor,
        scheme="MXFP4",
        nsamples=2,
        seqlen=32,
        iters=1,
        ignore_layers="self_attn,lm_head,mlp.gate",
    )
    quantized_model, output_dir = autoround.quantize_and_save(format="auto_round", output_dir=output_dir)
    assert quantized_model is not None, "Quantized model should not be None."
    loaded_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(output_dir, device_map="cpu")

    for n, m in quantized_model.named_modules():
        if m.__class__.__name__ == "QuantLinear":
            loaded_m = loaded_model.get_submodule(n)
            assert (loaded_m.weight_packed.to("cpu") == m.weight_packed.to("cpu")).all()
    # test generation
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    text = "There is a girl who likes adventure,"
    inputs = tokenizer(text, return_tensors="pt").to(device=loaded_model.device)
    print(tokenizer.decode(loaded_model.generate(**inputs, max_new_tokens=50)[0]))
    # clean the output directory after test
    shutil.rmtree(output_dir, ignore_errors=True)
