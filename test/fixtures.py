import os
import shutil
from unittest.mock import patch

import datasets
import pytest
import torch
import transformers

from .helpers import (
    DataLoader,
    deepseek_v2_name_or_path,
    gemma_name_or_path,
    get_model_path,
    get_tiny_model,
    gptj_name_or_path,
    lamini_name_or_path,
    opt_name_or_path,
    phi2_name_or_path,
    qwen2_5_omni_name_or_path,
    qwen3_omni_name_or_path,
    qwen_2_5_vl_name_or_path,
    qwen_moe_name_or_path,
    qwen_name_or_path,
    qwen_vl_name_or_path,
    save_tiny_model,
)


# Create tiny model path fixtures for testing
@pytest.fixture(scope="session")
def tiny_opt_model_path():
    model_name_or_path = opt_name_or_path
    tiny_model_path = "./tmp/tiny_opt_model_path"
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_lamini_model_path():
    model_name_or_path = lamini_name_or_path
    tiny_model_path = "./tmp/tiny_lamini_model_path"
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_gptj_model_path():
    model_name_or_path = gptj_name_or_path
    tiny_model_path = "./tmp/tiny_gptj_model_path"
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_phi2_model_path():
    model_name_or_path = phi2_name_or_path
    tiny_model_path = "./tmp/tiny_phi2_model_path"
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_deepseek_v2_model_path():
    model_name_or_path = deepseek_v2_name_or_path
    tiny_model_path = "./tmp/tiny_deepseek_v2_model_path"
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path, num_layers=2, trust_remote_code=False)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_gemma_model_path():
    model_name_or_path = gemma_name_or_path
    tiny_model_path = "./tmp/tiny_gemma_model_path"
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path, num_layers=2)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_qwen_model_path():
    model_name_or_path = qwen_name_or_path
    tiny_model_path = "./tmp/tiny_qwen_model_path"
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_fp8_qwen_model_path():
    from unittest.mock import patch

    with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
        model_name_or_path = get_model_path("Qwen/Qwen3-0.6B-FP8")
        tiny_model_path = "./tmp/tiny_fp8_qwen_model_path"
        tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_untied_qwen_model_path():
    model_name_or_path = qwen_name_or_path
    tiny_model_path = "./tmp/tiny_untied_qwen_model_path"
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path, force_untie=True)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_qwen_moe_model_path():
    model_name_or_path = qwen_moe_name_or_path
    tiny_model_path = "./tmp/tiny_qwen_moe_model_path"
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path, num_layers=2)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_qwen_vl_model_path():
    model_name_or_path = qwen_vl_name_or_path
    tiny_model_path = "./tmp/tiny_qwen_vl_model_path"
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path, num_layers=3, is_mllm=True)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_qwen_2_5_vl_model_path():
    model_name_or_path = qwen_2_5_vl_name_or_path
    tiny_model_path = "./tmp/tiny_qwen_2_5_vl_model_path"
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path, num_layers=2, is_mllm=True)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_fp8_qwen_moe_model_path():
    with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
        tiny_model_path = "./tmp/tiny_fp8_qwen_moe_model_path"
        model_name = get_model_path("Qwen/Qwen3-30B-A3B-FP8")
        config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.num_experts, config.num_hidden_layers, config.vocab_size = 4, 2, 2048
        model = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        from transformers.integrations.finegrained_fp8 import FP8Expert, FP8Linear

        for name, module in model.named_modules():
            if name == "lm_head":
                continue
            if "mlp.gate" in name:
                continue
            if isinstance(module, torch.nn.Linear):
                fp8_linear = FP8Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    block_size=[128, 128],
                )
                model.set_submodule(name, fp8_linear)
            if name.endswith("mlp.experts"):
                fp8_expert = FP8Expert(
                    config=model.config.get_text_config(),
                    block_size=[128, 128],
                )
                model.set_submodule(name, fp8_expert)

        model.save_pretrained(tiny_model_path)
        print(model)
        tokenizer.save_pretrained(tiny_model_path)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_gpt_oss_model_path():
    tiny_model_path = "./tmp/tiny_gpt_oss"
    from transformers import GptOssForCausalLM

    model_name = get_model_path("unsloth/gpt-oss-20b")
    config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.num_hidden_layers = 1  # Reduce layers for testing
    config.layer_types = config.layer_types[:1]  # Keep only the first layer type for testing
    delattr(config, "quantization_config")
    model = GptOssForCausalLM(config)
    model.save_pretrained(tiny_model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(tiny_model_path)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_llama4_model_path():
    tiny_model_path = "./tmp/tiny_llama4"
    from transformers import Llama4ForConditionalGeneration

    model_name = get_model_path("meta-llama/Llama-4-Scout-17B-16E-Instruct")
    config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    # TODO: Remove after https://github.com/huggingface/transformers/issues/43525 is resolved
    config.pad_token_id = None
    config.vision_config.num_hidden_layers = 1  # Reduce layers for testing
    config.text_config.num_hidden_layers = 1
    config.text_config.num_hidden_layers = 1
    model = Llama4ForConditionalGeneration(config)
    # Remove these parameters to avoid mismatch during quantized model loading
    model.config.text_config.no_rope_layers = []
    if hasattr(model.config.text_config, "moe_layers"):
        delattr(model.config.text_config, "moe_layers")
    if hasattr(model.config.text_config, "layer_types"):
        delattr(model.config.text_config, "layer_types")
    model.save_pretrained(tiny_model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(tiny_model_path)
    processor = transformers.AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    processor.save_pretrained(tiny_model_path)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_qwen3_vl_moe_model_path():
    tiny_model_path = "./tmp/tiny_qwen3_vl_moe"
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration

    model_name = get_model_path("Qwen/Qwen3-VL-30B-A3B-Instruct")
    config = transformers.AutoConfig.from_pretrained(model_name)
    config.vision_config.depth = 1  # Reduce layers for testing
    config.text_config.num_hidden_layers = 1
    config.text_config.num_experts = 16
    config.num_hidden_layers = 1
    model = Qwen3VLMoeForConditionalGeneration(config)
    model.save_pretrained(tiny_model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(tiny_model_path)
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    processor.save_pretrained(tiny_model_path)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_qwen35_moe_model_path():
    tiny_model_path = "./tmp/tiny_qwen35_moe"
    from transformers import Qwen3_5MoeForConditionalGeneration

    model_name = get_model_path("Qwen/Qwen3.5-35B-A3B")
    config = transformers.AutoConfig.from_pretrained(model_name)
    config.text_config.pad_token_id = None
    config.vision_config.depth = 1  # Reduce layers for testing
    config.text_config.num_hidden_layers = 4
    config.num_hidden_layers = 1
    config.text_config.layer_types = config.text_config.layer_types[: config.text_config.num_hidden_layers]
    config.text_config.use_cache = False
    model = Qwen3_5MoeForConditionalGeneration(config)
    model.save_pretrained(tiny_model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(tiny_model_path)
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    processor.save_pretrained(tiny_model_path)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_tiny_llama_model_path():
    tiny_model_path = "./tmp/tiny_TinyLlama"
    model_name = get_model_path("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    config = transformers.AutoConfig.from_pretrained(model_name)
    config.num_hidden_layers = 4
    model = transformers.AutoModelForCausalLM.from_config(config)
    model.save_pretrained(tiny_model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(tiny_model_path)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_qwen2_5_omni():
    """Tiny Qwen2.5-Omni-3B model built from real config with reduced layers.

    Uses random weights (no checkpoint loading) so it is fast for CPU unit
    tests while still exercising the real config structure.
    Skipped automatically when the model path does not exist locally.
    """

    from transformers import AutoConfig, AutoProcessor, AutoTokenizer, Qwen2_5OmniForConditionalGeneration

    model_name = qwen2_5_omni_name_or_path
    if not os.path.isdir(model_name):
        pytest.skip(f"Qwen2.5-Omni-3B not found at {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Reduce layers — keeps real config structure but uses random weights
    config.thinker_config.text_config.num_hidden_layers = 1
    config.thinker_config.vision_config.depth = 1
    config.thinker_config.audio_config.num_hidden_layers = 1
    config.talker_config.num_hidden_layers = 1
    if hasattr(config.thinker_config.text_config, "layer_types"):
        config.thinker_config.text_config.layer_types = config.thinker_config.text_config.layer_types[:1]
    if hasattr(config.talker_config, "layer_types"):
        config.talker_config.layer_types = config.talker_config.layer_types[:1]

    model = Qwen2_5OmniForConditionalGeneration(config)
    model.config.name_or_path = None
    yield model, tokenizer, processor


@pytest.fixture(scope="session")
def tiny_qwen3_omni_moe():
    """Tiny Qwen3-Omni-MoE model built from real config with reduced layers.

    Uses random weights (no checkpoint loading) so it is fast for CI while
    still exercising the real config structure.
    Skipped automatically when the model path does not exist locally.
    """

    from transformers import AutoConfig, AutoProcessor, AutoTokenizer, Qwen3OmniMoeForConditionalGeneration

    model_name = qwen3_omni_name_or_path
    if not os.path.isdir(model_name):
        pytest.skip(f"Qwen3-Omni-30B-A3B-Instruct not found at {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Reduce layers — keeps real config structure but uses random weights
    config.thinker_config.text_config.num_hidden_layers = 1
    config.thinker_config.vision_config.depth = 1
    config.thinker_config.audio_config.num_hidden_layers = 1
    if hasattr(config.thinker_config.text_config, "layer_types"):
        config.thinker_config.text_config.layer_types = config.thinker_config.text_config.layer_types[:1]
    # Talker
    if hasattr(config, "talker_config"):
        if hasattr(config.talker_config, "text_config"):
            config.talker_config.text_config.num_hidden_layers = 1
        elif hasattr(config.talker_config, "num_hidden_layers"):
            config.talker_config.num_hidden_layers = 1

    model = Qwen3OmniMoeForConditionalGeneration(config)
    model.config.name_or_path = None
    yield model, tokenizer, processor


# Mock torch.cuda.get_device_capability to always return (9, 0) like H100
@pytest.fixture()
def mock_fp8_capable_device():
    from unittest.mock import patch

    with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
        yield


@pytest.fixture(autouse=True, scope="session")
def clean_tmp_model_folder():
    yield
    shutil.rmtree("./tmp", ignore_errors=True)  # unittest default workspace
    shutil.rmtree("./tmp_autoround", ignore_errors=True)  # autoround default workspace


# Create objective fixtures for testing
@pytest.fixture(scope="function")
def tiny_opt_model():
    model_name_or_path = opt_name_or_path
    return get_tiny_model(model_name_or_path, num_layers=2)


@pytest.fixture(scope="function")
def opt_model():
    model_name_or_path = opt_name_or_path
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype="auto", trust_remote_code=True)
    return model


@pytest.fixture(scope="session")
def opt_tokenizer():
    model_name_or_path = opt_name_or_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    return tokenizer


@pytest.fixture(scope="function")
def model():
    model_name_or_path = opt_name_or_path
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype="auto", trust_remote_code=True)
    return model


@pytest.fixture(scope="session")
def tokenizer():
    model_name_or_path = opt_name_or_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    return tokenizer


@pytest.fixture(scope="session")
def dataloader():
    return DataLoader()
