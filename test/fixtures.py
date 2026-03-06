import os
import shutil

import datasets
import pytest
import torch
import transformers

from .helpers import (
    DataLoader,
    deepseek_v2_name_or_path,
    gemma_name_or_path,
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
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path, num_layers=2, is_mllm=True)
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
