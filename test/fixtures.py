import os
import shutil
from unittest.mock import patch

import datasets
import pytest
import torch
import transformers

from auto_round.utils import is_transformers_version_greater_or_equal_5_4_0

from .helpers import (
    DataLoader,
    deepseek_v2_name_or_path,
    flux_name_or_path,
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
    tiny_model_path = save_tiny_model(
        model_name_or_path, tiny_model_path, num_layers=2, trust_remote_code=False, use_config=True
    )
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
def tiny_flux_model_path():
    model_name_or_path = flux_name_or_path
    tiny_model_path = "./tmp/tiny_flux_model_path"
    tiny_model_path = save_tiny_model(
        model_name_or_path,
        tiny_model_path,
        num_layers=1,
        is_diffusion=True,
        from_config=True,
        config_overrides={
            "num_attention_heads": 2,
            "attention_head_dim": 128,
            "joint_attention_dim": 256,
            "pooled_projection_dim": 256,
            "hidden_size": 256,
            "max_position_embeddings": 128,
            "intermediate_size": 256,
        },
    )
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_z_image_model_path():
    model_name_or_path = "Tongyi-MAI/Z-Image"
    tiny_model_path = "./tmp/tiny_z_image_model_path"
    tiny_model_path = save_tiny_model(
        model_name_or_path,
        tiny_model_path,
        num_layers=1,
        is_diffusion=True,
        from_config=True,
        config_overrides={
            "dim": 256,
            "n_heads": 2,
            "n_kv_heads": 2,
            "n_layers": 1,
            "n_refiner_layers": 1,
            "cap_feat_dim": 512,
            "in_channels": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "attention_head_dim": 128,
            "joint_attention_dim": 256,
            "pooled_projection_dim": 256,
            "hidden_size": 512,
            "intermediate_size": 256,
        },
    )
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

        from transformers.integrations.finegrained_fp8 import FP8Linear

        if is_transformers_version_greater_or_equal_5_4_0():
            from transformers.integrations.finegrained_fp8 import FP8Experts as FP8Expert
        else:
            from transformers.integrations.finegrained_fp8 import FP8Expert

        for name, module in model.named_modules():
            if name == "lm_head":
                continue
            if "mlp.gate" in name:
                continue
            if isinstance(module, torch.nn.Linear):
                fp8_linear = FP8Linear(
                    module.in_features,
                    module.out_features,
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
def tiny_qwen2_5_omni_model_path():
    """Tiny Qwen2.5-Omni-3B model built from real config with reduced layers.

    Uses random weights (no checkpoint loading) so it is fast for CPU unit
    tests while still exercising the real config structure.
    Skipped automatically when the model path does not exist locally.
    """
    from huggingface_hub import hf_hub_download

    model_name = qwen2_5_omni_name_or_path
    tiny_model_path = "./tmp/tiny_qwen2_5_omni_model_path"
    tiny_model_path = save_tiny_model(model_name, tiny_model_path, num_layers=1, is_mllm=True, from_config=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    processor = transformers.AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(tiny_model_path)
    processor.save_pretrained(tiny_model_path)
    # Copy model-specific files required for from_pretrained (e.g. spk_dict.pt for token2wav)
    file_path = hf_hub_download(repo_id="Qwen/Qwen2.5-Omni-3B", filename="spk_dict.pt", local_dir=tiny_model_path)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_qwen3_omni_moe_model_path():
    """Tiny Qwen3-Omni-MoE model built from real config with reduced layers.

    Uses random weights (no checkpoint loading) so it is fast for CI while
    still exercising the real config structure.
    Skipped automatically when the model path does not exist locally.
    """
    model_name = qwen3_omni_name_or_path
    tiny_model_path = "./tmp/tiny_qwen3_omni_moe_model_path"
    tiny_model_path = save_tiny_model(model_name, tiny_model_path, num_layers=1, is_mllm=True, from_config=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    processor = transformers.AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.save_pretrained(tiny_model_path)
    processor.save_pretrained(tiny_model_path)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


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


@pytest.fixture(scope="session")
def tiny_stable_audio_pipe():
    """Build a tiny StableAudioPipeline from scratch (random weights, no download).

    StableAudioPipeline is a text-to-audio pipeline not supported by AutoPipelineForText2Image,
    so we construct it manually from individual components rather than using save_tiny_model.
    Saves to a temp directory and reloads via from_pretrained so that
    ``name_or_path`` and ``model_index.json`` are set correctly.
    """
    from diffusers import AutoencoderOobleck, StableAudioDiTModel, StableAudioPipeline
    from diffusers.pipelines.stable_audio.modeling_stable_audio import StableAudioProjectionModel
    from diffusers.schedulers import EDMDPMSolverMultistepScheduler
    from transformers import AutoTokenizer, T5Config, T5EncoderModel

    tiny_model_path = "./tmp/tiny_stable_audio_pipe"

    transformer = StableAudioDiTModel(
        sample_size=64,
        in_channels=8,
        num_layers=1,
        attention_head_dim=32,
        num_attention_heads=2,
        num_key_value_attention_heads=2,
        out_channels=8,
        cross_attention_dim=64,
        time_proj_dim=32,
        global_states_input_dim=64,
        cross_attention_input_dim=64,
    )
    t5_config = T5Config(vocab_size=100, d_model=64, d_ff=128, num_heads=2, num_layers=1, d_kv=32)
    text_encoder = T5EncoderModel(t5_config)
    projection = StableAudioProjectionModel(text_encoder_dim=64, conditioning_dim=64, min_value=0.0, max_value=47.0)
    vae = AutoencoderOobleck(
        encoder_hidden_size=32,
        downsampling_ratios=[2, 4],
        channel_multiples=[1, 2],
        decoder_channels=16,
        decoder_input_channels=8,
        audio_channels=1,
        sampling_rate=16000,
    )
    scheduler = EDMDPMSolverMultistepScheduler()
    tokenizer = AutoTokenizer.from_pretrained(get_model_path("google-t5/t5-small"))
    pipe = StableAudioPipeline(
        vae=vae,
        text_encoder=text_encoder,
        projection_model=projection,
        tokenizer=tokenizer,
        transformer=transformer,
        scheduler=scheduler,
    )
    pipe.save_pretrained(tiny_model_path, is_diffusers=True)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_mimo_audio_model_path():
    """Build a tiny MiMo-Audio model by patching a Qwen backbone with MiMo-Audio config.

    Follows the pattern of omni models (is_mllm=True, from_config=True) but uses Qwen
    as the base since MiMo-Audio requires custom code not available in standard transformers.
    Patches config.architectures to ["MiMoAudioModel"] so that resolve_model_type returns 'mimo_audio'.
    """
    model_name_or_path = qwen_name_or_path
    tiny_model_path = "./tmp/tiny_mimo_audio_model_path"
    tiny_model_path = save_tiny_model(
        model_name_or_path,
        tiny_model_path,
        num_layers=2,
        is_mllm=True,
        from_config=True,
    )
    # Patch the config to simulate MiMo-Audio architecture
    config = transformers.AutoConfig.from_pretrained(tiny_model_path)
    config.architectures = ["MiMoAudioModel"]
    config.save_pretrained(tiny_model_path)
    yield tiny_model_path
    shutil.rmtree(tiny_model_path, ignore_errors=True)
