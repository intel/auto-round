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
    make_tiny_qwen3_omni_moe_config,
    opt_name_or_path,
    phi2_name_or_path,
    qwen2_5_omni_name_or_path,
    qwen_2_5_vl_name_or_path,
    qwen_moe_name_or_path,
    qwen_name_or_path,
    qwen_vl_name_or_path,
    save_tiny_model,
)

_save_tiny_model = save_tiny_model
TINY_MODEL_ROOT = os.path.join(os.path.dirname(__file__), "tmp", "tiny_models")
_source_model_ids = set()


def tiny_model_dir(name):
    return os.path.join(TINY_MODEL_ROOT, os.path.basename(os.path.normpath(name)))


def _tiny_model_ready(path):
    return os.path.isfile(os.path.join(path, ".autoround_ready"))


def _mark_tiny_model(path):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, ".autoround_ready"), "w", encoding="utf-8") as marker:
        marker.write("ready\n")


def _release_source_model_cache():
    """Release source checkpoints after the pytest session has finished."""
    if os.environ.get("AUTOROUND_REUSE_TINY_MODELS") != "1" or not _source_model_ids:
        return

    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        revisions = [
            revision.commit_hash
            for repo in cache_info.repos
            if repo.repo_type == "model" and repo.repo_id in _source_model_ids
            for revision in repo.revisions
        ]
        if revisions:
            cache_info.delete_revisions(*revisions).execute()
    except Exception:  # pragma: no cover - cache cleanup must not hide a test result
        pass


def save_tiny_model(*args, **kwargs):
    requested_path = args[1] if len(args) > 1 else kwargs["tiny_model_path"]
    tiny_model_path = tiny_model_dir(requested_path)
    if _tiny_model_ready(tiny_model_path):
        return tiny_model_path

    args = list(args)
    if len(args) > 1:
        args[1] = tiny_model_path
    else:
        kwargs = dict(kwargs)
        kwargs["tiny_model_path"] = tiny_model_path
    model_name_or_path = args[0] if args else kwargs["model_name_or_path"]
    result = _save_tiny_model(*args, **kwargs)
    _mark_tiny_model(result)
    if os.environ.get("AUTOROUND_REUSE_TINY_MODELS") == "1" and not os.path.isdir(model_name_or_path):
        _source_model_ids.add(model_name_or_path)
    return result


# Create tiny model path fixtures for testing
@pytest.fixture(scope="session")
def tiny_opt_model_path():
    model_name_or_path = opt_name_or_path
    tiny_model_path = tiny_model_dir("tiny_opt_model_path")
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path)
    yield tiny_model_path


@pytest.fixture(scope="session")
def tiny_lamini_model_path():
    model_name_or_path = lamini_name_or_path
    tiny_model_path = tiny_model_dir("tiny_lamini_model_path")
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path)
    yield tiny_model_path


@pytest.fixture(scope="session")
def tiny_gptj_model_path():
    model_name_or_path = gptj_name_or_path
    tiny_model_path = tiny_model_dir("tiny_gptj_model_path")
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path)
    yield tiny_model_path


@pytest.fixture(scope="session")
def tiny_phi2_model_path():
    model_name_or_path = phi2_name_or_path
    tiny_model_path = tiny_model_dir("tiny_phi2_model_path")
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path)
    yield tiny_model_path


@pytest.fixture(scope="session")
def tiny_deepseek_v2_model_path():
    model_name_or_path = deepseek_v2_name_or_path
    tiny_model_path = tiny_model_dir("tiny_deepseek_v2_model_path")
    tiny_model_path = save_tiny_model(
        model_name_or_path, tiny_model_path, num_layers=2, trust_remote_code=False, use_config=True
    )
    yield tiny_model_path


@pytest.fixture(scope="session")
def tiny_deepseek_v2_model_path_cpu():
    """Reduced fixture for CPU-only tests (2 MoE layers, 8 experts)."""
    model_name_or_path = deepseek_v2_name_or_path
    tiny_model_path = tiny_model_dir("tiny_deepseek_v2_model_path_cpu")
    tiny_model_path = save_tiny_model(
        model_name_or_path,
        tiny_model_path,
        num_layers=2,
        num_experts=8,
        trust_remote_code=False,
        use_config=True,
        config_overrides={"first_k_dense_replace": 0},
    )
    yield tiny_model_path


@pytest.fixture(scope="session")
def tiny_gemma_model_path():
    model_name_or_path = gemma_name_or_path
    tiny_model_path = tiny_model_dir("tiny_gemma_model_path")
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path, num_layers=2)
    yield tiny_model_path


@pytest.fixture(scope="session")
def tiny_qwen_model_path():
    model_name_or_path = qwen_name_or_path
    tiny_model_path = tiny_model_dir("tiny_qwen_model_path")
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path)
    yield tiny_model_path


@pytest.fixture(scope="session")
def tiny_fp8_qwen_model_path():
    from unittest.mock import patch

    with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
        model_name_or_path = get_model_path("Qwen/Qwen3-0.6B-FP8")
        tiny_model_path = tiny_model_dir("tiny_fp8_qwen_model_path")
        tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path)
    yield tiny_model_path


@pytest.fixture(scope="session")
def tiny_flux_model_path():
    model_name_or_path = flux_name_or_path
    tiny_model_path = tiny_model_dir("tiny_flux_model_path")
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


@pytest.fixture(scope="session")
def tiny_z_image_model_path():
    model_name_or_path = "Tongyi-MAI/Z-Image"
    tiny_model_path = tiny_model_dir("tiny_z_image_model_path")
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


@pytest.fixture(scope="session")
def tiny_untied_qwen_model_path():
    model_name_or_path = qwen_name_or_path
    tiny_model_path = tiny_model_dir("tiny_untied_qwen_model_path")
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path, force_untie=True)
    yield tiny_model_path


@pytest.fixture(scope="session")
def tiny_qwen_moe_model_path():
    model_name_or_path = qwen_moe_name_or_path
    tiny_model_path = tiny_model_dir("tiny_qwen_moe_model_path")
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path, num_layers=2)
    yield tiny_model_path


@pytest.fixture(scope="session")
def tiny_qwen_vl_model_path():
    model_name_or_path = qwen_vl_name_or_path
    tiny_model_path = tiny_model_dir("tiny_qwen_vl_model_path")
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path, num_layers=3, is_mllm=True)
    yield tiny_model_path


@pytest.fixture(scope="session")
def tiny_qwen_2_5_vl_model_path():
    model_name_or_path = qwen_2_5_vl_name_or_path
    tiny_model_path = tiny_model_dir("tiny_qwen_2_5_vl_model_path")
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path, num_layers=2, is_mllm=True)
    yield tiny_model_path


@pytest.fixture(scope="session")
def tiny_fp8_qwen_moe_model_path():
    with patch("torch.cuda.get_device_capability", return_value=(9, 0)):
        tiny_model_path = tiny_model_dir("tiny_fp8_qwen_moe_model_path")
        if _tiny_model_ready(tiny_model_path):
            yield tiny_model_path
            return
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
    _mark_tiny_model(tiny_model_path)
    yield tiny_model_path


@pytest.fixture(scope="session")
def tiny_gpt_oss_model_path():
    tiny_model_path = tiny_model_dir("tiny_gpt_oss")
    if _tiny_model_ready(tiny_model_path):
        yield tiny_model_path
        return
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
    _mark_tiny_model(tiny_model_path)
    yield tiny_model_path


@pytest.fixture(scope="session")
def tiny_llama4_model_path():
    tiny_model_path = tiny_model_dir("tiny_llama4")
    if _tiny_model_ready(tiny_model_path):
        yield tiny_model_path
        return
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
    _mark_tiny_model(tiny_model_path)
    yield tiny_model_path


@pytest.fixture(scope="session")
def tiny_qwen3_vl_moe_model_path():
    tiny_model_path = tiny_model_dir("tiny_qwen3_vl_moe")
    if _tiny_model_ready(tiny_model_path):
        yield tiny_model_path
        return
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
    _mark_tiny_model(tiny_model_path)
    yield tiny_model_path


@pytest.fixture(scope="session")
def tiny_qwen35_moe_model_path():
    tiny_model_path = tiny_model_dir("tiny_qwen35_moe")
    if _tiny_model_ready(tiny_model_path):
        yield tiny_model_path
        return
    from transformers import Qwen3_5MoeForConditionalGeneration

    model_name = get_model_path("Qwen/Qwen3.5-35B-A3B")
    config = transformers.AutoConfig.from_pretrained(model_name)
    config.text_config.pad_token_id = None
    config.vision_config.depth = 1  # Reduce layers for testing
    config.text_config.num_hidden_layers = 4
    config.num_hidden_layers = 1
    config.text_config.layer_types = config.text_config.layer_types[: config.text_config.num_hidden_layers]
    config.text_config.use_cache = False
    # This tiny model doesn't materialize the MTP block, so keep block_count aligned
    # with the actual number of exported layers to avoid a gguf tensor mismatch
    # (e.g. missing "blk.N.attn_norm.weight") when loading with llama.cpp.
    config.text_config.mtp_num_hidden_layers = 0
    model = Qwen3_5MoeForConditionalGeneration(config)
    model.save_pretrained(tiny_model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(tiny_model_path)
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    processor.save_pretrained(tiny_model_path)
    _mark_tiny_model(tiny_model_path)
    yield tiny_model_path


@pytest.fixture(scope="session")
def tiny_qwen35_moe_text_model_path(tiny_opt_model_path):
    """Small text-only Qwen3.5 MoE fixture for the PR CUDA smoke test."""
    tiny_model_path = tiny_model_dir("tiny_qwen35_moe_text")
    if _tiny_model_ready(tiny_model_path):
        yield tiny_model_path
        return

    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    tokenizer = transformers.AutoTokenizer.from_pretrained(tiny_opt_model_path)
    config = Qwen3_5MoeTextConfig(
        architectures=["Qwen3_5MoeForCausalLM"],
        vocab_size=len(tokenizer),
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        num_experts=2,
        num_experts_per_tok=2,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=64,
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = Qwen3_5MoeForCausalLM(config)
    model.save_pretrained(tiny_model_path)
    tokenizer.save_pretrained(tiny_model_path)
    _mark_tiny_model(tiny_model_path)
    yield tiny_model_path


@pytest.fixture(scope="session")
def tiny_tiny_llama_model_path():
    tiny_model_path = tiny_model_dir("tiny_TinyLlama")
    if _tiny_model_ready(tiny_model_path):
        yield tiny_model_path
        return
    model_name = get_model_path("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    config = transformers.AutoConfig.from_pretrained(model_name)
    config.num_hidden_layers = 4
    model = transformers.AutoModelForCausalLM.from_config(config)
    model.save_pretrained(tiny_model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(tiny_model_path)
    _mark_tiny_model(tiny_model_path)
    yield tiny_model_path


@pytest.fixture(scope="session")
def tiny_qwen2_5_omni_model_path():
    """Tiny Qwen2.5-Omni-3B model built from real config with reduced layers.

    Uses random weights (no checkpoint loading) so it is fast for CPU unit
    tests while still exercising the real config structure.
    Skipped automatically when the model path does not exist locally.
    """
    model_name_or_path = get_model_path(qwen2_5_omni_name_or_path)
    if not os.path.isdir(model_name_or_path):
        pytest.skip("Qwen2.5-Omni fixture is not available locally")
    tiny_model_path = tiny_model_dir("tiny_qwen2_5_omni_model_path")
    if _tiny_model_ready(tiny_model_path):
        yield tiny_model_path
        return
    tiny_model_path = save_tiny_model(model_name_or_path, tiny_model_path, num_layers=1, is_mllm=True, from_config=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    processor = transformers.AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.save_pretrained(tiny_model_path)
    processor.save_pretrained(tiny_model_path)
    # Copy model-specific files required for from_pretrained (e.g. spk_dict.pt for token2wav)
    local_spk_dict = os.path.join(model_name_or_path, "spk_dict.pt")
    if os.path.exists(local_spk_dict):
        shutil.copy(local_spk_dict, tiny_model_path)
    else:
        pytest.skip("Qwen2.5-Omni spk_dict.pt is not available locally")
    yield tiny_model_path


@pytest.fixture(scope="session")
def tiny_qwen3_omni_moe_model_path():
    """Self-contained tiny Qwen3-Omni-MoE model for CUDA smoke coverage."""
    tiny_model_path = tiny_model_dir("tiny_qwen3_omni_moe_smoke_v2_model_path")
    if not _tiny_model_ready(tiny_model_path):
        model = transformers.Qwen3OmniMoeForConditionalGeneration(make_tiny_qwen3_omni_moe_config())
        model.save_pretrained(tiny_model_path)
        _mark_tiny_model(tiny_model_path)
    yield tiny_model_path


# Mock FP8 capability checks without letting the fake capability affect Inductor code generation.
@pytest.fixture()
def mock_fp8_capable_device():
    from unittest.mock import patch

    with patch("torch.cuda.get_device_capability", return_value=(9, 0)), patch(
        "torch.compile", side_effect=lambda function, *args, **kwargs: function
    ):
        yield


@pytest.fixture(autouse=True, scope="session")
def clean_tmp_model_folder():
    yield
    _release_source_model_cache()
    tmp_root = os.path.join(os.path.dirname(__file__), "tmp")
    tiny_model_cache = os.path.abspath(TINY_MODEL_ROOT)
    for entry in os.scandir(tmp_root) if os.path.isdir(tmp_root) else []:
        if os.path.abspath(entry.path) == tiny_model_cache:
            continue
        if entry.is_dir():
            shutil.rmtree(entry.path, ignore_errors=True)
        else:
            try:
                os.unlink(entry.path)
            except FileNotFoundError:
                pass
    shutil.rmtree("./ar_work_space", ignore_errors=True)
    shutil.rmtree("./tmp_autoround", ignore_errors=True)
    shutil.rmtree(os.path.expanduser("~/.cache/auto_round"), ignore_errors=True)


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

    tiny_model_path = tiny_model_dir("tiny_stable_audio_pipe")
    if _tiny_model_ready(tiny_model_path):
        yield tiny_model_path
        return

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
    _mark_tiny_model(tiny_model_path)
    yield tiny_model_path


@pytest.fixture(scope="session")
def tiny_mimo_audio_model_path():
    """Build a tiny MiMo-Audio model by patching a Qwen backbone with MiMo-Audio config.

    Follows the pattern of omni models (is_mllm=True, from_config=True) but uses Qwen
    as the base since MiMo-Audio requires custom code not available in standard transformers.
    Patches config.architectures to ["MiMoAudioModel"] so that resolve_model_type returns 'mimo_audio'.
    """
    model_name_or_path = qwen_name_or_path
    tiny_model_path = tiny_model_dir("tiny_mimo_audio_model_path")
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
