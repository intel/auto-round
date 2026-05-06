import concurrent.futures
import copy
import os
import re
import shutil

import pytest
import torch
import transformers
from packaging import version

from auto_round.eval.evaluation import simple_evaluate, simple_evaluate_user_model
from auto_round.utils import detect_device, diffusion_load_model, get_attr, llm_load_model, mllm_load_model, set_attr

transformers_version = version.parse(transformers.__version__)


def _raise_threaded_packing(*args, **kwargs):
    raise AssertionError("Packing should not create a thread pool or call threadpoolctl.")


def forbid_threaded_packing(monkeypatch, module):
    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", _raise_threaded_packing)
    monkeypatch.setattr(module, "ThreadPoolExecutor", _raise_threaded_packing, raising=False)
    tctl = getattr(module, "tctl", None)
    if tctl is not None:
        monkeypatch.setattr(tctl, "threadpool_limits", _raise_threaded_packing)


def generate_prompt(model_obj_or_str, tokenizer=None, text="The capital of France is,", max_new_tokens=10, device=None):
    """Generate text using a model and tokenizer.

    Args:
        model_obj_or_str: The model to use for generation.
        tokenizer: The tokenizer for the model.
        text: The input prompt text.
        max_new_tokens: Maximum number of new tokens to generate.

    Returns:
        str: The generated text.
    """
    if device is None:
        device = detect_device()
    if isinstance(model_obj_or_str, str):
        model, tokenizer = llm_load_model(model_obj_or_str, trust_remote_code=True)
    else:
        model = model_obj_or_str
        assert tokenizer is not None, "Tokenizer must be provided when model is a model object"
    if not (hasattr(model, "hf_device_map") and len(model.hf_device_map) > 1):
        model = model.to(device)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)[0]
    output = tokenizer.decode(generated_ids)
    print(output)
    return output


def eval_generated_prompt(
    model, tokenizer=None, prompt_text="The United States of", target_text="America", max_new_tokens=10, device=None
):
    generated_text = generate_prompt(model, tokenizer, prompt_text, max_new_tokens=max_new_tokens, device=device)
    assert target_text in generated_text, f"Expected {target_text} in generated text: {generated_text}"


def evaluate_accuracy(
    model_or_save_dir, tokenizer=None, task="lambada_openai", threshold=0.25, batch_size="auto", limit=None, device=None
):
    """Helper function to evaluate model accuracy on a given task.

    Supports both saved model directory and in-memory model object.

    Args:
        model_or_save_dir: Either a path to the saved model directory (str) or a model object.
        tokenizer: The tokenizer for the model (required when model_or_save_dir is a model object).
        task: The evaluation task.
        threshold: The minimum accuracy threshold.
        batch_size: Batch size for evaluation.
        limit: Limit the number of samples to evaluate (only for model object).

    Returns:
        float: The accuracy value.

    Raises:
        AssertionError: If accuracy is below threshold.
    """
    if isinstance(model_or_save_dir, str):
        # save_dir mode
        model_args = f"pretrained={model_or_save_dir}"
        result = simple_evaluate(model="hf", model_args=model_args, tasks=task, batch_size=batch_size, device=device)
    else:
        # model object mode
        if tokenizer is None:
            raise ValueError("tokenizer is required when model_or_save_dir is a model object")
        if device and device != "cpu":
            model_or_save_dir = model_or_save_dir.to(device)
        result = simple_evaluate_user_model(
            model_or_save_dir, tokenizer, batch_size=batch_size, tasks=task, limit=limit, device=device
        )

    acc = result["results"][task]["acc,none"]
    print(f"{task} accuracy: {acc}")
    assert acc > threshold, f"Accuracy {acc} is below threshold {threshold}"
    return acc


# Automatic choose local path or model name.
def get_model_path(model_name: str) -> str:
    model_name = model_name.rstrip("/")
    ut_path = f"/tf_dataset/auto_round/models/{model_name}"
    local_path = f"/models/{model_name.split('/')[-1]}"
    local_path_1 = f"/dataset/{model_name.split('/')[-1]}"

    if os.path.exists(ut_path):
        return ut_path
    elif os.path.exists(local_path):
        return local_path
    elif os.path.exists(local_path_1):
        return local_path_1
    else:
        return model_name


opt_name_or_path = get_model_path("facebook/opt-125m")
qwen_name_or_path = get_model_path("Qwen/Qwen3-0.6B")
lamini_name_or_path = get_model_path("MBZUAI/LaMini-GPT-124M")
gptj_name_or_path = get_model_path("hf-internal-testing/tiny-random-GPTJForCausalLM")
phi2_name_or_path = get_model_path("microsoft/phi-2")
deepseek_v2_name_or_path = get_model_path("deepseek-ai/DeepSeek-V2-Lite")
qwen_moe_name_or_path = get_model_path("Qwen/Qwen1.5-MoE-A2.7B")
qwen_vl_name_or_path = get_model_path("Qwen/Qwen2-VL-2B-Instruct")
qwen_2_5_vl_name_or_path = get_model_path("Qwen/Qwen2.5-VL-3B-Instruct")
qwen_3_vl_9b_name_or_path = get_model_path("Qwen/Qwen3.5-9B")
gemma_name_or_path = get_model_path("benzart/gemma-2b-it-fine-tuning-for-code-test")
qwen2_5_omni_name_or_path = get_model_path("Qwen/Qwen2.5-Omni-3B")
qwen3_omni_name_or_path = get_model_path("Qwen/Qwen3-Omni-30B-A3B-Instruct")
flux_name_or_path = get_model_path("black-forest-labs/FLUX.1-dev")


# Slice model into tiny model for speedup
def _reduce_config_layers(config, num_layers, num_experts=None):
    """Reduce num_layers and num_experts in a config object (in-place).

    Handles nested sub-configs (text_config, vision_config, audio_config,
    thinker_config, talker_config, etc.) commonly found in multi-modal and
    MoE architectures.
    """
    n_block_keys = [
        "n_layers",
        "num_hidden_layers",
        "n_layer",
        "num_layers",
        "depth",
        "encoder_layers",
        "num_single_layers",
        "num_decoder_layers",
    ]
    n_expert_keys = ["num_experts", "num_local_experts"]

    def _apply(cfg):
        for key in n_block_keys:
            if isinstance(cfg, dict) and key in cfg:
                cfg[key] = num_layers
            elif hasattr(cfg, key):
                setattr(cfg, key, num_layers)

        if isinstance(cfg, dict) and "layer_types" in cfg:
            cfg["layer_types"] = cfg["layer_types"][:num_layers]
        elif hasattr(cfg, "layer_types"):
            cfg.layer_types = cfg.layer_types[:num_layers]

        if num_experts is not None:
            for key in n_expert_keys:
                if isinstance(cfg, dict) and key in cfg:
                    cfg[key] = num_experts
                elif hasattr(cfg, key):
                    setattr(cfg, key, num_experts)

    # Top-level config
    _apply(config)

    # Common sub-configs
    for sub_name in [
        "text_config",
        "vision_config",
        "audio_config",
        "thinker_config",
        "talker_config",
    ]:
        sub_cfg = getattr(config, sub_name, None)
        if sub_cfg is not None:
            _apply(sub_cfg)
            # Handle deeper nesting, e.g. thinker_config.text_config
            for inner_name in ["text_config", "vision_config", "audio_config"]:
                inner_cfg = getattr(sub_cfg, inner_name, None)
                if inner_cfg is not None:
                    _apply(inner_cfg)


def _apply_config_overrides(config, overrides):
    """Apply arbitrary key-value overrides to a config (dict or object)."""
    if not overrides:
        return
    for key, value in overrides.items():
        if isinstance(config, dict):
            if key in config:
                config[key] = value
        elif hasattr(config, key):
            setattr(config, key, value)


def get_tiny_model(
    model_name_or_path,
    num_layers=2,
    num_experts=None,
    is_mllm=False,
    from_config=False,
    is_diffusion=False,
    config_overrides=None,
    **kwargs,
):
    """Generate a tiny model by slicing layers from the original model.

    Args:
        model_name_or_path: HuggingFace model name or local path.
        num_layers: Number of layers to keep.
        num_experts: Number of experts to keep for MoE models (None = unchanged).
        is_mllm: Whether the model is a multi-modal model.
        from_config: If True, initialise the model from config with random
            weights instead of downloading the full checkpoint.  This is much
            faster and avoids large downloads.
        **kwargs: Extra keyword arguments forwarded to the model loader or
            ``AutoConfig.from_pretrained``.
    """
    if "use_config" in kwargs:
        from_config = kwargs.pop("use_config")

    model_name_or_path = get_model_path(model_name_or_path)

    if from_config:
        # ---- lightweight path: config-only, random weights ----
        if is_diffusion:
            import importlib
            import json

            from diffusers import AutoPipelineForText2Image
            from huggingface_hub import snapshot_download

            diffusers_module = importlib.import_module("diffusers")
            transformers_module = importlib.import_module("transformers")

            existing = False
            if os.path.exists(model_name_or_path):
                local_dir = model_name_or_path
                existing = True
            else:
                local_dir = snapshot_download(
                    repo_id=model_name_or_path, ignore_patterns=["*.safetensors", "*.safetensors.index.json"]
                )

            def _get_module(cls_name, mod_name, folder_name):
                if cls_name == "diffusers":
                    with open(os.path.join(local_dir, folder_name, "config.json"), "r", encoding="utf-8") as f:
                        config = json.load(f)
                    _reduce_config_layers(config, num_layers, num_experts)
                    _apply_config_overrides(config, config_overrides)
                    return getattr(getattr(diffusers_module, mod_name), "from_config")(config)
                else:
                    config = transformers.AutoConfig.from_pretrained(
                        os.path.join(local_dir, folder_name, "config.json")
                    )
                    _reduce_config_layers(config, num_layers, num_experts)
                    _apply_config_overrides(config, config_overrides)
                    return getattr(getattr(transformers_module, mod_name), "_from_config")(config)

            with open(os.path.join(local_dir, "model_index.json"), "r", encoding="utf-8") as f:
                model_index = json.load(f)

            if not existing:
                for k, v in model_index.items():
                    if k in ["scheduler", "tokenizer", "tokenizer_2"]:
                        continue
                    if isinstance(v, list) and v[0] in ["diffusers", "transformers"]:
                        module = _get_module(v[0], v[1], k)
                        module.save_pretrained(os.path.join(local_dir, k))
                model = AutoPipelineForText2Image.from_pretrained(local_dir)
            else:
                model = AutoPipelineForText2Image.from_pretrained(local_dir)
                for k, v in model_index.items():
                    if (
                        k not in ["scheduler", "tokenizer", "tokenizer_2"]
                        and isinstance(v, list)
                        and v[0] in ["diffusers", "transformers"]
                    ):
                        tiny_module = _get_module(v[0], v[1], k)
                        setattr(model, k, tiny_module)
            return model
        else:
            trust_remote_code = kwargs.get("trust_remote_code", True)
            config = transformers.AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
            # Special cases, for transformers == 5.4.0
            if config.model_type == "qwen3_omni_moe":
                config.initializer_range = 0.02  # Default initializer range for weight initialization
            _reduce_config_layers(config, num_layers, num_experts)

            # Pick the right model class
            base_lib = transformers
            architectures = getattr(config, "architectures", [None])[0]
            if (
                is_mllm
                and architectures.endswith("Model")
                and hasattr(base_lib, n := architectures.replace("Model", "ForConditionalGeneration"))
            ):
                model_cls = getattr(base_lib, n)
            elif hasattr(base_lib, architectures):
                model_cls = getattr(base_lib, architectures)
            else:
                model_cls = transformers.AutoModelForCausalLM  # default to causal LM if we can't find a better match
            model = model_cls._from_config(config)
            model = model.eval()
            return model

    # ---- original path: load pretrained weights then slice ----
    def slice_layers(module):
        """slice layers in the model."""
        sliced = False
        for name, child in module.named_children():
            if isinstance(child, torch.nn.ModuleList) and len(child) > num_layers:
                new_layers = torch.nn.ModuleList(child[:num_layers])
                setattr(module, name, new_layers)
                sliced = True
            elif slice_layers(child):
                sliced = True
        return sliced

    kwargs["dtype"] = "auto" if "auto" not in kwargs else kwargs["dtype"]
    kwargs["trust_remote_code"] = True if "trust_remote_code" not in kwargs else kwargs["trust_remote_code"]
    if is_mllm:
        model, processor, tokenizer, image_processor = mllm_load_model(model_name_or_path, **kwargs)
        if hasattr(model.config, "vision_config"):
            if hasattr(model.config.vision_config, "num_hidden_layers"):  # mistral, etc.
                model.config.num_hidden_layers = num_layers
            elif hasattr(model.config.vision_config, "depth"):  # qwen vl
                model.config.vision_config.depth = num_layers
    elif is_diffusion:
        pipe, model = diffusion_load_model(model_name_or_path, **kwargs)
        if (
            hasattr(model, "config")
            and hasattr(model.config, "num_layers")
            and hasattr(model.config, "num_single_layers")
        ):
            model.config.num_layers = num_layers
            model.config.num_single_layers = num_layers
    else:
        model, tokenizer = llm_load_model(model_name_or_path, **kwargs)

    slice_layers(model)

    _reduce_config_layers(model.config, num_layers, num_experts)

    return model


# for fixture usage only
def save_tiny_model(
    model_name_or_path,
    tiny_model_path,
    num_layers=2,
    num_experts=None,
    is_mllm=False,
    force_untie=False,
    from_config=False,
    is_diffusion=False,
    config_overrides=None,
    **kwargs,
):
    """Generate  a tiny model and save to the specified path.

    Args:
        model_name_or_path: HuggingFace model name or local path.
        tiny_model_path: Where to save the tiny model.
        num_layers: Number of layers to keep.
        num_experts: Number of experts to keep for MoE models (None = unchanged).
        is_mllm: Whether the model is a multi-modal model.
        force_untie: Force untie word embeddings.
        from_config: If True, initialise the model from config with random
            weights instead of downloading the full checkpoint.
        **kwargs: Extra keyword arguments forwarded to the model loader.
    """
    if "use_config" in kwargs:
        from_config = kwargs.pop("use_config")

    model = get_tiny_model(
        model_name_or_path,
        num_layers=num_layers,
        num_experts=num_experts,
        is_mllm=is_mllm,
        from_config=from_config,
        is_diffusion=is_diffusion,
        config_overrides=config_overrides,
        **kwargs,
    )
    if force_untie:
        if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
            model.config.tie_word_embeddings = False
            for key in model._tied_weights_keys:
                weight = get_attr(model, key)
                set_attr(model, key, copy.deepcopy(weight))
    test_path = os.path.dirname(__file__)
    tiny_model_path = os.path.join(test_path, tiny_model_path.removeprefix("./"))
    shutil.rmtree(tiny_model_path, ignore_errors=True)

    if not kwargs.get("trust_remote_code", True) and getattr(getattr(model, "config", None), "auto_map", None):
        del model.config.auto_map

    model.save_pretrained(tiny_model_path)
    if not is_diffusion:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        tokenizer.save_pretrained(tiny_model_path)
    # copy tokenizer.model
    model_path = model_name_or_path
    if not os.path.isdir(model_name_or_path):
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, HFValidationError

        from auto_round.utils import download_hf_model

        try:
            model_path = download_hf_model(model_name_or_path)
        except (GatedRepoError, HfHubHTTPError, HFValidationError):
            model_path = model_name_or_path
    if os.path.isfile(os.path.join(model_path, "tokenizer.model")):
        shutil.copy(os.path.join(model_path, "tokenizer.model"), os.path.join(tiny_model_path, "tokenizer.model"))
    if is_mllm:
        sources = [model_path]
        if model_name_or_path not in sources:
            sources.insert(0, model_name_or_path)

        for source in sources:
            try:
                processor = transformers.AutoProcessor.from_pretrained(source, **kwargs)
                processor.save_pretrained(tiny_model_path)
                break
            except (OSError, ValueError):
                continue

        for source in sources:
            try:
                image_processor = transformers.AutoImageProcessor.from_pretrained(source, **kwargs)
                image_processor.save_pretrained(tiny_model_path)
                break
            except (OSError, ValueError):
                continue
    print(f"[Fixture]: built tiny model path:{tiny_model_path} for testing in session")
    return tiny_model_path


# HPU mode checking
def is_pytest_mode_compile():
    return pytest.mode == "compile"


def is_pytest_mode_lazy():
    return pytest.mode == "lazy"


def check_version(lib):
    try:
        from transformers.utils.versions import require_version

        require_version(lib)
        return True
    except Exception:
        return False


# General model inference code
def model_infer(model, tokenizer, apply_chat_template=False):
    """Run model inference and print generated outputs."""
    prompts = [
        "Hello,my name is",
        # "The president of the United States is",
        # "The capital of France is",
        # "The future of AI is",
    ]
    if apply_chat_template:
        texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)
        prompts = texts

    inputs = tokenizer(prompts, return_tensors="pt", padding=False, truncation=True)

    outputs = model.generate(
        input_ids=inputs["input_ids"].to(model.device),
        attention_mask=inputs["attention_mask"].to(model.device),
        do_sample=False,  ## change this to follow official usage
        max_new_tokens=5,
    )
    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs["input_ids"], outputs)]

    decoded_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    for i, prompt in enumerate(prompts):
        print(f"Prompt: {prompt}")
        print(f"Generated: {decoded_outputs[i]}")
        print("-" * 50)
    return decoded_outputs[0]


# Dummy dataloader for testing
class DataLoader:

    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


fixed_input = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)


def get_output(model_name_or_path):
    """Get model output for fixed input."""
    try:
        model, tokenizer = llm_load_model(model_name_or_path)
    except:
        model, processor, tokenizer, image_processor = mllm_load_model(model_name_or_path)
    outputs = model(fixed_input)[0]
    return outputs.detach().cpu()


def is_cuda_support_fp8(major=9, minor=0):
    """Check if the current CUDA device capability is >= (major, minor).

    Args:
        major: Required major compute capability (default: 9).
        minor: Required minor compute capability (default: 0).

    Returns:
        bool: True if CUDA is available and device capability >= (major, minor).
    """
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap >= (major, minor)
