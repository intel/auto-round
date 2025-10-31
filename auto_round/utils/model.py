# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
import json
import os
import re
from collections import UserDict
from dataclasses import asdict, fields
from pathlib import Path
from typing import Union

import torch
import transformers

from auto_round.export.export_to_gguf.config import ModelType
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme


def convert_dtype_str2torch(str_dtype):
    """Converts a string dtype to its corresponding PyTorch dtype.

    Args:
        str_dtype (str): The string representation of the dtype.

    Returns:
        torch.dtype: The PyTorch dtype.

    Raises:
        ValueError: If the input str_dtype is unsupported.
    """
    if isinstance(str_dtype, torch.dtype) or str_dtype is None:
        return str_dtype
    if str_dtype == "int8":
        return torch.int8
    elif str_dtype == "fp32" or str_dtype == "float32" or str_dtype == "auto":
        return torch.float
    elif str_dtype == "fp16" or str_dtype == "float16":
        return torch.float16
    elif str_dtype == "bf16" or str_dtype == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported string dtype '{str_dtype}' for conversion to torch dtype.")


def convert_dtype_torch2str(dtype):
    """Converts a PyTorch dtype to its corresponding string representation.

    Args:
        dtype: PyTorch dtype or str. The dtype to convert.

    Returns:
        str: The string representation of the dtype.

    Raises:
        ValueError: If the input dtype is unsupported.
    """
    if isinstance(dtype, str) or dtype is None:
        return dtype
    if dtype == torch.int8:
        return "int8"
    elif dtype == torch.float:
        return "fp32"
    elif dtype == torch.float16:
        return "fp16"
    elif dtype == torch.bfloat16:
        return "bf16"
    elif isinstance(dtype, str) and dtype in ["int8", "fp32", "fp16", "bf16"]:
        return dtype
    else:
        raise ValueError(f"Unsupported PyTorch dtype '{dtype}' for conversion to string dtype.")


def convert_dtype_torch2str_hf(dtype):
    """Converts a PyTorch dtype to its corresponding huggingface string dtype, e.g. torch.float32 -> 'float32'.

    Args:
        dtype: PyTorch dtype or str. The dtype to convert.

    Returns:
         str: The string representation of the dtype.

    Raises:
        ValueError: If the input str_dtype is unsupported.
    """
    if dtype is None:
        return dtype
    if isinstance(dtype, str):
        if "float" not in dtype and "int" not in dtype:
            dtype = convert_dtype_str2torch(dtype)
        else:
            return dtype
    str_dtype = str(dtype)
    if "." not in str_dtype:
        raise ValueError(f"Unsupported pytorch dtype '{dtype}' for conversion to huggingface str dtype")
    str_dtype = str_dtype.split(".")[1]
    return str_dtype


def check_and_mark_fp8_model(model: torch.nn.Module) -> bool:
    if is_fp8_model(model):
        return True
    for n, m in model.named_modules():
        if is_fp8_linear(m):
            m.is_fp8_linear = True
            if not hasattr(model, "is_fp8"):
                model.is_fp8 = True
    if hasattr(model, "is_fp8"):
        return True
    return False


def check_diffusers_installed():  # pragma: no cover
    try:
        import diffusers  # noqa: F401

        return True
    except ImportError:
        logger.error("Please install diffusers via 'pip install diffusers'" " to run diffusion model")
        exit(-1)


def check_start_with_block_name(name: str, block_name_to_quantize: list):
    """
    Checks if the given layer name starts with any of the block names to be quantized.

    Args:
        name (str): The name of the layer.
        block_name_to_quantize (list): A list of block names to check against.

    Returns:
        bool: True if the layer name starts with any of the block names, False otherwise.
    """
    for block_name in block_name_to_quantize:
        if name.startswith(block_name):
            return True
    return False


def download_hf_model(repo_id, cache_dir=None, repo_type=None, revision=None):
    """Download hugging face model from hf hub."""
    from huggingface_hub.constants import DEFAULT_REVISION, HUGGINGFACE_HUB_CACHE
    from huggingface_hub.file_download import REGEX_COMMIT_HASH, repo_folder_name

    if cache_dir is None:
        cache_dir = HUGGINGFACE_HUB_CACHE
    if revision is None:
        revision = DEFAULT_REVISION
    if repo_type is None:
        repo_type = "model"
    storage_folder = os.path.join(cache_dir, repo_folder_name(repo_id=repo_id, repo_type=repo_type))
    commit_hash = None
    if REGEX_COMMIT_HASH.match(revision):
        commit_hash = revision
    else:
        ref_path = os.path.join(storage_folder, "refs", revision)
        if os.path.exists(ref_path):
            with open(ref_path) as f:
                commit_hash = f.read()
    if storage_folder and commit_hash:
        pointer_path = os.path.join(storage_folder, "snapshots", commit_hash)
        if os.path.isdir(pointer_path):
            return pointer_path
    else:  # pragma: no cover
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(repo_id)
        return model_path


def llm_load_model(
    pretrained_model_name_or_path,
    trust_remote_code=True,
    model_dtype=None,
    device="cpu",
    **kwargs,
):
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    from auto_round.utils.device import (
        _use_hpu_compile_mode,
        get_device_and_parallelism,
        set_fake_cuda_device_capability,
    )

    device_str, use_auto_mapping = get_device_and_parallelism(device)
    torch_dtype = "auto"
    if device_str is not None and "hpu" in device_str:
        torch_dtype = torch.bfloat16

    is_glm = bool(re.search("chatglm", pretrained_model_name_or_path.lower()))

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust_remote_code)

    model_cls = AutoModel if is_glm else AutoModelForCausalLM
    if "deepseek" in pretrained_model_name_or_path.lower() and trust_remote_code:
        logger.warning("trust_remote_code is enabled by default, please ensure its correctness.")

    if _use_hpu_compile_mode():
        model = model_cls.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
            trust_remote_code=trust_remote_code,
            device_map="auto" if use_auto_mapping else None,
        )
    else:
        try:
            model = model_cls.from_pretrained(
                pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                device_map="auto" if use_auto_mapping else None,
            )
        except ValueError as e:
            if "FP8 quantized" in str(e):
                orig_func = set_fake_cuda_device_capability()
                model = model_cls.from_pretrained(
                    pretrained_model_name_or_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code,
                    device_map="auto" if use_auto_mapping else None,
                )
                torch.cuda.get_device_capability = orig_func
                logger.warning("the support for fp8 model as input is experimental, please use with caution.")
            else:
                raise

        except OSError as e:
            logger.warning(f"fail to load {pretrained_model_name_or_path}, set trust_remote_code to False and retry.")
            model = model_cls.from_pretrained(
                pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                trust_remote_code=False,
                device_map="auto" if use_auto_mapping else None,
            )

    model = model.eval()
    check_and_mark_fp8_model(model)
    model = _to_model_dtype(model, model_dtype)

    return model, tokenizer


def mllm_load_model(
    pretrained_model_name_or_path,
    device="cpu",
    torch_dtype="auto",
    use_auto_mapping=True,
    trust_remote_code=True,
    model_dtype=None,
    **kwargs,
):
    import transformers
    from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    from auto_round.utils.device import get_device_and_parallelism, set_fake_cuda_device_capability

    device_str, use_auto_mapping = get_device_and_parallelism(device)
    torch_dtype = "auto"
    if device_str is not None and "hpu" in device_str:
        torch_dtype = torch.bfloat16
    if os.path.isdir(pretrained_model_name_or_path):
        config = json.load(open(os.path.join(pretrained_model_name_or_path, "config.json")))
    else:
        from huggingface_hub import hf_hub_download, list_repo_files

        file_list = list_repo_files(pretrained_model_name_or_path)
        if "config.json" in file_list:
            # Load plain JSON
            config_path = hf_hub_download(pretrained_model_name_or_path, "config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        elif "config.json.gz" in file_list:
            # Load gzipped JSON
            import gzip

            config_path = hf_hub_download(pretrained_model_name_or_path, "config.json.gz")
            with gzip.open(config_path, "rt", encoding="utf-8") as f:
                config = json.load(f)
        else:
            raise FileNotFoundError(f"No config.json or config.json.gz found for {pretrained_model_name_or_path}")

    if "model_type" in config:
        model_type = config["model_type"]
    else:
        model_type = None

    processor, image_processor = None, None
    if "deepseek_vl_v2" == model_type:
        from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor  # pylint: disable=E0401

        processor = DeepseekVLV2Processor.from_pretrained(pretrained_model_name_or_path)
        tokenizer = processor.tokenizer
        model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map="auto" if use_auto_mapping else None,
        )
    else:
        architectures = config["architectures"][0]
        if architectures == "LlavaLlamaForCausalLM":
            from llava.model.builder import load_pretrained_model  # pylint: disable=E0401

            tokenizer, model, image_processor, _ = load_pretrained_model(
                pretrained_model_name_or_path,
                model_base=None,
                model_name=pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
            )
        else:
            if architectures.endswith("Model") and hasattr(
                transformers, n := architectures.replace("Model", "ForConditionalGeneration")
            ):
                cls = getattr(transformers, n)
            elif hasattr(transformers, architectures):
                cls = getattr(transformers, architectures)
            else:
                cls = AutoModelForCausalLM
            try:
                model = cls.from_pretrained(
                    pretrained_model_name_or_path,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=torch_dtype,
                    device_map="auto" if use_auto_mapping else None,
                )
            except ValueError as e:
                if "FP8 quantized" in str(e):
                    orig_func = set_fake_cuda_device_capability()
                    model = cls.from_pretrained(
                        pretrained_model_name_or_path,
                        trust_remote_code=trust_remote_code,
                        torch_dtype=torch_dtype,
                        device_map="auto" if use_auto_mapping else None,
                    )
                    torch.cuda.get_device_capability = orig_func
                    logger.warning("the support for fp8 model as input is experimental, please use with caution.")
                else:
                    raise

            if "Mistral-Small-3.2" in pretrained_model_name_or_path:
                from mistral_common.tokens.tokenizers.mistral import MistralTokenizer  # pylint: disable=E0401

                if os.path.isdir(pretrained_model_name_or_path):
                    tokenizer = MistralTokenizer.from_file(os.path.join(pretrained_model_name_or_path, "tekken.json"))
                else:
                    tokenizer = MistralTokenizer.from_hf_hub(pretrained_model_name_or_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code
                )
                processor = AutoProcessor.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code
                )
            try:
                from transformers import AutoImageProcessor

                image_processor = AutoImageProcessor.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code
                )
            except Exception as e:
                pass

    model = model.eval()
    check_and_mark_fp8_model(model)
    model = _to_model_dtype(model, model_dtype)

    return model, processor, tokenizer, image_processor


def diffusion_load_model(
    pretrained_model_name_or_path: str,
    device: Union[str, torch.device] = "cpu",
    torch_dtype: Union[str, torch.dtype] = "auto",
    use_auto_mapping: bool = False,
    trust_remote_code: bool = True,
    model_dtype: str = None,
    **kwargs,
):
    from auto_round.utils.common import LazyImport
    from auto_round.utils.device import get_device_and_parallelism

    device_str, use_auto_mapping = get_device_and_parallelism(device)
    torch_dtype = "auto"
    if device_str is not None and "hpu" in device_str:
        torch_dtype = torch.bfloat16

    pipelines = LazyImport("diffusers.pipelines")

    pipe = pipelines.auto_pipeline.AutoPipelineForText2Image.from_pretrained(
        pretrained_model_name_or_path, torch_dtype=torch_dtype
    )
    pipe = _to_model_dtype(pipe, model_dtype)
    model = pipe.transformer
    return pipe, model.to(device)


def is_pure_text_model(model):
    """verify on: phi-3.5, Mistral-Small-3.1, gemma-3, qwen2-vl,"""
    if hasattr(model, "config") and hasattr(model.config, "vision_config"):
        return False
    if hasattr(model.__class__, "main_input_name") and model.__class__.main_input_name != "input_ids":
        return False
    for module in model.modules():
        if hasattr(module.__class__, "main_input_name") and module.__class__.main_input_name != "input_ids":
            return False
        if "vision" in str(module.__class__).lower():
            return False
        if "image" in str(module.__class__).lower():
            return False
        if "img" in str(module.__class__).lower():
            return False
    return True


def is_mllm_model(model_or_path: Union[str, torch.nn.Module]):
    MM_KEYS = [
        "multi_modal_projector",
        "vision_tower",
        "multimodal_projector",
        "thinker",
        "visual",
        "audio",
        "talker",
        "token2wav",
        "vision_model",
        "audio_tower",
        "vision_encoder",
        "vision_language_adapter",
        "patch_merger",
        "pre_mm_projector_norm",
        "vision",
    ]

    model_path = model_or_path if isinstance(model_or_path, str) else model_or_path.name_or_path
    if not os.path.isdir(model_path):
        model_path = download_hf_model(model_path)

    if isinstance(model_path, str):
        if os.path.exists(os.path.join(model_path, "preprocessor_config.json")):
            return True
        if os.path.exists(os.path.join(model_path, "processor_config.json")):
            return True
        if os.path.exists(os.path.join(model_path, "config.json")):
            with open(os.path.join(model_path, "config.json")) as f:
                config = json.load(f)
            for key in config.keys():
                if any([k in key for k in MM_KEYS]):
                    return True

    if isinstance(model_or_path, torch.nn.Module):
        for name, module in model_or_path.named_modules():
            if any([k in name for k in MM_KEYS]):
                return True

    return False


def is_diffusion_model(model_or_path: Union[str, object]) -> bool:
    from auto_round.utils.common import LazyImport

    if isinstance(model_or_path, str):
        index_file = None
        if not os.path.isdir(model_or_path):
            try:
                from huggingface_hub import hf_hub_download

                index_file = hf_hub_download(model_or_path, "model_index.json")
                check_diffusers_installed()
            except Exception as e:
                print(e)
                index_file = None

        elif os.path.exists(os.path.join(model_or_path, "model_index.json")):
            check_diffusers_installed()
            index_file = os.path.join(model_or_path, "model_index.json")
        return index_file is not None
    elif not isinstance(model_or_path, torch.nn.Module):
        check_diffusers_installed()
        pipeline_utils = LazyImport("diffusers.pipelines.pipeline_utils")
        return isinstance(model_or_path, pipeline_utils.DiffusionPipeline)
    else:
        return False


def is_moe(module: torch.nn.Module) -> bool:
    """Returns whether the module is an MOE layer."""
    return any(
        key in type(module).__name__.lower()
        for key in [
            "MixtralSparseMoeBlock".lower(),
            "ArcticMoE".lower(),
            "DbrxFFN".lower(),
            "MoELayer".lower(),
            "PhimoeSparseMoeBlock".lower(),
            "DeepseekMoE".lower(),
            "DeepseekV2MoE".lower(),
            "DeepseekV3MoE".lower(),
            "Qwen2MoeSparseMoeBlock".lower(),
            "Qwen3MoeSparseMoeBlock".lower(),
        ]
    )


def is_fp8_model(model: torch.nn.Module) -> bool:
    if not hasattr(model, "is_fp8"):
        return False
    else:
        return model.is_fp8


def is_fp8_linear(module: torch.nn.Module) -> bool:
    if hasattr(module, "is_fp8_linear"):
        return module.is_fp8_linear
    if not (type(module) == torch.nn.Linear or module.__class__.__name__ == "FP8Linear"):
        return False
    if module.weight is None:
        return False
    if str(module.weight.dtype).startswith("torch.float8"):
        return True
    else:
        return False


def get_block_names(model, quant_vision=False):
    """Get the block names for transformers-like networks.

    Args:
    model: The model.

    Returns:
    block_names: A list whose elements are list of block's layer names
    """
    from auto_round.special_model_handler import SPECIAL_MULTIMODAL_BLOCK

    def _search_block(name, module):
        if hasattr(type(module), "__name__") and "ModuleList" in type(module).__name__:
            return [(name, module)]
        target_modules = []
        for n, m in module.named_children():
            if hasattr(type(m), "__name__") and "ModuleList" in type(m).__name__:
                target_modules.append((".".join(filter(None, (name, n))), m))
            else:
                target_modules.extend(_search_block(".".join(filter(None, (name, n))), m))
        return target_modules

    def _get_llm_block_names(model):
        block_names = []
        target_modules = _search_block("", model)

        for i, target_m in enumerate(target_modules):
            block_names.append([])
            for n, m in target_m[1].named_children():
                block_names[i].append(target_m[0] + "." + n)
        return block_names

    def _get_vlm_block_names(model, quant_vision=False):
        if (
            hasattr(model, "config")
            and hasattr(model.config, "model_type")
            and model.config.model_type in SPECIAL_MULTIMODAL_BLOCK.keys()
        ):
            return SPECIAL_MULTIMODAL_BLOCK.get(model.config.model_type)(model, quant_vision=quant_vision)
        block_names = []
        target_modules = []
        vision_blocks_tuple = ("vision", "visual", "image", "img")
        last_block_name = ""
        for n, m in model.named_modules():
            if hasattr(type(m), "__name__") and "ModuleList" in type(m).__name__:
                if quant_vision or all(key not in n.lower() for key in (vision_blocks_tuple)):
                    if last_block_name and last_block_name in n:
                        continue
                    target_modules.append((n, m))
                    last_block_name = n
        for i, target_m in enumerate(target_modules):
            block_names.append([])
            for n, m in target_m[1].named_children():
                block_names[i].append(target_m[0] + "." + n)
        return block_names

    if quant_vision or not is_pure_text_model(model):
        return _get_vlm_block_names(model, quant_vision=quant_vision)
    else:
        return _get_llm_block_names(model)


def get_lm_head_name(model):
    block_names = get_block_names(model, True)
    last_name = None
    for n, m in model.named_modules():
        if any(m.children()):
            continue
        last_name = n
    for l in block_names:
        if last_name in l:
            last_name = None
            break
    return last_name


# please refer to https://github.com/NVIDIA/TensorRT-Model-Optimizer
# /blob/4c611e47a60084a86e1de7e48690a692a1b8170c/modelopt/torch/export/layer_utils.py#L976
def get_expert_linear_names(module: torch.nn.Module) -> list[str]:
    """Get the list of linear names for the experts."""

    def module_match_name_list(module, name_list):
        """Check if the module name matches any of the names in the list.

        e.g. module_match_name_list(QuantQwen3MoeSparseMoeBlock, ['Qwen3MoeSparseMoeBlock']) -> True

        """
        return any(name.lower() in type(module).__name__.lower() for name in name_list)

    if module_match_name_list(
        module, ["Qwen2MoeSparseMoeBlock", "Qwen3MoeSparseMoeBlock", "DeepseekMoE", "DeepseekV2MoE", "DeepseekV3MoE"]
    ):
        return ["gate_proj", "down_proj", "up_proj"]
    elif module_match_name_list(module, ["MixtralMoeSparseMoeBlock"]):
        return ["linear_fc1", "linear_fc2"]
    elif module_match_name_list(module, ["DBRXMoeSparseMoeBlock"]):
        return ["w1_linear", "w2_linear", "v1_linear"]
    else:
        # assuming w1, w2, w3 by default
        return ["w1", "w2", "w3"]


def get_model_dtype(model_dtype, default="auto"):
    if model_dtype is None or model_dtype == "auto":
        model_dtype = default
    elif model_dtype in ["bf16", "bfloat16"]:
        model_dtype = "bfloat16"
    elif model_dtype in ["f16", "float16", "fp16"]:
        model_dtype = "float16"
    elif model_dtype in ["f32", "float32", "fp32"]:
        model_dtype = "float32"
    else:
        logger.warning(f"Unable to identify model_dtype {model_dtype}, reset to default model_dtype {default}")
        model_dtype = default
    return model_dtype


def get_nested_attr(module, attr_name: str):
    """Recursively get nested attribute (e.g., 'orig_layer.act_max')."""
    attrs = attr_name.split(".")
    for attr in attrs:
        if not hasattr(module, attr):
            return None
        module = getattr(module, attr)
    return module


def get_gguf_architecture(dir_model, model_type=ModelType.TEXT):
    from auto_round.export.export_to_gguf.convert_hf_to_gguf import (
        ModelBase,
        get_model_architecture,
    )

    is_mistral_format = False
    if isinstance(dir_model, str):
        dir_model = Path(dir_model)

    hparams = ModelBase.load_hparams(dir_model, is_mistral_format)
    if isinstance(hparams, dict):
        tmp_model_type = hparams["model_type"]
    else:
        tmp_model_type = hparams.model_type
    if "mistral" == tmp_model_type:
        is_mistral_format = True
        hparams = ModelBase.load_hparams(dir_model, is_mistral_format)
    if not is_mistral_format:
        model_class = get_model_architecture(hparams, model_type)
    elif model_type == ModelType.MMPROJ:
        assert hparams.get("vision_encoder") is not None, "This model does not support multimodal"
        model_class = "PixtralModel"
    else:
        model_class = "MistralModel"
    return model_class


def get_layer_names_in_block(
    model: torch.nn.Module,
    supported_types=(torch.nn.Linear, transformers.pytorch_utils.Conv1D),
    quant_block_list: list = None,
    class_names: tuple = None,
) -> list[str]:
    """Retrieves the names of layers within each block of the model.

    Returns:
        list: A list of strings, where each string is the name of a layer
              within a block of the model.
    """
    if class_names is None:
        class_names = []
    for n, m in model.named_modules():
        if type(m) in supported_types or (class_names is not None and m.__class__.__name__ in class_names):
            m.bk_tmp_name = n
    layers_in_block = []
    if bool(quant_block_list):
        all_blocks = quant_block_list
    else:
        all_blocks = get_block_names(model)
    for block_names in all_blocks:
        for block_name in block_names:
            block = get_module(model, block_name)
            for n, m in block.named_modules():
                if hasattr(m, "bk_tmp_name"):
                    layers_in_block.append(m.bk_tmp_name)
                    delattr(m, "bk_tmp_name")
    return layers_in_block


def set_nested_attr(module, attr_name: str, value):
    """Recursively set nested attribute (e.g., 'orig_layer.act_max' = value)."""
    attrs = attr_name.split(".")
    for attr in attrs[:-1]:
        if not hasattr(module, attr):
            return None  # No need to set act_max for fp layers
        module = getattr(module, attr)
    setattr(module, attrs[-1], value)


def _pad_weight(weight: torch.Tensor, block_size: list) -> tuple[torch.Tensor, int, int]:
    """Pads a matrix to make its dimensions multiples of block_size."""
    M, N = weight.shape[-2:]
    block_size_m, block_size_n = block_size
    pad_M = (block_size_m - M % block_size_m) % block_size_m
    pad_N = (block_size_n - N % block_size_n) % block_size_n

    if pad_M == 0 and pad_N == 0:
        return weight, M, N  # No padding needed
    padded_weight = torch.nn.functional.pad(weight, (0, pad_N, 0, pad_M), mode="constant", value=0)
    return padded_weight, M, N  # Return original dimensions for unpadding


def _unpad_weight(weight: torch.Tensor, original_M: int, original_N: int, keep_first_dim: bool = False) -> torch.Tensor:
    """Removes padding from the matrix to restore its original shape."""
    if (weight.shape[-2] == original_M) and (weight.shape[-1] == original_N):
        return weight
    if keep_first_dim:
        return weight[:, :original_M, :original_N]
    else:
        return weight[:original_M, :original_N]


def pad_block_fp8_weight_naive(
    weight: torch.Tensor, weight_scale: torch.Tensor, block_size: list
) -> tuple[torch.Tensor, int, int]:
    assert len(block_size) == 2

    block_size_m, block_size_n = block_size
    weight_scale_m, weight_scale_n = weight_scale.shape[-2:]

    weight, orig_M, orig_N = _pad_weight(weight, block_size)
    M, N = weight.shape[-2:]

    assert weight_scale_m == M // block_size_m
    assert weight_scale_n == N // block_size_n

    return weight, orig_M, orig_N


def dequant_block_fp8_weight(weight: torch.Tensor, weight_scale: torch.Tensor, block_size: list) -> torch.Tensor:
    dtype = torch.bfloat16
    if weight_scale is None:
        return weight
    assert len(block_size) == 2

    weight, orig_M, orig_N = pad_block_fp8_weight_naive(weight, weight_scale, block_size)

    weight_shape_len = len(weight.shape)

    block_size_m, block_size_n = block_size

    # mul scale
    if weight_shape_len == 2:
        weight_scale_m, weight_scale_n = weight_scale.shape
        weight_scale = weight_scale.view(weight_scale_m, 1, weight_scale_n, 1)
        weight = weight.view(weight_scale_m, block_size_m, weight_scale_n, block_size_n)
        dequant_weight = weight.to(dtype) * weight_scale.to(dtype)
        dequant_weight = dequant_weight.view(weight_scale_m * block_size_m, weight_scale_n * block_size_n)
        keep_first_dim = False
    elif weight_shape_len == 3:
        fd, weight_scale_m, weight_scale_n = weight_scale.shape
        weight_scale = weight_scale.view(fd, weight_scale_m, 1, weight_scale_n, 1)
        weight = weight.view(fd, weight_scale_m, block_size_m, weight_scale_n, block_size_n)
        dequant_weight = weight.to(dtype) * weight_scale.to(dtype)
        dequant_weight = dequant_weight.view(fd, weight_scale_m * block_size_m, weight_scale_n * block_size_n)
        keep_first_dim = True
    else:
        raise ValueError("Only support original weight shape is either 2 or 3")

    dequant_weight = _unpad_weight(dequant_weight, orig_M, orig_N, keep_first_dim=keep_first_dim)

    return dequant_weight


def check_to_quantized(config):
    """Checks if the configuration is valid for quantization.

    Args:
        config (dict or object): The configuration to check. It can be either a
            dictionary with a 'bits' key or an object with a 'bits' attribute.

    Returns:
        bool: True if the configuration is valid for quantization (bits <= 8),
            False otherwise.
    """

    if isinstance(config, (dict, QuantizationScheme)):
        bits = int(config.get("bits", 16))
        act_bits = int(config.get("act_bits", 16))
    elif hasattr(config, "orig_layer"):
        bits = int(config.orig_layer.bits) if hasattr(config.orig_layer, "bits") else 16
        act_bits = int(config.orig_layer.act_bits) if hasattr(config.orig_layer, "act_bits") else 16
    else:
        bits = int(config.bits) if hasattr(config, "bits") else 16
        act_bits = int(config.act_bits) if hasattr(config, "act_bits") else 16

    return bits <= 8 or act_bits <= 8


def check_seqlen_compatible(input_seqlen, tokenizer=None, model=None):
    """
    Check whether the input sequence length is within the limits defined
    by the tokenizer and the model configuration.

    Args:
        input_seqlen (int): The length of the input sequence.
        tokenizer: Optional, a HuggingFace tokenizer object.
        model: Optional, a HuggingFace model object.

    Returns:
        ValueError: if the input length is not valid, riase Error.
    """
    if model is not None and hasattr(model, "config"):
        model_config = model.config
        if hasattr(model_config, "max_position_embeddings") and input_seqlen > model_config.max_position_embeddings:
            raise ValueError(
                f"seqlen({input_seqlen}) exceeds model.config.max_position_embeddings("
                f"{model_config.max_position_embeddings}). Please lowering '--seqlen'"
            )
    if tokenizer is not None and hasattr(tokenizer, "model_max_length") and input_seqlen > tokenizer.model_max_length:
        raise ValueError(
            f"seqlen({input_seqlen}) exceeds tokenizer.model_max_length({tokenizer.model_max_length}). "
            "Please oncider Consider lowering the '--seqlen' or increasing tokenizer.model_max_length."
        )


def convert_fp8_layer_to_linear(layer, dtype=torch.bfloat16, device: str = "cpu"):
    """ """
    from auto_round.schemes import QuantizationScheme

    new_layer = torch.nn.Linear(layer.in_features, layer.out_features, bias=layer.bias is not None, dtype=dtype)
    if layer.bias is not None:
        new_layer.bias.data.copy_(layer.bias.data.to(dtype=dtype))
    scheme_keys = (f.name for f in fields(QuantizationScheme))
    keys = tuple(scheme_keys) + ("tmp_name", "scale_dtype")
    for key in keys:
        setattr(new_layer, key, getattr(layer, key, None))

    layer = layer.to(device)
    if layer.__class__.__name__ == "CompressedLinear":
        dq_weight = layer.compressor.decompress_module(layer)
    else:
        weight_scale = layer.weight_scale if hasattr(layer, "weight_scale") else layer.weight_scale_inv
        dq_weight = dequant_block_fp8_weight(layer.weight, weight_scale, layer.block_size)
    new_layer.weight.data.copy_(dq_weight.to(dtype=dtype))
    return new_layer


def convert_fp8_model_to_16b_model(model, dtype=torch.bfloat16, device: str = "cpu"):
    """
    Convert a model with FP8 quantized layers to a model with 16-bit linear layers.
    This is useful for compatibility with other frameworks or for further processing.
    """
    from auto_round.utils.device import clear_memory

    cnt = 0
    for n, m in model.named_modules():
        if m.__class__.__name__ == "FP8Linear":
            new_module = convert_fp8_layer_to_linear(m, dtype=dtype, device=device)
            set_module(model, n, new_module)
            cnt += 1
            if cnt % 10 == 0:  # Tricky setting
                clear_memory()
    return model


def _to_model_dtype(model, model_dtype):
    if model_dtype is not None:
        try:
            if (model_dtype == "float16" or model_dtype == "fp16") and model.dtype != torch.float16:
                model = model.to(torch.float16)
            elif (
                model_dtype == "bfloat16" or model_dtype == "bfp16" or model_dtype == "bf16"
            ) and model.dtype != torch.bfloat16:
                model = model.to(torch.bfloat16)
            elif model_dtype == "float32" or model_dtype == "fp32" and model.dtype != torch.bfloat32:
                model = model.to(torch.float32)
        except:
            logger.error("please use more device to fit the device or just use one device")
            exit()
    return model


def get_module(module, key):
    """Get module from model by key name.

    Args:
        module (torch.nn.Module): original model
        key (str): module name to be replaced
    """
    name_list = key.split(".")
    for name in name_list:
        module = getattr(module, name, None)
    return module


def set_module(model, key, new_module):
    """Set new module into model by key name.

    Args:
        model (torch.nn.Module): original model
        key (str): module name to be replaced
        new_module (torch.nn.Module): new module to be inserted
    """
    module = model
    name_list = key.split(".")
    for name in name_list[:-1]:
        if hasattr(module, name):
            module = getattr(module, name)
    setattr(module, name_list[-1], new_module)


def get_layer_features(layer):
    """Extracts input and output feature dimensions for supported layers."""
    from auto_round.utils import deepspeed_exists

    if deepspeed_exists:
        from deepspeed.module_inject import LinearAllreduce, LinearLayer
    if type(layer) == torch.nn.Linear:
        return layer.in_features, layer.out_features
    elif type(layer) == transformers.pytorch_utils.Conv1D:  # TODO: Verify correctness
        return layer.weight.shape[0], layer.weight.shape[1]
    elif isinstance(layer, torch.nn.Embedding):
        return layer.num_embeddings, layer.embedding_dim
    elif deepspeed_exists and type(layer) in (LinearLayer, LinearAllreduce):
        return layer.weight.shape[1], layer.weight.shape[0]  # (input_dim, output_dim)
    elif "FP8Linear" in layer.__class__.__name__:
        return layer.in_features, layer.out_features
    return None, None  # Unsupported layer type


def get_common_prefix(paths):
    # Split each path into components and find the common prefix
    split_paths = [path.split(".") for path in paths]
    common_prefix = split_paths[0]
    for path in split_paths[1:]:
        common_prefix = [comp for comp, other in zip(common_prefix, path) if comp == other]
    return ".".join(common_prefix)


def unsupported_meta_device(model):
    """Checks if the model is a valid model for auto_round.

    Args:
    model: The model to be checked.

    Returns:
    bool: True if the model is valid, False otherwise.
    """
    target_device = None
    for param in model.parameters():
        if target_device is None:
            target_device = param.device
        if param.device != target_device:
            if param.device.type == "meta" or target_device.type == "meta":
                return True
    if target_device.type == "meta":
        if hasattr(model, "path"):
            return False
        else:
            return True
    return False


def to_device(input, device=torch.device("cpu")):
    """Moves input data to the specified device.

    Args:
    input: The input data to be moved.
    device: The target device.

    Returns:
    The input data on the specified device.
    """
    if input is None:
        return None
    if isinstance(input, torch.Tensor):
        return input.to(device)
    if isinstance(input, dict) or isinstance(input, UserDict):
        for inp in input.keys():
            input[inp] = to_device(input[inp], device)

    elif isinstance(input, list) or isinstance(input, tuple):
        if len(input) == 0:
            return input
        input_res = []
        for inp in input:
            input_res.append(to_device(inp, device))
        if isinstance(input, tuple):
            input_res = tuple(input_res)
        input = input_res

    return input


def mv_module_from_gpu(module):
    """Moves module from gpu to cpu.

    Args:
    module: The module to be moved.

    Returns:
    The module on the specified device.
    """
    if hasattr(module, "device"):
        target_device = "cpu"
        if module.device.type == target_device:
            return module
        else:
            return module.to(target_device)
    else:
        return module.to("cpu")


def to_dtype(input, dtype=torch.float32):
    """Moves input data to the specified data type.

    Args:
    input: The input data to be moved.
    dtype: The target data type.

    Returns:
    The input data on the specified data type.
    """
    if input is None:
        return None
    if isinstance(input, torch.Tensor):
        return input.to(dtype)
    if isinstance(input, dict) or isinstance(input, UserDict):
        for inp in input.keys():
            input[inp] = to_dtype(input[inp], dtype)

    elif isinstance(input, list) or isinstance(input, tuple):
        if len(input) == 0:
            return input
        input_res = []
        for inp in input:
            input_res.append(to_dtype(inp, dtype))
        if isinstance(input, tuple):
            input_res = tuple(input_res)
        input = input_res

    return input


def set_amax_for_uncalibrated_experts(
    experts: torch.nn.Module, set_amax_value: float | None = None, attr_name="act_max"
):
    """Set amax of uncalibrated experts to a given value or the max of existing amax value from other experts.

    Args:
        experts: a list of experts
        set_amax_value: set amax value to the given value.
                        If None, set amax value to the max of existing amax value from other experts.

    Returns:
        uncalibrated_experts: a list of uncalibrated experts
    """
    uncalibrated_experts = []
    # get the max amax value from all experts
    if set_amax_value is None:
        amax_values = [
            get_nested_attr(module, attr_name) for module in experts if get_nested_attr(module, attr_name) is not None
        ]
        if len(amax_values) == 0:
            return uncalibrated_experts
        # Flatten all tensors to 1D before concatenation
        flat_values = [t.reshape(-1) for t in amax_values]
        all_values = torch.cat(flat_values)
        set_amax_value = torch.max(all_values)

    for module in experts:
        if get_nested_attr(module, attr_name) is None:
            logger.warning_once(
                "Missing amax value of expert layers."
                "This typically occurs in MoE models when certain experts are not activated during calibration. "
                "Consider increasing your calibration dataset size to ensure all experts are exercised."
            )
            # Use float32 dtype explicitly to ensure we create a floating point tensor
            if not isinstance(set_amax_value, torch.Tensor):
                set_amax_value = torch.tensor(set_amax_value, dtype=torch.float32)
            set_nested_attr(module, attr_name, set_amax_value)
            # uncalibrated_experts.append(module)


# Please refer to: https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/
# 4c611e47a60084a86e1de7e48690a692a1b8170c/modelopt/torch/export/unified_export_hf.py#L195-L207
def set_amax_for_all_moe_layers(model: torch.nn.Module, layer_name=None, attr_name="act_max"):
    if layer_name is not None:
        parts = layer_name.split(".")
        if "experts" not in parts:
            raise ValueError
        idx = parts.index("experts")
        moe_name = ".".join(parts[:idx])
        model = get_module(model, moe_name)
    # Handle input quantizers of experts that are not calibrated
    for name, sub_module in model.named_modules():
        if not (is_moe(sub_module) and hasattr(sub_module, "experts")):
            continue
        expert_linear_names = get_expert_linear_names(sub_module)
        for linear_name in expert_linear_names:
            if isinstance(sub_module.experts, collections.abc.Iterable):
                # For other MoE models (like Mixtral) with iterable experts
                try:
                    set_amax_for_uncalibrated_experts(
                        [getattr(expert, linear_name, None) for expert in sub_module.experts], attr_name=attr_name
                    )
                except AttributeError as e:
                    # Provide more helpful debugging information
                    expert_types = list(set(type(expert).__name__ for expert in sub_module.experts))
                    raise AttributeError(
                        f"Failed to access attribute '{linear_name}' on experts. "
                        f"MoE module type: {type(sub_module).__name__}, "
                        f"Expert types: {expert_types}, "
                        f"Expected linear names: {expert_linear_names}. "
                        f"This suggests the get_expert_linear_names function may need "
                        f"to be updated for this model architecture. "
                        f"Original error: {e}"
                    ) from e
            else:
                # Unsupported MoE model structure
                raise NotImplementedError(
                    f"MoE model with experts type '{type(sub_module.experts).__name__}' is not supported in export."
                    f"Please file an issue or add support for this model architecture."
                )


# Adapted from https://github.com/vllm-project/llm-compressor/blob/
# 5b3ddff74cae9651f24bef15d3255c4ee053fc60/src/llmcompressor/pytorch/model_load/helpers.py#L144
def copy_python_files_from_model_cache(model, save_path: str):
    config = model.config
    cache_path = None
    if hasattr(config, "_name_or_path"):
        import os
        import shutil

        from huggingface_hub import hf_hub_download
        from transformers import TRANSFORMERS_CACHE
        from transformers.utils import http_user_agent

        cache_path = config._name_or_path
        if not os.path.exists(cache_path):
            user_agent = http_user_agent()
            config_file_path = hf_hub_download(
                repo_id=cache_path,
                filename="config.json",
                cache_dir=TRANSFORMERS_CACHE,
                force_download=False,
                user_agent=user_agent,
            )
            cache_path = os.path.sep.join(config_file_path.split(os.path.sep)[:-1])

        for file in os.listdir(cache_path):
            full_file_name = os.path.join(cache_path, file)
            if file.endswith(".py") and os.path.isfile(full_file_name):
                logger.debug(f"Transferring {full_file_name} to {save_path}")
                shutil.copy(full_file_name, save_path)


def extract_block_names_to_str(quant_block_list):
    if not isinstance(quant_block_list, (list, tuple)):
        return None
    # Extract common prefix for each list
    prefixes = [get_common_prefix(blocks) for blocks in quant_block_list]
    # Join prefixes into a single string
    return ",".join(prefixes)


def find_matching_blocks(model, all_blocks, to_quant_block_names):
    """
    Find and return matching blocks in the model based on to_quant_block_names.

    Args:
        model: The model (not used in this specific function but kept for completeness).
        all_blocks: List of lists, where each inner list contains full block names in the model.
        to_quant_block_names: Comma-separated string of target block names to match.

    Returns:
        target_blocks: List of lists containing full paths of matching blocks in the model.
    """
    if not to_quant_block_names:
        return all_blocks
    to_quant_block_list = to_quant_block_names
    if isinstance(to_quant_block_names, list) or isinstance(to_quant_block_names, tuple):
        return to_quant_block_names
    if isinstance(to_quant_block_names, str):
        to_quant_block_list = [name.strip() for name in to_quant_block_names.split(",")]
    target_blocks = []
    for block_list in all_blocks:
        matched_sublist = []
        for name in to_quant_block_list:
            matches = [block for block in block_list if re.search(name, block)]
            if matches:
                matched_sublist.extend(matches)
        if matched_sublist:
            target_blocks.append(matched_sublist)
    if not target_blocks:
        raise ValueError(
            "No block names matched. Please check the input for to_quant_block_name,"
            "or set to_quant_block_name to None to automatically match quantizable blocks."
        )
    return target_blocks


def is_separate_lm_head(model: torch.nn.Module) -> bool:
    dir_path = model.name_or_path
    if not os.path.isdir(dir_path):
        dir_path = download_hf_model(dir_path)
    lm_head_name: str = get_lm_head_name(model)
    lm_head_name += ".weight"

    if "model.safetensors.index.json" in os.listdir(dir_path):
        with open(os.path.join(dir_path, "model.safetensors.index.json")) as f:
            index_mapping = json.load(f)
            if lm_head_name in index_mapping["weight_map"]:
                return True
            else:
                return False
    else:
        from safetensors import safe_open

        f = safe_open(os.path.join(dir_path, "model.safetensors"), framework="pt")
        if lm_head_name in f.keys():
            return True
        else:
            return False
