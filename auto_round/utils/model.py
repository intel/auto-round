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
from pathlib import Path
from typing import Union

import psutil
import torch
import transformers
from packaging import version

from auto_round import envs
from auto_round.export.export_to_gguf.config import ModelType
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme
from auto_round.utils.weight_handler import (
    _dequant_fp8_linear_weight,
    check_and_mark_quantized_module,
    convert_module_to_hp_if_necessary,
    is_quantized_input_module,
)


def clean_module_parameter(submodule: torch.nn.Module, param_name: str) -> None:
    """This function is recommended to be used instead of module.weight = None.
    For models like `tie_word_embeddings`, setting the embedding weight to None
    causes `lm_head` to reallocate memory for its weight instead of treating it as a "bound shared weight,"
    it's now iterated over as an independent parameter,
    resulting in an additional `lm_head` parameter in `named_parameters`.

    Args:
        submodule (torch.nn.Module): submodule to clean
        param_name (str): "weight" or "bias"
    """
    if submodule is None:
        return
    is_buffer = param_name in submodule._buffers
    with torch.no_grad():
        if is_buffer:
            buf = submodule._buffers[param_name]
            if buf is not None:
                buf.data = torch.empty(0, dtype=buf.dtype, device=buf.device)
                buf.requires_grad = False
        else:
            param = submodule._parameters[param_name]
            if param is not None:
                param.data = torch.empty(0, dtype=param.dtype, device=param.device)
                param.requires_grad = False


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


def download_or_get_path(repo_id: str, platform: str = None) -> str:
    from auto_round import envs

    if platform is None:
        if envs.AR_USE_MODELSCOPE:
            platform = "model_scope"
        else:
            platform = "hf"

    if platform == "model_scope":
        return download_modelscope_model(repo_id)
    else:
        return download_hf_model(repo_id)


def download_modelscope_model(repo_id: str, local_dir: str = None, cache_dir: str = None):
    from modelscope.utils.file_utils import get_modelscope_cache_dir  # pylint: disable=E0401

    system_cache = cache_dir if cache_dir is not None else get_modelscope_cache_dir()
    if local_dir:
        directory = os.path.abspath(local_dir)
    elif cache_dir:
        directory = os.path.join(system_cache, *repo_id.split("/"))
    else:
        directory = os.path.join(system_cache, "models", *repo_id.split("/"))
    if os.path.exists(directory):
        return directory
    else:
        from modelscope.hub.snapshot_download import snapshot_download  # pylint: disable=E0401

        return snapshot_download(repo_id)


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


def _check_accelerate_version():
    from auto_round.utils.common import get_library_version

    accelerate_version = get_library_version("accelerate")
    from packaging.version import Version

    if Version(accelerate_version) > Version("1.5.1") and Version(accelerate_version) < Version("1.10.0"):
        logger.warning(
            f"Detected accelerate version {accelerate_version}. "
            "Versions between 1.5.1 and 1.10.0 may cause high RAM usage during model loading. "
            "It is recommended to upgrade to version 1.10.0 or above."
        )


def llm_load_model(
    pretrained_model_name_or_path: str,
    platform: str = "hf",
    trust_remote_code: bool = True,
    model_dtype: str = None,
    device: str = "cpu",
    **kwargs,
):

    assert platform.lower() in [
        "hf",
        "model_scope",
    ], "current only support hf or model_scope platform to load pretrained model."
    if platform.lower() == "model_scope" and not envs.AR_USE_MODELSCOPE:
        envs.set_config(AR_USE_MODELSCOPE=True)

    _check_accelerate_version()

    if platform == "model_scope":
        from modelscope import AutoModel, AutoModelForCausalLM, AutoTokenizer  # pylint: disable=E0401
    else:
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    from auto_round.utils.device import (
        _use_hpu_compile_mode,
        get_device_and_parallelism,
        override_cuda_device_capability,
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
                with override_cuda_device_capability():
                    model = model_cls.from_pretrained(
                        pretrained_model_name_or_path,
                        torch_dtype=torch_dtype,
                        trust_remote_code=trust_remote_code,
                        device_map="auto" if use_auto_mapping else None,
                    )
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
    check_and_mark_quantized_module(model)
    model = _to_model_dtype(model, model_dtype)

    return model, tokenizer


def mllm_load_model(
    pretrained_model_name_or_path: str,
    platform: str = "hf",
    device: str = "cpu",
    torch_dtype: str = "auto",
    use_auto_mapping: bool = True,
    trust_remote_code: bool = True,
    model_dtype: str = None,
    **kwargs,
):
    from auto_round.special_model_handler import MISTRAL_3_2_MODELS

    _check_accelerate_version()

    assert platform.lower() in [
        "hf",
        "model_scope",
    ], "current only support hf or model_scope platform to load pretrained model."
    if platform.lower() == "model_scope" and not envs.AR_USE_MODELSCOPE:
        envs.set_config(AR_USE_MODELSCOPE=True)

    if platform == "model_scope":
        import modelscope  # pylint: disable=E0401
        from modelscope import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer  # pylint: disable=E0401

        base_lib = modelscope
    else:
        import transformers
        from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        base_lib = transformers

    from auto_round.utils.device import get_device_and_parallelism, override_cuda_device_capability

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
                base_lib, n := architectures.replace("Model", "ForConditionalGeneration")
            ):
                cls = getattr(base_lib, n)
            elif hasattr(base_lib, architectures):
                cls = getattr(base_lib, architectures)
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
                    with override_cuda_device_capability():
                        model = cls.from_pretrained(
                            pretrained_model_name_or_path,
                            trust_remote_code=trust_remote_code,
                            torch_dtype=torch_dtype,
                            device_map="auto" if use_auto_mapping else None,
                        )
                    logger.warning("the support for fp8 model as input is experimental, please use with caution.")
                else:
                    raise

            if any([name in model.name_or_path for name in MISTRAL_3_2_MODELS]):
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
                if platform == "model_scope":
                    from modelscope import AutoImageProcessor  # pylint: disable=E0401
                else:
                    from transformers import AutoImageProcessor

                image_processor = AutoImageProcessor.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code
                )
            except Exception as e:
                pass

    model = model.eval()
    check_and_mark_quantized_module(model)
    model = _to_model_dtype(model, model_dtype)

    return model, processor, tokenizer, image_processor


def diffusion_load_model(
    pretrained_model_name_or_path: str,
    platform: str = "hf",
    device: Union[str, torch.device] = "cpu",
    torch_dtype: Union[str, torch.dtype] = "auto",
    use_auto_mapping: bool = False,
    trust_remote_code: bool = True,
    model_dtype: str = None,
    **kwargs,
):
    from auto_round.utils.common import LazyImport
    from auto_round.utils.device import get_device_and_parallelism

    _check_accelerate_version()

    if platform != "hf":
        raise NotImplementedError(
            f"auto_round current only support hf as platform for diffusion model, but get {platform}"
        )

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


def is_mllm_model(model_or_path: Union[str, torch.nn.Module], platform: str = None):
    from auto_round.utils.common import MM_KEYS

    model_path = model_or_path if isinstance(model_or_path, str) else model_or_path.name_or_path
    # For dummy model, model_path could be "".
    if model_path and not os.path.isdir(model_path):
        model_path = download_or_get_path(model_path, platform=platform)

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


def is_moe_layer(module: torch.nn.Module) -> bool:
    """Returns whether the module is an MOE layer."""
    return "moe" in type(module).__name__.lower() or any(
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
            if hasattr(type(m), "__name__") and "NgramEmbedding" in type(m).__name__:
                continue
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
        target_modules = _search_block("", model)

        for i, target_m in enumerate(target_modules):
            if quant_vision or all(key not in target_m[0].lower() for key in (vision_blocks_tuple)):
                block_names.append([])
                for n, m in target_m[1].named_children():
                    block_names[-1].append(target_m[0] + "." + n)
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
        module,
        [
            "Qwen2MoeSparseMoeBlock",
            "Qwen3MoeSparseMoeBlock",
            "DeepseekMoE",
            "DeepseekV2MoE",
            "DeepseekV3MoE",
        ],
    ):
        return ["gate_proj", "down_proj", "up_proj"]
    elif module_match_name_list(module, ["MixtralMoeSparseMoeBlock"]):
        return ["linear_fc1", "linear_fc2"]
    elif module_match_name_list(module, ["DBRXMoeSparseMoeBlock"]):
        return ["w1_linear", "w2_linear", "v1_linear"]
    else:
        # assuming w1, w2, w3 by default
        return ["w1", "w2", "w3"]


def get_expert_input_proj_names(module: torch.nn.Module) -> list[str]:
    """Get the list of input projection names for MoE experts.

    Input projections are the first linear layers that receive the expert's input directly.
    For FP8 dispatch efficiency, these projections need unified input scales across all experts.

    Args:
        module: The MoE module (e.g., SparseMoeBlock)

    Returns:
        List of input projection names (e.g., ['gate_proj', 'up_proj'])
    """

    def module_match_name_list(module, name_list):
        """Check if the module name matches any of the names in the list."""
        return any(name.lower() in type(module).__name__.lower() for name in name_list)

    if module_match_name_list(
        module, ["Qwen2MoeSparseMoeBlock", "Qwen3MoeSparseMoeBlock", "DeepseekMoE", "DeepseekV2MoE", "DeepseekV3MoE"]
    ):
        # gate_proj and up_proj are input projections, down_proj is output
        return ["gate_proj", "up_proj"]
    elif module_match_name_list(module, ["MixtralMoeSparseMoeBlock"]):
        # Mixtral uses linear_fc1 as input projection, linear_fc2 is output
        return ["linear_fc1"]
    elif module_match_name_list(module, ["DBRXMoeSparseMoeBlock"]):
        # w1_linear and v1_linear are input projections, w2_linear is output
        return ["w1_linear", "v1_linear"]
    else:
        logger.warning_once("Using default input projection names ['w1', 'w3'] for MoE expert alignment. ")
        # Default: w1 and w3 are input projections, w2 is output
        return ["w1", "w3"]


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
        try:
            hparams = ModelBase.load_hparams(dir_model, is_mistral_format)
        except Exception:
            is_mistral_format = False
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
            m.bk_global_name = n
    layers_in_block = []
    if bool(quant_block_list):
        all_blocks = quant_block_list
    else:
        all_blocks = get_block_names(model)
    for block_names in all_blocks:
        for block_name in block_names:
            block = get_module(model, block_name)
            for n, m in block.named_modules():
                if hasattr(m, "bk_global_name"):
                    layers_in_block.append(m.bk_global_name)
                    delattr(m, "bk_global_name")
    return layers_in_block


def set_nested_attr(module, attr_name: str, value):
    """Recursively set nested attribute (e.g., 'orig_layer.act_max' = value)."""
    attrs = attr_name.split(".")
    for attr in attrs[:-1]:
        if not hasattr(module, attr):
            return None  # No need to set act_max for fp layers
        module = getattr(module, attr)
    setattr(module, attrs[-1], value)


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
        bits = config.get("bits", None)
        act_bits = config.get("act_bits", None)

    elif hasattr(config, "orig_layer"):
        bits = getattr(config.orig_layer, "bits", None)
        act_bits = getattr(config.orig_layer, "act_bits", None)

    else:
        bits = getattr(config, "bits", None)
        act_bits = getattr(config, "act_bits", None)

    bits = int(bits) if bits is not None else 16
    act_bits = int(act_bits) if act_bits is not None else 16

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


# For getting and setting attribution, such as 'lm_head.weight'
get_attr = get_module
set_attr = set_module


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


def is_moe_model(model: torch.nn.Module) -> bool:
    if hasattr(model, "config") and hasattr(model.config, "to_dict"):
        for key in model.config.to_dict().keys():
            if "moe" in key or "expert" in key:
                return True
    for n, m in model.named_modules():
        if "expert" in n:
            return True
    return False


def is_moe_model_via_config(config) -> bool:
    if hasattr(config, "to_dict"):
        for key in config.to_dict().keys():
            if "moe" in key or "expert" in key:
                return True
    return False


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
    experts: torch.nn.Module, set_amax_value: float | None = None, attr_name="act_max", unify_all: bool = False
):
    """Set amax of uncalibrated experts to a given value or the max of existing amax value from other experts.

    Args:
        experts: a list of experts
        set_amax_value: set amax value to the given value.
                        If None, set amax value to the max of existing amax value from other experts.
        attr_name: attribute name to set (default: "act_max")
        unify_all: if True, unify the amax value for ALL experts (not just uncalibrated ones).
                   This is needed for FP8 dispatch where all experts must share the same input scale.

    Returns:
        uncalibrated_experts: a list of uncalibrated experts
    """
    uncalibrated_experts = []

    def _get_attr(module, name):
        """Get attribute from module or its orig_layer."""
        if hasattr(module, name):
            return getattr(module, name)
        if hasattr(module, "orig_layer") and hasattr(module.orig_layer, name):
            return getattr(module.orig_layer, name)
        return None

    def _get_amax_value(module):
        value = get_nested_attr(module, attr_name)
        if value is None and hasattr(module, "orig_layer"):
            value = get_nested_attr(module.orig_layer, attr_name)
        return value

    # get the max amax value from all experts
    if set_amax_value is None:
        amax_values = [_get_amax_value(m) for m in experts if _get_amax_value(m) is not None]
        if len(amax_values) == 0:
            # Check if any expert actually needs act_max (act_bits < 8, not dynamic, not already quantized)
            sample = next((m for m in experts if m is not None), None)
            if sample is not None:
                act_bits = _get_attr(sample, "act_bits")
                act_dynamic = _get_attr(sample, "act_dynamic")
                is_quantized = "Quant" in sample.__class__.__name__ or hasattr(sample, "is_mx")
                needs_warning = (
                    not is_quantized and isinstance(act_bits, (int, float)) and act_bits < 8 and not act_dynamic
                )
                if needs_warning:
                    logger.warning_once(
                        f"All {len(experts)} expert layers are missing '{attr_name}' values. "
                        f"This may indicate calibration hooks were not attached to expert layers."
                    )
            return uncalibrated_experts
        else:
            # Flatten all tensors to 1D before concatenation
            flat_values = [t.reshape(-1) for t in amax_values]
            all_values = torch.cat(flat_values)
            set_amax_value = torch.max(all_values)

    for module in experts:
        current_amax = _get_amax_value(module)

        # Set amax if it's None (uncalibrated) OR if unify_all is True
        if current_amax is None or unify_all:
            # Use float32 dtype explicitly to ensure we create a floating point tensor
            if not isinstance(set_amax_value, torch.Tensor):
                set_amax_value = torch.tensor(set_amax_value, dtype=torch.float32)
            set_nested_attr(module, attr_name, set_amax_value.clone())
            if current_amax is None:
                uncalibrated_experts.append(module)

    if uncalibrated_experts:
        logger.info_once(
            f"Found {len(uncalibrated_experts)} uncalibrated expert layers. "
            "Using max amax from calibrated experts to fill missing values. "
        )

    return uncalibrated_experts


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
        if not (is_moe_layer(sub_module) and hasattr(sub_module, "experts")):
            continue

        # Handle router (gate) layer - it's a Linear layer used for token routing
        # It needs act_max for quantization but may not be calibrated if it wasn't exercised
        _set_amax_for_moe_auxiliary_layers(sub_module, attr_name=attr_name)

        expert_linear_names = get_expert_linear_names(sub_module)
        # Get input projection names for FP8 dispatch unification
        expert_input_proj_names = get_expert_input_proj_names(sub_module)

        # Check experts structure and handle accordingly
        if _is_unfused_experts_module(sub_module.experts):
            # Unfused experts: gate_up_proj/down_proj are nn.ModuleList
            _set_amax_for_unfused_experts(sub_module.experts, attr_name=attr_name)
        elif _is_fused_experts_module(sub_module.experts):
            # Fused experts: 3D Parameters (e.g., DeepseekV2Experts)
            # For fused experts, act_max is set on the parent MOE module, not individual experts
            # Skip processing here as they don't have individual Linear layers to calibrate
            logger.debug(
                f"Skipping act_max setting for fused experts module '{name}': "
                f"fused experts use parent module's act_max"
            )
            continue
        elif isinstance(sub_module.experts, collections.abc.Iterable):
            # Iterable experts: list of expert modules (e.g., Mixtral)
            for linear_name in expert_linear_names:
                try:
                    # Determine if this is an input projection that needs scale unification
                    unify_scale = linear_name in expert_input_proj_names and envs.AR_ENABLE_UNIFY_MOE_INPUT_SCALE

                    set_amax_for_uncalibrated_experts(
                        [getattr(expert, linear_name, None) for expert in sub_module.experts],
                        attr_name=attr_name,
                        unify_all=unify_scale,  # Unify scales for input projections (gate/up)
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
            # Unknown experts structure
            logger.warning(
                f"Unknown experts structure in '{name}': type={type(sub_module.experts).__name__}. "
                f"Skipping act_max setting. This may cause issues during export."
            )


def _is_unfused_experts_module(module: torch.nn.Module) -> bool:
    """Check if the module is an unfused experts module (has ModuleList gate_up_proj/down_proj)."""
    if not hasattr(module, "gate_up_proj") or not hasattr(module, "down_proj"):
        return False
    return isinstance(module.gate_up_proj, torch.nn.ModuleList) and isinstance(module.down_proj, torch.nn.ModuleList)


def _is_fused_experts_module(module: torch.nn.Module) -> bool:
    """Check if the module is a fused experts module (has 3D Parameter gate_up_proj/down_proj)."""
    if not hasattr(module, "gate_up_proj") or not hasattr(module, "down_proj"):
        return False
    return (
        isinstance(module.gate_up_proj, torch.nn.Parameter)
        and isinstance(module.down_proj, torch.nn.Parameter)
        and module.gate_up_proj.dim() == 3
        and module.down_proj.dim() == 3
    )


def _set_amax_for_unfused_experts(experts_module: torch.nn.Module, attr_name: str = "act_max"):
    """Set amax for unfused experts module with ModuleList attributes.

    This handles experts modules that have been unfused to have:
    - gate_up_proj: nn.ModuleList of nn.Linear (input projections, unified scale)
    - down_proj: nn.ModuleList of nn.Linear (output projections)
    """
    if hasattr(experts_module, "gate_up_proj") and isinstance(experts_module.gate_up_proj, torch.nn.ModuleList):
        unify_scale = envs.AR_ENABLE_UNIFY_MOE_INPUT_SCALE
        set_amax_for_uncalibrated_experts(
            list(experts_module.gate_up_proj),
            attr_name=attr_name,
            unify_all=unify_scale,
        )

    if hasattr(experts_module, "down_proj") and isinstance(experts_module.down_proj, torch.nn.ModuleList):
        set_amax_for_uncalibrated_experts(
            list(experts_module.down_proj),
            attr_name=attr_name,
            unify_all=False,
        )


def _set_amax_for_moe_auxiliary_layers(moe_module: torch.nn.Module, attr_name: str = "act_max"):
    """Set amax for auxiliary layers in MOE modules (gate/router, shared_experts).

    These layers are not part of the experts structure but are siblings in the MOE module.
    They need act_max for quantization but may be missing if not all paths were exercised
    during calibration.

    Args:
        moe_module: The MOE module (e.g., DeepseekV2MoE)
        attr_name: The attribute name for amax (default: "act_max")
    """
    # Collect all Linear layers that have act_bits set but missing act_max
    layers_needing_amax = []

    # Check gate (router) layer - it's typically a Linear layer for token routing
    if hasattr(moe_module, "gate") and isinstance(moe_module.gate, torch.nn.Linear):
        gate = moe_module.gate
        if hasattr(gate, "act_bits") and gate.act_bits < 8:
            if get_nested_attr(gate, attr_name) is None:
                layers_needing_amax.append(gate)

    # Check shared_experts - may have Linear layers that need act_max
    if hasattr(moe_module, "shared_experts"):
        shared_experts = moe_module.shared_experts
        if shared_experts is not None:
            for child_name, child in shared_experts.named_modules():
                if isinstance(child, torch.nn.Linear):
                    if hasattr(child, "act_bits") and child.act_bits < 8:
                        if get_nested_attr(child, attr_name) is None:
                            layers_needing_amax.append(child)

    if not layers_needing_amax:
        return

    # Try to get reference amax from calibrated experts
    reference_amax = _get_reference_amax_from_experts(moe_module, attr_name)

    if reference_amax is not None:
        for layer in layers_needing_amax:
            if not isinstance(reference_amax, torch.Tensor):
                reference_amax = torch.tensor(reference_amax, dtype=torch.float32)
            set_nested_attr(layer, attr_name, reference_amax.clone())
        logger.info_once(
            f"Set act_max for {len(layers_needing_amax)} MOE auxiliary layers (gate/shared_experts) "
            f"using reference value from calibrated experts."
        )
    else:
        logger.warning_once(
            f"Cannot set act_max for {len(layers_needing_amax)} MOE auxiliary layers: "
            f"no calibrated experts found to use as reference."
        )


def _get_reference_amax_from_experts(moe_module: torch.nn.Module, attr_name: str = "act_max"):
    """Get a reference amax value from calibrated expert layers.

    Args:
        moe_module: The MOE module containing experts
        attr_name: The attribute name for amax

    Returns:
        A reference amax tensor, or None if no calibrated experts found
    """
    amax_values = []

    if not hasattr(moe_module, "experts"):
        return None

    experts = moe_module.experts

    # Handle unfused experts (ModuleList)
    if _is_unfused_experts_module(experts):
        for proj_list in [getattr(experts, "gate_up_proj", None), getattr(experts, "down_proj", None)]:
            if proj_list is not None and isinstance(proj_list, torch.nn.ModuleList):
                for layer in proj_list:
                    amax = get_nested_attr(layer, attr_name)
                    if amax is not None:
                        amax_values.append(amax)

    # Handle iterable experts (list of modules)
    elif isinstance(experts, collections.abc.Iterable):
        expert_linear_names = get_expert_linear_names(moe_module)
        for expert in experts:
            for linear_name in expert_linear_names:
                layer = getattr(expert, linear_name, None)
                if layer is not None:
                    amax = get_nested_attr(layer, attr_name)
                    if amax is not None:
                        amax_values.append(amax)

    if not amax_values:
        return None

    # Return max of all amax values
    flat_values = [t.reshape(-1) for t in amax_values]
    all_values = torch.cat(flat_values)
    return torch.max(all_values)


# Adapted from https://github.com/vllm-project/llm-compressor/blob/
# 5b3ddff74cae9651f24bef15d3255c4ee053fc60/src/llmcompressor/pytorch/model_load/helpers.py#L144
def copy_python_files_from_model_cache(model, save_path: str):
    config = model.config
    cache_path = None
    if hasattr(config, "_name_or_path"):
        import os
        import shutil

        from huggingface_hub import hf_hub_download

        if version.parse(transformers.__version__) < version.parse("5.0.0"):
            from transformers import TRANSFORMERS_CACHE

            cache_dir = TRANSFORMERS_CACHE
            from huggingface_hub.constants import HF_HUB_CACHE

            cache_dir = os.environ.get("HF_HOME") or HF_HUB_CACHE

            cache_dir = os.environ.get("HF_HOME", None)
        from transformers.utils import http_user_agent

        cache_path = config._name_or_path
        if not os.path.exists(cache_path):
            user_agent = http_user_agent()
            config_file_path = hf_hub_download(
                repo_id=cache_path,
                filename="config.json",
                cache_dir=cache_dir,
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


def is_separate_tensor(model: torch.nn.Module, tensor_name: str) -> bool:
    dir_path = model.name_or_path
    if not os.path.isdir(dir_path):
        dir_path = download_hf_model(dir_path)
    if not tensor_name.endswith(".weight"):
        tensor_name += ".weight"

    if "model.safetensors.index.json" in os.listdir(dir_path):
        with open(os.path.join(dir_path, "model.safetensors.index.json")) as f:
            index_mapping = json.load(f)
            if tensor_name in index_mapping["weight_map"]:
                return True
            else:
                return False
    else:
        from safetensors import safe_open

        f = safe_open(os.path.join(dir_path, "model.safetensors"), framework="pt")
        if tensor_name in f.keys():
            return True
        else:
            return False
