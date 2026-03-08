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
import json
import os
import shutil

import torch.nn as nn

from auto_round.utils import copy_python_files_from_model_cache, logger, unsupported_meta_device


def is_local_pipeline_model_dir(model_dir: str) -> bool:
    if not model_dir or not os.path.isdir(model_dir):
        return False
    return os.path.isfile(os.path.join(model_dir, "model_index.json"))


def is_remote_pipeline_model_dir(model_dir: str) -> bool:
    if not model_dir or os.path.isdir(model_dir):
        return False
    try:
        from huggingface_hub import list_repo_files

        return "model_index.json" in list_repo_files(model_dir)
    except Exception:
        return False


def is_pipeline_model_dir(model_dir: str) -> bool:
    return is_local_pipeline_model_dir(model_dir) or is_remote_pipeline_model_dir(model_dir)


def _resolve_pipeline_source_dir(model: nn.Module) -> str | None:
    candidates = [
        getattr(model, "name_or_path", None),
        getattr(getattr(model, "config", None), "_name_or_path", None),
        getattr(getattr(model, "config", None), "name_or_path", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and is_pipeline_model_dir(candidate):
            return candidate
    return None


def _copy_pipeline_artifact(model_dir: str, relative_path: str, output_dir: str) -> None:
    target_path = os.path.join(output_dir, relative_path)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if is_local_pipeline_model_dir(model_dir):
        source_path = os.path.join(model_dir, relative_path)
    else:
        from huggingface_hub import hf_hub_download

        source_path = hf_hub_download(model_dir, relative_path)
    shutil.copy2(source_path, target_path)


def _copy_pipeline_artifacts(source_dir: str, output_dir: str, exclude_components: set[str] | None = None):
    exclude_components = exclude_components or set()
    os.makedirs(output_dir, exist_ok=True)

    model_index_path = os.path.join(source_dir, "model_index.json") if is_local_pipeline_model_dir(source_dir) else None
    if model_index_path:
        with open(model_index_path, "r", encoding="utf-8") as f:
            model_index = json.load(f)
    else:
        from huggingface_hub import hf_hub_download, list_repo_files

        with open(hf_hub_download(source_dir, "model_index.json"), "r", encoding="utf-8") as f:
            model_index = json.load(f)

    component_dirs = [k for k, v in model_index.items() if not k.startswith("_") and isinstance(v, list)]
    is_local = is_local_pipeline_model_dir(source_dir)

    # Copy root-level files
    if is_local:
        for name in os.listdir(source_dir):
            src = os.path.join(source_dir, name)
            if os.path.isfile(src) and (
                name in ("model_index.json", ".gitattributes") or name.lower().startswith(("readme", "license"))
            ):
                shutil.copy2(src, os.path.join(output_dir, name))
    else:
        all_files = list(list_repo_files(source_dir))
        for name in all_files:
            if "/" not in name and (
                name in ("model_index.json", ".gitattributes") or name.lower().startswith(("readme", "license"))
            ):
                _copy_pipeline_artifact(source_dir, name, output_dir)

    # Copy component directories
    for component_name in component_dirs:
        if component_name in exclude_components:
            continue
        if is_local:
            src = os.path.join(source_dir, component_name)
            dst = os.path.join(output_dir, component_name)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            prefix = f"{component_name}/"
            for f in all_files:
                if f.startswith(prefix):
                    _copy_pipeline_artifact(source_dir, f, output_dir)


def resolve_pipeline_export_layout(model: nn.Module, output_dir: str) -> tuple[str, str, bool]:
    model_component = getattr(model, "_autoround_pipeline_subfolder", None)
    if model_component is None:
        return output_dir, output_dir, False

    source_dir = _resolve_pipeline_source_dir(model)
    processor_component = None
    if source_dir is not None:
        try:
            model_index_path = (
                os.path.join(source_dir, "model_index.json") if is_local_pipeline_model_dir(source_dir) else None
            )
            if model_index_path:
                with open(model_index_path, "r", encoding="utf-8") as f:
                    model_index = json.load(f)
            else:
                from huggingface_hub import hf_hub_download

                with open(hf_hub_download(source_dir, "model_index.json"), "r", encoding="utf-8") as f:
                    model_index = json.load(f)
            if "processor" in model_index and isinstance(model_index["processor"], list):
                processor_component = "processor"
            excluded = {model_component}
            if processor_component:
                excluded.add(processor_component)
            _copy_pipeline_artifacts(source_dir, output_dir, exclude_components=excluded)
        except Exception as e:
            logger.warning("Failed to copy pipeline artifacts from %s: %s", source_dir, e)

    model_output_dir = os.path.join(output_dir, model_component)
    processor_output_dir = os.path.join(output_dir, processor_component) if processor_component else output_dir
    return model_output_dir, processor_output_dir, True


def save_model(
    model: nn.Module,
    save_dir: str,
    max_shard_size: str = "5GB",
    safe_serialization: bool = True,
    dtype=None,
    config_file="quantization_config.json",
):
    """Save model state dict and configs.

    Args:
        model (`nn.Module`):
            Model to be saved. The model can be wrapped or unwrapped.
        save_dir (`str`):
            Directory to which to save. Will be created if it doesn't exist.
        max_shard_size (`str`, defaults to `"10GB"`):
            The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
            lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
            <Tip warning={true}>

            If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
            which will be bigger than `max_shard_size`.

            </Tip>
        safe_serialization (`bool`, defaults to `True`):
            Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
    """
    os.makedirs(save_dir, exist_ok=True)

    if unsupported_meta_device(model):
        if hasattr(model, "config") and model.config is not None:
            model.config.save_pretrained(save_dir)

        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.save_pretrained(save_dir)
    else:
        model.save_pretrained(save_dir, max_shard_size=max_shard_size, safe_serialization=safe_serialization)

    config_path = os.path.join(save_dir, "config.json")
    if dtype is not None and dtype != model.dtype and os.path.exists(os.path.join(save_dir, "config.json")):
        with open(config_path, "r") as file:
            data = json.load(file)
        data["torch_dtype"] = str(dtype).split(".")[-1]
        with open(config_path, "w") as file:
            json.dump(data, file, indent=2)

    config_file = "quantization_config.json"
    if hasattr(model, "config") and hasattr(model.config, "quantization_config"):
        with open(os.path.join(save_dir, config_file), "w", encoding="utf-8") as f:
            json.dump(model.config.quantization_config, f, indent=2)

    try:
        copy_python_files_from_model_cache(model, save_dir)
    except Exception as e:
        logger.warning("Skipping source model Python file copy due to error: %s", e)


def get_autogptq_packing_qlinear(backend, bits=4, group_size=128, sym=False):
    """
    Configures and returns a QuantLinear class based on the specified backend and parameters.

    Args:
        backend (str): The backend to be used for quantization. Supported values include "qigen", "triton", "marlin",
                       "exllama", and "cuda".
        bits (int, optional): The number of bits for quantization. Default is 4.
        group_size (int, optional): The group size for quantization. Default is 128.
        sym (bool, optional): Flag indicating whether to use symmetric quantization. Default is False.

    Returns:
        class: The dynamically imported QuantLinear class configured according to the specified parameters.
    """
    use_triton = True
    if bits not in [2, 4, 8]:
        use_triton = False
    disable_exllamav2 = True
    disable_exllamav1 = False
    disable_marlin = True
    use_qigen = False
    if "qigen" in backend:
        use_triton = False
        use_qigen = True
    elif "triton" in backend:
        use_triton = True
    elif "marlin" in backend and sym:
        use_triton = False
        disable_marlin = False
    elif "exllama" in backend:  ##need v1 code to export
        use_triton = True  ##same with triton
        disable_marlin = True
    elif "cuda" in backend:
        use_triton = False
        disable_marlin = True
        disable_exllamav2 = True
        disable_exllamav1 = True
    if use_triton:
        from auto_round.export.export_to_autogptq.qlinear_triton import QuantLinear

        return QuantLinear
    try:
        import auto_gptq  # pylint: disable=E0401
    except:
        logger.error(f"please install auto_gptq via 'pip install auto-gptq' to support exporting to {backend}")
        exit()

    from auto_gptq.utils.import_utils import dynamically_import_QuantLinear  # pylint: disable=E0401

    from auto_round.utils.common import get_library_version

    version = get_library_version("auto_gptq")
    from packaging.version import Version

    if Version(version) < Version("0.7.2"):
        QuantLinear = dynamically_import_QuantLinear(
            use_triton=use_triton,
            desc_act=False,
            group_size=group_size,
            bits=bits,
            disable_exllama=disable_exllamav1,
            disable_exllamav2=disable_exllamav2,
            use_qigen=use_qigen,
            disable_marlin=disable_marlin,
        )
    else:
        QuantLinear = dynamically_import_QuantLinear(  # pylint: disable=E1123
            use_triton=use_triton,
            desc_act=False,
            group_size=group_size,
            bits=bits,
            disable_exllama=disable_exllamav1,
            disable_exllamav2=disable_exllamav2,
            use_qigen=use_qigen,
            use_marlin=not disable_marlin,
        )
    return QuantLinear


def filter_quantization_config(quantization_config):
    default_dict = {
        "amp": True,
        "batch_size": 8,
        "data_type": int,
        "dataset": "NeelNanda/pile-10k",
        "enable_minmax_tuning": True,
        "enable_norm_bias_tuning": False,
        "enable_quanted_input": True,
        "gradient_accumulate_steps": 1,
        "iters": 200,
        "low_gpu_mem_usage": False,
        "nsamples": 128,
        "scale_dtype": "torch.float16",
        "seqlen": 2048,
    }
    iters = quantization_config.get("iters", 200)

    default_dict["lr"] = 1.0 / iters if iters > 0 else 5e-3
    default_dict["minmax_lr"] = default_dict["lr"]

    for key in default_dict:
        if key in quantization_config and default_dict[key] == quantization_config[key]:
            quantization_config.pop(key)
    for k in list(quantization_config.keys()):
        if quantization_config[k] is None:
            quantization_config.pop(k)

    if quantization_config.get("act_bits", 16) >= 16:
        quantization_config.pop("act_bits", None)
        quantization_config.pop("act_data_type", None)
        quantization_config.pop("act_dynamic", None)
        quantization_config.pop("act_sym", None)
        quantization_config.pop("act_group_size", None)

    clean_list = ("supported_types", "quant_block_list")
    for key in list(quantization_config.keys()):
        if callable(key):
            quantization_config.pop(key)
        elif isinstance(quantization_config[key], (list, tuple)):
            if any([callable(item) for item in quantization_config[key]]):
                quantization_config.pop(key)
        if key in clean_list and key in quantization_config:
            quantization_config.pop(key)
    return quantization_config


def release_layer_safely(layer: nn.Module):
    """
    Safely releases the weight and bias tensors of a layer to free memory.
    Handles the case where attributes might not exist or are already None.
    """
    for attr in ["weight", "bias"]:
        if hasattr(layer, attr):
            tensor = getattr(layer, attr)
            if tensor is not None:
                # Detach and delete to avoid memory leaks
                tensor.detach_()
                setattr(layer, attr, None)
