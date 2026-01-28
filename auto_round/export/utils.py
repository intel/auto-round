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

import torch
import torch.nn as nn

from auto_round.utils import copy_python_files_from_model_cache, logger, unsupported_meta_device


def save_model(
    model: nn.Module,
    save_dir: str,
    max_shard_size: str = "5GB",
    safe_serialization: bool = True,
    dtype=None,
    config_file="quantization_config.json",
    disable_conversion_mapping: bool = False,
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
        disable_conversion_mapping (`bool`, defaults to `False`):
            Whether to disable HuggingFace's checkpoint conversion mapping during save.
            This is needed for models with unfused MOE experts (linear_loop implementation).
    """
    os.makedirs(save_dir, exist_ok=True)

    # If not explicitly set, try to detect unfused MOE experts
    if not disable_conversion_mapping:
        # Detect unfused MOE experts (nn.ModuleList structure) that need conversion mapping bypass
        # This can happen with linear_loop implementation or other unfusing mechanisms
        for _, module in model.named_modules():
            # Check if this module itself has unfused gate_up_proj/down_proj (experts module)
            gate_up = getattr(module, "gate_up_proj", None)
            down = getattr(module, "down_proj", None)
            if isinstance(gate_up, nn.ModuleList) and isinstance(down, nn.ModuleList):
                disable_conversion_mapping = True
                break
            # Also check if this module has an 'experts' submodule with unfused structure
            experts = getattr(module, "experts", None)
            if experts is not None:
                gate_up = getattr(experts, "gate_up_proj", None)
                down = getattr(experts, "down_proj", None)
                if isinstance(gate_up, nn.ModuleList) and isinstance(down, nn.ModuleList):
                    disable_conversion_mapping = True
                    break

    if unsupported_meta_device(model):
        if hasattr(model, "config") and model.config is not None:
            model.config.save_pretrained(save_dir)

        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.save_pretrained(save_dir)
    elif disable_conversion_mapping:
        # For unfused MOE models, save state_dict directly to avoid HF's conversion mapping
        _save_model_without_conversion(model, save_dir, max_shard_size, safe_serialization)
    else:
        try:
            model.save_pretrained(save_dir, max_shard_size=max_shard_size, safe_serialization=safe_serialization)
        except ValueError as e:
            if hasattr(model, "generation_config"):
                setattr(model.generation_config, "do_sample", True)
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


def _save_model_without_conversion(
    model: nn.Module,
    save_dir: str,
    max_shard_size: str = "5GB",
    safe_serialization: bool = True,
):
    """Save model without HuggingFace's checkpoint conversion mapping.

    This is needed for models with unfused MOE experts where the conversion
    mapping would incorrectly try to fuse the weights back.
    """
    # Save config and generation_config
    if hasattr(model, "config") and model.config is not None:
        model.config.save_pretrained(save_dir)

    if hasattr(model, "generation_config") and model.generation_config is not None:
        try:
            model.generation_config.save_pretrained(save_dir)
        except Exception:
            pass

    # Get state dict directly without conversion
    state_dict = model.state_dict()

    # Try to use transformers' split_torch_state_dict_into_shards
    try:
        from huggingface_hub import split_torch_state_dict_into_shards

        state_dict_split = split_torch_state_dict_into_shards(state_dict, max_shard_size=max_shard_size)
        shards = state_dict_split.filename_to_tensors
        index = None
        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
    except ImportError:
        # Fallback: save as single file
        shards = {"model.safetensors" if safe_serialization else "pytorch_model.bin": state_dict}
        index = None

    # Save each shard
    for shard_file, shard_tensors in shards.items():
        shard_path = os.path.join(save_dir, shard_file)
        # Get the actual tensors from state_dict
        if isinstance(shard_tensors, (list, set)):
            shard = {k: state_dict[k] for k in shard_tensors}
        else:
            shard = shard_tensors

        if safe_serialization:
            # Use safetensors
            from safetensors.torch import save_file

            # safetensors requires all tensors to be contiguous
            shard = {k: v.contiguous() if not v.is_contiguous() else v for k, v in shard.items()}
            save_file(shard, shard_path)
        else:
            torch.save(shard, shard_path)

    # Save the index if sharded
    if index is not None:
        index_file = "model.safetensors.index.json" if safe_serialization else "pytorch_model.bin.index.json"
        with open(os.path.join(save_dir, index_file), "w") as f:
            json.dump(index, f, indent=2)


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
