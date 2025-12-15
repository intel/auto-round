# Copyright (c) 2024 Intel Corporation
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

import os
import shutil
import sys
import time
from pathlib import Path

import requests
import torch

from auto_round.logger import logger

# from auto_round.export.export_to_gguf.convert_hf_to_gguf import ModelBase, ModelType, get_model_architecture
from auto_round.utils import (
    LazyImport,
    check_to_quantized,
    clear_memory,
    download_or_get_path,
    flatten_list,
    get_block_names,
    get_gguf_architecture,
    get_module,
)

convert_hf_to_gguf = LazyImport("auto_round.export.export_to_gguf.convert_hf_to_gguf")
from auto_round.export.export_to_gguf.convert import wrapper_model_instance
from auto_round.export.export_to_gguf.special_handle import handle_special_model

TMP_DIR_NAME = "tmp_dir"

gguf = LazyImport("gguf")

FTYPE_MAP: dict[str, gguf.LlamaFileType] = {
    "f32": gguf.LlamaFileType.ALL_F32,
    "f16": gguf.LlamaFileType.MOSTLY_F16,
    "bf16": gguf.LlamaFileType.MOSTLY_BF16,
    "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
    "q4_1": gguf.LlamaFileType.MOSTLY_Q4_1,
    "q5_0": gguf.LlamaFileType.MOSTLY_Q5_0,
    "q5_1": gguf.LlamaFileType.MOSTLY_Q5_1,
    "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
    "q2_k_s": gguf.LlamaFileType.MOSTLY_Q2_K_S,
    "q3_k_s": gguf.LlamaFileType.MOSTLY_Q3_K_S,
    "q3_k_m": gguf.LlamaFileType.MOSTLY_Q3_K_M,
    "q3_k_l": gguf.LlamaFileType.MOSTLY_Q3_K_L,
    "q4_k_s": gguf.LlamaFileType.MOSTLY_Q4_K_S,
    "q4_k_m": gguf.LlamaFileType.MOSTLY_Q4_K_M,
    "q5_k_s": gguf.LlamaFileType.MOSTLY_Q5_K_S,
    "q5_k_m": gguf.LlamaFileType.MOSTLY_Q5_K_M,
    "q6_k": gguf.LlamaFileType.MOSTLY_Q6_K,
    "q6_k_s": gguf.LlamaFileType.MOSTLY_Q6_K,
    "auto": gguf.LlamaFileType.GUESSED,
}


def create_model_class(
    output_dir,
    model,
    layer_config,
    backend="gguf:q4_0",
    low_cpu_mem_usage=False,
    model_type=convert_hf_to_gguf.ModelType.TEXT,
    device="cpu",
):
    tmp_work_dir = model.name_or_path
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.isdir(tmp_work_dir):
        tmp_work_dir = download_or_get_path(tmp_work_dir)
    with torch.inference_mode():
        model_architecture = get_gguf_architecture(tmp_work_dir, model_type=model_type)
        try:
            model_class = convert_hf_to_gguf.ModelBase.from_model_architecture(
                model_architecture, model_type=model_type
            )
        except NotImplementedError:
            logger.error(f"Model {model_architecture} is not supported to export gguf format.")
            sys.exit(1)
        model_name = model.name_or_path.split("/")
        if len(model_name[-1]) == 0:
            model_name = model_name[-2]
        else:
            model_name = model_name[-1]

        output_type = backend.split(":")[-1]
        if output_type.lower() not in FTYPE_MAP:
            raise TypeError(f"{output_type} type is not supported")
        output_type = FTYPE_MAP.get(output_type.lower())

        hparams = convert_hf_to_gguf.ModelBase.load_hparams(Path(tmp_work_dir), "mistral" in model.config.model_type)
        hparams.pop("quantization_config", None)
        model_instance = model_class(
            dir_model=Path(tmp_work_dir),
            ftype=output_type,
            hparams=hparams,
            fname_out=Path(output_dir),
            is_big_endian=False,
            model_name=model_name,
            split_max_tensors=False,
            split_max_size=0,
            dry_run=False,
            small_first_shard=False,
        )
        model_instance = wrapper_model_instance(
            model_instance, model=model, layer_config=layer_config, low_cpu_mem_usage=low_cpu_mem_usage, device=device
        )
        model_instance = handle_special_model(model_instance, model_architecture)
    return model_instance


@torch.inference_mode()
def pack_gguf_layer(
    name,
    model,
    backend,
    output_dir,
    layer_config,
    tokenizer,
    processor=None,
    image_processor=None,
    model_type=convert_hf_to_gguf.ModelType.TEXT,
    device="cpu",
):
    """Export the model to gguf format."""
    global gguf_model_instance_global
    # if output_dir is not None and os.path.exists(output_dir):
    #     logger.warning_once(f"{output_dir} already exists, this may cause model conflict")
    if "gguf_model_instance_global" not in globals():
        config = model.config

        gguf_model_instance_global = [
            create_model_class(
                output_dir,
                model,
                layer_config,
                backend,
                low_cpu_mem_usage=True,
                model_type=convert_hf_to_gguf.ModelType.TEXT,
            )
        ]
        if model_type == convert_hf_to_gguf.ModelType.MMPROJ:
            gguf_model_instance_global.append(
                create_model_class(
                    output_dir,
                    model,
                    layer_config,
                    backend,
                    low_cpu_mem_usage=True,
                    model_type=convert_hf_to_gguf.ModelType.MMPROJ,
                    device=device,
                )
            )

        if not hasattr(model, "last_layer_name_to_block_name"):
            block_name_to_last_layer_name = {}
            block_names = get_block_names(model, quant_vision=True)
            block_names_flatten = flatten_list(block_names)
            all_qlayer_name = []
            for n, m in model.named_modules():
                if not check_to_quantized(m):
                    continue
                all_qlayer_name.append(n)
                for block_name in block_names_flatten:
                    block_name_split = block_name.split(".")
                    name_split = n.split(".")
                    if (
                        len(name_split) < len(block_name_split)
                        or name_split[: len(block_name_split)] != block_name_split
                    ):
                        continue
                    block_name_to_last_layer_name[block_name] = n
            last_layer_name_to_block_name = {v: k for k, v in block_name_to_last_layer_name.items()}
            model.last_layer_name_to_block_name = last_layer_name_to_block_name
            names_in_blocks = []
            for block_name in block_names_flatten:
                block = get_module(model, block_name)
                for n, m in block.named_modules():
                    if check_to_quantized(m):
                        names_in_blocks.append(m.tmp_name)

    if name in model.last_layer_name_to_block_name:
        # Packing block
        block = get_module(model, model.last_layer_name_to_block_name[name])
        for gguf_model in gguf_model_instance_global:
            gguf_model.current_packing_block = model.last_layer_name_to_block_name[name]
            gguf_model.prepare_tensors()

        for n, m in block.named_modules():
            if hasattr(m, "weight"):
                m.weight = None
            if hasattr(m, "bias"):
                m.bias = None
        model.last_layer_name_to_block_name.pop(name)
        if len(model.last_layer_name_to_block_name) == 0:
            for gguf_model in gguf_model_instance_global:
                gguf_model.current_packing_block = None


@torch.inference_mode()
def save_quantized_as_gguf(output_dir, backend="gguf:q4_0", layer_config=None, vlm=False, device="cpu", **kwargs):
    """Export the model to gguf format."""
    st = time.time()
    global gguf_model_instance_global

    model = kwargs["model"]
    if "gguf_model_instance_global" not in globals():
        gguf_model_instance_global = [
            create_model_class(
                output_dir, model, layer_config, backend, model_type=convert_hf_to_gguf.ModelType.TEXT, device=device
            )
        ]
        if vlm:
            gguf_model_instance_global.append(
                create_model_class(
                    output_dir,
                    model,
                    layer_config,
                    backend,
                    model_type=convert_hf_to_gguf.ModelType.MMPROJ,
                    device=device,
                )
            )

    for gguf_model in gguf_model_instance_global:
        gguf_model.write()
        rt = time.time() - st
        logger.info(f"Model successfully exported to {gguf_model.fname_out}, running time={rt}")
    del gguf_model_instance_global

    return model
