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

import torch

from auto_round.export.export_to_gguf.convert import ModelBase, ModelType, get_model_architecture
from auto_round.utils import (
    LazyImport,
    check_to_quantized,
    clear_memory,
    flatten_list,
    get_block_names,
    get_module,
    logger,
)

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
    output_dir, model, layer_config, backend="gguf:q4_0", low_cpu_mem_usage=False, model_type=ModelType.TEXT
):
    tmp_work_dir = Path(os.path.join(output_dir, TMP_DIR_NAME))
    with torch.inference_mode():
        hparams = ModelBase.load_hparams(tmp_work_dir)
        model_architecture = get_model_architecture(hparams=hparams, model_type=model_type)
        # if "architectures" in hparams:
        #     model_architecture = hparams["architectures"][0]
        # else:
        #     model_architecture = type(model).__name__
        #     if model_architecture not in ModelBase._model_classes:
        #         if model_architecture.replace("CausalLM", "ConditionalGeneration") in ModelBase._model_classes:
        #             model_architecture = model_architecture.replace("CausalLM", "ConditionalGeneration")
        #         elif model_architecture.replace("ConditionalGeneration", "CausalLM") in ModelBase._model_classes:
        #             model_architecture = model_architecture.replace("ConditionalGeneration", "CausalLM")
        try:
            model_class = ModelBase.from_model_architecture(model_architecture, model_type=model_type)
        except NotImplementedError:
            logger.error(f"Model {model_architecture} is not supported")
            sys.exit(1)
        model_class = ModelBase.from_model_architecture(model_architecture, model_type=model_type)
        model_name = model.name_or_path.split("/")
        if len(model_name[-1]) == 0:
            model_name = model_name[-2]
        else:
            model_name = model_name[-1]

        output_type = backend.split(":")[-1]
        if output_type.lower() not in FTYPE_MAP:
            raise TypeError(f"{output_type} type is not supported")
        output_type = FTYPE_MAP.get(output_type.lower())

        model_instance = model_class(
            model,
            layer_config,
            dir_model=tmp_work_dir,
            ftype=output_type,
            fname_out=Path(output_dir),
            low_cpu_mem_usage=low_cpu_mem_usage,  # pylint: disable=E0401
            is_big_endian=False,
            model_name=model_name,
            split_max_tensors=False,
            split_max_size=0,
            dry_run=False,
            small_first_shard=False,
        )
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
    model_type=ModelType.TEXT,
):
    """Export the model to gguf format."""
    global gguf_model_instance_global
    tmp_work_dir = Path(os.path.join(output_dir, TMP_DIR_NAME))
    if output_dir is not None and os.path.exists(output_dir) and not os.path.exists(tmp_work_dir):
        logger.warning_once(f"{output_dir} already exists, this may cause model conflict")
    tmp_work_dir = Path(os.path.join(output_dir, TMP_DIR_NAME))
    if "gguf_model_instance_global" not in globals():
        config = model.config
        config.save_pretrained(tmp_work_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(tmp_work_dir)
        if processor is not None:
            processor.save_pretrained(tmp_work_dir)
        if image_processor is not None:
            image_processor.save_pretrained(tmp_work_dir)

        gguf_model_instance_global = [
            create_model_class(
                output_dir, model, layer_config, backend, low_cpu_mem_usage=True, model_type=ModelType.TEXT
            )
        ]
        if model_type == ModelType.MMPROJ:
            gguf_model_instance_global.append(
                create_model_class(
                    output_dir, model, layer_config, backend, low_cpu_mem_usage=True, model_type=ModelType.MMPROJ
                )
            )
        if not hasattr(model, "last_layer_name_to_block_name"):
            block_name_to_last_layer_name = {}
            block_names = get_block_names(model, quant_vision=True)
            block_names_flatten = flatten_list(block_names)
            for n, m in model.named_modules():
                if not check_to_quantized(m):
                    continue
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
    if name in model.last_layer_name_to_block_name:
        ##packing block
        for gguf_model in gguf_model_instance_global:
            gguf_model.current_packing_block = model.last_layer_name_to_block_name[name]
            gguf_model.prepare_tensors()

        block = get_module(model, model.last_layer_name_to_block_name[name])
        for n, m in block.named_modules():
            if hasattr(m, "weight"):
                m.weight = None
            if hasattr(m, "bias"):
                m.bias = None
        clear_memory()
        model.last_layer_name_to_block_name.pop(name)
        if len(model.last_layer_name_to_block_name) == 0:
            for gguf_model in gguf_model_instance_global:
                gguf_model.current_packing_block = None


@torch.inference_mode()
def save_quantized_as_gguf(output_dir, backend="gguf:q4_0", layer_config=None, vlm=False, **kwargs):
    """Export the model to gguf format."""
    tmp_work_dir = Path(os.path.join(output_dir, TMP_DIR_NAME))
    if output_dir is not None and os.path.exists(output_dir) and not os.path.exists(tmp_work_dir):
        logger.warning(f"{output_dir} already exists, this may cause model conflict")

    st = time.time()
    global gguf_model_instance_global

    model = kwargs["model"]
    if "gguf_model_instance_global" not in globals():
        config = model.config
        config.save_pretrained(tmp_work_dir)
        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is not None:
            tokenizer.save_pretrained(tmp_work_dir)
        processor = kwargs.get("processor", None)
        if processor is not None:
            processor.save_pretrained(tmp_work_dir)
        image_processor = kwargs.get("image_processor", None)
        if image_processor is not None:
            image_processor.save_pretrained(tmp_work_dir)

        gguf_model_instance_global = [
            create_model_class(output_dir, model, layer_config, backend, model_type=ModelType.TEXT)
        ]
        if vlm:
            gguf_model_instance_global.append(
                create_model_class(output_dir, model, layer_config, backend, model_type=ModelType.MMPROJ)
            )

    for gguf_model in gguf_model_instance_global:
        gguf_model.write()
        rt = time.time() - st
        logger.info(f"Model successfully exported to {gguf_model.fname_out}, running time={rt}")
    del gguf_model_instance_global
    shutil.rmtree(tmp_work_dir, ignore_errors=True)

    return model
