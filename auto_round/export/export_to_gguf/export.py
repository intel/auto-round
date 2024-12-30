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
import sys
import shutil
import torch
from .convert import Model
from auto_round.utils import logger, LazyImport
from pathlib import Path

gguf = LazyImport("gguf")


FTYPE_MAP: dict[str, gguf.LlamaFileType] = {
        "f32": gguf.LlamaFileType.ALL_F32,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        "q4_0": gguf.LlamaFileType.MOSTLY_Q4_0,
        "q4_1": gguf.LlamaFileType.MOSTLY_Q4_1,
        "q4_k": gguf.LlamaFileType.MOSTLY_Q4_K_S,
        "auto": gguf.LlamaFileType.GUESSED,
    }

def save_quantized_as_gguf(output_dir, backend="gguf:q4_0", **kwargs):
    """Export the model to gguf format."""

    model = kwargs["model"]
    tokenizer = kwargs.get("tokenizer", None)
    config = model.config

    tmp_work_dir = Path(os.path.join(output_dir, 'tmp_dir'))
    if tokenizer is not None:
        tokenizer.save_pretrained(tmp_work_dir)
    config.save_pretrained(tmp_work_dir)

    with torch.inference_mode():
        hparams = Model.load_hparams(tmp_work_dir)
        model_architecture = hparams["architectures"][0]
        try:
            model_class = Model.from_model_architecture(model_architecture)
        except NotImplementedError:
            logger.error(f"Model {model_architecture} is not supported")
            sys.exit(1)
        model_class = Model.from_model_architecture(model_architecture)
        model_name = model.name_or_path.split('/')
        if len(model_name[-1]) == 0:
            model_name = model_name[-2]
        else:
            model_name = model_name[-1]
        
        output_type = backend.split(":")[-1]
        output_type = FTYPE_MAP.get(output_type.lower())

        layer_config = kwargs.get("layer_config")

        model_instance = model_class(model, dir_model=tmp_work_dir, ftype=output_type, fname_out=Path(output_dir),
                                     layer_config=layer_config, is_big_endian=False, model_name=model_name,
                                     split_max_tensors=False, split_max_size=0, dry_run=False,
                                     small_first_shard=False)
        model_instance.write()
        logger.info(f"Model successfully exported to {model_instance.fname_out}")
    
    shutil.rmtree(tmp_work_dir, ignore_errors=True)

    return model