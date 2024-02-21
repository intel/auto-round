# Copyright (c) 2023 Intel Corporation
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


# MIT License
#
# Copyright (c) 2023 潘其威(William)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import torch
from logging import getLogger

logger = getLogger(__name__)
import os
from typing import Dict, List, Optional, Union
from safetensors.torch import save_file as safe_save
from os.path import join, isfile, isdir
import json


def save_quantized_to_autogptq(model, save_dir: str, bits=4, group_size=128, sym=False, iters=200, lr=5e-3,
                               minmax_lr=5e-3,
                               enable_minmax_tuning=True, use_quant_input=True, use_safetensors: bool = True,
                               safetensors_metadata: Optional[Dict[str, str]] = None, modules_in_block_to_quantize=None):
    """save quantized model and configs to local disk for cuda """
    os.makedirs(save_dir, exist_ok=True)
    model.to("cpu")

    model_base_name = f"model"
    if use_safetensors:
        model_save_name = model_base_name + ".safetensors"
        state_dict = model.state_dict()
        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        if safetensors_metadata is None:
            safetensors_metadata = {}
        elif not isinstance(safetensors_metadata, dict):
            raise TypeError("safetensors_metadata must be a dictionary.")
        else:
            logger.debug(f"Received safetensors_metadata: {safetensors_metadata}")
            new_safetensors_metadata = {}
            converted_keys = False
            for key, value in safetensors_metadata.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    converted_keys = True
                    try:
                        new_key = str(key)
                        new_value = str(value)
                    except Exception as e:
                        raise TypeError(
                            f"safetensors_metadata: both keys and values must be strings and an error occured when trying to convert them: {e}")
                    if new_key in new_safetensors_metadata:
                        logger.warning(
                            f"After converting safetensors_metadata keys to strings, the key '{new_key}' is duplicated. Ensure that all your metadata keys are strings to avoid overwriting.")
                    new_safetensors_metadata[new_key] = new_value
            safetensors_metadata = new_safetensors_metadata
            if converted_keys:
                logger.debug(
                    f"One or more safetensors_metadata keys or values had to be converted to str(). Final safetensors_metadata: {safetensors_metadata}")

        # Format is required to enable Accelerate to load the metadata
        # otherwise it raises an OSError
        safetensors_metadata['format'] = "pt"

        # Store the quantization configuration as safetensors metadata
        from auto_round import __version__
        safetensors_metadata['autoround_version'] = str(__version__)
        safetensors_metadata['bits'] = str(bits)
        safetensors_metadata['group_size'] = str(group_size)
        safetensors_metadata['iters'] = str(iters)
        safetensors_metadata['lr'] = str(lr)
        safetensors_metadata['minmax_lr'] = str(minmax_lr)
        safetensors_metadata['enable_minmax_tuning'] = str(enable_minmax_tuning)
        safetensors_metadata['use_quant_input'] = str(use_quant_input)
        safe_save(state_dict, join(save_dir, model_save_name), safetensors_metadata)
    else:
        model_save_name = model_base_name + ".bin"
        torch.save(model.state_dict(), join(save_dir, model_save_name))

    from auto_gptq.modeling._base import BaseQuantizeConfig

    quantization_config = BaseQuantizeConfig(bits=bits, group_size=group_size, desc_act=False, sym=sym,
                                             true_sequential=False, static_groups=False,
                                             model_file_base_name=model_base_name)
    quantization_config.model_file_base_name = model_base_name

    config_dict = quantization_config.to_dict()
    config_dict["quant_method"] = "intel/auto-round"
    config_dict['autoround_version'] = __version__
    config_dict['iters'] = iters
    config_dict['lr'] = lr
    config_dict['minmax_lr'] = minmax_lr
    config_dict['enable_minmax_tuning'] = enable_minmax_tuning
    config_dict['use_quant_input'] = use_quant_input
    if modules_in_block_to_quantize is not None:
        config_dict["modules_in_block_to_quantize"] = modules_in_block_to_quantize

    with open(join(save_dir, "quantize_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    config_dict["quant_method"] = "gptq"  ##hf transformers could only recognize this value
    model.config.quantization_config = config_dict
    model.config.save_pretrained(save_dir)
