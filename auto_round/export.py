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
import copy
import json


def compress_model(
        model,
        weight_config:Union[str, dict],
        enable_full_range=False,
        compression_dtype=torch.int32,
        compression_dim=1,
        scale_dtype=torch.float32,
        device="cpu",
        use_optimum_format=True,
    ):
    """Convert Linear to WeightOnlyLinear for low memory inference.

    Args:
        weight_config (str|dict): qconfig dict or Path of qconfig.json.
        enable_full_range (bool, optional): Whether to leverage the full compression range
                                            under symmetric quantization. Defaults to False.
        compression_dtype (torch.Tensor, optional): The target dtype after comoression.
                                                    Defaults to torch.int32.
        compression_dim (int, optional): Select from [0, 1], 0 is output channel,
                                            1 is input channel. Defaults to 1.
        scale_dtype (torch.Tensor, optional): Use float32 or float16.
                                                Defaults to torch.float32.
        device (str, optional): choose device for compression. Defaults to cpu.
        use_optimum_format (bool, optional): use the popular huggingface compression format.
            1: compression_dim: weight = 1, zeros = 0 and both are transposed.
            2: zeros -= 1 before compression. Why we need it?
            3: g_idx: use same number for one group instead of recording the channel order.
            4. parameter name changed, such as 'packed_weight' -> 'qweight'.
            5. zeros is always needed even for sym.
    """
    from .model_wrapper import WeightOnlyLinear
    from .autoround import quant_weight_w_scale, get_module, set_module
    compressed_model = copy.deepcopy(model)
    if isinstance(weight_config, str):
        with open(weight_config, "r") as f:
            q_config = json.load(f)
    else:
        q_config = weight_config
    for k, v in q_config.items():
        logger.info(f"Compressing {k} on device {device}")
        if "float" in v["data_type"]:
            continue
        dtype = v["data_type"]
        num_bits = v["bits"]
        group_size = v["group_size"]
        scheme = v["scheme"]
        scale_dtype = v["scale_dtype"]
        m = get_module(compressed_model, k)
        fp_weight = m.weight.data
        scale = torch.tensor(v["scale"], dtype=scale_dtype)
        zp = None if scheme == "sym" else torch.tensor(v["zp"], dtype=torch.int32)
        int_weight = quant_weight_w_scale(fp_weight, scale, zp, group_size)
        int_weight = int_weight.type(torch.int32)
        new_module = WeightOnlyLinear(
            m.in_features,
            m.out_features,
            num_bits,
            group_size,
            dtype=dtype,
            scale_dtype=scale_dtype,
            zp=zp is not None,
            bias=m.bias is not None,
            device=device,
            use_optimum_format=True,
        )
        new_module.pack(int_weight, scale, zp, m.bias)
        set_module(compressed_model, k, new_module)
    return compressed_model


def save_compressed_model(model, output_dir, quantize_config=None, tokenizer=None):
        """Save configure file and weights for CPU backend inference."""
        
        if quantize_config is not None:
            config = model.config
            setattr(config, "quantization_config", quantize_config.to_dict())
            config.save_pretrained(output_dir)
            quantize_config.save_pretrained(output_dir)
            
        try:
            model.save_pretrained(output_dir, safe_serialization=True)
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)
            logger.info("Saved config file and weights of quantized model to {}.".format(output_dir))
        except IOError as e:  # pragma: no cover
            logger.error("Fail to save configure file and weights due to {}.".format(e))


def save_quantized_to_autogptq(model, save_dir: str, bits=4, group_size=128, sym=False, iters=200, lr=5e-3,
                               minmax_lr=5e-3,
                               enable_minmax_tuning=True, use_quant_input=True, use_safetensors: bool = True,
                               safetensors_metadata: Optional[Dict[str, str]] = None):
    """save quantized model and configs to local disk for cuda """
    os.makedirs(save_dir, exist_ok=True)
    model.to("cpu")

    model_base_name = f"autoround-model-{str(bits)}bit-{str(group_size)}g"
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
        safetensors_metadata['auto_round_version'] = str(__version__)
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

    model.config.save_pretrained(save_dir)
    from auto_gptq.modeling._base import BaseQuantizeConfig
    quantization_config = BaseQuantizeConfig(bits=bits, group_size=group_size, desc_act=False, sym=sym,
                                             true_sequential=False, static_groups=False,
                                             model_file_base_name=model_base_name)
    quantization_config.model_file_base_name = model_base_name

    quantization_config.save_pretrained(save_dir)

