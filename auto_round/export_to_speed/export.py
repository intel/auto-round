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

import torch
from logging import getLogger

logger = getLogger(__name__)
import os
from typing import Dict, List, Optional, Union
from safetensors.torch import save_file as safe_save
from os.path import join, isfile, isdir
import copy
import json
from .config import QuantConfig
from .model_wrapper import WeightOnlyLinear
from ..utils import quant_weight_w_scale, get_module, set_module


def compress_model(
        model,
        weight_config:Union[str, dict],
        enable_full_range=False,
        compression_dtype=torch.int32,
        compression_dim=1,
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
        device (str, optional): choose device for compression. Defaults to cpu.
        use_optimum_format (bool, optional): use the popular huggingface compression format.
            1: compression_dim: weight = 1, zeros = 0 and both are transposed.
            2: zeros -= 1 before compression. Why we need it?
            3: g_idx: use same number for one group instead of recording the channel order.
            4. parameter name changed, such as 'packed_weight' -> 'qweight'.
            5. zeros is always needed even for sym.
    """
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
        # scale = torch.tensor(v["scale"], dtype=scale_dtype)
        scale, zp = v["scale"], v["zp"]
        convert_dtype=torch.float32 if fp_weight.device.type == "cpu" else scale_dtype
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=convert_dtype)
            zp = None if scheme == "sym" else torch.tensor(zp, dtype=torch.int32)
        else:
            scale = scale.to(dtype=convert_dtype)
            zp = None if scheme == "sym" else zp.to(dtype=torch.int32)
        int_weight = quant_weight_w_scale(fp_weight, scale, zp, group_size, fp_weight.device)
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
    
    quantize_config = QuantConfig(bits=num_bits, sym=(scheme=="sym"), group_size=group_size)
    return compressed_model, quantize_config


def save_compressed_model(model,
                          weight_config:Union[str, dict],
                          output_dir,
                          tokenizer=None):
        """Save configure file and weights for CPU backend inference."""
        
        compressed_model, quantize_config = compress_model(model, weight_config)
        if quantize_config is not None:
            config = compressed_model.config
            setattr(config, "quantization_config", quantize_config.to_dict())
            config.save_pretrained(output_dir)
            quantize_config.save_pretrained(output_dir)
            
        try:
            compressed_model.save_pretrained(output_dir, safe_serialization=True)
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)
            logger.info("Saved config file and weights of quantized model to {}.".format(output_dir))
        except IOError as e:  # pragma: no cover
            logger.error("Fail to save configure file and weights due to {}.".format(e))
