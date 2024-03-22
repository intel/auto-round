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

import copy
import json
import os
from os.path import isdir, isfile, join
from typing import Dict, List, Optional, Union

import torch
from safetensors.torch import save_file as safe_save

from auto_round.export.register import register_format
from auto_round.utils import get_module, logger, quant_weight_w_scale, set_module

from .config import QuantConfig
from .model_wrapper import WeightOnlyLinear


@register_format("itrex")
def save_quantized_as_itrex(output_dir, inplace=True, **kwargs):
    """Save configure file and weights for CPU backend inference."""
    model = kwargs["model"]
    weight_config = kwargs["weight_config"]
    sym = kwargs["sym"]
    bits = kwargs["bits"]
    group_size = kwargs["group_size"]
    iters = kwargs["iters"]
    lr = kwargs["lr"]
    minmax_lr = kwargs["minmax_lr"]
    enable_minmax_tuning = kwargs["enable_minmax_tuning"]
    use_quant_input = kwargs["use_quant_input"]
    scale_dtype = kwargs["scale_dtype"]
    tokenizer = kwargs["tokenizer"]

    compressed_model = pack_model(model, weight_config, inplace=inplace)
    if output_dir is None:
        return compressed_model
    quantize_config = QuantConfig(
        bits=bits,
        group_size=group_size,
        sym=sym,
        iters=iters,
        lr=lr,
        minmax_lr=minmax_lr,
        enable_minmax_tuning=enable_minmax_tuning,
        use_quant_input=use_quant_input,
        scale_dtype=str(scale_dtype),
    )
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
    return compressed_model


def pack_model(
    model,
    weight_config: Union[str, dict],
    enable_full_range=False,
    compression_dtype=torch.int32,
    compression_dim=1,
    device="cpu",
    use_optimum_format=True,
    inplace=False,
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
        inplace (bool, optional): Compress the model in place, or copy the model and compress it.
    """
    if inplace:
        compressed_model = model
    else:
        compressed_model = copy.deepcopy(model)
    if isinstance(weight_config, str):
        with open(weight_config, "r") as f:
            q_config = json.load(f)
    else:
        q_config = weight_config
    for k, v in q_config.items():
        logger.info(f"Packing {k}")
        if "float" in v["data_type"]:
            continue
        dtype = v["data_type"]
        num_bits = v["bits"]
        group_size = v["group_size"]
        sym = v["sym"]
        scale_dtype = v["scale_dtype"]
        m = get_module(compressed_model, k)
        fp_weight = m.weight.data
        scale, zp = v["scale"], v["zp"]
        convert_dtype = scale_dtype
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=convert_dtype)
            zp = torch.tensor(zp, dtype=torch.int32)
        else:
            if not inplace:
                scale = scale.clone()
                zp = zp.clone()
            scale = scale.to(dtype=convert_dtype)
            zp = zp.to(dtype=torch.int32)
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

    return compressed_model
