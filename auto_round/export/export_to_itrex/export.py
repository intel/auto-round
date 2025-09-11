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
from typing import Dict, List, Optional, Union

import torch
import transformers

from auto_round.export.register import register_format
from auto_round.logger import logger
from auto_round.utils import check_to_quantized, detect_device, get_module, set_module

from .config import QuantConfig
from .model_wrapper import WeightOnlyLinear


def quant_weight_w_scale(weight, scale, zp, group_size=-1, device="cpu"):
    """Quant and dequant tensor with group size.

    Args:
        weight: input weight
        scale: scale
        zp: zero point
        group_size (int, optional): how many elements share one scale/zp. Defaults to -1.

    Returns:
        output: int weight.
    """
    scale = scale.to(device)
    if zp is not None and isinstance(zp, torch.Tensor):
        zp = zp.to(device)
    if group_size == -1:
        return torch.round(weight / scale) if zp is None else torch.round(weight / scale + zp)
    int_weight = torch.zeros(weight.shape).to(device)
    leng = weight.shape[1] // group_size
    tail_flag = False if weight.shape[1] % group_size == 0 else True
    for i in range(leng):
        int_weight_tmp = weight[:, i * group_size : (i + 1) * group_size] / scale[:, i].unsqueeze(1)
        if zp is not None:
            int_weight_tmp += zp[:, i].unsqueeze(1)
        int_weight[:, i * group_size : (i + 1) * group_size] = torch.round(int_weight_tmp)
    if tail_flag:
        int_weight_tmp = weight[:, leng * group_size :] / scale[:, -1].unsqueeze(1)
        if zp is not None:
            int_weight_tmp += zp[:, -1].unsqueeze(1)
        int_weight[:, leng * group_size :] = torch.round(int_weight_tmp)
    return int_weight


def save_quantized_as_itrex(output_dir, inplace=True, **kwargs):
    """Save configure file and weights for CPU backend inference."""
    model = kwargs["model"]
    layer_config = kwargs["layer_config"]
    sym = kwargs["sym"]
    bits = kwargs["bits"]
    group_size = kwargs["group_size"]
    iters = kwargs["iters"]
    lr = kwargs["lr"]
    minmax_lr = kwargs["minmax_lr"]
    enable_minmax_tuning = kwargs["enable_minmax_tuning"]
    enable_quanted_input = kwargs["enable_quanted_input"]
    scale_dtype = kwargs["scale_dtype"]
    tokenizer = kwargs["tokenizer"]
    safe_serialization = True if "safe_serialization" not in kwargs.keys() else kwargs["safe_serialization"]

    compressed_model = pack_model(model, layer_config, inplace=inplace)
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
        enable_quanted_input=enable_quanted_input,
        scale_dtype=scale_dtype,
    )
    if quantize_config is not None:
        quantize_config.post_init()
        config = compressed_model.config
        setattr(config, "quantization_config", quantize_config.to_dict())
        config.save_pretrained(output_dir)
        quantize_config.save_pretrained(output_dir)
    try:
        compressed_model.save_pretrained(output_dir, safe_serialization=safe_serialization)
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(output_dir)
        logger.info("Saved config file and weights of quantized model to {}.".format(output_dir))
    except IOError as e:  # pragma: no cover
        logger.error("Fail to save configure file and weights due to {}.".format(e))
    return compressed_model


def save_quantized_as_itrex_xpu(output_dir, inplace=True, **kwargs):
    """Save configure file and weights for XPU backend inference."""
    model = kwargs["model"]
    layer_config = kwargs["layer_config"]
    sym = kwargs["sym"]
    bits = kwargs["bits"]
    group_size = kwargs["group_size"]
    iters = kwargs["iters"]
    lr = kwargs["lr"]
    minmax_lr = kwargs["minmax_lr"]
    enable_minmax_tuning = kwargs["enable_minmax_tuning"]
    enable_quanted_input = kwargs["enable_quanted_input"]
    scale_dtype = kwargs["scale_dtype"]
    tokenizer = kwargs.get("tokenizer", None)
    processor = kwargs.get("processor", None)

    compressed_model = pack_model(inplace=inplace, **kwargs)
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
        enable_quanted_input=enable_quanted_input,
        scale_dtype=scale_dtype,
        export_to_xpu=True,
    )
    if quantize_config is not None:
        quantize_config.post_init_xpu()
        quantize_config.remove_redundant_parameters()
        config = compressed_model.config
        setattr(config, "quantization_config", quantize_config.to_dict())
        config.save_pretrained(output_dir)
        quantize_config.save_pretrained(output_dir)
    try:
        compressed_model.save_pretrained(output_dir, safe_serialization=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        if processor is not None:
            processor.save_pretrained(output_dir)
        logger.info("Saved config file and weights of quantized model to {}.".format(output_dir))
    except IOError as e:  # pragma: no cover
        logger.error("Fail to save configure file and weights due to {}.".format(e))
    return compressed_model


def pack_model(
    model,
    layer_config: Union[str, dict],
    enable_full_range=False,
    compression_dtype=torch.int32,
    compression_dim=1,
    device="cpu",
    use_optimum_format=True,
    inplace=False,
    **kwargs,
):
    """Convert Linear to WeightOnlyLinear for low memory inference.

    Args:
        layer_config (str|dict): qconfig dict or Path of qconfig.json.
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
    # Due to XPU doesn't support tuning yet
    device = "cpu" if device == "xpu" else device
    if model.device.type == "meta":
        model = model.to(device)
    if inplace:
        compressed_model = model
    else:
        compressed_model = copy.deepcopy(model)
    if isinstance(layer_config, str):
        with open(layer_config, "r") as f:
            q_config = json.load(f)
    else:
        q_config = layer_config

    for k, v in q_config.items():
        if check_to_quantized(v) is False:
            continue
        logger.info(f"Packing {k}")
        dtype = v["data_type"]
        bits = v["bits"]
        group_size = v["group_size"]
        sym = v["sym"]
        scale_dtype = v["scale_dtype"]
        m = get_module(compressed_model, k)
        fp_weight = m.weight.data
        scale, zp = m.scale, m.zp
        if isinstance(zp, int | float):
            zp = torch.full_like(scale, zp)
        convert_dtype = scale_dtype
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=convert_dtype)
            zp = torch.tensor(zp, dtype=torch.int32)
        else:
            if not inplace:
                scale = scale.clone()
                zp = zp.clone() if isinstance(zp, torch.Tensor) else zp
            else:
                scale = scale.to(dtype=convert_dtype)
                zp = zp.to(dtype=torch.int32) if isinstance(zp, torch.Tensor) else zp
        if isinstance(m, transformers.pytorch_utils.Conv1D):
            fp_weight = fp_weight.t_().contiguous()
        int_weight = quant_weight_w_scale(fp_weight, scale, zp, group_size, fp_weight.device)
        if isinstance(m, torch.nn.Linear):
            in_features = m.in_features
            out_features = m.out_features
        elif isinstance(m, transformers.pytorch_utils.Conv1D):
            in_features = m.weight.shape[0]
            out_features = m.weight.shape[1]
        int_weight = int_weight.type(torch.int32)
        new_module = WeightOnlyLinear(
            in_features,
            out_features,
            bits,
            group_size,
            dtype=dtype,
            scale_dtype=scale_dtype,
            zp=zp is not None,
            bias=m.bias is not None,
            device=device,
            compression_dtype=compression_dtype,
            compression_dim=compression_dim,
            use_optimum_format=use_optimum_format,  # xpu is False
        )
        new_module.pack(int_weight, scale, zp, m.bias)
        set_module(compressed_model, k, new_module)
    return compressed_model
