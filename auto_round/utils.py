
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
import copy
from torch.amp import autocast
from collections import UserDict


def is_optimum_habana_available():
    import importlib
    from transformers.utils.import_utils import is_optimum_available

    return is_optimum_available() and importlib.util.find_spec("optimum.habana") != None

try:
    import habana_frameworks.torch.hpu as hthpu
    import habana_frameworks.torch.core as htcore
    if is_optimum_habana_available():
        is_hpu_available = True
    else:
        print("Should install optimum-habana when the environment has habana frameworks")
        is_hpu_available = False
except ImportError:
    is_hpu_available = False

def round_ste(x: torch.Tensor):
    """Straight-Through Estimator for rounding.
    This function is adapted from omniquant.

    Args:
        x: torch.Tensor

    Returns:
        torch.Tensor
    """
    return (x.round() - x).detach() + x


def quant_weight_asym(weight, num_bits=4, v=0, min_scale=0, max_scale=0, scale_dtype=torch.float16):
    """Quantizes and dequantizes weight asymmetrically.

    Args:
        weight: Tensor containing the weight to be quantized
        num_bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for weight
        max_scale: Maximum scale coefficient for weight

    Returns:
        Quantized and dequantized weight, scale, zero-point
    """
    maxq = torch.tensor(2 ** num_bits - 1)
    zeros = torch.zeros(weight.shape[0], device=weight.device, dtype=scale_dtype)
    # zeros = torch.zeros(weight.shape[0], device=weight.device)
    if isinstance(min_scale, torch.Tensor):
        wmin_tmp = torch.minimum(weight.min(1)[0], zeros)
        wmax_tmp = torch.maximum(weight.max(1)[0], zeros)
        wmin_tmp *= min_scale + 1.0
        wmax_tmp *= max_scale + 1.0
        wmax = torch.maximum(wmax_tmp, wmin_tmp)
        wmin = torch.minimum(wmax_tmp, wmin_tmp)
    else:
        wmin = torch.minimum(weight.min(1)[0], zeros)
        wmax = torch.maximum(weight.max(1)[0], zeros)
        
    tmp = (wmin == 0) & (wmax == 0)
    wmin[tmp] = -1
    wmax[tmp] = +1
    scale = ((wmax - wmin) / maxq).to(scale_dtype)
    zp = round_ste(-wmin / scale)
    scale = scale.unsqueeze(dim=-1)
    zp = zp.unsqueeze(dim=-1)
    int_w = round_ste(weight / scale + v)
    q = torch.clamp(int_w + zp, 0, maxq)
    return scale * (q - zp), scale, zp


def quant_weight_sym(weight, num_bits=4, v=0, min_scale=0, max_scale=0, scale_dtype=torch.float16):
    """Quantizes and dequantizes weight symmetrically.

    Args:
        weight: Tensor containing the weight to be quantized
        num_bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for weight
        max_scale: Maximum scale coefficient for weight

    Returns:
        Quantized and dequantized weight, scale, zero-point
    """
    maxq = torch.tensor(2 ** (num_bits - 1) - 1).to(weight.device)
    minq = torch.tensor(-(2 ** (num_bits - 1))).to(weight.device)
    if num_bits == 1:
        maxq = torch.tensor(2 ** (num_bits - 1))
        minq = torch.tensor(2 ** (num_bits - 1) - 1)

    wmax = torch.abs(weight).max(1)[0]
    wmax *= 1 + max_scale
    tmp = wmax == 0
    wmax[tmp] = +1
    scale = (wmax / ((maxq - minq) / 2)).to(scale_dtype)
    scale.unsqueeze_(dim=-1)
    q = torch.clamp(round_ste(weight / scale + v), minq, maxq)
    return scale * q, scale, None


def quant_weight_actor(weight, num_bits, scheme, v, min_scale, max_scale, scale_dtype=torch.float16):
    """Quantizes and dequantizes weight symmetrically or asymmetrically .

    Args:
        weight: Tensor containing the weight to be quantized
        num_bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        scheme: Sym or asym
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for weight
        max_scale: Maximum scale coefficient for weight

    Returns:
        Quantized and dequantized weight, scale, zero-point
    """
    assert num_bits > 0, "num_bits should be larger than 0"
    if scheme == "sym":
        return quant_weight_sym(weight, num_bits, v, min_scale, max_scale, scale_dtype)
    else:
        return quant_weight_asym(weight, num_bits, v, min_scale, max_scale, scale_dtype)


def quant_weight(weight, num_bits=4, group_size=-1, scheme="asym", v=0, min_scale=0, max_scale=0, scale_dtype=torch.float16):
    """Quantizes and dequantizes weight, handing the group size issue .

    Args:
        weight: Tensor containing the weight to be quantized
        num_bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: The number of elements shares scale and zero point
        scheme: Sym or asym
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for weight
        max_scale: Maximum scale coefficient for weight

    Returns:
        Quantized and dequantized weight, scale, zero-point
    """
    if group_size == -1 or weight.shape[1] < group_size:
        return quant_weight_actor(weight, num_bits, scheme=scheme, v=v, 
                                  min_scale=min_scale, max_scale=max_scale, scale_dtype=scale_dtype)
    orig_shape = weight.shape
    if weight.shape[1] % group_size == 0:
        weight = weight.reshape(-1, group_size)
        if isinstance(v, torch.Tensor):
            v = v.reshape(-1, group_size)

        weight, scale, zp = quant_weight_actor(
            weight, num_bits, scheme=scheme, v=v, min_scale=min_scale, max_scale=max_scale, scale_dtype=scale_dtype
        )
        weight = weight.reshape(orig_shape)
        scale = scale.reshape(weight.shape[0], -1)  ##only for linear, conv1d
        if zp is not None:
            zp = zp.reshape(weight.shape[0], -1)
        return weight, scale, zp

    else:
        pad_len = (weight.shape[1] + group_size - 1) // group_size * group_size - weight.shape[1]
        weight_new = torch.nn.functional.pad(weight, (0, pad_len))
        v = torch.nn.functional.pad(v, (0, pad_len))
        weight_new = weight_new.reshape(-1, group_size)
        if isinstance(v, torch.Tensor):
            v = v.reshape(-1, group_size)
        weight_new, scale, zp = quant_weight_actor(
            weight_new, num_bits, scheme=scheme, v=v, min_scale=min_scale, max_scale=max_scale, scale_dtype=scale_dtype
        )
        weight_new = weight_new.reshape(orig_shape[0], -1)

        weight_new = weight_new[:, :-pad_len]
        scale = scale.reshape(weight_new.shape[0], -1)  ##only for linear, conv1d
        if zp is not None:
            zp = zp.reshape(weight_new.shape[0], -1)
        return weight_new, scale, zp
    
    
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
    if zp is not None:
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


def get_module(model, key):
    """Get module from model by key name.

    Args:
        model (torch.nn.Module): original model
        key (str): module name to be replaced
    """
    module = model
    name_list = key.split(".")
    for name in name_list:
        if hasattr(module, name):
            module = getattr(module, name)
            module = module
    return module


def set_module(model, key, new_module):
    """Set new module into model by key name.

    Args:
        model (torch.nn.Module): original model
        key (str): module name to be replaced
        new_module (torch.nn.Module): new module to be inserted
    """
    module = model
    name_list = key.split(".")
    for name in name_list[:-1]:
        if hasattr(module, name):
            module = getattr(module, name)
        else:
            module = module
    setattr(module, name_list[-1], new_module)


def get_scale_shape(weight, group_size):
    """Computes the shape of the scale tensor for quantization based on the weight tensor and group size.

    Args:
      weight (torch.Tensor): The weight tensor of the layer.
      group_size (int): The size of the groups for quantization.

    Returns:
      The shape of the scale tensor to be used for quantization.
    """
    if group_size == -1 or weight.shape[1] < group_size:
        shape = weight.shape[0]
    else:
        shape = weight.shape[0] * ((weight.shape[1] + group_size - 1) // group_size)

    return shape


def move_input_to_device(input, device=torch.device("cpu")):
    """Moves input data to the specified device.

    Args:
    input: The input data to be moved.
    device: The target device.

    Returns:
    The input data on the specified device.
    """
    if isinstance(input, torch.Tensor):
        return input.to(device)
    if isinstance(input, dict) or isinstance(input, UserDict):
        for inp in input.keys():
            input[inp] = move_input_to_device(input[inp], device)
    elif isinstance(input, list) or isinstance(input, tuple):
        input_res = []
        for inp in input:
            input_res.append(move_input_to_device(inp, device))
        input = input_res
    return input


def check_is_cpu(device):
    """Check if the device is a CPU.

    Args:
        device: The device to be checked.

    Returns:
        bool: True if the device is a CPU, False otherwise.
    """
    return device == torch.device("cpu") or device == "cpu"


def get_block_names(model):
    """Get the block names for transformers-like networks.

    Args:
    model: The model.

    Returns:
    block_names: A list of block names.
    """
    block_names = []
    target_m = None
    for n, m in model.named_modules():
        if hasattr(type(m), "__name__") and "ModuleList" in type(m).__name__:
            target_m = (n, m)
            break  ## only find the first modulelist, may be not robust
    for n, m in target_m[1].named_children():
        block_names.append(target_m[0] + "." + n)
    return block_names




def collect_round_v(block):
    """Collects the round values for wrapped linear modules in the given block.

    Args:
    block: The input block.

    Returns:
    vs: A dictionary of round values for the wrapped linear modules.
    """
    vs = {}
    for n, m in block.named_modules():
        if hasattr(m, "orig_layer"):
            v = m.value.data
            vs[n] = copy.deepcopy(v)
    return vs


def collect_minmax_scale(block):
    """Collects the min-max scaling values for wrapped linear modules in the given block.

    Args:
    block: The input block.

    Returns:
    min_scales: A dictionary of minimum scaling values.
    max_scales: A dictionary of maximum scaling values.
    """
    min_scales = {}
    max_scales = {}
    for n, m in block.named_modules():
        if hasattr(m, "orig_layer"):
            min_scales[n] = copy.deepcopy(torch.clamp(m.min_scale.data, -1, 0))
            max_scales[n] = copy.deepcopy(torch.clamp(m.max_scale.data, -1, 0))
    return min_scales, max_scales


@torch.no_grad()
def get_batch_dim(input_others):
    """Gets the batch dimension based on the input positional inputs.

    Args:
    input_others: A dictionary containing input data.

    Returns:
    dim: The batch dimension.
    """
    dim = int(len(input_others["positional_inputs"]) > 0)
    return dim
            
            
def sampling_inputs(input_ids, input_others, indices, seqlen):
    """Samples inputs based on the given indices and sequence length.

    Args:
    input_ids: The input tensor containing IDs.
    input_others: A dictionary containing other input data.
    indices: The indices to sample from the input.
    seqlen: The sequence length.

    Returns:
    current_input_ids: The sampled input IDs.
    current_input_others: The sampled other input data.
    """
    if len(input_ids.shape) == 3:
        if int(len(input_others["positional_inputs"]) > 0):
            current_input_ids = input_ids[:, indices, :]
        else:
            current_input_ids = input_ids[indices, :, :]
    else:
        n_samples = input_ids.shape[0] // seqlen
        current_input_ids = input_ids.view(n_samples, seqlen, -1)
        current_input_ids = current_input_ids[indices, :, :]
        current_input_ids = current_input_ids.reshape(-1, input.shape[-1])

    current_input_others = {"positional_inputs": input_others["positional_inputs"]}
    for key in input_others.keys():
        if "attention_mask" in key or "alibi" in key:
            current_input_others[key] = None
            if input_others[key] is not None:
                current_input_others[key] = input_others[key][indices, ...]
        else:
            current_input_others[key] = input_others[key]

    return current_input_ids, current_input_others


def block_forward(block, input_ids, input_others, amp=False, amp_dtype=torch.float16, device=torch.device("cpu")):
    """Performs a forward pass through a block with the given inputs.

    Args:
    block: The block to perform the forward pass on.
    input_ids: The input IDs.
    input_others: A dictionary containing other input data.
    amp: A boolean indicating whether to use automatic mixed precision.
    amp_dtype: The data type for automatic mixed precision.
    device: The target device.

    Returns:
    output: The output of the forward pass.
    """
    if input_ids.device != device:
        # input_ids, input_others = move_to_device(input_ids, input_others, device)
        input_ids = move_input_to_device(input_ids, device)
        input_others = move_input_to_device(input_others, device)
    if "alibi" in input_others.keys():
        attention_mask = input_others["attention_mask"]
        alibi = input_others["alibi"]
        if alibi is not None:
            alibi = alibi.reshape(-1, alibi.shape[2], alibi.shape[3])
        if amp and not check_is_cpu(device):
            with autocast(device_type="cuda", dtype=amp_dtype):  # pragma: no cover
                output = block(
                    input_ids, attention_mask=attention_mask, alibi=alibi
                )  ##TODO is this correct for all models with alibi?
        elif amp and check_is_cpu(device):
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                output = block(input_ids, attention_mask=attention_mask, alibi=alibi)
        else:
            output = block(input_ids, attention_mask=attention_mask, alibi=alibi)
    else:
        input_tuple = input_others.pop("positional_inputs", None)
        if amp and not check_is_cpu(device):
            with autocast(device_type="cuda", dtype=amp_dtype):  # pragma: no cover
                output = block.forward(input_ids, *input_tuple, **input_others)
        elif amp and check_is_cpu(device):
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                output = block.forward(input_ids, *input_tuple, **input_others)
        else:
            output = block.forward(input_ids, *input_tuple, **input_others)
    if isinstance(output, list) or isinstance(output, tuple):
        output = output[0]
    return output