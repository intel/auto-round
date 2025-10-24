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
import gc

import torch


def bytes_to_gigabytes(bytes) -> int:
    """
    Converts bytes to gigabytes.

    Args:
        bytes (int): The number of bytes.

    Returns:
        int: The equivalent number of gigabytes.
    """
    return bytes / 1024 / 1024 / 1024


def _clear_memory_for_cpu_and_cuda(tensor=None):
    if isinstance(tensor, list):
        for i in range(len(tensor)):
            tensor[i] = None
    if tensor is not None:
        del tensor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.xpu.is_available():
        torch.xpu.empty_cache()


@torch._dynamo.disable()
def clear_memory(tensor=None):
    from auto_round.utils.device_utils import is_hpex_available

    if is_hpex_available():
        # hpu does not have empty_cache
        return
    else:
        _clear_memory_for_cpu_and_cuda(tensor)


def check_memory_availability(device, inputs, weight, org_seqlen, org_bs):
    """Checks the availability of memory on the specified device for processing inputs using a given weight tensor.

    Args:
        device (str): The device type ('cuda' for GPU or 'hpu' for HPU).
        inputs (torch.Tensor): Input tensor.
        weight (torch.Tensor): Weight tensor.
        org_seqlen (int): Original sequence length.
        org_bs (int): Original batch size.

    Returns:
        tuple: A tuple containing availability status (bool), modified sequence length (int),
               and modified batch size (int).
    """
    weight_memory = weight.numel() * weight.element_size()
    if "cuda" in device:
        current_gpu_index = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(current_gpu_index).total_memory
        used_memory = torch.cuda.memory_allocated(current_gpu_index)
        free_space = total_memory - used_memory
    elif "hpu" in device:  # pragma: no cover
        current_hpu_index = torch.hpu.current_device()
        free_space = torch.hpu.memory_reserved(current_hpu_index)
    else:
        return True, org_seqlen, org_bs

    free_space = free_space - weight_memory * 10  # for min_max_scale & grad usage
    seqlen = org_seqlen
    bs = org_bs
    in_feature = weight.shape[1]
    out_feature = weight.shape[0]
    while seqlen >= 128:
        input_size = bs * seqlen * in_feature
        output_size = bs * seqlen * out_feature
        input_output_memory = 2 * (input_size * inputs.element_size() + output_size * inputs.element_size())
        if input_output_memory < free_space:
            return True, seqlen, bs
        seqlen = seqlen // 2
        bs = 1

    return False, seqlen, bs


def estimate_tuning_block_mem(block: torch.nn.Module, input_ids: list[torch.Tensor]) -> tuple[float, float]:
    """
    Calculates the memory consumption of a specific block in the model.

    Args:
        block (torch.nn.Module): The block of the model to analyze.
        input_ids (list[torch.Tensor]): A list of input tensors for the block.

    Returns:
        tuple: A tuple containing the following:
            - block_memory (float): The memory consumption (in GB) of the block's linear layers.
            - input_output_memory (float): The memory consumption (in GB) for input and output
                tensors of the block.
    """
    # Calculate all block parameters memory
    from auto_round.utils.quantization_utils import check_to_quantized

    total_param_mem = 0
    for name, module in block.named_modules():
        if check_to_quantized(module):
            param_size = module.weight.nbytes
            total_param_mem += param_size
    block_memory = total_param_mem / 1024**3  # Convert to GB

    # Assuming bfloat16 or float32, input and output
    input_output_memory = 2 * sum(tensor.nbytes for tensor in input_ids) / 1024**3

    return block_memory, input_output_memory


def out_of_vram(error_msg):
    error_msg = str(error_msg)
    # CUDA
    if "CUDA out of memory" in error_msg:
        return True
    # gaudi
    if "MODULE:PT_DEVMEM" in error_msg:
        return True
    # XPU
    if "UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY" in error_msg:
        return True
    # ROCM
    if "HIP out of memory. Tried to allocate" in error_msg:
        return True
    return False


def get_max_vram(ratio: float = 0.9) -> dict:
    max_memory = {}
    if torch.cuda.is_available():  # NVIDIA CUDA
        num_devices = torch.cuda.device_count()
        for i in range(num_devices):
            total_mem = torch.cuda.get_device_properties(i).total_memory
            max_mem_gb = int(total_mem / 1024**3 * ratio)
            max_memory[i] = f"{max_mem_gb}GiB"
    elif torch.xpu.is_available():  # TODO need verification
        num_devices = torch.xpu.device_count()
        for i in range(num_devices):
            total_mem = torch.xpu.get_device_properties(i).total_memory
            max_mem_gb = int(total_mem / 1024**3 * ratio)
            max_memory[i] = f"{max_mem_gb}GiB"

    else:
        raise RuntimeError("No CUDA or XPU devices found.")
    return max_memory


def get_device_memory(i: int = 0) -> int:
    """
    Gets the available memory on the specified device.

    Args:
        i (int, optional): Device index. Defaults to 0.

    Returns:
        int: Available memory in gigabytes.
    """
    if torch.cuda.is_available():
        total_memory = bytes_to_gigabytes(torch.cuda.get_device_properties(i).total_memory)
    elif torch.xpu.is_available():
        raise RuntimeError("XPU does not support device_map='auto' currently.")
    else:
        raise RuntimeError("No supported device found (CUDA or XPU).")
    return total_memory
