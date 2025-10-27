from typing import Union

import torch

from auto_round.logger import logger

from auto_round.utils import detect_device, get_module, get_device_memory, estimate_tuning_block_mem, check_to_quantized
import re


def get_major_device(device_map: Union[str, torch.device, int, dict]) -> str:
    if isinstance(device_map, (str, torch.device, int)):
        device = detect_device(device_map)
        return device

    if isinstance(device_map, dict) and device_map:
        tmp_devices = []
        for val in device_map.values():
            if isinstance(val, (str, torch.device, int)):  # could optimize
                tmp_device = detect_device(val)
                tmp_device = tmp_device.split(":")[0]
                tmp_devices.append(tmp_device)
        tmp_devices = list(set(tmp_devices))
        if len(tmp_devices) > 1:
            logger.warning(
                f"there are multiple device types in the device_map, "
                f"please make sure they are correct,use the first none-cpu device {tmp_devices[0]} as the core device "
            )

        for device in tmp_devices:
            if device != "cpu":
                return device

        device = tmp_devices[0]
        return device
    logger.warning(f"device_map should be [str, torch.device, int, dict], but got {type(device_map)}")
    return "cpu"


def set_tuning_device_for_layer(model, name: str, device: str) -> None:
    """Sets the device for a module if it matches the given name."""
    module = get_module(model, name)
    if hasattr(module, "tuning_device") and module.tuning_device != device:
        logger.warning(
            f"multiple devices have been set for layer {name}, keeping original device {module.tuning_device}"
        )
    else:
        module.tuning_device = device


def set_non_auto_device_map(model: torch.nn.Module, device_map, quant_layer_names=None):
    if not device_map:
        return
    if device_map == "auto":
        return
    if isinstance(device_map, str) and "," in device_map:  # auto device map
        return
    if isinstance(device_map, int):
        return
    if isinstance(device_map, str):
        device_map = device_map.replace(" ", "")
        infos = device_map.split(",")
        device_map_dict = {}
        for info in infos:
            index = info.find(":")
            key = info[:index]
            value = info[index + 1:]
            device_map_dict[key] = value
        device_map = device_map_dict
    if quant_layer_names is not None:
        names = quant_layer_names
    else:
        names = [n for n, m in model.model.named_modules() if len(list(m.children())) == 0] # if it's a block, it will be incorrect
    for key, device in device_map.items():
        if isinstance(device, str) and device.isdigit():
            device = int(device)
        device = detect_device(device)
        if key in names:
            module = get_module(model, key)
            module.tuning_device = device
        else:
            matching_names = [name for name in names if re.match(key, name)]
            for name in matching_names:
                set_non_auto_device_map(model, name, device)
            if not matching_names:
                logger.warning(f"{key} in `device_map` dose not match any modules, please have a check")



def set_auto_device_map_for_block(block: torch.nn.Module, device_map, input_ids: list[torch.Tensor],
                                  low_gpu_mem_usage=False,mem_per_param_scale=13.0) -> None:
    """Automatically sets the device map for the block based on available GPUs and memory constraints."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    elif torch.xpu.is_available():
        logger.warning_once("XPU does not support auto device map yet, using device 0 for tuning.")
        return
    else:
        raise RuntimeError("No CUDA or XPU devices found.")
    device_list = None
    if isinstance(device_map, str) and "," in device_map:
        device_list = [int(dev) for dev in device_map.split(",") if dev.isdigit()]

    if device_list:
        cuda_devices = [f"cuda:{i}" for i in device_list]
        device_0 = cuda_devices[0]
    else:
        cuda_devices = [f"cuda:{i}" for i in range(num_gpus)]
        device_0 = "cuda:0"

    device_0_memory = get_device_memory(
    device_list[0] if device_list else 0
    )
    block_memory, input_output_memory = estimate_tuning_block_mem(block, input_ids)
    if low_gpu_mem_usage:
        input_output_memory = 0


    if (block_memory * mem_per_param_scale + input_output_memory) < device_0_memory:
        return  # fit in one GPU

    device_map = {}
    device_memory = {device: get_device_memory(int(device.split(":")[1])) for device in cuda_devices}
    device_memory[device_0] = device_0_memory - input_output_memory

    device_idx = 0
    names = []
    # First, fill device 0 to its maximum capacity, then distribute the remaining layers evenly across other devices
    for n, m in block.named_modules():
        if check_to_quantized(m):
            layer_name = m.tmp_name
            names.append(layer_name)
            layer_memory = m.weight.nbytes / 1024**3
            if device_idx == 0 and layer_memory * mem_per_param_scale < device_memory[cuda_devices[device_idx]]:
                device_map[layer_name] = cuda_devices[device_idx]
                device_memory[cuda_devices[device_idx]] -= layer_memory * mem_per_param_scale
            elif device_idx == 0:
                device_idx += 1  # Move to the next device once device 0 is full
                device_map[layer_name] = cuda_devices[device_idx]
                device_memory[cuda_devices[device_idx]] -= layer_memory * mem_per_param_scale
            else:
                # Calculate the target device index based on even distribution
                sorted_devices = sorted(cuda_devices, key=lambda d: device_memory[d], reverse=True)
                device_idx = sorted_devices[0]
                if layer_memory * mem_per_param_scale < device_memory[device_idx]:
                    device_map[layer_name] = device_idx
                    device_memory[device_idx] -= layer_memory * mem_per_param_scale
                else:
                    logger.warning_once(
                        f"Block {block.tmp_name} not fit in available GPU memory. "
                        "Consider using more GPUs or reducing mem_per_param_scale if OOM occurs."
                    )
    set_non_auto_device_map(block,device_map,names)
