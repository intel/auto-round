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
import os
import re
from functools import lru_cache
from itertools import combinations
from typing import Callable, Union

import cpuinfo
import torch

from auto_round.logger import logger
from auto_round.utils.model import check_to_quantized, get_block_names, get_layer_features, get_module

# Note on HPU usage:
# There are two modes available for enabling auto-round on HPU:
# 1. Compile Mode
#   1) Use PyTorch version ≥ 2.4 (Intel® Gaudi® v1.18 or later)
#   2) Set `PT_HPU_LAZY_MODE=0` and `PT_ENABLE_INT64_SUPPORT=1`
#   The compile mode can speed up quantization process but still in experimental stage.
# 2. Lazy Mode (By default)


################ Check available sys.module to decide behavior #################
def is_package_available(package_name: str) -> bool:
    """Check if the package exists in the environment without importing.

    Args:
        package_name (str): package name
    """
    from importlib.util import find_spec

    package_spec = find_spec(package_name)
    return package_spec is not None


def is_autoround_exllamav2_available():
    """Checks if the AutoRound ExLlamaV2 kernels are available.

    Returns:
        bool:
            True if the AutoRound ExLlamaV2 kernels are available, False otherwise.
    """
    res = True
    try:
        from autoround_exllamav2_kernels import gemm_half_q_half, make_q_matrix
    except ImportError as e:
        res = False
    return res


def is_hpu_lazy_mode():
    return os.getenv("PT_HPU_LAZY_MODE") != "0"


def _use_hpu_compile_mode():
    from auto_round.utils.common import TORCH_VERSION_AT_LEAST_2_4

    return TORCH_VERSION_AT_LEAST_2_4 and not is_hpu_lazy_mode()


def compile_func_on_hpu(func):
    if _use_hpu_compile_mode():
        return torch.compile(func, backend="hpu_backend")
    return func


def compile_func_on_cuda_or_cpu(func):
    return torch.compile(func)


def compile_func(
    fun: Union[torch.nn.Module, Callable], device: Union[str, torch.device, int]
) -> Union[torch.nn.Module, Callable]:
    """Compile function on the specified device."""
    if "hpu" in str(device):
        return compile_func_on_hpu(fun)  ## use auto by default
    else:
        return compile_func_on_cuda_or_cpu(fun)


def is_numba_available():  # pragma: no cover
    """Check if Numba is available."""
    try:
        import numba

        return True
    except ImportError:
        return False


def _is_tbb_installed():  # pragma: no cover
    import importlib.metadata

    try:
        importlib.metadata.version("tbb")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def _is_tbb_configured():  # pragma: no cover
    try:
        from numba.np.ufunc.parallel import _check_tbb_version_compatible

        # check if TBB is present and compatible
        _check_tbb_version_compatible()

        return True
    except ImportError as e:
        logger.warning_once(f"TBB not available: {e}")
        return False


def is_tbb_available():  # pragma: no cover
    """Check if TBB is available."""
    if not _is_tbb_installed():
        logger.warning_once("TBB is not installed, please install it with `pip install tbb`.")
        return False
    if not _is_tbb_configured():
        logger.warning_once(
            (
                "TBB is installed but not configured correctly. \n"
                "Please add the TBB library path to `LD_LIBRARY_PATH`, "
                "for example: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/`."
            )
        )
        return False
    return True


def can_pack_with_numba():  # pragma: no cover
    """Check if Numba and TBB are available for packing.

    To pack tensor with Numba, both Numba and TBB are required, and TBB should be configured correctly.
    """
    if not is_numba_available():
        logger.warning_once("Numba is not installed, please install it with `pip install numba`.")
        return False
    if not is_tbb_available():
        return False
    return True


## check hpex
if is_package_available("habana_frameworks"):
    _hpex_available = True
    import habana_frameworks.torch.hpex  # pylint: disable=E0401
else:
    _hpex_available = False


@torch._dynamo.disable()
@lru_cache(None)
def is_hpex_available():
    return _hpex_available


def check_is_cpu(device):
    """Check if the device is a CPU.

    Args:
        device: The device to be checked.

    Returns:
        bool: True if the device is a CPU, False otherwise.
    """
    return device == torch.device("cpu") or device == "cpu"


def detect_device_count():
    """Detects the number of available computation devices.

    This function checks if CUDA is available. If it is, it returns the count
    of available CUDA devices. If not, it attempts to import the Habana
    device framework to return the count of Habana devices. If the import
    fails or no devices are found, it returns 0.

    Returns:
        int: The number of available devices (CUDA or Habana).
    """
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        try:
            import habana_frameworks.torch.hpu as hthpu  # pylint: disable=E0401

            return hthpu.device_count()
        except ImportError:
            return 0


def detect_device(device: Union[None, str, int, torch.device] = None) -> str:
    """Detects the appropriate computation device.

    This function determines the device to use for computations. It can take
    a specific device index or default to 'auto'. The function checks for
    available devices in the following order: CUDA, Habana, and finally CPU.

    Args:
        device (str, int, or torch.device, optional): The desired device.
            If 'auto' or None, the function will determine the best device
            automatically.

    Returns:
        str: The device to use for computations, formatted as a string.
    """

    def is_valid_digit(s):
        try:
            num = int(s)
            return 0 <= num
        except:
            return False

    dev_idx = None
    if is_valid_digit(device):
        dev_idx = int(device)
        device = "auto"
    if isinstance(device, str) and "," in device:  # device is "0,1,2"
        device_list = [int(dev) for dev in device.split(",") if dev.isdigit()]
        dev_idx = device_list[0] if device_list else None
        device = "auto"
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # logger.info("Using GPU device")
        elif is_hpex_available():  # pragma: no cover
            device = torch.device("hpu")
            # logger.info("Using HPU device")
        elif torch.xpu.is_available():  # pragma: no cover
            device = torch.device("xpu")
        # Use CPU as a fallback
        else:
            device = torch.device("cpu")
            # logger.info("Using CPU device")
        if dev_idx is not None and str(device) != "cpu":
            device = str(device) + f":{dev_idx}"
        return str(device)
    elif isinstance(device, torch.device):
        device = str(device)
    elif isinstance(device, str):  ## for cuda:0
        if device == "tp":  # pragma: no cover
            # should not specify card, e.g., cuda:0
            if torch.cuda.is_available():
                device = "cuda"
            elif is_hpex_available():
                device = "hpu"
            else:
                device = "cpu"
        else:
            device = device
    return device


def get_device_and_parallelism(device: Union[str, torch.device, int]) -> tuple[str, bool]:
    if isinstance(device, str):
        devices = device.replace(" ", "").split(",")
    elif isinstance(device, int):
        devices = [str(device)]
    else:
        devices = [device]
    if all(s.isdigit() for s in devices) and len(devices) > 1 and torch.cuda.is_available():
        device = "cuda"
        parallelism = True
    elif all(s.isdigit() for s in devices) and len(devices) > 1 and torch.xpu.is_available():
        device = "xpu"
        parallelism = False
    # pragma: no cover
    elif device == "auto":
        device = detect_device(device)
        parallelism = True
    else:
        device = detect_device(device)
        parallelism = False
    return device, parallelism


def set_cuda_visible_devices(device):
    devices = device.replace(" ", "").split(",")
    if all(s.isdigit() for s in devices):
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            current_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            current_visible_devices = current_visible_devices.split(",")
            indices = [int(device) for device in devices]
            try:
                pick_device = [current_visible_devices[i] for i in indices]
            except:
                raise ValueError(
                    "Invalid '--device' value: It must be smaller than the number of available devices."
                    " For example, with CUDA_VISIBLE_DEVICES=4,5, "
                    "--device 0,1 is valid, but --device 4,5 is not supported."
                )
            visible_devices = ",".join(pick_device)
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = device


def set_fake_cuda_device_capability(func=None):
    if func is not None:
        torch.cuda.get_device_capability = func
        return func

    def fake_cuda():
        return 100, 1

    orig_func = torch.cuda.get_device_capability
    torch.cuda.get_device_capability = fake_cuda
    return orig_func


def get_packing_device(device: str | torch.device | None = "auto") -> torch.device:
    """
    Selects the packing device.
    - "auto": choose best available (CUDA > XPU > CPU).
    - str: parsed by torch.device (e.g., "cuda:2", "cpu").
    - torch.device: returned as-is.
    - None: treated as "auto".

    Args:
        device: Target device spec ("auto", "cuda:0", "xpu:0", "cpu", or torch.device).

    Returns:
        torch.device: The resolved device.
    """
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu:0")
        return torch.device("cpu")

    if isinstance(device, torch.device):
        return device

    if isinstance(device, str):
        try:
            return torch.device(device)
        except Exception as e:
            raise ValueError(f"Invalid device string: {device}") from e

    raise TypeError(f"Unsupported device type: {type(device)} ({device})")


def is_complex_device_mapping(device_map):
    if device_map is None or isinstance(device_map, int):
        return False
    elif device_map == "auto":
        return True
    elif isinstance(device_map, str) and "," in device_map:
        return True
    elif isinstance(device_map, dict):
        return True
    else:
        return False


class CpuInfo(object):
    """Get CPU Info."""

    def __init__(self):
        """Get whether the cpu numerical format is bf16, the number of sockets, cores and cores per socket."""
        self._bf16 = False
        info = cpuinfo.get_cpu_info()
        if "arch" in info and "X86" in info["arch"]:
            cpuid = cpuinfo.CPUID()
            max_extension_support = cpuid.get_max_extension_support()
            if max_extension_support >= 7:
                eax = cpuid._run_asm(
                    b"\xb9\x01\x00\x00\x00",  # mov ecx, 1
                    b"\xb8\x07\x00\x00\x00" b"\x0f\xa2" b"\xc3",  # mov eax, 7  # cpuid  # ret
                )
                self._bf16 = bool(eax & (1 << 5))

    @property
    def bf16(self):
        """Get whether it is bf16."""
        return self._bf16


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
    from auto_round.utils.device import is_hpex_available

    if is_hpex_available():
        # hpu does not have empty_cache
        return
    else:
        _clear_memory_for_cpu_and_cuda(tensor)


def clear_memory_if_reached_threshold(threshold=0.85):
    """Check all available devices and clear memory if any device is using close to the threshold.

    Args:
        threshold (float): Memory usage threshold (default: 0.85 for 85%).
                            If any device exceeds this percentage, clear_memory() will be called.

    Returns:
        bool: True if memory was cleared, False otherwise.
    """
    # Detect CUDA/XPU devices
    if torch.cuda.is_available():
        name, device_api = "cuda", torch.cuda
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        name, device_api = "xpu", torch.xpu
    else:
        return False

    num_devices = device_api.device_count()
    for i in range(num_devices):
        try:
            total_memory = device_api.get_device_properties(i).total_memory
            reserved_memory = device_api.memory_reserved(i)
            memory_usage_ratio = reserved_memory / total_memory

            if memory_usage_ratio >= threshold:
                logger.warning_once(
                    f"Major device ({name}:{i}) has reached memory threshold. "
                    + "Memory clearing operation will be called during each iteration, which "
                    + "will result in more time consumption."
                )
                logger.warning_once(
                    "To alleviate high memory usage on the major device, consider reducing the `batch_size` "
                    + "(and correspondingly increasing `gradient_accumulation_steps) or shortening the seqlen."
                )
                clear_memory()
                return True
        except Exception as e:
            logger.warning_once(f"Failed to check memory for {name}:{i}: {e}")
    return False


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
        total_memory = bytes_to_gigabytes(torch.xpu.get_device_properties(i).total_memory)
    else:
        raise RuntimeError("No supported device found (CUDA or XPU).")
    return total_memory


def get_major_device(device_map: Union[None, str, torch.device, int, dict]) -> str:
    if device_map is None or isinstance(device_map, (str, torch.device, int)):
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
        device = None
        for tmp_device in tmp_devices:
            if tmp_device != "cpu":
                device = tmp_device
                break
        if device is None:
            device = tmp_devices[0]
        if len(tmp_devices) > 1:
            logger.warning_once(
                f"there are multiple device types in the device_map, "
                f"please make sure they are correct,use the first none-cpu device {device} as the core device "
            )

        return device
    logger.warning_once(f"device_map should be [str, torch.device, int, dict], but got {type(device_map)}")
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


def set_non_auto_device_map(
    model: torch.nn.Module, device_map: Union[str, int, dict], quant_layer_names: Union[None, list, tuple] = None
) -> None:
    if not device_map or device_map == "auto" or isinstance(device_map, int):
        return
    if isinstance(device_map, str):
        if "," in device_map:  # auto device map
            return
        device_map = device_map.replace(" ", "")
        infos = device_map.split(",")
        device_map_dict = {}
        for info in infos:
            index = info.find(":")
            key = info[:index]
            value = info[index + 1 :]
            device_map_dict[key] = value
        device_map = device_map_dict
    if quant_layer_names is not None:
        names = quant_layer_names
    else:
        names = [
            n for n, m in model.named_modules() if len(list(m.children())) == 0
        ]  # if it's a block, it will be incorrect
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
                set_tuning_device_for_layer(model, name, device)
            if not matching_names:
                logger.warning(f"{key} in `device_map` dose not match any modules, please have a check")


def _allocate_layers_to_devices(
    layer_memory_dict: dict, device_memory: dict, gpu_devices: list, mem_per_param: float
) -> tuple[dict, list]:
    """
    Allocates layers to devices using a load-balancing strategy.

    Strategy:
    1. Sort layers by memory size (descending), preserve order for equal sizes
    2. Assign largest N layers to higher-index devices (N = num_devices)
    3. Remaining layers use memory availability + layer continuity scorings

    Args:
        layer_memory_dict (dict): Mapping of layer names to memory info (order preserved)
        device_memory (dict): Available memory for each device (will be modified)
        gpu_devices (list): List of device names (e.g., ["cuda:0", "cuda:1"])
        mem_per_param (float): Memory multiplier per parameter GB

    Returns:
        tuple[dict, list]: (device_map, names)

    Example:
        Input:
            device_memory = {"cuda:0": 30.0, "cuda:1": 40.0, "cuda:2": 40.0}
            layer_memory_dict = {
                "q_proj": {"param_memory": 4.0}, "k_proj": {"param_memory": 1.0},
                "v_proj": {"param_memory": 1.0}, "o_proj": {"param_memory": 4.0},
                "gate_proj": {"param_memory": 11.0}, "up_proj": {"param_memory": 11.0},
                "down_proj": {"param_memory": 11.0}
            }
            mem_per_param = 2.0

        Result (allocation order by size):
            1. gate_proj (22GB) -> cuda:2 (largest, prefer last device)
            2. up_proj (22GB) -> cuda:1 (2nd largest, prefer 2nd last device)
            3. down_proj (22GB) -> cuda:0 (3rd largest, cuda:0 has 30GB available)
            4. q_proj (8GB) -> cuda:2 (neighbor of gate_proj, continuity bonus)
            5. o_proj (8GB) -> cuda:2 (neighbor of q_proj, continuity bonus)
            6. k_proj (2GB) -> cuda:1 (neighbor of q_proj via original order)
            7. v_proj (2GB) -> cuda:1 (neighbor of k_proj, continuity bonus)
    """
    device_map = {}
    names = []
    layer_names_in_order = list(layer_memory_dict.keys())
    layer_order = {name: idx for idx, name in enumerate(layer_names_in_order)}
    sorted_layers = sorted(layer_memory_dict.items(), key=lambda x: (-x[1]["param_memory"], -layer_order[x[0]]))
    num_devices = len(gpu_devices)

    def find_best_device(layer_name, estimated_memory, layer_idx):
        """Find the best device for a layer."""
        # Phase 1: Direct assign largest layers to higher-index devices first
        if layer_idx < num_devices - 1:
            return gpu_devices[-(layer_idx + 1)]

        # Phase 2: Choose device with best score (memory + continuity)
        best_device = None
        best_score = float("-inf")
        current_layer_order = layer_order[layer_name]

        for device in gpu_devices:
            if device_memory[device] < estimated_memory:
                continue

            # Memory score (normalized)
            memory_score = device_memory[device] / estimated_memory

            # Continuity bonus for adjacent layers
            continuity_bonus = 0
            for offset in [-1, 1]:  # Check previous and next neighbors
                neighbor_idx = current_layer_order + offset
                if 0 <= neighbor_idx < len(layer_names_in_order):
                    neighbor_name = layer_names_in_order[neighbor_idx]
                    if neighbor_name in device_map and device_map[neighbor_name] == device:
                        continuity_bonus += 1.0

            total_score = memory_score + continuity_bonus
            if total_score > best_score:
                best_score = total_score
                best_device = device

        # Fallback: device with most available memory
        return best_device or max(gpu_devices, key=lambda d: device_memory[d])

    # Allocate layers
    for layer_idx, (layer_name, mem_info) in enumerate(sorted_layers):
        names.append(layer_name)
        estimated_memory = mem_info["param_memory"] * mem_per_param
        best_device = find_best_device(layer_name, estimated_memory, layer_idx)
        device_map[layer_name] = best_device
        device_memory[best_device] -= estimated_memory

    # Restore original order
    ordered_device_map = {name: device_map[name] for name in layer_names_in_order if name in device_map}
    return ordered_device_map, names


def get_first_available_attr(obj, attr_names: list[str], default=None):
    """
    Get the first available attribute from a list of attribute names.

    Args:
        obj: The object to get the attribute from.
        attr_names (list[str]): List of attribute names to try in order.
        default: Default value to return if none of the attributes exist.

    Returns:
        The value of the first available attribute, or default if none exist.
    """
    for attr_name in attr_names:
        value = getattr(obj, attr_name, None)
        if value is not None:
            return value
    return default


def get_moe_memory_ratio(block: torch.nn.Module) -> float:
    """
    Calculate the memory ratio for MoE (Mixture of Experts) models.

    For MoE models, only num_experts_per_tok experts are activated per token,
    not all experts. This function returns the ratio of active experts to total experts.

    Args:
        block (torch.nn.Module): The model block to analyze.

    Returns:
        float: Memory ratio (num_experts_per_tok / num_experts).
               Returns 1.0 for non-MoE models.
        bool: True if the model is MoE, False otherwise.

    Examples:
        - Non-MoE model: returns 1.0
        - Mixtral (2/8 experts): returns 0.25
        - Qwen2MoE (4/60 experts): returns ~0.067
    """
    from auto_round.utils.model import is_moe

    for name, module in block.named_modules():
        if not is_moe(module):
            continue

        config = getattr(block, "config", None)
        if config is None:
            break

        # Try to get num_experts_per_tok (active experts count)
        # Mixtral, Qwen2MoE, DeepSeek, GPT-OSS, Llama4, LLaDAMoE, SmallThinker
        num_experts_per_tok = get_first_available_attr(
            config, ["num_experts_per_tok", "moe_num_active_primary_experts"]
        )

        # HunYuan MoE uses moe_topk (array), get first element
        if num_experts_per_tok is None:
            moe_topk = getattr(config, "moe_topk", None)  # HunYuan MoE V1
            if moe_topk is not None and isinstance(moe_topk, (list, tuple)) and len(moe_topk) > 0:
                num_experts_per_tok = moe_topk[0]
            elif moe_topk is not None:
                num_experts_per_tok = moe_topk

        if num_experts_per_tok is None:
            break

        # Get total number of experts
        # Mixtral, PhiMoE, Grok, Llama4, Qwen2MoE, Olmo, BailingMoE, GroveMoE, HunYuan, LLaDAMoE, SmallThinker, DeepSeek
        num_experts = get_first_available_attr(
            config, ["num_local_experts", "num_experts", "moe_num_primary_experts", "n_routed_experts"]
        )

        if num_experts is not None and num_experts > 0:
            moe_ratio = num_experts_per_tok / num_experts
            logger.debug(
                f"MoE detected: {num_experts_per_tok}/{num_experts} experts active per token, "
                f"activation memory ratio: {moe_ratio:.2f}"
            )
            logger.debug(f"Using MoE memory ratio: {moe_ratio:.4f}")
            return moe_ratio, True
        break  # Only check once per block

    return 1.0, False  # Default ratio for non-MoE models


def estimate_tuning_block_mem(
    block: torch.nn.Module, input_ids: list[torch.Tensor], batch_size: int
) -> tuple[dict, float]:
    """
    Calculates the memory consumption of a specific block in the model.

    Args:
        block (torch.nn.Module): The block of the model to analyze.
        input_ids (list[torch.Tensor]): A list of input tensors for the block.
        batch_size (int): Number of samples to consider for memory estimation.

    Returns:
        tuple: A tuple containing the following:
            - layer_memory_dict (dict): A dictionary mapping layer names to their memory consumption (in GB).
                Format: {layer_name: {"param_memory": float, "output_memory": float}}
            - input_output_memory (float): The memory consumption (in GB) for input and output
                tensors of the block.
            - additional_memory (float): Additional memory overhead (in GB) for operations like attention.
    """
    # Calculate all block parameters memory and build layer-wise memory dict
    from auto_round.utils.model import get_layer_features, is_moe

    layer_memory_dict = {}

    # Calculate batch_size and sequence_length from input_ids for output memory estimation
    seq_len = input_ids[0].shape[1] if input_ids and len(input_ids[0].shape) >= 2 else 1
    element_size = input_ids[0].element_size() if input_ids else 2  # Default to 2 bytes (fp16/bf16)

    moe_ratio, has_moe = get_moe_memory_ratio(block)  # Get MoE memory ratio (1.0 for non-MoE models)

    for name, module in block.named_modules():
        if check_to_quantized(module):
            enable_act_quant = module.act_bits <= 8
            layer_name = name
            param_size = module.weight.nbytes
            param_memory_gb = param_size / 1024**3
            param_memory_gb *= 2  # considering the v tensor for weight rounding

            # Estimate output memory based on input_features and out_features
            in_features, out_features = get_layer_features(module)
            if in_features is not None and out_features is not None:
                # Output tensor size: batch_size * seq_len * out_features * element_size
                output_size = batch_size * seq_len * out_features * element_size
                output_memory_gb = output_size / 1024**3

                # If enable_act_quant, add input tensor memory to param_memory
                if enable_act_quant:
                    input_size = batch_size * seq_len * in_features * element_size
                    input_memory_gb = input_size / 1024**3
                    param_memory_gb += input_memory_gb
            else:
                output_memory_gb = 0.0

            if has_moe:
                pparent_module = get_module(block, layer_name.rsplit(".", 2)[0]) if "." in layer_name else block
                is_moe_expert = "expert" in layer_name.lower() and isinstance(pparent_module, torch.nn.ModuleList)
            else:
                is_moe_expert = False

            # memory * 2, because it contains grad tensor.
            layer_memory_dict[layer_name] = {
                "param_memory": param_memory_gb * 2,
                "output_memory": output_memory_gb * 2,
                "is_moe_expert": is_moe_expert,
            }

    # Assuming bfloat16 or float32, input and output
    block_input_output_memory = 2 * sum(tensor.nbytes for tensor in input_ids) / 1024**3

    # Roughly estimate additional memory for attention and other operations
    # For MoE expert layers, multiply activation memory by the ratio of active experts
    # For non-MoE layers (attention, norm, etc.), use full activation memory
    layer_activation_memory = 0.0
    for layer_name, info in layer_memory_dict.items():
        if info.get("is_moe_expert", False):
            # MoE expert layer: only a fraction of experts are active
            layer_activation_memory += info["output_memory"] * moe_ratio
        else:
            # Non-MoE layer: use full activation memory
            layer_activation_memory += info["output_memory"]

    # layer_activation_memory considers other ops activation memory
    # 1GB considers norm weight, sdpa, reference_output, etc.
    additional_memory = layer_activation_memory + 1  # GB
    if has_moe:
        # TODO: Cannot estimate the memory usage correctly for MoE models yet.
        # For MoE models, additional memory usage can be higher due to routing, gating,
        # and multiple expert activations. Here we use a conservative estimate.
        moe_additional_memory = additional_memory * 6  # GB
        additional_memory += moe_additional_memory
    if torch.xpu.is_available():
        # https://github.com/intel/torch-xpu-ops/issues/2232
        # TODO: XPU takes more memory than expected. for llama 8B, it's about 12 GB
        xpu_additional_memory = 12  # GB
        additional_memory += xpu_additional_memory
    logger.warning_once(
        "[Memory Estimation]: If there is an abnormal memory issue, please collect log with "
        + "AR_LOG_LEVEL=debug and raise issue to us."
    )

    return layer_memory_dict, layer_activation_memory, block_input_output_memory, additional_memory


def set_auto_device_map_for_block_with_tuning(
    block: torch.nn.Module,
    device_map,
    input_ids: list[torch.Tensor],
    low_gpu_mem_usage: bool = False,
    batch_size: int = 8,
    output_device: str | torch.device = None,
    card_0_threshold: float = 0.9,
):
    """
    Automatically sets the device map for the block based on available GPUs and memory constraints.

    Args:
        block (torch.nn.Module): The model block whose device map is to be set.
        device_map (str | int | dict): Specifies the device mapping.
        input_ids (list[torch.Tensor]): List of input tensors used for estimating memory requirements.
        low_gpu_mem_usage (bool, optional): If True, ignoring input/output memory. Defaults to False.
        batch_size (int, optional): Number of samples to consider for memory estimation. Defaults to 8.
        output_device (str | torch.device, optional): Device to move unassigned modules to. Defaults to None.
        card_0_threshold (float, optional): Threshold ratio to determine if the first device is at high risk of
            running out of memory. Defaults to 0.9 (90%).

    Returns:
        card_0_in_high_risk (bool): True if the first device is at risk of running out of memory, False otherwise.
            card_0_in_high_risk = card_0_used_memory / device_0_memory > card_0_threshold
            card_0_used_memory = card_0_left_memory + block_input_output_memory + additional_memory
            We may need to clear card 0 memory more frequently during training/inference in that case.

    Raises:
        RuntimeError: If no CUDA or XPU devices are found.

    Note:
        This function is intended for internal use in device memory management and tuning.
    """
    if not (device_map == "auto" or ((isinstance(device_map, str) and "," in device_map))):
        block = block.to(output_device)
        card_0_in_high_risk = False  # card 0 contains weight, clear_memory will not help much
        loss_device = output_device
        return card_0_in_high_risk, loss_device

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        device_name = "cuda"
    elif torch.xpu.is_available():
        num_devices = torch.xpu.device_count()
        device_name = "xpu"
    else:
        return
    device_list = None
    if isinstance(device_map, str) and "," in device_map:
        device_list = [int(dev) for dev in device_map.split(",") if dev.isdigit()]

    if device_list:
        gpu_devices = [f"{device_name}:{i}" for i in device_list]
        device_0 = gpu_devices[0]
        device_1 = gpu_devices[1]
    else:
        gpu_devices = [f"{device_name}:{i}" for i in range(num_devices)]
        device_0 = f"{device_name}:0"
        device_1 = f"{device_name}:1"

    device_0_memory = get_device_memory(device_list[0] if device_list else 0)
    device_1_memory = get_device_memory(device_list[1] if device_list else 1)
    layer_memory_dict, layer_activation_memory, block_input_output_memory, additional_memory = (
        estimate_tuning_block_mem(block, input_ids, batch_size)
    )
    loss_memory = block_input_output_memory / 2  # GB, rough estimate for loss tensor memory
    if low_gpu_mem_usage:
        block_input_output_memory = 0

    total_block_param_memory = sum(info["param_memory"] for info in layer_memory_dict.values())

    # Average dispatch strategy
    # card_0_left_memory = card_0_mem - block_input_output_memory - additional_memory - layer_outputs_memory
    card_0_used_memory = block_input_output_memory + layer_activation_memory + additional_memory
    logger.debug(f"Card 0 used memory details [Estimated]: {card_0_used_memory} GB")
    logger.debug(f"  Block input output cache memory: {block_input_output_memory} GB")
    logger.debug(f"  Quantized layer outputs memory: {layer_activation_memory} GB")
    logger.debug(f"  Additional_memory from other ops: {additional_memory} GB")

    card_0_left_memory = max(0, (device_0_memory - card_0_used_memory))
    card_0_in_high_risk = card_0_used_memory / device_0_memory >= card_0_threshold
    card_1_left_memory = max(0, device_1_memory - loss_memory) if card_0_in_high_risk else device_1_memory
    loss_device = device_1 if card_0_in_high_risk else output_device

    # Calculate total available memory across all devices
    total_available_memory = card_0_left_memory + card_1_left_memory
    for i in range(2, len(gpu_devices)):
        device_idx = device_list[i] if device_list else i
        total_available_memory += get_device_memory(device_idx)

    # Calculate total params (in GB, considering param_memory only for calculation)
    total_params = total_block_param_memory
    mem_per_param = total_available_memory / total_params

    # Initialize device memory tracking
    device_memory = {}
    device_memory[device_0] = card_0_left_memory
    for i in range(1, len(gpu_devices)):
        device_idx = device_list[i] if device_list else i
        device_memory[gpu_devices[i]] = get_device_memory(device_idx)

    # Allocate layers to devices using load-balancing strategy
    device_map, names = _allocate_layers_to_devices(layer_memory_dict, device_memory, gpu_devices, mem_per_param)

    logger.debug(f"Auto device map for block: {device_map}")
    set_non_auto_device_map(block, device_map, names)

    # Ensure all remaining modules with parameters/buffers are moved to expected device, by default device_0
    output_device = device_0 if output_device is None else output_device
    for name, module in block.named_modules():
        if name not in names:  # This module wasn't assigned a device
            # Check if module has any parameters or buffers
            has_params = any(True for _ in module.parameters(recurse=False))
            has_buffers = any(True for _ in module.buffers(recurse=False))
            if has_params or has_buffers:
                module = module.to(output_device)

    return card_0_in_high_risk, loss_device


def partition_dict_numbers(number_dict, n):
    """
    Partition a dictionary of numbers into N groups with approximately equal sums
    """
    # Edge cases
    if n > len(number_dict):
        groups = []
        for key, value in number_dict.items():
            groups.append({key: value})
        for _ in range(n - len(number_dict)):
            groups.append({})
        return groups

    if n == len(number_dict):
        return [{key: value} for key, value in number_dict.items()]

    total_sum = sum(number_dict.values())
    # target = total_sum / n  # Use float for better precision

    items = list(number_dict.items())
    result = []
    remaining = items.copy()

    def find_optimal_subset(arr, target):
        """Find subset with sum closest to target"""
        best_subset = []
        best_diff = float("inf")

        # Try all possible subset sizes
        for r in range(1, len(arr) + 1):
            for combo in combinations(arr, r):
                current_sum = sum(value for _, value in combo)
                current_diff = abs(current_sum - target)

                # If we found a perfect match, return immediately
                if current_diff == 0:
                    return list(combo)

                # Update the best subset if this is better
                if current_diff < best_diff and current_sum <= total_sum:
                    best_diff = current_diff
                    best_subset = list(combo)

        return best_subset

    # Distribute items into n-1 groups
    for i in range(n - 1):
        if not remaining:
            break

        # Calculate dynamic target based on remaining items
        remaining_target = sum(value for _, value in remaining) / (n - i)
        subset = find_optimal_subset(remaining, remaining_target)

        result.append(dict(subset))

        # Remove allocated items
        for item in subset:
            remaining.remove(item)

    # Last group gets all remaining items
    result.append(dict(remaining))

    return result


def set_avg_auto_device_map(model: torch.nn.Module, device_map):
    block_name_list = get_block_names(model)
    device_list = None
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        device_name = "cuda"
    elif torch.xpu.is_available():
        num_devices = torch.xpu.device_count()
        device_name = "xpu"
    else:
        return

    if isinstance(device_map, str) and "," in device_map:
        device_list = [int(dev) for dev in device_map.split(",") if dev.isdigit()]
        num_devices = len(device_list)

    if device_list:
        gpu_devices = [f"{device_name}:{i}" for i in device_list]
    else:
        gpu_devices = [f"{device_name}:{i}" for i in range(num_devices)]

    for block_names in block_name_list:
        for block_name in block_names:
            params_dict = {}
            block_module = get_module(model, block_name)
            for n, m in block_module.named_modules():
                in_features, out_features = get_layer_features(m)
                if in_features is None:
                    continue
                params_dict[n] = in_features * out_features

            res_list = partition_dict_numbers(params_dict, num_devices)
            device_index = 0
            for res in res_list:
                for key in res.keys():
                    set_tuning_device_for_layer(block_module, key, gpu_devices[device_index])
                device_index += 1


if __name__ == "__main__":
    # Example usage
    number_dict = {"item1": 90, "item2": 20, "item3": 30, "item4": 40, "item5": 50, "item6": 60}

    groups = partition_dict_numbers(number_dict, 10)
    for i, group in enumerate(groups):
        print(f"Group {i + 1}: {group}, Sum: {sum(group.values())}")

    groups = partition_dict_numbers(number_dict, 6)
    for i, group in enumerate(groups):
        print(f"Group {i + 1}: {group}, Sum: {sum(group.values())}")

    groups = partition_dict_numbers(number_dict, 4)
    for i, group in enumerate(groups):
        print(f"Group {i + 1}: {group}, Sum: {sum(group.values())}")

    groups = partition_dict_numbers(number_dict, 3)
    for i, group in enumerate(groups):
        print(f"Group {i + 1}: {group}, Sum: {sum(group.values())}")

    groups = partition_dict_numbers(number_dict, 2)
    for i, group in enumerate(groups):
        print(f"Group {i + 1}: {group}, Sum: {sum(group.values())}")
