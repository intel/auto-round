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
from functools import lru_cache
from typing import Any, Callable, Dict, List, Tuple, Union

import cpuinfo
import torch

from auto_round.logger import logger

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


def detect_device(device: Union[str, int, torch.device] = None) -> str:
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


def get_device_and_parallelism(device: Union[str, torch.device, int]) -> Tuple[str, bool]:
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
