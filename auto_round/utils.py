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
import logging
import os
import sys
import subprocess
from collections import UserDict
import re
# for cpu usage
import cpuinfo
import numpy as np
import psutil
import torch
from torch.amp import autocast

from functools import lru_cache
from packaging import version
import gc
from .special_model_handler import shareable_keywords


@lru_cache(None)
def warning_once(self, msg: str):
    self.warning(msg)

class AutoRoundFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;1m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    _format = "%(asctime)s %(levelname)s %(filename)s L%(lineno)d: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + _format + reset,
        logging.INFO: grey + _format + reset,
        logging.WARNING: yellow + _format + reset,
        logging.ERROR: bold_red + _format + reset,
        logging.CRITICAL: bold_red + _format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, "%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

logging.Logger.warning_once = warning_once
logger = logging.getLogger("autoround")
logger.setLevel(logging.INFO)
logger.propagate = False
fh = logging.StreamHandler()
fh.setFormatter(AutoRoundFormatter())
logger.addHandler(fh)

import importlib
import transformers


class LazyImport(object):
    """Lazy import python module till use."""

    def __init__(self, module_name):
        """Init LazyImport object.

        Args:
           module_name (string): The name of module imported later
        """
        self.module_name = module_name
        self.module = None

    def __getattr__(self, name):
        """Get the attributes of the module by name."""
        try:
            self.module = importlib.import_module(self.module_name)
            mod = getattr(self.module, name)
        except:
            spec = importlib.util.find_spec(str(self.module_name + "." + name))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        return mod

    def __call__(self, *args, **kwargs):
        """Call the function in that module."""
        function_name = self.module_name.split(".")[-1]
        module_name = self.module_name.split(f".{function_name}")[0]
        self.module = importlib.import_module(module_name)
        function = getattr(self.module, function_name)
        return function(*args, **kwargs)


auto_gptq = LazyImport("auto_gptq")
htcore = LazyImport("habana_frameworks.torch.core")


def is_optimum_habana_available():
    from transformers.utils.import_utils import is_optimum_available

    return is_optimum_available() and importlib.util.find_spec("optimum.habana") is not None


def get_module(module, key):
    """Get module from model by key name.

    Args:
        module (torch.nn.Module): original model
        key (str): module name to be replaced
    """
    name_list = key.split(".")
    for name in name_list:
        module = getattr(module, name, None)
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


def unsupport_meta_device(model):
    """Checks if the model is a valid model for auto_round.

    Args:
    model: The model to be checked.

    Returns:
    bool: True if the model is valid, False otherwise. 
    """
    target_device = None
    for param in model.parameters():
        if target_device is None:
            target_device = param.device
        if param.device != target_device:
            if param.device.type == 'meta' or target_device.type == 'meta':
                return True
    if target_device.type == 'meta':
        if hasattr(model, "path"):
            return False
        else:
            return True
    return False


def to_device(input, device=torch.device("cpu")):
    """Moves input data to the specified device.

    Args:
    input: The input data to be moved.
    device: The target device.

    Returns:
    The input data on the specified device.
    """
    if input is None:
        return None
    if isinstance(input, torch.Tensor):
        return input.to(device)
    if isinstance(input, dict) or isinstance(input, UserDict):
        for inp in input.keys():
            input[inp] = to_device(input[inp], device)

    elif isinstance(input, list) or isinstance(input, tuple):
        if len(input) == 0:
            return input
        input_res = []
        for inp in input:
            input_res.append(to_device(inp, device))
        if isinstance(input, tuple):
            input_res = tuple(input_res)
        input = input_res

    return input


def mv_module_from_gpu(module, low_cpu_mem_usage=False):
    """Moves module from gpu to cpu or meta if low_cpu_mem_usage is true.

    Args:
    module: The module to be moved.
    low_cpu_mem_usage: Whether to use low CPU memory. If true, move module to meta.

    Returns:
    The module on the specified device.
    """
    if hasattr(module, "device"):
        target_device = "meta" if low_cpu_mem_usage else "cpu"
        if module.device.type == target_device:
            return module
        else:
            return module.to(target_device)
    else:
        if low_cpu_mem_usage:
            return module.to('meta')
        else:
            return module.to('cpu')


def to_dtype(input, dtype=torch.float32):
    """Moves input data to the specified data type.

    Args:
    input: The input data to be moved.
    dtype: The target data type.

    Returns:
    The input data on the specified data type.
    """
    if input is None:
        return None
    if isinstance(input, torch.Tensor):
        return input.to(dtype)
    if isinstance(input, dict) or isinstance(input, UserDict):
        for inp in input.keys():
            input[inp] = to_dtype(input[inp], dtype)

    elif isinstance(input, list) or isinstance(input, tuple):
        if len(input) == 0:
            return input
        input_res = []
        for inp in input:
            input_res.append(to_dtype(inp, dtype))
        if isinstance(input, tuple):
            input_res = tuple(input_res)
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


def validate_modules(module_names, quant_vision=False, vison_blocks_names=None):
    """
    Test a list of modules' validity.

    Args:
    modules (list of str): List of strings to be validated.

    Returns:
    bool: True if all modules have equal length or not dependent, otherwise False.
    """
    if not bool(module_names):  # pragma: no cover
        raise ValueError(f"Empty modules")
    if len(module_names) < 2:
        return True
    split_modules = [s.split('.') for s, _ in module_names]
    lengths = [len(parts) for parts in split_modules]
    if len(set(lengths)) == 1:  # pragma: no cover
        return True
    max_length = max(lengths)
    min_length = min(lengths)
    longest_module = next(s for s in split_modules if len(s) == max_length)
    shortest_module = next(s for s in split_modules if len(s) == min_length)
    shortest_module = '.'.join(shortest_module)
    longest_module = '.'.join(longest_module)
    # Check if the shortest name is a substring of the longest name
    if shortest_module in longest_module:  # pragma: no cover
        raise ValueError(f"Invalid modules, at least two modules detected" \
                         " as dependent, {shortest_module} and {longest_module}")
    flag = False
    if quant_vision:
        for n,_ in module_names:
            flag = any(key not in n.lower() for key in (vison_blocks_names))
    if quant_vision and not flag:
        raise ValueError(f"Cannot find the visual block. Please reset the quant_nontext_module parameter to False, " \
                         "or raise an issue at https://github.com/intel/auto-round/issues.")
    return


def find_matching_blocks(model, all_blocks, to_quant_block_names):
    """
    Find and return matching blocks in the model based on to_quant_block_names.
    
    Args:
        model: The model (not used in this specific function but kept for completeness).
        all_blocks: List of lists, where each inner list contains full block names in the model.
        to_quant_block_names: Comma-separated string of target block names to match.
    
    Returns:
        target_blocks: List of lists containing full paths of matching blocks in the model.
    """
    if not to_quant_block_names:
        return all_blocks
    to_quant_block_list = to_quant_block_names
    if isinstance(to_quant_block_names, list) or isinstance(to_quant_block_names,tuple):
        return to_quant_block_names
    if isinstance(to_quant_block_names, str):
        to_quant_block_list = [name.strip() for name in to_quant_block_names.split(",")]
    target_blocks = []
    for block_list in all_blocks:
        matched_sublist = []
        for name in to_quant_block_list:
            matches = [block for block in block_list if re.search(name, block)]
            if matches:
                matched_sublist.extend(matches)
        if matched_sublist:
            target_blocks.append(matched_sublist)
        if not target_blocks:
            raise ValueError("No block names matched. Please check the input for to_quant_block_name," \
                             "or set to_quant_block_name to None to automatically match quantizable blocks.")
    return target_blocks
    
    
def get_block_names(model):
    """Get the block names for transformers-like networks.

    Args:
    model: The model.

    Returns:
    block_names: A list whose elements are list of block's layer names
    """
    block_names = []
    target_modules = []
    for n, m in model.named_modules():
        if hasattr(type(m), "__name__") and "ModuleList" in type(m).__name__:
            target_modules.append((n, m))
            break  ## only find the first modulelist, may be not robust
    for i, target_m in enumerate(target_modules):
        block_names.append([])
        for n, m in target_m[1].named_children():
            block_names[i].append(target_m[0] + "." + n)
    return block_names


def get_multimodal_block_names(model, quant_vision=False):
    """Get the multimodal model block names for transformers-like networks.

    Args:
    model: The model.

    Returns:
    block_names: A list whose elements are list of block's layer names
    """
    block_names = []
    target_modules = []
    vison_blocks_tuple = ("vision", "visual",)
    for n, m in model.named_modules():
        if hasattr(type(m), "__name__") and "ModuleList" in type(m).__name__:
            if quant_vision or all(key not in n.lower() for key in (vison_blocks_tuple)):
                target_modules.append((n, m))
    validate_modules(target_modules, quant_vision, vison_blocks_tuple)
    for i, target_m in enumerate(target_modules):
        block_names.append([])
        for n, m in target_m[1].named_children():
            block_names[i].append(target_m[0] + "." + n)
    return block_names


def collect_best_params(block):
    params = {}
    for n, m in block.named_modules():
        if hasattr(m, "orig_layer"):
            params[n] = {}
            for key in m.params.keys():
                params[n][key] = copy.deepcopy(m.params[key].data)
    return params


@torch.no_grad()
def sampling_inputs(input_ids, input_others, indices, seqlen,
                    batch_dim=0):
    """Samples inputs based on the given indices and sequence length.

    Args:
    input_ids: The list of input tensor containing  input_ids.
    input_others: A dictionary containing other input data.
    indices: The indices to sample from the input.
    seqlen: The sequence length.

    Returns:
    current_input_ids: The sampled input IDs.
    current_input_others: The sampled other input data.
    """
    current_input_ids = [input_ids[i] for i in indices]

    current_input_ids = torch.cat(current_input_ids, dim=batch_dim)

    current_input_others = {"positional_inputs": input_others["positional_inputs"]}
    for key in input_others.keys():
        if "positional_inputs" in key:
            continue
        if (key not in shareable_keywords or len(indices) == 1) \
                and not isinstance(input_others[key], (str, bool, type(None))):
            current_input_others[key] = None
            if input_others[key] is not None:
                current_input_others[key] = [input_others[key][i] for i in indices]
                if len(indices) == 1:
                    current_input_others[key] = current_input_others[key][0]
                else:
                    try:
                        current_input_others[key] = torch.cat(current_input_others[key], dim=0)
                    except TypeError as err:
                        logger.warning_once("Please check the model cache inputs or try setting batch_size to 1.")
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
        input_ids = to_device(input_ids, device)
        input_others = to_device(input_others, device)
    input_tuple = input_others.pop("positional_inputs", None)
    if "alibi" in input_others.keys() and input_others["alibi"] is not None:
        alibi = input_others["alibi"]
        input_others["alibi"] = alibi.reshape(-1, alibi.shape[2], alibi.shape[3])
    if amp:
        with autocast(device_type=device.split(":")[0], dtype=amp_dtype):  # pragma: no cover
            output = block(input_ids, *input_tuple, **input_others)
    else:
        output = block(input_ids, *input_tuple, **input_others)
    if isinstance(output, list) or isinstance(output, tuple):
        output = output[0]
    return output


def check_to_quantized(config):
    """Checks if the configuration is valid for quantization.

    Args:
        config (dict or object): The configuration to check. It can be either a
            dictionary with a 'bits' key or an object with a 'bits' attribute.

    Returns:
        bool: True if the configuration is valid for quantization (bits <= 8),
            False otherwise.
    """
    if isinstance(config, dict):
        
        if int(config["bits"]) > 8:
            return False
        return True
    else:
        if int(config.bits) > 8:
            return False
        return True


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


def detect_device(device=None):
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
    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using GPU device")
        elif is_optimum_habana_available():  # pragma: no cover
            device = torch.device("hpu")
            logger.info("Using HPU device")
        # Use CPU as a fallback
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        if dev_idx is not None and str(device) != "cpu":
            device = str(device) + f":{dev_idx}"
        return str(device)
    elif isinstance(device, torch.device):
        device = str(device)
    return device


class CpuInfo(object):
    """Get CPU Info."""

    def __init__(self):
        """Get whether the cpu numerical format is bf16, the number of sockets, cores and cores per socket."""
        self._bf16 = False
        self._vnni = False
        info = cpuinfo.get_cpu_info()
        if "arch" in info and "X86" in info["arch"]:
            cpuid = cpuinfo.CPUID()
            max_extension_support = cpuid.get_max_extension_support()
            if max_extension_support >= 7:
                ecx = cpuid._run_asm(
                    b"\x31\xC9",  # xor ecx, ecx
                    b"\xB8\x07\x00\x00\x00" b"\x0f\xa2" b"\x89\xC8" b"\xC3",  # mov eax, 7  # cpuid  # mov ax, cx  # ret
                )
                self._vnni = bool(ecx & (1 << 11))
                eax = cpuid._run_asm(
                    b"\xB9\x01\x00\x00\x00",  # mov ecx, 1
                    b"\xB8\x07\x00\x00\x00" b"\x0f\xa2" b"\xC3",  # mov eax, 7  # cpuid  # ret
                )
                self._bf16 = bool(eax & (1 << 5))
        if "arch" in info and "ARM" in info["arch"]:  # pragma: no cover
            self._sockets = 1
        else:
            self._sockets = self.get_number_of_sockets()
        self._cores = psutil.cpu_count(logical=False)
        self._cores_per_socket = int(self._cores / self._sockets)

    @property
    def bf16(self):
        """Get whether it is bf16."""
        return self._bf16

    @property
    def vnni(self):
        """Get whether it is vnni."""
        return self._vnni

    @property
    def cores_per_socket(self):
        """Get the cores per socket."""
        return self._cores_per_socket

    def get_number_of_sockets(self) -> int:
        """Get number of sockets in platform."""
        cmd = "cat /proc/cpuinfo | grep 'physical id' | sort -u | wc -l"
        if psutil.WINDOWS:
            cmd = r'wmic cpu get DeviceID | C:\Windows\System32\find.exe /C "CPU"'
        elif psutil.MACOS:  # pragma: no cover
            cmd = "sysctl -n machdep.cpu.core_count"

        with subprocess.Popen(
                args=cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=False,
        ) as proc:
            proc.wait()
            if proc.stdout:
                for line in proc.stdout:
                    return int(line.decode("utf-8", errors="ignore").strip())
        return 0


def is_local_path(path):
    """Checks if a given path exists locally.

    Args:
        path (str): The path to check.

    Returns:
        bool: True if the path exists locally, False otherwise.
    """
    format_list = ("json", "txt",)
    flag = None
    for x in format_list:
        flag = True if x in path else flag
    return flag and os.path.exists(path)


def convert_dtype_str2torch(str_dtype):
    """Converts a string dtype to its corresponding PyTorch dtype.

    Args:
        str_dtype (str): The string representation of the dtype.

    Returns:
        torch.dtype: The PyTorch dtype.

    Raises:
        AssertionError: If the input str_dtype is unsupported.
    """
    if isinstance(str_dtype, torch.dtype) or str_dtype is None:
        return str_dtype
    if str_dtype == "int8":
        return torch.int8
    elif str_dtype == "fp32" or str_dtype == "float32" or str_dtype == "auto":
        return torch.float
    elif str_dtype == "fp16" or str_dtype == "float16":
        return torch.float16
    elif str_dtype == "bf16" or str_dtype == "bfloat16":
        return torch.bfloat16
    else:
        assert False, "Unsupported str dtype {} to torch dtype".format(str_dtype)


def convert_dtype_torch2str(dtype):
    """Converts a PyTorch dtype to its corresponding string representation.

    Args:
        dtype: PyTorch dtype or str. The dtype to convert.

    Returns:
        str: The string representation of the dtype.

    Raises:
        AssertionError: If the input dtype is unsupported.
    """
    if isinstance(dtype, str) or dtype is None:
        return dtype
    if dtype == torch.int8:
        return "int8"
    elif dtype == torch.float:
        return "fp32"
    elif dtype == torch.float16:
        return "fp16"
    elif dtype == torch.bfloat16:
        return "bf16"
    elif isinstance(dtype, str) and dtype in ["int8", "fp32", "fp16", "bf16"]:
        return dtype
    else:
        assert False, "Unsupported pytorch dtype {} to str dtype".format(dtype)


def convert_dtype_torch2str_hf(dtype):
    """Converts a PyTorch dtype to its corresponding huggingface string dtype, e.g. torch.float32 -> 'float32'.

    Args:
        dtype: PyTorch dtype or str. The dtype to convert.

    Returns:
         str: The string representation of the dtype.

    Raises:
        AssertionError: If the input str_dtype is unsupported.
    """
    if dtype is None:
        return dtype
    if isinstance(dtype, str):
        if "float" not in dtype and "int" not in dtype:
            dtype = convert_dtype_str2torch(dtype)
        else:
            return dtype
    str_dtype = str(dtype)
    if "." not in str_dtype:
        assert False, "Unsupported pytorch dtype {} to huggingface str dtype".format(dtype)
    str_dtype = str_dtype.split(".")[1]
    return str_dtype


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


def get_layer_names_in_block(model, supported_types=[torch.nn.Linear,
                                                     transformers.modeling_utils.Conv1D], to_quant_block_names=None):
    """Retrieves the names of layers within each block of the model.

    Returns:
        list: A list of strings, where each string is the name of a layer
              within a block of the model.
    """
    for n, m in model.named_modules():
        if isinstance(m, tuple(supported_types)):
            m.tmp_name = n
    layers_in_block = []
    if bool(to_quant_block_names):
        all_blocks = to_quant_block_names
    else:
        all_blocks = get_block_names(model)
    for block_names in all_blocks:
        for block_name in block_names:
            block = get_module(model, block_name)
            for n, m in block.named_modules():
                if hasattr(m, "tmp_name"):
                    layers_in_block.append(m.tmp_name)
    for n, m in model.named_modules():
        if hasattr(m, "tmp_name"):
            delattr(m, "tmp_name")
    return layers_in_block


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


@lru_cache(None)
def is_hpu_supported():  # pragma: no cover
    try:
        import habana_frameworks.torch.core as htcore  # pylint: disable=E0401
    except ImportError as e:
        return False
    return True


def get_library_version(library_name):
    from packaging.version import Version
    python_vesion = Version(sys.version.split()[0])
    if python_vesion < Version("3.8"):
        import warnings
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        import pkg_resources  # pylint: disable=E0401
        try:
            version = pkg_resources.get_distribution(library_name).version
            return version
        except pkg_resources.DistributionNotFound:
            return f"{library_name} is not installed"
    else:
        import importlib.metadata  # pylint: disable=E0401
        try:
            version = importlib.metadata.version(library_name)
            return version
        except importlib.metadata.PackageNotFoundError:
            return f"{library_name} is not installed"


def get_autogptq_packing_qlinear(backend, bits=4, group_size=128, sym=False):
    """
    Configures and returns a QuantLinear class based on the specified backend and parameters.

    Args:
        backend (str): The backend to be used for quantization. Supported values include "qigen", "triton", "marlin",
                       "exllama", and "cuda".
        bits (int, optional): The number of bits for quantization. Default is 4.
        group_size (int, optional): The group size for quantization. Default is 128.
        sym (bool, optional): Flag indicating whether to use symmetric quantization. Default is False.

    Returns:
        class: The dynamically imported QuantLinear class configured according to the specified parameters.
    """
    use_triton = True
    if bits not in [2, 4, 8]:
        use_triton = False
    disable_exllamav2 = True
    disable_exllamav1 = False
    disable_marlin = True
    use_qigen = False
    if "qigen" in backend:
        use_triton = False
        use_qigen = True
    elif "triton" in backend:
        use_triton = True
    elif "marlin" in backend and sym:
        use_triton = False
        disable_marlin = False
    elif "exllama" in backend:  ##need v1 code to export
        use_triton = True  ##same with triton
        disable_marlin = True
    elif "cuda" in backend:
        use_triton = False
        disable_marlin = True
        disable_exllamav2 = True
        disable_exllamav1 = True
    if use_triton:
        from auto_round.export.export_to_autogptq.qlinear_triton import QuantLinear
        return QuantLinear
    try:
        import auto_gptq  # pylint: disable=E0401
    except:
        logger.error(f"please install auto_gptq via 'pip install auto-gptq' to support exporting to {backend}")
        exit()

    from auto_gptq.utils.import_utils import dynamically_import_QuantLinear  # pylint: disable=E0401
    version = get_library_version("auto_gptq")
    from packaging.version import Version
    if Version(version) < Version("0.7.2"):
        QuantLinear = dynamically_import_QuantLinear(
            use_triton=use_triton,
            desc_act=False,
            group_size=group_size,
            bits=bits,
            disable_exllama=disable_exllamav1,
            disable_exllamav2=disable_exllamav2,
            use_qigen=use_qigen,
            disable_marlin=disable_marlin,
        )
    else:
        QuantLinear = dynamically_import_QuantLinear(  # pylint: disable=E1123
            use_triton=use_triton,
            desc_act=False,
            group_size=group_size,
            bits=bits,
            disable_exllama=disable_exllamav1,
            disable_exllamav2=disable_exllamav2,
            use_qigen=use_qigen,
            use_marlin=not disable_marlin,
        )
    return QuantLinear


def _clear_memory_for_cpu_and_cuda(tensor=None):
    if tensor is not None:
        del tensor
    gc.collect()
    torch.cuda.empty_cache()


def clear_memory(tensor=None):
    if is_hpu_supported():
        # hpu does not have empty_cache
        return
    else:
        _clear_memory_for_cpu_and_cuda(tensor)


def compare_versions(v1, v2):
    return version.parse(v1) >= version.parse(v2)


def torch_version_at_least(version_string):
    return compare_versions(torch.__version__, version_string)


TORCH_VERSION_AT_LEAST_2_6_PRE_RELEASE = torch_version_at_least("2.5.99")
TORCH_VERSION_AT_LEAST_2_6 = torch_version_at_least("2.6.0")
TORCH_VERSION_AT_LEAST_2_5 = torch_version_at_least("2.5.0")
TORCH_VERSION_AT_LEAST_2_4 = torch_version_at_least("2.4.0")


def check_hpu_compile_mode():
    assert (
            os.getenv("PT_HPU_LAZY_MODE") == "0"
    ), "Please set `PT_HPU_LAZY_MODE=0` to use HPU compile mode"
    # Note: this is a temporary solution, will be removed in the future
    assert (
            os.getenv("PT_ENABLE_INT64_SUPPORT") == "1"
    ), "Please set `PT_ENABLE_INT64_SUPPORT=1` to use HPU compile mode"


def compile_func_on_hpu(func):
    if TORCH_VERSION_AT_LEAST_2_4:
        check_hpu_compile_mode()
        return torch.compile(func, backend="hpu_backend")
    return func


def compile_func_on_cuda_or_cpu(func, enable_torch_compile):
    if enable_torch_compile or (TORCH_VERSION_AT_LEAST_2_6_PRE_RELEASE and enable_torch_compile!=False):
        return torch.compile(func)
    else:
        return func


def compile_func(fun, device, enable_torch_compile):
    if "hpu" in str(device):
        return compile_func_on_hpu(fun)  ## use auto by default
    else:
        return compile_func_on_cuda_or_cpu(fun, enable_torch_compile)



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
