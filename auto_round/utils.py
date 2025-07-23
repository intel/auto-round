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
import cpuinfo
import psutil
import torch
from torch.amp import autocast

from functools import lru_cache
from packaging import version
import gc
from auto_round.special_model_handler import SPECIAL_MULTIMODAL_BLOCK, SPECIAL_SHARED_CACHE_KEYS
import transformers
from auto_round.export.export_to_gguf.config import GGUF_CONFIG, GGML_QUANT_SIZES, GGUF_INNER_CONFIG, QK_K, ModelType

SHARED_CACHE_KEYS = ("position_ids", "cache_position", "position_embeddings")


class SupportedFormats:
    def __init__(self):
        self._support_format = (
            "auto_round", "auto_gptq", "auto_awq", "auto_round:auto_gptq", "auto_round:gptqmodel",
            "auto_round:auto_awq", "itrex", "itrex_xpu", "fake")
        self._gguf_format = tuple(sorted(GGUF_CONFIG.keys()))
        self._support_list = self._support_format + self._gguf_format

    def __contains__(self, key):
        return True if key in self._support_list else False

    def __str__(self):
        ##return "(%s)" % ', '.join(self._support_format + ("gguf:q*_0", "gguf:q*_1", "gguf:q*_k_s"))
        return "(%s)" % ', '.join(self._support_list)

    def __getitem__(self, key):
        return self._support_list[key]


SUPPORTED_FORMATS = SupportedFormats()

SUPPORTED_LAYER_TYPES = (torch.nn.Linear, transformers.pytorch_utils.Conv1D)

SUPPORTED_DTYPES = ("int", "mx_fp", "fp", "nv_fp")


def infer_bits_by_data_type(data_type: str):
    for supported_dtype in SUPPORTED_DTYPES:
        if data_type.startswith(supported_dtype) and len(data_type) > len(supported_dtype):
            ##first check the following two bits
            suc_2str = data_type[len(supported_dtype):len(supported_dtype) + 2]
            if str.isdigit(suc_2str):
                return int(suc_2str)
            if str.isdigit(data_type[len(supported_dtype)]):
                return int(data_type[len(supported_dtype)])
    return 16


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
    if group_size == 0:
        return (1)
    elif group_size == -1 or weight.shape[1] < group_size:
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


def get_common_prefix(paths):
    # Split each path into components and find the common prefix
    split_paths = [path.split('.') for path in paths]
    common_prefix = split_paths[0]
    for path in split_paths[1:]:
        common_prefix = [comp for comp, other in zip(common_prefix, path) if comp == other]
    return '.'.join(common_prefix)


def extract_block_names_to_str(quant_block_list):
    if not isinstance(quant_block_list, (list, tuple)):
        return None
    # Extract common prefix for each list
    prefixes = [get_common_prefix(blocks) for blocks in quant_block_list]
    # Join prefixes into a single string
    return ','.join(prefixes)


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
    if isinstance(to_quant_block_names, list) or isinstance(to_quant_block_names, tuple):
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


def get_block_names(model, quant_vision=False):
    """Get the block names for transformers-like networks.

    Args:
    model: The model.

    Returns:
    block_names: A list whose elements are list of block's layer names
    """

    def _get_llm_block_names(model):
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

    def _get_vlm_block_names(model, quant_vision=False):
        if hasattr(model, "config") and model.config.model_type in SPECIAL_MULTIMODAL_BLOCK.keys():
            return SPECIAL_MULTIMODAL_BLOCK.get(model.config.model_type)(model, quant_vision=quant_vision)
        block_names = []
        target_modules = []
        vision_blocks_tuple = ("vision", "visual", "image", "img")
        last_block_name = ""
        for n, m in model.named_modules():
            if hasattr(type(m), "__name__") and "ModuleList" in type(m).__name__:
                if quant_vision or all(key not in n.lower() for key in (vision_blocks_tuple)):
                    if last_block_name and last_block_name in n:
                        continue
                    target_modules.append((n, m))
                    last_block_name = n
        for i, target_m in enumerate(target_modules):
            block_names.append([])
            for n, m in target_m[1].named_children():
                block_names[i].append(target_m[0] + "." + n)
        return block_names

    if quant_vision or not is_pure_text_model(model):
        return _get_vlm_block_names(model, quant_vision=quant_vision)
    else:
        return _get_llm_block_names(model)


def collect_best_params(block):
    params = {}
    for n, m in block.named_modules():
        if hasattr(m, "orig_layer"):
            params[n] = {}
            for key in m.params.keys():
                params[n][key] = copy.deepcopy(m.params[key].data)
    return params


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
        bits = int(config.get("bits", 16))
        act_bits = int(config.get("act_bits", 16))
    else:
        bits = int(config.bits) if hasattr(config, "bits") else 16
        act_bits = int(config.act_bits) if hasattr(config, "act_bits") else 16

    return bits <= 8 or act_bits <= 8


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
            # logger.info("Using GPU device")
        elif is_optimum_habana_available():  # pragma: no cover
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
        ValueError: If the input str_dtype is unsupported.
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
        raise ValueError(f"Unsupported string dtype '{str_dtype}' for conversion to torch dtype.")


def convert_dtype_torch2str(dtype):
    """Converts a PyTorch dtype to its corresponding string representation.

    Args:
        dtype: PyTorch dtype or str. The dtype to convert.

    Returns:
        str: The string representation of the dtype.

    Raises:
        ValueError: If the input dtype is unsupported.
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
        raise ValueError(f"Unsupported PyTorch dtype '{dtype}' for conversion to string dtype.")


def convert_dtype_torch2str_hf(dtype):
    """Converts a PyTorch dtype to its corresponding huggingface string dtype, e.g. torch.float32 -> 'float32'.

    Args:
        dtype: PyTorch dtype or str. The dtype to convert.

    Returns:
         str: The string representation of the dtype.

    Raises:
        ValueError: If the input str_dtype is unsupported.
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
        raise ValueError(f"Unsupported pytorch dtype '{dtype}' for conversion to huggingface str dtype")
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


def get_layer_names_in_block(model, supported_types=(torch.nn.Linear,
                                                     transformers.pytorch_utils.Conv1D), quant_block_list=None):
    """Retrieves the names of layers within each block of the model.

    Returns:
        list: A list of strings, where each string is the name of a layer
              within a block of the model.
    """
    for n, m in model.named_modules():
        if isinstance(m, supported_types):
            m.tmp_name = n
    layers_in_block = []
    if bool(quant_block_list):
        all_blocks = quant_block_list
    else:
        all_blocks = get_block_names(model)
    for block_names in all_blocks:
        for block_name in block_names:
            block = get_module(model, block_name)
            for n, m in block.named_modules():
                if hasattr(m, "tmp_name"):
                    layers_in_block.append(m.tmp_name)
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


# Note on HPU usage:
# There are two modes available for enabling auto-round on HPU:
# 1. Compile Mode
#   1) Use PyTorch version ≥ 2.4 (Intel® Gaudi® v1.18 or later)
#   2) Set `PT_HPU_LAZY_MODE=0` and `PT_ENABLE_INT64_SUPPORT=1`
#   The compile mode can speed up quantization process but still in experimental stage.
# 2. Lazy Mode (By default)


def is_hpu_lazy_mode():
    return os.getenv("PT_HPU_LAZY_MODE") != "0"


def _use_hpu_compile_mode():
    return TORCH_VERSION_AT_LEAST_2_4 and not is_hpu_lazy_mode()


def compile_func_on_hpu(func):
    if _use_hpu_compile_mode():
        return torch.compile(func, backend="hpu_backend")
    return func


def compile_func_on_cuda_or_cpu(func):
    return torch.compile(func)


def compile_func(fun, device):
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


def get_fp_layer_names(model, fp_layers):
    """Identifies and returns layers in the model to exclude from quantization.

    This function processes a comma-separated list of fully precision (FP) layers,
    matches them to the names of layers in the model, and returns a list of such
    layers to exclude from quantization.

    Args:
        model (torch.nn.Module): The model whose layers will be inspected.
        fp_layers (str): A comma-separated string of layer names to be excluded
            from quantization. Whitespace is ignored in this string.

    Returns:
        list: A list of layer names that match the specified FP layers or are
        subcomponents of those layers.
    """
    fp_layers = fp_layers.replace(" ", "").split(",")
    all_layer_names = []
    for n, m in model.named_modules():
        if isinstance(m, (torch.nn.Linear, transformers.pytorch_utils.Conv1D)):
            all_layer_names.append(n)
    not_to_quantized_layers = []

    for fp_layer in fp_layers:
        if fp_layer == "":
            continue
        if fp_layer in all_layer_names:
            not_to_quantized_layers.append(fp_layer)
            continue
        if fp_layer[-1].isdigit():
            fp_layer = fp_layer + "."  ##tricky setting
        for name in all_layer_names:
            if fp_layer in name:
                not_to_quantized_layers.append(name)

    return not_to_quantized_layers


def check_awq_gemm_compatibility(model, bits, group_size, sym, layer_configs=None):
    """Checks if a model is compatible with the AutoAWQ GEMM kernel.

    Args:
        model: The model object to evaluate, typically a PyTorch model.
        bits (int): The number of bits for quantization (must be 4 for compatibility).
        group_size (int): The group size for quantization.
        sym (bool): Whether symmetric quantization is used (not utilized in the current function logic).
        layer_configs (dict, optional): A dictionary mapping layer names to configurations, where each
            configuration can specify a custom number of bits for the layer.

    Returns:
        tuple: A tuple containing:
            - bool: `True` if the model is compatible, `False` otherwise.
            - str: An error message describing why the model is incompatible, or an empty string if compatible.
    """
    if bits != 4:
        return False, f"AutoAWQ GEMM kernel only supports 4 bits"
    for n, m in model.named_modules():
        if isinstance(m, transformers.pytorch_utils.Conv1D):
            return False, "AutoAWQ GEMM kernel does not support conv1d"

    layer_names = get_layer_names_in_block(model)
    for layer_name in layer_names:
        if layer_configs is not None and layer_name in layer_configs.keys() and layer_configs[layer_name].get("bits",
                                                                                                              bits) > 8:
            continue

        layer = get_module(model, layer_name)
        if layer.in_features % group_size != 0:
            return False, f"Layer {layer_name} in_features is not multiple of group_size {group_size}"
        if layer.out_features % (32 // bits) != 0:
            return False, f"Layer {layer_name} out_features is not multiple of 32 // bits"

    return True, ""


def get_device_and_parallelism(device):
    from auto_round.utils import detect_device
    devices = device.replace(" ", "").split(',')
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
    devices = device.replace(" ", "").split(',')
    if all(s.isdigit() for s in devices):
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            current_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            current_visible_devices = current_visible_devices.split(',')
            indices = [int(device) for device in devices]
            try:
                pick_device = [current_visible_devices[i] for i in indices]
            except:
                raise ValueError(
                    "Invalid '--device' value: It must be smaller than the number of available devices."
                    " For example, with CUDA_VISIBLE_DEVICES=4,5, "
                    "--device 0,1 is valid, but --device 4,5 is not supported.")
            visible_devices = ','.join(pick_device)
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = device


def is_debug_mode():
    """Checks if the Python interpreter is running in debug mode.

    Returns:
        bool: True if debugging is enabled, False otherwise.
    """
    return sys.gettrace() is not None or sys.flags.debug == 1


def get_layer_features(layer):
    """Extracts input and output feature dimensions for supported layers."""
    if isinstance(layer, torch.nn.Linear):
        return layer.in_features, layer.out_features
    elif isinstance(layer, transformers.pytorch_utils.Conv1D):  # TODO: Verify correctness
        return layer.weight.shape[0], layer.weight.shape[1]
    elif isinstance(layer, torch.nn.Embedding):
        return layer.num_embeddings, layer.embedding_dim
    return None, None  # Unsupported layer type


def _gguf_args_check(args_or_ar, format_str=None, model_type=ModelType.TEXT):
    from auto_round.utils import logger
    import argparse

    if format_str is None:
        args_or_ar.format = args_or_ar.format.replace("q*_", f"q{args_or_ar.bits}_")
        format_str = args_or_ar.format
    else:
        format_str = format_str.replace("q*_", f"q{args_or_ar.bits}_")
    formats = format_str.lower().replace(' ', '').split(",")
    formats = sorted(formats, key=lambda x: len(x))
    for f in formats:
        if f.startswith("gguf") and f not in GGUF_CONFIG:
            logger.error(f"{f} is not supported, please check.")
    pattern = re.compile("q\d_k")
    pre_dq_format = ""
    unsupport_list, reset_list = [], []
    for format in GGUF_CONFIG:
        if format in formats:
            if format == "q6_k_s":
                logger.warning("Please note that q6_k_s is q6_k.")
            try:
                from auto_round.export.export_to_gguf.convert import ModelBase
            except:
                raise ImportError(
                    f"Please use the latest gguf-py for {format}, you can use the following command to install it:\n"
                    "git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp/gguf-py && pip install .")
            if re.search(pattern, format):
                if pre_dq_format and re.search(pattern, format).group() not in pre_dq_format:
                    logger.error(f"Cannot export {pre_dq_format} and {format} at the same time.")
                    sys.exit(-1)
                else:
                    pre_dq_format = format

            if isinstance(args_or_ar.model, str) and os.path.isdir(args_or_ar.model):
                from pathlib import Path
                from auto_round.export.export_to_gguf.convert import ModelBase
                hparams = ModelBase.load_hparams(Path(args_or_ar.model))
                model_architecture = hparams["architectures"][0]
                try:
                    model_class = ModelBase.from_model_architecture(model_architecture, model_type=model_type)
                except NotImplementedError:
                    logger.error(f"Model {model_architecture} is not supported to export GGUF format")
                    sys.exit(1)

                if re.search(pattern, format) and ("hidden_size" in hparams and hparams["hidden_size"] % QK_K != 0):
                    model_name = args_or_ar.model.split('/')
                    model_name = model_name[-1] if model_name[-1] else model_name[-2]
                    hidden_size = hparams["hidden_size"]
                    logger.error(
                        f"Currently only support pure mode for format: {format}. "
                        f"{model_name} is not supported, cause hidden_size({hidden_size}) % 256 !=0")
                    sys.exit(-1)

            unsupport_list, reset_list = [], []
            gguf_config = GGUF_CONFIG[format]
            for k, v in gguf_config.items():
                if not hasattr(args_or_ar, k):
                    continue
                if k == "data_type":
                    if re.search("q\d_1", format) and len(formats) > 1:
                        v = "int"
                if k == "sym" and isinstance(args_or_ar, argparse.Namespace):
                    k = "asym"
                    v = not v
                if getattr(args_or_ar, k) != v:
                    unsupport_list.append(f"{k}={getattr(args_or_ar, k)}")
                    reset_list.append(f"{k}={v}")
                    setattr(args_or_ar, k, v)
            if len(unsupport_list) > 0:
                logger.info(
                    f"format {format} does not support for {', '.join(unsupport_list)},"
                    f" reset to {', '.join(reset_list)}.")
# Removed obsolete commented-out block for improved readability and maintainability.
    return args_or_ar


def _to_model_dtype(model, model_dtype):
    if model_dtype is not None:
        try:
            if model_dtype == "float16" or model_dtype == "fp16":
                model = model.to(torch.float16)
            elif model_dtype == "bfloat16" or model_dtype == "bfp16" or model_dtype == "bf16":
                model = model.to(torch.bfloat16)
            elif model_dtype == "float32" or model_dtype == "fp32":
                model = model.to(torch.float32)
        except:
            logger.error("please use more device to fit the device or just use one device")
            exit()
    return model


def llm_load_model(
        pretrained_model_name_or_path,
        torch_dtype="auto",
        use_auto_mapping=True,
        trust_remote_code=True,
        model_dtype=None,
        device="cpu",
        low_cpu_mem_mode=0,
        low_cpu_mem_tmp_dir=None,
        **kwargs):
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

    is_glm = bool(re.search("chatglm", pretrained_model_name_or_path.lower()))
    low_cpu_mem_usage = False

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust_remote_code)

    model_cls = AutoModel if is_glm else AutoModelForCausalLM

    if low_cpu_mem_tmp_dir is None:
        low_cpu_mem_tmp_dir = "low_cpu_mem_tmp"
    if low_cpu_mem_mode == 2:
        from auto_round.low_cpu_mem.utils import load_model_with_hooks
        model = load_model_with_hooks(
            pretrained_model_name_or_path,
            model_cls,
            device=device,
            clean_weight=True,
            saved_path=low_cpu_mem_tmp_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code)
    elif low_cpu_mem_mode == 1:
        from auto_round.low_cpu_mem.utils import load_empty_model
        low_cpu_mem_usage = True
        model = load_empty_model(
            pretrained_model_name_or_path,
            model_cls,
            device=device,
            saved_path=low_cpu_mem_tmp_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code)
    else:
        if _use_hpu_compile_mode():
            model = model_cls.from_pretrained(
                pretrained_model_name_or_path, torch_dtype=torch_dtype,
                attn_implementation="eager",
                trust_remote_code=trust_remote_code, device_map="auto" if use_auto_mapping else None
            )
        else:
            try:
                model = model_cls.from_pretrained(
                    pretrained_model_name_or_path, torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code, device_map="auto" if use_auto_mapping else None
                )
            except OSError as e:
                logger.warning(
                    f"fail to load {pretrained_model_name_or_path}, set trust_remote_code to False and retry.")
                model = model_cls.from_pretrained(
                    pretrained_model_name_or_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=False,
                    device_map="auto" if use_auto_mapping else None)

    model = model.eval()
    model = _to_model_dtype(model, model_dtype)

    return model, tokenizer, low_cpu_mem_usage


def mllm_load_model(
        pretrained_model_name_or_path,
        torch_dtype="auto",
        use_auto_mapping=True,
        trust_remote_code=True,
        model_dtype=None,
        **kwargs):
    import json
    import transformers
    from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, AutoModel
    from huggingface_hub import HfApi, hf_hub_download, HfFileSystem

    if os.path.isdir(pretrained_model_name_or_path):
        config = json.load(open(os.path.join(pretrained_model_name_or_path, "config.json")))
    else:
        from huggingface_hub import hf_hub_download, list_repo_files
        file_list = list_repo_files(pretrained_model_name_or_path)
        if "config.json" in file_list:
            # Load plain JSON
            config_path = hf_hub_download(pretrained_model_name_or_path, "config.json")
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        elif "config.json.gz" in file_list:
            # Load gzipped JSON
            import gzip
            config_path = hf_hub_download(pretrained_model_name_or_path, "config.json.gz")
            with gzip.open(config_path, "rt", encoding="utf-8") as f:
                config = json.load(f)
        else:
            raise FileNotFoundError(f"No config.json or config.json.gz found for {pretrained_model_name_or_path}")

    if "model_type" in config:
        model_type = config["model_type"]
    else:
        model_type = None

    processor, image_processor = None, None
    if "deepseek_vl_v2" == model_type:
        from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM  # pylint: disable=E0401
        processor = DeepseekVLV2Processor.from_pretrained(pretrained_model_name_or_path)
        tokenizer = processor.tokenizer
        model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map="auto" if use_auto_mapping else None)
    else:
        architectures = config["architectures"][0]
        if architectures == "LlavaLlamaForCausalLM":
            from llava.model.builder import load_pretrained_model  # pylint: disable=E0401
            tokenizer, model, image_processor, _ = load_pretrained_model(
                pretrained_model_name_or_path,
                model_base=None,
                model_name=pretrained_model_name_or_path,
                torch_dtype=torch_dtype)
        else:
            if architectures.endswith("Model") \
                    and hasattr(transformers, n := architectures.replace("Model", "ForConditionalGeneration")):
                cls = getattr(transformers, n)
            elif hasattr(transformers, architectures):
                cls = getattr(transformers, architectures)
            else:
                cls = AutoModelForCausalLM
            model = cls.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
                device_map="auto" if use_auto_mapping else None)
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
            processor = AutoProcessor.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
            try:
                from transformers import AutoImageProcessor
                image_processor = AutoImageProcessor.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code)
            except Exception as e:
                pass

    model = model.eval()
    model = _to_model_dtype(model, model_dtype)

    return model, processor, tokenizer, image_processor


def is_pure_text_model(model):
    """verify on: phi-3.5, Mistral-Small-3.1, gemma-3, qwen2-vl, """
    if hasattr(model, "config") and hasattr(model.config, "vision_config"):
        return False
    if hasattr(model.__class__, "main_input_name") and model.__class__.main_input_name != "input_ids":
        return False
    for module in model.modules():
        if hasattr(module.__class__, "main_input_name") and module.__class__.main_input_name != "input_ids":
            return False
        if "vision" in str(module.__class__).lower():
            return False
        if "image" in str(module.__class__).lower():
            return False
        if "img" in str(module.__class__).lower():
            return False
    return True


def reset_params(inputs):
    """
    Resets specific input parameters to avoid saving the key-value cache during fine-tuning.

    Args:
        inputs (dict): Dictionary of model inputs.

    Modifies:
        inputs (dict): Sets "use_cache" to False if the key is present.
    """
    if "use_cache" in inputs.keys():  # Not storing kv cache
        inputs['use_cache'] = False


def check_skippable_keywords(key):
    """
    Prints a reminder if a key is not stored during quantization fine-tuning.
    """
    skippable_cache_keys = ("past_key_value",)
    for cache_key in skippable_cache_keys:
        if cache_key not in key:
            return True
    return False


def init_cache(positional_inputs, inputs):
    """
    Initializes special model inputs by adding positional inputs if missing.

    Args:
        positional_inputs (list): List of positional inputs to add to inputs.
        inputs (dict): Dictionary of model inputs.

    Modifies:
        inputs (dict): Adds "positional_inputs" key if not present.
    """
    if "positional_inputs" not in inputs:  # for chatglm Series
        inputs["positional_inputs"] = []
    for idx, item in enumerate(positional_inputs):
        inputs["positional_inputs"] = to_device(positional_inputs)


def get_shared_keys(model):
    """
    Retrieves shared keys from the model's state dictionary.

    Args:
        model (torch.nn.Module): The model to retrieve shared keys from.

    Returns:
        tuple: tuple of shared keys.
    """
    shared_keys = SHARED_CACHE_KEYS
    shared_keys += SPECIAL_SHARED_CACHE_KEYS.get(model.__class__.__name__, ())
    return shared_keys


def get_model_dtype(model_dtype, default="auto"):
    if model_dtype is None or model_dtype == "auto":
        model_dtype = default
    elif model_dtype in ["bf16", "bfloat16"]:
        model_dtype = "bfloat16"
    elif model_dtype in ["f16", "float16", "fp16"]:
        model_dtype = "float16"
    elif model_dtype in ["f32", "float32", "fp32"]:
        model_dtype = "float32"
    else:
        logger.warning(f"Unable to identify model_dtype {model_dtype}, reset to default model_dtype {default}")
        model_dtype = default
    return model_dtype


def str2bool(v):
    import argparse
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def filter_quantization_config(quantization_config):
    default_dict = {"amp": True, "batch_size": 8, "data_type": int, "dataset": "NeelNanda/pile-10k",
                    "enable_minmax_tuning": True, "enable_norm_bias_tuning": False, "enable_quanted_input": True,
                    "gradient_accumulate_steps": 1, "iters": 200, "low_gpu_mem_usage": False, "nsamples": 128,
                    "scale_dtype": "torch.float16", "seqlen": 2048}
    iters = quantization_config.get("iters", 200)

    default_dict["lr"] = 1.0 / iters if iters > 0 else 5e-3
    default_dict["minmax_lr"] = default_dict["lr"]

    for key in default_dict:
        if key in quantization_config and default_dict[key] == quantization_config[key]:
            quantization_config.pop(key)
    for k in list(quantization_config.keys()):
        if quantization_config[k] is None:
            quantization_config.pop(k)

    if quantization_config.get("act_bits", 16) >= 16:
        quantization_config.pop("act_bits", None)
        quantization_config.pop("act_data_type", None)
        quantization_config.pop("act_dynamic", None)
        quantization_config.pop("act_sym", None)
        quantization_config.pop("act_group_size", None)


def check_start_with_block_name(name: str, block_name_to_quantize: list):
    """
    Checks if the given layer name starts with any of the block names to be quantized.

    Args:
        name (str): The name of the layer.
        block_name_to_quantize (list): A list of block names to check against.

    Returns:
        bool: True if the layer name starts with any of the block names, False otherwise.
    """
    for block_name in block_name_to_quantize:
        if name.startswith(block_name):
            return True
    return False


def check_seqlen_compatible(input_seqlen, tokenizer=None, model=None):
    """
    Check whether the input sequence length is within the limits defined
    by the tokenizer and the model configuration.

    Args:
        input_seqlen (int): The length of the input sequence.
        tokenizer: Optional, a HuggingFace tokenizer object.
        model: Optional, a HuggingFace model object.

    Returns:
        ValueError: if the input length is not valid, riase Error.
    """
    if model is not None and hasattr(model, 'config'):
        model_config = model.config
        if hasattr(model_config, 'max_position_embeddings') and input_seqlen > model_config.max_position_embeddings:
            raise ValueError(f"seqlen({input_seqlen}) exceeds model.config.max_position_embeddings(" \
                             f"{model_config.max_position_embeddings}). Please lowering '--seqlen'")
    if tokenizer is not None and hasattr(tokenizer, 'model_max_length') and input_seqlen > tokenizer.model_max_length:
        raise ValueError(f"seqlen({input_seqlen}) exceeds tokenizer.model_max_length({tokenizer.model_max_length}). " \
                         "Please oncider Consider lowering the '--seqlen' or increasing tokenizer.model_max_length.")


def _use_more_bits(i_layer: int, n_layer: int):
    return (i_layer < n_layer // 8) or (i_layer >= 7 * n_layer // 8) or ((i_layer - n_layer // 8) % 3 == 2)


def _get_digital_in_layer_name(layer_name):
    pattern = re.compile("([a-zA-Z]+\.){1,}(\d+)")
    res = re.search(pattern, layer_name)
    if res:
        return int(res[2])
    else:
        return None

def _search_gguf_type(gguf_type):
    if gguf_type in GGUF_INNER_CONFIG:
        return gguf_type
    pattern = re.compile("gguf:q([0-9]{1,})_[01k]")
    bits = re.search(pattern, gguf_type)
    if not bits:
        raise KeyError(f"{gguf_type} is not a correct gguf type, please check")

    for suffix in ["_k", "_0", "_1"]:
        if gguf_type.endswith(suffix):
            continue
        if (tmp_type := re.sub("_[01k]", suffix, gguf_type)) in GGUF_INNER_CONFIG:
            return tmp_type
    return None

def _gguf_type_fallback(gguf_type):
    if gguf_type in ("gguf:q2_k", "gguf:q3_k", "gguf:q4_k"):
        gguf_type = "gguf:q5_0"
    elif gguf_type == "gguf:q5_k":
        gguf_type = "gguf:q5_0"
    elif gguf_type == "gguf:q6_k":
        gguf_type = "gguf:q8_0"
    return gguf_type

##https://github.com/ggml-org/llama.cpp/blob/9e31bec4fd53634c9e5b04650488a09a055f5dab/src/llama-quant.cpp#L129
def get_layer_config_by_gguf_format(layer_config, gguf_format, model, model_type=ModelType.TEXT):
    # TODO: support for other format later
    target_gguf_format = next((fmt for fmt in gguf_format if fmt != "fake"), None)

    import gguf  # pylint: disable=E0401
    from auto_round.export.export_to_gguf.convert import ModelBase, get_model_architecture
    model_architecture = get_model_architecture(hparams=model.config.to_dict(), model_type=model_type)
    try:
        model_class = ModelBase.from_model_architecture(model_architecture, model_type=model_type)
    except NotImplementedError:
        return layer_config, {}

    n_layer = None
    for name in ["n_layers", "num_hidden_layers", "n_layer", "num_layers"]:
        sub_attr = "text_config" if model_type == ModelType.TEXT else "vision_config"
        if hasattr(model.config, name):
            n_layer = getattr(model.config, name)
            break
        if hasattr(model.config, sub_attr):
            if hasattr(getattr(model.config, sub_attr), name):
                n_layer = getattr(getattr(model.config, sub_attr), name)
                break
    if n_layer is None:
        return layer_config, {}

    tensor_map = gguf.get_tensor_name_map(model_class.model_arch, n_layer)

    def _set_config(config, target_config):
        for k, v in target_config.items():
            if isinstance(config, dict):
                config[k] = v
            else:
                setattr(config, k, v)
        return config

    gguf_format_config = {}
    lm_head_name = get_lm_head_name(model)
    inner_gguf_format = GGUF_CONFIG[target_gguf_format]["mostly"]
    # ggml_type =  getattr(gguf.GGMLQuantizationType,inner_gguf_format.split(":")[-1].upper())
    block_size = GGML_QUANT_SIZES[inner_gguf_format.split(":")[-1].lower()][0]
    tie_word_embeddings = True
    if hasattr(model, "config") and hasattr(model.config, "tie_word_embeddings"):
        tie_word_embeddings = model.config.tie_word_embeddings

    n_gqa = 1
    if hasattr(model, "config") and hasattr(model.config, "num_attention_heads") and hasattr(model.config,
                                                                                             "num_key_value_heads"):
        n_gqa = model.config.num_attention_heads // model.config.num_key_value_heads
    n_expert = 0
    for name in ["num_experts", "num_local_experts", "n_routed_experts"]:
        if hasattr(model.config, name):
            n_expert = getattr(model.config, name)

    i_attention_wv = 0
    i_ffn_down = 0
    layer_config_copy = copy.deepcopy(layer_config)
    target_bits = None
    if inner_gguf_format.startswith("gguf:q") and len(inner_gguf_format) >= 7 and (inner_gguf_format[6]).isdigit():
        target_bits = int(inner_gguf_format[6])

    for layer_name, config in layer_config_copy.items():
        if not check_to_quantized(config):
            continue
        new_type = GGUF_CONFIG[target_gguf_format]["mostly"]
        layer = get_module(model, layer_name)
        if isinstance(layer, transformers.pytorch_utils.Conv1D):
            input_features = layer.weight.shape[0]
        else:
            input_features = layer.weight.shape[-1]
        i_layer = _get_digital_in_layer_name(layer_name)

        if lm_head_name is not None and layer_name == lm_head_name:
            target_bits = int(re.search("gguf:q([0-9]{1,})_[01k]", GGUF_CONFIG[target_gguf_format]['lm_head']).group(1))
        if isinstance(layer, torch.nn.Embedding):
            target_bits = int(
                re.search("gguf:q([0-9]{1,})_[01k]", GGUF_CONFIG[target_gguf_format]['embedding']).group(1))

        gguf_name = tensor_map.get_name(layer_name)
        bits_index = 6
        if config.get("fixed_by_user", False):
            if "bits" not in config:
                logger.warning(
                    f"Setting layer_config requires providing bits, {layer_name} has not bits,"
                    f" using bits={target_bits} instead.")
                new_type = new_type[:bits_index] + target_bits + new_type[bits_index + 1:]
            else:
                new_type = new_type[:bits_index] + str(config["bits"]) + new_type[bits_index + 1:]
            new_type = _search_gguf_type(new_type)
            if new_type is None:
                raise ValueError(f"invalid bit setting for {layer_name}")
        elif (target_bits is not None and
                "bits" in config and config["bits"] != target_bits):
            new_type = new_type[:bits_index] + str(config["bits"]) + new_type[bits_index + 1:]
            new_type = _search_gguf_type(new_type)
            if new_type is None:
                raise ValueError(f"invalid bit setting for {layer_name}")
        elif lm_head_name is not None and layer_name == lm_head_name and not tie_word_embeddings:
            if gguf.MODEL_ARCH.FALCON == model_class.model_arch or input_features % block_size != 0:
                new_type = "gguf:q8_0"
            elif "lm_head" in GGUF_CONFIG[target_gguf_format]:
                new_type = GGUF_CONFIG[target_gguf_format]["lm_head"]
            elif new_type != "gguf:q8_0":
                new_type = "gguf:q6_k"
        elif lm_head_name is not None and layer_name == lm_head_name and tie_word_embeddings:
            pass
        elif isinstance(layer, torch.nn.Embedding):
            if "embedding" in GGUF_CONFIG[target_gguf_format]:
                new_type = GGUF_CONFIG[target_gguf_format]["embedding"]
        elif gguf_name is None:
            pass
        # attn_v
        elif "attn_v" in gguf_name:
            if target_gguf_format == "gguf:q2_k":
                new_type = "gguf:q4_k" if n_gqa >= 4 else "gguf:q3_k"
            elif target_gguf_format == "gguf:q2_k_s" and n_gqa >= 4:
                new_type = "gguf:q4_k"
            elif target_gguf_format == "gguf:q3_k_m":
                new_type = "gguf:q5_k" if i_attention_wv < 2 else "gguf:q4_k"
            elif target_gguf_format == "gguf:q3_k_l":
                new_type = "gguf:q5_k"
            elif (target_gguf_format == "gguf:q4_k_m" or target_gguf_format == "gguf:q5_k_m") and _use_more_bits(
                    i_layer, n_layer):
                new_type = "gguf:q6_k"
            elif target_gguf_format == "gguf:q4_k_s" and i_attention_wv < 4:
                new_type = "gguf:q5_k"
            ##TODO check which models are be grouped into to LLM_TYPE_70B
            # if (qs.model.type == LLM_TYPE_70B) {
            # // In the 70B model we have 8 heads sharing the same attn_v weights.
            # As a result, the attn_v.weight tensor is
            # // 8x smaller compared to attn_q.weight.Hence, we can get a nice boost in quantization accuracy with
            # // nearly negligible increase in model size by quantizing this tensor with more bits:
            #     if
            # (new_type == GGML_TYPE_Q3_K | | new_type == GGML_TYPE_Q4_K)
            # new_type = GGML_TYPE_Q5_K;
            # }
            if n_expert == 8:
                new_type = "gguf:q8_k"
            i_attention_wv += 1

        elif "attn_k" in gguf_name:
            if n_expert == 8:
                new_type = "gguf:q8_0"
        # ffn_down
        elif "ffn_down" in gguf_name:
            if target_gguf_format == "gguf:q2_k":
                new_type = "gguf:q3_k"
            elif target_gguf_format == "gguf:q2_k_s":
                if i_layer < n_layer / 8:
                    new_type = "gguf:q4_k"
            elif target_gguf_format == "gguf:q3_k_m":
                if i_layer < n_layer / 16:
                    new_type = "gguf:q5_k"
                elif gguf.MODEL_ARCH.FALCON == model_class.model_arch or _use_more_bits(i_layer, n_layer):
                    new_type = "gguf:q4_k"
                else:
                    new_type = "gguf:q3_k"
            elif target_gguf_format == "gguf:q3_k_l":
                if gguf.MODEL_ARCH.FALCON == model_class.model_arch:
                    new_type = "gguf:q4_k"
                else:
                    new_type = "gguf:q5_k"
            elif target_gguf_format == "gguf:q4_k_m":
                if gguf.MODEL_ARCH.FALCON == model_class.model_arch:
                    if i_layer < n_layer // 16:
                        new_type = "gguf:q6_k"
                    elif _use_more_bits(i_layer, n_layer):
                        new_type = "gguf:q5_k"
                    else:
                        new_type = "gguf:q4_k"
                else:
                    if _use_more_bits(i_layer, n_layer):
                        new_type = "gguf:q6_k"
            elif target_gguf_format == "gguf:q5_k_m" and _use_more_bits(i_layer, n_layer):
                new_type = "gguf:q6_k"
            elif (target_gguf_format == "gguf:q4_k_s" and
                  model_class.model_arch != gguf.MODEL_ARCH.FALCON and i_layer < n_layer / 8):
                new_type = "gguf:q5_k"
            elif (target_gguf_format == "gguf:q4_0" or target_gguf_format == "gguf:q5_0") and i_layer < n_layer / 8:
                if target_gguf_format == "gguf:q4_0":
                    new_type = "gguf:q4_1"
                else:
                    new_type = "gguf:q5_1"
            i_ffn_down += 1

        # attn_output
        elif "attn_output" in gguf_name:
            if gguf.MODEL_ARCH.FALCON != model_class.model_arch:
                if n_expert == 8:
                    if target_gguf_format in ("gguf:q2_k", "gguf:q3_k_s", "gguf:q3_k_m", "gguf:q4_k_s", "gguf:q4_k_m",
                                              "gguf:q5_k"):
                        new_type = "gguf:q5_k"
                    elif target_gguf_format == "gguf:q2_k":
                        new_type = "gguf:q3_k"
                    elif target_gguf_format == "gguf:q3_k_m":
                        new_type = "gguf:q4_k"
                    elif target_gguf_format == "gguf:q3_k_l":
                        new_type = "gguf:q5_k"
            else:
                if target_gguf_format == "gguf:q3_k_l":
                    new_type = "gguf:q4_k"
        # attn_qkv
        elif "attn_qkv" in gguf_name:
            if target_gguf_format in ("gguf:q3_k_m", "gguf:q3_k_l"):
                new_type = "gguf:q4_k"
            elif target_gguf_format == "gguf:q4_k_m":
                new_type = "gguf:q5_k"
            elif target_gguf_format == "gguf:q5_k_m":
                new_type = "gguf:q5_k"
        new_block_size = GGML_QUANT_SIZES[new_type.split(":")[-1].lower()][0]
        if input_features % new_block_size != 0:
            new_type = _gguf_type_fallback(new_type)
            new_block_size = GGML_QUANT_SIZES[new_type.split(":")[-1].lower()][0]
            if input_features % new_block_size != 0:
                new_type = "gguf:bf16"
            logger.warning(
                f"fallback {layer_name} to {new_type}, "
                f"because input_features({input_features}) % block_size({block_size}) != 0")
        # for deepseek v2
        if layer_name.endswith("kv_b_proj") and new_type.endswith("_k") \
            and 'Deepseek' in model.config.architectures[0]:
            fallback = False

            # calc if need fallback
            qk_nope_head_dim = model.config.qk_nope_head_dim
            kv_b_shape = get_module(model, layer_name).weight.shape

            if qk_nope_head_dim < QK_K or qk_nope_head_dim % QK_K != 0 \
                or kv_b_shape[-1] < QK_K or kv_b_shape[-1] % QK_K != 0:
                fallback = True
            if fallback:
                tmp_type = _gguf_type_fallback(new_type)
                logger.warning_once(
                    f"self_attn.kv_b_proj does not support the use of {new_type}, replace it with {tmp_type}")
                new_type = tmp_type

        target_config = GGUF_INNER_CONFIG[new_type]

        _set_config(layer_config[layer_name], target_config)
        _set_config(layer, target_config)
        gguf_format_config[layer_name] = new_type

    return layer_config, gguf_format_config


def get_lm_head_name(model):
    block_names = get_block_names(model, True)
    last_name = None
    for n, m in model.named_modules():
        if any(m.children()):
            continue
        last_name = n
    for l in block_names:
        if last_name in l:
            last_name = None
            break
    return last_name


def get_gguf_qtype_by_layer_config(layer_config):
    import gguf  # pylint: disable=E0401
    if layer_config['bits'] >= 16:
        return None
    bits = layer_config['bits']
    super_bits = layer_config.get("super_bits", None)
    sym = layer_config["sym"]
    group_size = layer_config.get("group_size", None)
    super_group_size = layer_config.get("super_group_size", None)
    if bits == 2 and super_bits == 4 and not sym and group_size == 16 and super_group_size == 16:
        return gguf.GGMLQuantizationType.Q2_K
    if bits == 3 and super_bits == 6 and sym and group_size == 16 and super_group_size == 16:
        return gguf.GGMLQuantizationType.Q3_K
    if bits == 4:
        if super_bits is not None and super_bits == 6 and not sym and group_size == 32 and super_group_size == 8:
            return gguf.GGMLQuantizationType.Q4_K
        if super_bits is None and sym and group_size == 32:
            return gguf.GGMLQuantizationType.Q4_0
        if super_bits is None and not sym and group_size == 32:
            return gguf.GGMLQuantizationType.Q4_1
    if bits == 5:
        if super_bits == 6 and not sym and group_size == 32 and super_group_size == 8:
            return gguf.GGMLQuantizationType.Q5_K
        if super_bits is None and sym and group_size == 32:
            return gguf.GGMLQuantizationType.Q5_0
        if super_bits is None and not sym and group_size == 32:
            return gguf.GGMLQuantizationType.Q5_1
    if bits == 6 and super_bits == 8 and group_size == 16 and super_group_size == 16:
        return gguf.GGMLQuantizationType.Q6_K
    if bits == 8 and sym and group_size == 32:
        return gguf.GGMLQuantizationType.Q8_0
    raise ValueError(f"Unknown layer config")

def flatten_list(nested_list):
    flattened = []
    for item in nested_list:
        if isinstance(item, (list, tuple)):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened

def clean_module_parameter(submodule, parameter):
    if submodule is None:
        return
    is_buffer = parameter in submodule._buffers
    with torch.no_grad():
        if is_buffer:
            submodule._buffers[parameter] = None
        else:
            submodule._parameters[parameter] = None

def get_reciprocal(tensor):
    if torch.dtype is torch.float16:
        tensor =  torch.sign(tensor) * torch.clamp(torch.abs(tensor), min=1e-5)
    else:
        tensor = torch.where(torch.abs(tensor) < 1e-30, 0, tensor)
    return  torch.where(tensor != 0, 1 / tensor, torch.zeros_like(tensor))