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
import importlib
import logging
import subprocess
from collections import UserDict

# for cpu usage
import cpuinfo
import psutil
import torch
from torch.amp import autocast

SHARE_ATTENTION_MASK_LIST = ["Baichuan2-13B-Base", "Baichuan2-13B-Chat"]

logger = logging.getLogger("autoround")
logger.setLevel(logging.INFO)
fh = logging.StreamHandler()
fh_formatter = logging.Formatter("%(asctime)s %(levelname)s %(filename)s L%(lineno)d: %(message)s", "%Y-%m-%d %H:%M:%S")
fh.setFormatter(fh_formatter)
logger.addHandler(fh)


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


def is_optimum_habana_available():

    from transformers.utils.import_utils import is_optimum_available

    return is_optimum_available() and importlib.util.find_spec("optimum.habana") is not None


htcore = LazyImport("habana_frameworks.torch.core")


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
    maxq = torch.tensor(2**num_bits - 1)
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
    maxq = torch.tensor(2**num_bits - 1)
    zeros = torch.zeros(weight.shape[0], device=weight.device, dtype=scale_dtype)
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
    wmax_new = torch.max(wmin.abs(), wmax)
    tmp = wmin < 0
    wmin_new = wmin.clone()  ##must clone, otherwise inplace backward will occur
    if torch.any(tmp):
        wmin_new[tmp] = -wmax_new[tmp]

    tmp = (wmin_new == 0) & (wmax_new == 0)
    wmin_new[tmp] = -1
    wmax_new[tmp] = +1
    scale = ((wmax_new - wmin_new) / maxq).to(scale_dtype)

    scale = scale.unsqueeze(dim=-1)
    zp = torch.full_like(scale, (maxq + 1) / 2)

    int_w = round_ste(weight / scale + v)
    q = torch.clamp(int_w + zp, 0, maxq)
    return scale * (q - zp), scale, zp


def quant_weight_actor(weight, num_bits, sym, v, min_scale, max_scale, scale_dtype=torch.float16):
    """Quantizes and dequantizes weight symmetrically or asymmetrically .

    Args:
        weight: Tensor containing the weight to be quantized
        num_bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        sym: Sym or asym
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for weight
        max_scale: Maximum scale coefficient for weight

    Returns:
        Quantized and dequantized weight, scale, zero-point
    """
    assert num_bits > 0, "num_bits should be larger than 0"
    if sym:
        return quant_weight_sym(weight, num_bits, v, min_scale, max_scale, scale_dtype)
    else:
        return quant_weight_asym(weight, num_bits, v, min_scale, max_scale, scale_dtype)


def quant_weight(
    weight, num_bits=4, group_size=-1, sym=False, v=0, min_scale=0, max_scale=0, scale_dtype=torch.float16
):
    """Quantizes and dequantizes weight, handing the group size issue .

    Args:
        weight: Tensor containing the weight to be quantized
        num_bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: The number of elements shares scale and zero point
        sym: Sym or asym
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for weight
        max_scale: Maximum scale coefficient for weight

    Returns:
        Quantized and dequantized weight, scale, zero-point
    """
    if group_size == -1 or weight.shape[1] < group_size:
        return quant_weight_actor(
            weight, num_bits, sym=sym, v=v, min_scale=min_scale, max_scale=max_scale, scale_dtype=scale_dtype
        )
    orig_shape = weight.shape
    if weight.shape[1] % group_size == 0:
        weight = weight.reshape(-1, group_size)
        if isinstance(v, torch.Tensor):
            v = v.reshape(-1, group_size)

        weight, scale, zp = quant_weight_actor(
            weight, num_bits, sym=sym, v=v, min_scale=min_scale, max_scale=max_scale, scale_dtype=scale_dtype
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
            weight_new, num_bits, sym=sym, v=v, min_scale=min_scale, max_scale=max_scale, scale_dtype=scale_dtype
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


def get_module(module, key):
    """Get module from model by key name.

    Args:
        module (torch.nn.Module): original model
        key (str): module name to be replaced
    """
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


def sampling_inputs(input_ids, input_others, indices, seqlen, share_attention_mask_flag=False):
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
        if not share_attention_mask_flag and "attention_mask" in key or "alibi" in key:
            current_input_others[key] = None
            if input_others[key] is not None:
                current_input_others[key] = input_others[key][indices, ...]
        else:
            current_input_others[key] = input_others[key]

    return current_input_ids, current_input_others


def block_forward(
    block,
    input_ids,
    input_others,
    amp=False,
    amp_dtype=torch.bfloat16,
    amp_device_type="hpu",
    device=torch.device("cpu"),
):
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
            with autocast(device_type=amp_device_type, dtype=amp_dtype):  # pragma: no cover
                output = block(
                    input_ids, attention_mask=attention_mask, alibi=alibi
                )  ##TODO is this correct for all models with alibi?
        elif amp and check_is_cpu(device):
            with torch.autocast(device_type=amp_device_type, dtype=torch.bfloat16):
                output = block(input_ids, attention_mask=attention_mask, alibi=alibi)
        else:
            output = block(input_ids, attention_mask=attention_mask, alibi=alibi)
    else:
        input_tuple = input_others.pop("positional_inputs", None)
        if amp and not check_is_cpu(device):
            with autocast(device_type=amp_device_type, dtype=amp_dtype):  # pragma: no cover
                output = block.forward(input_ids, *input_tuple, **input_others)
        elif amp and check_is_cpu(device):
            with torch.autocast(device_type=amp_device_type, dtype=torch.bfloat16):
                output = block.forward(input_ids, *input_tuple, **input_others)
        else:
            output = block.forward(input_ids, *input_tuple, **input_others)
    if isinstance(output, list) or isinstance(output, tuple):
        output = output[0]
    return output


def check_to_quantized(config):
    if isinstance(config, dict):
        if config["bits"] > 8 or "fp" in config["data_type"] or "float" in config["data_type"]:
            return False
        return True
    else:
        if config.bits > 8 or "fp" in config.data_type or "float" in config.data_type:
            return False
        return True


def is_share_attention_mask_model(model):
    model_name = None
    if not hasattr(model, "config") or not hasattr(model.config, "_name_or_path"):
        logger.warn("Unable to get model name via config, assumed to be a normal model.")
        return True
    model_name = model.config._name_or_path
    for key in SHARE_ATTENTION_MASK_LIST:
        if key in model_name:
            return True
    return False


def detect_device(device=None):
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
        elif is_optimum_habana_available():
            device = torch.device("hpu")
            logger.info("Using HPU device")
        # Use CPU as a fallback
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        if dev_idx is not None:
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
