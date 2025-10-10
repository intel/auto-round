# Copyright (c) 2024 Intel Corporation
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

import functools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from transformers.utils.versions import require_version

import auto_round_extension.cuda.gptqmodel_marlin
from auto_round import schemes as ar_schemes
from auto_round.experimental import qmodules as ar_qmodules
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme
from auto_round.utils import get_library_version

BackendInfos = {}

import cpuinfo

if TYPE_CHECKING:
    from auto_quantizer import AutoRoundConfig


def get_cpu_manufacturer():
    cpu_info = cpuinfo.get_cpu_info()
    if "brand_raw" in cpu_info and "intel" in cpu_info["brand_raw"].lower():
        return "intel"
    else:
        return "others"


@dataclass
class BackendInfo:
    """Stores configuration details for various backend formats.

    Attributes:
        device: A list of strings representing the devices the backend supports
            (e.g., 'cuda', 'cpu').
        sym: A list of booleans indicating whether the backend supports symmetric
            quantization for weights (True if symmetric, False if not).
        packing_format: A list of strings representing the packing formats used by the backend
            (e.g., 'triton', 'qbits').
        bits: A list of integers specifying the bit-widths supported by the backend
            for weight quantization (e.g., [2, 4, 8]).
        group_size: An optional list of integers specifying the group sizes supported
            for weight quantization. Group size determines how weights are grouped
            during quantization. Defaults to None.
        compute_dtype: An optional list of strings representing the compute data types
            supported by the backend (e.g., 'float32', 'bfloat16'). Defaults to None.
        data_type: An optional list of strings representing the data types
            supported for weight quantization (e.g., 'int', 'nv_fp'). Defaults to None.
        act_bits: An optional list of integers specifying the bit-widths supported
            for activation quantization (e.g., [8, 16]). Defaults to None.
        act_group_size: An optional list of integers specifying the group sizes
            supported for activation quantization. Defaults to None.
        act_sym: An optional list of booleans indicating whether the backend supports
            symmetric quantization for activations (True if symmetric, False if not).
            Defaults to None.
        act_data_type: An optional list of strings representing the data types
            supported for activations (e.g., 'mx_fp_rceil'). Defaults to None.
        act_dynamic: An optional list of booleans indicating whether the backend
            supports dynamic quantization for activations. Defaults to None.
        priority: An integer representing the backend's priority, where higher values
            indicate higher priority. Defaults to 0.
        checkers: A list of check functions (e.g., validation methods)
            used to verify whether the backend supports certain features. Defaults to
            an empty list.
        alias: An optional list of strings representing alternative names for the
            backend. Defaults to None.
    """

    device: list[str]  # TODO change to tuple
    sym: list[bool]
    packing_format: list[str]
    bits: list[int]
    compute_dtype: list[str] = None
    data_type: Optional[list[str]] = None
    group_size: Optional[list[int]] = None
    act_bits: Optional[list[int]] = None
    act_group_size: Optional[list[int]] = None
    act_sym: Optional[list[bool]] = None
    act_data_type: Optional[list[str]] = None
    act_dynamic: Optional[list[bool]] = None
    priority: int = 0  ##higher is better
    checkers: list[Any] = field(default_factory=list)
    alias: Optional[list[str]] = None
    requirements: Optional[list[str]] = None


BACKEND_ACT_ATTRS = [
    "act_bits",
    "act_group_size",
    "act_sym",
    "act_data_type",
    "act_dynamic",
]


def feature_multiply_checker(in_feature, out_feature, config, in_feature_multiplier, out_feature_multiplier=None):
    if out_feature_multiplier is None:
        out_feature_multiplier = in_feature_multiplier
    return in_feature % in_feature_multiplier == 0 and out_feature % out_feature_multiplier == 0


def feature_multiply_checker_group_size(
    in_feature, out_feature, config, in_feature_multiplier, out_feature_multiplier=None
):
    group_size = config["group_size"]
    if out_feature_multiplier is None:
        out_feature_multiplier = in_feature_multiplier
    return (
        in_feature % in_feature_multiplier == 0
        and out_feature % out_feature_multiplier == 0
        and in_feature % group_size == 0
    )


feature_multiply_checker_32 = functools.partial(feature_multiply_checker, in_feature_multiplier=32)
feature_multiply_checker_16 = functools.partial(feature_multiply_checker, in_feature_multiplier=16)
in_output_feature_multiply_checker_32 = functools.partial(
    feature_multiply_checker, in_feature_multiplier=32, out_feature_multiplier=32
)

exllamav2_feature_checker = functools.partial(
    feature_multiply_checker_group_size, in_feature_multiplier=32, out_feature_multiplier=32
)

gptqmodel_marlin_feature_checker = functools.partial(
    feature_multiply_checker_group_size, in_feature_multiplier=1, out_feature_multiplier=64
)


def fp8_static_scheme_checker(
    in_feature: int,
    out_feature: int,
    config: QuantizationScheme,
    in_feature_multiplier: Optional[int] = None,
    out_feature_multiplier: Optional[int] = None,
):
    from auto_round.schemes import FP8_STATIC

    return config == FP8_STATIC


GPTQ_FORMAT = ["auto_round:auto_gptq"]  # zp+-1
GPTQ_FORMAT_NO_ZP = ["auto_round", "auto_round:gptqmodel"]
AWQ_FORMAT = ["auto_round:auto_awq"]
LLM_COMPRESSOR_FORMAT = ["auto_round:llm_compressor"]
WOQ_DEFAULT_ACT_BITS = [None, 16, 32]

BackendInfos["auto_gptq:exllamav2"] = BackendInfo(
    device=["cuda"],
    sym=[True, False],
    packing_format=GPTQ_FORMAT,
    bits=[4],
    priority=5,
    compute_dtype=["float16"],
    data_type=["int"],
    act_bits=WOQ_DEFAULT_ACT_BITS,
    # 16, 384,768 accuracy issue
    group_size=[-1, 32, 64, 128, 256, 512, 1024, 2048],
    checkers=[exllamav2_feature_checker],
    alias=["gptq", "auto_gptq", "exllamav2", "gptq:exllamav2", "auto_gptq:exllamav2"],
    requirements=["auto-gptq>=0.7.1"],
)

BackendInfos["auto_gptq:tritonv2"] = BackendInfo(
    device=["cuda"],
    sym=[True, False],
    packing_format=GPTQ_FORMAT,
    bits=[2, 4, 8],
    group_size=None,
    compute_dtype=["float16"],
    data_type=["int"],
    act_bits=WOQ_DEFAULT_ACT_BITS,
    priority=0,
    checkers=[exllamav2_feature_checker],
    alias=["auto_gptq:tritonv2"],
    requirements=["auto-gptq>=0.7.1", "triton>=2.0"],
)

BackendInfos["auto_gptq:cuda"] = BackendInfo(
    device=["cuda"],
    sym=[True, False],
    packing_format=GPTQ_FORMAT,
    bits=[2, 3, 4, 8],
    group_size=None,
    priority=1,
    checkers=[exllamav2_feature_checker],
    compute_dtype=["float16"],
    data_type=["int"],
    act_bits=WOQ_DEFAULT_ACT_BITS,
    alias=["auto_gptq:cuda"],
    requirements=[
        "auto-gptq>=0.7.1",
    ],
)

# FP8 static quant
# Weight: FP8, per-channel, may be extended to per-tensor in future
# Activation: FP8, per-tensor
BackendInfos["auto_round:torch_fp8_static"] = BackendInfo(
    device=["xpu", "cuda", "cpu"],
    packing_format=["auto_round:fp8_static"],
    sym=[True],
    compute_dtype=["float32", "float16", "bfloat16"],
    data_type=["fp"],
    bits=[8],
    priority=0,
    checkers=[fp8_static_scheme_checker],
    alias=["auto_round", "torch"],
    requirements=["auto-round>0.6.0"],
)

# MXFP8
BackendInfos["auto_round:torch_mxfp8"] = BackendInfo(
    device=["xpu", "cuda", "cpu"],
    packing_format=LLM_COMPRESSOR_FORMAT,
    sym=[True],
    compute_dtype=["float32", "float16", "bfloat16"],
    data_type=["mx_fp", "max_fp_rceil"],
    group_size=[32],
    bits=[8],
    act_bits=[8],
    act_group_size=[32],
    act_sym=[True],
    act_data_type=["mx_fp_rceil"],
    act_dynamic=[True],
    priority=0,
    checkers=[feature_multiply_checker_32],
    alias=["auto_round", "torch"],
    requirements=["auto-round>0.7.0"],
)

# MXFP4
BackendInfos["auto_round:torch_mxfp4"] = BackendInfo(
    device=["xpu", "cuda", "cpu"],
    packing_format=LLM_COMPRESSOR_FORMAT,
    sym=[True],
    compute_dtype=["float32", "float16", "bfloat16"],
    data_type=["mx_fp"],
    group_size=[32],
    bits=[4],
    act_bits=[4],
    act_group_size=[32],
    act_sym=[True],
    act_data_type=["mx_fp_rceil"],
    act_dynamic=[True],
    priority=0,
    checkers=[feature_multiply_checker_32],
    alias=["auto_round", "torch"],
    requirements=["auto-round>0.7.0"],
)

# NVFP4

BackendInfos["auto_round:torch_nvfp4"] = BackendInfo(
    device=["xpu", "cuda", "cpu"],
    packing_format=LLM_COMPRESSOR_FORMAT,
    sym=[True],
    compute_dtype=["float32", "float16", "bfloat16"],
    data_type=["nv_fp"],
    group_size=[16],
    bits=[4],
    act_bits=[4],
    act_group_size=[16],
    act_sym=[True],
    act_data_type=["nv_fp4_with_static_gs"],
    act_dynamic=[True],
    priority=0,
    checkers=[feature_multiply_checker_16],
    alias=["auto_round", "torch"],
    requirements=["auto-round>0.7.0"],
)

BackendInfos["auto_round:tritonv2"] = BackendInfo(
    device=["cuda", "xpu"],
    sym=[True, False],
    packing_format=GPTQ_FORMAT_NO_ZP,
    compute_dtype=["float16", "bfloat16"],
    bits=[2, 4, 8],
    priority=2,
    checkers=[feature_multiply_checker_32],
    alias=["auto_round", "tritonv2", "triton"],
    requirements=["triton>=2.0", "auto-round>=0.5.0"],
)

BackendInfos["auto_round:tritonv2_zp"] = BackendInfo(
    device=["cuda", "xpu"],
    sym=[True],
    packing_format=GPTQ_FORMAT,
    compute_dtype=["float16", "bfloat16"],
    data_type=["int"],
    act_bits=WOQ_DEFAULT_ACT_BITS,
    bits=[2, 4, 8],
    priority=2,
    checkers=[feature_multiply_checker_32],
    alias=["tritonv2", "tritonv2_zp", "triton"],
    requirements=["triton>=2.0", "auto-round>=0.5.0"],
)

BackendInfos["auto_round:torch"] = BackendInfo(
    device=["cuda", "xpu", "cpu"],
    sym=[True, False],
    packing_format=GPTQ_FORMAT_NO_ZP,
    compute_dtype=["float16", "bfloat16"],
    data_type=["int"],
    act_bits=WOQ_DEFAULT_ACT_BITS,
    bits=[2, 3, 4, 8],
    priority=0,
    checkers=[exllamav2_feature_checker],
    alias=["auto_round", "torch"],
    requirements=["auto-round>=0.5.1"],
)


BackendInfos["auto_round:torch_zp"] = BackendInfo(
    device=["cuda", "xpu", "cpu"],
    sym=[True],
    packing_format=GPTQ_FORMAT,
    compute_dtype=["float16", "bfloat16"],
    data_type=["int"],
    act_bits=WOQ_DEFAULT_ACT_BITS,
    bits=[2, 3, 4, 8],
    priority=0,
    checkers=[exllamav2_feature_checker],
    alias=["torch", "torch_zp"],
    requirements=["auto-round>=0.5.1"],
)

BackendInfos["gptqmodel:marlin"] = BackendInfo(
    device=["cuda"],
    sym=[True],
    packing_format=GPTQ_FORMAT_NO_ZP,
    bits=[4, 8],
    group_size=[-1, 32, 64, 128],
    compute_dtype=["float16", "bfloat16"],
    data_type=["int"],
    act_bits=WOQ_DEFAULT_ACT_BITS,
    priority=6,
    checkers=[gptqmodel_marlin_feature_checker],
    alias=["marlin", "gptqmodel"],
    requirements=["gptqmodel>=2.0"],
)

BackendInfos["gptqmodel:marlin_zp"] = BackendInfo(
    device=["cuda"],
    sym=[True],
    packing_format=GPTQ_FORMAT,
    bits=[4, 8],
    group_size=[-1, 32, 64, 128],
    compute_dtype=["float16", "bfloat16"],
    data_type=["int"],
    act_bits=WOQ_DEFAULT_ACT_BITS,
    priority=6,
    checkers=[gptqmodel_marlin_feature_checker],
    alias=["marlin", "gptqmodel"],
    requirements=["gptqmodel>=2.0"],
)

BackendInfos["gptqmodel:exllamav2"] = BackendInfo(
    device=["cuda"],
    sym=[True, False],
    packing_format=GPTQ_FORMAT_NO_ZP,
    bits=[4],
    group_size=[-1, 32, 64, 128],  ##16 seems has accuracy issue
    compute_dtype=["float16", "bfloat16"],
    data_type=["int"],
    act_bits=WOQ_DEFAULT_ACT_BITS,
    priority=5,
    checkers=[exllamav2_feature_checker],
    alias=["exllamav2"],
    requirements=["gptqmodel>=2.0"],
)

BackendInfos["auto_awq:gemm"] = BackendInfo(
    device=["cuda"],
    sym=[True, False],  # Actually it is GEMM
    packing_format=AWQ_FORMAT,
    bits=[4],
    group_size=None,
    priority=5,
    compute_dtype=["float16"],
    data_type=["int"],
    act_bits=WOQ_DEFAULT_ACT_BITS,
    alias=["auto_awq:gemm", "awq", "awq:gemm", "auto_awq"],
    requirements=["autoawq", "transformers<4.57.0"],
)

BackendInfos["qbits"] = BackendInfo(
    device=["cpu"],
    sym=[True, False],
    packing_format=GPTQ_FORMAT_NO_ZP,
    bits=[2, 4, 8],
    group_size=None,
    priority=1,
    checkers=[],
    alias=["itrex", "qbits"],
    compute_dtype=["float16", "bfloat16"],
    data_type=["int"],
    act_bits=WOQ_DEFAULT_ACT_BITS,
    requirements=["torch<2.7.0", "intel-extension-for-transformers"],
)

BackendInfos["qbits_zp"] = BackendInfo(
    device=["cpu"],
    sym=[True, False],
    packing_format=GPTQ_FORMAT,
    bits=[2, 4, 8],
    group_size=None,
    compute_dtype=["float16", "bfloat16"],
    data_type=["int"],
    act_bits=WOQ_DEFAULT_ACT_BITS,
    priority=1,
    checkers=[],
    alias=["itrex", "qbits"],
    requirements=["torch<2.7.0", "intel-extension-for-transformers"],
)


BackendInfos["qbits_awq"] = BackendInfo(
    device=["cpu"],
    sym=[True, False],
    packing_format=AWQ_FORMAT,
    bits=[2, 4, 8],
    group_size=None,
    compute_dtype=["float16", "bfloat16"],
    data_type=["int"],
    act_bits=WOQ_DEFAULT_ACT_BITS,
    priority=1,
    checkers=[],
    alias=["itrex", "qbits"],
    requirements=["torch<2.7.0", "intel-extension-for-transformers"],
)
BackendInfos["ipex_gptq"] = BackendInfo(
    device=["cpu", "xpu"],
    sym=[True, False],
    packing_format=GPTQ_FORMAT,
    bits=[4],
    group_size=None,
    priority=5,
    checkers=[],
    compute_dtype=["float16", "bfloat16"],
    data_type=["int"],
    act_bits=WOQ_DEFAULT_ACT_BITS,
    alias=["ipex"],
    requirements=["intel-extension-for-pytorch>=2.5"],
)

BackendInfos["ipex_awq"] = BackendInfo(
    device=["cpu", "xpu"],
    sym=[True, False],
    packing_format=AWQ_FORMAT,
    bits=[4],
    group_size=None,
    priority=5,
    checkers=[],
    compute_dtype=["float16", "bfloat16"],
    data_type=["int"],
    act_bits=WOQ_DEFAULT_ACT_BITS,
    alias=["ipex"],
    requirements=["intel-extension-for-pytorch>=2.5"],
)
BackendInfos["hpu"] = BackendInfo(
    device=["hpu"],
    sym=[True, False],
    packing_format=GPTQ_FORMAT_NO_ZP,
    bits=[4],
    compute_dtype=["bfloat16"],
    data_type=["int"],
    act_bits=WOQ_DEFAULT_ACT_BITS,
    alias=["hpu"],
    priority=0,
)

BackendInfos["hpu_zp"] = BackendInfo(
    device=["hpu"],
    sym=[True, False],
    packing_format=GPTQ_FORMAT,
    bits=[4],
    compute_dtype=["bfloat16"],
    data_type=["int"],
    act_bits=WOQ_DEFAULT_ACT_BITS,
    alias=["hpu"],
    priority=0,
)


def check_compatible(
    backend_name: str,
    device: str,
    config: dict,
    packing_format: str,
    in_features: int,
    out_features: int,
    check_requirements=True,
):
    """Checks if the given configuration is compatible with the specified backend.

    Args:
        backend_name (str): The name of the backend to check compatibility for.
        device (str): The device on which the backend operates (e.g., 'cuda', 'cpu').
        config(dict): scheme
        packing_format (str): The packing format used by the backend (e.g., 'triton').
        in_features (int): The number of input features for the model layer.
        out_features (int): The number of output features for the model layer.
        check_requirements (bool): Whether check the requirement

    Returns:
        bool: True if the configuration is compatible with the backend, False otherwise.

    Raises:
        KeyError: If the backend_name is not found in BackendInfos.

    Compatibility checks:
    - Device must match one of the backend's supported devices.
    - Bit-width must be supported by the backend.
    - If group_size is required by the backend, it must match.
    - Symmetric or asymmetric quantization must be supported.
    - If the packing format matches exactly, all feature checks must pass.
    - If the packing format does not match, it must be convertible.
    """
    backend = BackendInfos[backend_name]
    # Check if the format is convertible when packing formats differ
    if packing_format in backend.packing_format:
        pass
    else:
        return False
    # Check scheme
    for key, value in config.items():
        backend_value = getattr(backend, key, None)
        if backend_value is not None and value not in backend_value:
            return False

    # Check if device is supported by the backend
    if device not in backend.device:
        return False

    for check in backend.checkers:
        if not check(in_features, out_features, config):
            return False

    if check_requirements and backend.requirements is not None:
        for requirement in backend.requirements:
            if isinstance(requirement, str):
                try:
                    require_version(requirement)
                except ImportError:
                    return False
            else:
                res, _ = requirement()
                return res

    return True


def dynamic_import_inference_linear(backend, config):
    """Dynamically imports and returns the appropriate QuantLinear class based on the given backend.

    This function dynamically loads the correct `QuantLinear` class based on the backend and quantization
    configuration (e.g., qbits, marlin, hpu, gptq, awq, auto_round). It imports specific modules or raises
    errors if the required packages are not installed or the environment is not set up.

    Args:
        backend (str):
            The backend to be used for quantization (e.g., 'qbits', 'marlin', 'hpu', 'gptq', 'awq', 'auto_round').
        config (QuantizationScheme):
            The quantization configuration containing parameters like bits, group_size, and sym.

    Returns:
        class:
            The dynamically imported QuantLinear class that corresponds to the given backend configuration.

    Raises:
        ImportError:
            If required modules are missing for a backend (e.g., Intel Extension, GPTQ, auto_awq).
    """
    bits, group_size, sym = config["bits"], config["group_size"], config["sym"]

    if "torch_fp8_static" in backend:
        return ar_qmodules.WeightFP8ActFP8StaticQuantLinear
    if "torch_mxfp8" in backend:
        return ar_qmodules.MXFP8QuantLinear
    if "torch_mxfp4" in backend:
        return ar_qmodules.MXFP4QuantLinear
    if "torch_nvfp4" in backend:
        return ar_qmodules.NVFP4QuantLinear

    if "qbits" in backend:
        try:
            from intel_extension_for_transformers import qbits  # pylint: disable=E0401
        except Exception as e:
            raise ImportError(
                "Please install Intel Extension for Transformers via 'pip install "
                "intel-extension-for-transformers' to inference on X86 CPU"
            )
        if "zp" in backend:
            import auto_round_extension.qbits.qlinear_qbits_gptq as qlinear_qbits_gptq

            return qlinear_qbits_gptq.QuantLinear
        elif "awq" in backend:
            import auto_round_extension.qbits.qbits_awq as qlinear_qbits_awq

            return qlinear_qbits_awq.QuantLinear
        else:  # auto_round must be at the end
            import auto_round_extension.qbits.qlinear_qbits as qlinear_qbits_autoround

            return qlinear_qbits_autoround.QuantLinear
    if "ipex_gptq" in backend:
        from auto_round_extension.ipex.qlinear_ipex_gptq import QuantLinear

        return QuantLinear

    if "ipex_awq" in backend:
        from auto_round_extension.ipex.qlinear_ipex_awq import QuantLinear

        return QuantLinear

    if "hpu" in backend:
        try:
            import habana_frameworks.torch.hpu  # pylint: disable=E0401
        except ImportError:
            raise ImportError("Please setup hpu environment before using hpu backend")

        if "zp" in backend:
            from auto_round_extension.hpu.qlinear_hpu_gptq import QuantLinear as QuantLinear_gptq

            return QuantLinear_gptq
        else:  # auto_round must be at the end
            from auto_round_extension.hpu.qlinear_hpu import QuantLinear

            return QuantLinear
    if "gptqmodel" in backend:
        return get_gptqmodel_infer_linear(backend, bits, group_size, sym)

    if "gptq" in backend and "gptqmodel" not in backend:
        return get_autogptq_infer_linear(backend, bits, group_size, sym)

    if "awq" in backend:
        try:
            from awq.modules.linear import WQLinear_GEMM  # pylint: disable=E0401
        except ImportError:
            raise ImportError(
                "autoawq is required. Please install it by 'pip install autoawq' to support auto_awq format."
            )
        return WQLinear_GEMM

    if backend == "auto_round:tritonv2":
        from auto_round_extension.triton.qlinear_tritonv2 import QuantLinear

        return QuantLinear

    if backend == "auto_round:tritonv2_zp":
        from auto_round_extension.triton.qlinear_tritonv2_zp import QuantLinear

        return QuantLinear

    if backend == "auto_round:torch":
        from auto_round_extension.torch.qlinear_torch import QuantLinear

        return QuantLinear

    if backend == "auto_round:torch_zp":
        from auto_round_extension.torch.qlinear_torch_zp import QuantLinear

        return QuantLinear

    raise ValueError(f"unsupported backend {backend}, please set it to `auto` and retry")


def get_gptqmodel_infer_linear(backend, bits=4, group_size=128, sym=False):
    import torch

    dtype = torch.get_default_dtype()
    if dtype != torch.float32:
        torch.set_default_dtype(torch.float32)
    import gptqmodel  # pylint: disable=E0401

    torch.set_default_dtype(dtype)

    if "marlin" in backend:
        return auto_round_extension.cuda.gptqmodel_marlin.get_marlin_layer()
        # return gptqmodel.nn_modules.qlinear.marlin.MarlinQuantLinear
    elif "exllamav2" in backend:
        return gptqmodel.nn_modules.qlinear.exllamav2.ExllamaV2QuantLinear
    elif "tritonv2" in backend:
        return gptqmodel.nn_modules.qlinear.tritonv2.TritonV2QuantLinear
    elif "torch" in backend:
        return gptqmodel.nn_modules.qlinear.torch.TorchQuantLinear
    else:
        raise ValueError(f"Unsupported {backend}")


def get_autogptq_infer_linear(backend, bits=4, group_size=128, sym=False):
    """Returns the appropriate QuantLinear class based on backend configuration.

    This function selects and dynamically imports the `QuantLinear` class according to the specified backend
    and its features, such as using Triton, ExLlama, Marlin, or Qigen for quantization.

    Args:
        backend (str):
            The backend to be used for quantization (e.g., 'triton', 'qigen', 'marlin', 'exllamav2').
        bits (int, optional):
            The number of bits used for quantization. Default is 4.
        group_size (int, optional):
            The group size for quantization. Default is 128.
        sym (bool, optional):
            Whether symmetric quantization is enabled. Default is False.

    Returns:
        class:
            The dynamically imported QuantLinear class for the given configuration.

    Raises:
        ImportError:
            If required packages or backends are not installed.
    """
    use_triton = False
    disable_exllamav2 = False
    disable_exllamav1 = False
    disable_marlin = True
    use_qigen = False
    use_tritonv2 = False

    # Determine backend configurations based on input string
    if "qigen" in backend:
        use_qigen = True
    elif "triton" in backend:
        use_triton = True
    elif "tritonv2" in backend:
        use_triton = False
        use_tritonv2 = True
    elif "marlin" in backend:
        use_triton = False
        disable_marlin = False
    elif "exllamav2" in backend:
        use_triton = False
        disable_exllamav2 = False
        disable_marlin = True
    elif "exllamav1" in backend:
        use_triton = False
        disable_marlin = True
    elif "cuda" in backend:
        use_triton = False
        disable_marlin = True
        disable_exllamav2 = True
        disable_exllamav1 = True

    from auto_gptq.utils.import_utils import dynamically_import_QuantLinear  # pylint: disable=E0401

    version = get_library_version("auto_gptq")
    from packaging.version import Version

    # Import the appropriate QuantLinear based on the version of auto_gptq
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
            use_tritonv2=use_tritonv2,
        )

    return QuantLinear


def find_backend(backend: str, orig_backend: str = None):
    """
    Finds the matching backend key based on the target backend name or its aliases.

    Args:
        backend (str): Name or alias of the target backend.
        orig_backend (str, optional): Original backend name to check compatibility. Defaults to None.

    Returns:
        str or None: Matching backend key if found and compatible; otherwise, None.
    """
    logger.trace(f"Finding backend for target: {backend}, original: {orig_backend}")

    matched_keys = [
        key for key, info in BackendInfos.items() if key == backend or (info.alias and backend in info.alias)
    ]

    if not matched_keys:
        return None

    if orig_backend is None:
        return matched_keys[0] if len(matched_keys) >= 1 else None

    orig_info = BackendInfos[orig_backend]

    for key in matched_keys:
        target_info = BackendInfos[key]
        if (
            target_info.packing_format == orig_info.packing_format
            or orig_info.packing_format in target_info.convertible_format
        ):
            return key

    raise ValueError(f"{backend} is not compatible with {orig_backend}. " f"Please set `backend` to `auto` and retry.")


def get_all_compatible_backend(
    device: str, packing_format: str, config: dict, in_features: int, out_features: int
) -> list[str]:
    # Find compatible backends
    compatible_backends = [
        key
        for key in BackendInfos.keys()
        if check_compatible(key, device, config, packing_format, in_features, out_features, check_requirements=False)
    ]

    # Return the first compatible backend or an empty list if none found
    return compatible_backends


def get_layer_backend(
    device: str, backend: str, packing_format: str, config: dict, in_features: int, out_features: int
) -> str:
    """Selects the most suitable backend for the layer based on compatibility and priority.

    This function first checks if the specified backend supports the layer with the provided configuration.
    If not, it iterates through other available backends,
    checking compatibility and returning the one with the highest priority.

    Args:
        device (str):
            The device on which the layer will run, e.g., 'cpu', 'cuda'.
        backend (str):
            The target backend to be used for this layer."auto","triton","gptqmodel", etc,
        packing_format (str):
            The original backend from which packing format information is retrieved.
        config (dict): Layer config.
        in_features (int):
            The number of input features for the layer.
        out_features (int):
            The number of output features for the layer.

    Returns:
        str:
            The selected backend that is compatible with the layer configuration.

    Raises:
        ValueError:
            If the specified backend is not supported.
            If no compatible backend is found for the given layer configuration.
    """

    backends = []
    if backend == "auto":
        backends = BackendInfos.keys()
    else:
        for key in BackendInfos.keys():
            if backend == key or (BackendInfos[key].alias and backend in BackendInfos[key].alias):
                backends.append(key)

    # Find and store other compatible backends
    supported_backends = []
    for key in backends:
        if check_compatible(key, device, config, packing_format, in_features, out_features):
            logger.trace(f"Backend {key} is compatible")
            supported_backends.append(key)

    # Raise an error if no compatible backends are found
    if len(supported_backends) == 0:
        supported_backends_need_package = get_all_compatible_backend(
            device, packing_format, config, in_features, out_features
        )

        if len(supported_backends_need_package) > 0:
            supported_backends_need_package = sorted(
                supported_backends_need_package,
                key=lambda support_backend: BackendInfos[support_backend].priority,
                reverse=True,
            )
            backend_info = BackendInfos[supported_backends_need_package[0]]
            process_requirement(backend_info.requirements, target_device=device)

        return ""

    # Sort the compatible backends by priority and return the one with the highest priority
    supported_backends = sorted(
        supported_backends, key=lambda support_backend: BackendInfos[support_backend].priority, reverse=True
    )

    return supported_backends[0]


def get_highest_priority_backend(
    quantization_config: "AutoRoundConfig", device: str, packing_format: str
) -> str | None:
    supported_backends = []
    for key in BackendInfos.keys():
        backend = BackendInfos[key]
        # Check if device is supported by the backend
        if device not in backend.device:
            continue

        # Check if bit-width is supported
        if quantization_config.bits not in backend.bits:
            continue

        # Check if group_size is valid (if required by backend)
        if backend.group_size is not None and quantization_config.group_size not in backend.group_size:
            continue

        # Check if symmetric/asymmetric quantization is supported
        if quantization_config.sym not in backend.sym:
            continue

        # Check if the format is convertible when packing formats differ
        if packing_format in backend.packing_format:
            pass
        else:
            continue

        def _is_act_field_supported(backend, quantization, field_name):
            q_val = getattr(quantization, field_name, None)
            b_val = getattr(backend, field_name, None)
            # Case 1. quantization field is None, assume it is not used, so supported
            # Case 2. backend field is not None and contains the quantization field value
            return (q_val is None) or (b_val is not None and q_val in b_val)

        if not all(_is_act_field_supported(backend, quantization_config, field) for field in BACKEND_ACT_ATTRS):
            continue

        supported_backends.append(key)

    if len(supported_backends) > 0:

        supported_backends = sorted(
            supported_backends, key=lambda support_backend: BackendInfos[support_backend].priority, reverse=True
        )
        return supported_backends[0]
    else:
        return None


def process_requirement(requirements: list, target_device="cuda", logger_level="error"):
    def log(message):
        (logger.warning if logger_level != "error" else logger.error)(message)

    def build_pip_commands(gptq_req, other_reqs):
        commands = []

        if gptq_req:
            commands.append(f"pip install -v {gptq_req} --no-build-isolation")
            try:
                require_version("numpy<2.0")
            except:
                commands.append("pip install 'numpy<2.0'")

        if other_reqs:
            other_str = " ".join(other_reqs)
            commands.append(f"pip install {other_str}")

        return commands

    # Filter requirements
    missing_requirements = []
    for req in requirements:
        try:
            require_version(req)
        except:
            missing_requirements.append(req)

    gptq_req = next((f'"{req}"' for req in missing_requirements if "gptqmodel" in req), None)
    other_reqs = [f'"{req}"' for req in missing_requirements if "gptqmodel" not in req]

    pip_cmds = build_pip_commands(gptq_req, other_reqs)
    if not pip_cmds:
        return

    # Instructional messages
    install_instructions = []

    for cmd in pip_cmds:
        if "intel-extension-for-pytorch" in cmd and target_device == "xpu":
            install_instructions.append(
                "Please refer to https://pytorch-extension.intel.com/installation?platform=gpu "
                "to install intel-extension-for-pytorch. Ensure that the version matches your installed PyTorch."
            )

    prefix_msg = (
        "Better backend is found, please install all the following requirements to enable it."
        if logger_level != "error"
        else "Inference requires the following libraries. Please install all of them."
    )
    log(prefix_msg)

    for msg in install_instructions:
        log(msg)
        if logger_level == "error" and len(pip_cmds) == 0:
            exit(-1)

    joined_cmds = " and ".join(f"`{cmd}`" for cmd in pip_cmds)
    if joined_cmds:
        log(joined_cmds)
        if logger_level == "error":
            exit(-1)

