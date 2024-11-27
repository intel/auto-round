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
import logging
from dataclasses import dataclass, field
from typing import List, Any, Optional

from transformers.utils.versions import require_version

from auto_round.utils import get_library_version

BackendInfos = {}

import cpuinfo

def get_cpu_manufacturer():
    cpu_info = cpuinfo.get_cpu_info()
    if "brand_raw" in cpu_info and "intel"  in cpu_info["brand_raw"].lower():
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
            quantization (True if symmetric, False if not).
        packing_format: A string representing the packing format used by the backend
            (e.g., 'triton', 'qbits').
        bits: A list of integers specifying the bit-widths supported by the backend
            (e.g., [2, 4, 8]).
        group_size: An optional list of integers specifying the group size for
            quantization. Defaults to None.
        priority: An integer representing the backend's priority, where higher values
            indicate higher priority. Defaults to 0.
        convertable_format: A list of strings specifying the formats that the backend
            can convert from. Defaults to an empty list.
        feature_checks: A list of feature check functions (e.g., validation methods)
            used to verify whether the backend supports certain features. Defaults to
            an empty list.
        alias: An optional list of strings representing alternative names for the
            backend. Defaults to None.
    """
    device: List[str]
    sym: List[bool]
    packing_format: str
    bits: List[int]
    group_size: Optional[List[int]] = None
    priority: int = 0  ##higher is better
    convertable_format: List[str] = field(default_factory=list)
    feature_checks: List[Any] = field(default_factory=list)
    alias: Optional[List[str]] = None
    requirements: Optional[List[str]] = None


def feature_multiply_checker(in_feature, out_feature, in_feature_multiplier, out_feature_multiplier=None):
    if out_feature_multiplier is None:
        out_feature_multiplier = in_feature_multiplier
    return in_feature % in_feature_multiplier == 0 and out_feature % out_feature_multiplier == 0


def feature_num_greater_checker(in_feature, out_feature, num):
    return in_feature * out_feature > num


feature_multiply_checker_32 = functools.partial(feature_multiply_checker, in_feature_multiplier=32)

feature_multiply_checker_marlin = functools.partial(feature_multiply_checker, in_feature_multiplier=128,
                                                    out_feature_multiplier=256)

feature_num_greater_checker_1024 = functools.partial(feature_num_greater_checker, num=1024)

BackendInfos['auto_round:exllamav2'] = BackendInfo(device=["cuda"], sym=[True, False],
                                                   packing_format="triton",
                                                   bits=[4], group_size=None,
                                                   priority=5,
                                                   feature_checks=[feature_multiply_checker_32],
                                                   alias=["auto_round"]
                                                   )

BackendInfos['auto_round:tritonv2'] = BackendInfo(device=["cuda"], sym=[True, False],
                                                  packing_format="triton",
                                                  bits=[2, 4, 8], group_size=None,
                                                  priority=0, feature_checks=[feature_multiply_checker_32],
                                                  requirements=["triton<3.0,>=2.0"]
                                                  )

BackendInfos['gptq:exllamav2'] = BackendInfo(device=["cuda"], sym=[True, False],
                                             packing_format="triton_zp+-1",
                                             bits=[4], group_size=None,
                                             priority=5,
                                             feature_checks=[feature_multiply_checker_32],
                                             alias=["auto_round:gptq:exllamav2", "auto_round:auto_gptq:exllamav2",
                                                    'gptq', 'auto_gptq', "auto_round:gptq", "auto_round:auto_gptq"],
                                             requirements=["auto-gptq>=0.7.1"]
                                             )

BackendInfos['gptq:tritonv2'] = BackendInfo(device=["cuda"], sym=[True, False],
                                            packing_format="triton_zp+-1",
                                            bits=[2, 4, 8], group_size=None,
                                            priority=0, feature_checks=[feature_multiply_checker_32],
                                            alias=["auto_round:gptq:tritonv2", "auto_round:auto_gptq:tritonv2",
                                                   "auto_gptq:tritonv2"],
                                            requirements=["auto-gptq>=0.7.1","triton<3.0,>=2.0"]
                                            )

BackendInfos['gptq:cuda'] = BackendInfo(device=["cuda"], sym=[True, False],
                                            packing_format="triton_zp+-1",
                                            bits=[2, 3, 4, 8], group_size=None,
                                            priority=1, feature_checks=[feature_multiply_checker_32],
                                            alias=["auto_round:auto_gptq:cuda,auto_gptq:cuda, auto_round:gptq:cuda"],
                                            convertable_format=["triton_zp+-1"],
                                            requirements=["auto-gptq>=0.7.1"]
                                            )

BackendInfos['awq:gemm'] = BackendInfo(device=["cuda"], sym=[True, False],  ##actrally is gemm
                                       packing_format="awq",
                                       bits=[4], group_size=None,
                                       priority=4, feature_checks=[feature_num_greater_checker_1024],
                                       alias=["auto_awq:gemm", "auto_round:awq:gemm", "auto_round:auto_awq:gemm", "awq",
                                              "auto_awq", "auto_round:awq", "aut_round:auto_awq"],
                                       requirements=["autoawq"]
                                       )

BackendInfos['auto_round:qbits'] = BackendInfo(device=["cpu"], sym=[True, False],
                                               packing_format="qbits",
                                               bits=[2, 4, 8], group_size=None,
                                               priority=0 if "intel" in get_cpu_manufacturer() else 5,
                                               feature_checks=[],
                                               convertable_format=["triton"],
                                               requirements=["intel-extension-for-transformers"])

BackendInfos['auto_round:qbits_zp'] = BackendInfo(device=["cpu"], sym=[True, False],
                                                  packing_format="qbits_zp+-1",
                                                  bits=[2, 4, 8], group_size=None,
                                                  priority=0 if "intel" in get_cpu_manufacturer() else 5,
                                                  feature_checks=[],
                                                  convertable_format=["triton_zp+-1"],
                                                  requirements=["intel-extension-for-transformers"]
                                                  )

BackendInfos['auto_round:ipex_gptq'] = BackendInfo(device=["cpu"], sym=[True, False],
                                              packing_format="ipex_gptq",
                                              bits=[4], group_size=None,
                                              priority=5 if "intel" in get_cpu_manufacturer() else 5,
                                              feature_checks=[],
                                              convertable_format=["triton_zp+-1"],
                                              requirements=["intel-extension-for-pytorch>=2.4"]
                                              )

BackendInfos['auto_round:ipex_awq'] = BackendInfo(device=["cpu"], sym=[True, False],
                                              packing_format="ipex_awq",
                                              bits=[4], group_size=None,
                                              priority=5 if "intel" in get_cpu_manufacturer() else 5,
                                              feature_checks=[],
                                              ##convertable_format=["triton_zp+-1", "awq"],
                                              convertable_format=["awq"],
                                              requirements=["intel-extension-for-pytorch>=2.4"]
                                              )

# BackendInfos['auto_round:marlin'] = BackendInfo(device=["gpu"], sym=[True],
#                                                 packing_format="marlin",
#                                                 bits=[4], group_size=[-1, 128],
#                                                 priority=6,
#                                                 feature_checks=[feature_multiply_checker_marlin],
#                                                 alias=["marlin", "auto_gptq:marlin", "auto_round:gptq:marlin",
#                                                        "auto_round:auto_gptq:marlin"])

BackendInfos['auto_round:hpu'] = BackendInfo(device=["hpu"], sym=[True, False],
                                             packing_format="hpu",
                                             bits=[4],
                                             priority=0,
                                             convertable_format=["triton"]
                                             )

BackendInfos['auto_round:hpu_zp'] = BackendInfo(device=["hpu"], sym=[True, False],
                                                packing_format="hpu_zp+-1",
                                                bits=[4],
                                                priority=0,
                                                convertable_format=["triton_zp+-1"])


def check_compatible(backend_name, device, bits, group_size, sym, packing_format, in_features, out_features,
                     check_requirements=True):
    """Checks if the given configuration is compatible with the specified backend.

    Args:
        backend_name (str): The name of the backend to check compatibility for.
        device (str): The device on which the backend operates (e.g., 'cuda', 'cpu').
        bits (int): The bit-width of the quantization (e.g., 2, 4, 8).
        group_size (Optional[int]): The size of the quantization group. Can be None if
            not required by the backend.
        sym (bool): Whether symmetric quantization is required (True for symmetric).
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

    # Check if device is supported by the backend
    if not device in backend.device:
        return False

    # Check if bit-width is supported
    if bits not in backend.bits:
        return False

    # Check if group_size is valid (if required by backend)
    if backend.group_size is not None and group_size not in backend.group_size:
        return False

    # Check if symmetric/asymmetric quantization is supported
    if sym not in backend.sym:
        return False

    # Check packing format and apply feature checks
    if packing_format == backend.packing_format:
        for check in backend.feature_checks:
            if not check(in_features, out_features):
                return False

    # Check if the format is convertible when packing formats differ
    if packing_format != backend.packing_format and packing_format not in backend.convertable_format:
        return False

    if check_requirements and backend.requirements is not None:
        for requirement in backend.requirements:
            try:
                require_version(requirement)
            except ImportError:
                return False

    return True


def dynamic_import_inference_linear(backend, bits, group_size, sym):
    """Dynamically imports and returns the appropriate QuantLinear class based on the given backend.

    This function dynamically loads the correct `QuantLinear` class based on the backend and quantization
    configuration (e.g., qbits, marlin, hpu, gptq, awq, auto_round). It imports specific modules or raises
    errors if the required packages are not installed or the environment is not set up.

    Args:
        backend (str):
            The backend to be used for quantization (e.g., 'qbits', 'marlin', 'hpu', 'gptq', 'awq', 'auto_round').
        bits (int):
            The number of bits to be used for quantization.
        group_size (Optional[int]):
            The size of the quantization group (if applicable).
        sym (bool):
            Whether symmetric quantization is required.

    Returns:
        class:
            The dynamically imported QuantLinear class that corresponds to the given backend configuration.

    Raises:
        ImportError:
            If required modules are missing for a backend (e.g., Intel Extension, GPTQ, auto_awq).
    """
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
        else:  # auto_round must be at the end
            import auto_round_extension.qbits.qlinear_qbits as qlinear_qbits_autoround
            return qlinear_qbits_autoround.QuantLinear
    if "ipex_gptq" in backend:
        from auto_round_extension.ipex.qlinear_ipex_gptq import QuantLinear
        return QuantLinear

    if "ipex_awq" in backend:
        from auto_round_extension.ipex.qlinear_ipex_awq import QuantLinear
        return QuantLinear

    if "marlin" in backend:
        from transformers.utils.versions import require_version
        require_version(
            "gptqmodel",
            "marlin format requires gptqmodel to be installed, `pip install -v gptqmodel --no-build-isolation`"
        )
        from gptqmodel.nn_modules.qlinear.qlinear_marlin_inference import \
            MarlinInferenceQuantLinear  # pylint: disable=E0401
        return MarlinInferenceQuantLinear

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

    if "gptq" in backend:
        return get_autogptq_infer_linear(backend, bits, group_size, sym)

    if "awq" in backend:
        try:
            from awq.modules.linear import WQLinear_GEMM  # pylint: disable=E0401
        except ImportError:
            raise ImportError(
                "autoawq is required. Please install it by 'pip install autoawq' to support auto_awq format.")
        return WQLinear_GEMM

    if "auto_round" in backend:
        if "exllamav2" in backend:
            import auto_round_extension.cuda.qlinear_exllamav2
            return auto_round_extension.cuda.qlinear_exllamav2.QuantLinear
        else:
            import auto_round_extension.cuda.qlinear_tritonv2
            return auto_round_extension.cuda.qlinear_tritonv2.QuantLinear


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
            disable_marlin=disable_marlin
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
            use_tritonv2=use_tritonv2
        )

    return QuantLinear


def get_layer_backend(device, backend, orig_backend, bits, group_size, sym, in_features, out_features):
    """Selects the most suitable backend for the layer based on compatibility and priority.

    This function first checks if the specified backend supports the layer with the provided configuration.
    If not, it iterates through other available backends,
    checking compatibility and returning the one with the highest priority.

    Args:
        device (str):
            The device on which the layer will run, e.g., 'cpu', 'cuda'.
        backend (str):
            The target backend to be used for this layer.
        orig_backend (str):
            The original backend from which packing format information is retrieved.
        bits (int):
            The number of bits used for quantization.
        group_size (int):
            The group size for quantization.
        sym (bool):
            Whether symmetric quantization is enabled.
        in_features (int):
            The number of input features for the layer.
        out_features (int):
            The number of output features for the layer.

    Returns:
        str:
            The selected backend that is compatible with the layer configuration.

    Raises:
        AssertionError:
            If the specified backend is not supported.
        ValueError:
            If no compatible backend is found for the given layer configuration.
    """
    # Check if the provided backend is in BackendInfos
    assert backend in BackendInfos.keys(), \
        f"Unsupported backend {backend}, please set it to `auto` to try automatic selection"

    packing_format = BackendInfos[orig_backend].packing_format

    # Check if the provided backend supports the layer configuration
    if check_compatible(backend, device, bits, group_size, sym, packing_format, in_features, out_features):
        return backend

    # Find and store other compatible backends
    supported_backends = []
    for key in BackendInfos.keys():
        if key == backend:
            continue
        if check_compatible(key, device, bits, group_size, sym, packing_format, in_features, out_features):
            supported_backends.append(key)

    # Raise an error if no compatible backends are found
    if len(supported_backends) == 0:
        supported_backends_need_package = []
        for key in BackendInfos.keys():
            if check_compatible(key, device, bits, group_size, sym, packing_format, in_features, out_features,
                                check_requirements=False):
                supported_backends_need_package.append(key)

        if len(supported_backends_need_package) > 0:
            supported_backends_need_package = sorted(supported_backends_need_package,
                                                     key=lambda support_backend: BackendInfos[support_backend].priority,
                                                     reverse=True)
            backend_info = BackendInfos[supported_backends_need_package[0]]
            str_info = ",".join(backend_info.requirements)
            logging.error(f"`pip install {str_info}` to support inference")
            exit(-1)

        raise ValueError(f"None of the backends support this layer")

    # Sort the compatible backends by priority and return the one with the highest priority
    supported_backends = sorted(supported_backends, key=lambda support_backend: BackendInfos[support_backend].priority,
                                reverse=True)

    return supported_backends[0]
