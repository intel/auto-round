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
from typing import List, Any, Optional

from transformers.utils.versions import require_version

import auto_round_extension.cuda.gptqmodel_marlin
from auto_round.utils import get_library_version, logger

BackendInfos = {}

import cpuinfo


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
    dtype: List[str] = None
    group_size: Optional[List[int]] = None
    priority: int = 0  ##higher is better
    convertable_format: List[str] = field(default_factory=list)
    feature_checks: List[Any] = field(default_factory=list)
    alias: Optional[List[str]] = None
    requirements: Optional[List[str]] = None


def feature_multiply_checker(in_feature, out_feature, group_size, in_feature_multiplier, out_feature_multiplier=None):
    if out_feature_multiplier is None:
        out_feature_multiplier = in_feature_multiplier
    return in_feature % in_feature_multiplier == 0 and out_feature % out_feature_multiplier == 0


def feature_multiply_checker_group_size(in_feature, out_feature, group_size, in_feature_multiplier,
                                        out_feature_multiplier=None):
    if out_feature_multiplier is None:
        out_feature_multiplier = in_feature_multiplier
    return (in_feature % in_feature_multiplier == 0 and out_feature % out_feature_multiplier == 0
            and in_feature % group_size == 0)


feature_multiply_checker_32 = functools.partial(feature_multiply_checker, in_feature_multiplier=32)
in_output_feature_multiply_checker_32 = functools.partial(feature_multiply_checker, in_feature_multiplier=32,
                                                          out_feature_multiplier=32)

exllamav2_feature_check = functools.partial(feature_multiply_checker_group_size, in_feature_multiplier=32,
                                            out_feature_multiplier=32)

gptqmodel_marlin_feature_check = functools.partial(feature_multiply_checker_group_size, in_feature_multiplier=1,
                                                   out_feature_multiplier=64)

BackendInfos['auto_gptq:exllamav2'] = BackendInfo(device=["cuda"], sym=[True, False],
                                                  packing_format="int32_zp",
                                                  bits=[4],
                                                  priority=5,
                                                  dtype=["float16"],
                                                  ##16, 384,768 accuracy issue
                                                  group_size=[-1, 32, 64, 128, 256, 512, 1024, 2048],
                                                  feature_checks=[exllamav2_feature_check],
                                                  alias=['gptq', 'auto_gptq', 'exllamav2', "gptq:exllamav2",
                                                         "auto_gptq:exllamav2"],
                                                  requirements=["auto-gptq>=0.7.1"]
                                                  )

BackendInfos['auto_gptq:tritonv2'] = BackendInfo(device=["cuda"], sym=[True, False],
                                                 packing_format="int32_zp",
                                                 bits=[2, 4, 8], group_size=None,
                                                 dtype=["float16"],
                                                 priority=0, feature_checks=[exllamav2_feature_check],
                                                 alias=["auto_gptq:tritonv2"],
                                                 requirements=["auto-gptq>=0.7.1", "triton>=2.0"]
                                                 )

BackendInfos['auto_gptq:cuda'] = BackendInfo(device=["cuda"], sym=[True, False],
                                             packing_format="int32_zp",
                                             bits=[2, 3, 4, 8], group_size=None,
                                             priority=0, feature_checks=[exllamav2_feature_check],
                                             alias=["auto_gptq:cuda"],
                                             dtype=["float16"],
                                             convertable_format=["int32_zp"],
                                             requirements=["auto-gptq>=0.7.1"]
                                             )

BackendInfos['auto_round:tritonv2'] = BackendInfo(device=["cuda"], sym=[True, False],
                                                  packing_format="int32",
                                                  dtype=["float16", "bfloat16"],
                                                  bits=[2, 4, 8],
                                                  priority=1, feature_checks=[feature_multiply_checker_32],
                                                  alias=["auto_round", "tritonv2", "triton"],
                                                  requirements=["auto-round>=0.5.0", "triton>=2.0"]
                                                  )

BackendInfos['auto_round:tritonv2_zp'] = BackendInfo(device=["cuda"], sym=[True],  ## asym has accuracy issue
                                                     packing_format="int32_zp",
                                                     dtype=["float16", "bfloat16"],
                                                     bits=[2, 4, 8],
                                                     priority=1, feature_checks=[feature_multiply_checker_32],
                                                     alias=["tritonv2", "tritonv2_zp", "triton"],
                                                     requirements=["auto-round>=0.5.0", "triton>=2.0"]
                                                     )

BackendInfos['gptqmodel:marlin'] = BackendInfo(device=["cuda"], sym=[True],
                                               packing_format="int32",
                                               bits=[4, 8],
                                               group_size=[-1, 32, 64, 128],
                                               dtype=["float16", "bfloat16"],
                                               priority=6, feature_checks=[gptqmodel_marlin_feature_check],
                                               alias=["marlin", "gptqmodel"],
                                               requirements=["gptqmodel>=2.0"],
                                               )

BackendInfos['gptqmodel:marlin_zp'] = BackendInfo(device=["cuda"], sym=[True],
                                                  packing_format="int32_zp",
                                                  bits=[4, 8],
                                                  group_size=[-1, 32, 64, 128],
                                                  dtype=["float16", "bfloat16"],
                                                  priority=6, feature_checks=[gptqmodel_marlin_feature_check],
                                                  alias=["marlin", "gptqmodel"],
                                                  requirements=["gptqmodel>=2.0"]
                                                  )

BackendInfos['gptqmodel:exllamav2'] = BackendInfo(device=["cuda"], sym=[True, False],
                                                  packing_format="int32",
                                                  bits=[4], group_size=[-1, 32, 64, 128],  ##16 seems has accuracy issue
                                                  dtype=["float16", "bfloat16"],
                                                  priority=5, feature_checks=[exllamav2_feature_check],
                                                  alias=["exllamav2"],
                                                  requirements=["gptqmodel>=2.0"]
                                                  )

BackendInfos['auto_awq:gemm'] = BackendInfo(device=["cuda"], sym=[True, False],  ##actually is gemm
                                            packing_format="awq",
                                            bits=[4], group_size=None,
                                            priority=5,
                                            dtype=["float16"],
                                            alias=["auto_awq:gemm", "awq", "awq:gemm",
                                                   "auto_awq"],
                                            requirements=["autoawq"]
                                            )

BackendInfos['qbits'] = BackendInfo(device=["cpu"], sym=[True, False],
                                    packing_format="qbits",
                                    bits=[2, 4, 8], group_size=None,
                                    priority=0,
                                    feature_checks=[],
                                    alias=["itrex", "qbits"],
                                    dtype=["float16", "bfloat16"],
                                    convertable_format=["int32"],
                                    requirements=["intel-extension-for-transformers"])

BackendInfos['qbits_zp'] = BackendInfo(device=["cpu"], sym=[True, False],
                                       packing_format="qbits_zp",
                                       bits=[2, 4, 8], group_size=None,
                                       dtype=["float16", "bfloat16"],
                                       priority=0,
                                       feature_checks=[],
                                       alias=["itrex", "qbits"],
                                       convertable_format=["int32_zp"],
                                       requirements=["intel-extension-for-transformers"]
                                       )

BackendInfos['auto_round:qbits_awq'] = BackendInfo(device=["cpu"], sym=[True, False],  ## for awq, not robust
                                                   packing_format="awq",
                                                   bits=[2, 4, 8], group_size=None,
                                                   priority=0,
                                                   feature_checks=[],
                                                   requirements=["intel-extension-for-transformers"]
                                                   )

BackendInfos['ipex_gptq'] = BackendInfo(device=["cpu", "xpu"], sym=[True, False],
                                        packing_format="ipex_gptq",
                                        bits=[4], group_size=None,
                                        priority=5,
                                        feature_checks=[],
                                        dtype=["float16", "bfloat16"],
                                        convertable_format=["int32_zp"],
                                        alias=["ipex"],
                                        requirements=["intel-extension-for-pytorch>=2.5"]
                                        )

BackendInfos['ipex_awq'] = BackendInfo(device=["cpu", "xpu"], sym=[True, False],
                                       packing_format="ipex_awq",
                                       bits=[4], group_size=None,
                                       priority=1,
                                       dtype=["float16", "bfloat16"],
                                       feature_checks=[],
                                       alias=["ipex"],
                                       convertable_format=["awq"],
                                       requirements=["intel-extension-for-pytorch>=2.6"]
                                       )

BackendInfos['hpu'] = BackendInfo(device=["hpu"], sym=[True, False],
                                  packing_format="hpu",
                                  bits=[4],
                                  dtype=["bfloat16"],
                                  alias=["hpu"],
                                  priority=0,
                                  convertable_format=["int32"]
                                  )

BackendInfos['hpu_zp'] = BackendInfo(device=["hpu"], sym=[True, False],
                                     packing_format="hpu_zp",
                                     bits=[4],
                                     dtype=["bfloat16"],
                                     alias=["hpu"],
                                     priority=0,
                                     convertable_format=["int32_zp"])


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

    # Check if the format is convertible when packing formats differ
    if packing_format == backend.packing_format or packing_format in backend.convertable_format:
        pass
    else:
        return False

    for check in backend.feature_checks:
        if not check(in_features, out_features, group_size):
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
                "autoawq is required. Please install it by 'pip install autoawq' to support auto_awq format.")
        return WQLinear_GEMM

    if backend == "auto_round:tritonv2":
        from auto_round_extension.cuda.qlinear_tritonv2 import QuantLinear
        return QuantLinear

    if backend == "auto_round:tritonv2_zp":
        from auto_round_extension.cuda.qlinear_tritonv2_zp import QuantLinear
        return QuantLinear

    raise ValueError(f"unsupported backend {backend}, please set it to `auto` and retry")


def get_gptqmodel_infer_linear(backend, bits=4, group_size=128, sym=False):
    import gptqmodel  # pylint: disable=E0401
    if "marlin" in backend:
        return auto_round_extension.cuda.gptqmodel_marlin.get_marlin_layer()
        # return gptqmodel.nn_modules.qlinear.marlin.MarlinQuantLinear
    elif "exllamav2" in backend:
        return gptqmodel.nn_modules.qlinear.exllamav2.ExllamaV2QuantLinear
    elif "tritonv2" in backend:
        return gptqmodel.nn_modules.qlinear.tritonv2.TritonV2QuantLinear
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


def find_backend(target_backend: str, orig_backend: str = None):
    """
    Finds the matching backend key based on the target backend name or its aliases.

    Args:
        target_backend (str): Name or alias of the target backend.
        orig_backend (str, optional): Original backend name to check compatibility. Defaults to None.

    Returns:
        str or None: Matching backend key if found and compatible; otherwise, None.
    """
    matched_keys = [
        key for key, info in BackendInfos.items()
        if key == target_backend or (info.alias and target_backend in info.alias)
    ]

    if not matched_keys:
        return None

    if orig_backend is None:
        return matched_keys[0] if len(matched_keys) >= 1 else None

    orig_info = BackendInfos[orig_backend]

    for key in matched_keys:
        target_info = BackendInfos[key]
        if (target_info.packing_format == orig_info.packing_format or
                orig_info.packing_format in target_info.convertable_format):
            return key

    raise ValueError(
        f"{target_backend} is not compatible with {orig_backend}. "
        f"Please set `backend` to `auto` and retry."
    )


def get_all_compatible_backend(device, backend, orig_backend, bits, group_size, sym, in_features, out_features):
    # Get packing format from the original backend
    packing_format = BackendInfos[orig_backend].packing_format

    # Find compatible backends
    compatible_backends = [
        key for key in BackendInfos.keys()
        if check_compatible(key, device, bits, group_size, sym, packing_format, in_features, out_features,
                            check_requirements=False)
    ]

    # Return the first compatible backend or an empty list if none found
    return compatible_backends


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
    backend = find_backend(backend)
    assert backend in BackendInfos.keys(), \
        f"Unsupported backend {backend}, please set it to `auto` to try automatic selection"

    packing_format = BackendInfos[orig_backend].packing_format

    # Find and store other compatible backends
    supported_backends = []
    for key in BackendInfos.keys():
        if check_compatible(key, device, bits, group_size, sym, packing_format, in_features, out_features):
            supported_backends.append(key)

    # Raise an error if no compatible backends are found
    if len(supported_backends) == 0:
        supported_backends_need_package = get_all_compatible_backend(device, backend, orig_backend, bits, group_size,
                                                                     sym,
                                                                     in_features, out_features)

        if len(supported_backends_need_package) > 0:
            supported_backends_need_package = sorted(supported_backends_need_package,
                                                     key=lambda support_backend: BackendInfos[support_backend].priority,
                                                     reverse=True)
            backend_info = BackendInfos[supported_backends_need_package[0]]
            logger.error("please install all the following packages to support inference")
            for requirement in backend_info.requirements:
                if isinstance(requirement, str):
                    try:
                        require_version(requirement)
                    except ImportError:
                        if "gptqmodel" in requirement:
                            logger.error(f"pip install -v '{requirement}' --no-build-isolation")
                        else:
                            logger.error(f"pip install '{requirement}' ")
                else:
                    str_info = requirement()[1]
                    logger.error(str_info)
            exit(-1)

        return None

    # Sort the compatible backends by priority and return the one with the highest priority
    supported_backends = sorted(supported_backends, key=lambda support_backend: BackendInfos[support_backend].priority,
                                reverse=True)

    return supported_backends[0]


def get_highest_priority_backend(bits, sym, group_size, device, packing_format):
    supported_backends = []
    for key in BackendInfos.keys():
        backend = BackendInfos[key]
        # Check if device is supported by the backend
        if device not in backend.device:
            continue

        # Check if bit-width is supported
        if bits not in backend.bits:
            continue

        # Check if group_size is valid (if required by backend)
        if backend.group_size is not None and group_size not in backend.group_size:
            continue

        # Check if symmetric/asymmetric quantization is supported
        if sym not in backend.sym:
            continue

        # Check if the format is convertible when packing formats differ
        if packing_format == backend.packing_format or packing_format in backend.convertable_format:
            pass
        else:
            continue
        supported_backends.append(key)

    if len(supported_backends) > 0:

        supported_backends = sorted(supported_backends,
                                    key=lambda support_backend: BackendInfos[support_backend].priority,
                                    reverse=True)
        return supported_backends[0]
    else:
        return None


def process_requirement(requirements: list):
    gptqmodel_requirements = None
    other_requirements = []
    for requirement in requirements:
        try:
            require_version(requirement)
        except:
            if "gptqmodel" in requirement:
                gptqmodel_requirements = requirement
            else:
                other_requirements.append(requirement)

    infos = []

    if gptqmodel_requirements is not None:
        infos.append(f"pip install -v '{gptqmodel_requirements}' --no-build-isolation")
        try:
            require_version("numpy<2.0")
        except:
            infos.append(f"pip install 'numpy<2.0'")

    other_info = f"pip install"
    if len(other_requirements) > 0:
        for requirement in other_requirements:
            other_info += f" {requirement}"
        infos.append(other_info)
    return infos
