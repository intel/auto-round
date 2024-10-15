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
import torch

import auto_round_extension.cuda.qlinear_exllamav2
from auto_round.utils import is_hpu_supported, get_autogptq_infer_linear

BackendInfos = {}


@dataclass
class BackendInfo:
    device: List[str]
    sym: List[bool]
    packing_format: str
    bits: List[int]
    group_size: Optional[List[int]] = None
    priority: int = 0  ##higher is better
    # require_packages: List[str] = field(default_factory=list)
    convertable_format: List[str] = field(default_factory=list)
    feature_checks: List[Any] = field(default_factory=list)
    inference_layer: Any = None


def feature_multiply_checker(in_feature, out_feature, multiply):
    return in_feature % multiply == 0 and out_feature % multiply == 0


def feature_num_greater_checker(in_feature, out_feature, num):
    return in_feature * out_feature > num


feature_multiply_checker_32 = functools.partial(feature_multiply_checker, multiply=32)

feature_num_greater_checker_1024 = functools.partial(feature_num_greater_checker, num=1024)

BackendInfos['auto_round:exllamav2'] = BackendInfo(device=["cuda"], sym=[True, False],
                                                   packing_format="triton",
                                                   bits=[4], group_size=None,
                                                   priority=5,
                                                   feature_checks=[feature_multiply_checker_32],
                                                   )

BackendInfos['auto_round:tritonv2'] = BackendInfo(device=["cuda"], sym=[True, False],
                                                  packing_format="triton",
                                                  bits=[2, 4, 8], group_size=None,
                                                  priority=0, feature_checks=[feature_multiply_checker_32],
                                                  )

BackendInfos['gptq:exllamav2'] = BackendInfo(device=["cuda"], sym=[True, False],
                                             packing_format="triton_zp+-1",
                                             bits=[4], group_size=None,
                                             priority=5,
                                             feature_checks=[feature_multiply_checker_32]
                                             )

BackendInfos['gptq:tritonv2'] = BackendInfo(device=["cuda"], sym=[True, False],
                                            packing_format="triton_zp+-1",
                                            bits=[2, 4, 8], group_size=None,
                                            priority=0, feature_checks=[feature_multiply_checker_32])

BackendInfos['awq:gemm'] = BackendInfo(device=["cuda"], sym=[True, False],  ##actrally is gemm
                                       packing_format="awq",
                                       bits=[4], group_size=None,
                                       priority=4, feature_checks=[feature_num_greater_checker_1024])

BackendInfos['auto_round:qbits'] = BackendInfo(device=["cpu"], sym=[True, False],
                                               packing_format="qbits",
                                               bits=[2, 4, 8], group_size=None,
                                               priority=0,
                                               feature_checks=[],
                                               convertable_format=["triton"])

BackendInfos['auto_round:qbits_zp'] = BackendInfo(device=["cpu"], sym=[True, False],
                                                  packing_format="qbits",
                                                  bits=[2, 4, 8], group_size=None,
                                                  priority=0,
                                                  feature_checks=[],
                                                  convertable_format=["triton_zp+-1"])

# BackendInfos['auto_round:gptq:qbits'] = BackendInfo(device=["cpu"], sym=[True, False],
#                                                     packing_format="qbits_zp+-1",
#                                                     bits=[2, 4, 8], group_size=None,
#                                                     priority=0,
#                                                     feature_checks=[],
#                                                     convertable_format=["triton_zp+-1"])


BackendInfos['gptq:marlin'] = BackendInfo(device=["gpu"], sym=[True],
                                          packing_format="marlin",
                                          bits=[4], group_size=[-1, 32, 128],
                                          priority=5,
                                          feature_checks=[feature_multiply_checker_32])

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


def check_compatible(backend_name, device, bits, group_size, sym, packing_format, in_features, out_features):
    backend = BackendInfos[backend_name]
    if not device in backend.device:
        return False
    if bits not in backend.bits:
        return False
    if backend.group_size is not None and group_size not in backend.group_size:
        return False
    if sym not in backend.sym:
        return False
    if packing_format == backend.packing_format:
        for check in backend.feature_checks:
            if not check(in_features, out_features):
                return False
    if packing_format != backend.packing_format and packing_format not in backend.convertable_format:  ##need to convert
        return False
    return True


def dynamic_import_inference_linear(backend, bits, group_size, sym):
    """Dynamically imports and returns the appropriate QuantLinear class based on the given bits and backend.

       Args:
           bits (int):
               The number of bits for quantization.
           backend (str):
               The backend to be used for quantization, such as "qbits", "cpu", or "exllamav2".

       Returns:
           class:
               The appropriate QuantLinear class for the given configuration.
       """
    if "qbits" in backend:
        try:
            from intel_extension_for_transformers import qbits  # pylint: disable=E0401
        except Exception as e:
            raise ImportError("Please install Intel Extension for Transformers via 'pip install "
                              "intel-extension-for-transformers' to  inference on X86 CPU")
        if "zp" in backend:
            import auto_round_extension.qbits.qlinear_qbits_gptq as qlinear_qbits_gptq
            return qlinear_qbits_gptq.QuantLinear
        else:  ## auto_round must in the end
            import auto_round_extension.qbits.qlinear_qbits as qlinear_qbits_autoround
            return qlinear_qbits_autoround.QuantLinear

    if "hpu" in backend:
        try:
            import habana_frameworks.torch.hpu  # pylint: disable=E0401
        except:
            raise ImportError("Please setup hpu environment before using hpu backend")

        if "zp" in backend:
            from auto_round_extension.hpu.qlinear_hpu_gptq import QuantLinear as QuantLinear_gptq
            return QuantLinear_gptq
        else:  ## auto_round must in the end
            from auto_round_extension.hpu.qlinear_hpu import QuantLinear
            return QuantLinear

    if "gptq" in backend:
        return get_autogptq_infer_linear(backend, bits, group_size, sym)

    if "awq" in backend:
        try:
            from awq.modules.linear import WQLinear_GEMM  # pylint: disable=E0401
        except:
            raise ImportError("autoawq is required. Please install it by 'pip install autoawq' to \
                                 support auto_awq format.")
        return WQLinear_GEMM

    if "auto_round" in backend:
        if "exllamav2" in backend:
            return auto_round_extension.cuda.qlinear_exllamav2.QuantLinear
        else:
            return auto_round_extension.cuda.qlinear_tritonv2.QuantLinear


def get_layer_backend(device, backend, orig_backend, bits, group_size, sym, in_features, out_features):
    ##check device

    assert backend in BackendInfos.keys(), f"Unsupported backend f{backend}, please set it to `auto` to have a try"
    packing_format = BackendInfos[orig_backend].packing_format
    ##first check the current backend whether support this layer
    if check_compatible(backend, device, bits, group_size, sym, packing_format, in_features, out_features):
        return backend

    supported_backends = []
    for key in BackendInfos.keys():
        if key == backend:
            continue
        if check_compatible(key, device, bits, group_size, sym, packing_format, in_features, out_features):
            supported_backends.append(key)

    if len(supported_backends) == 0:
        raise ValueError(f"None of the backends support this layer")

    supported_backends = sorted(supported_backends, key=lambda support_backend: BackendInfos[support_backend].priority,
                                reverse=True)
    return supported_backends[0]


if __name__ == "__main__":
    res = get_layer_backend("cuda", "gptq:exllamav2", "gptq:exllamav2",4, 128, sym=False , in_features=128,
                            out_features=128)
    assert res == "gptq:exllamav2"

    res = get_layer_backend("cuda", "gptq:exllamav2", "gptq:exllamav2",2, 128, sym=False , in_features=128,
                            out_features=128)
    assert res == "gptq:tritonv2"

    res = get_layer_backend("cpu", "auto_round:exllamav2", "auto_round:exllamav2",4, 128, sym=False , in_features=128,
                            out_features=128)
    assert res == "auto_round:qbits"

    res = get_layer_backend("cpu", "gptq:exllamav2", "gptq:exllamav2",4, 128, sym=False , in_features=128,
                            out_features=128)
    assert res == "auto_round:qbits_zp"

