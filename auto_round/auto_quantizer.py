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

# coding=utf-8
# Copyright 2023 HuggingFace Inc. team and GPTQ and AutoGPTQ authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import importlib.util
import warnings
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from packaging import version
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import Conv1D
from transformers.quantizers import AutoQuantizationConfig, HfQuantizer
from transformers.quantizers.auto import AUTO_QUANTIZER_MAPPING
from transformers.utils.quantization_config import AwqConfig, GPTQConfig, QuantizationConfigMixin, QuantizationMethod

from auto_round.utils import get_module, set_module, is_hpu_supported

from auto_round.backend import get_layer_backend, dynamic_import_inference_linear

from auto_round.backend import BackendInfos
from transformers.utils.versions import require_version
from enum import Enum
from tqdm import tqdm
import copy
import re

logger = getLogger(__name__)
import sys

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

AUTOROUND_MINIMUM_VERSION = version.parse("0.2")


def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    try:  ##TODO remove it later
        import auto_round
        return True, auto_round.__version__
    except:
        pass

    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


_auto_round_available = _is_package_available("auto_round")


def is_auto_round_available():
    try:
        import auto_round
        return True
    except:
        pass
    if _auto_round_available:
        version_autoround = version.parse(importlib_metadata.version("auto_round"))
        if AUTOROUND_MINIMUM_VERSION < version_autoround:
            return True
        else:
            raise ImportError(
                f"Found an incompatible version of auto-round. Found version {version_autoround},"
                f" but only version above {AUTOROUND_MINIMUM_VERSION} are supported"
            )


#
def get_device(obj: Union[torch.Tensor, nn.Module]):
    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device


class AutoHfQuantizer:
    """The Auto-HF quantizer class that takes care of automatically instantiating to the correct
    `HfQuantizer` given the `QuantizationConfig`."""

    @classmethod
    def from_config(cls, quantization_config: Union[QuantizationConfigMixin, Dict], **kwargs):
        # Convert it to a QuantizationConfig if the q_config is a dict
        if isinstance(quantization_config, dict):
            if "auto-round" in quantization_config["quant_method"]:
                quantization_config = AutoRoundConfig.from_dict(quantization_config)
            else:
                quantization_config = AutoQuantizationConfig.from_dict(quantization_config)  # pylint: disable=E1101
        quant_method = quantization_config.quant_method

        # Again, we need a special care for bnb as we have a single quantization config
        # class for both 4-bit and 8-bit quantization
        if quant_method == QuantizationMethod.BITS_AND_BYTES:
            if quantization_config.load_in_8bit:
                quant_method += "_8bit"
            else:
                quant_method += "_4bit"

        if quant_method not in AUTO_QUANTIZER_MAPPING.keys() and "auto-round" not in quant_method:
            raise ValueError(
                f"Unknown quantization type, got {quant_method} - supported types are:"
                f" {list(AUTO_QUANTIZER_MAPPING.keys())}"
            )
        if "auto-round" in quant_method or is_hpu_supported():  # pragma: no cover
            target_cls = AutoRoundQuantizer
        else:
            target_cls = AUTO_QUANTIZER_MAPPING[quant_method]

        return target_cls(quantization_config, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        quantization_config = AutoQuantizationConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls.from_config(quantization_config)

    @classmethod
    def merge_quantization_configs(
            cls,
            quantization_config: Union[dict, QuantizationConfigMixin],
            quantization_config_from_args: Optional[QuantizationConfigMixin],
    ):
        """Handles situations where both quantization_config
        from args and quantization_config from model config are present."""
        if quantization_config_from_args is not None:
            warning_msg = (
                "You passed `quantization_config` or equivalent parameters to "
                "`from_pretrained` but the model you're loading"
                " already has a `quantization_config` attribute. The `quantization_config` from the model will be used."
            )
        else:
            warning_msg = ""

        if isinstance(quantization_config, dict):
            if "auto-round" in quantization_config["quant_method"]:
                quantization_config = AutoRoundConfig.from_dict(quantization_config)
            else:
                quantization_config = AutoQuantizationConfig.from_dict(quantization_config)  # pylint: disable=E1101

        if isinstance(quantization_config,
                      (GPTQConfig, AwqConfig, AutoRoundConfig)) and quantization_config_from_args is not None:
            # special case for GPTQ / AWQ config collision
            loading_attr_dict = quantization_config_from_args.get_loading_attributes()
            for attr, val in loading_attr_dict.items():
                setattr(quantization_config, attr, val)
            warning_msg += (
                f"However, loading attributes (e.g. {list(loading_attr_dict.keys())}) "
                f"will be overwritten with the one you passed to `from_pretrained`. The rest will be ignored."
            )

        if warning_msg != "":
            warnings.warn(warning_msg)

        return quantization_config


class AutoRoundQuantizationMethod(str, Enum):
    AutoRound = "intel/auto-round"


@dataclass
class AutoRoundConfig(QuantizationConfigMixin):
    """This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded AutoRound quantization.

    Args:
        bits (`int`):
            The number of bits to quantize to, supported numbers are (2, 3, 4, 8).
        tokenizer (`str` or `PreTrainedTokenizerBase`, *optional*):
            The tokenizer used to process the dataset. You can pass either:
                - A custom tokenizer object.
                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                    using the [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.
    """

    def __init__(
            self,
            bits: int = 4,
            tokenizer: Any = None,
            dataset: str = None,
            group_size: int = 128,
            sym: bool = False,
            backend="auto",
            layer_config: dict = None,
            **kwargs,
    ):

        self.bits = bits
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.group_size = group_size
        self.sym = sym
        self.backend = backend
        self.layer_config = layer_config
        if kwargs is not None:
            for key in kwargs.keys():
                setattr(self, key, kwargs[key])
        self.quant_method = AutoRoundQuantizationMethod.AutoRound
        self.post_init()

    def post_init(self):
        r"""Safety checker that arguments are correct."""
        if self.bits not in [2, 4, 8]:
            raise ValueError(f"Only support quantization to [2,4,8] bits but found {self.bits}")
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("group_size must be greater than 0 or equal to -1")

    def get_loading_attributes(self):
        # attributes_dict = copy.deepcopy(self.__dict__)
        loading_attibutes_dict = {"target_backend": self.backend}
        # loading_attributes = ["backend"]
        # loading_attibutes_dict = {i: j for i, j in attributes_dict.items() if i in loading_attributes}
        return loading_attibutes_dict

    def to_dict(self):
        config_dict = super().to_dict()
        config_dict.pop("disable_exllama", None)
        return config_dict


class AutoRoundQuantizer(HfQuantizer):
    """Quantizer of the AutoRound method, currently only triton and exllamav2 backend has been supported."""

    requires_calibration = False
    required_packages = ["auto_round"]
    optimum_quantizer = None

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        if not is_auto_round_available():
            raise ImportError("Loading a AutoRound quantized model requires auto-round library (`pip install "
                              "auto-round`)")
        else:
            try:
                import auto_round
                autoround_version = version.parse(auto_round.__version__)
            except:
                autoround_version = version.parse(importlib.metadata.version("auto_round"))
            if autoround_version < version.parse("0.2.0"):
                raise ImportError("You need a version of auto_round > 0.2.0 to use AutoRound: `pip install --upgrade "
                                  "auto-round` or install from source")

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            torch_dtype = torch.float16
        elif torch_dtype != torch.float16 and not is_hpu_supported():
            logger.info("We suggest you to set `torch_dtype=torch.float16` for better efficiency with AutoRound.")
        return torch_dtype

    def find_backend(self, target_backend: str):
        """Finds the matching backend key based on the target backend or its alias.

        This function checks if the provided `target_backend` is directly present in `BackendInfos`.
        If not, it iterates through the backends to see if the `target_backend` matches any backend's alias.

        Args:
            target_backend (str):
                The name of the backend or alias to find.

        Returns:
            str or None:
                The backend key if a match is found, otherwise `None`.
        """
        # Directly return if target_backend exists in BackendInfos
        if target_backend in BackendInfos:
            return target_backend

        # Search through BackendInfos to check if target_backend matches any backend alias
        for key in BackendInfos.keys():
            backendInfo = BackendInfos[key]
            if backendInfo.alias is not None and target_backend in backendInfo.alias:
                return key

        # Return None if no matching backend or alias is found
        return None

    def detect_device(self, target_backend, orig_backend):
        """Detects the appropriate device for the specified backend.

        This function determines the device type based on the target backend. If the target backend is
        not specified, it defaults to the original backend. The function checks for the availability
        of CUDA, HPU, or CPU, and returns the appropriate device type.

        Args:
            target_backend (str or None):
                The name of the target backend. If None, defaults to `orig_backend`.
            orig_backend (str):
                The original backend name to fall back on if `target_backend` is None.

        Returns:
            str:
                The type of device detected ('cuda', 'hpu', or 'cpu').

        Raises:
            ValueError:
                If the specified backend cannot be found.
        """
        # Default to the original backend if target_backend is not provided
        if target_backend is None:
            target_backend = orig_backend

        # Check for specific device types based on the target backend
        if "cuda" in target_backend:
            return "cuda"
        elif "hpu" in target_backend:
            return "hpu"
        elif "cpu" in target_backend:
            return "cpu"

        # Determine the device automatically based on availability
        if target_backend.split(":")[0] == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif is_hpu_supported():
                return "hpu"
            else:
                return "cpu"

        # Find the backend and determine the device type from BackendInfos
        backend = self.find_backend(target_backend)
        if backend is None:
            raise ValueError("Backend not found, please set it to 'auto' to have a try ")

        return BackendInfos[backend].device[0]

    def convert_model(self, model: nn.Module):
        """Converts the given model to an AutoRound model by replacing its layers with quantized layers.

        This method extracts the quantization configuration from the model and adjusts its layers
        according to the specified quantization parameters. It supports different backends and
        ensures that the model's data type is compatible with the selected hardware.

        Args:
            model (nn.Module):
                The model to be converted into an AutoRound model.

        Returns:
            nn.Module:
                The converted AutoRound model with quantized layers.

        Raises:
            ValueError:
                If the quantization backend is not specified in the configuration.
        """

        from auto_round.utils import get_layer_names_in_block

        quantization_config = model.config.quantization_config
        if not hasattr(quantization_config, "target_backend"):
            quantization_config.target_backend = quantization_config.backend

        target_device = self.detect_device(quantization_config.target_backend, quantization_config.backend)
        self.target_device = target_device

        if hasattr(quantization_config, "backend"):  # pragma: no cover
            if ("hpu" == target_device or "cpu" == target_device) and model.dtype != torch.bfloat16:
                logger.info(f"Change the dtype to `bfloat16` as {target_device.upper()} does not support float16")
                model = model.to(torch.bfloat16)
            else:
                if model.dtype != torch.float16:
                    logger.info(f"Change the dtype to `float16` for better performance")
                    model = model.to(torch.float16)

        bits = quantization_config.bits
        group_size = quantization_config.group_size
        data_type = quantization_config.data_type if hasattr(quantization_config,
                                                             "data_type") else "int"  # pragma: no cover
        sym = quantization_config.sym
        to_quant_block_names = quantization_config.to_quant_block_names if hasattr(quantization_config,
                                                                                   "to_quant_block_names") else None
        layer_names = get_layer_names_in_block(model, to_quant_block_names=to_quant_block_names)

        extra_config = {}
        if hasattr(quantization_config, "extra_config"):
            extra_config = quantization_config.extra_config

        layer_names += extra_config.keys()
        layer_names = list(set(layer_names))

        layer_configs = {}
        for layer_name in layer_names:
            layer_configs[layer_name] = {}
            if layer_name not in extra_config:
                layer_configs[layer_name]["bits"] = bits
                layer_configs[layer_name]["group_size"] = group_size
                layer_configs[layer_name]["data_type"] = data_type
                layer_configs[layer_name]["sym"] = sym
                layer_configs[layer_name]["clip"] = False
            else:
                layer_configs[layer_name]["bits"] = extra_config[layer_name].get("bits", bits)
                layer_configs[layer_name]["group_size"] = extra_config[layer_name].get("group_size", group_size)
                layer_configs[layer_name]["data_type"] = extra_config[layer_name].get("data_type", data_type)
                layer_configs[layer_name]["sym"] = extra_config[layer_name].get("sym", sym)
                layer_configs[layer_name]["clip"] = extra_config[layer_name].get("clip", False)

        if hasattr(quantization_config, "backend"):  # pragma: no cover
            backend = quantization_config.backend
        elif 'gptq' in quantization_config.quant_method:  # pragma: no cover
            backend = 'gptq'
        else:  # pragma: no cover
            logger.error("Please specify quantization backend")
            raise ValueError("Quantization backend must be specified.")

        self._replace_by_quant_layers(model, layer_configs, quantization_config.target_backend, target_device, backend)
        return model

    def _replace_by_quant_layers(self, module: nn.Module, layer_configs, target_backend, target_device, orig_backend):
        """Replaces linear layers in the given module with quantized layers.

        This method iterates over the specified layer configurations and replaces
        the original layers in the module with instances of `QuantLinear`. It handles
        various layer types and ensures that the correct quantization parameters are applied.

        Args:
            module (nn.Module):
                The module containing layers to be quantized.
            layer_configs (dict):
                A dictionary containing configuration for each layer's quantization.
            target_backend (str):
                The backend to use for quantization, which includes device and format information.
            target_device (str):
                The device on which the model will run (e.g., 'cuda', 'cpu', 'hpu').
            orig_backend (str):
                The original backend of the packing.

        Raises:
            AssertionError:
                If any condition related to backend or quantization configuration is not met.
        """

        def remove_device_str(s, device_str):
            if s and s.startswith(device_str):
                return s[len(device_str):].lstrip(":")
            return s

        if "auto" == target_backend.split(':')[0]:
            target_backend = target_backend[4:]  # Remove 'auto'
            if len(target_backend) >= 1 and target_backend[0] == ":":
                target_backend = target_backend[1:]

        # Remove device info from target_backend
        target_backend = remove_device_str(target_backend, "cpu")
        target_backend = remove_device_str(target_backend, "hpu")
        target_backend = remove_device_str(target_backend, "cuda")
        orig_backend = self.find_backend(orig_backend)

        if target_backend == "":
            target_backend = orig_backend

        self.need_marlin_repacking = False

        for layer_name in layer_configs.keys():
            config = layer_configs[layer_name]
            bits = config["bits"]
            group_size = config["group_size"]
            data_type = config["data_type"]
            sym = config["sym"]
            clip = config["clip"]

            if not (bits <= 8):
                continue

            layer = get_module(module, layer_name)
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                out_features = layer.out_features
            elif isinstance(layer, nn.Conv2d):  # Not supported currently
                in_features = layer.in_channels
                out_features = layer.out_channels
            elif isinstance(layer, Conv1D):  # TODO: Needs verification
                in_features = layer.weight.shape[0]
                out_features = layer.weight.shape[1]
            else:
                continue

            if "marlin" in target_backend and "marlin" not in orig_backend:
                # Need to repack
                assert sym == True, "Marlin only supports symmetric quantization"
                assert target_device == "cuda", "Marlin only supports CUDA device"
                assert not "awq" in orig_backend, "Marlin does not support repacking from AWQ format"
                self.need_marlin_repacking = True
                # Using original backend to load the layer then replace
                layer_backend = orig_backend
            else:
                target_backend = self.find_backend(target_backend)  # TODO: Move out if have supported marlin
                layer_backend = get_layer_backend(
                    target_device, target_backend, orig_backend, bits, group_size, sym, in_features, out_features
                )
            if "gptq" in layer_backend and "exllamav2" in layer_backend:
                try:
                    from exllamav2_kernels import gemm_half_q_half, make_q_matrix  # pylint: disable=E0611
                except:
                    logger.warning_once(
                        "For better inference performance, please install exllamav2 kernel "
                        "via `pip install git+https://github.com/AutoGPTQ/AutoGPTQ.git@b8b4127`")

            QuantLinear = dynamic_import_inference_linear(layer_backend, bits, group_size, sym)

            layer_device = get_device(layer)

            bias = layer.bias is not None
            if "awq" in layer_backend:
                new_layer = QuantLinear.from_linear(  # pylint: disable=E1123
                    layer,
                    bits,
                    group_size,
                    init_only=True
                )
            else:
                try:
                    new_layer = QuantLinear(  # pylint: disable=E1123
                        bits,
                        group_size,
                        in_features,
                        out_features,
                        bias,
                        weight_dtype=layer.weight.dtype,
                        clip=clip
                    )
                except:
                    new_layer = QuantLinear(  # pylint: disable=E1123
                        bits,
                        group_size,
                        in_features,
                        out_features,
                        bias,
                        weight_dtype=layer.weight.dtype,
                    )

            new_layer.device = layer_device
            set_module(module, layer_name, new_layer)

    def cpu_post_init(self, model):
        dep_check = True
        message = "Repacking to CPU format"
        layers = []  ## ipex post_init  will add one more layer
        for n, m in model.named_modules():
            layers.append((n, m))

        for n, layer in tqdm(layers, desc=message, total=len(layers),
                             leave=True):
            from auto_round_extension.qbits import qbits_qlinear_classes
            from auto_round_extension.ipex import ipex_qlinear_classes
            if isinstance(layer, qbits_qlinear_classes):
                if dep_check:
                    layer.req_check()
                layer.post_init()
                dep_check = False
            if isinstance(layer, ipex_qlinear_classes):
                layer.post_init()

        return model

    def repack_marlin(self, model):
        """Repack the model to use Marlin format for quantized layers.

        This method iterates through the model's modules, identifies instances of
        `QuantLinear`, and replaces them with `MarlinInferenceQuantLinear`. It
        handles the initialization of various parameters and the repacking of
        quantized weights and scales for optimized performance on Marlin.

        Args:
            model (nn.Module):
                The model to be repacked into Marlin format.

        Raises:
            ImportError:
                If the required modules for Marlin inference cannot be imported.
        """
        message = "Repacking to Marlin format"

        for n, m in tqdm(model.named_modules(), desc=message, total=len(list(model.named_modules()))):
            if m.__class__.__name__ == "QuantLinear":
                try:
                    from gptqmodel.nn_modules.qlinear.qlinear_marlin_inference import (  # pylint: disable=E0401
                        MarlinInferenceQuantLinear,
                        marlin_permute_scales,
                        marlin_make_workspace
                    )
                except ImportError:
                    raise ImportError("Failed to import Marlin inference modules.")

                with torch.device("meta"):
                    # Create a new MarlinInferenceQuantLinear module with the appropriate parameters.
                    new_module = MarlinInferenceQuantLinear(
                        bits=4,
                        group_size=m.group_size,
                        sym=True,
                        desc_act=False,
                        infeatures=m.infeatures,
                        outfeatures=m.outfeatures,
                        bias=m.bias is not None,
                    )

                device = m.qweight.device
                import gptqmodel_marlin_cuda_inference  # pylint: disable=E0401

                # Initialize the necessary parameters for the new module.
                new_module.g_idx = torch.nn.Parameter(torch.empty(0, dtype=torch.int, device=device),
                                                      requires_grad=False)
                new_module.g_idx_sort_indices = torch.nn.Parameter(torch.empty(0, dtype=torch.int, device=device),
                                                                   requires_grad=False)
                new_module.zp = torch.nn.Parameter(torch.empty(0, dtype=torch.int, device=device),
                                                   requires_grad=False)
                new_module.bias = m.bias

                # Repack the quantized weight for the Marlin format.
                marlin_qweight = gptqmodel_marlin_cuda_inference.gptq_marlin_repack(  # pylint: disable=E0401
                    m.qweight,
                    new_module.g_idx_sort_indices,
                    m.infeatures,
                    m.outfeatures,
                    m.bits
                )
                new_module.qweight.resize_(marlin_qweight.shape)
                new_module.qweight = nn.Parameter(marlin_qweight, requires_grad=False)

                # Permute scales for the new module's configuration.
                marlin_scales = marlin_permute_scales(
                    m.scales,
                    size_k=m.infeatures,
                    size_n=m.outfeatures,
                    group_size=m.group_size
                )

                new_module.scales.resize_(marlin_scales.shape)
                new_module.scales = nn.Parameter(marlin_scales, requires_grad=False)

                # Create a workspace for the new module.
                new_module.workspace = marlin_make_workspace(  # TODO: Consider moving this to post-init.
                    new_module.outfeatures, device
                )

                # Replace the original module in the model with the new Marlin module.
                set_module(model, n, new_module)

                # Clear cache and perform garbage collection to free memory.
                torch.cuda.empty_cache()
                gc.collect()

    def post_init_model(self, model):
        """Post-initialization that require device information, for example buffers initialization on device.

        Args:
            model (`nn.Module`):
                The input model
        """

        class StoreAttr(object):
            pass

        model.quantize_config = StoreAttr()
        if self.need_marlin_repacking:
            require_version("gptqmodel",
                            "marlin format requires gptqmodel to be installed, "
                            "`pip install -v gptqmodel --no-build-isolation `")
            self.repack_marlin(model)
        from auto_round_extension.cuda.post_init import autoround_post_init
        model = autoround_post_init(model)
        # there are no side-effects after call qbits_post_init when model quant-type not equal to qbits.
        if self.target_device == "cpu":
            model = self.cpu_post_init(model)

        return model

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        if model.__class__.main_input_name != "input_ids":
            logger.warning("We can only quantize pure text models and " \
                           "certain types(Llava/Qwen-VL/Phi-3-vision) of multimodal models.")

        if self.pre_quantized:
            model = self.convert_model(model)

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        if self.pre_quantized:
            model = self.post_init_model(model)
        else:
            raise NotImplementedError

    @property
    def is_trainable(self, model: Optional["PreTrainedModel"] = None):
        return True

    @property
    def is_serializable(self):
        return True


import transformers

if version.parse(transformers.__version__) < version.parse("4.38.0"):
    logger.error("Please upgrade transformers>=4.38.0 to support lm-head quantization")

transformers.quantizers.auto.AutoHfQuantizer = AutoHfQuantizer
transformers.modeling_utils.AutoHfQuantizer = AutoHfQuantizer
