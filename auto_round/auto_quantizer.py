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

from auto_round.utils import get_module, set_module, dynamic_import_inference_linear
import auto_round_extension.qbits.qlinear_qbits as qlinear_qbits
from enum import Enum
import copy

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


def is_autoround_exllamav2_available():
    res = True
    try:
        from autoround_exllamav2_kernels import gemm_half_q_half, make_q_matrix
    except ImportError as e:
        res = False
    return res


def is_hpu_supported():  # pragma: no cover
    try:
        import subprocess
        import habana_frameworks.torch.core as htcore  # pylint: disable=E0401
        hqt_version = subprocess.check_output(['pip', 'show', \
                                               'habana_quantization_toolkit']).decode().split('\n')[1].split(': ')[1]
        assert (hqt_version >= "1.17")
    except ImportError as e:
        return False
    return True


if is_auto_round_available():
    from auto_round_extension.cuda.post_init import autoround_post_init


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
        if "auto" == backend:
            if torch.cuda.is_available():
                backend = "auto_round:exllamav2"
            elif is_hpu_supported():
                backend = "hpu"
            else:
                backend = "cpu"
        elif "cuda" == backend:
            backend = "auto_round:exllamav2"
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
        ##TODO add more check

    def get_loading_attributes(self):
        attributes_dict = copy.deepcopy(self.__dict__)
        loading_attributes = ["backend"]
        loading_attibutes_dict = {i: j for i, j in attributes_dict.items() if i in loading_attributes}
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
        self.exllama2_available = is_autoround_exllamav2_available()

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

    def convert_model(self, model: nn.Module):
        """Convert the model to an AutoRound model by getting and replacing the layers.

        Args:
            model (`nn.Module`):
                Model to be converted
        """
        from auto_round.utils import get_layer_names_in_block

        quantization_config = model.config.quantization_config
        if hasattr(quantization_config, "backend"):  # pragma: no cover
            backend = quantization_config.backend
            if "hpu" in backend and model.dtype != torch.bfloat16:
                logger.info("change the dtype to `bfloat16` as HPU does not support float16")
                model = model.to(torch.bfloat16)
        bits = quantization_config.bits
        group_size = quantization_config.group_size
        data_type = quantization_config.data_type if hasattr(quantization_config, "data_type") \
            else "int"  # pragma: no cover
        sym = quantization_config.sym
        quant_block_list = quantization_config.quant_block_list \
                           if hasattr(quantization_config, "quant_block_list") else None
        layer_names = get_layer_names_in_block(model, quant_block_list=quant_block_list)
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

        self._replace_by_quant_layers(model, layer_configs, backend)
        return model

    def _replace_by_quant_layers(self, module: nn.Module, layer_configs, backend):
        """Replaces linear layers in `module` by `QuantLinear`

        Args:
            module (`nn.Module`):
                Module to quantize
            names (`List[str]`):
                List of names of the module to quantize
            name (`str`, defaults to `""`):
                To keep track of the name of the current module
        """
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
            device = get_device(layer)
            QuantLinear = dynamic_import_inference_linear(backend, bits, group_size, sym)
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                out_features = layer.out_features
            elif isinstance(layer, nn.Conv2d):  ##not supported now
                in_features = layer.in_channels
                out_features = layer.out_channels
            elif isinstance(layer, Conv1D):  ##TODO need to have a check
                in_features = layer.weight.shape[0]
                out_features = layer.weight.shape[1]
            bias = layer.bias is not None
            try:
                new_layer = QuantLinear(  # pylint: disable=E1123
                    bits,
                    group_size,
                    in_features,
                    out_features,
                    bias,
                    weight_dtype=layer.weight.dtype,
                    clip = clip
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

            new_layer.device = device
            set_module(module, layer_name, new_layer)

    def qbits_post_init(self, model):
        dep_check = True
        for layer in model.modules():
            if isinstance(layer, qlinear_qbits.QuantLinear):
                if dep_check:
                    layer.req_check()
                layer.post_init()
                dep_check = False
        return model

    def post_init_model(self, model):
        """Post-initialization that require device information, for example buffers initialization on device.

        Args:
            model (`nn.Module`):
                The input model
        """

        class StoreAttr(object):
            pass

        model.quantize_config = StoreAttr()
        model = autoround_post_init(model)
        # there are no side-effects after call qbits_post_init when model quant-type not equal to qbits. 
        model = self.qbits_post_init(model)

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

transformers_version = [int(item) for item in transformers.__version__.split('.')[:2]]
if transformers_version[0] == 4 and transformers_version[1] < 38:
    logger.error("Please upgrade transformers>=4.38.0 to support lm-head quantization")

transformers.quantizers.auto.AutoHfQuantizer = AutoHfQuantizer
transformers.modeling_utils.AutoHfQuantizer = AutoHfQuantizer


