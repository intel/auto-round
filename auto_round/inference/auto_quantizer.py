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
from enum import Enum
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from packaging import version
from transformers.modeling_utils import PreTrainedModel
from transformers.quantizers import AutoQuantizationConfig, HfQuantizer
from transformers.quantizers.auto import AUTO_QUANTIZER_MAPPING
from transformers.utils.quantization_config import AwqConfig, GPTQConfig, QuantizationConfigMixin, QuantizationMethod

from auto_round.inference.convert_model import convert_hf_model, infer_target_device, post_init
from auto_round.utils import is_hpex_available

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
        if "auto-round" in quant_method or is_hpex_available():  # pragma: no cover
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
        if quantization_config_from_args is None or not hasattr(
            quantization_config_from_args, "get_loading_attributes"
        ):
            # If the quantization_config_from_args is None or does not have get_loading_attributes method,
            # we will not use it to load the model.
            quantization_config_from_args = None
        else:
            loading_attr_dict = quantization_config_from_args.get_loading_attributes()

        if isinstance(quantization_config, dict):
            if (
                "auto-round" in quantization_config["quant_method"]
                or quantization_config_from_args.__class__.__name__ == "AutoRoundConfig"
            ):
                quantization_config = AutoRoundConfig.from_dict(quantization_config)
            else:
                quantization_config = AutoQuantizationConfig.from_dict(quantization_config)  # pylint: disable=E1101

        if (
            isinstance(quantization_config, (GPTQConfig, AwqConfig, AutoRoundConfig))
            and quantization_config_from_args is not None
        ):
            # special case for GPTQ / AWQ config collision

            for attr, val in loading_attr_dict.items():
                setattr(quantization_config, attr, val)
            warning_msg += (
                f"However, loading attributes (e.g. {list(loading_attr_dict.keys())}) "
                f"will be overwritten with the one you passed to `from_pretrained`. The rest will be ignored."
            )

        if warning_msg != "":
            warnings.warn(warning_msg)

        return quantization_config

    @staticmethod
    def supports_quant_method(quantization_config_dict):
        from transformers.quantizers.auto import AUTO_QUANTIZATION_CONFIG_MAPPING

        AUTO_QUANTIZATION_CONFIG_MAPPING["auto-round"] = AutoRoundConfig
        AUTO_QUANTIZATION_CONFIG_MAPPING["auto_round"] = AutoRoundConfig
        quant_method = quantization_config_dict.get("quant_method", None)
        if quantization_config_dict.get("load_in_8bit", False) or quantization_config_dict.get("load_in_4bit", False):
            suffix = "_4bit" if quantization_config_dict.get("load_in_4bit", False) else "_8bit"
            quant_method = QuantizationMethod.BITS_AND_BYTES + suffix
        elif quant_method is None:
            raise ValueError(
                "The model's quantization config from the arguments has no `quant_method` attribute."
                "Make sure that the model has been correctly quantized"
            )

        if quant_method not in AUTO_QUANTIZATION_CONFIG_MAPPING.keys():
            logger.warning(
                f"Unknown quantization type, got {quant_method} - supported types are:"
                f" {list(AUTO_QUANTIZER_MAPPING.keys())}. Hence, we will skip the quantization. "
                "To remove the warning, you can delete the quantization_config attribute in config.json"
            )
            return False
        return True


class AutoRoundQuantizationMethod(str, Enum):
    AutoRound = "auto-round"


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
        self.packing_format = "auto_round:auto_gptq"
        self.backend = backend
        self.layer_config = layer_config
        if kwargs is not None:
            for key in kwargs.keys():
                setattr(self, key, kwargs[key])
        self.quant_method = AutoRoundQuantizationMethod.AutoRound
        self.post_init()

    def post_init(self):
        r"""Safety checker that arguments are correct."""
        if self.bits not in [2, 3, 4, 8]:
            raise ValueError(f"Only support quantization to [2,3,4,8] bits but found {self.bits}")
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("group_size must be greater than 0 or equal to -1")

    def get_loading_attributes(self):
        loading_attibutes_dict = {"backend": self.backend}
        return loading_attibutes_dict

    def to_dict(self):
        config_dict = super().to_dict()
        return config_dict

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
        quant_method = config_dict["quant_method"]
        if "auto-round" not in quant_method and "gptq" not in quant_method and "awq" not in quant_method:
            raise NotImplementedError(
                "Failed to convert to auto_round format. Only `gptqv1`, `awq`, and `auto-round` formats are supported."
            )

        if "gptq" in quant_method and "meta" in config_dict:
            raise NotImplementedError("Failed to convert gptq format to auto_round format. Only supports `gptqv1`")

        if "awq" in quant_method and config_dict.get("version", "gemm") != "gemm":
            raise NotImplementedError(
                "Failed to convert awq format to auto_round format. Only supports  awq format with gemm version"
            )

        if "auto-round" not in quant_method:
            config_dict["packing_format"] = f"auto_round:{quant_method}"

        return super().from_dict(config_dict, return_unused_kwargs=return_unused_kwargs, **kwargs)


class AutoRoundQuantizer(HfQuantizer):
    """Quantizer of the AutoRound method, currently only triton and exllamav2 backend has been supported."""

    requires_calibration = False
    required_packages = ["auto_round"]
    optimum_quantizer = None

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        self.device_map = None
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        self.device_map = kwargs.get("device_map", None)
        if not is_auto_round_available():
            raise ImportError(
                "Loading a AutoRound quantized model requires auto-round library (`pip install " "auto-round`)"
            )
        else:
            try:
                import auto_round

                autoround_version = version.parse(auto_round.__version__)
            except:
                autoround_version = version.parse(importlib.metadata.version("auto_round"))
            if autoround_version < version.parse("0.2.0"):
                raise ImportError(
                    "You need a version of auto_round > 0.2.0 to use AutoRound: `pip install --upgrade "
                    "auto-round` or install from source"
                )

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            torch_dtype = torch.bfloat16
        return torch_dtype

    def post_init_model(self, model):
        """Post-initialization that require device information, for example buffers initialization on device.

        Args:
            model (`nn.Module`):
                The input model
        """

        class StoreAttr(object):
            pass

        model.quantize_config = StoreAttr()

        post_init(model, self.used_backends)

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        if self.pre_quantized:
            target_device = infer_target_device(self.device_map)
            model, used_backends = convert_hf_model(model, target_device)
            self.used_backends = used_backends

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        if self.pre_quantized:
            self.post_init_model(model)
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
