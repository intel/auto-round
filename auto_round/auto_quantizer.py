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
import transformers
from transformers.quantizers import AutoQuantizationConfig, HfQuantizer
from transformers.quantizers.auto import AUTO_QUANTIZER_MAPPING
from transformers.utils.quantization_config import AwqConfig, GPTQConfig, QuantizationConfigMixin, QuantizationMethod

from auto_round.utils import get_module, set_module

logger = getLogger(__name__)
import sys

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

AUTOGPTQ_MINIMUM_VERSION = version.parse("0.4.99")  # Allows 0.5.0.dev0


def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
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


_auto_gptq_available = _is_package_available("auto_gptq")


def is_auto_gptq_available():
    if _auto_gptq_available:
        version_autogptq = version.parse(importlib_metadata.version("auto_gptq"))
        if AUTOGPTQ_MINIMUM_VERSION < version_autogptq:
            return True
        else:
            raise ImportError(
                f"Found an incompatible version of auto-gptq. Found version {version_autogptq},"
                f" but only version above {AUTOGPTQ_MINIMUM_VERSION} are supported"
            )


if is_auto_gptq_available():
    from auto_gptq import exllama_set_max_input_length
    from auto_gptq.modeling._utils import autogptq_post_init
    from auto_gptq.quantization import GPTQ
    from auto_gptq.utils.import_utils import dynamically_import_QuantLinear


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
        if "auto-round" in quant_method:
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

        if isinstance(quantization_config, (GPTQConfig, AwqConfig)) and quantization_config_from_args is not None:
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
            bits: int,
            tokenizer: Any = None,
            dataset: str = None,
            group_size: int = 128,
            sym: bool = False,
            backend="gptq:exllamav2",
            iters: int = 200,
            weight_config: dict = None,
            enable_quanted_input=True,
            enable_minmax_tuning=True,
            lr=None,
            minmax_lr=None,
            n_samples=512,
            seqlen=2048,
            **kwargs,
    ):
        self.bits = bits
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.group_size = group_size
        self.sym = sym
        self.backend = backend
        self.inters = iters
        self.weight_config = weight_config
        self.enable_quanted_input = enable_quanted_input
        self.enable_minmax_tuning = enable_minmax_tuning
        self.lr = lr
        self.minmax_lr = minmax_lr
        self.n_samples = n_samples
        self.seqlen = seqlen
        if kwargs is not None:
            for key in kwargs.keys():
                setattr(self, key, kwargs[key])

        self.post_init()

    def get_loading_attributes(self):
        pass
        # attibutes_dict = copy.deepcopy(self.__dict__)
        # loading_attibutes = ["disable_exllama", "use_exllama", "exllama_config", "use_cuda_fp16", "max_input_length"]
        # loading_attibutes_dict = {i: j for i, j in attibutes_dict.items() if i in loading_attibutes}
        # return loading_attibutes_dict

    def post_init(self):
        r"""Safety checker that arguments are correct."""
        if self.bits not in [2, 3, 4, 8]:
            raise ValueError(f"Only support quantization to [2,3,4,8] bits but found {self.bits}")
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("group_size must be greater than 0 or equal to -1")
        ##TODO add more check

    def to_dict(self):
        config_dict = super().to_dict()
        config_dict.pop("disable_exllama", None)
        return config_dict


class AutoRoundQuantizer(HfQuantizer):
    """Quantizer of the Autoround method, currently only gptq backend has been supported."""

    requires_calibration = False
    required_packages = ["auto_gptq"]
    optimum_quantizer = None

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        gptq_supports_cpu = version.parse(importlib.metadata.version("auto-gptq")) > version.parse("0.4.2")
        if not gptq_supports_cpu and not torch.cuda.is_available():
            raise RuntimeError("GPU is required to quantize or run quantize model.")
        elif not is_auto_gptq_available():
            raise ImportError("Loading a GPTQ quantized model requires auto-gptq library (`pip install auto-gptq`)")
        elif version.parse(importlib.metadata.version("auto_gptq")) < version.parse("0.4.2"):
            raise ImportError("You need a version of auto_gptq >= 0.4.2 to use GPTQ: `pip install --upgrade auto-gptq`")

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            torch_dtype = torch.float16
        elif torch_dtype != torch.float16:
            logger.info("We suggest you to set `torch_dtype=torch.float16` for better efficiency with AutoRound.")
        return torch_dtype

    def convert_model(self, model: nn.Module):
        """Convert the model to a GPTQ model by getting and replacing the layers.

        Args:
            model (`nn.Module`):
                Model to be converted
        """
        from auto_round.utils import get_layer_names_in_block

        layer_names = get_layer_names_in_block(model)
        quantization_config = model.config.quantization_config
        bits = quantization_config.bits
        group_size = quantization_config.group_size
        data_type = quantization_config.data_type
        sym = quantization_config.sym
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
            else:
                layer_configs[layer_name]["bits"] = extra_config.get("bits", bits)
                layer_configs[layer_name]["group_size"] = extra_config.get("group_size", group_size)
                layer_configs[layer_name]["data_type"] = extra_config.get("data_type", data_type)
                layer_configs[layer_name]["sym"] = extra_config.get("sym", sym)
        backend = quantization_config.backend

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
            if not (bits <= 8 and data_type == "int"):
                continue
            from auto_round.export.export_to_autoround.export_to_autoround import get_autogptq_backend_config

            use_triton, disable_exllama, disable_exllamav2, use_qigen, disable_marlin = get_autogptq_backend_config(
                backend, bits
            )
            QuantLinear = dynamically_import_QuantLinear(
                use_triton=False,
                desc_act=False,
                group_size=group_size,
                bits=bits,
                disable_exllama=True,
                disable_exllamav2=False,
                use_qigen=use_qigen,
                disable_marlin=disable_marlin,
            )
            layer = get_module(module, layer_name)
            device = get_device(layer)
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

    def post_init_model(self, model):
        """Post-initialization that require device information, for example buffers initialization on device.

        Args:
            model (`nn.Module`):
                The input model
        """

        # if self.bits == 4 and not self.disable_exllama:
        #     if get_device(model) == torch.device("cpu") or (
        #             hasattr(model, "hf_device_map") and any(d in model.hf_device_map for d in ["cpu", "disk"])
        #     ):
        #         raise ValueError(
        #             "Found modules on cpu/disk. Using Exllama
        #             or Exllamav2 backend requires all the modules to be on GPU."
        #             "You can deactivate exllama backend by
        #             setting `disable_exllama=True` in the quantization config object"
        #         )

        class StoreAttr(object):
            pass

        model.quantize_config = StoreAttr()
        model.quantize_config.desc_act = False
        model = autogptq_post_init(model, use_act_order=False)
        return model

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        if model.__class__.main_input_name != "input_ids":
            raise RuntimeError("We can only quantize pure text model.")

        if self.pre_quantized:
            model = self.convert_model(model)

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        if self.pre_quantized:
            model = self.post_init_model(model)
        else:
            raise NotImplementedError
            # if self.quantization_config.tokenizer is None:
            #     self.quantization_config.tokenizer = model.name_or_path
            #
            # self.optimum_quantizer.quantize_model(model, self.quantization_config.tokenizer)
            # model.config.quantization_config = GPTQConfig.from_dict(self.optimum_quantizer.to_dict())

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
from transformers import AutoModelForCausalLM as AutoModelForCausalLM
