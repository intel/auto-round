import importlib.util
import warnings
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from packaging import version
from transformers.pytorch_utils import Conv1D
from transformers.quantizers import AutoQuantizationConfig, HfQuantizer
from transformers.quantizers.auto import AUTO_QUANTIZER_MAPPING
from transformers.utils.quantization_config import AwqConfig, GPTQConfig, QuantizationConfigMixin, QuantizationMethod

logger = getLogger(__name__)
import sys

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

AUTOGPTQ_MINIMUM_VERSION = version.parse("0.4.99")  # Allows 0.5.0.dev0
from enum import Enum


class ExllamaVersion(int, Enum):
    ONE = 1
    TWO = 2


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
                f"Found an incompatible version of auto-gptq. Found version {version_autogptq}, but only version above {AUTOGPTQ_MINIMUM_VERSION} are supported"
            )


if is_auto_gptq_available():
    from auto_gptq import exllama_set_max_input_length
    from auto_gptq.modeling._utils import autogptq_post_init
    from auto_gptq.quantization import GPTQ
    from auto_gptq.utils.import_utils import dynamically_import_QuantLinear

SEQLEN_KEYS_TRANFORMERS = ["max_position_embeddings", "seq_length", "n_positions"]
BLOCK_PATTERNS = [
    "transformer.h",
    "model.decoder.layers",
    "gpt_neox.layers",
    "model.layers",
]


def get_device(obj: Union[torch.Tensor, nn.Module]):
    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device


def get_layers(module: nn.Module, layers=[Conv1D, nn.Conv2d, nn.Linear], prefix: Optional[str] = None, name: str = ""):
    """
    Get all the layers with a specific prefix in the module
    Args:
        module (`nn.Module`):
            The module that contains our layers
        layers (`list`, defaults to `[Conv1D, nn.Conv2d, nn.Linear]`):
            Type of the layers that we want to get
        prefix (`Optional[str]`, defaults to `None`):
            Prefix of layers
        name (`str`, defaults to `""`):
            Used for recursion. Don't modify

    Returns:
        `Dict[str,Union[Conv1D, nn.Conv2d, nn.Linear]]`: Mapping of the name of the layer and the actual layer
    """
    for layer in layers:
        if isinstance(module, layer):
            if prefix is not None:
                if name.startswith(prefix):
                    return {name: module}
            else:
                return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(get_layers(child, layers=layers, prefix=prefix, name=name + "." + name1 if name != "" else name1))
    return res


def get_block_name_with_pattern(model: nn.Module):
    """Get the name of the module that contains the transformers blocks by checking if any modules has a specific pattern.

    Args:
        model (`nn.Module`):
        The input model
    Returns:
        `str`: The name of the module that contains the Transformer blocks.
    """
    modules_names = [n for n, _ in model.named_modules()]
    for pattern_candidate in BLOCK_PATTERNS:
        pattern_candidate = pattern_candidate
        if any(pattern_candidate in name for name in modules_names):
            return pattern_candidate
    raise ValueError("Block pattern could not be match. Pass `block_name_to_quantize` argument in `quantize_model`")


class AutoHfQuantizer:
    """The Auto-HF quantizer class that takes care of automatically instantiating to the correct
    `HfQuantizer` given the `QuantizationConfig`."""

    @classmethod
    def from_config(cls, quantization_config: Union[QuantizationConfigMixin, Dict], **kwargs):
        # Convert it to a QuantizationConfig if the q_config is a dict
        if isinstance(quantization_config, dict):
            quantization_config = AutoQuantizationConfig.from_dict(quantization_config)
        print("AUTOROUND INTERFACE", flush=True)
        quant_method = quantization_config.quant_method

        # Again, we need a special care for bnb as we have a single quantization config
        # class for both 4-bit and 8-bit quantization
        if quant_method == QuantizationMethod.BITS_AND_BYTES:
            if quantization_config.load_in_8bit:
                quant_method += "_8bit"
            else:
                quant_method += "_4bit"

        if quant_method not in AUTO_QUANTIZER_MAPPING.keys():
            raise ValueError(
                f"Unknown quantization type, got {quant_method} - supported types are:"
                f" {list(AUTO_QUANTIZER_MAPPING.keys())}"
            )

        ##target_cls = AUTO_QUANTIZER_MAPPING[quant_method]
        target_cls = AutoRoundQuantizer  ##changed by wenhauch
        return target_cls(quantization_config, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        print("AutoRound interface", flush=True)
        quantization_config = AutoQuantizationConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls.from_config(quantization_config)

    @classmethod
    def merge_quantization_configs(
        cls,
        quantization_config: Union[dict, QuantizationConfigMixin],
        quantization_config_from_args: Optional[QuantizationConfigMixin],
    ):
        """Handles situations where both quantization_config from args and quantization_config from model config are present."""
        if quantization_config_from_args is not None:
            warning_msg = (
                "You passed `quantization_config` or equivalent parameters to `from_pretrained` but the model you're loading"
                " already has a `quantization_config` attribute. The `quantization_config` from the model will be used."
            )
        else:
            warning_msg = ""

        if isinstance(quantization_config, dict):
            quantization_config = AutoQuantizationConfig.from_dict(quantization_config)

        if isinstance(quantization_config, (GPTQConfig, AwqConfig)) and quantization_config_from_args is not None:
            # special case for GPTQ / AWQ config collision
            loading_attr_dict = quantization_config_from_args.get_loading_attributes()
            for attr, val in loading_attr_dict.items():
                setattr(quantization_config, attr, val)
            warning_msg += f"However, loading attributes (e.g. {list(loading_attr_dict.keys())}) will be overwritten with the one you passed to `from_pretrained`. The rest will be ignored."

        if warning_msg != "":
            warnings.warn(warning_msg)

        return quantization_config


@dataclass
class AutoRoundConfig(QuantizationConfigMixin):
    """This is a wrapper class about all possible attributes and features that you can play with a model that has been
    loaded using `optimum` api for gptq quantization relying on auto_gptq backend.

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
        sym: bool = True,
        backend="gptq:triton",
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
        self.quant_method = QuantizationMethod.GPTQ
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
        self.kwargs = kwargs

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
    """
    Quantizer of the GPTQ method - for GPTQ the quantizer support calibration of the model through
    `auto_gptq` package. Quantization is done under the hood for users if they load a non-prequantized model.
    """

    requires_calibration = False
    required_packages = ["optimum", "auto_gptq"]  ##TODO change
    optimum_quantizer = None

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)
        # from optimum.gptq import GPTQQuantizer
        #
        # self.optimum_quantizer = GPTQQuantizer.from_dict(self.quantization_config.to_dict_optimum())

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
            logger.info("We suggest you to set `torch_dtype=torch.float16` for better efficiency with GPTQ.")
        return torch_dtype

    def convert_model(self, model: nn.Module):
        """Convert the model to a GPTQ model by getting and replacing the layers.

        Args:
            model (`nn.Module`):
                Model to be converted
        """

        self.block_name_to_quantize = get_block_name_with_pattern(model)
        block_name = self.block_name_to_quantize
        layers_to_be_replaced = get_layers(model, prefix=block_name)
        layers_to_be_replaced["lm_head"] = model.lm_head ##TODO hard coded
        self.modules_in_block_to_quantize = None
        if self.modules_in_block_to_quantize is not None:
            layers_to_keep = sum(self.modules_in_block_to_quantize, [])
            for name in list(layers_to_be_replaced.keys()):
                if not any(name.endswith(layer) for layer in layers_to_keep):
                    logger.info(
                        f"Quantization disabled for {name} (only modules_in_block_to_quantize={self.modules_in_block_to_quantize} are quantized)"
                    )
                    del layers_to_be_replaced[name]
        self._replace_by_quant_layers(model, layers_to_be_replaced)
        return model

    def _replace_by_quant_layers(self, module: nn.Module, names: List[str], name: str = ""):
        """Replaces linear layers in `module` by `QuantLinear`

        Args:
            module (`nn.Module`):
                Module to quantize
            names (`List[str]`):
                List of names of the module to quantize
            name (`str`, defaults to `""`):
                To keep track of the name of the current module
        """
        self.bits = 4  ##TODO CHANGE
        self.group_size = 128  ##TODO change
        self.use_cuda_fp16 = True
        self.disable_exllama = True
        QuantLinear = dynamically_import_QuantLinear(
            use_triton=False,
            desc_act=False,
            group_size=self.group_size,
            bits=self.bits,
            disable_exllama=True,
            disable_exllamav2=False,
        )
        if isinstance(module, QuantLinear):
            return
        for attr in dir(module):
            layer = getattr(module, attr)
            name1 = name + "." + attr if name != "" else attr
            if name1 in names:
                device = get_device(layer)
                delattr(module, attr)
                if isinstance(layer, nn.Linear):
                    in_features = layer.in_features
                    out_features = layer.out_features
                elif isinstance(layer, nn.Conv2d):
                    in_features = layer.in_channels
                    out_features = layer.out_channels
                elif isinstance(layer, Conv1D):
                    in_features = layer.weight.shape[0]
                    out_features = layer.weight.shape[1]
                # bias = layer.bias is not None and torch.any(layer.bias)
                bias = True ## autogptq always set bias to True

                new_layer = QuantLinear(
                    self.bits,
                    self.group_size,
                    in_features,
                    out_features,
                    bias,
                    use_cuda_fp16=self.use_cuda_fp16,
                    weight_dtype=layer.weight.dtype,
                )

                new_layer.device = device
                setattr(module, attr, new_layer.to(device))
        for name1, child in module.named_children():
            self._replace_by_quant_layers(child, names, name + "." + name1 if name != "" else name1)

    def post_init_model(self, model):
        """Post-initialization that require device information, for example buffers initialization on device.

        Args:
            model (`nn.Module`):
                The input model
        """
        if self.bits == 4 and not self.disable_exllama:
            if get_device(model) == torch.device("cpu") or (
                hasattr(model, "hf_device_map") and any(d in model.hf_device_map for d in ["cpu", "disk"])
            ):
                raise ValueError(
                    "Found modules on cpu/disk. Using Exllama or Exllamav2 backend requires all the modules to be on GPU."
                    "You can deactivate exllama backend by setting `disable_exllama=True` in the quantization config object"
                )

        class StoreAttr(object):
            pass

        model.quantize_config = StoreAttr()
        model.quantize_config.desc_act = False
        model = autogptq_post_init(model, use_act_order=False)
        # if (
        #         False
        #         and (not self.disable_exllama and self.exllama_version == ExllamaVersion.ONE)
        #         and self.max_input_length is not None
        # ):
        #     model = exllama_set_max_input_length(model, self.max_input_length)
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
            if self.quantization_config.tokenizer is None:
                self.quantization_config.tokenizer = model.name_or_path

            self.optimum_quantizer.quantize_model(model, self.quantization_config.tokenizer)
            model.config.quantization_config = GPTQConfig.from_dict(self.optimum_quantizer.to_dict())

    @property
    def is_trainable(self, model: Optional["PreTrainedModel"] = None):
        return True

    @property
    def is_serializable(self):
        return True
