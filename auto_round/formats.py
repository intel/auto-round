# Copyright (c) 2025 Intel Corporation
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

from __future__ import annotations

import copy
import os
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict
from enum import Enum
from typing import TYPE_CHECKING, Callable, Union

import torch
import transformers

from auto_round.compressors.utils import (
    is_mx_fp,
    is_nv_fp,
    is_standard_fp,
    is_static_wfp8afp8,
    is_wfp8afp8,
)
from auto_round.export.export_to_gguf.config import ModelType
from auto_round.schemes import (
    PRESET_SCHEMES,
    QuantizationScheme,
    get_gguf_scheme,
)
from auto_round.utils import (
    SUPPORTED_FORMATS,
    check_to_quantized,
    copy_python_files_from_model_cache,
    get_module,
    logger,
)


class AutoRoundExportFormat(str, Enum):
    # Weight: FP8, per-channel, may be extended to per-tensor in future
    # Activation: FP8, per-tensor
    FP8_STATIC = "fp8_static"
    MXFP8 = "mxfp8"
    MXFP4 = "mxfp4"
    NVFP4 = "nvfp4"
    FP8 = "fp8"
    MX_FP = "mx_fp"
    NV_FP = "nv_fp"
    MX_FP_RCEIL = "mx_fp_rceil"
    NV_FP4_WITH_STATIC_GS = "nv_fp4_with_static_gs"


if TYPE_CHECKING:
    from auto_round.compressors.base import BaseCompressor


def _check_compatibility(formats: list[str], ar: BaseCompressor):
    if (
        any(["gguf" in f.lower() for f in formats])
        and len([f for f in formats if f.lower() != "fake" and not f.lower().startswith("gguf")]) > 1
    ):
        raise ValueError(
            f"GGUF format is not compatible with other formats, but got {formats}, please choose only one of them"
        )
    gguf_format_name = get_gguf_scheme(ar.scheme)
    if gguf_format_name:
        if gguf_format_name.lower().endswith("mixed"):
            gguf_format_name = gguf_format_name.lower().replace("_mixed", "_s")
        if any([f.lower() not in ["fake", gguf_format_name.lower()] for f in formats]):
            tmp_format_name = gguf_format_name.lower() if "fake" not in formats else f"{gguf_format_name.lower()},fake"
            logger.warning(
                f"reset format {','.join(formats)} to {tmp_format_name} "
                f"since scheme {gguf_format_name} can only be exported to format {gguf_format_name.lower()} or fake"
            )
            formats = tmp_format_name.split(",")
    return formats


def get_formats(
    format: str,
    ar: BaseCompressor,
) -> list[OutputFormat]:
    """Get the list of OutputFormat instances based on the provided name."""

    def remove_duplicates(lst):
        seen = set()
        return [x for x in lst if not (x in seen or seen.add(x))]

    formats = format.lower().replace("q*_", f"q{ar.bits}_").replace(" ", "").split(",")
    formats = remove_duplicates(formats)  # need the keep origin order

    formats = _check_compatibility(formats, ar)

    formats = remove_duplicates(formats)

    for i in range(len(formats)):
        if formats[i].startswith("gguf:"):
            formats[i] = GGUFFormat(formats[i], ar)
        elif formats[i] not in OutputFormat._format_list:
            raise KeyError(f"Unsupported format {formats[i]}, please choose from {SUPPORTED_FORMATS}")
        else:
            formats[i] = OutputFormat._format_list[formats[i]](formats[i], ar)

        new_format = formats[i].check_and_reset_format(ar)
        if new_format is not None:
            if new_format not in format:
                formats[i] = OutputFormat._format_list[new_format](new_format, ar)
            else:
                formats[i] = None

    formats = [fmt for fmt in formats if fmt is not None]

    if len(formats) == 1 and formats[0].is_gguf() and ar.scale_dtype != torch.float32:
        ar.scale_dtype = torch.float32
        logger.info("change `scale_dtype` to `torch.float32` for gguf format")

    return formats


def _check_divisible_by_32(ar):
    from auto_round.schemes import preset_name_to_scheme

    if isinstance(ar.scheme, str):
        default_dict = asdict(preset_name_to_scheme(ar.scheme.upper()))
    else:
        default_dict = asdict(ar.scheme)
    if default_dict["data_type"] == "int" and default_dict["act_bits"] >= 16:
        for n, m in ar.model.named_modules():
            if (
                type(m) in ar.supported_types
                or m.__class__.__name__ in ar.inner_supported_types
                and check_to_quantized(m)
            ):
                if hasattr(m, "weight") and m.weight is not None and (m.weight.shape[0] % 32 or m.weight.shape[1] % 32):
                    if ar.layer_config is None:
                        ar.layer_config = {}
                    if ar.layer_config.get(n) is not None and ar.layer_config[n]["bits"] >= 16:
                        continue
                    ar.layer_config.setdefault(n, copy.deepcopy(default_dict))
                    ar.layer_config[n].update({"bits": 16, "data_type": "fp", "fixed_by_user": True})
                    logger.warning_once(f"{n} skipped quantization (shape not divisible by 32).")


class OutputFormat(ABC):
    """ "Base class for different output formats.

    format: determines which method from export module to use for exporting.
            For example, auto_round, gguf, llm_compressor etc.
    backend: determines the specific export process within the format.
            For example, auto_round:fp8_static, auto_round:auto_awq etc.
    """

    support_schemes: list = []
    _format_list: dict[str, OutputFormat] = {}
    format_name = "base"

    def __init__(self, format: str, ar: BaseCompressor):
        """Initialize the OutputFormat class."""
        self.output_format = format
        self.backend = None

        if not self.is_fake() and not self.is_support_scheme(ar.scheme):
            logger.error(
                f"Currently, the {self.format_name} format only supports {self.support_schemes}, "
                f"but got scheme {ar.scheme}, please change to fake or auto_round etc."
            )
            exit(-1)

    @classmethod
    def register(cls, *names: str) -> Callable[[OutputFormat], OutputFormat]:
        assert names

        def func(output_format: OutputFormat) -> OutputFormat:
            for name in names:
                cls._format_list[name] = output_format
            return output_format

        return func

    @classmethod
    def get_support_matrix(cls: OutputFormat) -> str:
        output_str = ""
        for k, v in sorted(cls._format_list.items()):
            if k == "fake":
                support_schemes = "All schemes"
            else:
                if ":" in k and k.split(":")[1] in cls._format_list:
                    support_schemes = cls._format_list[k.split(":")[1]].support_schemes
                else:
                    support_schemes = v.support_schemes
                support_schemes = ", ".join(support_schemes).rstrip(",")
            output_str += f"\x1b[31;1m{k}\x1b[0m support scheme:\n\t{support_schemes}\n"
        return output_str

    def get_backend_name(self) -> str:
        if self.backend is None:
            return self.output_format

        # auto_round:llm_compressor:fp8_static
        if self.backend.backend is not None:
            return f"{self.output_format}:{self.backend.get_backend_name()}"
        # auto_round:fp8_static, llm_compressor:fp8_static, auto_round:auto_awq
        else:
            return self.backend.get_backend_name()

    @classmethod
    def is_support_scheme(cls: OutputFormat, scheme: Union[str, QuantizationScheme]) -> bool:
        if isinstance(scheme, str) and scheme.upper() in cls.support_schemes:
            return True
        if isinstance(scheme, QuantizationScheme):
            return True
        return False

    def check_and_reset_format(self, ar: BaseCompressor) -> str:
        if self.backend is not None:
            new_format = self.backend.check_and_reset_format(ar)
            self.backend = OutputFormat._format_list[new_format](new_format, ar) if new_format else self.backend

        w_fp8 = ar.data_type.startswith("fp") and ar.bits == 8
        act_fp8 = ar.act_data_type.startswith("fp") and ar.act_bits == 8
        if w_fp8 or act_fp8:
            error_msg = (
                f"is only supported to export auto_round or llm_compressor format,"
                f" but got {self.format_name}, please check."
            )
            error_msg = ("act_data_type<fp8> " + error_msg) if act_fp8 else error_msg
            error_msg = ("data_type<fp8> " + error_msg) if w_fp8 else error_msg
            logger.error(error_msg)
            sys.exit(-1)

        if ar.act_bits <= 8 and (not is_standard_fp(ar.act_data_type) or ar.act_dynamic):
            logger.warning(
                f"{self.format_name} format not support for current activation quantization configuration,"
                " reset to fake format and save."
            )
            return "fake"

        return None

    @abstractmethod
    def pack_layer(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_quantized(self, *args, **kwargs):
        pass

    def immediate_pack(self, name: str, model: torch.nn.Module, device: torch.device | str, **kwargs):
        m = get_module(model, name)
        if not check_to_quantized(m):
            return

        self.pack_layer(name, model, device=device)

    def is_gguf(self) -> bool:
        return "gguf" in self.output_format

    def is_fake(self) -> bool:
        return self.output_format == "fake"

    def is_gptq(self) -> bool:
        return "gptq" in self.output_format or (self.backend is not None and self.backend.is_gptq())

    def is_awq(self) -> bool:
        return "awq" in self.output_format or (self.backend is not None and self.backend.is_awq())

    def is_llm_compressor(self) -> bool:
        return "llm_compressor" in self.output_format or (self.backend is not None and self.backend.is_llm_compressor())


@OutputFormat.register("fake")
class FakeFormat(OutputFormat):
    support_schemes = None
    format_name = "fake"

    def check_and_reset_format(self, ar: BaseCompressor) -> str:
        return None

    # fake format will not execute pack_layer.
    def pack_layer(self, *args, **kwargs):
        pass

    def save_quantized(
        self,
        output_dir: str,
        model: torch.nn.Module = None,
        tokenizer: Callable = None,
        layer_config: dict = None,
        inplace: bool = True,
        device: Union[str, torch.device] = "cpu",
        serialization_dict: dict = None,
        **kwargs,
    ):
        model = model.to("cpu")
        model.save_pretrained(output_dir)
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(output_dir)
        processor = kwargs.get("processor", None)
        if processor is not None:
            processor.save_pretrained(output_dir)
        try:
            copy_python_files_from_model_cache(model, output_dir)
        except Exception as e:
            logger.warning("Skipping source model Python file copy due to error: %s", e)
        return model


@OutputFormat.register("llm_compressor")
class LLMCompressorFormat(OutputFormat):
    support_schemes = ["MXFP4", "MXFP8", "NVFP4", "FPW8A16", "FP8_STATIC"]
    format_name = "llm_compressor"

    def __init__(self, format, ar):
        if not self.is_support_scheme(ar.scheme):
            logger.error(
                f"Currently, the llm_compressor format only supports {self.support_schemes}, "
                f"but got scheme {ar.scheme}, please change to fake or auto_round etc."
            )
            exit(-1)
        # if format.startswith("llm_compressor"):
        if re.search("^(auto_round:)?llm_compressor", format):
            self.output_format = format
            self.backend = None
            if is_nv_fp(ar.data_type) or is_mx_fp(ar.data_type):
                from auto_round.export.export_to_llmcompressor import check_compressed_tensors_supported

                check_compressed_tensors_supported()
                self.backend = LLMCompressorFormat(ar.data_type, ar)
            elif is_static_wfp8afp8(ar):
                self.backend = LLMCompressorFormat(AutoRoundExportFormat.FP8_STATIC.value, ar)
                if ar.act_group_size != 0:
                    logger.warning(
                        f"scheme FP8_STATIC export to llm_compressor format only support for act_group_size 0,"
                        f" ,but got act_group_size={ar.act_group_size}, reset = 0"
                    )
                    ar.act_group_size = 0
                if ar.group_size > 0:
                    logger.warning(
                        f"please note that group_size={ar.group_size}"
                        " may not be supported for llm_compressor format, and cannot be loaded in llm_compressor"
                    )
        else:
            if format.upper() not in list(AutoRoundExportFormat.__members__.keys()):
                raise KeyError(f"Unsupported backend format llm_compressor:{format}, please check")
            self.output_format = f"llm_compressor:{format}"
            self.backend = None

    def check_and_reset_format(self, ar: BaseCompressor) -> str:
        if self.backend is not None:
            new_format = self.backend.check_and_reset_format(ar)
            self.backend = OutputFormat._format_list[new_format](new_format, ar) if new_format else self.backend

        if ar.act_bits <= 8 and (not is_standard_fp(ar.act_data_type) or ar.act_dynamic):
            if (is_nv_fp(ar.act_data_type) and "static_gs" in ar.act_data_type) or (is_mx_fp(ar.act_data_type)):
                return None
            else:
                bits, group_size, sym, act_bits = 8, -1, True, 8
                assert (
                    ar.bits == bits
                    and ar.group_size == group_size
                    and ar.sym == sym
                    and ar.act_bits == act_bits
                    and ar.act_dynamic
                ), (
                    f"Currently only support to export llm_compressor format for sym dynamic quantized"
                    f" W{ar.bits}A{ar.act_bits} model with group_size={group_size},"
                    f" but got bits={ar.bits}, group_size={ar.group_size}, sym={ar.sym},"
                    f" act_bits={ar.act_bits}"
                )
            return None
        return None

    def pack_layer(self, layer_name, model, device=None, **kwargs):
        if self.backend is not None:
            return self.backend.pack_layer(layer_name, model, device=device, **kwargs)
        if re.search(f"{AutoRoundExportFormat.MX_FP.value}|{AutoRoundExportFormat.NV_FP.value}", self.output_format):
            from auto_round.export.export_to_llmcompressor.export_to_fp import pack_layer

            return pack_layer(layer_name, model, device=device)
        elif re.search(f"{AutoRoundExportFormat.FP8_STATIC.value}", self.output_format):
            from auto_round.export.export_to_llmcompressor.export_to_static_fp import pack_layer

            return pack_layer(layer_name, model, self.get_backend_name(), device=device)

        ## passed as no other llm_compressor format is supported yet
        logger.warning("No other llm_compressor packing format(except NVFP&MXFP) is supported yet, skip packing")
        return

    def save_quantized(
        self,
        output_dir: str,
        model: torch.nn.Module = None,
        tokenizer: Callable = None,
        layer_config: dict = None,
        inplace: bool = True,
        device: Union[str, torch.device] = "cpu",
        serialization_dict: dict = None,
        **kwargs,
    ) -> torch.nn.Module:
        backend = self.get_backend_name()
        if re.search(f"{AutoRoundExportFormat.MX_FP.value}|{AutoRoundExportFormat.NV_FP.value}", backend):
            from auto_round.export.export_to_llmcompressor.export_to_fp import save_quantized_as_fp

            export_func = save_quantized_as_fp
        elif re.search(f"{AutoRoundExportFormat.FP8_STATIC.value}", backend):
            from auto_round.export.export_to_llmcompressor.export_to_static_fp import save_quantized_as_static_fp

            export_func = save_quantized_as_static_fp
        else:
            from auto_round.export.export_to_llmcompressor.export import save_quantized_as_llmcompressor

            export_func = save_quantized_as_llmcompressor
        return export_func(
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            layer_config=layer_config,
            inplace=inplace,
            device=device,
            backend=backend,
            serialization_dict=serialization_dict,
            **kwargs,
        )


@OutputFormat.register("auto_gptq", "gptqmodel")
class AutoGPTQFormat(OutputFormat):
    support_schemes = ["W4A16", "W2A16", "W3A16", "W8A16", "BF16", "W2A16G64", "W2A16G32"]
    format_name = "auto_gptq"

    def check_and_reset_format(self, ar):
        if not ar.sym:
            logger.warning(
                "the asymmetrical kernel of the GPTQ format may result in a noticeable accuracy drop,"
                " particularly for 2-bit quantization and smaller models."
                " We recommend exporting to either the AutoAWQ format ( only 4 bits) or "
                "the AutoRound format(2/3/4/8 bits)."
            )
        if self.backend is None:
            _check_divisible_by_32(ar)
        return super().check_and_reset_format(ar)

    def pack_layer(self, layer_name, model, device=None, **kwargs):
        if self.output_format.startswith("auto_round"):
            from auto_round.export.export_to_autoround.export import pack_layer

            pack_layer(layer_name, model, backend=self.output_format, device=device)
        else:
            from auto_round.export.export_to_autogptq.export import pack_layer

            pack_layer(layer_name, model, backend=self.output_format, device=device)

    def save_quantized(
        self,
        output_dir: str,
        model: torch.nn.Module = None,
        tokenizer: Callable = None,
        layer_config: dict = None,
        inplace: bool = True,
        device: Union[str, torch.device] = "cpu",
        serialization_dict: dict = None,
        **kwargs,
    ) -> torch.nn.Module:
        backend = self.get_backend_name()
        if backend == "auto_round:auto_gptq" or backend == "auto_round:gptqmodel":
            from auto_round.export.export_to_autoround.export import save_quantized_as_autoround

            export_func = save_quantized_as_autoround
        else:
            from auto_round.export.export_to_autogptq.export import save_quantized_as_autogptq

            export_func = save_quantized_as_autogptq
        return export_func(
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            layer_config=layer_config,
            inplace=inplace,
            device=device,
            backend=backend,
            serialization_dict=serialization_dict,
            **kwargs,
        )


@OutputFormat.register("auto_awq")
class AutoAWQFormat(OutputFormat):
    support_schemes = ["W4A16"]
    format_name = "auto_awq"

    @staticmethod
    def check_awq_gemm_compatibility(model, bits, group_size, sym, layer_configs=None):
        """Checks if a model is compatible with the AutoAWQ GEMM kernel.

        Args:
            model: The model object to evaluate, typically a PyTorch model.
            bits (int): The number of bits for quantization (must be 4 for compatibility).
            group_size (int): The group size for quantization.
            sym (bool): Whether symmetric quantization is used (not utilized in the current function logic).
            layer_configs (dict, optional): A dictionary mapping layer names to configurations, where each
                configuration can specify a custom number of bits for the layer.

        Returns:
            tuple: A tuple containing:
                - bool: `True` if the model is compatible, `False` otherwise.
                - str: An error message describing why the model is incompatible, or an empty string if compatible.
        """
        from auto_round.utils.model import get_layer_names_in_block, get_module

        if bits != 4:
            return False, "AutoAWQ GEMM kernel only supports 4 bits"
        for n, m in model.named_modules():
            if type(m) == transformers.pytorch_utils.Conv1D:
                return False, "AutoAWQ GEMM kernel does not support conv1d"

        layer_names = get_layer_names_in_block(model)
        for layer_name in layer_names:
            if (
                layer_configs is not None
                and layer_name in layer_configs.keys()
                and layer_configs[layer_name].get("bits", bits) > 8
            ):
                continue

            layer = get_module(model, layer_name)
            if layer.in_features % group_size != 0:
                return False, f"Layer {layer_name} in_features is not multiple of group_size {group_size}"
            if layer.out_features % (32 // bits) != 0:
                return False, f"Layer {layer_name} out_features is not multiple of 32 // bits"

        return True, ""

    def check_and_reset_format(self, ar):
        awq_supported, info = self.check_awq_gemm_compatibility(
            ar.model, ar.bits, ar.group_size, ar.sym, ar.layer_config
        )
        if not awq_supported:
            logger.warning(f"The AutoAWQ format may not be supported due to {info}")
        if ar.bits != 4:
            raise ValueError("The AWQ format only supports W4 quantization ")

        if self.backend is None:
            _check_divisible_by_32(ar)

        return super().check_and_reset_format(ar)

    def pack_layer(self, layer_name, model, device=None, **kwargs):
        from auto_round.export.export_to_awq.export import pack_layer

        pack_layer(layer_name, model, backend=self.output_format, device=device)

    def save_quantized(
        self,
        output_dir: str,
        model: torch.nn.Module = None,
        tokenizer: Callable = None,
        layer_config: dict = None,
        inplace: bool = True,
        device: Union[str, torch.device] = "cpu",
        serialization_dict: dict = None,
        **kwargs,
    ) -> torch.nn.Module:
        backend = self.get_backend_name()
        if backend == "auto_round:auto_awq":
            from auto_round.export.export_to_autoround.export import save_quantized_as_autoround

            export_func = save_quantized_as_autoround
        else:
            from auto_round.export.export_to_awq.export import save_quantized_as_autoawq

            export_func = save_quantized_as_autoawq

        return export_func(
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            layer_config=layer_config,
            inplace=inplace,
            backend=backend,
            device=device,
            serialization_dict=serialization_dict,
            **kwargs,
        )


@OutputFormat.register("itrex")
@OutputFormat.register("itrex_xpu")
class ITREXFormat(OutputFormat):
    support_schemes = ["W4A16", "W2A16", "W3A16", "W8A16", "BF16", "W2A16G64", "W2A16G32"]
    format_name = "itrex"

    def pack_layer(self, *args, **kwargs):
        pass

    def save_quantized(
        self,
        output_dir: str,
        model: torch.nn.Module = None,
        tokenizer: Callable = None,
        layer_config: dict = None,
        inplace: bool = True,
        device: Union[str, torch.device] = "cpu",
        serialization_dict: dict = None,
        **kwargs,
    ) -> torch.nn.Module:
        backend = self.get_backend_name()
        if backend == "itrex":
            from auto_round.export.export_to_itrex.export import save_quantized_as_itrex

            export_func = save_quantized_as_itrex
        else:
            from auto_round.export.export_to_itrex.export import save_quantized_as_itrex_xpu

            export_func = save_quantized_as_itrex_xpu
        return export_func(
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            layer_config=layer_config,
            inplace=inplace,
            device=device,
            backend=backend,
            serialization_dict=serialization_dict,
            **kwargs,
        )


@OutputFormat.register("gguf")
class GGUFFormat(OutputFormat):
    support_schemes = [
        "GGUF:Q4_0",
        "GGUF:Q4_1",
        "GGUF:Q5_0",
        "GGUF:Q5_1",
        "GGUF:Q2_K_S",
        "GGUF:Q3_K_S",
        "GGUF:Q3_K_M",
        "GGUF:Q3_K_L",
        "GGUF:Q4_K_S",
        "GGUF:Q4_K_M",
        "GGUF:Q5_K_S",
        "GGUF:Q5_K_M",
        "GGUF:Q6_K",
        "GGUF:Q8_0",
        "GGUF:Q2_K_MIXED",
    ]
    format_name = "gguf"

    def __init__(self, format: str, ar: BaseCompressor):
        if format.startswith("gguf:"):
            self.gguf_args_check(ar, format, model_type=ModelType.TEXT)
            if ar.mllm:
                self.gguf_args_check(ar, format, model_type=ModelType.MMPROJ)

            self.output_format = "gguf"
            self.backend_cls = GGUFFormat
            self.backend = GGUFFormat(format.split(":")[-1], ar)
        else:
            scheme = ar.scheme
            gguf_format = f"gguf:{format.lower()}"
            if format.lower().endswith("_mixed"):
                from auto_round.schemes import _handle_special_schemes

                ar.layer_config = _handle_special_schemes(gguf_format, ar.layer_config, ar.model)
                gguf_format = gguf_format.lower().replace("_mixed", "_s")
            if isinstance(scheme, str) and scheme.lower() != gguf_format:
                # logger.warning(f"reset scheme {scheme.lower()} to {gguf_format} for gguf format export")
                ar.scheme = gguf_format
            self.output_format = gguf_format
            self.backend = None
        self.mllm = ar.mllm

    def check_and_reset_format(self, ar):
        if ar.iters != 0 and ar.bits != 3 and not ar.enable_alg_ext:
            logger.warning_once(
                "`iters=0` is recommended when exporting to current GGUF format"
                " or add `enable_alg_ext` for better accuracy with much more tuning cost."
                " Please refer to https://github.com/intel/auto-round/tree/main/docs/gguf_alg_ext_acc.md"
                " for the accuracy results."
            )
        elif ar.bits >= 8 and ar.iters != 0:
            logger.warning_once("`iters=0` is recommended for bits>=8")

        return super().check_and_reset_format(ar)

    def pack_layer(
        self,
        name,
        model,
        backend,
        output_dir,
        layer_config,
        tokenizer,
        processor=None,
        image_processor=None,
        model_type=ModelType.TEXT,
        device="cpu",
    ):
        from auto_round.export.export_to_gguf.export import pack_gguf_layer

        pack_gguf_layer(
            name,
            model,
            backend,
            output_dir,
            layer_config,
            tokenizer,
            processor,
            image_processor,
            model_type,
            device,
        )

    def save_quantized(
        self,
        output_dir: str,
        model: torch.nn.Module = None,
        tokenizer: Callable = None,
        layer_config: dict = None,
        inplace: bool = True,
        device: Union[str, torch.device] = "cpu",
        serialization_dict: dict = None,
        **kwargs,
    ) -> torch.nn.Module:
        from auto_round.export.export_to_gguf.export import save_quantized_as_gguf

        backend = self.get_backend_name()
        return save_quantized_as_gguf(
            output_dir=output_dir,
            model=model,
            backend=backend,
            layer_config=layer_config,
            mllm=self.mllm,
            device=device,
            serialization_dict=serialization_dict,
            **kwargs,
        )

    @staticmethod
    def gguf_args_check(args_or_ar, formats: Union[str, list[str]] = None, model_type=ModelType.TEXT):
        import argparse

        from auto_round.export.export_to_gguf.config import GGUF_CONFIG
        from auto_round.export.export_to_gguf.convert import download_convert_file
        from auto_round.logger import logger
        from auto_round.utils.model import download_or_get_path, get_gguf_architecture

        formats = [formats] if isinstance(formats, str) else formats
        formats = sorted(formats, key=lambda x: len(x))
        export_gguf = False
        for f in formats:
            if f.startswith("gguf"):
                export_gguf = True

            if f.startswith("gguf") and f not in GGUF_CONFIG:
                logger.error(f"{f} is not supported, please check.")

        redownload = False
        if export_gguf:
            try:
                from auto_round.export.export_to_gguf.convert_hf_to_gguf import (  # pylint: disable=E0401
                    ModelBase,
                    ModelType,
                    get_model_architecture,
                )

                if isinstance(args_or_ar.model, str):
                    model_path = args_or_ar.model
                else:
                    model_path = args_or_ar.model.name_or_path
                if not os.path.isdir(model_path):
                    model_path = download_or_get_path(model_path, args_or_ar.platform)
                model_architecture = get_gguf_architecture(model_path, model_type=ModelType.TEXT)
                if model_architecture not in ModelBase._model_classes[ModelType.TEXT]:
                    logger.warning(
                        f"Current version of gguf export does not support for {model_architecture},"
                        " will re-download dependency file. Please restart the task."
                    )
                    redownload = True
            except ModuleNotFoundError as e:
                if "convert_hf_to_gguf" in str(e):
                    logger.warning("GGUF export dependency file is not found, download from github.")
                    redownload = True
            except AttributeError as e:
                raise ImportError(
                    "Please use the latest gguf-py, you can use the following command to install it:\n"
                    "git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp/gguf-py"
                    " && pip install . sentencepiece"
                )
            download_convert_file(redownload)

            try:
                from auto_round.export.export_to_gguf.convert_hf_to_gguf import (  # pylint: disable=E0401
                    ModelBase,
                    ModelType,
                )
            except ImportError as e:
                raise ImportError(
                    "Please use the latest gguf-py, you can use the following command to install it:\n"
                    "git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp/gguf-py"
                    " && pip install . sentencepiece"
                )
            if isinstance(args_or_ar.model, str):
                model_path = args_or_ar.model
            else:
                model_path = args_or_ar.model.name_or_path
            if not os.path.isdir(model_path):
                model_path = download_or_get_path(model_path, args_or_ar.platform)
            model_architecture = get_gguf_architecture(model_path, model_type=ModelType.TEXT)
            if model_architecture not in ModelBase._model_classes[ModelType.TEXT]:
                logger.error(f"Model {model_architecture} is not supported to export gguf format.")
                sys.exit(1)

        pattern = re.compile(r"q\d_k")
        pre_dq_format = ""
        unsupported_list, reset_list = [], []
        for format in GGUF_CONFIG:
            if format in formats:
                if format == "q6_k_s":
                    logger.warning("Please note that q6_k_s is q6_k.")

                if re.search(pattern, format):
                    if pre_dq_format and re.search(pattern, format).group() not in pre_dq_format:
                        logger.error(f"Cannot export {pre_dq_format} and {format} at the same time.")
                        sys.exit(-1)
                    else:
                        pre_dq_format = format

                unsupported_list, reset_list = [], []
                gguf_config = GGUF_CONFIG[format]
                for k, v in gguf_config.items():
                    if not hasattr(args_or_ar, k):
                        continue
                    if k == "data_type":
                        if re.search(r"q\d_1", format) and len(formats) > 1:
                            v = "int"
                    if k == "sym" and isinstance(args_or_ar, argparse.Namespace):
                        k = "asym"
                        v = not v
                    if getattr(args_or_ar, k) != v:
                        unsupported_list.append(f"{k}={getattr(args_or_ar, k)}")
                        reset_list.append(f"{k}={v}")
                        setattr(args_or_ar, k, v)
                if len(unsupported_list) > 0:
                    logger.info(
                        f"format {format} does not support for {', '.join(unsupported_list)},"
                        f" reset to {', '.join(reset_list)}."
                    )
        # Removed obsolete commented-out block for improved readability and maintainability.
        return args_or_ar

    def immediate_pack(
        self,
        name: str,
        model: torch.nn.Module,
        device: torch.device,
        output_dir: str = None,
        mllm: bool = False,
        layer_config: dict = None,
        tokenizer=None,
        processor=None,
        image_processor=None,
        **kwargs,
    ):
        m = get_module(model, name)
        if not check_to_quantized(m):
            return
        model_type = ModelType.MMPROJ if mllm else ModelType.TEXT
        self.pack_layer(
            name,
            model,
            self.get_backend_name(),
            output_dir,
            layer_config=layer_config,
            tokenizer=tokenizer,
            processor=processor,
            image_processor=image_processor,
            model_type=model_type,
            device=device,
        )


@OutputFormat.register("auto_round")
@OutputFormat.register("auto_round:auto_awq")
@OutputFormat.register("auto_round:llm_compressor")
@OutputFormat.register("auto_round:gptqmodel", "auto_round:auto_gptq")
class AutoRoundFormat(OutputFormat):
    support_schemes = [
        "W4A16",
        "W2A16",
        "W3A16",
        "W8A16",
        "MXFP4",
        "MXFP8",
        "NVFP4",
        "FPW8A16",
        "W2A16G64",
        "W2A16G32",
        "FP8_STATIC",
        "BF16",
    ]
    format_name = "auto_round"

    def __init__(self, format: str, ar: BaseCompressor):
        self.output_format = "auto_round"
        self.backend = None

        if format == "auto_round":
            if ar.sym and "int" in ar.data_type:
                self.backend = AutoGPTQFormat("auto_round:auto_gptq", ar)
            elif ar.bits == 4 and not ar.sym and "int" in ar.data_type:
                if ar.layer_config is None:
                    enable_awq = True
                else:
                    enable_awq = all(
                        config["bits"] == ar.bits or config["bits"] >= 16 for config in ar.layer_config.values()
                    )
                if enable_awq:
                    self.backend = AutoAWQFormat("auto_round:auto_awq", ar)
            elif is_nv_fp(ar.data_type) or is_mx_fp(ar.data_type):
                self.backend = AutoRoundFormat(ar.data_type, ar)
            elif is_static_wfp8afp8(ar):  # static wfp8afp8
                self.backend = AutoRoundFormat(AutoRoundExportFormat.FP8_STATIC.value, ar)
            elif ar.data_type.startswith("fp") and ar.bits == 8 and ar.act_bits >= 16:  # woq fp8
                self.backend = AutoRoundFormat(AutoRoundExportFormat.FP8.value, ar)
            elif ar.act_bits < 16:
                raise ValueError(
                    "AutoRound format does not support exporting "
                    "for the current quantization configuration, "
                    "please change to `fake` format for research purpose"
                )
        # for auto_round:fp8_static, auto_round:nv_fp etc.
        elif not format.startswith("auto_round"):
            if format.upper() not in list(AutoRoundExportFormat.__members__.keys()):
                raise KeyError(f"Unsupported backend format auto_round:{format}, please check")
            self.output_format = f"auto_round:{format}"
            self.backend = None
        else:
            backend = format.split(":")[1] if ":" in format else None
            self.backend = self._format_list.get(backend)(format, ar) if backend else None

        if self.backend is not None:
            self.support_schemes = self.backend.support_schemes

    def check_and_reset_format(self, ar):
        if self.backend is not None:
            new_format = self.backend.check_and_reset_format(ar)
            self.backend = OutputFormat._format_list[new_format](new_format, ar) if new_format else self.backend

        if ar.act_bits <= 8:
            if is_standard_fp(ar.act_data_type) and not ar.act_dynamic:
                if (
                    ar.act_group_size != 0
                    and not ar.act_dynamic
                    and self.get_backend_name() == f"auto_round:{AutoRoundExportFormat.FP8.value}"
                ):
                    logger.warning(
                        f"Please note that quantize activation with act_group_size={ar.act_group_size}"
                        " may result in failure to export or import normally."
                    )
        if self.backend is None:
            _check_divisible_by_32(ar)
        return None

    def pack_layer(self, layer_name, model, device=None, **kwargs):
        if self.backend is not None:
            return self.backend.pack_layer(layer_name, model, device=device, **kwargs)

        backend = self.get_backend_name()

        if self.output_format in [
            f"auto_round:{AutoRoundExportFormat.NV_FP.value}",
            f"auto_round:{AutoRoundExportFormat.MX_FP.value}",
            f"auto_round:{AutoRoundExportFormat.MX_FP_RCEIL.value}",
            f"auto_round:{AutoRoundExportFormat.NV_FP4_WITH_STATIC_GS.value}",
        ]:
            from auto_round.export.export_to_autoround.export_to_nvfp_mxfp import pack_layer

            pack_func = pack_layer
        elif self.output_format in [
            f"auto_round:{AutoRoundExportFormat.FP8.value}",
            f"auto_round:{AutoRoundExportFormat.FP8_STATIC.value}",
            f"auto_round:{AutoRoundExportFormat.FP8_STATIC.value}",
        ]:
            from auto_round.export.export_to_autoround.export_to_fp8 import pack_layer

            pack_func = pack_layer
        else:
            from auto_round.export.export_to_autoround.export import pack_layer

            pack_func = pack_layer
        return pack_func(layer_name, model, backend, device)

    def save_quantized(
        self,
        output_dir: str,
        model: torch.nn.Module = None,
        tokenizer: Callable = None,
        layer_config: dict = None,
        inplace: bool = True,
        device: Union[str, torch.device] = "cpu",
        serialization_dict: dict = None,
        **kwargs,
    ) -> torch.nn.Module:
        if self.backend is not None:
            return self.backend.save_quantized(
                output_dir=output_dir,
                model=model,
                tokenizer=tokenizer,
                layer_config=layer_config,
                inplace=inplace,
                device=device,
                serialization_dict=serialization_dict,
                **kwargs,
            )
        backend = self.get_backend_name()
        if re.search(f"{AutoRoundExportFormat.MX_FP.value}|{AutoRoundExportFormat.NV_FP.value}", backend):
            from auto_round.export.export_to_autoround.export_to_nvfp_mxfp import save_quantized_as_fp

            backend = "auto_round:llm_compressor"
            export_func = save_quantized_as_fp
        elif (
            serialization_dict.get("data_type", "int") == "fp"
            and serialization_dict.get("bits", 16) == 8
            and serialization_dict.get("act_bits", 16) >= 16
        ):
            from auto_round.export.export_to_autoround.export_to_fp8 import save_quantized_as_autoround

            backend = "auto_round"
            export_func = save_quantized_as_autoround
        else:
            from auto_round.export.export_to_autoround.export import save_quantized_as_autoround

            export_func = save_quantized_as_autoround
        return export_func(
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            layer_config=layer_config,
            inplace=inplace,
            device=device,
            backend=backend,
            serialization_dict=serialization_dict,
            **kwargs,
        )
