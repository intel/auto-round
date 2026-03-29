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

"""Output format classes for exporting AutoRound-quantized models.

This module defines the :class:`OutputFormat` abstract base class and all
concrete format implementations used to pack quantized weights and save models
in deployment-ready formats.

Supported formats include:
    - ``fake``: Saves the model without packing (research/debugging).
    - ``auto_round``: AutoRound-native format (delegates to auto_gptq/awq/fp8
      depending on the quantization scheme).
    - ``auto_gptq`` / ``gptqmodel``: GPTQ-compatible packing.
    - ``auto_awq``: AWQ-compatible packing (W4 only).
    - ``llm_compressor``: LLM-Compressor / compressed-tensors format
      (FP8, MX-FP, NV-FP, INT8 W8A8).
    - ``gguf``: GGUF format for llama.cpp-compatible models.
    - ``fp8``: Block-wise FP8 packing.
"""

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
    is_block_wfp8,
    is_dynamic_afp8,
    is_dynamic_wint8aint8,
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
    compress_layer_names,
    copy_python_files_from_model_cache,
    find_matching_blocks,
    get_block_names,
    get_module,
    logger,
    unsupported_meta_device,
)


class AutoRoundExportFormat(str, Enum):
    """String enum of AutoRound backend export format identifiers.

    These values are used as backend specifiers within compound format strings
    such as ``"auto_round:fp8_static"`` or ``"llm_compressor:mxfp8"``.
    """

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
    INT8_W8A8 = "int8_w8a8"
    FP8_BLOCK = "fp8_block"


if TYPE_CHECKING:
    from auto_round.compressors.base import BaseCompressor


def _check_compatibility(formats: list[str], ar: BaseCompressor):
    """Validates and normalizes a list of requested export formats.

    Ensures that GGUF formats are not mixed with other non-fake formats and
    that the GGUF scheme derived from the compressor matches the requested
    format.  Also converts block-wise FP8 ``auto_round`` requests to ``fp8``.

    Args:
        formats (list[str]): List of format name strings to validate.
        ar (BaseCompressor): The compressor instance whose scheme and
            configuration determine compatibility.

    Returns:
        list[str]: The (possibly modified) list of validated format strings.

    Raises:
        ValueError: If GGUF is requested alongside incompatible formats.
    """
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
    if isinstance(ar.group_size, tuple) and any(["auto_round" in f.lower() for f in formats]):
        logger.warning(
            "`auto_round` format can't be used for deploying block-wise fp8 quantization now, use `fp8` instead."
        )
        formats = ["fp8" if "auto_round" in f.lower() else f for f in formats]
    return formats


def get_formats(
    format: str,
    ar: BaseCompressor,
) -> list[OutputFormat]:
    """Parses a format string and returns the corresponding OutputFormat instances.

    Args:
        format (str): Comma-separated format string (e.g. ``"auto_round,fake"``).
            May contain ``q*_`` wildcard patterns.
        ar (BaseCompressor): Compressor instance providing scheme and model info.

    Returns:
        list[OutputFormat]: Ordered list of :class:`OutputFormat` objects ready
        for packing and serialization.

    Raises:
        KeyError: If an unknown (non-GGUF) format name is requested.
    """

    def remove_duplicates(lst):
        """Return a list with duplicates removed, preserving original order."""
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
    """Marks layers whose weight dimensions are not divisible by 32 as full-precision.

    Layers with output or input feature sizes not divisible by 32 are
    incompatible with most INT packing kernels and are therefore set to
    ``bits=16`` (i.e. skipped) in ``ar.layer_config``.

    Args:
        ar (BaseCompressor): Compressor instance with ``model``, ``scheme``,
            ``supported_types``, ``inner_supported_types``, and
            ``layer_config`` attributes.
    """
    from auto_round.schemes import preset_name_to_scheme

    if isinstance(ar.scheme, str):
        default_dict = asdict(preset_name_to_scheme(ar.scheme.upper()))
    else:
        default_dict = asdict(ar.scheme)
    skipped_layers = []
    if default_dict["data_type"] == "int" and default_dict["act_bits"] >= 16:
        for n, m in ar.model.named_modules():
            if type(m) in ar.supported_types or m.__class__.__name__ in ar.inner_supported_types:
                if m.weight.shape[0] % 32 or m.weight.shape[1] % 32:
                    if ar.layer_config is None:
                        ar.layer_config = {}
                    if ar.layer_config.get(n) is not None and ar.layer_config[n]["bits"] >= 16:
                        continue
                    ar.layer_config.setdefault(n, copy.deepcopy(default_dict))
                    ar.layer_config[n].update({"bits": 16, "data_type": "fp", "fixed_by_user": True})
                    skipped_layers.append(n)
    compressed_skipped_layers = compress_layer_names(skipped_layers)
    logger.warning_once(
        f"some layers are skipped quantization (shape not divisible by 32): {compressed_skipped_layers}"
    )


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
        """Initializes the output format, validating scheme compatibility.

        Args:
            format (str): The format name string (e.g. ``"auto_round"``).
            ar (BaseCompressor): Compressor instance supplying the scheme to
                validate against ``support_schemes``.
        """
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
        """Class decorator that registers an OutputFormat subclass under one or more names.

        Args:
            *names (str): One or more format name strings to register.

        Returns:
            Callable[[OutputFormat], OutputFormat]: Decorator function that adds the
            class to ``_format_list`` under each name and returns it unchanged.

        Raises:
            AssertionError: If no names are provided.
        """
        assert names

        def func(output_format: OutputFormat) -> OutputFormat:
            """Register the decorated OutputFormat class under all given names."""
            for name in names:
                cls._format_list[name] = output_format
            return output_format

        return func

    @classmethod
    def get_support_matrix(cls: OutputFormat) -> str:
        """Builds a formatted string listing all registered formats and their supported schemes.

        Returns:
            str: ANSI-colored multi-line string with one entry per registered format.
        """
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
        """Returns the fully-qualified backend name for this format.

        For simple formats (no backend) this is the ``output_format`` string.
        For compound formats the backend names are joined with colons, e.g.
        ``"auto_round:llm_compressor:fp8_static"``.

        Returns:
            str: Backend name string.
        """
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
        """Checks whether a given scheme is supported by this format.

        Args:
            scheme (str | QuantizationScheme): Scheme name or object to check.

        Returns:
            bool: ``True`` if the scheme is compatible with this format.
        """
        if isinstance(scheme, str) and scheme.upper() in cls.support_schemes:
            return True
        if isinstance(scheme, QuantizationScheme):
            return cls.check_scheme_args(scheme)
        return False

    @classmethod
    def check_scheme_args(cls: OutputFormat, scheme: QuantizationScheme) -> bool:
        """Validates detailed scheme arguments for compatibility with this format.

        Override in subclasses to enforce format-specific restrictions on
        ``bits``, ``data_type``, ``group_size``, etc. The base implementation
        performs no validation and always returns ``True``.

        Args:
            scheme (QuantizationScheme): The quantization scheme to validate.

        Returns:
            bool: ``True`` if the scheme arguments are valid.

        Raises:
            ValueError: If any scheme argument violates format constraints.
        """
        return True

    def check_and_reset_format(self, ar: BaseCompressor) -> str:
        """Checks format compatibility and optionally resets to a fallback format.

        Inspects the compressor configuration (FP8 data types, activation
        quantization settings) and returns a replacement format name when the
        current format cannot support the configuration.

        Args:
            ar (BaseCompressor): Compressor instance with quantization config.

        Returns:
            str | None: Name of a replacement format (e.g. ``"fake"``) or
            ``None`` if no reset is needed.
        """
        if self.backend is not None:
            new_format = self.backend.check_and_reset_format(ar)
            self.backend = OutputFormat._format_list[new_format](new_format, ar) if new_format else self.backend

        w_fp8 = ar.data_type.startswith("fp") and ar.bits == 8
        act_fp8 = ar.act_data_type.startswith("fp") and ar.act_bits == 8
        is_block_dynamic_fp8 = (
            self.format_name in ["fp8", "auto_round:fp8"] and isinstance(ar.group_size, tuple) and ar.act_dynamic
        )
        if (w_fp8 or act_fp8) and not is_block_dynamic_fp8:
            error_msg = (
                f"is only supported to export auto_round or llm_compressor format,"
                f" but got {self.format_name}, please check."
            )
            error_msg = ("act_data_type<fp8> " + error_msg) if act_fp8 else error_msg
            error_msg = ("data_type<fp8> " + error_msg) if w_fp8 else error_msg
            logger.error(error_msg)
            sys.exit(-1)

        if ar.act_bits <= 8 and (not is_standard_fp(ar.act_data_type) or ar.act_dynamic) and not is_block_dynamic_fp8:
            logger.warning(
                f"{self.format_name} format not support for current activation quantization configuration,"
                " reset to fake format and save."
            )
            return "fake"

        return None

    @abstractmethod
    def pack_layer(self, *args, **kwargs):
        """Packs quantization parameters into the layer in-place.

        Args:
            *args: Format-specific positional arguments.
            **kwargs: Format-specific keyword arguments.
        """
        pass

    @abstractmethod
    def save_quantized(self, *args, **kwargs):
        """Saves the quantized model to the given output directory.

        Args:
            *args: Format-specific positional arguments.
            **kwargs: Format-specific keyword arguments.
        """
        pass

    def immediate_pack(self, name: str, model: torch.nn.Module, device: torch.device, **kwargs):
        """Packs a single named layer immediately if it is marked for quantization.

        Args:
            name (str): Dot-separated layer name within ``model``.
            model (torch.nn.Module): The model containing the layer.
            device (torch.device): Device on which packing should be performed.
            **kwargs: Additional keyword arguments forwarded to :meth:`pack_layer`.
        """
        m = get_module(model, name)
        if not check_to_quantized(m):
            return

        self.pack_layer(name, model, device=device)

    def is_gguf(self) -> bool:
        """Returns True if this is a GGUF format."""
        return "gguf" in self.output_format

    def is_fake(self) -> bool:
        """Returns True if this is a fake (no-packing) format."""
        return self.output_format == "fake"

    def is_gptq(self) -> bool:
        """Returns True if this format uses GPTQ kernel packing."""
        return "gptq" in self.output_format or (self.backend is not None and self.backend.is_gptq())

    def is_awq(self) -> bool:
        """Returns True if this format uses AWQ kernel packing."""
        return "awq" in self.output_format or (self.backend is not None and self.backend.is_awq())

    def is_llm_compressor(self) -> bool:
        """Returns True if this format targets llm_compressor / compressed-tensors."""
        return "llm_compressor" in self.output_format or (self.backend is not None and self.backend.is_llm_compressor())


@OutputFormat.register("fake")
class FakeFormat(OutputFormat):
    """Fake/no-op output format used for testing or dry-run quantization.

    Saves the model in standard HuggingFace format without any weight packing.
    """

    support_schemes = None
    format_name = "fake"

    def check_and_reset_format(self, ar: BaseCompressor) -> str:
        """Validate and optionally reset the format for the given compressor (no-op for fake).

        Args:
            ar (BaseCompressor): Compressor instance (unused).

        Returns:
            str: Always returns ``None`` – fake format requires no special format string.
        """
        return None

    # fake format will not execute pack_layer.
    def pack_layer(self, *args, **kwargs):
        """No-op: fake format does not pack layers."""
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
        """Saves the model in standard HuggingFace format without weight packing.

        Args:
            output_dir (str): Directory path where the model will be saved.
            model (torch.nn.Module, optional): The model to save.
            tokenizer (Callable, optional): Tokenizer to save alongside the model.
            layer_config (dict, optional): Layer configuration (unused for fake format).
            inplace (bool, optional): Whether to operate in-place. Defaults to ``True``.
            device (str | torch.device, optional): Device for the model. Defaults to ``"cpu"``.
            serialization_dict (dict, optional): Extra serialization metadata (unused).
            **kwargs: Additional keyword arguments (e.g. ``processor``).

        Returns:
            torch.nn.Module: The (possibly CPU-moved) model after saving.
        """
        if not unsupported_meta_device(model):
            model = model.to("cpu")
            model.save_pretrained(output_dir)
        elif hasattr(model, "config") and model.config is not None:
            model.config.save_pretrained(output_dir)

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
    """Output format targeting the LLMCompressor / llm-compressor export pipeline.

    Supports MXFP4, MXFP8, NVFP4, FPW8A16, FP8_STATIC, INT8_W8A8, and FP8_BLOCK schemes.
    """

    support_schemes = ["MXFP4", "MXFP8", "NVFP4", "FPW8A16", "FP8_STATIC", "INT8_W8A8", "FP8_BLOCK"]
    format_name = "llm_compressor"

    def __init__(self, format, ar):
        """Initializes the LLMCompressor format, selecting the appropriate backend.

        Args:
            format (str): Format string (e.g. ``"llm_compressor"`` or a sub-backend
                such as ``"fp8_static"``).
            ar (BaseCompressor): Compressor instance supplying scheme and config.

        Raises:
            ValueError: If the scheme is not supported by llm_compressor.
            KeyError: If an unsupported backend sub-format is specified.
        """
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
            elif is_dynamic_afp8(ar) and is_block_wfp8(ar):
                self.backend = LLMCompressorFormat(AutoRoundExportFormat.FP8_BLOCK.value, ar)
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
            elif is_dynamic_wint8aint8(ar):
                from auto_round.export.export_to_llmcompressor import check_compressed_tensors_supported

                check_compressed_tensors_supported()
                self.backend = LLMCompressorFormat(AutoRoundExportFormat.INT8_W8A8.value, ar)
        else:
            if format.upper() not in list(AutoRoundExportFormat.__members__.keys()):
                raise KeyError(f"Unsupported backend format llm_compressor:{format}, please check")
            self.output_format = f"llm_compressor:{format}"
            self.backend = None

    @classmethod
    def check_scheme_args(cls: OutputFormat, scheme: QuantizationScheme) -> bool:
        """Validate that the quantization scheme is compatible with LLMCompressor format.

        Args:
            cls (OutputFormat): The format class.
            scheme (QuantizationScheme): Quantization scheme to validate.

        Returns:
            bool: ``True`` if the scheme is valid.

        Raises:
            ValueError: If the scheme uses unsupported bits, data types, or group sizes.
        """
        error_logs = []
        if scheme.bits not in [4, 8, 16]:
            error_logs.append(f"bits={scheme.bits}")
        if not re.search("mxfp|fp|nvfp|int", scheme.data_type):
            error_logs.append(f"data_type={scheme.data_type}")
        if scheme.data_type in ["fp", "int"] and scheme.bits != 8:
            error_logs.append(f"data_type={scheme.data_type}, bits={scheme.bits}")
        if scheme.super_bits:
            error_logs.append(f"super_bits={scheme.super_bits}")
        if scheme.super_group_size:
            error_logs.append(f"super_group_size={scheme.super_group_size}")
        if isinstance(scheme.group_size, tuple):
            if scheme.bits != 8:
                error_logs.append(f"bits={scheme.bits}")
            if scheme.data_type != "fp":
                error_logs.append(f"data_type={scheme.data_type}")
            if len(scheme.group_size) != 2:
                error_logs.append(f"group_size={scheme.group_size}")
            if not scheme.act_dynamic:
                error_logs.append(f"act_dynamic={scheme.act_dynamic}")
            if not isinstance(scheme.act_group_size, int):
                error_logs.append(f"act_group_size={scheme.act_group_size}")
            if scheme.act_bits != 8:
                error_logs.append(f"act_bits={scheme.act_bits}")
            if scheme.act_data_type != "fp":
                error_logs.append(f"act_data_type={scheme.act_data_type}")
        if error_logs:
            raise ValueError(
                f"LLMCompressor format support quantization scheme with {','.join(cls.support_schemes)} "
                f"but got {', '.join(error_logs)}, please have a check."
            )
        return True

    def check_and_reset_format(self, ar: BaseCompressor) -> str | None:
        """Validate the format against the compressor config and optionally switch to a sub-backend.

        Args:
            ar (BaseCompressor): Compressor instance with scheme and activation settings.

        Returns:
            str | None: Sub-backend format string if a change is needed, or ``None``.
        """
        if self.backend is not None:
            new_format = self.backend.check_and_reset_format(ar)
            self.backend = OutputFormat._format_list[new_format](new_format, ar) if new_format else self.backend

        if ar.act_bits <= 8 and (not is_standard_fp(ar.act_data_type) or ar.act_dynamic):
            if (is_nv_fp(ar.act_data_type) and "static_gs" in ar.act_data_type) or (is_mx_fp(ar.act_data_type)):
                return None
            elif is_dynamic_afp8(ar) and is_block_wfp8(ar):
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
        """Packs quantization parameters into the layer using the llm_compressor backend.

        Args:
            layer_name (str): Dot-separated layer name.
            model (torch.nn.Module): The model containing the layer.
            device (torch.device | None, optional): Device for packing.
            **kwargs: Additional keyword arguments.
        """
        if self.backend is not None:
            return self.backend.pack_layer(layer_name, model, device=device, **kwargs)
        if re.search(f"{AutoRoundExportFormat.MX_FP.value}|{AutoRoundExportFormat.NV_FP.value}", self.output_format):
            from auto_round.export.export_to_llmcompressor.export_to_fp import pack_layer

            return pack_layer(layer_name, model, device=device)
        elif re.search(f"{AutoRoundExportFormat.FP8_STATIC.value}", self.output_format):
            from auto_round.export.export_to_llmcompressor.export_to_static_fp import pack_layer

            return pack_layer(layer_name, model, self.get_backend_name(), device=device)
        elif re.search(f"{AutoRoundExportFormat.INT8_W8A8.value}", self.output_format):
            from auto_round.export.export_to_llmcompressor.export import pack_layer

            return pack_layer(layer_name, model, device=device)
        elif re.search(f"{AutoRoundExportFormat.FP8_BLOCK.value}", self.output_format):
            from auto_round.export.export_to_llmcompressor.export import pack_layer

            return pack_layer(layer_name, model, device=device)
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
        """Saves the model in llm_compressor / compressed-tensors format.

        Args:
            output_dir (str): Destination directory.
            model (torch.nn.Module, optional): Model to serialize.
            tokenizer (Callable, optional): Tokenizer to save.
            layer_config (dict, optional): Per-layer quantization configuration.
            inplace (bool, optional): Operate in-place. Defaults to ``True``.
            device (str | torch.device, optional): Device. Defaults to ``"cpu"``.
            serialization_dict (dict, optional): Serialization metadata.
            **kwargs: Extra keyword arguments forwarded to the export function.

        Returns:
            torch.nn.Module: The serialized model.
        """
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
    """Output format targeting the AutoGPTQ / GPTQModel export pipeline.

    Supports W2A16, W3A16, W4A16, W8A16, BF16, and mixed-precision variants.
    """

    support_schemes = ["W4A16", "W2A16", "W3A16", "W8A16", "BF16", "W2A16G64", "W2A16G32", "W4A16_MIXED"]
    format_name = "auto_gptq"

    def check_and_reset_format(self, ar):
        """Warns about asymmetric GPTQ accuracy and checks layer divisibility.

        Args:
            ar (BaseCompressor): Compressor instance.

        Returns:
            str | None: Replacement format name, or ``None``.
        """
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

    @classmethod
    def check_scheme_args(cls: OutputFormat, scheme: QuantizationScheme) -> bool:
        """Validate that the quantization scheme is compatible with AutoGPTQ format.

        Args:
            cls (OutputFormat): The format class.
            scheme (QuantizationScheme): Quantization scheme to validate.

        Returns:
            bool: ``True`` if valid.

        Raises:
            ValueError: If the scheme uses unsupported bits, data type, or super-group settings.
        """
        error_logs = []
        if scheme.bits not in [2, 3, 4, 8, 16]:
            error_logs.append(f"bits={scheme.bits}")
        if not re.search("int", scheme.data_type):
            error_logs.append(f"data_type={scheme.data_type}")
        if scheme.super_bits:
            error_logs.append(f"super_bits={scheme.super_bits}")
        if scheme.super_group_size:
            error_logs.append(f"super_group_size={scheme.super_group_size}")
        if error_logs:
            raise ValueError(
                f"{cls.format_name} format support quantization scheme with {','.join(cls.support_schemes)} "
                f"but got {', '.join(error_logs)}, please have a check."
            )
        return True

    def pack_layer(self, layer_name, model, device=None, **kwargs):
        """Packs quantization data into a layer using AutoGPTQ or AutoRound packing.

        Args:
            layer_name (str): Dot-separated layer name.
            model (torch.nn.Module): The model containing the layer.
            device (torch.device | None, optional): Device for packing.
            **kwargs: Additional keyword arguments.
        """
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
        """Saves the model in AutoGPTQ format.

        Args:
            output_dir (str): Destination directory.
            model (torch.nn.Module, optional): Model to serialize.
            tokenizer (Callable, optional): Tokenizer to save.
            layer_config (dict, optional): Per-layer quantization configuration.
            inplace (bool, optional): Operate in-place. Defaults to ``True``.
            device (str | torch.device, optional): Device. Defaults to ``"cpu"``.
            serialization_dict (dict, optional): Serialization metadata.
            **kwargs: Extra keyword arguments forwarded to the export function.

        Returns:
            torch.nn.Module: The serialized model.
        """
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
    """Output format targeting the AutoAWQ export pipeline.

    Only supports W4A16 (4-bit symmetric integer weight quantization).
    """

    support_schemes = ["W4A16"]
    format_name = "auto_awq"

    @classmethod
    def check_scheme_args(cls: OutputFormat, scheme: QuantizationScheme) -> bool:
        """Validate that the quantization scheme is compatible with AutoAWQ format.

        Args:
            cls (OutputFormat): The format class.
            scheme (QuantizationScheme): Quantization scheme to validate.

        Returns:
            bool: ``True`` if valid.

        Raises:
            ValueError: If the scheme uses unsupported bits, data type, or super-group settings.
        """
        error_logs = []
        if scheme.bits != 4:
            error_logs.append(f"bits={scheme.bits}")
        if not re.search("int", scheme.data_type):
            error_logs.append(f"data_type={scheme.data_type}")
        if scheme.super_bits:
            error_logs.append(f"super_bits={scheme.super_bits}")
        if scheme.super_group_size:
            error_logs.append(f"super_group_size={scheme.super_group_size}")
        if error_logs:
            raise ValueError(
                f"{cls.format_name} format support quantization scheme with {','.join(cls.support_schemes)} "
                f"but got {', '.join(error_logs)}, please have a check."
            )
        return True

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
        """Verifies AWQ GEMM kernel compatibility and validates W4 constraint.

        Args:
            ar (BaseCompressor): Compressor instance.

        Returns:
            str | None: Replacement format name, or ``None``.

        Raises:
            ValueError: If ``ar.bits != 4``.
        """
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
        """Packs quantization data into a layer using the AWQ kernel.

        Args:
            layer_name (str): Dot-separated layer name.
            model (torch.nn.Module): The model containing the layer.
            device (torch.device | None, optional): Device for packing.
            **kwargs: Additional keyword arguments.
        """
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
        """Saves the model in AutoAWQ format.

        Args:
            output_dir (str): Destination directory.
            model (torch.nn.Module, optional): Model to serialize.
            tokenizer (Callable, optional): Tokenizer to save.
            layer_config (dict, optional): Per-layer quantization configuration.
            inplace (bool, optional): Operate in-place. Defaults to ``True``.
            device (str | torch.device, optional): Device. Defaults to ``"cpu"``.
            serialization_dict (dict, optional): Serialization metadata.
            **kwargs: Extra keyword arguments forwarded to the export function.

        Returns:
            torch.nn.Module: The serialized model.
        """
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


@OutputFormat.register("gguf")
class GGUFFormat(OutputFormat):
    """Output format targeting the GGUF model format for llama.cpp-compatible inference.

    Supports a wide range of GGUF quantization sub-types (Q2_K through Q8_0).
    """

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
        """Initializes the GGUFFormat, validating the requested GGUF sub-type.

        Handles both ``"gguf:<type>"`` (top-level GGUF with nested backend) and
        bare sub-type strings (used when constructing the nested backend).

        Args:
            format (str): GGUF format string (e.g. ``"gguf:q4_k_m"``).
            ar (BaseCompressor): Compressor instance with model and config.
        """
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
                logger.warning(f"reset scheme {scheme.lower()} to {gguf_format} for gguf format export")
                ar.scheme = gguf_format
            self.output_format = gguf_format
            self.backend = None
        self.mllm = ar.mllm

    @classmethod
    def check_scheme_args(cls: OutputFormat, scheme: QuantizationScheme) -> bool:
        """Validate that the quantization scheme is compatible with GGUF format.

        Args:
            cls (OutputFormat): The format class.
            scheme (QuantizationScheme): Quantization scheme to validate.

        Returns:
            bool: ``True`` if valid.

        Raises:
            ValueError: If the data type is not integer-based.
        """
        error_logs = []
        if not re.search("int", scheme.data_type):
            error_logs.append(f"data_type={scheme.data_type}")
        if error_logs:
            raise ValueError(
                f"{cls.format_name} format support quantization scheme with {','.join(cls.support_schemes)} "
                f"but got {', '.join(error_logs)}, please have a check."
            )
        return True

    def check_and_reset_format(self, ar):
        """Warns if ``iters != 0`` for GGUF and adjusts quant block list for MLLMs.

        Args:
            ar (BaseCompressor): Compressor instance.

        Returns:
            str | None: Replacement format name, or ``None``.
        """
        if ar.iters != 0 and ar.bits != 3 and not ar.enable_alg_ext:
            logger.warning_once(
                "`iters=0` is recommended when exporting to current GGUF format"
                " or add `enable_alg_ext` for better accuracy with much more tuning cost."
                " Please refer to https://github.com/intel/auto-round/tree/main/docs/gguf_alg_ext_acc.md"
                " for the accuracy results."
            )
        elif ar.bits >= 8 and ar.iters != 0:
            logger.warning_once("`iters=0` is recommended for bits>=8")

        if getattr(ar, "quant_nontext_module", False):
            # for gguf export, leave vl model for gguf itself
            all_blocks = get_block_names(ar.model, False)
            ar.quant_block_list = find_matching_blocks(ar.model, all_blocks, None)
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
        quant_nontext_module=False,
    ):
        """Packs a single layer into the GGUF output file.

        Args:
            name (str): Dot-separated layer name.
            model (torch.nn.Module): The model.
            backend (str): Backend format string (e.g. ``"gguf:q4_k_m"``).
            output_dir (str): Output directory for the GGUF file.
            layer_config (dict): Per-layer quantization configuration.
            tokenizer: Tokenizer instance.
            processor (optional): Multimodal processor.
            image_processor (optional): Image processor.
            model_type (ModelType): Text or MMPROJ model type.
            device (str, optional): Device string. Defaults to ``"cpu"``.
            quant_nontext_module (bool, optional): Whether to quantize non-text
                modules. Defaults to ``False``.
        """
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
            quant_nontext_module,
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
        """Saves the model in GGUF format.

        Args:
            output_dir (str): Destination directory.
            model (torch.nn.Module, optional): Model to serialize.
            tokenizer (Callable, optional): Tokenizer to save.
            layer_config (dict, optional): Per-layer quantization configuration.
            inplace (bool, optional): Operate in-place. Defaults to ``True``.
            device (str | torch.device, optional): Device. Defaults to ``"cpu"``.
            serialization_dict (dict, optional): Serialization metadata.
            **kwargs: Extra keyword arguments forwarded to the GGUF export function.

        Returns:
            torch.nn.Module: The serialized model.
        """
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
        """Validate GGUF format arguments and download required conversion files if needed.

        Checks that the requested GGUF sub-type is supported and that the model
        architecture has a corresponding GGUF export implementation.  Downloads
        the ``convert_hf_to_gguf`` dependency automatically when missing or when
        the model architecture is not yet supported.

        Args:
            args_or_ar (argparse.Namespace | BaseCompressor): CLI arguments or compressor
                instance providing the model path/object and platform.
            formats (str | list[str], optional): One or more format strings to validate
                (e.g. ``"gguf:q4_k_m"``). Defaults to ``None``.
            model_type (ModelType): Whether to validate for text or MMPROJ model.
                Defaults to ``ModelType.TEXT``.

        Raises:
            ImportError: If ``gguf-py`` is not installed or is out-of-date.
        """
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
        quant_nontext_module: bool = False,
        **kwargs,
    ):
        """Packs a single layer immediately to GGUF format if it should be quantized.

        Args:
            name (str): Dot-separated layer name within ``model``.
            model (torch.nn.Module): The full model.
            device (torch.device): Device for packing.
            output_dir (str, optional): Output directory for the GGUF file.
            mllm (bool, optional): Whether this is a multimodal LM projection.
            layer_config (dict, optional): Per-layer quantization configuration.
            tokenizer (optional): Tokenizer instance.
            processor (optional): Multimodal processor.
            image_processor (optional): Image processor.
            quant_nontext_module (bool, optional): Quantize non-text modules.
            **kwargs: Additional keyword arguments.
        """
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
            quant_nontext_module=quant_nontext_module,
        )


@OutputFormat.register("fp8")
class FP8Format(OutputFormat):
    """Output format for block-wise FP8 weight quantization (direct safetensors export).

    Supports the FP8_BLOCK scheme (8-bit floating-point with block-wise scaling).
    """

    support_schemes = ["FP8_BLOCK"]
    format_name = "fp8"

    @classmethod
    def check_scheme_args(cls: OutputFormat, scheme: QuantizationScheme) -> bool:
        """Validates that the scheme is compatible with block-wise FP8 packing.

        Args:
            scheme (QuantizationScheme): Scheme to validate.

        Returns:
            bool: ``True`` if valid.

        Raises:
            ValueError: If bits, data_type, group_size, or activation settings
                are incompatible with block FP8.
        """
        error_logs = []
        if scheme.bits != 8:
            error_logs.append(f"bits={scheme.bits}")
        if scheme.data_type != "fp":
            error_logs.append(f"data_type={scheme.data_type}")
        if not isinstance(scheme.group_size, tuple):
            error_logs.append(f"group_size={scheme.group_size}")
        if not scheme.act_dynamic:
            error_logs.append(f"act_dynamic={scheme.act_dynamic}")
        if not isinstance(scheme.act_group_size, int):
            error_logs.append(f"act_group_size={scheme.act_group_size}")
        if scheme.act_bits != 8:
            error_logs.append(f"act_bits={scheme.act_bits}")
        if scheme.act_data_type != "fp":
            error_logs.append(f"act_data_type={scheme.act_data_type}")
        if error_logs:
            raise ValueError(
                f"{cls.format_name} format support quantization scheme with {','.join(cls.support_schemes)} "
                f"but got {', '.join(error_logs)}, please have a check."
            )
        return True

    def pack_layer(self, layer_name, model, device=None, **kwargs):
        """Packs a single layer using FP8 block-wise quantization.

        Args:
            layer_name (str): Dot-separated layer name.
            model (torch.nn.Module): The model containing the layer.
            device (torch.device | None, optional): Device for packing.
            **kwargs: Additional keyword arguments.
        """
        from auto_round.export.export_to_autoround.export_to_fp8 import pack_layer

        pack_layer(layer_name, model, self.output_format, device=device)

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
        """Saves the model in FP8 block-wise format.

        Args:
            output_dir (str): Destination directory.
            model (torch.nn.Module, optional): Model to serialize.
            tokenizer (Callable, optional): Tokenizer to save.
            layer_config (dict, optional): Per-layer quantization configuration.
            inplace (bool, optional): Operate in-place. Defaults to ``True``.
            device (str | torch.device, optional): Device. Defaults to ``"cpu"``.
            serialization_dict (dict, optional): Serialization metadata.
            **kwargs: Extra keyword arguments forwarded to the export function.

        Returns:
            torch.nn.Module: The serialized model.
        """
        from auto_round.export.export_to_autoround.export_to_fp8 import save_quantized_as_autoround

        backend = self.get_backend_name()

        # weight_block_size & ignored_layers are required by fp8 format, skip them in auto_round:fp8 format
        if isinstance(serialization_dict["group_size"], tuple) and "auto_round" not in backend:
            serialization_dict["weight_block_size"] = serialization_dict["group_size"]

            ignored_layers = []
            for layer_name, cfg in layer_config.items():
                if cfg["bits"] >= 16 and cfg["act_bits"] >= 16:
                    ignored_layers.append(layer_name)
            if len(ignored_layers) > 0:
                serialization_dict["ignored_layers"] = ignored_layers

        return save_quantized_as_autoround(
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            layer_config=layer_config,
            inplace=inplace,
            device=device,
            serialization_dict=serialization_dict,
            quant_method=backend,
            **kwargs,
        )


@OutputFormat.register("auto_round")
@OutputFormat.register("auto_round:auto_awq")
@OutputFormat.register("auto_round:llm_compressor")
@OutputFormat.register("auto_round:gptqmodel", "auto_round:auto_gptq")
@OutputFormat.register("auto_round:fp8")
class AutoRoundFormat(OutputFormat):
    """Primary AutoRound output format with automatic backend selection.

    Chooses the optimal packing backend (AutoGPTQ, AutoAWQ, LLMCompressor, FP8, etc.)
    based on the quantization scheme.  Also handles compound format strings such as
    ``"auto_round:auto_gptq"`` to force a specific backend.
    """

    support_schemes = [
        "W4A16",
        "W4A16_MIXED",
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
        "FP8_BLOCK",
    ]
    format_name = "auto_round"

    def __init__(self, format: str, ar: BaseCompressor):
        """Initializes the AutoRound format, selecting the appropriate backend.

        The backend is chosen based on the quantization scheme:
        symmetric INT → AutoGPTQ; asymmetric W4 INT → AutoAWQ;
        NV-FP / MX-FP → NV/MX backend; static FP8 → FP8_STATIC backend;
        WOQ FP8 → FP8 backend; block FP8 → fp8 sub-format.

        Args:
            format (str): Format string (e.g. ``"auto_round"`` or a compound
                string such as ``"auto_round:auto_gptq"``).
            ar (BaseCompressor): Compressor instance supplying scheme and config.

        Raises:
            ValueError: If activation quantization is requested but the
                configuration is unsupported.
            KeyError: If an unknown backend sub-format is specified.
        """
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
            elif ar.data_type.startswith("fp") and ar.bits == 8 and isinstance(ar.group_size, tuple):
                self.backend = AutoRoundFormat("auto_round:fp8", ar)
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
        """Validates and adjusts the auto_round format for the given configuration.

        Args:
            ar (BaseCompressor): Compressor instance.

        Returns:
            None: Auto_round format never resets to another format at the top level.
        """
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
        """Packs a layer using the appropriate auto_round backend packing function.

        Args:
            layer_name (str): Dot-separated layer name.
            model (torch.nn.Module): The model containing the layer.
            device (torch.device | None, optional): Device for packing.
            **kwargs: Additional keyword arguments forwarded to the backend.
        """
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
        """Saves the model using the auto_round backend serialization.

        Args:
            output_dir (str): Destination directory.
            model (torch.nn.Module, optional): Model to serialize.
            tokenizer (Callable, optional): Tokenizer to save.
            layer_config (dict, optional): Per-layer quantization configuration.
            inplace (bool, optional): Operate in-place. Defaults to ``True``.
            device (str | torch.device, optional): Device. Defaults to ``"cpu"``.
            serialization_dict (dict, optional): Serialization metadata.
            **kwargs: Extra keyword arguments forwarded to the export function.

        Returns:
            torch.nn.Module: The serialized model.
        """
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
        elif serialization_dict.get("data_type", "int") == "fp" and serialization_dict.get("bits", 16) == 8:
            from auto_round.export.export_to_autoround.export_to_fp8 import save_quantized_as_autoround

            backend = "auto_round:fp8_static" if serialization_dict.get("act_bits", 16) == 8 else None
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
