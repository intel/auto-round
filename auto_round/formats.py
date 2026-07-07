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

from auto_round.export.export_to_gguf.config import ModelType
from auto_round.schemes import (
    PRESET_SCHEMES,
    QuantizationScheme,
    get_gguf_scheme,
)
from auto_round.utils import (
    INNER_SUPPORTED_LAYER_TYPES,
    SUPPORTED_FORMATS,
    SUPPORTED_LAYER_TYPES,
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
    INT8 = "int8_w8a8"
    FP8_BLOCK = "fp8_block"
    MXINT4 = "mxint4"
    MX_INT = "mx_int"
    WINT_A16 = "wint_a16"


if TYPE_CHECKING:
    from auto_round.compressors.base import BaseCompressor


def _check_compatibility(formats: list[str], scheme: QuantizationScheme):
    if (
        any(["gguf" in f.lower() for f in formats])
        and len([f for f in formats if f.lower() != "fake" and not f.lower().startswith("gguf")]) > 1
    ):
        raise ValueError(
            f"GGUF format is not compatible with other formats, but got {formats}, please choose only one of them"
        )
    gguf_format_name = get_gguf_scheme(scheme)
    if gguf_format_name:
        if gguf_format_name.lower().endswith("mixed"):
            gguf_format_name = gguf_format_name.lower().replace("_mixed", "_s")
        if any([f.lower() not in ["fake", gguf_format_name.lower()] for f in formats]):
            has_gguf_format = any(f.lower().startswith("gguf") for f in formats if f.lower() != "fake")
            if has_gguf_format:
                logger.warning(
                    f"scheme {gguf_format_name} is GGUF, but format {','.join(formats)} specifies "
                    f"a different GGUF type. The scheme-driven per-layer quantization may differ from the "
                    f"file-level GGUF format type."
                )
            else:
                tmp_format_name = (
                    gguf_format_name.lower() if "fake" not in formats else (f"{gguf_format_name.lower()},fake")
                )
                logger.warning(
                    f"reset format {','.join(formats)} to {tmp_format_name} "
                    f"since scheme {gguf_format_name} can only be exported to format "
                    f"{gguf_format_name.lower()} or fake"
                )
                formats = tmp_format_name.split(",")

    if isinstance(scheme.group_size, tuple) and any(["auto_round" in f.lower() for f in formats]):
        logger.warning(
            "auto_round:fp8 format only supports vLLM inference for now. "
            "We recommend using the FP8 format via `--format fp8` instead."
        )

    return formats


def get_formats(
    format: str,
    scheme: QuantizationScheme,
    *,
    model=None,
    layer_config: dict = None,
    scale_dtype=None,
    mllm: bool = False,
    iters: int = 0,
    enable_alg_ext: bool = False,
    quant_nontext_module: bool = False,
    quant_block_list: list = None,
    platform: str = None,
) -> tuple[list[OutputFormat], QuantizationScheme, dict, object, list]:
    """Get the list of OutputFormat instances based on the provided name.

    Returns ``(formats, scheme, layer_config, scale_dtype, quant_block_list)`` since format
    resolution may replace the scheme (GGUF preset correction), create ``layer_config`` if it
    was ``None``, force ``scale_dtype`` to ``torch.float32`` for GGUF, or narrow ``quant_block_list``.
    """

    def remove_duplicates(lst):
        seen = set()
        return [x for x in lst if not (x in seen or seen.add(x))]

    fwd_kwargs = dict(
        model=model,
        layer_config=layer_config,
        scale_dtype=scale_dtype,
        mllm=mllm,
        iters=iters,
        enable_alg_ext=enable_alg_ext,
        quant_nontext_module=quant_nontext_module,
        quant_block_list=quant_block_list,
        platform=platform,
    )

    formats = format.lower().replace("q*_", f"q{scheme.bits}_").replace(" ", "").split(",")
    formats = remove_duplicates(formats)  # need the keep origin order

    formats = _check_compatibility(formats, scheme)

    formats = remove_duplicates(formats)

    for fmt in formats:
        if fmt not in SUPPORTED_FORMATS:
            raise ValueError(f"{fmt} is not supported, we only support {SUPPORTED_FORMATS}")

    for i in range(len(formats)):
        if formats[i].startswith("gguf:"):
            formats[i], scheme, layer_config = GGUFFormat.build(formats[i], scheme, **fwd_kwargs)
        elif formats[i] not in OutputFormat._format_list:
            raise KeyError(f"Unsupported format {formats[i]}, please choose from {SUPPORTED_FORMATS}")
        else:
            fwd_kwargs["layer_config"] = layer_config
            formats[i] = OutputFormat._format_list[formats[i]](formats[i], scheme, **fwd_kwargs)

        fwd_kwargs["layer_config"] = layer_config
        new_format, scheme, layer_config, quant_block_list = formats[i].check_and_reset_format(
            scheme, **{**fwd_kwargs, "quant_block_list": quant_block_list}
        )
        fwd_kwargs["layer_config"] = layer_config
        fwd_kwargs["quant_block_list"] = quant_block_list
        if new_format is not None:
            if new_format not in format:
                formats[i] = OutputFormat._format_list[new_format](new_format, scheme, **fwd_kwargs)
            else:
                formats[i] = None

    formats = [fmt for fmt in formats if fmt is not None]

    # Ensure fake format is processed before GGUF — GGUF export may clear
    # model weights via low_cpu_mem_usage, causing zeroed weights for fake.
    if any(fmt.is_gguf() for fmt in formats) and any(fmt.is_fake() for fmt in formats):
        formats.sort(key=lambda f: 0 if f.is_fake() else 1)

    if len(formats) == 1 and formats[0].is_gguf() and scale_dtype != torch.float32:
        scale_dtype = torch.float32
        logger.info("change `scale_dtype` to `torch.float32` for gguf format")

    return formats, scheme, layer_config, scale_dtype, quant_block_list


def _check_divisible_by_32(scheme: QuantizationScheme, model, layer_config: dict) -> dict:
    default_dict = asdict(scheme)
    skipped_layers = []
    if default_dict["data_type"] == "int" and default_dict["act_bits"] >= 16:
        for n, m in model.named_modules():
            if type(m) in SUPPORTED_LAYER_TYPES or m.__class__.__name__ in INNER_SUPPORTED_LAYER_TYPES:
                if m.weight.shape[0] % 32 or m.weight.shape[1] % 32:
                    if layer_config is None:
                        layer_config = {}
                    if layer_config.get(n) is not None and layer_config[n]["bits"] >= 16:
                        continue
                    layer_config.setdefault(n, copy.deepcopy(default_dict))
                    layer_config[n].update({"bits": 16, "data_type": "fp", "fixed_by_user": True})
                    skipped_layers.append(n)
    compressed_skipped_layers = compress_layer_names(skipped_layers)
    if compressed_skipped_layers:
        logger.warning_once(
            f"some layers are skipped quantization (shape not divisible by 32): {compressed_skipped_layers}"
        )
    return layer_config


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

    def __init__(
        self,
        format: str,
        scheme: QuantizationScheme,
        *,
        model=None,
        layer_config: dict = None,
        scale_dtype=None,
        mllm: bool = False,
        iters: int = 0,
        enable_alg_ext: bool = False,
        quant_nontext_module: bool = False,
        quant_block_list: list = None,
        platform: str = None,
    ):
        """Initialize the OutputFormat class."""
        self.output_format = format
        self.backend = None
        self.mllm = mllm

        if not self.is_fake() and not self.is_support_scheme(scheme):
            logger.error(
                f"Currently, the {self.format_name} format only supports {self.support_schemes}, "
                f"but got scheme {scheme}, please change to fake or auto_round etc."
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
            return cls.check_scheme_args(scheme)
        return False

    @classmethod
    def check_scheme_args(cls: OutputFormat, scheme: QuantizationScheme) -> bool:
        return True

    def check_and_reset_format(
        self,
        scheme: QuantizationScheme,
        *,
        model=None,
        layer_config: dict = None,
        scale_dtype=None,
        mllm: bool = False,
        iters: int = 0,
        enable_alg_ext: bool = False,
        quant_nontext_module: bool = False,
        quant_block_list: list = None,
        platform: str = None,
    ) -> tuple[str, QuantizationScheme, dict, list]:
        fwd_kwargs = dict(
            model=model,
            layer_config=layer_config,
            scale_dtype=scale_dtype,
            mllm=mllm,
            iters=iters,
            enable_alg_ext=enable_alg_ext,
            quant_nontext_module=quant_nontext_module,
            quant_block_list=quant_block_list,
            platform=platform,
        )
        if self.backend is not None:
            new_format, scheme, layer_config, quant_block_list = self.backend.check_and_reset_format(
                scheme, **{**fwd_kwargs, "layer_config": layer_config, "quant_block_list": quant_block_list}
            )
            fwd_kwargs["layer_config"] = layer_config
            fwd_kwargs["quant_block_list"] = quant_block_list
            self.backend = (
                OutputFormat._format_list[new_format](new_format, scheme, **fwd_kwargs) if new_format else self.backend
            )

        w_fp8 = scheme.data_type.startswith("fp") and scheme.bits == 8
        act_fp8 = scheme.act_data_type.startswith("fp") and scheme.act_bits == 8
        is_block_dynamic_fp8 = (
            self.format_name in ["fp8", "auto_round:fp8"]
            and isinstance(scheme.group_size, tuple)
            and scheme.act_dynamic
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

        if (
            scheme.act_bits <= 8
            and (not scheme.is_act_standard_fp() or scheme.act_dynamic)
            and not is_block_dynamic_fp8
        ):
            logger.warning(
                f"{self.format_name} format does not support the current activation quantization configuration,"
                " reset to fake format and save."
            )
            return "fake", scheme, layer_config, quant_block_list

        return None, scheme, layer_config, quant_block_list

    @abstractmethod
    def pack_layer(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_quantized(self, *args, **kwargs):
        pass

    def immediate_pack(self, name: str, model: torch.nn.Module, device: torch.device, **kwargs):
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

    def check_and_reset_format(
        self, scheme: QuantizationScheme, **kwargs
    ) -> tuple[None, QuantizationScheme, dict, list]:
        return None, scheme, kwargs.get("layer_config"), kwargs.get("quant_block_list")

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
    support_schemes = [
        "MXFP4",
        "MXFP8",
        "NVFP4",
        "FPW8A16",
        "FP8_STATIC",
        "INT8",
        "INT8_W8A8",
        "FP8_BLOCK",
        "W4A16",
        "W8A16",
    ]
    format_name = "llm_compressor"

    def __init__(
        self,
        format: str,
        scheme: QuantizationScheme,
        *,
        model=None,
        layer_config: dict = None,
        scale_dtype=None,
        mllm: bool = False,
        iters: int = 0,
        enable_alg_ext: bool = False,
        quant_nontext_module: bool = False,
        quant_block_list: list = None,
        platform: str = None,
    ):
        fwd_kwargs = dict(
            model=model,
            layer_config=layer_config,
            scale_dtype=scale_dtype,
            mllm=mllm,
            iters=iters,
            enable_alg_ext=enable_alg_ext,
            quant_nontext_module=quant_nontext_module,
            quant_block_list=quant_block_list,
            platform=platform,
        )
        if not self.is_support_scheme(scheme):
            logger.error(
                f"Currently, the llm_compressor format only supports {self.support_schemes}, "
                f"but got scheme {scheme}, please change to fake or auto_round etc."
            )
            exit(-1)
        # if format.startswith("llm_compressor"):
        if re.search("^(auto_round:)?llm_compressor", format):
            self.output_format = format
            self.backend = None
            if scheme.is_nv_fp() or scheme.is_mx_fp():
                from auto_round.export.export_to_llmcompressor import check_compressed_tensors_supported

                check_compressed_tensors_supported(raise_error=True)
                self.backend = LLMCompressorFormat(scheme.data_type, scheme, **fwd_kwargs)
            elif scheme.is_dynamic_afp8() and scheme.is_block_wfp8():
                self.backend = LLMCompressorFormat(AutoRoundExportFormat.FP8_BLOCK.value, scheme, **fwd_kwargs)
            elif scheme.is_static_wfp8afp8():
                self.backend = LLMCompressorFormat(AutoRoundExportFormat.FP8_STATIC.value, scheme, **fwd_kwargs)
                if scheme.act_group_size != 0:
                    logger.warning(
                        f"scheme FP8_STATIC export to llm_compressor format only support for act_group_size 0,"
                        f" ,but got act_group_size={scheme.act_group_size}, reset = 0"
                    )
                    scheme.act_group_size = 0
                if scheme.group_size > 0:
                    logger.warning(
                        f"please note that group_size={scheme.group_size}"
                        " may not be supported for llm_compressor format, and cannot be loaded in llm_compressor"
                    )
            elif scheme.is_dynamic_wint8aint8():
                from auto_round.export.export_to_llmcompressor import check_compressed_tensors_supported

                check_compressed_tensors_supported()
                self.backend = LLMCompressorFormat(AutoRoundExportFormat.INT8.name, scheme, **fwd_kwargs)
                self.backend.output_format = f"llm_compressor:{AutoRoundExportFormat.INT8.value}"
            elif scheme.is_wint_woq():
                from auto_round.export.export_to_llmcompressor import check_compressed_tensors_supported

                check_compressed_tensors_supported()
                self.backend = LLMCompressorFormat(AutoRoundExportFormat.WINT_A16.value, scheme, **fwd_kwargs)
        else:
            if format.upper() not in list(AutoRoundExportFormat.__members__.keys()):
                raise KeyError(f"Unsupported backend format llm_compressor:{format}, please check")
            self.output_format = f"llm_compressor:{format}"
            self.backend = None

    @classmethod
    def check_scheme_args(cls: OutputFormat, scheme: QuantizationScheme) -> bool:
        error_logs = []
        if scheme.bits not in [4, 8, 16]:
            error_logs.append(f"bits={scheme.bits}")
        if not re.search("mxfp|fp|nvfp|int", scheme.data_type):
            error_logs.append(f"data_type={scheme.data_type}")
        if scheme.data_type == "fp" and scheme.bits != 8:
            error_logs.append(f"data_type={scheme.data_type}, bits={scheme.bits}")
        if scheme.data_type == "int" and scheme.bits not in [4, 8]:
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

    def check_and_reset_format(
        self,
        scheme: QuantizationScheme,
        *,
        model=None,
        layer_config: dict = None,
        scale_dtype=None,
        mllm: bool = False,
        iters: int = 0,
        enable_alg_ext: bool = False,
        quant_nontext_module: bool = False,
        quant_block_list: list = None,
        platform: str = None,
    ) -> tuple[str, QuantizationScheme, dict, list]:
        fwd_kwargs = dict(
            model=model,
            layer_config=layer_config,
            scale_dtype=scale_dtype,
            mllm=mllm,
            iters=iters,
            enable_alg_ext=enable_alg_ext,
            quant_nontext_module=quant_nontext_module,
            quant_block_list=quant_block_list,
            platform=platform,
        )
        if self.backend is not None:
            new_format, scheme, layer_config, quant_block_list = self.backend.check_and_reset_format(
                scheme, **{**fwd_kwargs, "layer_config": layer_config, "quant_block_list": quant_block_list}
            )
            fwd_kwargs["layer_config"] = layer_config
            fwd_kwargs["quant_block_list"] = quant_block_list
            self.backend = (
                OutputFormat._format_list[new_format](new_format, scheme, **fwd_kwargs) if new_format else self.backend
            )

        if scheme.act_bits <= 8 and (not scheme.is_act_standard_fp() or scheme.act_dynamic):
            if (scheme.is_act_nv_fp() and "static_gs" in scheme.act_data_type) or scheme.is_act_mx_fp():
                return None, scheme, layer_config, quant_block_list
            elif scheme.is_dynamic_afp8() and scheme.is_block_wfp8():
                return None, scheme, layer_config, quant_block_list
            else:
                bits, group_size, sym, act_bits = 8, -1, True, 8
                assert (
                    scheme.bits == bits
                    and scheme.group_size == group_size
                    and scheme.sym == sym
                    and scheme.act_bits == act_bits
                    and scheme.act_dynamic
                ), (
                    f"Currently only support to export llm_compressor format for sym dynamic quantized"
                    f" W{scheme.bits}A{scheme.act_bits} model with group_size={group_size},"
                    f" but got bits={scheme.bits}, group_size={scheme.group_size}, sym={scheme.sym},"
                    f" act_bits={scheme.act_bits}"
                )
            return None, scheme, layer_config, quant_block_list
        return None, scheme, layer_config, quant_block_list

    def pack_layer(self, layer_name, model, device=None, **kwargs):
        if self.backend is not None:
            return self.backend.pack_layer(layer_name, model, device=device, **kwargs)
        if re.search(f"{AutoRoundExportFormat.MX_FP.value}|{AutoRoundExportFormat.NV_FP.value}", self.output_format):
            from auto_round.export.export_to_llmcompressor.export_to_fp import pack_layer

            return pack_layer(layer_name, model, device=device)
        elif re.search(f"{AutoRoundExportFormat.FP8_STATIC.value}", self.output_format):
            from auto_round.export.export_to_llmcompressor.export_to_static_fp import pack_layer

            return pack_layer(layer_name, model, self.get_backend_name(), device=device)
        elif re.search(f"{AutoRoundExportFormat.INT8.value}", self.output_format):
            from auto_round.export.export_to_llmcompressor.export import pack_layer

            return pack_layer(layer_name, model, device=device)
        elif re.search(f"{AutoRoundExportFormat.FP8_BLOCK.value}", self.output_format):
            from auto_round.export.export_to_llmcompressor.export import pack_layer

            return pack_layer(layer_name, model, device=device)
        elif re.search(f"{AutoRoundExportFormat.WINT_A16.value}", self.output_format):
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
    support_schemes = ["W4A16", "W2A16", "W3A16", "W8A16", "BF16", "W2A16G64", "W2A16G32", "W4A16_MIXED"]
    format_name = "auto_gptq"

    def check_and_reset_format(
        self,
        scheme: QuantizationScheme,
        *,
        model=None,
        layer_config: dict = None,
        scale_dtype=None,
        mllm: bool = False,
        iters: int = 0,
        enable_alg_ext: bool = False,
        quant_nontext_module: bool = False,
        quant_block_list: list = None,
        platform: str = None,
    ):
        if not scheme.sym:
            logger.warning(
                "the asymmetrical kernel of the GPTQ format may result in a noticeable accuracy drop,"
                " particularly for 2-bit quantization and smaller models."
                " We recommend exporting to either the AutoAWQ format ( only 4 bits) or "
                "the AutoRound format(2/3/4/8 bits)."
            )
        if self.backend is None:
            layer_config = _check_divisible_by_32(scheme, model, layer_config)
        return super().check_and_reset_format(
            scheme,
            model=model,
            layer_config=layer_config,
            scale_dtype=scale_dtype,
            mllm=mllm,
            iters=iters,
            enable_alg_ext=enable_alg_ext,
            quant_nontext_module=quant_nontext_module,
            quant_block_list=quant_block_list,
            platform=platform,
        )

    @classmethod
    def check_scheme_args(cls: OutputFormat, scheme: QuantizationScheme) -> bool:
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

    @classmethod
    def check_scheme_args(cls: OutputFormat, scheme: QuantizationScheme) -> bool:
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

    def check_and_reset_format(
        self,
        scheme: QuantizationScheme,
        *,
        model=None,
        layer_config: dict = None,
        scale_dtype=None,
        mllm: bool = False,
        iters: int = 0,
        enable_alg_ext: bool = False,
        quant_nontext_module: bool = False,
        quant_block_list: list = None,
        platform: str = None,
    ):
        awq_supported, info = self.check_awq_gemm_compatibility(
            model, scheme.bits, scheme.group_size, scheme.sym, layer_config
        )
        if not awq_supported:
            logger.warning(f"The AutoAWQ format may not be supported due to {info}")
        if scheme.bits != 4:
            raise ValueError(f"auto_awq format support quantization scheme with W4A16 but got bits={scheme.bits}")

        if self.backend is None:
            layer_config = _check_divisible_by_32(scheme, model, layer_config)

        return super().check_and_reset_format(
            scheme,
            model=model,
            layer_config=layer_config,
            scale_dtype=scale_dtype,
            mllm=mllm,
            iters=iters,
            enable_alg_ext=enable_alg_ext,
            quant_nontext_module=quant_nontext_module,
            quant_block_list=quant_block_list,
            platform=platform,
        )

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

    def __init__(
        self,
        format: str,
        scheme: QuantizationScheme,
        *,
        model=None,
        layer_config: dict = None,
        scale_dtype=None,
        mllm: bool = False,
        iters: int = 0,
        enable_alg_ext: bool = False,
        quant_nontext_module: bool = False,
        quant_block_list: list = None,
        platform: str = None,
    ):
        fwd_kwargs = dict(
            model=model,
            layer_config=layer_config,
            scale_dtype=scale_dtype,
            mllm=mllm,
            iters=iters,
            enable_alg_ext=enable_alg_ext,
            quant_nontext_module=quant_nontext_module,
            quant_block_list=quant_block_list,
            platform=platform,
        )
        if format.startswith("gguf:"):
            self._original_format = format  # preserve "gguf:q2_k_mixed" etc. for Phase 2b
            self.output_format = "gguf"
            self.backend_cls = GGUFFormat
            self.backend, scheme, layer_config = GGUFFormat.build(format.split(":")[-1], scheme, **fwd_kwargs)

            resolved_format = self.backend.output_format
            scheme = self.gguf_args_check(scheme, model, platform, resolved_format, model_type=ModelType.TEXT)
            if mllm:
                scheme = self.gguf_args_check(scheme, model, platform, resolved_format, model_type=ModelType.MMPROJ)
        else:
            gguf_format = f"gguf:{format.lower()}"
            if format.lower().endswith("_mixed"):
                from auto_round.schemes import _handle_special_schemes
                from auto_round.utils.model import is_moe_model

                if format.lower() == "q2_k_mixed" and (iters or 0) > 0 and not is_moe_model(model):
                    logger.warning(
                        "gguf:q2_k_mixed only supports MoE models with iters>0. "
                        "It is not an MoE model, falling back to gguf:q2_k_s."
                    )
                    gguf_format = "gguf:q2_k_s"
                else:
                    layer_config = _handle_special_schemes(
                        gguf_format, layer_config, model, quant_nontext_module=quant_nontext_module
                    )
                    gguf_format = gguf_format.lower().replace("_mixed", "_s")
            if isinstance(scheme, str) and scheme.lower() != gguf_format:
                # Defensive legacy branch: only reachable if a caller constructs OutputFormat
                # objects before scheme resolution (scheme still a raw string). No call site in
                # this codebase does this today (compressors/base.py's _resolve_formats documents
                # scheme as already-resolved to a QuantizationScheme by this point) — preserved
                # verbatim from the pre-refactor code in case an external caller relies on it.
                # NOTE: main's is_auto_scheme guard (added for GGUF+AutoScheme accuracy, #1960) is
                # moot here: resolve_scheme() (Phase 1) always builds self.scheme_context via
                # QuantizationScheme.from_dict(...) before _resolve_format_string() ever calls
                # get_formats(), so `scheme` is never a str on the live call path regardless of
                # is_auto_scheme; there is no `ar` in scope to guard with post-decoupling anyway.
                logger.warning(f"reset scheme {scheme.lower()} to {gguf_format} for gguf format export")
                scheme = gguf_format
            self.output_format = gguf_format
            self.backend = None
        self.mllm = mllm
        self._resolved_layer_config = layer_config
        self._resolved_scheme = scheme

    @classmethod
    def build(
        cls,
        format: str,
        scheme: QuantizationScheme,
        **kwargs,
    ) -> tuple["GGUFFormat", QuantizationScheme, dict]:
        """Construct a GGUFFormat and surface the (possibly corrected) scheme/layer_config.

        Plain ``__init__`` cannot report a corrected ``scheme``/``layer_config`` back to its
        caller, but GGUF resolution can replace either (the ``_mixed`` -> ``_s`` rewrite, and the
        legacy string-scheme correction above). ``get_formats`` calls this instead of the bare
        constructor for every ``gguf:``-prefixed dispatch.
        """
        instance = cls(format, scheme, **kwargs)
        return instance, instance._resolved_scheme, instance._resolved_layer_config

    @classmethod
    def check_scheme_args(cls: OutputFormat, scheme: QuantizationScheme) -> bool:
        error_logs = []
        if not re.search("int", scheme.data_type):
            error_logs.append(f"data_type={scheme.data_type}")
        if error_logs:
            raise ValueError(
                f"{cls.format_name} format support quantization scheme with {','.join(cls.support_schemes)} "
                f"but got {', '.join(error_logs)}, please have a check."
            )
        return True

    def check_and_reset_format(
        self,
        scheme: QuantizationScheme,
        *,
        model=None,
        layer_config: dict = None,
        scale_dtype=None,
        mllm: bool = False,
        iters: int = 0,
        enable_alg_ext: bool = False,
        quant_nontext_module: bool = False,
        quant_block_list: list = None,
        platform: str = None,
    ):
        if iters != 0 and scheme.bits != 3 and not enable_alg_ext:
            logger.warning_once(
                "`iters=0` is recommended when exporting to current GGUF format"
                " or add `enable_alg_ext` for better accuracy with much more tuning cost."
                " Please refer to https://github.com/intel/auto-round/tree/main/docs/gguf_alg_ext_acc.md"
                " for the accuracy results."
            )
        elif scheme.bits >= 8 and iters != 0:
            logger.warning_once("`iters=0` is recommended for bits>=8")

        if quant_nontext_module:
            # for gguf export, leave vl model for gguf itself
            all_blocks = get_block_names(model, False)
            quant_block_list = find_matching_blocks(model, all_blocks, None)
        return super().check_and_reset_format(
            scheme,
            model=model,
            layer_config=layer_config,
            scale_dtype=scale_dtype,
            mllm=mllm,
            iters=iters,
            enable_alg_ext=enable_alg_ext,
            quant_nontext_module=quant_nontext_module,
            quant_block_list=quant_block_list,
            platform=platform,
        )

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
    def gguf_args_check(
        scheme: QuantizationScheme,
        model,
        platform: str,
        formats: Union[str, list[str]] = None,
        model_type=ModelType.TEXT,
    ) -> QuantizationScheme:
        import argparse

        from auto_round.export.export_to_gguf.config import GGUF_CONFIG
        from auto_round.export.export_to_gguf.llama_cpp_conversion import get_conversion
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

        if export_gguf:
            if isinstance(model, str):
                model_path = model
            else:
                model_path = model.name_or_path
            if not os.path.isdir(model_path):
                model_path = download_or_get_path(model_path, platform)
            conversion = get_conversion(model_path, model_type=ModelType.TEXT)
            model_architecture = get_gguf_architecture(model_path, model_type=ModelType.TEXT)
            if not conversion.is_supported(model_architecture, ModelType.TEXT):
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
                    if not hasattr(scheme, k):
                        continue
                    if k == "data_type":
                        if re.search(r"q\d_1", format) and len(formats) > 1:
                            v = "int"
                    # `scheme` is never an argparse.Namespace (no caller in this codebase passes
                    # one — confirmed via repo-wide grep); preserved verbatim for parity with the
                    # pre-refactor dual-mode contract.
                    if k == "sym" and isinstance(scheme, argparse.Namespace):
                        k = "asym"
                        v = not v
                    if getattr(scheme, k) != v:
                        unsupported_list.append(f"{k}={getattr(scheme, k)}")
                        reset_list.append(f"{k}={v}")
                        setattr(scheme, k, v)
                if len(unsupported_list) > 0:
                    logger.info(
                        f"format {format} does not support for {', '.join(unsupported_list)},"
                        f" reset to {', '.join(reset_list)}."
                    )
        return scheme

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
    support_schemes = ["FP8_BLOCK"]
    format_name = "fp8"

    @classmethod
    def check_scheme_args(cls: OutputFormat, scheme: QuantizationScheme) -> bool:
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
        from auto_round.export.export_to_autoround.export_to_fp8 import save_quantized_as_autoround

        backend = self.get_backend_name()

        if isinstance(serialization_dict["group_size"], tuple):
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


@OutputFormat.register("mlx")
class MLXFormat(OutputFormat):
    support_schemes = ["W2A16", "W2A16G32", "W2A16G64", "W3A16", "W4A16", "W5A16", "W6A16", "W8A16", "BF16"]
    format_name = "mlx"

    @classmethod
    def check_scheme_args(cls: OutputFormat, scheme: QuantizationScheme) -> bool:
        error_logs = []
        if scheme.bits not in [2, 3, 4, 5, 6, 8, 16]:
            error_logs.append(f"bits={scheme.bits}")
        if scheme.act_bits != 16:
            error_logs.append(f"act_bits={scheme.act_bits}")
        if not re.search("int", scheme.data_type):
            error_logs.append(f"data_type={scheme.data_type}")
        if scheme.super_bits:
            error_logs.append(f"super_bits={scheme.super_bits}")
        if scheme.super_group_size:
            error_logs.append(f"super_group_size={scheme.super_group_size}")
        if error_logs:
            raise ValueError(
                f"MLX format support quantization scheme with {','.join(cls.support_schemes)} "
                f"but got {', '.join(error_logs)}, please have a check."
            )
        return True

    def pack_layer(self, layer_name, model, device=None, **kwargs):
        from auto_round.export.export_to_mlx.export import pack_layer

        pack_layer(layer_name, model, device=device, **kwargs)

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
        from auto_round.export.export_to_mlx.export import save_quantized_as_mlx

        return save_quantized_as_mlx(
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            layer_config=layer_config,
            inplace=inplace,
            device=device,
            serialization_dict=serialization_dict,
            **kwargs,
        )


@OutputFormat.register("auto_round")
@OutputFormat.register("auto_round:auto_awq")
@OutputFormat.register("auto_round:llm_compressor")
@OutputFormat.register("auto_round:gptqmodel", "auto_round:auto_gptq")
@OutputFormat.register("auto_round:fp8")
@OutputFormat.register("auto_round:mlx")
class AutoRoundFormat(OutputFormat):
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
        "MXINT4",
    ]
    format_name = "auto_round"

    def __init__(
        self,
        format: str,
        scheme: QuantizationScheme,
        *,
        model=None,
        layer_config: dict = None,
        scale_dtype=None,
        mllm: bool = False,
        iters: int = 0,
        enable_alg_ext: bool = False,
        quant_nontext_module: bool = False,
        quant_block_list: list = None,
        platform: str = None,
    ):
        fwd_kwargs = dict(
            model=model,
            layer_config=layer_config,
            scale_dtype=scale_dtype,
            mllm=mllm,
            iters=iters,
            enable_alg_ext=enable_alg_ext,
            quant_nontext_module=quant_nontext_module,
            quant_block_list=quant_block_list,
            platform=platform,
        )
        self.output_format = "auto_round"
        self.backend = None

        if format == "auto_round":
            if scheme.sym and "int" in scheme.data_type and "mx" not in scheme.data_type:
                self.backend = AutoGPTQFormat("auto_round:auto_gptq", scheme, **fwd_kwargs)
            elif scheme.bits == 4 and not scheme.sym and "int" in scheme.data_type:
                if layer_config is None:
                    enable_awq = True
                else:
                    enable_awq = all(
                        config["bits"] == scheme.bits or config["bits"] >= 16 for config in layer_config.values()
                    )
                if enable_awq:
                    self.backend = AutoAWQFormat("auto_round:auto_awq", scheme, **fwd_kwargs)
            elif scheme.is_nv_fp() or scheme.is_mx_fp():
                self.backend = AutoRoundFormat(scheme.data_type, scheme, **fwd_kwargs)
            elif scheme.is_mx_int() and scheme.bits == 4:  # only add mx_int4 now
                self.backend = AutoRoundFormat(scheme.data_type, scheme, **fwd_kwargs)
            elif scheme.is_static_wfp8afp8():  # static wfp8afp8
                self.backend = AutoRoundFormat(AutoRoundExportFormat.FP8_STATIC.value, scheme, **fwd_kwargs)
            elif scheme.data_type.startswith("fp") and scheme.bits == 8 and scheme.act_bits >= 16:  # woq fp8
                self.backend = AutoRoundFormat(AutoRoundExportFormat.FP8.value, scheme, **fwd_kwargs)
            elif scheme.data_type.startswith("fp") and scheme.bits == 8 and isinstance(scheme.group_size, tuple):
                self.backend = AutoRoundFormat("auto_round:fp8", scheme, **fwd_kwargs)
            elif scheme.act_bits < 16:
                raise ValueError(
                    "AutoRound format does not support exporting "
                    "for the current quantization configuration, "
                    "please change to `fake` format for research purpose"
                )
        # for auto_round:fp8_static, auto_round:nv_fp etc.
        elif not format.startswith("auto_round"):
            if format == "mlx":
                self.backend = MLXFormat("mlx", scheme, **fwd_kwargs)
            elif format.upper() not in list(AutoRoundExportFormat.__members__.keys()):
                raise KeyError(f"Unsupported backend format auto_round:{format}, please check")
            else:
                self.output_format = f"auto_round:{format}"
                self.backend = None
        elif format == "auto_round:mlx":
            self.backend = MLXFormat("mlx", scheme, **fwd_kwargs)
        else:
            backend = format.split(":")[1] if ":" in format else None
            self.backend = self._format_list.get(backend)(format, scheme, **fwd_kwargs) if backend else None

        if self.backend is not None:
            self.support_schemes = self.backend.support_schemes

    def check_and_reset_format(
        self,
        scheme: QuantizationScheme,
        *,
        model=None,
        layer_config: dict = None,
        scale_dtype=None,
        mllm: bool = False,
        iters: int = 0,
        enable_alg_ext: bool = False,
        quant_nontext_module: bool = False,
        quant_block_list: list = None,
        platform: str = None,
    ):
        fwd_kwargs = dict(
            model=model,
            layer_config=layer_config,
            scale_dtype=scale_dtype,
            mllm=mllm,
            iters=iters,
            enable_alg_ext=enable_alg_ext,
            quant_nontext_module=quant_nontext_module,
            quant_block_list=quant_block_list,
            platform=platform,
        )
        if self.backend is not None:
            new_format, scheme, layer_config, quant_block_list = self.backend.check_and_reset_format(
                scheme, **{**fwd_kwargs, "layer_config": layer_config, "quant_block_list": quant_block_list}
            )
            fwd_kwargs["layer_config"] = layer_config
            fwd_kwargs["quant_block_list"] = quant_block_list
            self.backend = (
                OutputFormat._format_list[new_format](new_format, scheme, **fwd_kwargs) if new_format else self.backend
            )

        if scheme.act_bits <= 8:
            if scheme.is_act_standard_fp() and not scheme.act_dynamic:
                if (
                    scheme.act_group_size != 0
                    and not scheme.act_dynamic
                    and self.get_backend_name() == f"auto_round:{AutoRoundExportFormat.FP8.value}"
                ):
                    logger.warning(
                        f"Please note that quantize activation with act_group_size={scheme.act_group_size}"
                        " may result in failure to export or import normally."
                    )
        if self.backend is None:
            layer_config = _check_divisible_by_32(scheme, model, layer_config)
        return None, scheme, layer_config, quant_block_list

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
            from auto_round.export.export_to_autoround.export_to_nvfp_mx import pack_layer

            pack_func = pack_layer
        elif self.output_format in [f"auto_round:{AutoRoundExportFormat.MX_INT.value}"]:
            from auto_round.export.export_to_autoround.export_to_nvfp_mx import pack_layer

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
            extra_kwargs = {}
            if isinstance(self.backend, MLXFormat):
                extra_kwargs["autoround_format"] = True
            return self.backend.save_quantized(
                output_dir=output_dir,
                model=model,
                tokenizer=tokenizer,
                layer_config=layer_config,
                inplace=inplace,
                device=device,
                serialization_dict=serialization_dict,
                **extra_kwargs,
                **kwargs,
            )
        backend = self.get_backend_name()
        if re.search(f"{AutoRoundExportFormat.MX_FP.value}|{AutoRoundExportFormat.NV_FP.value}", backend):
            from auto_round.export.export_to_autoround.export_to_nvfp_mx import save_quantized_as_fp

            backend = "auto_round:llm_compressor"
            export_func = save_quantized_as_fp
        elif serialization_dict.get("data_type", "int") == "fp" and serialization_dict.get("bits", 16) == 8:
            from auto_round.export.export_to_autoround.export_to_fp8 import save_quantized_as_autoround

            backend = "auto_round:fp8_static" if serialization_dict.get("act_bits", 16) == 8 else None
            export_func = save_quantized_as_autoround
        elif re.search(f"{AutoRoundExportFormat.MX_INT.value}", backend):
            from auto_round.export.export_to_autoround.export_to_nvfp_mx import save_quantized_as_fp

            backend = "auto_round"
            export_func = save_quantized_as_fp
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
