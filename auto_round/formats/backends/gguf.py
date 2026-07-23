# Copyright (c) 2026 Intel Corporation
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

import copy
import os
import re
from dataclasses import fields
from typing import Any, Callable, Union

import torch
import transformers

from auto_round.export.export_to_gguf.config import GGML_QUANT_SIZES, GGUF_CONFIG, GGUF_INNER_CONFIG, QK_K, ModelType
from auto_round.export.export_to_gguf.gguf_dtype import GGUFDTypeSelector, gguf_format_to_ftype
from auto_round.formats.base import OutputFormat
from auto_round.logger import logger
from auto_round.planning.contracts import thaw_mapping
from auto_round.planning.errors import FormatCompatibilityError
from auto_round.schemes import QuantizationScheme
from auto_round.utils import check_to_quantized, find_matching_blocks, get_block_names, get_module


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

    def __init__(self, format: str, scheme: QuantizationScheme, ctx: Any):
        self.is_auto_scheme = ctx.is_auto_scheme
        layer_config = ctx.layer_config
        if format.startswith("gguf:"):
            self._original_format = format  # preserve "gguf:q2_k_mixed" etc. for Phase 2b
            self.output_format = "gguf"
            self.backend_cls = GGUFFormat
            self.backend, scheme, layer_config = GGUFFormat.build(format.split(":")[-1], scheme, ctx)
        else:
            gguf_format = f"gguf:{format.lower()}"
            if format.lower().endswith("_mixed"):
                from auto_round.schemes import _handle_special_schemes
                from auto_round.utils.model import is_moe_model

                if format.lower() == "q2_k_mixed" and (ctx.iters or 0) > 0 and not is_moe_model(ctx.model):
                    logger.warning(
                        "gguf:q2_k_mixed only supports MoE models with iters>0. "
                        "It is not an MoE model, falling back to gguf:q2_k_s."
                    )
                    gguf_format = "gguf:q2_k_s"
                else:
                    layer_config = _handle_special_schemes(
                        gguf_format, layer_config, ctx.model, quant_nontext_module=ctx.quant_nontext_module
                    )
                    gguf_format = gguf_format.lower().replace("_mixed", "_s")
            if isinstance(scheme, str) and scheme.lower() != gguf_format:
                # Defensive: no live call site passes a raw string scheme here today, but this
                # is kept for external callers that construct OutputFormat before scheme resolution.
                logger.warning(f"reset scheme {scheme.lower()} to {gguf_format} for gguf format export")
                scheme = gguf_format
            self.output_format = gguf_format
            self.backend = None
            from auto_round.schemes import GGUF_PRESET_ALIASES, GGUF_SCHEME_FACTS

            facts_name = GGUF_PRESET_ALIASES.get(gguf_format)
            if facts_name is not None:
                scheme = copy.deepcopy(scheme)
                for key, value in GGUF_SCHEME_FACTS[facts_name].items():
                    setattr(scheme, key, value)
        self.mllm = ctx.mllm
        self._resolved_layer_config = layer_config
        self._resolved_scheme = scheme

    @classmethod
    def build(
        cls,
        format: str,
        scheme: QuantizationScheme,
        ctx: Any,
    ) -> tuple["GGUFFormat", QuantizationScheme, dict]:
        """Construct a GGUFFormat and surface the (possibly corrected) scheme/layer_config.

        Plain ``__init__`` cannot report a corrected ``scheme``/``layer_config`` back to its
        caller, but GGUF resolution can replace either (the ``_mixed`` -> ``_s`` rewrite, and the
        legacy string-scheme correction above). ``resolve_formats`` calls this instead of the bare
        constructor for every ``gguf:``-prefixed dispatch.
        """
        instance = cls(format, scheme, ctx)
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

    def check_and_reset_format(self, scheme: QuantizationScheme, ctx: Any):
        if ctx.iters != 0 and scheme.bits != 3 and not ctx.enable_alg_ext:
            logger.warning_once(
                "`iters=0` is recommended when exporting to current GGUF format"
                " or add `enable_alg_ext` for better accuracy with much more tuning cost."
                " Please refer to https://github.com/intel/auto-round/tree/main/docs/gguf_alg_ext_acc.md"
                " for the accuracy results."
            )
        elif scheme.bits >= 8 and ctx.iters != 0:
            logger.warning_once("`iters=0` is recommended for bits>=8")

        if ctx.quant_nontext_module:
            # for gguf export, leave vl model for gguf itself
            all_blocks = get_block_names(ctx.model, False)
            ctx.quant_block_list = find_matching_blocks(ctx.model, all_blocks, None)
        return super().check_and_reset_format(scheme, ctx)

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
            is_auto_scheme=self.is_auto_scheme,
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
            is_auto_scheme=self.is_auto_scheme,
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
            try:
                conversion = get_conversion(model_path, model_type=ModelType.TEXT)
            except AttributeError as e:
                print(f"[gguf conversion] error: {e}")
                raise ImportError(
                    "Please use the latest gguf-py, you can use the following command to install it:\n"
                    "git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp/gguf-py"
                    " && pip install . sentencepiece"
                )
            model_architecture = get_gguf_architecture(model_path, model_type=ModelType.TEXT)
            if not conversion.is_supported(model_architecture, ModelType.TEXT):
                raise FormatCompatibilityError(f"Model {model_architecture} is not supported to export gguf format.")

        pattern = re.compile(r"q\d_k")
        pre_dq_format = ""
        unsupported_list, reset_list = [], []
        for format in GGUF_CONFIG:
            if format in formats:
                if format == "q6_k_s":
                    logger.warning("Please note that q6_k_s is q6_k.")

                if re.search(pattern, format):
                    if pre_dq_format and re.search(pattern, format).group() not in pre_dq_format:
                        raise FormatCompatibilityError(f"Cannot export {pre_dq_format} and {format} at the same time.")
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
                    # No caller passes an argparse.Namespace here; kept for dual-mode parity.
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


def apply_gguf_layer_defaults(
    layer_config,
    model,
    gguf_name,
    lm_head_name,
    tie_word_embeddings,
    embedding_layer_names,
    default_scale_dtype,
    enable_gguf_official_mixed,
    is_mllm,
    has_qlayer_outside_block,
) -> tuple[dict, bool]:
    """Step 10 of layer-config resolution: embed/lm_head GGUF defaults plus the
    GGUF-format-specific per-layer mapping. Only called when `gguf_name` is truthy."""
    from auto_round.utils.model import is_separate_lm_head

    # embed + lm_head defaults for gguf
    tie_word_embeddings &= not is_separate_lm_head(model)
    if lm_head_name not in layer_config and not tie_word_embeddings:
        cfg = GGUF_INNER_CONFIG[GGUF_CONFIG[gguf_name.lower()]["lm_head"]]
        cfg = {**cfg, "fixed_by_user": False, "scale_dtype": default_scale_dtype}
        layer_config[lm_head_name] = cfg
        has_qlayer_outside_block = True
    for emd_name in embedding_layer_names:
        if emd_name in layer_config:
            continue
        if not tie_word_embeddings:
            cfg = GGUF_INNER_CONFIG[GGUF_CONFIG[gguf_name.lower()]["embedding"]]
        else:
            cfg = GGUF_INNER_CONFIG[GGUF_CONFIG[gguf_name.lower()]["lm_head"]]
        cfg = {**cfg, "fixed_by_user": False, "scale_dtype": default_scale_dtype}
        layer_config[emd_name] = cfg

    if enable_gguf_official_mixed:
        model_type = ModelType.MMPROJ if is_mllm else ModelType.TEXT
        layer_config, _ = get_layer_config_by_gguf_format(layer_config, gguf_name.lower(), model, model_type)

    _apply_gguf_shape_fallback(layer_config, model)
    return layer_config, has_qlayer_outside_block


def _use_more_bits(i_layer: int, n_layer: int):
    return (i_layer < n_layer // 8) or (i_layer >= 7 * n_layer // 8) or ((i_layer - n_layer // 8) % 3 == 2)


def _search_gguf_type(gguf_type):
    if gguf_type in GGUF_INNER_CONFIG:
        return gguf_type
    pattern = re.compile("gguf:q([0-9]{1,})_[01k]")
    bits = re.search(pattern, gguf_type)
    if not bits:
        raise KeyError(f"{gguf_type} is not a correct gguf type, please check")

    for suffix in ["_k", "_0", "_1"]:
        if gguf_type.endswith(suffix):
            continue
        if (tmp_type := re.sub("_[01k]", suffix, gguf_type)) in GGUF_INNER_CONFIG:
            return tmp_type
    return None


def gguf_type_fallback(gguf_type: str) -> str:
    gguf_type = gguf_type.lower()
    if gguf_type in ("gguf:q2_k", "gguf:q3_k", "gguf:q4_k"):
        gguf_type = "gguf:q5_0"
    elif gguf_type == "gguf:q5_k":
        gguf_type = "gguf:q5_0"
    elif gguf_type == "gguf:q6_k":
        gguf_type = "gguf:q8_0"
    return gguf_type


def get_gguf_qtype_by_layer_config(layer_config):
    import gguf  # pylint: disable=E0401

    if layer_config["bits"] >= 16:
        return None
    bits = layer_config["bits"]
    super_bits = layer_config.get("super_bits", None)
    sym = layer_config["sym"]
    group_size = layer_config.get("group_size", None)
    super_group_size = layer_config.get("super_group_size", None)
    if bits == 2 and super_bits == 4 and not sym and group_size == 16 and super_group_size == 16:
        return gguf.GGMLQuantizationType.Q2_K
    if bits == 3 and super_bits == 6 and sym and group_size == 16 and super_group_size == 16:
        return gguf.GGMLQuantizationType.Q3_K
    if bits == 4:
        if super_bits is not None and super_bits == 6 and not sym and group_size == 32 and super_group_size == 8:
            return gguf.GGMLQuantizationType.Q4_K
        if super_bits is None and sym and group_size == 32:
            return gguf.GGMLQuantizationType.Q4_0
        if super_bits is None and not sym and group_size == 32:
            return gguf.GGMLQuantizationType.Q4_1
    if bits == 5:
        if super_bits == 6 and not sym and group_size == 32 and super_group_size == 8:
            return gguf.GGMLQuantizationType.Q5_K
        if super_bits is None and sym and group_size == 32:
            return gguf.GGMLQuantizationType.Q5_0
        if super_bits is None and not sym and group_size == 32:
            return gguf.GGMLQuantizationType.Q5_1
    if bits == 6 and super_bits == 8 and group_size == 16 and super_group_size == 16:
        return gguf.GGMLQuantizationType.Q6_K
    if bits == 8 and sym and group_size == 32:
        return gguf.GGMLQuantizationType.Q8_0
    raise ValueError("Unknown layer config")


def _apply_gguf_shape_fallback(layer_config, model):
    from auto_round.utils.model import get_module

    for layer_name, config in layer_config.items():
        if not check_to_quantized(config):
            continue
        layer = get_module(model, layer_name)
        if layer is None or not hasattr(layer, "weight"):
            continue
        try:
            qtype = get_gguf_qtype_by_layer_config(config)
        except ValueError:
            continue
        if qtype is None:
            continue

        gguf_type = f"gguf:{qtype.name.lower()}"
        block_size = GGML_QUANT_SIZES[gguf_type.split(":")[-1]][0]
        input_features = (
            layer.weight.shape[0] if type(layer) == transformers.pytorch_utils.Conv1D else layer.weight.shape[-1]
        )
        if input_features % block_size == 0:
            continue

        fallback_type = _gguf_type_fallback(gguf_type)
        fallback_block_size = GGML_QUANT_SIZES[fallback_type.split(":")[-1]][0]
        if input_features % fallback_block_size != 0:
            fallback_type = "gguf:bf16"

        preserved = {key: config[key] for key in ("fixed_by_user", "scale_dtype") if key in config}
        config.update(GGUF_INNER_CONFIG[fallback_type])
        config.update(preserved)
        logger.warning(
            "fallback %s to %s before quantization, because input_features(%s) is not divisible by %s block_size(%s)",
            layer_name,
            fallback_type,
            input_features,
            gguf_type,
            block_size,
        )


def _get_digital_in_layer_name(layer_name):
    pattern = re.compile(r"([a-zA-Z]+\.){1,}(\d+)")
    res = re.search(pattern, layer_name)
    if res:
        return int(res[2])
    else:
        return None


def _gguf_type_fallback(gguf_type: str) -> str:
    gguf_type = gguf_type.lower()
    if gguf_type in ("gguf:q2_k", "gguf:q3_k", "gguf:q4_k"):
        gguf_type = "gguf:q5_0"
    elif gguf_type == "gguf:q5_k":
        gguf_type = "gguf:q5_0"
    elif gguf_type == "gguf:q6_k":
        gguf_type = "gguf:q8_0"
    return gguf_type


def _select_gguf_layer_type(
    layer_name,
    config,
    layer,
    model,
    model_class,
    *,
    target_gguf_format,
    lm_head_name,
    tie_word_embeddings,
    block_size,
    base_target_bits,
    dtype_selector,
    gguf_name,
    i_layer,
) -> str | None:
    """Per-layer GGUF-type-selection cascade extracted from
    `get_layer_config_by_gguf_format`'s loop body.

    Returns the selected gguf type string, or ``None`` as the sentinel for the
    `lm_head_name == layer_name and tie_word_embeddings` skip case, which the
    caller must handle with `continue`.
    """
    import gguf  # pylint: disable=E0401

    from auto_round.schemes import QuantizationScheme, get_gguf_scheme

    if type(layer) == transformers.pytorch_utils.Conv1D:
        input_features = layer.weight.shape[0]
    else:
        input_features = layer.weight.shape[-1]

    # Reset target_bits each iteration to prevent lm_head/embedding settings
    # from bleeding into subsequent block layers and bypassing their special logic.
    target_bits = base_target_bits
    new_type = GGUF_CONFIG[target_gguf_format]["mostly"]

    if lm_head_name is not None and layer_name == lm_head_name:
        target_bits = int(re.search("gguf:q([0-9]{1,})_[01k]", GGUF_CONFIG[target_gguf_format]["lm_head"]).group(1))
    if isinstance(layer, torch.nn.Embedding):
        embedding_format_key = "lm_head" if tie_word_embeddings else "embedding"
        target_bits = int(
            re.search("gguf:q([0-9]{1,})_[01k]", GGUF_CONFIG[target_gguf_format][embedding_format_key]).group(1)
        )

    bits_index = 6
    if config.get("fixed_by_user", False):
        if "bits" not in config:
            logger.warning(
                f"Setting layer_config requires providing bits, {layer_name} has not bits,"
                f" using bits={target_bits} instead."
            )
            new_type = new_type[:bits_index] + str(target_bits) + new_type[bits_index + 1 :]
        else:
            config_tmp = config.copy()
            scheme_keys = [f.name for f in fields(QuantizationScheme)]
            for key in config.keys():
                if key not in scheme_keys:
                    config_tmp.pop(key, None)
            matched_scheme = get_gguf_scheme(QuantizationScheme.from_dict(config_tmp))  # check matched
            if not matched_scheme:
                if config.get("super_group_size", None) is not None or config.get("super_bits", None) is not None:
                    new_type = new_type[:bits_index] + str(config["bits"]) + "_k"
                if new_type not in GGUF_INNER_CONFIG:
                    prefix_idx = 0 if config.get("sym", True) else 1
                    new_type = new_type[:bits_index] + str(config["bits"]) + f"_{prefix_idx}"
                    if new_type not in GGUF_INNER_CONFIG:
                        new_type = new_type[:bits_index] + str(config["bits"]) + f"_{1-prefix_idx}"
                if new_type not in GGUF_INNER_CONFIG:
                    raise ValueError(
                        f"the setting in layer_config {layer_name} "
                        f"could not match any supported gguf format, please have a check."
                    )

            new_type = new_type[:bits_index] + str(config["bits"]) + new_type[bits_index + 1 :]
        new_type = _search_gguf_type(new_type)
        if new_type is None:
            raise ValueError(f"invalid bit setting for {layer_name}")
    elif lm_head_name is not None and layer_name == lm_head_name and not tie_word_embeddings:
        if gguf.MODEL_ARCH.FALCON == model_class.model_arch or input_features % block_size != 0:
            new_type = "gguf:q8_0"
        elif "lm_head" in GGUF_CONFIG[target_gguf_format]:
            new_type = GGUF_CONFIG[target_gguf_format]["lm_head"]
        elif new_type != "gguf:q8_0":
            new_type = "gguf:q6_k"
    elif lm_head_name is not None and layer_name == lm_head_name and tie_word_embeddings:
        return None
    elif isinstance(layer, torch.nn.Embedding):
        embedding_format_key = "lm_head" if tie_word_embeddings else "embedding"
        if embedding_format_key in GGUF_CONFIG[target_gguf_format]:
            new_type = GGUF_CONFIG[target_gguf_format][embedding_format_key]
    elif target_bits is not None and "bits" in config and config["bits"] != target_bits:
        new_type = new_type[:bits_index] + str(config["bits"]) + new_type[bits_index + 1 :]
        new_type = _search_gguf_type(new_type)
        if new_type is None:
            raise ValueError(f"invalid bit setting for {layer_name}")
    elif gguf_name is not None:
        gguf_weight_name = gguf_name if gguf_name.endswith(".weight") else f"{gguf_name}.weight"
        new_type = dtype_selector.select_gguf_type(gguf_weight_name, len(layer.weight.shape), i_layer or 0)
    new_block_size = GGML_QUANT_SIZES[new_type.split(":")[-1].lower()][0]
    if input_features % new_block_size != 0:
        new_type = _gguf_type_fallback(new_type)
        new_block_size = GGML_QUANT_SIZES[new_type.split(":")[-1].lower()][0]
        if input_features % new_block_size != 0:
            new_type = "gguf:bf16"
        logger.warning(
            f"fallback {layer_name} to {new_type}, "
            f"because input_features({input_features}) % block_size({block_size}) != 0"
        )
    # for deepseek v2
    if layer_name.endswith("kv_b_proj") and new_type.endswith("_k") and "Deepseek" in model.config.architectures[0]:
        fallback = False

        # calc if need fallback
        qk_nope_head_dim = model.config.qk_nope_head_dim
        kv_b_shape = get_module(model, layer_name).weight.shape

        if (
            qk_nope_head_dim < QK_K
            or qk_nope_head_dim % QK_K != 0
            or kv_b_shape[-1] < QK_K
            or kv_b_shape[-1] % QK_K != 0
        ):
            fallback = True
        if fallback:
            tmp_type = _gguf_type_fallback(new_type)
            logger.warning_once(
                f"self_attn.kv_b_proj does not support the use of {new_type}, replace it with {tmp_type}"
            )
            new_type = tmp_type

    return new_type


def _infer_gguf_n_layers_from_model(model, model_type=ModelType.TEXT):
    if model is None or not hasattr(model, "named_modules"):
        return None, None

    from auto_round.utils.common import MM_MODULE_KEYS

    text_layer_ids = set()
    vision_layer_ids = set()
    layer_pattern = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
    for module_name, _ in model.named_modules():
        match = layer_pattern.search(module_name)
        if match is None:
            continue
        layer_id = int(match.group(1))
        if model_type != ModelType.TEXT and any(key in module_name.lower() for key in MM_MODULE_KEYS):
            vision_layer_ids.add(layer_id)
        else:
            text_layer_ids.add(layer_id)

    n_layer = max(text_layer_ids) + 1 if text_layer_ids else None
    n_layer_vision = max(vision_layer_ids) + 1 if vision_layer_ids else None
    return n_layer, n_layer_vision


def _resolve_gguf_n_layers(config, model_type=ModelType.TEXT, model=None):
    """Resolve text (and vision) layer counts from config, then model structure.

    Multimodal configs (e.g. Qwen3.5, Qwen3-VL) keep the text hparams only in
    ``config.text_config``, so both it and the top-level config must be checked.
    If config naming changes, fall back to the loaded module hierarchy so GGUF
    official mixed formats still receive a usable block count.
    """
    n_layer = None
    n_layer_vision = None
    text_config = getattr(config, "text_config", None)
    vision_config = None
    audio_config = getattr(config, "audio_config", None)
    for name in ["n_layers", "num_hidden_layers", "n_layer", "num_layers", "depth"]:
        if hasattr(config, name):
            n_layer = getattr(config, name)
        if text_config is not None and hasattr(text_config, name):
            # the text hparams in text_config are authoritative for multimodal models;
            # for text-only usage they still fill in when the top-level config lacks them
            if model_type != ModelType.TEXT or n_layer is None:
                n_layer = getattr(text_config, name)
        if model_type != ModelType.TEXT:
            for config_name in ["vision_config", "vision_encoder"]:
                if hasattr(config, config_name):
                    vision_config = getattr(config, config_name)
                    if hasattr(vision_config, name):
                        n_layer_vision = getattr(vision_config, name)
                        break
            if n_layer and n_layer_vision:
                break

    inferred_n_layer, inferred_n_layer_vision = _infer_gguf_n_layers_from_model(model, model_type)
    n_layer = n_layer or inferred_n_layer
    n_layer_vision = n_layer_vision or inferred_n_layer_vision

    if (
        model_type != ModelType.TEXT
        and n_layer_vision is None
        and vision_config is not None
        and audio_config is not None
    ):
        # llama.cpp's Gemma4 mmproj converter uses a fixed tensor-name map size
        # when vision and audio encoders coexist, because the two encoders do not
        # expose a single transformer block count in config.json.
        n_layer_vision = 128
    return n_layer, n_layer_vision


##https://github.com/ggml-org/llama.cpp/blob/9e31bec4fd53634c9e5b04650488a09a055f5dab/src/llama-quant.cpp#L129
def get_layer_config_by_gguf_format(layer_config, target_gguf_format: str, model, model_type=ModelType.TEXT):
    import gguf  # pylint: disable=E0401

    from auto_round.export.export_to_gguf.llama_cpp_conversion import get_conversion
    from auto_round.utils.common import MM_MODULE_KEYS
    from auto_round.utils.model import get_lm_head_name, get_module, is_separate_lm_head

    try:
        hparams = model.config.to_dict()
        conversion = get_conversion(hparams=hparams, model_type=model_type)
        model_architecture = conversion.get_model_architecture(
            hparams=hparams, model_type=conversion.model_type(model_type)
        )
    except AttributeError as e:
        raise ImportError(
            "Please use the latest gguf-py, you can use the following command to install it:\n"
            "git clone https://github.com/ggml-org/llama.cpp.git && cd llama.cpp/gguf-py"
            " && pip install . sentencepiece"
        )
    try:
        if model_type != ModelType.TEXT:
            model_class_vision = conversion.get_model_class(model_architecture, model_type=model_type)
        model_class = conversion.get_model_class(model_architecture, model_type=ModelType.TEXT)

    except NotImplementedError:
        return layer_config, {}

    n_layer, n_layer_vision = _resolve_gguf_n_layers(model.config, model_type, model=model)

    if n_layer is None:
        return layer_config, {}

    tensor_map = gguf.get_tensor_name_map(model_class.model_arch, n_layer)
    if model_type != ModelType.TEXT:
        tensor_map_vision = gguf.get_tensor_name_map(model_class_vision.model_arch, n_layer_vision)

    def _set_config(config, target_config):
        for k, v in target_config.items():
            config[k] = v
        return config

    gguf_format_config = {}
    lm_head_name = get_lm_head_name(model)
    inner_gguf_format = GGUF_CONFIG[target_gguf_format]["mostly"]
    # ggml_type =  getattr(gguf.GGMLQuantizationType,inner_gguf_format.split(":")[-1].upper())
    block_size = GGML_QUANT_SIZES[inner_gguf_format.split(":")[-1].lower()][0]
    tie_word_embeddings = True
    if hasattr(model, "config") and hasattr(model.config, "tie_word_embeddings"):
        tie_word_embeddings = model.config.tie_word_embeddings
    tie_word_embeddings &= not is_separate_lm_head(model)

    dtype_selector = GGUFDTypeSelector(
        hparams, gguf_format_to_ftype(target_gguf_format), model_class.model_arch, n_layer
    )
    layer_config_copy = thaw_mapping(layer_config)
    base_target_bits = None
    if inner_gguf_format.startswith("gguf:q") and len(inner_gguf_format) >= 7 and (inner_gguf_format[6]).isdigit():
        base_target_bits = int(inner_gguf_format[6])

    def _resolve_gguf_name(layer_name):
        if model_type != ModelType.TEXT and any([key in layer_name for key in MM_MODULE_KEYS]):
            gguf_layer_name = tensor_map_vision.get_name(layer_name)
            if gguf_layer_name is None:
                for key in MM_MODULE_KEYS:
                    gguf_layer_name = tensor_map_vision.get_name(layer_name.replace(f".{key}", ""))
                    if gguf_layer_name is not None:
                        break
        else:
            gguf_layer_name = tensor_map.get_name(layer_name)
            if gguf_layer_name is None:
                gguf_layer_name = tensor_map.get_name(layer_name.replace(".language_model", ""))
        return gguf_layer_name

    dtype_selector.n_attention_wv = sum(
        1
        for layer_name, config in layer_config_copy.items()
        if check_to_quantized(config)
        and (gguf_name := _resolve_gguf_name(layer_name))
        and any(key in gguf_name for key in ("attn_v", "attn_qkv", "attn_kv_b"))
    )

    for layer_name, config in layer_config_copy.items():
        if not check_to_quantized(config):
            continue
        layer = get_module(model, layer_name)
        i_layer = _get_digital_in_layer_name(layer_name)
        new_type = _select_gguf_layer_type(
            layer_name,
            config,
            layer,
            model,
            model_class,
            target_gguf_format=target_gguf_format,
            lm_head_name=lm_head_name,
            tie_word_embeddings=tie_word_embeddings,
            block_size=block_size,
            base_target_bits=base_target_bits,
            dtype_selector=dtype_selector,
            gguf_name=_resolve_gguf_name(layer_name),
            i_layer=i_layer,
        )
        if new_type is None:
            continue

        target_config = GGUF_INNER_CONFIG[new_type]

        _set_config(layer_config[layer_name], target_config)
        gguf_format_config[layer_name] = new_type

    return layer_config, gguf_format_config
