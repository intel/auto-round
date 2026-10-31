# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import os
import re
from collections.abc import Callable
from dataclasses import fields

import torch

from auto_round.export.base import OutputFormat, _check_divisible_by_32
from auto_round.export.common.artifacts import save_artifacts
from auto_round.export.common.packing import pack_layers, prepare_model
from auto_round.export.common.serialization import (
    filter_quantization_config,
    is_immediate_saving_mode,
    resolve_pipeline_export_layout,
    save_model,
)
from auto_round.export.formats.autoround.utils import check_neq_config
from auto_round.export.formats.gptq.config import GPTQ_REQUIRED_CONFIG_KEYS, convert_to_autogptq_dynamic
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme
from auto_round.utils import (
    check_start_with_block_name,
    check_to_quantized,
    get_block_names,
    get_module,
    matches_any_regex,
)


@OutputFormat.register("auto_gptq")
@OutputFormat.register("gptqmodel", public=False)
class AutoGPTQFormat(OutputFormat):
    support_schemes = [
        "W4A16",
        "W2A16",
        "W3A16",
        "W8A16",
        "BF16",
        "W2A16G64",
        "W2A16G32",
        "W4A16_MIXED",
    ]
    format_name = "auto_gptq"

    # 5/6/7-bit weights straddle int32 word boundaries. AutoRound packs them with the
    # generic bit-stream layout (see auto_round.utils.bit_packing), which upstream
    # AutoGPTQ/GPTQModel kernels cannot read -- so restrict them to ``auto_round:*``.
    EXTENDED_BITS = (5, 6, 7)

    def check_and_reset_format(self, scheme: QuantizationScheme, ctx):
        if scheme.bits in self.EXTENDED_BITS and not self.output_format.startswith("auto_round"):
            raise ValueError(
                f"{self.output_format} format does not support bits={scheme.bits}. "
                f"{self.EXTENDED_BITS} bit packing is only defined for the `auto_round` format family, "
                "please export to `auto_round` or `auto_round:auto_gptq`."
            )
        if not scheme.sym:
            logger.warning(
                "the asymmetrical kernel of the GPTQ format may result in a noticeable accuracy drop,"
                " particularly for 2-bit quantization and smaller models."
                " We recommend exporting to either the AutoAWQ format ( only 4 bits) or "
                "the AutoRound format(2/3/4/5/6/7/8 bits)."
            )
        if self.backend is None:
            ctx.layer_config = _check_divisible_by_32(scheme, ctx.model, ctx.layer_config)
        return super().check_and_reset_format(scheme, ctx)

    @classmethod
    def check_scheme_args(cls, scheme: QuantizationScheme) -> bool:
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
            from auto_round.export.formats.autoround.integer import pack_layer
        else:
            from auto_round.export.formats.gptq.packing import pack_layer
        pack_layer(layer_name, model, backend=self.output_format, device=device)

    def save_quantized(
        self,
        output_dir: str,
        model: torch.nn.Module = None,
        tokenizer: Callable | None = None,
        layer_config: dict | None = None,
        inplace: bool = True,
        device: str | torch.device = "cpu",
        serialization_dict: dict | None = None,
        **kwargs,
    ) -> torch.nn.Module:
        """Pack the model and save GPTQ checkpoint metadata."""

        backend = self.get_backend_name()
        if backend.startswith("auto_round:"):
            from auto_round.export.formats.autoround.integer import save_checkpoint

            return save_checkpoint(
                output_dir,
                model=model,
                tokenizer=tokenizer,
                layer_config=layer_config,
                inplace=inplace,
                device=device,
                serialization_dict=serialization_dict,
                backend=backend,
                _pack_layer=self.pack_layer,
                **kwargs,
            )
        model = prepare_model(model, inplace, device=None)
        # --- 1️⃣ Extract inputs & configs ---
        quantization_config = serialization_dict
        quant_block_list = serialization_dict.get("quant_block_list") or get_block_names(model)
        processor = kwargs.get("processor")
        image_processor = kwargs.get("image_processor")
        safe_serialization = kwargs.get("safe_serialization", True)

        # --- Save metadata (tokenizer, processor, etc.) ---
        immediate_saving = is_immediate_saving_mode(model, serialization_dict)
        processor_output_dir = output_dir
        model_output_dir = output_dir
        if output_dir:
            model_output_dir, processor_output_dir, _ = resolve_pipeline_export_layout(model, output_dir)

        if output_dir:
            if os.path.exists(output_dir) and not immediate_saving:
                logger.warning(f"{output_dir} already exists, this may cause model conflict")
            save_artifacts(processor_output_dir, tokenizer, processor, image_processor)

        # --- Handle quantization structure ---
        all_blocks = quant_block_list
        flattened = [x for sub in all_blocks for x in sub]
        common_prefix = os.path.commonprefix(flattened).rstrip(".")

        if "BLOCK_PATTERNS" in kwargs and common_prefix not in kwargs["BLOCK_PATTERNS"]:
            logger.error(f"Unsupported block prefix '{common_prefix}' for AutoGPTQ format.")
            quantization_config["block_name_to_quantize"] = common_prefix
        quantization_config.pop("to_quant_block_names", None)

        # --- Build per-layer dynamic overrides ---
        regex_config = quantization_config.pop("regex_config", {})
        block_name_to_quantize = quantization_config.get("block_name_to_quantize")
        extra_config = {}
        lm_head_quantized = False
        scheme_keys = [f.name for f in fields(QuantizationScheme)]
        for layer_name, cfg in layer_config.items():
            bits = cfg.get("bits", 16)
            in_blocks = cfg.get("in_blocks", False)
            # Handle non-block layers (e.g., LM head)
            if not in_blocks and bits <= 8:
                lm_head_quantized = True
                extra_config[layer_name] = {k: cfg[k] for k in GPTQ_REQUIRED_CONFIG_KEYS}
                continue
            # Handle block layers
            if in_blocks or (
                block_name_to_quantize and check_start_with_block_name(layer_name, block_name_to_quantize)
            ):
                neq_keys = check_neq_config(cfg, **{k: quantization_config.get(k) for k in scheme_keys})
                if neq_keys:
                    if matches_any_regex(layer_name, regex_config):
                        continue
                    extra_config[layer_name] = {k: cfg[k] for k in GPTQ_REQUIRED_CONFIG_KEYS}

        # --- Merge regex_config + extra_config into GPTQ dynamic config ---
        dynamic = {}
        if regex_config:
            dynamic.update(convert_to_autogptq_dynamic(regex_config))
        if extra_config:
            dynamic.update(convert_to_autogptq_dynamic(extra_config))
        if dynamic:
            quantization_config["dynamic"] = dynamic

        # --- Block-wise quantization verification ---
        for n, m in model.named_modules():
            m.global_name = n

        all_to_quantized = True
        modules_in_block_to_quantize = []
        # for backward compatibility
        for block_names in all_blocks:
            quantized_in_group = set()
            not_quantized_in_group = set()
            for block_name in block_names:
                block = get_module(model, block_name)
                for n, m in block.named_modules():
                    if m.global_name not in layer_config:
                        continue
                    if not check_to_quantized(layer_config[m.global_name]):
                        not_quantized_in_group.add(n)
                    else:
                        quantized_in_group.add(n)
            if not_quantized_in_group:
                all_to_quantized = False
            modules_in_block_to_quantize.append(sorted(quantized_in_group))

        if all_to_quantized:
            modules_in_block_to_quantize = None

        for _, m in model.named_modules():
            delattr(m, "global_name")

        pack_layers(model, layer_config, self.pack_layer, device=device)
        if output_dir is None:
            return model
        quantization_config["lm_head"] = lm_head_quantized
        quantization_config["provider"] = "auto-round"
        quantization_config["quant_method"] = "gptq"
        quantization_config.pop("dataset", None)  ## pile-10k is not supported in gptq
        quantization_config["desc_act"] = False  ## for autogptq API
        quantization_config["true_sequential"] = False
        quantization_config["damp_percent"] = 0.01
        if modules_in_block_to_quantize is not None:
            quantization_config["modules_in_block_to_quantize"] = modules_in_block_to_quantize
        filter_quantization_config(quantization_config)
        if hasattr(model, "config"):
            model.config.quantization_config = quantization_config

        dtype = torch.float16  ##force dtype to fp16
        save_model(
            model,
            model_output_dir,
            safe_serialization=safe_serialization,
            dtype=dtype,
            config_file="quantize_config.json",
            immediate_saving=immediate_saving,
        )
        return model
