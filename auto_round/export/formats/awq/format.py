# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import copy
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict

import torch
import transformers

from auto_round.export.base import OutputFormat, _check_divisible_by_32
from auto_round.export.common.artifacts import save_artifacts
from auto_round.export.common.packing import pack_layers, prepare_model
from auto_round.export.common.serialization import filter_quantization_config, is_immediate_saving_mode, save_model
from auto_round.export.formats.awq.config import _collect_modules_to_not_convert
from auto_round.export.formats.awq.utils import awq_gemm_kernel_supported, awq_layer_user_forced
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme
from auto_round.utils import (
    INNER_SUPPORTED_LAYER_TYPES,
    SUPPORTED_LAYER_TYPES,
    compress_layer_names,
    get_module,
    unsupported_meta_device,
)
from auto_round.utils.model import get_layer_names_in_block


def mark_awq_unservable_layers(model, layer_config: dict, default_dict: dict) -> dict:
    """Exclude layers the AWQ GEMM kernel cannot serve by marking them fp16.

    Serving stacks such as vLLM crash on shapes violating the kernel's
    divisibility constraints ("OC is not multiple of Group size"), e.g. the
    small ``in_proj_ba`` projections in Gated-DeltaNet hybrids. Marking such
    layers fp16 keeps them out of quantization and packing and lands them in
    ``modules_to_not_convert`` in the saved config.

    ``layer_config`` may hold raw user entries at format-resolution time
    (partial dicts, preset strings, not-yet-expanded regex keys) or fully
    resolved per-layer dicts at plan time; both are tolerated.

    Layers the user explicitly configured for quantization (exact name or
    expanded regex entry, ``fixed_by_user != False``) are honored: they keep
    their configuration, the packers pick the flag up from
    ``layer.fixed_by_user`` (written by ``apply_plan_to_model``, see
    ``awq_layer_user_forced``), and only a warning is logged that the shape
    may fail on the AWQ GEMM path.
    """
    if model is None or default_dict["data_type"] != "int":
        return layer_config
    from auto_round.export.formats.awq.utils import (
        awq_gemm_kernel_supported,
        awq_user_forced_quantization,
    )

    scheme_bits, scheme_group_size = default_dict["bits"], default_dict["group_size"]
    skipped_layers = []
    user_forced_layers = []
    for name, module in model.named_modules():
        if not (type(module) in SUPPORTED_LAYER_TYPES or module.__class__.__name__ in INNER_SUPPORTED_LAYER_TYPES):
            continue
        if not hasattr(module, "in_features") or not hasattr(module, "out_features"):
            continue
        cfg = layer_config.get(name) if layer_config else None
        if hasattr(cfg, "get"):
            bits = cfg.get("bits") or scheme_bits
            group_size = cfg.get("group_size") or scheme_group_size
        else:
            bits, group_size = scheme_bits, scheme_group_size
        if not isinstance(bits, int) or bits > 8:
            continue  # already excluded from quantization or exotic config
        if not isinstance(group_size, int) or group_size <= 0:
            continue
        if awq_gemm_kernel_supported(module.in_features, module.out_features, bits, group_size):
            continue
        if awq_user_forced_quantization(cfg):
            # The user explicitly configured this layer for quantization;
            # keep the configuration. fixed_by_user is applied to the module
            # by apply_plan_to_model, which the packers consult.
            user_forced_layers.append(name)
            continue
        if layer_config is None:
            layer_config = {}
        # Build a fresh entry rather than updating in place: regex expansion
        # shares one dict object across all matched layer names, so an in-place
        # update would leak the fp16 mark to sibling layers.
        new_cfg = copy.deepcopy(default_dict)
        if isinstance(cfg, Mapping):
            new_cfg.update(cfg)
        new_cfg.update({"bits": 16, "data_type": "fp", "fixed_by_user": True})
        layer_config[name] = new_cfg
        skipped_layers.append(name)
    compressed_skipped_layers = compress_layer_names(skipped_layers)
    if compressed_skipped_layers:
        logger.warning_once(
            "some layers are skipped quantization (in/out features not divisible by the AWQ "
            f"group_size {scheme_group_size} or out_features not divisible by 64, "
            "which the AWQ GEMM kernel cannot serve); they are exported in fp16 and listed "
            f"in `modules_to_not_convert`: {compressed_skipped_layers}"
        )
    compressed_user_forced = compress_layer_names(user_forced_layers)
    if compressed_user_forced:
        logger.warning_once(
            "some layers are quantized per their explicit `layer_config` entry but their "
            "shapes cannot be served by the AWQ GEMM kernel (in/out features not divisible "
            "by group_size or out_features not divisible by 64); they are packed as "
            "configured and may fail on the AWQ GEMM path in serving stacks such as vLLM: "
            f"{compressed_user_forced}"
        )
    return layer_config


@OutputFormat.register("auto_awq")
class AutoAWQFormat(OutputFormat):
    support_schemes = ["W4A16", "W5A16", "W6A16", "W7A16"]
    format_name = "auto_awq"

    # See AutoGPTQFormat.EXTENDED_BITS -- 5/6/7-bit uses the generic bit-stream
    # layout, which upstream AutoAWQ kernels cannot read.
    EXTENDED_BITS = (5, 6, 7)

    @classmethod
    def check_scheme_args(cls, scheme: QuantizationScheme) -> bool:
        error_logs = []
        if scheme.bits not in (4,) + cls.EXTENDED_BITS:
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
        """Check whether a model is compatible with the AutoAWQ GEMM kernel."""
        from auto_round.export.formats.awq.utils import SUPPORTED_AWQ_BITS, awq_gemm_kernel_supported
        from auto_round.utils.model import get_module

        if bits not in SUPPORTED_AWQ_BITS:
            return False, f"AutoAWQ GEMM kernel only supports bits in {SUPPORTED_AWQ_BITS}"
        for _, module in model.named_modules():
            if type(module) == transformers.pytorch_utils.Conv1D:
                return False, "AutoAWQ GEMM kernel does not support conv1d"

        layer_names = get_layer_names_in_block(model)
        for layer_name in layer_names:
            layer_cfg = layer_configs.get(layer_name) if layer_configs else None
            if hasattr(layer_cfg, "get") and (layer_cfg.get("bits") or bits) > 8:
                continue

            layer = get_module(model, layer_name)
            layer_bits = layer_cfg.get("bits") if hasattr(layer_cfg, "get") else None
            layer_group_size = layer_cfg.get("group_size") if hasattr(layer_cfg, "get") else None
            if not awq_gemm_kernel_supported(
                layer.in_features,
                layer.out_features,
                layer_bits or bits,
                layer_group_size or group_size,
            ):
                return False, (
                    f"Layer {layer_name} (in_features={layer.in_features}, "
                    f"out_features={layer.out_features}, group_size={layer_group_size or group_size}) "
                    "cannot be served by the AWQ GEMM kernel"
                )

        return True, ""

    def check_and_reset_format(self, scheme: QuantizationScheme, ctx):
        if self.backend is None:
            ctx.layer_config = _check_divisible_by_32(scheme, ctx.model, ctx.layer_config)
            # Also re-applied after layer-config resolution (regex/partial user
            # entries are expanded then) via apply_layer_config_special_cases.
            ctx.layer_config = mark_awq_unservable_layers(ctx.model, ctx.layer_config, asdict(scheme))
        awq_supported, info = self.check_awq_gemm_compatibility(
            ctx.model, scheme.bits, scheme.group_size, scheme.sym, ctx.layer_config
        )
        if not awq_supported:
            logger.warning(f"The AutoAWQ format may not be supported due to {info}")
        if scheme.bits in self.EXTENDED_BITS and not self.output_format.startswith("auto_round"):
            raise ValueError(
                f"{self.output_format} format does not support bits={scheme.bits}. "
                f"{self.EXTENDED_BITS} bit packing is only defined for the `auto_round` format family, "
                "please export to `auto_round:auto_awq`."
            )
        if scheme.bits not in (4,) + self.EXTENDED_BITS:
            raise ValueError(
                f"{self.format_name} format support quantization scheme with {','.join(self.support_schemes)} "
                f"but got bits={scheme.bits}, please have a check."
            )

        return super().check_and_reset_format(scheme, ctx)

    def pack_layer(self, layer_name, model, device=None, **kwargs):
        from auto_round.export.formats.awq.packing import pack_layer

        pack_layer(layer_name, model, backend=self.output_format, device=device)

    def save_quantized_as_format(
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
        """Pack the model and save AWQ checkpoint metadata."""
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
        model = prepare_model(model, inplace, device="cpu")
        to_quant_block_names = serialization_dict.get("to_quant_block_names", None)
        processor = kwargs.get("processor", None)
        image_processor = kwargs.get("image_processor", None)

        immediate_saving = is_immediate_saving_mode(model, serialization_dict)
        if output_dir is not None and os.path.exists(output_dir) and not immediate_saving:
            logger.warning(f"{output_dir} already exists, this may cause model conflict")

        logger.info("Saving quantized model to auto_awq format")
        save_artifacts(output_dir, tokenizer, processor, image_processor)
        compressed_model = model

        names = list(layer_config.keys())

        # Layers the AWQ GEMM kernel cannot serve (e.g. out_features not divisible by
        # group_size, as in Gated-DeltaNet in_proj_ba) must stay in full precision;
        # packing them crashes serving stacks such as vLLM. They are reported via
        # modules_to_not_convert below. Layers the user explicitly configured for
        # quantization in layer_config are exempt: their setting is honored and they
        # are packed as requested. This also covers the split
        # quantize() -> save_quantized(format="auto_awq") flow, where format
        # resolution (and the fp16 downgrade in check_and_reset_format) happens only
        # after the layer has already been quantized.
        unservable_layers = set()
        meta_device = unsupported_meta_device(model)
        if not meta_device:
            for name in names:
                layer = get_module(compressed_model, name)
                if (
                    type(layer) not in SUPPORTED_LAYER_TYPES
                    or not hasattr(layer, "in_features")
                    or not hasattr(layer, "out_features")
                ):
                    continue
                cfg = layer_config.get(name, {})
                cfg_get = cfg.get if hasattr(cfg, "get") else (lambda *a, **k: None)
                bits = getattr(layer, "bits", None)
                if bits is None:
                    bits = cfg_get("bits") or serialization_dict.get("bits")
                if not isinstance(bits, int) or bits > 8:
                    continue
                group_size = getattr(layer, "group_size", None)
                if group_size is None:
                    group_size = cfg_get("group_size") or serialization_dict.get("group_size")
                if not awq_gemm_kernel_supported(layer.in_features, layer.out_features, bits, group_size):
                    if not awq_layer_user_forced(layer, cfg):
                        unservable_layers.add(name)
            if unservable_layers:
                logger.warning_once(
                    f"{len(unservable_layers)} layer(s) cannot be served by the AWQ GEMM kernel "
                    "(in/out features not divisible by group_size or out_features not divisible by 64) "
                    "and are kept in full precision: "
                    f"{compress_layer_names(sorted(unservable_layers))}"
                )

        pack_layers(
            compressed_model, [name for name in names if name not in unservable_layers], self.pack_layer, device=device
        )
        if output_dir is None:
            return compressed_model

        quantization_config = serialization_dict
        regex_config = quantization_config.pop("regex_config", {})  # awq do not support mixed bits config saving

        if output_dir is None:
            return compressed_model

        # Collect all modules that should not be converted (not quantized)
        modules_to_not_convert = _collect_modules_to_not_convert(
            compressed_model,
            layer_config,
            regex_config,
            to_quant_block_names=to_quant_block_names,
        )

        quantization_config["provider"] = "auto-round"
        quantization_config["quant_method"] = "awq"
        quantization_config["zero_point"] = not quantization_config["sym"]
        quantization_config["version"] = "gemm"
        if unservable_layers:
            modules_to_not_convert = sorted(set(modules_to_not_convert) | unservable_layers)
        quantization_config["modules_to_not_convert"] = modules_to_not_convert
        ##check module quantized in block, this may have bug for mixed precision quantization
        filter_quantization_config(quantization_config)
        if hasattr(compressed_model, "config"):
            compressed_model.config.quantization_config = quantization_config
        safe_serialization = kwargs.get("safe_serialization", True)
        # Force torch_dtype in the saved config to fp16 for vLLM CUDA kernel compatibility;
        # this does not change in-memory model weights.
        dtype = torch.float16
        save_model(
            compressed_model,
            output_dir,
            safe_serialization=safe_serialization,
            dtype=dtype,
            immediate_saving=immediate_saving,
        )

        return compressed_model
