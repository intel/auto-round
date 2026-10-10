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
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict

import torch
import transformers

from auto_round.export.formats.base import OutputFormat, _check_divisible_by_32
from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme
from auto_round.utils import INNER_SUPPORTED_LAYER_TYPES, SUPPORTED_LAYER_TYPES, compress_layer_names


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
    from auto_round.export.export_to_awq.utils import (
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
        from auto_round.export.export_to_awq.utils import SUPPORTED_AWQ_BITS, awq_gemm_kernel_supported
        from auto_round.utils.model import get_layer_names_in_block, get_module

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
        from auto_round.export.export_to_awq.export import pack_layer

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
