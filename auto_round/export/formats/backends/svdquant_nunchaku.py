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

from __future__ import annotations

import os
from typing import Any, Callable, Union

import torch

from auto_round.export.formats.base import OutputFormat
from auto_round.schemes import PRESET_SCHEMES, QuantizationScheme


def _invoke_save_hook(hook: Callable[[str], Any], output_dir: str) -> None:
    hook(output_dir)


@OutputFormat.register("svdquant_nunchaku")
class SVDQuantNunchakuFormat(OutputFormat):
    support_schemes = ["MXFP4"]
    format_name = "svdquant_nunchaku"
    _e2m1_aliases = frozenset({"mx_fp", "mx_fp4", "mx_fp4e2m1"})

    def __init__(self, format: str, scheme: QuantizationScheme, ctx: Any):
        self.output_format = format
        self.backend = None
        self.mllm = ctx.mllm
        self.check_scheme_args(scheme)
        self._resolved_scheme = scheme.copy()

    def is_supported_immediate_packing(self) -> bool:
        return False

    def is_supported_immediate_saving(self) -> bool:
        return False

    @classmethod
    def check_scheme_args(cls, scheme: QuantizationScheme) -> bool:
        rules = {
            "data_type": cls._e2m1_aliases,
            "bits": 4,
            "group_size": 32,
            "sym": True,
            "act_data_type": cls._e2m1_aliases,
            "act_bits": 4,
            "act_group_size": 32,
            "act_sym": True,
            "act_dynamic": True,
        }
        for name, expected in rules.items():
            actual = getattr(scheme, name, None)
            if isinstance(expected, frozenset):
                valid = isinstance(actual, str) and actual in expected
                expected_text = f"one of {sorted(expected)}"
            elif isinstance(expected, int) and not isinstance(expected, bool):
                valid = isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
                expected_text = repr(expected)
            else:
                valid = actual is expected
                expected_text = repr(expected)
            if not valid:
                raise ValueError(
                    f"{cls.format_name} got {name}={actual!r}; expected {name}={expected_text} "
                    "for Nunchaku E2M1 group32 export."
                )
        return True

    def _validate_svd_layer_overrides(self, model: torch.nn.Module, layer_config: dict | None) -> None:
        if model is None or not layer_config:
            return
        from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear

        for name, module in model.named_modules():
            if not isinstance(module, SVDQuantLinear):
                continue
            for layer_name in (name, f"{name}.residual_linear"):
                if layer_name not in layer_config:
                    continue
                override = layer_config[layer_name]
                if isinstance(override, str):
                    scheme = PRESET_SCHEMES[override.upper()].copy()
                elif isinstance(override, QuantizationScheme):
                    scheme = override.copy()
                elif isinstance(override, dict):
                    values = dict(override)
                    preset_name = values.pop("scheme", None)
                    scheme = (
                        PRESET_SCHEMES[preset_name.upper()].copy()
                        if preset_name is not None
                        else self._resolved_scheme.copy()
                    )
                    scheme.update_from_dict(values)
                else:
                    raise TypeError(
                        f"Unsupported layer_config value for SVDQuant residual {layer_name!r}: {type(override)}"
                    )
                try:
                    self.check_scheme_args(scheme)
                except ValueError as exc:
                    raise ValueError(
                        f"{self.format_name} layer {layer_name!r} has an incompatible residual scheme: {exc}"
                    ) from exc

    def check_and_reset_format(self, scheme: QuantizationScheme, ctx: Any):
        self._validate_svd_layer_overrides(ctx.model, ctx.layer_config)
        return None, scheme, ctx.layer_config, ctx.quant_block_list

    def pack_layer(self, *args, **kwargs):
        return None

    def save_quantized(
        self,
        output_dir: str,
        model: torch.nn.Module = None,
        tokenizer: Callable = None,
        layer_config: dict = None,
        inplace: bool = True,
        device: Union[str, torch.device] = "cpu",
        serialization_dict: dict = None,
        *,
        config=None,
        residual_provider=None,
        adapter=None,
        model_adapter=None,
        **kwargs,
    ) -> torch.nn.Module:
        if output_dir is None:
            return model
        from auto_round.export.svdquant_nunchaku import (
            NUNCHAKU_WEIGHT_FILENAME,
            SVDQuantExportConfig,
            save_svdquant_nunchaku_safetensors,
        )

        self._validate_svd_layer_overrides(model, layer_config)
        save_config = getattr(model, "save_config", None)
        if callable(save_config):
            _invoke_save_hook(save_config, output_dir)
        else:
            model_config = getattr(model, "config", None)
            save_pretrained = getattr(model_config, "save_pretrained", None)
            if callable(save_pretrained):
                _invoke_save_hook(save_pretrained, output_dir)
        model_adapter = model_adapter or getattr(model, "_autoround_svdquant_model_adapter", "auto")
        if isinstance(model_adapter, str):
            from auto_round.export.svdquant_adapters import resolve_svdquant_model_adapter

            model_adapter = resolve_svdquant_model_adapter(model_adapter, model, decomposition_device=device)
        if model_adapter is not None:
            if adapter is not None:
                raise TypeError("Pass only one of model_adapter and adapter.")
            adapter = model_adapter
        from auto_round.export.svdquant_nunchaku import IdentitySVDQuantModelAdapter

        if isinstance(adapter, IdentitySVDQuantModelAdapter):
            raise ValueError(
                "svdquant_nunchaku requires a runtime model adapter; "
                "use a supported architecture or select one with model_adapter"
            )
        if config is None:
            config = SVDQuantExportConfig(runtime_loadable=True)
        elif not config.runtime_loadable:
            raise ValueError("svdquant_nunchaku requires config.runtime_loadable=True")

        output_path = os.path.join(os.fspath(output_dir), NUNCHAKU_WEIGHT_FILENAME)
        save_svdquant_nunchaku_safetensors(
            model,
            output_path,
            config=config,
            residual_provider=residual_provider,
            adapter=adapter,
        )
        return model
