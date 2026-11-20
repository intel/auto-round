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
from collections.abc import Callable
from typing import Any

import torch

from auto_round.export.base import OutputFormat
from auto_round.export.formats.svdquant.base import SVDQuantFormatBase
from auto_round.schemes import QuantizationScheme


@OutputFormat.register("svdquant_nunchaku")
class SVDQuantNunchakuFormat(SVDQuantFormatBase):
    support_schemes = ["MXFP4"]
    format_name = "svdquant_nunchaku"
    _e2m1_aliases = frozenset({"mx_fp", "mx_fp4", "mx_fp4e2m1"})

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

    def check_and_reset_format(self, scheme: QuantizationScheme, ctx: Any):
        self._validate_svd_layer_overrides(ctx.model, ctx.layer_config)
        quant_block_list = ctx.quant_block_list
        if quant_block_list is None and ctx.model is not None:
            from auto_round.export.formats.svdquant.adapters import resolve_svdquant_model_adapter

            model_adapter = resolve_svdquant_model_adapter("auto", ctx.model)
            adapter_quant_block_list = getattr(model_adapter, "quant_block_list", None)
            if callable(adapter_quant_block_list):
                quant_block_list = adapter_quant_block_list(ctx.model)
        return None, scheme, ctx.layer_config, quant_block_list

    def save_quantized(
        self,
        output_dir: str,
        model: torch.nn.Module = None,
        tokenizer: Callable | None = None,
        layer_config: dict | None = None,
        inplace: bool = True,
        device: str | torch.device = "cpu",
        serialization_dict: dict | None = None,
        *,
        config=None,
        residual_provider=None,
        adapter=None,
        model_adapter=None,
        **kwargs,
    ) -> torch.nn.Module:
        if output_dir is None:
            return model
        from auto_round.export.formats.svdquant.nunchaku import (
            NUNCHAKU_WEIGHT_FILENAME,
            SVDQuantExportConfig,
            save_svdquant_nunchaku_safetensors,
        )

        self._validate_svd_layer_overrides(model, layer_config)
        save_config = getattr(model, "save_config", None)
        if callable(save_config):
            save_config(output_dir)
        else:
            model_config = getattr(model, "config", None)
            save_pretrained = getattr(model_config, "save_pretrained", None)
            if callable(save_pretrained):
                save_pretrained(output_dir)
        model_adapter = model_adapter or getattr(model, "_autoround_svdquant_model_adapter", "auto")
        if isinstance(model_adapter, str):
            from auto_round.export.formats.svdquant.adapters import resolve_svdquant_model_adapter

            model_adapter = resolve_svdquant_model_adapter(model_adapter, model, decomposition_device=device)
        if model_adapter is not None:
            if adapter is not None:
                raise TypeError("Pass only one of model_adapter and adapter.")
            adapter = model_adapter
        from auto_round.export.formats.svdquant.common import IdentitySVDQuantModelAdapter

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
