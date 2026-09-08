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

"""Wan SVDQuant adapter for the Python Nunchaku runtime."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from auto_round.export.svdquant_nunchaku import SourceLinearRecord, SVDQuantExportRecord

WAN_SVDQUANT_TARGET_MODULES = (
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
    "ffn.net.0.proj",
    "ffn.net.2",
)

_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")


def _config_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    raise ValueError("Wan export requires a mapping-like config")


def _normalize_config(value: Any) -> Any:
    if isinstance(value, Path):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {key: _normalize_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_config(item) for item in value]
    return value


@dataclass
class WanSVDQuantNunchakuAdapter:
    """Keep Diffusers Wan projection names for direct Python-kernel replacement."""

    config: Mapping[str, Any] | None = None
    require_complete_model: bool = True

    def _resolved_config(self, model: torch.nn.Module) -> dict[str, Any]:
        value = self.config if self.config is not None else getattr(model, "config", None)
        config = _normalize_config(_config_dict(value))
        try:
            json.dumps(config)
        except (TypeError, ValueError) as exc:
            raise ValueError("Wan config must be JSON serializable") from exc
        return config

    def metadata(self, model: torch.nn.Module, rank: int) -> Mapping[str, str]:
        return {
            "model_class": "NunchakuWanTransformer3DModel",
            "config": json.dumps(self._resolved_config(model), sort_keys=True),
            "format": "pt",
            "comfy_config": "{}",
        }

    def map_modules(
        self, model: torch.nn.Module, records: Iterable[SourceLinearRecord]
    ) -> Iterable[SVDQuantExportRecord]:
        records = tuple(records)
        for source in records:
            match = _BLOCK_RE.match(source.name)
            if match is None or match.group(2) not in WAN_SVDQUANT_TARGET_MODULES:
                raise ValueError(f"unrecognized Wan SVDQuant source {source.name!r}")

        if self.require_complete_model:
            try:
                num_layers = int(self._resolved_config(model)["num_layers"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("complete Wan export requires integer num_layers") from exc
            expected = {
                f"blocks.{index}.{suffix}"
                for index in range(num_layers)
                for suffix in WAN_SVDQUANT_TARGET_MODULES
            }
            actual = {source.name for source in records}
            if actual != expected:
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                raise ValueError(
                    f"complete Wan projection mismatch: missing={missing[:5]}, extra={extra[:5]}"
                )

        return (
            SVDQuantExportRecord(
                prefix=source.name,
                residual_weight=source.residual_weight,
                lora_down=source.lora_down,
                lora_up=source.lora_up,
                smooth=source.smooth,
                smooth_orig=source.smooth_orig,
                bias=source.bias,
                scheme=source.scheme,
                sources=(source,),
            )
            for source in records
        )

    def extra_tensors(self, model: torch.nn.Module) -> Mapping[str, torch.Tensor]:
        from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear

        quantized_prefixes = tuple(f"{name}." for name, module in model.named_modules() if isinstance(module, SVDQuantLinear))
        tensors = {}
        for name, tensor in model.state_dict().items():
            if name.startswith(quantized_prefixes):
                continue
            value = tensor.detach()
            if value.is_floating_point():
                value = value.to(torch.bfloat16)
            tensors[name] = value.cpu().contiguous()
        return tensors

    def validate_records(
        self, sources: tuple[SourceLinearRecord, ...], records: tuple[SVDQuantExportRecord, ...]
    ) -> None:
        if len(sources) != len(records):
            raise ValueError("Wan adapter must map every source projection exactly once")

    def validate(self, tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, str]) -> None:
        if metadata.get("model_class") != "NunchakuWanTransformer3DModel":
            raise ValueError("Wan metadata has incorrect model_class")
        if metadata.get("format") != "pt" or metadata.get("comfy_config") != "{}":
            raise ValueError("Wan metadata requires format='pt' and empty comfy_config")
        config = json.loads(metadata.get("config", "{}"))
        if not isinstance(config, dict):
            raise ValueError("Wan metadata config must be a JSON object")
        for key, tensor in tensors.items():
            if tensor.device.type != "cpu" or not tensor.is_contiguous():
                raise ValueError(f"Wan tensor {key!r} must be contiguous on CPU")
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Wan tensor {key!r} must be finite")


__all__ = ["WAN_SVDQUANT_TARGET_MODULES", "WanSVDQuantNunchakuAdapter"]