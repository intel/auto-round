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

"""Architecture adapters for SVDQuant Nunchaku export."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from auto_round.export.svdquant_nunchaku import IdentitySVDQuantModelAdapter

from auto_round.export.svdquant_adapters.flux import (
    FLUX_TOP_LEVEL_TENSOR_KEYS,
    FLUX_SVDQUANT_TARGET_MODULES,
    FluxSVDQuantNunchakuAdapter,
    flux_onefile_tensor_count,
)


def _model_config(model: torch.nn.Module) -> dict:
    config = getattr(model, "config", None)
    if isinstance(config, Mapping):
        return dict(config)
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def resolve_svdquant_model_adapter(
    name: str,
    model: torch.nn.Module,
    *,
    decomposition_device: str | torch.device = "cpu",
):
    """Resolve a registered architecture adapter without runtime dependencies."""

    normalized = name.strip().lower()
    if normalized not in {"auto", "identity", "flux"}:
        raise ValueError(f"unknown SVDQuant model adapter {name!r}; expected auto, identity, or flux")
    config = _model_config(model)
    class_name = str(config.get("_class_name", type(model).__name__)).lower()
    if normalized == "auto":
        normalized = "flux" if "fluxtransformer" in class_name else "identity"
    if normalized == "flux":
        return FluxSVDQuantNunchakuAdapter(
            config=config or None,
            decomposition_device=decomposition_device,
            require_complete_model=True,
        )
    return IdentitySVDQuantModelAdapter()


__all__ = [
    "FLUX_TOP_LEVEL_TENSOR_KEYS",
    "FLUX_SVDQUANT_TARGET_MODULES",
    "FluxSVDQuantNunchakuAdapter",
    "flux_onefile_tensor_count",
    "resolve_svdquant_model_adapter",
]
