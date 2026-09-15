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
"""AWQ reference quantize-dequantize (QDQ).

AWQ searches smoothing scales and weight-clip thresholds by minimizing the
error of the *block quantizer's* weight QDQ. :class:`QDQTool` reproduces that
exact weight QDQ for every scheme AutoRound supports -- plain int, MX-FP,
NV-FP etc. -- behind a single
:meth:`QDQTool.qdq` entry point, so the search loss stays in lock-step with what
the downstream block quantizer will actually do and there is no second, parallel
implementation to drift from.

Usage (the same unified QDQ serves both the scale grid-search and clip-search)::

    tool = QDQTool(bits=..., group_size=..., sym=..., data_type=...)
    tool.layer_config = <per-layer overrides>      # wired in by AWQTransform
    tool.configure(compressor)                     # opt_rtn / v2 flags

    params = tool.resolve_params(layer)            # per-layer scheme
    quantizer = tool.resolve_quantizer(params)  # dispatch once, reuse in loop
    w_qdq = tool.qdq(weight, params, quantizer=quantizer)
"""

from __future__ import annotations

import torch

from auto_round.data_type.base import create_quantizer


class QDQTool:
    """Reproduce the block quantizer's weight QDQ for AWQ's search loss.

    The constructor takes the scheme-level *fallback* params (used when a layer
    is absent from ``layer_config``). :meth:`configure` derives the runtime
    quant-func-selection flags from the run's block quantizer; ``layer_config``
    is wired in by the owning ``AWQTransform``.
    """

    def __init__(self, *, bits, group_size, sym, data_type) -> None:
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self.data_type = data_type

        self.layer_config: dict | None = None
        self.disable_opt_rtn: bool | None = None
        self.use_v2_scale_search: bool = False

    # ── runtime wiring ────────────────────────────────────────────────────────
    @staticmethod
    def _block_quantizer_config(block_quantizer_or_composer):
        """Return the terminal block-quantizer config from new or legacy hosts."""
        block_quantizer = getattr(block_quantizer_or_composer, "block_quantizer", block_quantizer_or_composer)
        config = getattr(block_quantizer, "config", None)
        if config is not None:
            return config
        return getattr(block_quantizer_or_composer, "quantize_config", None)

    def configure(self, composer, awq_config=None) -> None:
        """Derive QDQ behaviour from the run's block quantizer."""
        block_config = self._block_quantizer_config(composer)
        awq_disable_opt_rtn = getattr(awq_config, "disable_opt_rtn", None)
        if awq_disable_opt_rtn is None:
            awq_disable_opt_rtn = getattr(block_config, "disable_opt_rtn", False)
        self.disable_opt_rtn = bool(awq_disable_opt_rtn)
        self.use_v2_scale_search = self._block_quantizer_is_signroundv2(composer)

    @staticmethod
    def _block_quantizer_is_signroundv2(block_quantizer_or_composer) -> bool:
        """Return ``True`` if the terminal block quantizer is SignRoundV2."""
        from auto_round.algorithms.quantization.sign_round.config import SignRoundV2Config

        config = QDQTool._block_quantizer_config(block_quantizer_or_composer)
        return isinstance(config, SignRoundV2Config)

    # ── per-layer scheme resolution + dispatch ────────────────────────────────
    def _layer_config_for(self, layer: torch.nn.Module) -> dict:
        name = getattr(layer, "global_name", None) or ""
        return (self.layer_config or {}).get(name, {})

    def resolve_params(self, layer: torch.nn.Module) -> dict:
        """Resolve the per-layer weight-quant params (``layer_config`` + fallbacks).

        Single source of every scheme field the QDQ needs, so callers pass the
        returned dict straight to :meth:`resolve_quantizer` and :meth:`qdq`.
        ``super_bits`` / ``super_group_size`` are the GGUF double-quant
        super-block params and are ``None`` for non-GGUF schemes.
        """
        cfg = self._layer_config_for(layer)
        return {
            "bits": cfg.get("bits", self.bits),
            "group_size": cfg.get("group_size", self.group_size),
            "sym": cfg.get("sym", self.sym),
            "data_type": cfg.get("data_type", self.data_type),
            "disable_opt_rtn": cfg.get("disable_opt_rtn", self.disable_opt_rtn),
            "super_bits": cfg.get("super_bits", None),
            "super_group_size": cfg.get("super_group_size", None),
        }

    def resolve_quantizer(self, params: dict):
        """Create one configured quantizer to reuse throughout a weight search."""
        use_optimized_init = self.use_v2_scale_search and params["sym"] and not params["disable_opt_rtn"]
        return create_quantizer(
            {**params, "scale_dtype": torch.float32},
            disable_opt_rtn=not use_optimized_init,
        )

    # ── the unified QDQ for AWQ search ──────────────────────
    @torch.no_grad()
    def qdq(
        self,
        weight: torch.Tensor,
        params: dict,
        *,
        quantizer=None,
        imatrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Simulate quantization for a search candidate without changing stored weights."""
        if quantizer is None:
            quantizer = self.resolve_quantizer(params)
        quantizer.initialize(weight, imatrix=imatrix)
        return quantizer.quantize(weight)
