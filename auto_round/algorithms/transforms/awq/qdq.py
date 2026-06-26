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
    qf, opt_qf = tool.resolve_quant_funcs(params)  # dispatch once, reuse in loop
    w_qdq = tool.qdq(weight, params, quant_func=qf, opt_quant_func=opt_qf)
"""

from __future__ import annotations

import torch

from auto_round.data_type.utils import (
    compute_optimized_init_scale,
    get_optimized_quant_func,
    get_quant_func,
)


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
    def configure(self, compressor) -> None:
        """Derive QDQ behaviour from the run's block quantizer."""
        block_config = getattr(compressor, "quantize_config", None)
        self.disable_opt_rtn = bool(getattr(block_config, "disable_opt_rtn", False))
        self.use_v2_scale_search = self._block_quantizer_is_signroundv2(compressor)

    @staticmethod
    def _block_quantizer_is_signroundv2(compressor) -> bool:
        """Return ``True`` if the terminal block quantizer is SignRoundV2."""
        block_config = getattr(compressor, "quantize_config", None)
        if block_config is None:
            return False
        from auto_round.algorithms.registry import resolve_pipeline_member

        try:
            return resolve_pipeline_member(block_config).__name__ == "SignRoundV2Quantizer"
        except Exception:
            return False

    # ── per-layer scheme resolution + dispatch ────────────────────────────────
    def _layer_config_for(self, layer: torch.nn.Module) -> dict:
        name = getattr(layer, "global_name", None) or ""
        return (self.layer_config or {}).get(name, {})

    def resolve_params(self, layer: torch.nn.Module) -> dict:
        """Resolve the per-layer weight-quant params (``layer_config`` + fallbacks).

        Single source of every scheme field the QDQ needs, so callers pass the
        returned dict straight to :meth:`resolve_quant_funcs` and :meth:`qdq`.
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

    def resolve_quant_funcs(self, params: dict):
        """Dispatch ``(quant_func, opt_quant_func)`` for ``params``.

        ``opt_quant_func`` is non-``None`` only when the SignRoundV2 optimized
        init-scale path applies (it depends solely on ``data_type`` / ``sym``).
        Hoisting this out of a search loop avoids repeated dispatch.
        """
        quant_func, _ = get_quant_func(
            params["data_type"],
            params["bits"],
            params["sym"],
            disable_opt_rtn=params["disable_opt_rtn"],
            group_size=params["group_size"],
            iters=0,
        )
        opt_quant_func = None
        if self.use_v2_scale_search and params["sym"]:
            opt_quant_func = get_optimized_quant_func(params["data_type"])
        return quant_func, opt_quant_func

    # ── the unified QDQ for AWQ search ──────────────────────
    @torch.no_grad()
    def qdq(
        self,
        weight: torch.Tensor,
        params: dict,
        *,
        quant_func=None,
        opt_quant_func=None,
        imatrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Quantize-dequantize ``weight`` under the resolved ``params``.

        The single QDQ entry point for every scheme: ``bits`` / ``group_size`` /
        ``sym`` / ``data_type`` plus the optional GGUF super-block params drive
        the dispatched ``quant_func``; a non-``None`` ``opt_quant_func`` enables
        the SignRoundV2 optimized init-scale path for this weight. Used for
        both the scale grid-search and the clip-search -- reference/loss only,
        it never mutates stored weights.
        """
        if quant_func is None:
            quant_func, opt_quant_func = self.resolve_quant_funcs(params)
        if quant_func is None:
            raise RuntimeError(
                "QDQTool: no quantization function resolved for "
                f"data_type={params['data_type']}, bits={params['bits']}, "
                f"sym={params['sym']}, group_size={params['group_size']}."
            )

        quant_kwargs = {
            "bits": params["bits"],
            "group_size": params["group_size"],
            "data_type": params["data_type"],
            "sym": params["sym"],
        }
        if params.get("super_bits") is not None:
            quant_kwargs["super_bits"] = params["super_bits"]
        if params.get("super_group_size") is not None:
            quant_kwargs["super_group_size"] = params["super_group_size"]

        active_quant_func = quant_func
        if opt_quant_func is not None:
            init_scale = compute_optimized_init_scale(
                weight, params["data_type"], params["bits"], params["group_size"], imatrix=imatrix
            )
            if init_scale is not None:
                quant_kwargs["init_scale"] = init_scale
                active_quant_func = opt_quant_func

        qdq_weight, _, _ = active_quant_func(weight, **quant_kwargs)
        return qdq_weight
