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
"""Quantize-dequantize (QDQ) tool.

:class:`QDQTool` is a reusable, scheme-aware primitive for
*"quantize-dequantize a candidate weight under a target quantization scheme"*.
Callers that need to evaluate a quantization-aware loss (e.g. pre-processing
transforms searching for smoothing / clipping parameters) **compose** one of
these instead of re-implementing the quantizer's dispatch internals:

* per-layer quant-param resolution (``layer_config`` overrides + fallbacks),
* quant-func dispatch incl. the ``opt_rtn`` (importance-weighted) variant,
* GGUF double-quant super-block params (``super_bits`` / ``super_group_size``),
* the optimized init-scale path (e.g. SignRoundV2),
* a clip-search QDQ primitive.

Keeping this logic in one place means a caller's reference loss stays in
lock-step with what the downstream block quantizer will actually do, removing
the silent drift risk of two parallel implementations.
"""

from __future__ import annotations

import torch

from auto_round.data_type.utils import (
    compute_optimized_init_scale,
    get_optimized_quant_func,
    get_quant_func,
)


class QDQTool:
    """Quantize-dequantize a candidate weight under the target scheme.

    The constructor takes the scheme-level *fallback* params (used when a layer
    is absent from ``layer_config``). Runtime quant-func-selection flags (the
    resolved ``disable_opt_rtn`` and whether the optimized init-scale path is
    active) are derived from the run's block quantizer by :meth:`configure`; the
    per-layer ``layer_config`` is wired in by the owning caller.
    """

    def __init__(self, *, bits, group_size, sym, data_type) -> None:
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self.data_type = data_type

        # Wired at runtime: ``layer_config`` by the owning caller, the rest by
        # :meth:`configure`.
        self.layer_config: dict | None = None
        self.disable_opt_rtn: bool | None = None
        self.use_v2_scale_search: bool = False

    def configure(self, compressor) -> None:
        """Derive the QDQ behaviour from the run's block quantizer.

        ``disable_opt_rtn`` (importance-weighted RTN) and ``use_v2_scale_search``
        (the SignRoundV2 optimized init-scale path) only influence quant-func
        selection, so they are resolved here from the block quantizer rather than
        threaded in by the calling transform.
        """
        block_config = getattr(compressor, "quantize_config", None)
        self.disable_opt_rtn = bool(getattr(block_config, "disable_opt_rtn", False))
        self.use_v2_scale_search = self._block_quantizer_is_signroundv2(compressor)

    @staticmethod
    def _block_quantizer_is_signroundv2(compressor) -> bool:
        """Return True if the terminal block quantizer is SignRoundV2.

        The block-quantizer config (``compressor.quantize_config``) is resolved
        to its registered quantizer class through the pipeline registry; only
        SignRoundV2 enables the optimized init-scale path.
        """
        block_config = getattr(compressor, "quantize_config", None)
        if block_config is None:
            return False
        from auto_round.algorithms.registry import resolve_pipeline_member

        try:
            return resolve_pipeline_member(block_config).__name__ == "SignRoundV2Quantizer"
        except Exception:
            return False

    def _layer_config_for(self, layer: torch.nn.Module) -> tuple[str, dict]:
        name = getattr(layer, "global_name", None) or ""
        return name, (self.layer_config or {}).get(name, {})

    def resolve_params(self, layer: torch.nn.Module) -> dict:
        """Resolve the per-layer quant params used by the QDQ loss."""
        _, config = self._layer_config_for(layer)
        return {
            "bits": config.get("bits", self.bits),
            "group_size": config.get("group_size", self.group_size),
            "sym": config.get("sym", self.sym),
            "data_type": config.get("data_type", self.data_type),
            "disable_opt_rtn": config.get("disable_opt_rtn", self.disable_opt_rtn),
        }

    def prepare_layer_funcs(self, layer: torch.nn.Module):
        """Pre-resolve ``(quant_func, opt_quant_func)`` once for a layer.

        Hoisting the dispatch out of a per-candidate search loop avoids repeated
        lookups. ``opt_quant_func`` is non-``None`` only when the optimized
        init-scale path applies (depends solely on ``data_type`` / ``sym``, not
        on the candidate weight).
        """
        _, cfg = self._layer_config_for(layer)
        data_type = cfg.get("data_type", self.data_type)
        bits = cfg.get("bits", self.bits)
        sym = cfg.get("sym", self.sym)
        group_size = cfg.get("group_size", self.group_size)
        quant_func, _ = get_quant_func(
            data_type,
            bits,
            sym,
            disable_opt_rtn=cfg.get("disable_opt_rtn", self.disable_opt_rtn),
            group_size=group_size,
            iters=0,
        )

        opt_quant_func = None
        if self.use_v2_scale_search and sym:
            opt_quant_func = get_optimized_quant_func(data_type)
        return quant_func, opt_quant_func

    def qdq(
        self,
        layer: torch.nn.Module,
        weight: torch.Tensor,
        *,
        quant_func=None,
        opt_quant_func=None,
    ) -> torch.Tensor:
        """Quantize-dequantize ``weight`` under ``layer``'s resolved scheme.

        Intended for reference/loss evaluation only; does NOT modify the layer's
        stored weights. ``quant_func`` / ``opt_quant_func`` may be pre-resolved
        by the caller (see :meth:`prepare_layer_funcs`); a non-``None``
        ``opt_quant_func`` enables the optimized init-scale path for this weight.
        """
        layer_name, config = self._layer_config_for(layer)
        bits = config.get("bits", self.bits)
        group_size = config.get("group_size", self.group_size)
        sym = config.get("sym", self.sym)
        data_type = config.get("data_type", self.data_type)
        disable_opt_rtn = config.get("disable_opt_rtn", self.disable_opt_rtn)
        # GGUF double-quant schemes need the per-layer super-block params to
        # reproduce the block quantizer's QDQ. Non-GGUF quant funcs ignore them
        # via ``**kwargs``.
        super_bits = config.get("super_bits", None)
        super_group_size = config.get("super_group_size", None)

        if quant_func is None:
            quant_func, _ = get_quant_func(
                data_type,
                bits,
                sym,
                disable_opt_rtn=disable_opt_rtn,
                group_size=group_size,
                iters=0,
            )

        if quant_func is None:
            raise RuntimeError(
                f"QDQTool: no quantization function resolved for '{layer_name}' "
                f"(data_type={data_type}, bits={bits}, sym={sym}, group_size={group_size})."
            )

        quant_kwargs = {
            "bits": bits,
            "group_size": group_size,
            "data_type": data_type,
            "sym": sym,
        }
        if super_bits is not None:
            quant_kwargs["super_bits"] = super_bits
        if super_group_size is not None:
            quant_kwargs["super_group_size"] = super_group_size

        active_quant_func = quant_func
        if opt_quant_func is not None:
            init_scale = compute_optimized_init_scale(
                weight, data_type, bits, group_size, imatrix=getattr(layer, "imatrix", None)
            )
            if init_scale is not None:
                quant_kwargs["init_scale"] = init_scale
                active_quant_func = opt_quant_func

        qdq_weight, _, _ = active_quant_func(weight, **quant_kwargs)
        return qdq_weight

    def resolve_clip_quant_func(self, params: dict, group_size: int):
        """Resolve the (non-opt) quant func for the clip search.

        Returns the resolved quant func, which may be ``None`` when the scheme
        has no clip-applicable quant func (a valid "not applicable" signal the
        caller skips on). Any resolution failure propagates to the caller.
        """
        quant_func, _ = get_quant_func(
            params["data_type"],
            params["bits"],
            params["sym"],
            disable_opt_rtn=params["disable_opt_rtn"],
            group_size=group_size,
            iters=0,
        )
        return quant_func

    @staticmethod
    @torch.no_grad()
    def clip_qdq(
        cur_w: torch.Tensor,
        quant_func,
        params: dict,
        group_size: int,
    ) -> torch.Tensor:
        """Quantize-dequantize the clamped weight for the clip-search loss."""
        oc_b, _, n_group, gs = cur_w.shape
        flat = cur_w.reshape(oc_b, n_group * gs)
        qdq, _, _ = quant_func(
            flat,
            bits=params["bits"],
            group_size=group_size,
            data_type=params["data_type"],
            sym=params["sym"],
        )
        return qdq.reshape(oc_b, 1, n_group, gs)
