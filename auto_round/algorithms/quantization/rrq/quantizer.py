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
"""RRQ RTN Quantizer: recursive multi-plane RTN quantization for INT2.

Phase 1 implementation: pure RTN (no sign-SGD tuning).  For each layer,
performs K rounds of RTN quantization where each round quantizes the
residual of the previous round.  The result is K independent INT2 planes
whose prefix sum reconstructs the weight at increasing precision.
"""

import torch
import torch.nn as nn

from auto_round.algorithms.quantization.base import BaseQuantizer
from auto_round.algorithms.quantization.rrq.config import RRQConfig
from auto_round.algorithms.registry import register_pipeline_member
from auto_round.data_type.utils import get_quant_func
from auto_round.utils import check_to_quantized


def _rrq_quant_linear_class(sym: bool):
    """Return the W2A16 ``QuantLinear`` class used to pack residual planes.

    RRQ uses weight-only INT2 (``act_bits=16``), so the class selection mirrors
    the standard ``auto_round`` export path exactly (see
    ``dynamic_import_quant_linear_for_packing``):

    - symmetric  -> ``auto_round:auto_gptq`` -> ``qlinear_torch_zp.QuantLinear``
    - asymmetric -> ``auto_round``            -> ``qlinear_torch.QuantLinear``

    Using the *same* class as the base model export guarantees the residual
    ``qweight``/``scales``/``qzeros`` tensors round-trip through the exact
    W2A16 pack/dequant code path.
    """
    if sym:
        from auto_round_extension.torch.qlinear_torch_zp import QuantLinear
    else:
        from auto_round_extension.torch.qlinear_torch import QuantLinear

    return QuantLinear


@register_pipeline_member(RRQConfig)
class RRQRTNQuantizer(BaseQuantizer):
    """Quantizer that produces multi-plane INT2 weights via iterative RTN.

    For each target layer:
        1. Quantize W  -> plane_0 (base), residual E_1 = W - QDQ(plane_0)
        2. Quantize E_1 -> plane_1, residual E_2 = W - QDQ(p0) - QDQ(p1)
        3. Quantize E_2 -> plane_2, residual E_3 = ...
        4. Quantize E_3 -> plane_3 (final residual)

    All planes share the same group_size and sym setting.
    Phase 1 uses pure RTN (no sign-SGD tuning).

    Storage:
        - plane 0 (base) is stored as the *dequantized float* weight in
          ``layer.weight`` plus ``layer.scale``/``layer.zp`` (standard INT2
          layout, exported via the regular ``auto_round`` path).
        - planes 1..K-1 (residual) are stored as *packed INT2* tensors
          (``rrq_qweight_k`` int32 + ``rrq_scales_k`` + ``rrq_qzeros_k``),
          produced by packing the dequantized residual plane through the W2A16
          ``QuantLinear.pack`` so the on-disk layout is identical to a single
          INT2 AutoRound model.  They are packed into a single
          ``auto_round:rrq`` artifact (all three planes together).
    """

    def __init__(self, config: RRQConfig) -> None:
        super().__init__(config)
        self.num_planes = config.total_planes  # 4 for Phase 1 (1 base + 3 residual)
        # Note: ``scale_dtype`` is a read-only property inherited from
        # ``BaseAlgorithm`` (sourced from the run context), so store our own
        # default in a distinct attribute to avoid an ``AttributeError``.
        self._rrq_scale_dtype = (
            config.scale_dtype if getattr(config, "scale_dtype", None) is not None else torch.float16
        )
        self._quant_linear = _rrq_quant_linear_class(config.sym)

    def _get_quant_func(self, bits: int, group_size: int):
        """Get the RTN quantization function for the given bits and group size.

        Uses plain RTN (disable_opt_rtn=True) for deterministic,
        calibration-free quantization suitable for each individual plane.
        """
        quant_func, _ = get_quant_func(
            dtype="int",
            bits=bits,
            sym=self.config.sym,
            disable_opt_rtn=True,
            group_size=group_size,
            iters=0,
        )
        return quant_func

    def _normalize_scale_zp(self, scale, zp, out_features: int):
        """Normalise RTN ``(scale, zp)`` to the shapes ``QuantLinear.pack`` expects.

        RTN returns ``scale`` as ``(out, num_groups, 1)`` and ``zp`` either as a
        Python ``int`` (symmetric -- the ``maxq`` scalar) or a ``(out, num_groups,
        1)`` tensor (asymmetric).  ``QuantLinear.pack`` (``pack_248_bits``)
        expects ``scale`` as ``(out, num_groups)`` and ``zp`` as an ``int``
        (symmetric) or ``(out, num_groups)`` tensor (asymmetric).  This reshape
        is exactly what the standard ``unwrapper`` does before ``pack_layer``.
        """
        if isinstance(scale, torch.Tensor):
            scale = scale.reshape(out_features, -1).to(self._rrq_scale_dtype)
        else:
            scale = torch.tensor(scale, dtype=self._rrq_scale_dtype)
        if isinstance(zp, torch.Tensor):
            zp = zp.reshape(out_features, -1)
        # otherwise: zp is a Python int (symmetric) or None -> pass through
        return scale, zp

    def _pack_plane(self, dequant: torch.Tensor, scale, zp, bits: int, group_size: int, in_features: int):
        """Pack a dequantized residual plane (``out, in``) into W2A16 INT2 tensors.

        Returns the packed ``(qweight, scales, qzeros)`` tensors in the exact
        layout the standard ``auto_round`` export produces, so each plane is
        bit-compatible with a single-plane INT2 AutoRound model and dequantizes
        through the stock ``QuantLinear.forward``.

        The ``scale``/``zp`` passed in are the raw RTN outputs; they are reshaped
        via :meth:`_normalize_scale_zp` to the ``pack`` convention.  Feeding the
        RTN scale/zp makes the pack self-consistent: ``code = round(W/scale + zp)``
        and ``QuantLinear.forward`` recovers ``scale * (code - zp)`` -- the RTN
        dequantized plane (exact for both symmetric and asymmetric).
        """
        out_features = dequant.shape[0]
        scale, zp = self._normalize_scale_zp(scale, zp, out_features)
        ql = self._quant_linear(bits, group_size, in_features, out_features, bias=False)
        plane_linear = nn.Linear(in_features, out_features, bias=False)
        plane_linear.weight.data = dequant.detach().clone()
        # Force float32 to be compatible with the pack math (torch 2.0).
        plane_linear.to(torch.float32)

        pack_device = plane_linear.weight.device
        try:
            from auto_round.utils.device_manager import get_packing_device

            pack_device = get_packing_device(pack_device)
        except Exception:  # pragma: no cover - fall back to the weight device
            pack_device = plane_linear.weight.device

        # ``pack(linear, scales, zeros, g_idx, device)``.  ``scale``/``zp`` are
        # already normalised to the ``pack`` convention by
        # :meth:`_normalize_scale_zp` (scale: (out, num_groups); zp: int for
        # symmetric, (out, num_groups) tensor for asymmetric) -- exactly what
        # the standard ``pack_layer`` feeds to this ``pack_248_bits`` path.
        ql.to("cpu")
        ql.pack(plane_linear, scale, zp, None, device=pack_device)
        ql.to("cpu")
        return ql.qweight.detach(), ql.scales.detach(), ql.qzeros.detach()

    @torch.no_grad()
    def quantize_block(
        self,
        block,
        fp_inputs,
        input_others,
        fp_outputs,
        q_inputs,
        block_ctx,
        input_ids=None,
        **kwargs,
    ) -> dict:
        """Apply recursive RTN quantization to all eligible layers in a block."""
        for _name, m in block.named_modules():
            if check_to_quantized(m):
                self._quantize_layer_rrq(m)
        return {}

    def quantize_layer_outside_block(
        self, layer, fp_inputs=None, q_inputs=None, disable_opt_rtn=None, input_ids=None
    ) -> None:
        """Quantize a single layer outside the block with RRQ."""
        if check_to_quantized(layer):
            self._quantize_layer_rrq(layer)

    def _quantize_layer_rrq(self, layer: torch.nn.Module) -> None:
        """Quantize a single layer into K INT2 planes using iterative RTN.

        Stores results on the module:
            - layer.weight, layer.scale, layer.zp  -> base plane (plane 0)
            - layer.rrq_qweight_k, layer.rrq_scales_k, layer.rrq_zp_k
              for k = 1..(num_planes-1)
        """
        original_weight = layer.weight.data.clone().to(torch.float32)
        device = original_weight.device
        group_size = layer.group_size
        bits = self.config.bits  # 2
        sym = layer.sym

        quant_func = self._get_quant_func(bits, group_size)

        accumulated = torch.zeros_like(original_weight)

        for plane_idx in range(self.num_planes):
            residual = original_weight - accumulated

            quantized, scale, zp = quant_func(
                residual,
                bits=bits,
                group_size=group_size,
                scale_dtype=self._rrq_scale_dtype,
                q_scale_thresh=1e-5,
            )

            quantized = quantized.to(device)
            accumulated = accumulated + quantized

            # Normalise scale/zp to the shapes the standard export feeds to
            # ``QuantLinear.pack`` (scale: (out, num_groups); zp: int for sym,
            # (out, num_groups) tensor for asym).  Stored as-is so the base
            # plane round-trips through the stock W2A16 pack/dequant path.
            scale_n, zp_n = self._normalize_scale_zp(scale, zp, original_weight.shape[0])

            if plane_idx == 0:
                layer.weight.data.copy_(quantized.to(layer.weight.data.dtype))
                layer.scale = scale_n.cpu()
                # symmetric: zp is the int ``maxq`` centre offset (kept as a
                # scalar so the standard ``pack`` scalar path is used);
                # asymmetric: (out, num_groups) tensor.
                layer.zp = zp_n.to(device) if isinstance(zp_n, torch.Tensor) else zp_n
            else:
                # Residual plane: store packed INT2 (W2A16 layout) so the
                # on-disk artifact is a standard single-plane INT2 layout.
                in_features = original_weight.shape[1]
                qweight, scales, qzeros = self._pack_plane(
                    quantized, scale, zp, bits, group_size, in_features
                )
                layer.register_buffer(f"rrq_qweight_{plane_idx}", qweight.cpu())
                layer.register_buffer(f"rrq_scales_{plane_idx}", scales.to(self._rrq_scale_dtype).cpu())
                layer.register_buffer(f"rrq_qzeros_{plane_idx}", qzeros.cpu())

        layer.rrq_total_planes = self.num_planes
        layer.rrq_bit_width = bits
        layer.rrq_group_size = group_size
        layer.rrq_sym = sym
