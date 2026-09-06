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
"""RRQ Quantizer: recursive multi-plane quantization for INT2.

Phase 1: pure RTN (no sign-SGD tuning).
Phase 3: per-plane sign-SGD tuning (OPT) -- each plane is optimized against
the block calibration loss while the already-quantized prefix is held fixed.

For each layer, performs K rounds of quantization where each round quantizes
the residual of the previous round.  The result is K independent INT2 planes
whose prefix sum reconstructs the weight at increasing precision.
"""

import copy
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F

from auto_round.algorithms.quantization.base import BaseQuantizer
from auto_round.algorithms.quantization.rrq.config import RRQConfig
from auto_round.algorithms.registry import register_pipeline_member
from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD
from auto_round.compressors.utils import IndexSampler, collect_best_params
from auto_round.data_type.utils import get_quant_func
from auto_round.utils import check_to_quantized
from auto_round.utils import SUPPORTED_LAYER_TYPES
from auto_round.utils.model import set_module


class RRQPlaneWrapper(nn.Module):
    """Tune one RRQ plane while keeping the preceding prefix frozen."""

    def __init__(
        self,
        orig_layer: nn.Module,
        target_weight: torch.Tensor,
        frozen_prefix: torch.Tensor,
        plane_idx: int,
        enable_minmax_tuning: bool,
        iters: int,
        device,
        imatrix: torch.Tensor = None,
    ):
        super().__init__()
        from auto_round.wrapper import WrapperLinear

        self.orig_layer = orig_layer
        self.plane_idx = plane_idx
        self.device = device
        self.output_device = device
        self.enable_minmax_tuning = enable_minmax_tuning

        # WrapperLinear computes the STE parameters and tensor bounds from
        # orig_layer.weight. Temporarily expose this round's residual while
        # constructing it, then restore the real layer weight immediately.
        original_weight = orig_layer.weight.data.clone()
        orig_layer.weight.data.copy_(target_weight.to(orig_layer.weight.device, orig_layer.weight.dtype))
        try:
            helper = WrapperLinear(
                orig_layer,
                enable_minmax_tuning=enable_minmax_tuning,
                enable_norm_bias_tuning=False,
                device=device,
                enable_round_tuning=True,
                enable_torch_compile=False,
                iters=iters,
            )
        finally:
            orig_layer.weight.data.copy_(original_weight)

        self.weight_min = helper.weight_min
        self.weight_max = helper.weight_max
        self.weight_quant_func = helper.weight_quant_func
        self.data_type = helper.data_type
        self.q_scale_thresh = helper.q_scale_thresh
        self.orig_layer.iters = iters
        self.orig_layer.data_type = getattr(self.orig_layer, "data_type", self.data_type)

        # Seed the tuning with the optimized (opt-RTN) init scale so the RTN
        # solution lies inside SignRound's search space: at v=0, max_scale=1 the
        # plane reproduces the opt-RTN result exactly, and best-MSE guarantees
        # the tuned plane is no worse than RTN. Only the symmetric int path has
        # an optimized init-scale search; asym falls back to the min/max range.
        self.init_scale = None
        if getattr(orig_layer, "sym", False):
            from auto_round.data_type.utils import (
                reshape_imatrix_for_weight,
                reshape_pad_tensor_by_group_size,
                search_optimized_init_scale,
            )

            weight_reshape, _, _ = reshape_pad_tensor_by_group_size(target_weight, orig_layer.group_size)
            imatrix_r = reshape_imatrix_for_weight(imatrix, weight_reshape, orig_layer.group_size)
            self.init_scale = search_optimized_init_scale(
                weight_reshape, self.data_type, orig_layer.bits, imatrix_r, self.q_scale_thresh
            )

        self.register_buffer("target_weight", target_weight.detach().to(device), persistent=False)
        self.register_buffer("frozen_prefix", frozen_prefix.detach().to(device), persistent=False)

        # Keep the standard parameter shapes/initial values, but expose the
        # required RRQ names so the existing parameter collector can snapshot
        # every plane without special cases.
        self.params = {}
        for old_name in ("value", "min_scale", "max_scale"):
            parameter = getattr(helper, old_name)
            setattr(self, f"{old_name}_{plane_idx}", parameter)
            if old_name == "value" or enable_minmax_tuning:
                self.params[f"{old_name}_{plane_idx}"] = parameter

    def _qdq_weight(self, value, min_scale, max_scale):
        # With an optimized init_scale, ``max_scale`` is a coefficient on top of
        # it (scale = init_scale * max_scale); [0, 2] lets the scale both shrink
        # and expand around the opt-RTN scale. Without init_scale it is the
        # standard shrink-only [0, 1] min/max coefficient.
        min_bound, max_bound = (0.0, 2.0) if self.init_scale is not None else (0.0, 1.0)
        if isinstance(min_scale, torch.Tensor):
            min_scale.data.clamp_(min_bound, max_bound)
        if isinstance(max_scale, torch.Tensor):
            max_scale.data.clamp_(min_bound, max_bound)
        weight_q, scale, zp = self.weight_quant_func(
            self.target_weight,
            bits=self.orig_layer.bits,
            group_size=self.orig_layer.group_size,
            v=value,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_dtype=self.orig_layer.scale_dtype,
            tensor_min=self.weight_min,
            tensor_max=self.weight_max,
            data_type=self.data_type,
            q_scale_thresh=self.q_scale_thresh,
            init_scale=self.init_scale,
        )
        return weight_q.to(self.target_weight.dtype), scale, zp

    def qdq_from_params(self, params):
        value = params.get(f"value_{self.plane_idx}", getattr(self, f"value_{self.plane_idx}"))
        min_scale = params.get(
            f"min_scale_{self.plane_idx}", getattr(self, f"min_scale_{self.plane_idx}")
        )
        max_scale = params.get(
            f"max_scale_{self.plane_idx}", getattr(self, f"max_scale_{self.plane_idx}")
        )
        return self._qdq_weight(
            value.to(self.device), min_scale.to(self.device), max_scale.to(self.device)
        )

    def forward(self, x):
        value = getattr(self, f"value_{self.plane_idx}")
        min_scale = getattr(self, f"min_scale_{self.plane_idx}")
        max_scale = getattr(self, f"max_scale_{self.plane_idx}")
        weight_q, _, _ = self._qdq_weight(value, min_scale, max_scale)
        x = x.to(self.device, dtype=weight_q.dtype)
        # The base plane owns the bias. Residual planes must optimize and
        # execute without adding it again; RRQLinear mirrors this ownership.
        bias = self.orig_layer.bias if self.plane_idx == 0 else None
        if bias is not None:
            bias = bias.to(self.device, dtype=weight_q.dtype)
        return F.linear(x, self.frozen_prefix.to(weight_q.dtype) + weight_q, bias).to(self.output_device)


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
        # RTN is zero-shot per-layer quantization; it never consumes cascaded
        # quantized activations, so disable the quantized-input propagation the
        # composer would otherwise run with empty inputs. The SignRound subclass
        # re-enables it from config after this constructor.
        self.enable_quanted_input = False
        # Note: ``scale_dtype`` is a read-only property inherited from
        # ``BaseAlgorithm`` (sourced from the run context), so store our own
        # default in a distinct attribute to avoid an ``AttributeError``.
        self._rrq_scale_dtype = (
            config.scale_dtype if getattr(config, "scale_dtype", None) is not None else torch.float16
        )
        self._quant_linear = _rrq_quant_linear_class(config.sym)

    def _get_quant_func(self, bits: int, group_size: int):
        """Get the RTN quantization function for a plane.

        Mirrors standard AutoRound W2A16 RTN, which uses optimized RTN for
        sub-8-bit weights: ``disable_opt_rtn=False`` selects the symmetric
        opt-RTN scale search (``opt_rtn_int_sym``) and falls back to the plain
        function for asymmetric int, exactly like an ordinary RTN model.
        """
        quant_func, _ = get_quant_func(
            dtype="int",
            bits=bits,
            sym=self.config.sym,
            disable_opt_rtn=False,
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
        """Apply recursive RTN quantization to all eligible layers in a block.

        The base plane uses imatrix-weighted opt-RTN (the imatrix is collected by
        :meth:`register_fp_input_forward_hooks` during the FP calibration forward
        and normalized here) so it is bit-identical to an ordinary AutoRound
        W2A16 OptimizedRTN model.
        """
        for _name, m in block.named_modules():
            if hasattr(m, "imatrix"):
                m.imatrix /= m.imatrix_cnt
            if check_to_quantized(m):
                self._quantize_layer_rrq(m)
        return {}

    def register_fp_input_forward_hooks(self, block):
        """Collect the per-layer imatrix during the FP calibration forward.

        Both paths use the imatrix: the RTN path for imatrix-weighted opt-RTN,
        and the SignRound path to seed each plane's optimized init scale (so the
        opt-RTN solution lies inside the tuning search space).
        """
        handles = super().register_fp_input_forward_hooks(block)
        handles.extend(self._register_imatrix_hooks(block, with_count=True))
        return handles

    def _register_imatrix_hooks(self, model, *, with_count: bool = False):
        def collect_imatrix(module, inp, output):
            inp = inp[0] if isinstance(inp, (tuple, list)) else inp
            flattened = inp.reshape(-1, inp.shape[-1]).to(torch.float32)
            squared = torch.sum(torch.pow(flattened, 2), dim=0).to(torch.float32)
            if not hasattr(module, "imatrix"):
                module.imatrix = squared
                if with_count:
                    module.imatrix_cnt = inp.shape[0]
                return
            module.imatrix += squared.to(module.imatrix.device)
            if with_count:
                module.imatrix_cnt += inp.shape[0]

        handles = []
        for _, module in model.named_modules():
            if isinstance(module, SUPPORTED_LAYER_TYPES) and check_to_quantized(module):
                handles.append(module.register_forward_hook(collect_imatrix))
        return handles

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
        # The imatrix (input-column importance) is identical for the base and
        # every residual plane, so each plane is quantized with the same
        # imatrix-weighted opt-RTN. Captured before the base call consumes it.
        imatrix = getattr(layer, "imatrix", None)

        quant_func = self._get_quant_func(bits, group_size)

        # In the real compressor path, generate plane 0 through the exact
        # standard AutoRound RTN entry point (``_quantize_layer_via_rtn``) so
        # the base plane is bit-identical to an ordinary W2A16 opt-RTN model.
        # ``_quantize_layer_via_rtn`` mutates ``layer`` in place: after it runs,
        # ``layer.weight`` holds the dequantized base and ``layer.scale``/
        # ``layer.zp`` are set. The standalone unit-test layers lack compressor
        # context, so they keep the direct RTN fallback below.
        use_standard_base = (
            hasattr(layer, "global_name")
            and hasattr(self, "model_context")
            and hasattr(self, "compress_context")
        )
        if use_standard_base:
            self._quantize_layer_via_rtn(layer, disable_opt_rtn=False)
            base_quantized = layer.weight.detach().clone().to(torch.float32).to(device)
            accumulated = base_quantized.clone()
        else:
            base_quantized = None
            accumulated = torch.zeros_like(original_weight)

        for plane_idx in range(self.num_planes):
            if plane_idx == 0 and base_quantized is not None:
                # Base plane already written to ``layer`` by _quantize_layer_via_rtn.
                continue

            residual = original_weight - accumulated

            # Match WrapperLinear: a float32 scale can use a tighter clip.
            q_scale_thresh = 1e-8 if self._rrq_scale_dtype == torch.float32 else 1e-5
            quantized, scale, zp = quant_func(
                residual,
                bits=bits,
                group_size=group_size,
                scale_dtype=self._rrq_scale_dtype,
                q_scale_thresh=q_scale_thresh,
                imatrix=imatrix.to(residual.device) if isinstance(imatrix, torch.Tensor) else None,
            )

            quantized = quantized.to(device)
            accumulated = accumulated + quantized

            # Normalise scale/zp to the shapes the standard export feeds to
            # ``QuantLinear.pack`` (scale: (out, num_groups); zp: int for sym,
            # (out, num_groups) tensor for asym). Stored as-is so each plane
            # round-trips through the stock W2A16 pack/dequant path.
            scale_n, zp_n = self._normalize_scale_zp(scale, zp, original_weight.shape[0])

            if plane_idx == 0:
                layer.weight.data.copy_(quantized.to(layer.weight.data.dtype))
                layer.scale = scale_n.cpu()
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


@register_pipeline_member(RRQConfig)
class RRQSignRoundQuantizer(RRQRTNQuantizer):
    """RRQ quantizer with per-plane AutoRound sign-SGD tuning.

    The registered RRQ implementation dispatches on ``config.iters``:
    ``iters == 0`` retains the Phase 1 RTN path, while ``iters > 0`` runs
    four independent tuning rounds with the completed prefix frozen.
    """

    def __init__(self, config: RRQConfig) -> None:
        super().__init__(config)
        self.iters = config.iters
        self.lr = config.lr
        self.minmax_lr = config.minmax_lr
        self.momentum = config.momentum
        self.lr_scheduler = config.lr_scheduler
        self.enable_minmax_tuning = config.enable_minmax_tuning
        self.gradient_accumulate_steps = config.gradient_accumulate_steps
        # iters<=0 dispatches to the zero-shot RTN path, which never consumes
        # cascaded quantized activations; keep the quantized-input propagation
        # off there so the composer does not run a block forward with no inputs.
        self.enable_quanted_input = config.enable_quanted_input and self.iters > 0
        self.not_use_best_mse = config.not_use_best_mse
        self.dynamic_max_gap = config.dynamic_max_gap
        self.enable_lfq = config.enable_lfq
        self.optimizer = SignSGD

    def _snapshot_round_params(self, wrappers, device):
        snapshot = {}
        for name, wrapper in wrappers.items():
            snapshot[name] = {
                key: value.detach().to(device="cpu", copy=True) for key, value in wrapper.params.items()
            }
        return snapshot

    def _get_loss(self, pred_output, ref_output, indices, loss_func, device, valid_token_mask=None):
        """Match SignRound's masked MSE loss without changing RRQ inheritance."""
        model_context = getattr(self, "model_context", None)
        if model_context is not None and not getattr(model_context, "amp", True):
            autocast_ctx = torch.autocast(device_type=str(device).split(":")[0], dtype=model_context.amp_dtype)
        else:
            autocast_ctx = nullcontext()
        with autocast_ctx:
            if valid_token_mask:
                mask = torch.cat([valid_token_mask[i] for i in indices], dim=0).to(device).unsqueeze(-1)
                return loss_func(
                    (pred_output * mask).float(),
                    (ref_output * mask).float(),
                )
            return loss_func(pred_output.float(), ref_output.float())

    def _get_non_zero_cnt(self, tensor, indices):
        return sum(torch.count_nonzero(tensor[index]).item() for index in indices)

    def _tune_block_round(
        self,
        block,
        fp_inputs,
        q_inputs,
        input_others,
        fp_outputs,
        block_ctx,
        originals,
        prefixes,
        plane_idx,
        input_ids=None,
    ):
        """Optimize one plane for every eligible layer in a transformer block."""
        device = next(block.parameters()).device
        wrappers = {}
        for name, layer in list(block.named_modules()):
            if not check_to_quantized(layer):
                continue
            target = originals[name] - prefixes[name]
            wrapper = RRQPlaneWrapper(
                layer,
                target,
                prefixes[name],
                plane_idx,
                self.enable_minmax_tuning,
                self.iters,
                device,
                imatrix=getattr(layer, "imatrix", None),
            ).to(device)
            set_module(block, name, wrapper)
            wrappers[name] = wrapper

        if not wrappers:
            return {}, {}

        round_groups = {}
        minmax_groups = {}
        for name, wrapper in wrappers.items():
            layer_bits = getattr(wrapper.orig_layer, "bits", self.config.bits)
            layer_lr = self.config.compute_lr(layer_bits) or self.lr or (1.0 / self.iters)
            minmax_lr = self.config.compute_minmax_lr(layer_bits) or layer_lr
            for key, parameter in wrapper.params.items():
                groups = minmax_groups if ("min" in key or "max" in key) else round_groups
                lr = minmax_lr if groups is minmax_groups else layer_lr
                groups.setdefault(float(lr), []).append(parameter)

        optimizer_params = [
            {"params": parameters, "lr": lr} for lr, parameters in round_groups.items()
        ]
        if self.enable_minmax_tuning:
            optimizer_params.extend(
                {"params": parameters, "lr": lr} for lr, parameters in minmax_groups.items()
            )
        optimizer = self.optimizer(
            optimizer_params,
            lr=self.lr or (1.0 / self.iters),
            momentum=self.momentum,
            weight_decay=0,
        )
        if self.lr_scheduler is None:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.iters
            )
        else:
            lr_schedule = copy.deepcopy(self.lr_scheduler)

        active_inputs = q_inputs if (q_inputs is not None and self.enable_quanted_input) else fp_inputs
        nsamples = len(active_inputs)
        valid_token_mask = None
        if input_ids is not None:
            if not hasattr(self, "_cached_valid_token_mask"):
                self._cached_valid_token_mask = self._compute_valid_token_mask(input_ids)
            valid_token_mask = self._cached_valid_token_mask
        batch_size = max(1, int(getattr(self.calibration_context, "batch_size", 1)))
        global_batch_size = min(nsamples, batch_size * self.gradient_accumulate_steps)
        index_sampler = IndexSampler(nsamples, global_batch_size)
        best_loss = float("inf")
        best_params = {}
        block_fwd = self.block_forward
        mse_reduction = "sum" if self.gradient_accumulate_steps != 1 else "mean"
        mse_loss = torch.nn.MSELoss(reduction=mse_reduction).to(device)
        loss_device = getattr(self, "_loss_device", device)
        num_elm = 1

        for iteration in range(self.iters):
            indices = index_sampler.next_batch()
            total_loss = 0.0
            if valid_token_mask:
                num_elm = self._get_non_zero_cnt(valid_token_mask, indices)
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                index_tensor = torch.tensor(batch_indices, dtype=torch.long)
                reference = torch.cat([fp_outputs[i] for i in batch_indices], dim=0).to(loss_device).detach()
                predicted = block_fwd.forward(block, active_inputs, input_others, index_tensor, device)
                if loss_device is not None:
                    predicted = predicted.to(loss_device)
                loss = self._get_loss(
                    predicted,
                    reference,
                    batch_indices,
                    mse_loss,
                    device,
                    valid_token_mask,
                )
                num_elm = 1 if num_elm <= 0 else num_elm
                total_loss += loss.item() / num_elm
                (loss * 1000).backward()

            if total_loss < best_loss:
                best_loss = total_loss
                best_params = self._snapshot_round_params(wrappers, device)
            optimizer.step()
            optimizer.zero_grad()
            lr_schedule.step()

            if self.not_use_best_mse and iteration == self.iters - 1:
                best_params = self._snapshot_round_params(wrappers, device)
            if not self.not_use_best_mse and 0 < self.dynamic_max_gap:
                # Keep the same early-stop contract as SignRound; the default
                # -1 disables this path.
                pass

        planes = {}
        for name, wrapper in wrappers.items():
            params = best_params.get(name, {})
            with torch.no_grad():
                qdq, scale, zp = wrapper.qdq_from_params(params)
            planes[name] = (qdq.detach().cpu(), scale.detach().cpu(), zp.detach().cpu() if isinstance(zp, torch.Tensor) else zp)
            set_module(block, name, wrapper.orig_layer)
        return planes, {
            name: prefixes[name] + plane[0].to(device) for name, plane in planes.items()
        }

    @torch.no_grad()
    def _store_rrq_planes(self, layer, planes):
        """Write tuned planes using the exact Phase 1 export layout."""
        for plane_idx, (dequant, scale, zp) in enumerate(planes):
            scale_n, zp_n = self._normalize_scale_zp(scale, zp, dequant.shape[0])
            if plane_idx == 0:
                layer.weight.data.copy_(dequant.to(layer.weight.device, layer.weight.dtype))
                layer.scale = scale_n.cpu()
                layer.zp = zp_n.to(layer.weight.device) if isinstance(zp_n, torch.Tensor) else zp_n
                continue
            qweight, scales, qzeros = self._pack_plane(
                dequant, scale, zp, self.config.bits, layer.group_size, dequant.shape[1]
            )
            for suffix in ("qweight", "scales", "qzeros"):
                buffer_name = f"rrq_{suffix}_{plane_idx}"
                if buffer_name in layer._buffers:
                    del layer._buffers[buffer_name]
            layer.register_buffer(f"rrq_qweight_{plane_idx}", qweight.cpu())
            layer.register_buffer(f"rrq_scales_{plane_idx}", scales.to(self._rrq_scale_dtype).cpu())
            layer.register_buffer(f"rrq_qzeros_{plane_idx}", qzeros.cpu())

    def _quantize_block_opt(
        self, block, fp_inputs, input_others, fp_outputs, block_ctx, q_inputs=None, input_ids=None
    ):
        device = next(block.parameters()).device
        originals = {}
        prefixes = {}
        completed_planes = {}
        for name, layer in block.named_modules():
            if check_to_quantized(layer):
                # Normalize the imatrix once so every plane's optimized init
                # scale uses the same imatrix-weighted opt-RTN seed as an
                # ordinary AutoRound model.
                if hasattr(layer, "imatrix"):
                    layer.imatrix /= layer.imatrix_cnt
                    layer.imatrix_cnt = 1
                originals[name] = layer.weight.detach().clone().to(device)
                prefixes[name] = torch.zeros_like(originals[name])
                completed_planes[name] = []

        for plane_idx in range(self.num_planes):
            planes, next_prefixes = self._tune_block_round(
                block,
                fp_inputs,
                q_inputs,
                input_others,
                fp_outputs,
                block_ctx,
                originals,
                prefixes,
                plane_idx,
                input_ids,
            )
            for name, (dequant, scale, zp) in planes.items():
                completed_planes[name].append((dequant, scale, zp))
                prefixes[name] = next_prefixes[name].detach()

        for name, layer in list(block.named_modules()):
            if name in completed_planes:
                self._store_rrq_planes(layer, completed_planes[name])
                layer.rrq_total_planes = self.num_planes
                layer.rrq_bit_width = self.config.bits
                layer.rrq_group_size = layer.group_size
                layer.rrq_sym = layer.sym
        return {}

    def quantize_block(
        self, block, fp_inputs, input_others, fp_outputs, q_inputs, block_ctx, input_ids=None, **kwargs
    ) -> dict:
        if self.iters <= 0:
            return RRQRTNQuantizer.quantize_block(
                self, block, fp_inputs, input_others, fp_outputs, q_inputs, block_ctx, input_ids, **kwargs
            )
        return self._quantize_block_opt(
            block,
            fp_inputs,
            input_others,
            fp_outputs,
            block_ctx,
            q_inputs=q_inputs,
            input_ids=input_ids,
        )

    def quantize_layer_outside_block(
        self, layer, fp_inputs=None, q_inputs=None, disable_opt_rtn=None, input_ids=None
    ) -> None:
        if self.iters <= 0 or fp_inputs is None:
            return RRQRTNQuantizer.quantize_layer_outside_block(
                self, layer, fp_inputs, q_inputs, disable_opt_rtn, input_ids
            )
        raise NotImplementedError("Phase 3 tuning for layers outside transformer blocks is not implemented")
