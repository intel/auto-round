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

"""Grouped experts forward for unfused (per-expert ``nn.Linear``) MoE modules.

Motivation
----------
``linear_loop_experts_forward`` iterates ``range(num_experts)`` in Python. Only the
experts that were actually routed to do any work -- the rest are skipped -- but *every*
iteration still pays ``nonzero()`` + ``sample_idx.numel()``, and that ``.numel()`` on a
data-dependent ``nonzero()`` result is a host<->device sync. For the experts that do have
tokens it then runs an ``index_select`` gather, three small GEMMs, and -- the expensive
part -- a separate ``weight_quant_func`` call, because each expert's ``WrapperLinear``
fake-quantizes its own weight. Block-wise tuning pays all of that on every forward *and*
backward of every calibration step.

This module keeps the math identical but restructures the layer around two independent
wins. Measured on A100 (sm80, bf16, W4G128, 2048x768 experts, 4096 routed pairs), speedup
over ``linear_loop`` for 16/32/64/128/256 experts:

    routing/GEMM only          1.09  1.09  1.21  1.10  1.12   (forward+backward)
    + fused fake-quantization  4.36  4.70  5.32  5.14  5.24

So the fused fake-quant is the dominant term. The routing rework looks small here only
because the per-expert quant dominates the total: on plain fp weights, where nothing hides
the GEMM, the same rework is worth 1.90-2.42x on its own.

How it works:

1. Flatten the ``(num_tokens, top_k)`` routing into ``S = num_tokens * top_k`` pairs.
2. ``sort`` the pairs by expert id so every expert's tokens are one contiguous slice --
   a plain view, so no ``nonzero`` and no gather kernel.
3. Take a **single** sync (``unique_consecutive``) to learn which experts are hit and how
   many rows each got, instead of one per expert. Experts with no tokens never appear.
4. **Fuse the fake-quantization**: concatenate the active experts' weights *and* their
   tuning tensors (``value``, ``min_scale``, ``max_scale``, ``weight_min``/``weight_max``,
   SignRoundV2's ``init_scale``) into one big tensor and call ``weight_quant_func`` once
   for the whole projection instead of once per expert. AutoRound's quantizers derive
   their scales per row of the group-reshaped tensor, so rows stay independent and the
   result is bit-identical; the backward likewise becomes a single big graph. See
   ``_batched_qdq_weights``.
5. Run the routed GEMM: by default one ``F.linear`` per *active* expert over its
   contiguous slice; ``AR_MOE_GROUPED_MM=1`` switches to torch's ``grouped_mm`` kernel
   instead (see the note above ``_native_grouped_mm_available`` for why the loop wins).
6. Scatter back and reduce over ``top_k``.

Gradients flow through the fake-quantized weights exactly as in the loop version, so this
is usable during tuning and not only for calibration/inference.

Fallbacks
---------
The grouped path only kicks in when the layer really is a stack of plain ``nn.Linear`` /
``WrapperLinear`` projections that all share one quantization scheme. Anything else
transparently falls back to ``linear_loop_experts_forward``:

* **mixed-bit MoE** — experts (or projections) quantized with different
  bits/group_size/data_type/sym, a mix of quantized and left-in-16-bit experts, or a mix
  of wrapper classes;
* Conv1D / LinearAllreduce experts, which need their own forward;
* forward hooks on the expert layers (e.g. online Hadamard rotation);
* activation quantization that is not row-independent (static ranges, per-tensor groups,
  per-expert tunable min/max, NVFP per-layer global scales);
* experts spread over different devices.

Step 4 has its own, narrower eligibility check on top of that. It covers plain group-wise
quantization, per-tensor (``group_size == 0``), 2-D blocks whose grid does not straddle
experts, NVFP ``global_scale`` and per-expert ``imatrix``; it steps aside (keeping only the
grouped GEMM) for GGUF super-blocks that would span two experts and for wrappers that
override ``_qdq_weight`` (GGUF double-quant search, AutoScheme scoring).
See ``_supports_batched_qdq``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import torch
import torch.nn.functional as F
from torch import nn

from auto_round import envs
from auto_round.modeling.fused_moe.utils import build_forced_routing, force_all_experts_routing_enabled
from auto_round.utils import logger

# Expert implementation name registered into transformers' ``ALL_EXPERTS_FUNCTIONS``.
GROUPED_LINEAR_IMPL = "linear_grouped"

# Set once if the native grouped_mm kernel raises; afterwards we always use the sliced loop.
_NATIVE_GROUPED_MM_DISABLED = False
_LOGGED_FALLBACK_REASONS: set[str] = set()


def _log_fallback_once(reason: str) -> None:
    if reason not in _LOGGED_FALLBACK_REASONS:
        _LOGGED_FALLBACK_REASONS.add(reason)
        logger.debug(f"[MoE grouped] falling back to linear_loop: {reason}")


# --------------------------------------------------------------------------------------
# Layer introspection helpers
# --------------------------------------------------------------------------------------


_WRAPPER_LINEAR_CLS: type | None = None


def _is_wrapper_linear(layer: nn.Module) -> bool:
    """``isinstance(layer, WrapperLinear)`` without importing wrapper.py at module import.

    The class is resolved once and cached: this runs a few times per expert per forward,
    and a MoE layer can have hundreds of experts.
    """
    global _WRAPPER_LINEAR_CLS
    if _WRAPPER_LINEAR_CLS is None:
        try:
            from auto_round.wrapper import WrapperLinear
        except Exception:  # pragma: no cover - defensive
            return False
        _WRAPPER_LINEAR_CLS = WrapperLinear
    return isinstance(layer, _WRAPPER_LINEAR_CLS)


def _act_signature(layer: nn.Module):
    """Quantization signature of a wrapped layer's *activation* path.

    Two projections may share one grouped activation-quantization call only when their
    signatures match, because we quantize the whole sorted batch in a single shot.
    """
    orig = layer.orig_layer
    return (
        getattr(orig, "act_bits", 16),
        getattr(orig, "act_group_size", -1),
        getattr(orig, "act_sym", True),
        getattr(orig, "act_data_type", None),
        bool(getattr(orig, "act_dynamic", True)),
    )


def _quant_signature(layer: nn.Module):
    """Quantization signature of one projection slot, or ``None`` if it is not quantized.

    AutoRound supports mixed-bit MoE: individual experts (or even individual projections
    inside one expert) can carry different schemes. Sharing one grouped GEMM across
    experts whose weights were produced by different quant functions is not something we
    want to reason about case by case, so experts must agree on this signature or the
    layer falls back to the per-expert loop. The common case — one uniform scheme for the
    whole experts module — is unaffected.

    The wrapper *class* is part of the signature too: SignRoundV2's optimized wrapper,
    the GGUF double-quant wrapper and the AutoScheme scoring wrappers all carry their own
    tuning state and ``minmax_scale_bound``, so they must not be mixed inside one group.
    """
    if not _is_wrapper_linear(layer):
        return None
    orig = layer.orig_layer
    enable_act_quant = bool(getattr(layer, "enable_act_quant", False))
    return (
        type(layer),
        getattr(orig, "bits", 16),
        getattr(orig, "group_size", -1),
        bool(getattr(orig, "sym", True)),
        getattr(orig, "data_type", None),
        getattr(layer, "data_type", None),  # resolved by get_quant_func, may differ from orig
        getattr(orig, "super_bits", None),
        getattr(orig, "super_group_size", None),
        bool(getattr(layer, "disable_opt_rtn", True)),
        enable_act_quant,
        _act_signature(layer) if enable_act_quant else None,
    )


def _weight_layout(layer: nn.Module):
    """``(shape, dtype)`` of the underlying weight; must match to be batched together."""
    weight = layer.weight
    return tuple(weight.shape), weight.dtype


def _act_quant_is_row_independent(layer: nn.Module) -> bool:
    """Whether activation quantization can be applied to the concatenated batch at once.

    Quantizing ``x[a:b]`` per expert equals quantizing the full ``x`` only when each row
    (or each group inside a row) gets its own scale and that scale is derived from the
    data itself. Per-tensor (``group_size == 0``), 2-D groups, static ranges and tunable
    per-expert min/max coefficients all couple rows of *different* experts together, so
    they disqualify the grouped path.
    """
    orig = layer.orig_layer
    if not getattr(orig, "act_dynamic", False):
        return False  # static: act_max is calibrated per expert
    group_size = getattr(orig, "act_group_size", -1)
    if isinstance(group_size, tuple) or group_size == 0:
        return False  # per-tensor / 2-D groups mix rows across experts
    if getattr(layer, "input_global_scale", None) is not None:
        return False  # NVFP-style per-layer global scale
    for name in ("act_max_scale", "act_min_scale"):
        if isinstance(getattr(layer, name, None), nn.Parameter):
            return False  # tunable, and tuned independently per expert
    return True


def _projection_is_supported(layer: nn.Module) -> bool:
    if type(layer) is nn.Linear:
        return True
    if not _is_wrapper_linear(layer):
        return False
    orig = layer.orig_layer
    if type(orig) is not nn.Linear:
        return False  # Conv1D / LinearAllreduce need their own forward
    if orig._forward_pre_hooks or orig._forward_hooks:
        return False  # e.g. online Hadamard rotation must run per layer
    if getattr(layer, "enable_act_quant", False) and not _act_quant_is_row_independent(layer):
        return False
    return True


def _compute_device(layer: nn.Module) -> torch.device:
    if _is_wrapper_linear(layer):
        return torch.device(layer.device)
    return layer.weight.device


def _output_device(layer: nn.Module) -> torch.device:
    if _is_wrapper_linear(layer):
        return torch.device(layer.output_device)
    return layer.weight.device


def _effective_weight_and_bias(layer: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return the weight the layer would actually multiply with, plus its bias.

    For a ``WrapperLinear`` this is the fake-quantized weight produced by the current
    tuning parameters, so autograd reaches ``value`` / ``min_scale`` / ``max_scale``.
    """
    if _is_wrapper_linear(layer):
        weight, _, _ = layer._qdq_weight(layer.value, layer.min_scale, layer.max_scale)
        return weight, _effective_bias(layer)
    return layer.weight, layer.bias


def _effective_bias(layer: nn.Module) -> torch.Tensor | None:
    if not _is_wrapper_linear(layer):
        return layer.bias
    bias = layer.orig_layer.bias
    if bias is not None and bias.device.type == "meta":
        bias = layer.orig_layer.get_bias().to(layer.device)
    if getattr(layer, "enable_norm_bias_tuning", False):
        bias, _, _ = layer._qdq_bias(bias, layer.bias_v)
    return bias


# --------------------------------------------------------------------------------------
# Batched fake-quantization
# --------------------------------------------------------------------------------------
#
# The grouped GEMM alone removes the per-expert launches/syncs of the *matmul*, but with
# ``WrapperLinear`` the dominant cost during tuning is the fake-quantization itself: one
# ``weight_quant_func`` call per expert, each on a small ``(out, in)`` tensor, plus the
# matching backward. With hundreds of experts that is hundreds of tiny elementwise graphs
# (and, under ``enable_torch_compile``, hundreds of compiled-graph launches).
#
# AutoRound's weight quantizers derive their scales from *rows* of the group-reshaped
# tensor (``reshape_pad_tensor_by_group_size``), so as long as the fused layout keeps each
# expert on its own set of rows, quantizing ``cat([w_0, ..., w_{E-1}])`` is identical to
# quantizing each ``w_i`` on its own. We then also concatenate the per-row tuning tensors
# (``value``, ``min_scale``, ``max_scale``, ``weight_min``/``weight_max``, SignRoundV2's
# ``init_scale``) and run a single quant call for the whole slot. Autograd routes each
# row-block's gradient back to the owning expert's parameters, so tuning is unchanged.
#
# Group layouts, and how they fuse:
#   group_size > 0 / -1  ``cat(dim=0)`` -> ``(E * out, in)``; groups never straddle rows.
#   group_size == 0      per-tensor, i.e. ONE scale per expert. Reshaping each expert to a
#                        single row and passing ``group_size=-1`` reproduces that exactly.
#   group_size == (M, N) 2-D blocks. ``cat(dim=0)`` is only equivalent when ``out % M == 0``
#                        -- otherwise per-expert zero-padding of the last block row would
#                        differ from the fused tensor's.
#
# Per-layer extras, and how they fuse:
#   NVFP ``global_scale``  a per-layer scalar that the quant func broadcasts against the
#                          ``(rows, 1)`` per-group max, so an expanded ``(rows, 1)`` column
#                          keeps every expert on its own scalar.
#   GGUF ``super_bits``    super-blocks group consecutive *rows*; GGUF requires
#                          ``in % QK_K == 0``, so every expert's row count is already a
#                          multiple of ``super_group_size`` and blocks cannot straddle
#                          experts. Guarded explicitly all the same.
#   ``imatrix``            declared as one value per *input channel*, which the quantizers
#                          broadcast across output rows with
#                          ``imatrix.reshape(1, -1).expand(tensor.numel() // imatrix.numel(),
#                          -1).reshape(tensor.shape)``. Handing them a *full-size* importance
#                          tensor makes that expand a no-op and the reshape an exact
#                          per-element mapping, so each expert keeps its own imatrix. The
#                          row-wise zero fixup (``_imatrix_handle_zero``) is row-independent,
#                          so the result stays bit-exact. When the experts happen to share one
#                          imatrix we pass it through unexpanded and pay nothing.
#
# Wrappers that override ``_qdq_weight`` (GGUF double-quant search, AutoScheme scoring)
# carry per-layer state and always keep the per-expert call. That also keeps the GGUF
# "row pattern" imatrix consumers (``_imatrix_row_pattern``, only reached from the
# double-quant search) out of the fused path.


def _grouped_row_count(out_features: int, in_features: int, group_size: int) -> int:
    """Rows of the group-reshaped weight, mirroring ``reshape_pad_tensor_by_group_size``."""
    if group_size == -1 or in_features < group_size:
        return out_features
    return out_features * ceil(in_features / group_size)


# Quant funcs that take ``global_scale`` and only ever broadcast it against per-row
# quantities, so a per-row column vector keeps each expert on its own scalar. Variants that
# recompute the scale from the tensor's global max (``*_with_static_gs``, ``nvfp4_v2*``,
# ``opt_rtn_nv_fp4``) would collapse it across experts and are deliberately excluded.
_PER_ROW_GLOBAL_SCALE_DTYPES = frozenset({"nv_fp4", "nv_fp4_rtn"})


def _fusion_group_size(layer: nn.Module) -> int | tuple | None:
    """``group_size`` to hand the fused quant call, or ``None`` if the slot cannot fuse."""
    group_size = layer.orig_layer.group_size
    out_features = layer.weight.shape[0]
    if isinstance(group_size, tuple):
        if len(group_size) != 2 or out_features % group_size[0] != 0:
            return None  # the block grid would straddle two experts
        return group_size
    if not isinstance(group_size, int) or isinstance(group_size, bool):
        return None
    if group_size == 0:
        return -1  # per-tensor == one row per expert
    return group_size


def _supports_batched_qdq(layer: nn.Module) -> bool:
    """Whether this projection's fake-quantization may be fused with its siblings'."""
    if not _is_wrapper_linear(layer):
        return False
    if type(layer)._qdq_weight is not _WRAPPER_LINEAR_CLS._qdq_weight:
        return False  # custom qdq: GGUF double-quant search, AutoScheme scoring, ...
    orig = layer.orig_layer
    if getattr(orig, "bits", 16) >= 16:
        return False  # not quantized: the plain weight is used as-is
    if hasattr(layer, "_extra_quant_kwargs"):
        return False  # unknown per-layer quant kwargs

    fusion_group_size = _fusion_group_size(layer)
    if fusion_group_size is None:
        return False

    if getattr(orig, "imatrix", None) is not None:
        # The fused importance tensor is flattened once and padded as a whole, so it only
        # lines up with the per-expert layout when no row-level padding is involved.
        if not isinstance(fusion_group_size, int):
            return False
        in_features = layer.weight.shape[1]
        if fusion_group_size > 0 and in_features % fusion_group_size != 0:
            return False

    if getattr(layer, "weight_global_scale", None) is not None:
        if getattr(layer, "data_type", None) not in _PER_ROW_GLOBAL_SCALE_DTYPES:
            return False
        if isinstance(fusion_group_size, tuple):
            return False  # no per-row column to broadcast against

    super_group_size = getattr(orig, "super_group_size", None)
    if getattr(orig, "super_bits", None) is not None:
        if not super_group_size or isinstance(fusion_group_size, tuple):
            return False
        out_features, in_features = layer.weight.shape
        rows = _grouped_row_count(out_features, in_features, fusion_group_size)
        if rows % super_group_size != 0:
            return False  # a super-block would span two experts

    return True


_UNBATCHABLE = object()


def _cat_tuning_tensors(layers: list[nn.Module], name: str) -> torch.Tensor | None | object:
    """Concatenate a per-row tuning tensor across experts, or pass a shared scalar through.

    Returns ``_UNBATCHABLE`` when the attribute is inconsistent (some experts have it and
    others do not), which disables the batched path for the slot.
    """
    values = [getattr(layer, name, None) for layer in layers]
    first = values[0]
    if first is None:
        return None if all(v is None for v in values) else _UNBATCHABLE
    if not isinstance(first, torch.Tensor):
        return _UNBATCHABLE
    if first.dim() == 0:
        # Non-tunable constant (e.g. value=0.0 when round tuning is off): identical for
        # every expert, and broadcast by the quant func.
        if any(v is None or not isinstance(v, torch.Tensor) or v.dim() != 0 for v in values):
            return _UNBATCHABLE
        return first
    if any(v is None or not isinstance(v, torch.Tensor) or v.dim() == 0 for v in values):
        return _UNBATCHABLE
    return torch.cat(values, dim=0)


def _fused_global_scale(layers: list[nn.Module], rows_per_expert: int, device) -> torch.Tensor | None | object:
    """Expand each expert's NVFP per-layer scalar into a ``(E * rows, 1)`` column."""
    scales = [getattr(layer, "weight_global_scale", None) for layer in layers]
    if all(scale is None for scale in scales):
        return None
    if any(scale is None or not isinstance(scale, torch.Tensor) or scale.numel() != 1 for scale in scales):
        return _UNBATCHABLE
    flat = torch.stack([scale.reshape(()) for scale in scales]).to(device=device, dtype=torch.float32)
    return flat.repeat_interleave(rows_per_expert).unsqueeze(-1)


def _shares_one_tensor(tensors: list[torch.Tensor]) -> bool:
    """Cheap, sync-free "are these all the same tensor" test (identity or same storage)."""
    first = tensors[0]
    for other in tensors[1:]:
        if other is first:
            continue
        if other.shape != first.shape or other.dtype != first.dtype or other.data_ptr() != first.data_ptr():
            return False
    return True


def _fused_imatrix(layers: list[nn.Module], out_features: int, device) -> torch.Tensor | None | object:
    """Build the importance matrix matching the fused weight layout.

    The quantizers treat ``imatrix`` as one value per input channel and broadcast it over
    the output rows with ``reshape(1, -1).expand(tensor.numel() // imatrix.numel(),
    -1).reshape(tensor.shape)``. Passing a full-size ``(E * out, in)`` tensor turns that
    expand into a no-op and the reshape into an exact per-element mapping, which is how
    each expert keeps its own importance instead of inheriting the first one's.

    When every expert already points at the same imatrix we hand that single vector over
    untouched, so the common case costs nothing; otherwise the fused tensor is
    materialized (roughly the size of the fused weight in fp32).
    """
    mats = [getattr(layer.orig_layer, "imatrix", None) for layer in layers]
    if all(mat is None for mat in mats):
        return None
    if any(mat is None or not isinstance(mat, torch.Tensor) for mat in mats):
        return _UNBATCHABLE

    mats = [mat.to(device) for mat in mats]
    if _shares_one_tensor(mats):
        # The quant func's own row-repeat already produces the right thing.
        return mats[0]

    numel = mats[0].numel()
    if any(mat.numel() != numel for mat in mats[1:]):
        return _UNBATCHABLE
    stacked = torch.stack([mat.reshape(-1) for mat in mats])  # (E, in)
    return stacked.repeat_interleave(out_features, dim=0)  # (E * out, in), expert-major


def _batched_qdq_weights(
    layers: list[nn.Module], *, allow_compiled: bool = True
) -> tuple[torch.Tensor | None, list[torch.Tensor]] | None:
    """Fake-quantize one projection of every expert in a single quant-func call.

    Args:
        layers: The projections to fuse, all from the same slot.
        allow_compiled: Whether the group size is a constant, so the compiled quant
            function can be used. Dynamo keys its cache on the traced code object, shared
            by every ``WrapperLinear``, so a group whose size follows the router would add
            a cache entry per distinct count.

    Returns ``(stacked, per_expert_weights)`` where ``stacked`` is the ``(E, out, in)``
    view of the fused result (free operand for the grouped-GEMM kernel, or ``None`` if the
    result is not contiguous). Returns ``None`` when the slot turned out not to be
    eligible and the caller must go per-expert.
    """
    ref = layers[0]
    orig = ref.orig_layer
    out_features, in_features = ref.weight.shape
    num_experts = len(layers)

    fusion_group_size = _fusion_group_size(ref)
    if fusion_group_size is None:  # pragma: no cover - already checked by the plan
        return None
    # Per-tensor slots fuse as one row per expert so each keeps its own single scale.
    per_tensor = orig.group_size == 0

    raw_weights = []
    for layer in layers:
        weight = layer.orig_layer.weight
        if weight.device.type == "meta":
            weight = layer.orig_layer.get_weight()
        weight = weight.to(ref.device)
        raw_weights.append(weight.reshape(1, -1) if per_tensor else weight)
    fused_input = torch.cat(raw_weights, dim=0)  # (E, out * in) or (E * out, in)

    # WrapperLinear._qdq_weight clamps the *parameters* in place; do the same here so the
    # optimizer sees identical state, then concatenate the clamped values.
    min_bound, max_bound = type(ref).minmax_scale_bound
    for layer in layers:
        for name in ("min_scale", "max_scale"):
            param = getattr(layer, name, None)
            if isinstance(param, torch.Tensor) and param.dim() > 0:
                param.data.clamp_(min_bound, max_bound)

    fused: dict[str, object] = {
        name: _cat_tuning_tensors(layers, name)
        for name in ("value", "min_scale", "max_scale", "weight_min", "weight_max", "init_scale")
    }
    if isinstance(fusion_group_size, tuple):
        # 2-D blocks never carry a per-layer global scale (checked by _supports_batched_qdq).
        fused["global_scale"] = None
    else:
        rows_per_expert = 1 if per_tensor else _grouped_row_count(out_features, in_features, fusion_group_size)
        fused["global_scale"] = _fused_global_scale(layers, rows_per_expert, ref.device)
    fused["imatrix"] = _fused_imatrix(layers, out_features, ref.device)
    if any(value is _UNBATCHABLE for value in fused.values()):
        _log_fallback_once("inconsistent tuning tensors across experts; batched qdq disabled")
        return None

    quant_kwargs = {}
    if getattr(orig, "super_bits", None) is not None:
        quant_kwargs["super_bits"] = orig.super_bits
        quant_kwargs["super_group_size"] = orig.super_group_size

    # Normally the *compiled* quant function, same as the per-expert path uses: this call
    # is reached from a function dynamo does not trace into (see ``_opaque_to_dynamo``), so
    # it is an ordinary call into a compiled artifact and the fused shape is a constant.
    # Only when the caller says the group size follows the router do we drop to eager,
    # because dynamo keys its cache on the traced code object -- shared by every
    # WrapperLinear -- and would gain an entry per distinct expert count.
    quant_func = ref.weight_quant_func
    if not allow_compiled:
        quant_func = getattr(ref, "weight_quant_func_eager", None) or quant_func

    weight_q, _, _ = quant_func(
        fused_input,
        bits=orig.bits,
        group_size=fusion_group_size,
        v=fused["value"],
        min_scale=fused["min_scale"],
        max_scale=fused["max_scale"],
        scale_dtype=orig.scale_dtype,
        tensor_min=fused["weight_min"],
        tensor_max=fused["weight_max"],
        data_type=ref.data_type,
        q_scale_thresh=ref.q_scale_thresh,
        imatrix=fused["imatrix"],
        global_scale=fused["global_scale"],
        init_scale=fused["init_scale"],
        **quant_kwargs,
    )
    weight_q = weight_q.to(fused_input.dtype)

    stacked = None
    if weight_q.is_contiguous():
        stacked = weight_q.view(num_experts, out_features, in_features)
        weights = list(stacked.unbind(0))
    elif per_tensor:
        weights = [row.reshape(out_features, in_features) for row in weight_q]
    else:
        weights = list(torch.split(weight_q, out_features, dim=0))
    return stacked, weights


@dataclass
class _SlotWeights:
    """The weights/biases of one projection slot for every active expert."""

    weights: list[torch.Tensor]
    biases: list[torch.Tensor | None]
    # (E, out, in) view of the fused qdq result, when available. Lets the native grouped_mm
    # kernel get its 3D operand for free instead of a torch.stack copy.
    stacked: torch.Tensor | None = None


# Experts fake-quantized per fused call. Fusing *all* active experts at once maximizes
# kernel size, but the working set it builds -- roughly
# ``chunk * out * in * (weight_itemsize + 4)`` bytes, the concatenated weight plus its fp32
# ``value`` -- grows with the expert count, and the quant func streams it several times
# (min/max, scale, round_ste, clamp, mul). Past a point that costs peak memory on GPU and
# cache locality on CPU.
#
# 16 comes from the chunk sweeps, not from taste:
#   CPU, 64x(512x256): fusing all 64 halved calibration throughput (0.50x vs the per-expert
#     loop); chunk=16 turned it into a win (1.28x calibration, 1.54x tuning).
#   A100, 128x(4096x1536): chunk=16 held the speed (1.58x tuning vs 1.63x for one fused
#     call) while cutting the calibration peak from 32.7 GB to 20.5 GB.
#
# ``AR_MOE_QDQ_CHUNK`` overrides it; 0 or negative means "fuse everything".
_DEFAULT_QDQ_CHUNK = 16

# Target working set for one fused qdq call. The chunk is derived from this budget and the
# per-expert weight size, instead of a flat count, so it adapts to the *shape* of the MoE:
#
#   * On the large experts the flat 16 was tuned on (4096x1536, bf16 -> ~38 MB/expert) the
#     budget reproduces ~16, preserving those sweeps.
#   * On small experts (e.g. 512-wide MoE, ~6 MB/expert) it fuses far more per call, so the
#     compiled quant graph is launched a handful of times instead of ~16 -- which is what
#     makes the batched qdq pay off once the block forward is eager under torch.compile.
#
# The tail below is fused into a single eager call, so a larger chunk never inflates the
# per-expert launch count; it only shifts work from many compiled calls to a few big ones.
_QDQ_FUSED_WORKINGSET_BYTES = 512 * 1024 * 1024


def _auto_qdq_chunk(layers: list[nn.Module]) -> int:
    """Experts-per-fused-call derived from a fixed working-set budget and the weight shape.

    Shape-derived (not routing-derived), so it stays constant across steps and does not
    break :func:`_static_shapes_required`.
    """
    weight = layers[0].weight
    # bytes streamed per expert: the concatenated weight plus its fp32 tuning ``value``.
    per_expert = int(weight.shape[0]) * int(weight.shape[1]) * (weight.element_size() + 4)
    if per_expert <= 0:
        return _DEFAULT_QDQ_CHUNK
    return max(_DEFAULT_QDQ_CHUNK, int(_QDQ_FUSED_WORKINGSET_BYTES // per_expert))


def _qdq_chunk_size(layers: list[nn.Module]) -> int:
    """Configured experts-per-fused-call. Not clamped to ``len(layers)``.

    Clamping here would defeat :func:`_static_shapes_required`: when fewer experts than
    the chunk are routed, the group would silently shrink to the active count and the
    fused shape would follow the router again.
    """
    setting = envs.AR_MOE_QDQ_CHUNK
    if setting in ("", "auto"):
        return _auto_qdq_chunk(layers)
    try:
        chunk = int(setting)
    except ValueError:
        logger.warning_once(f"Ignoring AR_MOE_QDQ_CHUNK={setting!r}: expected 'auto' or an integer.")
        return _auto_qdq_chunk(layers)
    return chunk


def _static_shapes_required(layer: nn.Module) -> bool:
    """Whether the fused group size must stay constant across steps.

    Only matters under ``torch.compile``: the fused call goes to the same compiled quant
    function the per-expert path uses, and dynamo keys its cache on the traced code object
    (shared by every ``WrapperLinear``). A group whose size followed the router would add
    a cache entry per distinct active-expert count -- i.e. per calibration sample.
    """
    return bool(getattr(layer, "enable_torch_compile", False))


def _plan_qdq_groups(layers: list[nn.Module]) -> tuple[list[list[nn.Module]], list[nn.Module], bool]:
    """Split the active experts into fused groups, a remainder tail, and a "fixed size" flag.

    Without ``torch.compile`` every expert can be fused, the last group simply being
    smaller. With it, only whole ``chunk``-sized groups are fused so the fused shape is a
    constant the compiled quant graph can cache; the router-sized remainder is returned as
    the tail, which :func:`_slot_weights` fuses into a single *eager* call (its shape may
    follow the router because eager needs no shape-keyed cache). The flag tells the caller
    whether the compiled quant function may be used for the whole groups -- it may not when
    an explicit ``AR_MOE_QDQ_CHUNK=0`` asks to fuse everything, since that group's size
    follows the router.
    """
    chunk = _qdq_chunk_size(layers)
    total = len(layers)
    if chunk <= 0:  # explicit "fuse everything": one group, sized by the routing
        return [layers], [], not _static_shapes_required(layers[0])
    if not _static_shapes_required(layers[0]):
        return [layers[i : i + chunk] for i in range(0, total, chunk)], [], True
    num_full = (total // chunk) * chunk
    groups = [layers[i : i + chunk] for i in range(0, num_full, chunk)]
    return groups, layers[num_full:], True


def _slot_weights(layers: list[nn.Module], batched: bool) -> _SlotWeights:
    biases = [_effective_bias(layer) for layer in layers]
    if batched:
        groups, tail, allow_compiled = _plan_qdq_groups(layers)
        if groups or tail:
            weights: list[torch.Tensor] = []
            stacked = None
            ok = True
            for group in groups:
                result = _batched_qdq_weights(group, allow_compiled=allow_compiled)
                if result is None:
                    ok = False
                    break
                if len(groups) == 1 and not tail:
                    stacked = result[0]  # free 3-D operand for the native kernel
                weights.extend(result[1])
            if ok and tail:
                # Fuse the router-sized remainder in a SINGLE eager qdq call. Its shape
                # follows the routing, so it must stay off the compiled quant graph
                # (``allow_compiled=False``), but batching it collapses up to ``chunk - 1``
                # per-expert launches into one -- the launches torch.compile would
                # otherwise hide inside the per-expert loop it competes with.
                tail_result = _batched_qdq_weights(tail, allow_compiled=False) if len(tail) > 1 else None
                if tail_result is not None:
                    weights.extend(tail_result[1])
                else:
                    weights.extend(_effective_weight_and_bias(layer)[0] for layer in tail)
            if ok and weights:
                return _SlotWeights(weights=weights, biases=biases, stacked=stacked)
    weights = [_effective_weight_and_bias(layer)[0] for layer in layers]
    return _SlotWeights(weights=weights, biases=biases)


def _quantize_activation(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply the layer's activation fake-quantization to a whole grouped batch."""
    if not _is_wrapper_linear(layer) or not getattr(layer, "enable_act_quant", False):
        return x
    act_max = getattr(layer.orig_layer, "act_max", None)
    x, _, _ = layer._qdq_act(x, act_max_scale=layer.act_max_scale, act_min_scale=layer.act_min_scale, act_max=act_max)
    return x


# --------------------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------------------


@dataclass
class _GroupedPlan:
    """Everything needed to run one grouped forward, resolved for the *active* experts."""

    experts: list[nn.Module]
    has_gate: bool
    device: torch.device
    output_device: torch.device
    # Per-slot: may the fake-quantization of all active experts be fused into one call?
    batched_qdq: dict[str, bool]


_SLOTS = ("gate_proj", "up_proj", "down_proj")


def _build_plan(module: nn.Module, active_ids: list[int]) -> _GroupedPlan | None:
    """Validate the active experts and gather them, or return ``None`` to fall back."""
    experts: list[nn.Module] = []
    has_gate: bool | None = None
    device: torch.device | None = None
    output_device: torch.device | None = None
    # Per-slot reference signatures: every active expert must agree on them.
    quant_signatures: dict[str, object] = {}
    layouts: dict[str, object] = {}
    batched_qdq: dict[str, bool] = {}

    for expert_id in active_ids:
        expert = getattr(module, str(expert_id), None)
        if expert is None:
            _log_fallback_once("expert container missing")
            return None
        if not hasattr(expert, "up_proj") or not hasattr(expert, "down_proj"):
            _log_fallback_once("expert does not expose up_proj/down_proj")
            return None

        expert_has_gate = hasattr(expert, "gate_proj")
        if has_gate is None:
            has_gate = expert_has_gate
        elif has_gate != expert_has_gate:
            _log_fallback_once("experts disagree on gate_proj")
            return None

        slots = _SLOTS if expert_has_gate else ("up_proj", "down_proj")
        for slot in slots:
            projection = getattr(expert, slot)
            if not _projection_is_supported(projection):
                _log_fallback_once(f"unsupported projection type {type(projection).__name__}")
                return None

            proj_device = _compute_device(projection)
            if device is None:
                device = proj_device
                output_device = _output_device(projection)
            elif proj_device != device:
                _log_fallback_once("experts live on different devices")
                return None

            # Mixed-bit MoE: bail out instead of batching differently-quantized experts.
            signature = _quant_signature(projection)
            if slot in quant_signatures:
                if quant_signatures[slot] != signature:
                    _log_fallback_once(f"mixed quantization schemes across experts for '{slot}'")
                    return None
            else:
                quant_signatures[slot] = signature

            layout = _weight_layout(projection)
            if slot in layouts:
                if layouts[slot] != layout:
                    _log_fallback_once(f"experts have different weight shape/dtype for '{slot}'")
                    return None
            else:
                layouts[slot] = layout

            supports_batched = _supports_batched_qdq(projection)
            batched_qdq[slot] = batched_qdq.get(slot, True) and supports_batched

        # gate_proj and up_proj consume the *same* tensor, which the grouped path
        # activation-quantizes once, so their activation settings must be identical.
        if expert_has_gate:
            gate_act = _act_signature(expert.gate_proj) if _is_wrapper_linear(expert.gate_proj) else None
            up_act = _act_signature(expert.up_proj) if _is_wrapper_linear(expert.up_proj) else None
            gate_enabled = bool(getattr(expert.gate_proj, "enable_act_quant", False))
            up_enabled = bool(getattr(expert.up_proj, "enable_act_quant", False))
            if gate_enabled != up_enabled or (gate_enabled and gate_act != up_act):
                _log_fallback_once("gate_proj/up_proj activation quantization differ")
                return None

        experts.append(expert)

    if not experts or device is None or output_device is None:
        return None
    return _GroupedPlan(
        experts=experts,
        has_gate=bool(has_gate),
        device=device,
        output_device=output_device,
        batched_qdq=batched_qdq,
    )


# --------------------------------------------------------------------------------------
# Grouped matmul
# --------------------------------------------------------------------------------------
#
# Two ways to run the routed GEMM, plus one we deliberately avoid:
#
#   sliced loop  (default) one ``F.linear`` per *active* expert over a contiguous slice.
#                Same kernels as transformers' ``grouped_mm_fallback``, but without its
#                extra ``offs.tolist()`` sync -- we already hold the counts on the host.
#   grouped_mm   ``torch.nn.functional.grouped_mm`` / ``torch._grouped_mm``: one kernel for
#                all experts, driven by ``offsets``. Differentiable, and bit-identical to
#                the loop (verified on A100). Opt in with ``AR_MOE_GROUPED_MM=1``.
#   batched_mm   transformers' ``_batched_linear``/``torch.bmm`` path gathers
#                ``weight[expert_ids]`` into an ``(S, out, in)`` tensor -- one weight copy
#                per routed token. That is a decode-time trick (S == top_k); at tuning
#                shapes it is orders of magnitude slower (measured ~87x) and the gather
#                alone would be hundreds of GB. Never used here.
#
# Why the loop is the default, even on CUDA where the grouped kernel exists: once the
# weights go through ``WrapperLinear`` the fake-quant dominates and the kernel's advantage
# inverts. Measured on A100 (sm80, bf16, 2048x768 experts, 4096 routed pairs), speedup over
# ``linear_loop`` for 16/32/64/128/256 experts:
#
#   forward+backward   grouped_mm 3.89 4.36 5.11 4.84 5.05  |  sliced 4.36 4.70 5.32 5.14 5.24
#   forward (wrapped)  grouped_mm 2.57 2.99 3.21 3.30 3.28  |  sliced 3.13 3.44 3.69 3.80  -
#   forward (plain fp) grouped_mm 2.07 2.53 2.76 2.99 2.93  |  sliced 1.90 2.21 2.29 2.42 2.39
#
# The kernel only wins on plain fp weights, where there is no fake-quant to hide the GEMM.
# Both AutoRound-relevant modes favour the loop, which additionally needs no ``torch.stack``
# copy (lower peak memory when the qdq was not fused) and has no dtype/alignment/compute-
# capability constraints. torch's CPU ``grouped_mm`` loses even harder, so the rule is
# device-independent.
#
# Deciding whether the native kernel is *legal* is fiddly (torch version, device, compute
# capability, dynamo, 16-byte alignment on CPU), so when it is requested we defer to
# transformers' own ``_can_use_grouped_mm`` rather than re-deriving it.

try:  # transformers >= 5.0
    from transformers.integrations.moe import _can_use_grouped_mm as _transformers_can_use_grouped_mm
except Exception:  # pragma: no cover - older/absent transformers
    _transformers_can_use_grouped_mm = None


def _native_grouped_mm_available() -> bool:
    """Cheap pre-check that does not need the stacked 3-D operand."""
    if _NATIVE_GROUPED_MM_DISABLED or envs.AR_MOE_GROUPED_MM != "1":
        return False
    return hasattr(F, "grouped_mm") or hasattr(torch, "_grouped_mm")


def _native_grouped_mm_preferred(device: torch.device) -> bool:
    """Whether the native kernel is opted into. Off by default; see the note above."""
    return envs.AR_MOE_GROUPED_MM == "1"


def _native_grouped_mm_usable(x: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor) -> bool:
    """Whether ``grouped_mm`` can and should run this batch. ``weight`` is ``(E, in, out)``."""
    if not _native_grouped_mm_available():
        return False
    if _transformers_can_use_grouped_mm is not None:
        return _transformers_can_use_grouped_mm(x, weight, offsets)
    # Backstop mirroring the parts of transformers' rules we can check on our own.
    if weight.device.type == "cuda":
        try:
            return torch.cuda.get_device_capability(weight.device) >= (8, 0)
        except Exception:  # pragma: no cover - defensive
            return False
    return True


def _native_grouped_mm(x: torch.Tensor, weight: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """``(S, in) @ grouped (E, in, out)`` -> ``(S, out)``."""
    x = x.to(weight.dtype)  # grouped_mm is not autocast-aware
    if hasattr(F, "grouped_mm"):
        return F.grouped_mm(x, weight, offs=offsets)
    return torch._grouped_mm(x, weight, offs=offsets)


def _sliced_linear(x: torch.Tensor, slot: _SlotWeights, counts: list[int]) -> torch.Tensor:
    """Per-expert ``F.linear`` over contiguous slices — no gather, no sync, no 3D copy."""
    outputs = []
    start = 0
    for weight, bias, count in zip(slot.weights, slot.biases, counts):
        end = start + count
        outputs.append(F.linear(x[start:end], weight, bias))  # pylint: disable=not-callable
        start = end
    return torch.cat(outputs, dim=0)


def _stack_weights(slot: _SlotWeights) -> torch.Tensor:
    """``(E, out, in)`` operand for the native kernel, as a view when the qdq was fused."""
    if slot.stacked is not None:
        return slot.stacked
    return torch.stack(slot.weights, dim=0)


def _grouped_linear(
    x: torch.Tensor,
    slot: _SlotWeights,
    counts: list[int],
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Run one projection for all active experts at once."""
    global _NATIVE_GROUPED_MM_DISABLED

    if len(slot.weights) == 1:
        # Single active expert: a plain GEMM beats every grouped path.
        return F.linear(x, slot.weights[0], slot.biases[0])  # pylint: disable=not-callable

    if _native_grouped_mm_available():
        try:
            weight = _stack_weights(slot).transpose(-2, -1)  # (E, in, out)
            if _native_grouped_mm_usable(x, weight, offsets):
                out = _native_grouped_mm(x, weight, offsets)
                if slot.biases[0] is not None:
                    row_expert = torch.repeat_interleave(
                        torch.arange(len(counts), device=x.device),
                        torch.tensor(counts, device=x.device),
                    )
                    out = out + torch.stack(slot.biases, dim=0)[row_expert]
                return out
        except Exception as err:  # pragma: no cover - kernel/shape constraints vary
            _NATIVE_GROUPED_MM_DISABLED = True
            logger.warning_once(
                f"torch grouped_mm failed ({err}); using the sliced per-expert GEMM loop for MoE tuning."
            )

    return _sliced_linear(x, slot, counts)


# --------------------------------------------------------------------------------------
# Forward
# --------------------------------------------------------------------------------------


def _run_routes(
    module: nn.Module,
    hidden_states: torch.Tensor,
    route_indices: torch.Tensor,
    route_weights: torch.Tensor,
    num_experts: int,
) -> torch.Tensor | None:
    """Grouped expert forward for one routing table, or ``None`` if unsupported."""
    device = hidden_states.device
    num_tokens, hidden_dim = hidden_states.shape
    num_top_k = route_indices.size(-1)

    expert_ids = route_indices.reshape(-1)  # (S,)
    sample_weights = route_weights.reshape(-1).to(hidden_states.dtype)  # (S,)
    num_pairs = expert_ids.numel()

    # Sort the token/expert pairs so each expert owns one contiguous slice.
    expert_ids_sorted, perm = torch.sort(expert_ids)

    # The only host<->device sync of the whole layer.
    unique_ids, unique_counts = torch.unique_consecutive(expert_ids_sorted, return_counts=True)
    active_ids = unique_ids.tolist()
    active_counts = unique_counts.tolist()

    # Expert-parallel sentinels (id >= num_experts) sort to the tail; drop them.
    keep = [i for i, expert_id in enumerate(active_ids) if 0 <= expert_id < num_experts]
    active_ids = [active_ids[i] for i in keep]
    active_counts = [active_counts[i] for i in keep]
    num_valid = sum(active_counts)
    if num_valid == 0:
        return torch.zeros_like(hidden_states)

    plan = _build_plan(module, active_ids)
    if plan is None:
        return None

    perm_valid = perm[:num_valid]
    token_idx = torch.div(perm_valid, num_top_k, rounding_mode="floor")
    x = hidden_states.index_select(0, token_idx).to(plan.device)

    offsets = torch.tensor(active_counts, device=plan.device, dtype=torch.int32).cumsum(0).to(torch.int32)

    # --- input projections (gate / up) -------------------------------------------------
    x = _quantize_activation(plan.experts[0].up_proj, x)

    up = _slot_weights([e.up_proj for e in plan.experts], plan.batched_qdq.get("up_proj", False))
    up_out = _grouped_linear(x, up, active_counts, offsets)

    if plan.has_gate:
        gate = _slot_weights([e.gate_proj for e in plan.experts], plan.batched_qdq.get("gate_proj", False))
        gate_out = _grouped_linear(x, gate, active_counts, offsets)
        if hasattr(module, "_apply_gate"):
            # Keep the module's own gating (clamping, alpha, ...) and its [gate; up] layout,
            # exactly as linear_loop_experts_forward does.
            hidden = module._apply_gate(torch.cat([gate_out, up_out], dim=-1))
        else:
            hidden = module.act_fn(gate_out) * up_out
    else:
        hidden = module.act_fn(up_out)

    # --- down projection ---------------------------------------------------------------
    hidden = _quantize_activation(plan.experts[0].down_proj, hidden)
    down = _slot_weights([e.down_proj for e in plan.experts], plan.batched_qdq.get("down_proj", False))
    out = _grouped_linear(hidden, down, active_counts, offsets)

    out = out.to(plan.output_device)
    out = out * sample_weights.index_select(0, perm_valid).to(out.dtype).unsqueeze(-1)

    # Scatter back to the original (token, top_k) order and reduce over top_k.
    out_per_sample = torch.zeros(num_pairs, hidden_dim, device=out.device, dtype=out.dtype)
    out_per_sample = out_per_sample.index_copy(0, perm_valid.to(out.device), out)
    return out_per_sample.view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(device)


def _opaque_to_dynamo(fn):
    """Stop ``torch.compile`` from tracing into this function.

    ``block_runner`` compiles the whole block forward, and the experts forward below is
    deliberately full of *Python-level* data-dependent control flow: the routed expert ids
    come back through ``.tolist()``, the number of loop iterations and fused groups follows
    them, and experts are looked up by name. Dynamo would retrace on every distinct routing
    -- i.e. on every calibration sample -- which is exactly the "fused MoE with
    shape-dependent control flow" case that already forces compile off for the DeepSeek and
    GLM-5 families (see ``special_model_handler``).

    ``linear_loop_experts_forward`` does not need this because its control flow is static
    (``for expert_idx in range(num_experts)``); only the tensor values inside vary.

    Making just this call opaque keeps the rest of the block compiled, and the quantization
    functions it calls are compiled in their own right, so nothing of value is lost.
    """
    disable = getattr(torch.compiler, "disable", None)
    if disable is None:  # pragma: no cover - torch < 2.1
        dynamo = getattr(torch, "_dynamo", None)
        disable = getattr(dynamo, "disable", None) if dynamo is not None else None
    return disable(fn) if disable is not None else fn


@_opaque_to_dynamo
def grouped_linear_experts_forward(
    self: nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Experts forward that batches the routed token/expert pairs into grouped GEMMs.

    Drop-in replacement for :func:`linear_loop_experts_forward` with the same module
    layout (numbered ``_ExpertContainer`` children holding ``gate_proj``/``up_proj``/
    ``down_proj``). Falls back to the loop implementation whenever the layer does not
    satisfy the grouped path's requirements.

    Args:
        self: The experts module.
        hidden_states: ``(num_tokens, hidden_dim)`` or ``(bs, seq_len, hidden_dim)``.
        top_k_index: Selected expert indices, ``(..., top_k)``.
        top_k_weights: Routing weights, ``(..., top_k)``.

    Returns:
        Output tensor with the same shape as ``hidden_states``.
    """
    from auto_round.modeling.fused_moe.moe_experts_interface import linear_loop_experts_forward

    num_experts = getattr(self, "num_experts", None)
    if not isinstance(num_experts, int) or num_experts <= 0:
        _log_fallback_once("num_experts is unavailable")
        return linear_loop_experts_forward(self, hidden_states, top_k_index, top_k_weights)

    if hidden_states.dim() == 3:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        flat_hidden_states = hidden_states.reshape(-1, hidden_dim)
        flat_index = top_k_index.reshape(-1, top_k_index.size(-1))
        flat_weights = top_k_weights.reshape(-1, top_k_weights.size(-1))
    else:
        batch_size, seq_len = None, None
        hidden_dim = hidden_states.size(-1)
        flat_hidden_states = hidden_states
        flat_index = top_k_index
        flat_weights = top_k_weights

    final_hidden_states = _run_routes(self, flat_hidden_states, flat_index, flat_weights, num_experts)
    if final_hidden_states is None:
        return linear_loop_experts_forward(self, hidden_states, top_k_index, top_k_weights)

    # Auxiliary coverage path: rotate the routing so every expert sees tokens (and its
    # hooks/statistics fire), discarding the output so model semantics stay untouched.
    if force_all_experts_routing_enabled():
        forced_indices, forced_weights = build_forced_routing(
            module=self,
            routing_scores=None,
            top_k=flat_index.size(-1),
            num_experts=num_experts,
            dtype=flat_hidden_states.dtype,
            num_tokens=flat_hidden_states.size(0),
            device=flat_hidden_states.device,
            normalize=True,
        )
        with torch.no_grad():
            _ = _run_routes(self, flat_hidden_states, forced_indices, forced_weights, num_experts)

    if batch_size is not None:
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_dim)
    return final_hidden_states

