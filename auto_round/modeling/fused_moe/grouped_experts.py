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
5. Run the routed GEMM: by default torch's native ``grouped_mm`` kernel (one launch for all
   experts); ``AR_MOE_EXPERTS_IMPL=linear_grouped_sliced`` forces one ``F.linear`` per
   *active* expert over its contiguous slice instead (see the note above
   ``_native_grouped_mm_available``).
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
# Same grouped forward, but forcing the sliced per-expert GEMM loop instead of the native
# ``grouped_mm`` kernel. Selected via ``AR_MOE_EXPERTS_IMPL=linear_grouped_sliced``.
GROUPED_LINEAR_SLICED_IMPL = "linear_grouped_sliced"

# Set once if the native grouped_mm kernel raises; afterwards we always use the sliced loop.
_NATIVE_GROUPED_MM_DISABLED = False
_LOGGED_FALLBACK_REASONS: set[tuple[str, str]] = set()
# Bumped whenever the plan-building path changes; logged once per process so a
# log unambiguously identifies which grouped implementation produced it.
GROUPED_PLANS_VERSION = "slot-stage-v3"
_LOGGED_VERSION = False


def _log_fallback_once(reason: str, detail: str = "") -> None:
    global _LOGGED_VERSION
    if not _LOGGED_VERSION:
        _LOGGED_VERSION = True
        logger.debug(f"[MoE grouped] plans version {GROUPED_PLANS_VERSION}")
    key = (reason, detail)
    if key in _LOGGED_FALLBACK_REASONS:
        return
    _LOGGED_FALLBACK_REASONS.add(key)
    msg = f"[MoE grouped] falling back to linear_loop: {reason}"
    if detail:
        msg = f"{msg} ({detail})"
    logger.debug(msg)


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


def _hooks_are_alignment_only(module: nn.Module) -> bool:
    """Whether every hook on ``module`` is one the grouped path satisfies manually.

    accelerate's ``AlignDevicesHook`` (mapped placement) and ``AddContiguousHook`` only
    move/contiguify inputs -- the grouped path does both explicitly (``.to(plan.device)``
    on the way in, contiguous ``index_select`` gathers). A hook that carries an offload
    (``io_has_offload`` / weights_offload) still must run: it loads the weights, which the
    grouped path reads directly. Calibration hooks (act_max collectors) are not alignment
    hooks and remain a hard fallback so they keep firing.
    """
    hooks = list(module._forward_pre_hooks.values()) + list(module._forward_hooks.values())
    if not hooks:
        return True
    hook_classes = []
    try:
        from accelerate.hooks import AlignDevicesHook

        hook_classes.append(AlignDevicesHook)
    except Exception:  # pragma: no cover - accelerate always present in our lanes
        pass
    try:  # newer accelerate only
        from accelerate.hooks import AddContiguousHook

        hook_classes.append(AddContiguousHook)
    except Exception:
        pass
    if not hook_classes:
        return False
    for h in hooks:
        if not isinstance(h, tuple(hook_classes)):
            return False
        # Offload-carrying variants (weights streaming in from host) must still run
        # their own forward; alignment-only variants are satisfied manually.
        if getattr(h, "offload", False) or getattr(h, "io_has_offload", False) or getattr(h, "weights_offload", None):
            return False
    return True


def _projection_is_supported(layer: nn.Module) -> bool:
    if type(layer) is nn.Linear:
        # A plain Linear carrying forward (pre-)hooks must run its own forward so the hooks
        # fire. This is how act_max is collected during calibration: the composer registers a
        # forward hook on each expert Linear, and the grouped path -- which multiplies the
        # weights directly and never calls ``Linear.forward`` -- would silently skip them,
        # leaving every expert without ``act_max`` (breaking static-act export, e.g. NVFP4).
        # Alignment-only accelerate hooks are exempt: the grouped path performs that exact
        # move itself (mapped multi-GPU placement attaches them to off-primary experts).
        if (layer._forward_pre_hooks or layer._forward_hooks) and not _hooks_are_alignment_only(layer):
            return False
        return True
    if not _is_wrapper_linear(layer):
        return False
    # accelerate's stage hooks attach to the tree module -- the wrapper itself.
    if (layer._forward_pre_hooks or layer._forward_hooks) and not _hooks_are_alignment_only(layer):
        return False
    orig = layer.orig_layer
    if type(orig) is not nn.Linear:
        return False  # Conv1D / LinearAllreduce need their own forward
    if (orig._forward_pre_hooks or orig._forward_hooks) and not _hooks_are_alignment_only(orig):
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
# ``AR_MOE_CHUNK`` overrides it; 0 or negative means "fuse everything".
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
    setting = envs.AR_MOE_CHUNK
    if setting in ("", "auto"):
        return _auto_qdq_chunk(layers)
    try:
        chunk = int(setting)
    except ValueError:
        logger.warning_once(f"Ignoring AR_MOE_CHUNK={setting!r}: expected 'auto' or an integer.")
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
    an explicit ``AR_MOE_CHUNK=0`` asks to fuse everything, since that group's size
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


def _sliced_grouped_mm_requested() -> bool:
    """Whether the sliced per-expert GEMM loop was explicitly requested.

    ``AR_MOE_EXPERTS_IMPL=linear_grouped_sliced`` forces the sliced loop; every other
    grouped setting (``auto`` / ``linear_grouped``) prefers the native ``grouped_mm`` kernel.
    """
    return str(envs.AR_MOE_EXPERTS_IMPL).lower() == GROUPED_LINEAR_SLICED_IMPL


def _native_grouped_mm_available() -> bool:
    """Cheap pre-check that does not need the stacked 3-D operand."""
    if _NATIVE_GROUPED_MM_DISABLED or _sliced_grouped_mm_requested():
        return False
    return hasattr(F, "grouped_mm") or hasattr(torch, "_grouped_mm")


def _native_grouped_mm_preferred(device: torch.device) -> bool:
    """Whether the native kernel is opted into. On by default; see the note above."""
    return not _sliced_grouped_mm_requested()


try:
    from transformers.integrations.moe import _can_use_grouped_mm as _transformers_can_use_grouped_mm
except Exception:  # pragma: no cover - transformers-internal API drift; degrade to the sliced loop
    _transformers_can_use_grouped_mm = None


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
    # Non-cuda devices (xpu/hpu/cpu): we cannot verify native grouped_mm support
    # here, and an unsupported F.grouped_mm would raise mid-forward -- degrade
    # to the sliced/loop fallback instead of gambling on the native path.
    return False


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


def _stack_weights_range(slot: _SlotWeights, start: int, stop: int) -> torch.Tensor:
    """``(E', out, in)`` operand for a slice of experts, as a view when the qdq was fused."""
    if slot.stacked is not None:
        return slot.stacked[start:stop]
    return torch.stack(slot.weights[start:stop], dim=0)


def _add_grouped_bias(out: torch.Tensor, slot: _SlotWeights, counts: list[int]) -> torch.Tensor:
    """Add per-expert bias to a grouped output whose rows are in sorted-expert order."""
    if slot.biases[0] is None:
        return out
    device = out.device
    row_expert = torch.repeat_interleave(
        torch.arange(len(counts), device=device),
        torch.tensor(counts, device=device),
    )
    return out + torch.stack(slot.biases, dim=0)[row_expert]


def _gemm_expert_chunk(num_experts: int, weight: torch.Tensor) -> int:
    """Experts per native grouped_mm tile. ``num_experts`` (the full count) disables tiling.

    Follows the single ``AR_MOE_CHUNK`` knob (same value as the qdq fusion), so a tile is
    exactly one qdq chunk: the ``torch.stack`` ``(E, out, in)`` operand the native kernel needs
    is bounded to ``chunk`` experts. ``auto`` derives the count from the working-set budget;
    a fixed int is used as-is; ``0``/negative means "fuse everything" -> no tiling. Tiling only
    actually runs when the qdq produced no free stacked view (see ``_grouped_linear``).
    """
    setting = envs.AR_MOE_CHUNK
    if setting in ("", "auto"):
        per_expert = int(weight.shape[0]) * int(weight.shape[1]) * weight.element_size()
        if per_expert <= 0:
            return num_experts
        return max(_DEFAULT_QDQ_CHUNK, min(num_experts, int(_QDQ_FUSED_WORKINGSET_BYTES // per_expert)))
    try:
        chunk = int(setting)
    except ValueError:
        return num_experts
    return num_experts if chunk <= 0 else min(chunk, num_experts)


def _grouped_mm_tiled(x: torch.Tensor, slot: _SlotWeights, counts: list[int], chunk: int) -> torch.Tensor | None:
    """Native grouped_mm tiled over experts, bounding the stacked weight to ``chunk`` experts.

    ``x`` is sorted so each expert owns a contiguous row slice; a tile of ``chunk`` experts
    owns the concatenation of their slices. Returns ``None`` (fall back to the sliced loop)
    if any tile is not eligible for the native kernel. Bias is added by the caller.
    """
    num_experts = len(slot.weights)
    outputs = []
    row = 0
    for start in range(0, num_experts, chunk):
        stop = min(start + chunk, num_experts)
        tile_counts = counts[start:stop]
        rows = sum(tile_counts)
        x_tile = x[row : row + rows]
        row += rows
        weight = _stack_weights_range(slot, start, stop).transpose(-2, -1)  # (E', in, out)
        sub_offsets = torch.tensor(tile_counts, device=x.device, dtype=torch.int32).cumsum(0).to(torch.int32)
        if not _native_grouped_mm_usable(x_tile, weight, sub_offsets):
            return None
        outputs.append(_native_grouped_mm(x_tile, weight, sub_offsets))
    return torch.cat(outputs, dim=0)


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
        num_experts = len(slot.weights)
        # Tile only when the qdq did not hand us a free stacked view: that is exactly the
        # case where feeding all experts costs a full torch.stack copy.
        chunk = num_experts if slot.stacked is not None else _gemm_expert_chunk(num_experts, slot.weights[0])
        try:
            if chunk >= num_experts:
                weight = _stack_weights(slot).transpose(-2, -1)  # (E, in, out)
                if _native_grouped_mm_usable(x, weight, offsets):
                    out = _native_grouped_mm(x, weight, offsets)
                    return _add_grouped_bias(out, slot, counts)
            else:
                out = _grouped_mm_tiled(x, slot, counts, chunk)
                if out is not None:
                    return _add_grouped_bias(out, slot, counts)
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

    # Slot-stage plans: the balancer scatters projections independently, so one
    # expert's gate/up/down can live on different devices. Each SLOT partitions its
    # active experts by that slot's home device and runs one grouped GEMM per group;
    # full pair-indexed buffers carry intermediates between stages. (token, top_k)
    # pairs are unique, so per-group index_copy_ into shared buffers is exact and
    # the final sum(dim=1) combine is unchanged.
    def _slot_groups(slot_name: str) -> list[tuple[torch.device, list[int]]]:
        groups: dict[torch.device, list[int]] = {}
        for pos, expert_id in enumerate(active_ids):
            expert = getattr(module, str(expert_id), None)
            proj = getattr(expert, slot_name, None) if expert is not None else None
            if proj is None:
                _log_fallback_once("expert container missing", detail=f"expert {expert_id} slot {slot_name}")
                return []
            if not _projection_is_supported(proj):
                # check BEFORE _compute_device: packed QuantLinear types
                # (e.g. MXFP4QuantLinear) have no .weight, and the support
                # probe must be what rejects them, not an AttributeError
                _log_fallback_once(
                    f"unsupported projection type {type(proj).__name__}", detail=f"expert {expert_id} slot {slot_name}"
                )
                return []
            groups.setdefault(_compute_device(proj), []).append(pos)
        return list(groups.items())

    has_gate = hasattr(getattr(module, str(active_ids[0]), None), "gate_proj")
    gate_groups = _slot_groups("gate_proj") if has_gate else []
    up_groups = _slot_groups("up_proj")
    down_groups = _slot_groups("down_proj")
    if not up_groups or not down_groups or (has_gate and not gate_groups):
        return None

    # Per-slot-group validation: projections supported; within a group (batched
    # together) quant signatures and weight layouts must agree. gate/up activation
    # settings must match per expert (they quantize the same input tensor).
    for slot_name, groups in (("up_proj", up_groups), ("gate_proj", gate_groups), ("down_proj", down_groups)):
        for _, positions in groups:
            sig = layout = None
            for p in positions:
                proj = getattr(getattr(module, str(active_ids[p])), slot_name)
                if not _projection_is_supported(proj):
                    _log_fallback_once(f"unsupported projection type {type(proj).__name__}")
                    return None
                cur_sig = _quant_signature(proj)
                if sig is None:
                    sig = cur_sig
                elif sig != cur_sig:
                    _log_fallback_once("mixed quantization schemes across experts in a device group")
                    return None
                cur_layout = _weight_layout(proj)
                if layout is None:
                    layout = cur_layout
                elif layout != cur_layout:
                    _log_fallback_once("experts have different weight shape/dtype in a device group")
                    return None
    if has_gate:
        for expert_id in active_ids:
            expert = getattr(module, str(expert_id))
            gate_act = _act_signature(expert.gate_proj) if _is_wrapper_linear(expert.gate_proj) else None
            up_act = _act_signature(expert.up_proj) if _is_wrapper_linear(expert.up_proj) else None
            gate_enabled = bool(getattr(expert.gate_proj, "enable_act_quant", False))
            up_enabled = bool(getattr(expert.up_proj, "enable_act_quant", False))
            if gate_enabled != up_enabled or (gate_enabled and gate_act != up_act):
                _log_fallback_once("gate_proj/up_proj activation quantization differ")
                return None

    perm_valid = perm[:num_valid]
    token_idx = torch.div(perm_valid, num_top_k, rounding_mode="floor")
    x_all = hidden_states.index_select(0, token_idx)  # (num_valid, hidden) on input device

    # Contiguous sorted-order row range for each active position.
    ranges = []
    start = 0
    for c in active_counts:
        ranges.append((start, start + c))
        start += c

    def _run_slot(slot_name: str, groups, inp: torch.Tensor) -> torch.Tensor:
        """Grouped GEMM for one slot; returns a full (num_valid, out) buffer on the input device."""
        out_buf = None
        for dev, positions in groups:
            counts = [active_counts[p] for p in positions]
            rows = torch.cat(
                [torch.arange(ranges[p][0], ranges[p][1], device=perm.device, dtype=torch.int64) for p in positions]
            )
            x_g = inp.index_select(0, rows).to(dev)
            offsets = torch.tensor(counts, device=dev, dtype=torch.int32).cumsum(0).to(torch.int32)
            experts_g = [getattr(module, str(active_ids[p])) for p in positions]
            x_g = _quantize_activation(getattr(experts_g[0], slot_name), x_g)
            batched = all(_supports_batched_qdq(getattr(e, slot_name)) for e in experts_g)
            w = _slot_weights([getattr(e, slot_name) for e in experts_g], batched)
            out_g = _grouped_linear(x_g, w, counts, offsets)
            if out_buf is None:
                out_buf = torch.zeros(num_valid, out_g.size(-1), device=inp.device, dtype=out_g.dtype)
            # device-only .to() would crash on cross-group dtype mixes; the
            # buffer dtype is the first group's expert-output dtype by design
            out_buf.index_copy_(0, rows.to(inp.device), out_g.to(device=inp.device, dtype=out_buf.dtype))
        return out_buf

    up_out = _run_slot("up_proj", up_groups, x_all)
    if has_gate:
        gate_out = _run_slot("gate_proj", gate_groups, x_all)
        if hasattr(module, "_apply_gate"):
            hidden_mid = module._apply_gate(torch.cat([gate_out, up_out], dim=-1))
        else:
            hidden_mid = module.act_fn(gate_out) * up_out
    else:
        hidden_mid = module.act_fn(up_out)

    out = _run_slot("down_proj", down_groups, hidden_mid)
    sample_weights_out = sample_weights.index_select(0, perm_valid).to(device=out.device, dtype=out.dtype)
    out = out * sample_weights_out.unsqueeze(-1)

    # Scatter the weighted rows into the shared (token, top_k) buffer. The buffer
    # carries the EXPERT OUTPUT dtype (bf16 weights produce bf16 rows even when the
    # input chain is fp32 -- the GDN lanes); the original single-device path used
    # out.dtype for exactly this reason.
    out_per_sample = torch.zeros(num_pairs, hidden_dim, device=device, dtype=out.dtype)
    out_per_sample.index_copy_(0, perm_valid.to(device), out.to(device))
    return out_per_sample.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)


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


# Config attributes various architectures store the expert count under. transformers' own
# experts forwards (``grouped_mm_experts_forward`` / ``batched_mm_experts_forward``) read
# ``self.num_experts`` directly, and every ``@use_experts_implementation`` module sets it in
# ``__init__``; AutoRound's unfuse sets it too. These config keys are the ones those
# ``__init__`` methods derive ``num_experts`` from (e.g. DeepSeek ``n_routed_experts``,
# Mixtral ``num_local_experts``), used only as a fallback so a module that leaves the
# attribute unset still resolves instead of silently dropping to the per-expert loop.
_NUM_EXPERTS_CONFIG_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")


def _resolve_num_experts(module: nn.Module) -> int | None:
    """Resolve the expert count, matching transformers' ``self.num_experts`` convention.

    top_k is deliberately *not* resolved this way: transformers reads it per-call from
    ``top_k_index.size(-1)`` rather than from an attribute, which is what the forward below
    already does, so it stays correct even for models that never store a ``top_k`` field.
    """
    num_experts = getattr(module, "num_experts", None)
    if isinstance(num_experts, int) and num_experts > 0:
        return num_experts
    config = getattr(module, "config", None)
    if config is not None:
        for key in _NUM_EXPERTS_CONFIG_KEYS:
            value = getattr(config, key, None)
            if isinstance(value, int) and value > 0:
                return value
    return None


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

    num_experts = _resolve_num_experts(self)
    if num_experts is None:
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
