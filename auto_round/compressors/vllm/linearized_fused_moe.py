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
"""Linearized FusedMoE replacement for vLLM quantization calibration.

vLLM stores all expert weights fused into two tensors:

  ``w13_weight``: ``[E, 2*intermediate, hidden]``  (gate + up, fused row-major)
  ``w2_weight``:  ``[E, hidden, intermediate]``    (down projection)

where ``E`` is the number of local experts.

This module replaces ``FusedMoE`` with per-expert ``nn.Linear`` modules so
that auto-round can calibrate activation statistics and quantize each expert
weight independently via the standard linear-layer pipeline.

The replacement module implements a simple top-k routing forward so that
``llm.generate()`` can drive calibration through the model normally.

After quantization, the individual expert weights are exported in a format
that vLLM's weight_loader can reassemble into ``FusedMoE``.

Memory efficiency
-----------------
Weight tensors are assigned as *views* of the original fused tensors (no
data copy).  For a model with E experts, this avoids 2× the peak memory
that cloning would require.  The views keep the underlying storage alive
even after the original ``FusedMoE`` module is removed from the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from auto_round.logger import logger
from auto_round.utils.common import global_state

# ---------------------------------------------------------------------------
# Activation helper
# ---------------------------------------------------------------------------


def _make_act_fn(activation):
    """Return a Python callable for the gated part of a ``MoEActivation``.

    Gated activations compute ``act(gate_proj(x)) * up_proj(x)``.
    We extract just the ``act`` portion here.

    Args:
        activation: A ``MoEActivation`` enum instance or a plain string.

    Returns:
        A callable ``f(x) -> x`` implementing the activation.
    """
    name = activation.value if hasattr(activation, "value") else str(activation)
    if "gelu" in name:
        return F.gelu
    elif "relu2" in name:
        return lambda x: F.relu(x) ** 2
    else:
        # Covers "silu" (most common) and unknown activations.
        if "silu" not in name:
            logger.warning_once(
                "Unknown MoE activation %r for linearized forward; falling back to SiLU.",
                name,
            )
        return F.silu


# ---------------------------------------------------------------------------
# Projection-name extraction from expert_mapping
# ---------------------------------------------------------------------------


def _extract_proj_names(
    expert_mapping,
) -> tuple[str, str, str]:
    """Derive checkpoint projection names from ``FusedMoE.expert_mapping``.

    ``make_expert_params_mapping`` stores tuples of the form::

        (param_name, weight_name, expert_id, shard_id)

    where ``shard_id`` is always one of ``"w1"`` (gate), ``"w2"`` (down),
    ``"w3"`` (up), and ``weight_name`` is the checkpoint key, e.g.
    ``"experts.0.gate_proj"`` (Qwen3) or ``"experts.0.w1"`` (Mixtral).
    The projection name is the third ``"."``-separated component.

    Returns:
        ``(gate_name, up_name, down_name)`` checkpoint projection names.
    """
    if not expert_mapping:
        return "gate_proj", "up_proj", "down_proj"

    gate_name = up_name = down_name = None
    for _param_name, weight_name, expert_id, shard_id in expert_mapping:
        if expert_id != 0:
            continue
        parts = weight_name.split(".")
        if len(parts) < 3:
            continue
        proj = parts[2]  # "experts.0.<proj_name>[.base_layer]"
        if shard_id == "w1":  # gate
            gate_name = proj
        elif shard_id == "w3":  # up
            up_name = proj
        elif shard_id == "w2":  # down
            down_name = proj

    return (
        gate_name or "gate_proj",
        up_name or "up_proj",
        down_name or "down_proj",
    )


# ---------------------------------------------------------------------------
# Per-expert module
# ---------------------------------------------------------------------------


class _VLLMLinearizedExpert(nn.Module):
    """Single MoE expert with gate / up / down as individual ``nn.Linear``.

    Sub-modules are registered under the checkpoint-specific projection names
    (e.g. ``w1`` / ``w3`` / ``w2`` for Mixtral, ``gate_proj`` / ``up_proj`` /
    ``down_proj`` for Qwen3), so that ``model.state_dict()`` immediately
    produces the keys expected by vLLM's weight_loader without any remapping.

    Weight tensors are *views* into the parent ``FusedMoE``'s fused weight
    buffers (see :class:`LinearizedVLLMFusedMoE`).
    """

    def __init__(
        self,
        gate_proj: nn.Linear,
        up_proj: nn.Linear,
        down_proj: nn.Linear,
        act_fn,
        gate_name: str = "gate_proj",
        up_name: str = "up_proj",
        down_name: str = "down_proj",
    ) -> None:
        super().__init__()
        # Register with checkpoint-specific names so state_dict keys are correct.
        self.add_module(gate_name, gate_proj)
        self.add_module(up_name, up_proj)
        self.add_module(down_name, down_proj)
        # Store attr names for forward – avoid re-registering as nn.Module children.
        self._gate_attr = gate_name
        self._up_attr = up_name
        self._down_attr = down_name
        self._act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = getattr(self, self._gate_attr)(x)
        up = getattr(self, self._up_attr)(x)
        return getattr(self, self._down_attr)(self._act_fn(gate) * up)


# ---------------------------------------------------------------------------
# Linearized MoE block
# ---------------------------------------------------------------------------


class LinearizedVLLMFusedMoE(nn.Module):
    """Linearized replacement for vLLM's ``FusedMoE``.

    Extracts per-expert weight *views* (zero extra memory) from
    ``w13_weight`` / ``w2_weight`` and wraps them as ``nn.Linear`` so
    auto-round can calibrate and quantize each expert independently.

    ``SharedFusedMoE`` support
    --------------------------
    ``SharedFusedMoE`` (used by Qwen3-MoE) inherits from ``FusedMoE`` and
    returns a ``(shared_out, routed_out)`` tuple.  When the constructor
    receives a ``shared_expert`` module (the sibling ``nn.Module`` in the
    parent MoE block), the forward runs it and returns the same tuple so
    that the parent block's forward code continues to work unchanged.

    The ``shared_expert`` module is stored as a plain Python reference
    (``self._shared_expert_ref``), *not* as a registered child, so it
    is not double-counted in the model's parameter inventory.
    """

    def __init__(
        self,
        fused_moe,
        shared_expert: nn.Module | None = None,
    ) -> None:
        super().__init__()

        num_experts: int = fused_moe.local_num_experts
        hidden_size: int = fused_moe.moe_config.hidden_dim
        inter: int = fused_moe.moe_config.intermediate_size_per_partition

        self.num_experts = num_experts
        self.top_k = fused_moe.top_k
        self.renormalize = fused_moe.renormalize
        self.use_grouped_topk: bool = getattr(fused_moe, "use_grouped_topk", False)
        self.num_expert_group: int | None = getattr(fused_moe, "num_expert_group", None)
        self.topk_group: int | None = getattr(fused_moe, "topk_group", None)
        self.scoring_func: str = getattr(fused_moe, "scoring_func", "softmax")
        self._act_fn = _make_act_fn(fused_moe.activation)

        # SharedFusedMoE detection – store as plain reference (not registered).
        _is_shared: bool = type(fused_moe).__name__ == "SharedFusedMoE"
        self._is_shared_moe: bool = _is_shared
        self._shared_expert_ref: nn.Module | None = shared_expert if _is_shared else None

        # Derive checkpoint projection names from the original expert_mapping so
        # state_dict() keys match vLLM's weight_loader expectations with no
        # post-export remapping.
        gate_name, up_name, down_name = _extract_proj_names(getattr(fused_moe, "expert_mapping", None))

        # ------------------------------------------------------------------
        # Build per-expert nn.Linear from *views* of the fused weight tensors.
        #
        # w13_weight: [E, 2*inter, hidden]
        #   rows  0 ..  inter-1  → gate weights
        #   rows inter .. 2*inter-1 → up weights
        #
        # w2_weight: [E, hidden, inter] → down weights
        #
        # Experts are registered directly as numbered children ("0", "1", ...)
        # on this module – NOT wrapped in a ModuleList – so the module path is
        # ``<fused_moe_attr>.{i}.{proj}`` instead of the double
        # ``<fused_moe_attr>.experts.{i}.{proj}`` that a ModuleList would give.
        # ------------------------------------------------------------------
        w13: torch.Tensor = fused_moe.w13_weight.data  # [E, 2*inter, hidden]
        w2: torch.Tensor = fused_moe.w2_weight.data  # [E, hidden, inter]

        for i in range(num_experts):
            gate = nn.Linear(hidden_size, inter, bias=False)
            up = nn.Linear(hidden_size, inter, bias=False)
            down = nn.Linear(inter, hidden_size, bias=False)

            # Views – contiguous slices, no allocation.
            gate.weight = nn.Parameter(w13[i, :inter, :], requires_grad=False)
            up.weight = nn.Parameter(w13[i, inter:, :], requires_grad=False)
            down.weight = nn.Parameter(w2[i], requires_grad=False)

            self.add_module(
                str(i),
                _VLLMLinearizedExpert(
                    gate,
                    up,
                    down,
                    self._act_fn,
                    gate_name=gate_name,
                    up_name=up_name,
                    down_name=down_name,
                ),
            )

        # Store original FusedMoE as a plain dict entry (NOT a registered
        # submodule) so that vLLM model code can reach attributes like
        # ``is_internal_router`` via __getattr__ delegation below.
        self.__dict__["_fused_moe_ref"] = fused_moe

    def __getattr__(self, name: str):
        """Delegate unknown attributes to the original FusedMoE.

        vLLM model code accesses several attributes on the ``experts`` object
        (e.g. ``is_internal_router``) that are defined on ``FusedMoE`` but not
        on ``LinearizedVLLMFusedMoE``.  Delegating to the original preserves
        those semantics without requiring us to enumerate every attribute.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        try:
            fused_moe = self.__dict__.get("_fused_moe_ref")
            if fused_moe is not None:
                return getattr(fused_moe, name)
        except AttributeError:
            pass
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing weights and selected expert indices.

        Supports standard softmax / sigmoid scoring and grouped top-k
        (DeepSeek-style).  For calibration purposes a close approximation
        to the original routing is sufficient.

        Args:
            router_logits: ``[num_tokens, num_experts]``

        Returns:
            ``(routing_weights, selected_experts)`` both of shape
            ``[num_tokens, top_k]``.
        """
        if self.scoring_func == "sigmoid":
            scores = torch.sigmoid(router_logits.float())
        else:
            scores = torch.softmax(router_logits.float(), dim=-1)

        if self.use_grouped_topk and self.num_expert_group and self.topk_group:
            # Grouped top-k (DeepSeek): select top-topk_group scores per group,
            # then pick top_k globally among the selected groups.
            E = self.num_experts
            G = self.num_expert_group
            group_size = E // G

            scores_grouped = scores.view(-1, G, group_size)
            # Sum of top-topk_group scores per group → group importance
            group_scores = scores_grouped.topk(self.topk_group, dim=-1)[0].sum(-1)
            n_groups_selected = max(1, G // 2)
            _, sel_groups = group_scores.topk(n_groups_selected, dim=-1)  # [T, n_sel]

            mask = torch.zeros_like(scores)
            for gi in range(G):
                in_sel = (sel_groups == gi).any(dim=-1)  # [T]
                mask[:, gi * group_size : (gi + 1) * group_size] = in_sel.unsqueeze(-1).to(mask.dtype)
            scores = scores * mask

        routing_weights, selected_experts = torch.topk(scores, self.top_k, dim=-1)

        if self.renormalize:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        return routing_weights, selected_experts

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor | None, torch.Tensor]:
        """Run linearized top-k MoE routing and expert computation.

        Args:
            hidden_states: ``[num_tokens, hidden_size]``
            router_logits: ``[num_tokens, num_experts]``

        Returns:
            For plain ``FusedMoE``: a tensor of shape
            ``[num_tokens, hidden_size]``.

            For ``SharedFusedMoE``: a tuple
            ``(shared_out, routed_out)`` where *shared_out* is the output of
            the shared expert (or ``None`` if no shared expert is present).
        """
        routing_weights, selected_experts = self._route(router_logits)
        routing_weights = routing_weights.to(hidden_states.dtype)

        out = torch.zeros_like(hidden_states)
        for expert_idx in range(self.num_experts):
            mask = selected_experts == expert_idx  # [tokens, top_k]
            token_idx, topk_idx = mask.nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue
            expert_out = self._modules[str(expert_idx)](hidden_states[token_idx])
            weight = routing_weights[token_idx, topk_idx].unsqueeze(-1)
            out.index_add_(0, token_idx, (expert_out * weight).to(out.dtype))

        if self._is_shared_moe:
            shared_out = self._shared_expert_ref(hidden_states) if self._shared_expert_ref is not None else None
            return shared_out, out

        return out


# ---------------------------------------------------------------------------
# Model-level linearize entry point
# ---------------------------------------------------------------------------


def linearize_vllm_moe(model: nn.Module) -> int:
    """Replace all ``FusedMoE`` instances in *model* with :class:`LinearizedVLLMFusedMoE`.

    Walks the module tree and replaces every ``FusedMoE`` child (including
    ``SharedFusedMoE`` subclass instances) with a
    :class:`LinearizedVLLMFusedMoE` that exposes per-expert ``nn.Linear``
    modules for calibration and quantization.

    Only gated MoE layers (``is_act_and_mul=True``) are processed.
    Non-gated layers are left unchanged with a warning.

    For ``SharedFusedMoE``, the function looks for a ``shared_expert``
    attribute on the *parent* module (the sibling MoE MLP) and passes it
    to :class:`LinearizedVLLMFusedMoE` so the forward can produce the correct
    ``(shared_out, routed_out)`` tuple.

    Args:
        model: The vLLM model (``nn.Module``) after loading.

    Returns:
        Number of ``FusedMoE`` layers replaced.
    """
    try:
        from vllm.model_executor.layers.fused_moe import FusedMoE
    except ImportError:
        return 0

    replaced = 0
    for parent_module in list(model.modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if not isinstance(child_module, FusedMoE):
                continue

            if not child_module.moe_config.is_act_and_mul:
                logger.warning_once(
                    "Skipping FusedMoE linearization at %r: non-gated activation "
                    "(is_act_and_mul=False) is not yet supported.",
                    child_name,
                )
                continue

            # For SharedFusedMoE: pass the sibling shared_expert from the parent.
            shared_expert = getattr(parent_module, "shared_expert", None)

            linearized = LinearizedVLLMFusedMoE(child_module, shared_expert=shared_expert)
            setattr(parent_module, child_name, linearized)
            replaced += 1

    if replaced:
        logger.info(
            "Linearized %d FusedMoE layer(s) into per-expert nn.Linear for quantization.",
            replaced,
        )
        # Inform safe_to_cpu_() that modules have been replaced so it does NOT
        # call model.to("cpu") — vLLM models must stay on GPU.
        global_state.replaced_module_count = replaced
    return replaced
