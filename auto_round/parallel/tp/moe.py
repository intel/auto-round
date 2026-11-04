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

"""Tensor-parallel MoE shard views preserving native communication."""

from types import MethodType

import torch
from torch import nn
from torch.distributed.tensor import DTensor, Shard


def _local_linear_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Apply an expert projection without duplicating parent-owned TP collectives.

    Args:
        self: Expert Linear module containing local or distributed weight storage.
        x: Local expert input activations.

    Returns:
        The shard-local linear projection output.
    """
    # The parent MoeExpertsParallel owns input-gradient and output reductions.
    # Installing colwise/rowwise communication on these children would double it.
    weight = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
    return torch.nn.functional.linear(x, weight, self.bias)


def _validate_experts(module: nn.Module) -> None:
    """Accept only the native concatenated [expert, output, input] TP layout.

    Args:
        module: Fused expert module with gate/up and down projection parameters.

    Raises:
        NotImplementedError: Storage, expert features, mesh, or placements are unsupported.
        ValueError: Projection shapes disagree or intermediate shards are uneven or empty.
    """
    gate_up, down = module.gate_up_proj, module.down_proj
    if not isinstance(gate_up, DTensor) or not isinstance(down, DTensor):
        raise NotImplementedError("Fused TP experts require DTensor gate_up_proj and down_proj weights.")
    if (
        not getattr(module, "is_concatenated", False)
        or getattr(module, "is_transposed", True)
        or getattr(module, "has_bias", True)
        or getattr(module, "has_post_expert_norm", False)
        or getattr(module, "_is_expert_parallel", False)
        or gate_up.ndim != 3
        or down.ndim != 3
        or gate_up.device_mesh.ndim != 1
        or down.device_mesh != gate_up.device_mesh
        or len(gate_up.placements) != 1
        or len(down.placements) != 1
    ):
        raise NotImplementedError("TP MoE requires concatenated, non-transposed, bias-free experts without EP.")
    packed, row = gate_up.placements[0], down.placements[0]
    # Do not import PyTorch's private _StridedShard: inspect its public layout
    # properties. An ordinary Shard(1) would split gate and up incorrectly.
    if (
        getattr(packed, "dim", -1) % 3 != 1
        or getattr(packed, "split_factor", None) != 2
        or not isinstance(row, Shard)
        or row.dim % 3 != 2
        or getattr(row, "split_factor", 1) != 1
    ):
        raise NotImplementedError("TP MoE requires packed_colwise gate/up and rowwise down projections.")
    experts, outputs, hidden = gate_up.shape
    intermediate = outputs // 2
    size = gate_up.device_mesh.size()
    if (
        outputs % 2
        or down.shape != (experts, hidden, intermediate)
        or intermediate == 0
        or intermediate % size
        or tuple(gate_up.to_local().shape) != (experts, 2 * intermediate // size, hidden)
        or tuple(down.to_local().shape) != (experts, hidden, intermediate // size)
    ):
        raise ValueError("TP MoE requires evenly sized, nonempty intermediate shards and matching expert shapes.")


def prepare_tp_moe(model: nn.Module) -> list[str]:
    """Expose local expert projections for QDQ; keep every native TP forward.

    Native fused parameters are retained as zero-copy aliases. Transformers
    checks for sharded parameters after forward to decide whether to reduce the
    output; QDQ wrappers hide their originals from registered parameter walks.
    These aliases preserve that check and are omitted from CPU export.

    Args:
        model: Native TP model to adapt in place after validating all candidate experts.

    Returns:
        Names of the fused expert modules adapted during this call.

    Raises:
        NotImplementedError: Fused projections or their native TP layouts are unsupported.
        ValueError: Expert shapes are incompatible or shards are uneven or empty.
        RuntimeError: Transformers does not provide the required experts interface.
    """
    from auto_round.modeling.fused_moe.fusion_spec import build_standard_moe_fusion_spec, register_moe_fusion_spec
    from auto_round.modeling.fused_moe.moe_experts_interface import (
        KNOWN_PROJECTION_PATTERNS,
        _ExpertContainer,
        _new_meta_linear,
        register_linear_loop_experts,
    )

    candidates = []
    for name, module in model.named_modules():
        if getattr(module, "_auto_round_tp_moe", False):
            continue
        parameters = dict(module.named_parameters(recurse=False))
        if not any(
            p.ndim == 3 and (isinstance(p, DTensor) or key in KNOWN_PROJECTION_PATTERNS)
            for key, p in parameters.items()
        ):
            continue
        if set(parameters) != {"gate_up_proj", "down_proj"}:
            raise NotImplementedError(f"Unsupported fused TP expert projections on {name!r}.")
        _validate_experts(module)
        candidates.append((name, module))
    if not candidates:
        return []
    if not register_linear_loop_experts():
        raise RuntimeError("Native TP MoE requires the Transformers experts interface.")
    # Validate all candidates before mutating any of them.
    for _, module in candidates:
        gate_up, down = module.gate_up_proj, module.down_proj
        count, _, hidden = gate_up.shape
        intermediate = down.shape[-1]
        gate, up = gate_up.to_local().detach().chunk(2, dim=1)
        down_local = down.to_local().detach()
        spec = build_standard_moe_fusion_spec(
            {key: KNOWN_PROJECTION_PATTERNS[key] for key in ("gate_up_proj", "down_proj")},
            count,
            False,
            module,
        )
        for index in range(count):
            expert = _ExpertContainer()
            for key, local, shape, axis in (
                ("gate_proj", gate[index], (intermediate, hidden), 0),
                ("up_proj", up[index], (intermediate, hidden), 0),
                ("down_proj", down_local[index], (hidden, intermediate), 1),
            ):
                linear = _new_meta_linear(shape[1], shape[0], False, local.dtype)
                linear.weight = nn.Parameter(
                    DTensor.from_local(
                        local,
                        gate_up.device_mesh,
                        [Shard(axis)],
                        run_check=False,
                        shape=torch.Size(shape),
                        stride=(shape[1], 1),
                    ),
                    requires_grad=False,
                )
                linear.forward = MethodType(_local_linear_forward, linear)
                linear._auto_round_tp_local_forward = True
                expert.add_module(key, linear)
            module.add_module(str(index), expert)
        module.config._experts_implementation = "linear_loop"
        module._unfused_experts = True
        module._auto_round_tp_moe = True
        register_moe_fusion_spec(module, spec)
    model.config._experts_implementation = "linear_loop"
    return [name for name, _ in candidates]
