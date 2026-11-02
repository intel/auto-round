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

"""Local quantization math with native DTensor layout and autograd semantics."""

import copy
from typing import cast

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard
from torch.func import functional_call

from auto_round.parallel.policy import validate_tp_quantization
from auto_round.schemes import needs_nvfp_global_scale
from auto_round.wrapper import WrapperLinear


class TensorParallelQDQLinear(WrapperLinear):
    """Quantize a local weight shard without replacing the native TP forward.

    Only ordinary, evenly sharded 2-D weight-only Linear layers are accepted.
    In particular, row shards must contain complete quantization groups. No
    process groups, communication hooks or sharding plans are created here.
    """

    _tp_layer: torch.nn.Linear

    def __init__(self, orig_layer: torch.nn.Linear, **kwargs):
        weight = orig_layer.weight
        if type(orig_layer) is not torch.nn.Linear or not isinstance(weight, DTensor):
            raise TypeError("TensorParallelQDQLinear requires an nn.Linear with a DTensor weight.")
        placements = weight.placements
        if weight.device_mesh.ndim != 1 or len(placements) != 1 or not isinstance(placements[0], Shard):
            raise NotImplementedError("QDQ TP currently supports one-dimensional, sharded Linear weights only.")
        axis = cast(Shard, placements[0]).dim % weight.ndim
        local_weight = weight.to_local()
        if weight.shape[axis] % weight.device_mesh.size() or local_weight.numel() == 0:
            raise ValueError("QDQ TP requires evenly sized, nonempty weight shards.")
        validate_tp_quantization(orig_layer)
        group_size = orig_layer.group_size
        if axis == 1 and (group_size == -1 or local_weight.shape[1] % group_size):
            raise ValueError(
                "Row-wise TP must not split a quantization group; use a group_size dividing the local width."
            )
        if kwargs.get("enable_norm_bias_tuning", False):
            raise NotImplementedError("QDQ TP does not support norm/bias tuning yet.")
        if kwargs.get("enable_torch_compile", False):
            raise NotImplementedError("QDQ TP does not support torch.compile yet.")

        # A separate view keeps the ordinary QDQ implementation unaware of
        # distributed layouts. Never replace parameters on the live TP module.
        local_layer = copy.copy(orig_layer)
        local_layer._parameters = dict(orig_layer._parameters)
        local_layer.weight = torch.nn.Parameter(local_weight.detach(), requires_grad=False)
        if orig_layer.bias is not None:
            bias = orig_layer.bias
            local_bias = bias.to_local() if isinstance(bias, DTensor) else bias
            local_layer.bias = torch.nn.Parameter(local_bias.detach(), requires_grad=False)
        local_layer.in_features = local_weight.shape[1]
        local_layer.out_features = local_weight.shape[0]
        # Placement hints must never move a shard onto another rank's device.
        local_layer.tuning_device = local_weight.device
        kwargs["device"] = local_weight.device
        if needs_nvfp_global_scale(orig_layer.data_type):
            from auto_round.data_type.nvfp import calculate_gparam_by_dtype

            # A per-shard scale changes the quantization grid. Reduce only a
            # scalar, not the weight, and use the mesh group rather than WORLD.
            amax = local_weight.detach().float().abs().amax()
            dist.all_reduce(amax, op=dist.ReduceOp.MAX, group=weight.device_mesh.get_group())
            local_layer.weight_global_scale = calculate_gparam_by_dtype(amax, orig_layer.data_type, group_size).to(
                local_weight.device
            )
        super().__init__(local_layer, **kwargs)
        # Do not register a second copy of the original TP module in the wrapper
        # tree: optimizer/state_dict traversal must see each parameter only once.
        object.__setattr__(self, "_tp_layer", orig_layer)

    def _distributed_weight(self, local_weight):
        weight = self._tp_layer.weight
        return DTensor.from_local(
            local_weight,
            device_mesh=weight.device_mesh,
            placements=weight.placements,
            run_check=False,
            shape=weight.shape,
            stride=weight.stride(),
        )

    def forward(self, x):
        """Use functional parameter substitution so native TP transforms still run."""
        quantized, _, _ = self._qdq_weight(self.value, self.min_scale, self.max_scale)
        return functional_call(self._tp_layer, {"weight": self._distributed_weight(quantized)}, (x,))

    def restore_original(self) -> torch.nn.Linear:
        """Restore the native module after an aborted tuning attempt."""
        return self._tp_layer

    @torch.no_grad()
    def unwrapper(self, best_params):
        """Bake the local shard and retain distributed scale/zero-point metadata."""
        local_layer = cast(torch.nn.Linear, super().unwrapper(best_params))
        original = self._tp_layer
        original.weight.to_local().copy_(local_layer.weight)
        groups = (
            1 if original.group_size == -1 else (original.in_features + original.group_size - 1) // original.group_size
        )
        metadata_shape = torch.Size((original.out_features, groups))
        for name in ("scale", "zp"):
            value = getattr(local_layer, name, None)
            if isinstance(value, torch.Tensor) and (name == "scale" or not original.sym or value.numel() > 1):
                value = value.reshape(local_layer.out_features, -1).to(original.weight.device)
                value = DTensor.from_local(
                    value,
                    original.weight.device_mesh,
                    original.weight.placements,
                    run_check=False,
                    shape=metadata_shape,
                    stride=(groups, 1),
                )
            setattr(original, name, value)
        if self.weight_global_scale is not None:
            original.weight_global_scale = self.weight_global_scale.detach().clone()
        original.q_scale_thresh = self.q_scale_thresh
        return original
