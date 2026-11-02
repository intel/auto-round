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
from functools import cache
from typing import cast

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard
from torch.func import functional_call

from auto_round.parallel.policy import validate_tp_quantization
from auto_round.schemes import needs_nvfp_global_scale
from auto_round.wrapper import WrapperLinear, WrapperWALayer


class TensorParallelQDQLinear(WrapperLinear):
    """Quantize a local weight shard without replacing the native TP forward.

    Only ordinary, evenly sharded 2-D Linear layers are accepted.
    In particular, row shards must contain complete quantization groups. No
    process groups, communication hooks or sharding plans are created here.
    """

    _tp_layer: torch.nn.Linear
    _local_wrapper_cls = WrapperLinear

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
        if orig_layer.act_bits <= 8 and axis == 1 and local_weight.shape[1] % orig_layer.act_group_size:
            raise ValueError("Row-wise TP must not split an activation quantization group.")
        if kwargs.get("enable_norm_bias_tuning", False):
            raise NotImplementedError("QDQ TP does not support norm/bias tuning yet.")
        if kwargs.get("enable_torch_compile", False):
            raise NotImplementedError("QDQ TP does not support torch.compile yet.")

        # A separate view keeps the ordinary QDQ implementation unaware of
        # distributed layouts. Never replace parameters on the live TP module.
        local_layer = copy.copy(orig_layer)
        local_layer._parameters = dict(orig_layer._parameters)
        local_layer._forward_pre_hooks = orig_layer._forward_pre_hooks.copy()
        local_layer.weight = torch.nn.Parameter(local_weight.detach(), requires_grad=False)
        if orig_layer.bias is not None:
            bias = orig_layer.bias
            local_bias = bias.to_local() if isinstance(bias, DTensor) else bias
            local_layer.bias = torch.nn.Parameter(local_bias.detach(), requires_grad=False)
        local_layer.in_features = local_weight.shape[1]
        local_layer.out_features = local_weight.shape[0]
        imatrix = getattr(local_layer, "imatrix", None)
        if imatrix is not None:
            if isinstance(imatrix, DTensor):
                imatrix = imatrix.full_tensor()
            if axis == 1 and imatrix.shape[-1] == weight.shape[1]:
                start = weight.device_mesh.get_local_rank() * local_weight.shape[1]
                imatrix = imatrix[..., start : start + local_weight.shape[1]]
            local_layer.imatrix = imatrix
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
        if orig_layer.act_bits <= 8 and needs_nvfp_global_scale(orig_layer.act_data_type):
            # Calibration observes local row inputs. Synchronize its tiny maximum
            # once, before tuning, rather than using different grids per rank.
            act_max = getattr(local_layer, "act_max", None)
            statistics = torch.zeros(2, dtype=torch.float32, device=local_weight.device)
            if act_max is not None:
                if isinstance(act_max, DTensor):
                    act_max = act_max.to_local()
                if act_max.numel():
                    statistics[0] = 1
                    statistics[1] = act_max.detach().to(local_weight.device).float().abs().amax()
            dist.all_reduce(statistics, op=dist.ReduceOp.MAX, group=weight.device_mesh.get_group())
            local_layer.act_max = statistics[1:] if statistics[0].item() else None
            input_scale = getattr(local_layer, "input_global_scale", None)
            if input_scale is not None:
                if isinstance(input_scale, DTensor):
                    input_scale = input_scale.to_local()
                input_scale = input_scale.detach().to(local_weight.device).float().clone()
                dist.all_reduce(input_scale, op=dist.ReduceOp.MIN, group=weight.device_mesh.get_group())
                local_layer.input_global_scale = input_scale
        super().__init__(local_layer, **kwargs)
        if self.enable_act_quant:
            from auto_round.parallel.activation import TensorParallelActivationQuantizer

            self.act_quant_func = TensorParallelActivationQuantizer(
                self.act_quant_func, weight, local_forward=getattr(orig_layer, "_auto_round_tp_local_forward", False)
            )
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
        handle = None
        if self.enable_act_quant:

            def quantize_input(module, inputs):
                activation, _, _ = self._qdq_act(
                    inputs[0],
                    act_min_scale=self.act_min_scale,
                    act_max_scale=self.act_max_scale,
                    act_max=getattr(self.orig_layer, "act_max", None),
                )
                return (activation,) + inputs[1:]

            # Existing smoothing hooks run first, as in ordinary WrapperLinear.
            handle = self._tp_layer.register_forward_pre_hook(quantize_input)
        try:
            return functional_call(self._tp_layer, {"weight": self._distributed_weight(quantized)}, (x,))
        finally:
            if handle is not None:
                handle.remove()

    def restore_original(self) -> torch.nn.Linear:
        """Restore the native module after an aborted tuning attempt."""
        return self._tp_layer

    @torch.no_grad()
    def unwrapper(self, best_params):
        """Bake the local shard and retain distributed scale/zero-point metadata."""
        local_result = super().unwrapper(best_params)
        local_layer = local_result.orig_layer if isinstance(local_result, WrapperWALayer) else local_result
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
        for name in ("_ar_warm_start", "_ar_warm_start_wrapper_cls"):
            if hasattr(original, name):
                delattr(original, name)
        if self.enable_act_quant:
            for name in (
                "act_max",
                "act_scale",
                "act_min_scale",
                "act_max_scale",
                "act_data_type",
                "act_quant_func",
                "input_global_scale",
            ):
                if hasattr(local_layer, name):
                    setattr(original, name, getattr(local_layer, name))
            result = WrapperWALayer(original, enable_torch_compile=False, device=self.device)
            if hasattr(original, "global_name"):
                result.global_name = original.global_name
            return result
        return original


@cache
def tensor_parallel_wrapper(wrapper_class):
    """Compose TP transport with explicitly shard-local algorithm weight math."""
    if wrapper_class in (WrapperLinear, TensorParallelQDQLinear):
        return TensorParallelQDQLinear
    if not issubclass(wrapper_class, WrapperLinear) or not wrapper_class.__dict__.get(
        "supports_tensor_parallel", False
    ):
        raise NotImplementedError(f"{wrapper_class.__name__} has not implemented the native TP wrapper contract.")
    if issubclass(wrapper_class, TensorParallelQDQLinear):
        return wrapper_class
    return type(
        f"TensorParallel{wrapper_class.__name__}",
        (TensorParallelQDQLinear, wrapper_class),
        {"_local_wrapper_cls": wrapper_class, "supports_tensor_parallel": True},
    )
