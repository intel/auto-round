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

"""Tensor-parallel activation QDQ preserving native shard layouts."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from auto_round.schemes import needs_nvfp_global_scale

if TYPE_CHECKING:
    from torch.autograd.function import FunctionCtx


class _ReplicatedScale(torch.autograd.Function):
    """Preserve a replicated scale while summing its shard-local gradients."""

    @staticmethod
    def forward(ctx: FunctionCtx, value: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        """Clone the scale and retain its gradient-reduction group.

        Args:
            ctx: Autograd context used to retain the process group.
            value: Replicated coefficient tensor.
            group: Process group spanning the contributing shards.

        Returns:
            A clone of the coefficient tensor.
        """
        ctx.group = group
        return value.clone()

    @staticmethod
    def backward(ctx: FunctionCtx, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Sum coefficient gradients across the saved process group.

        Args:
            ctx: Autograd context containing the process group.
            *grad_outputs: Output gradients; the first belongs to the cloned scale.

        Returns:
            A tuple of the reduced value gradient and None for the process group.
        """
        grad = grad_outputs[0].clone()
        dist.all_reduce(grad, group=ctx.group)
        return grad, None


class _GlobalAmax(torch.autograd.Function):
    """Compute a distributed absolute maximum with gradients shared across ties."""

    @staticmethod
    def forward(ctx: FunctionCtx, value: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        """Reduce the absolute maximum and count tied extrema across shards.

        Args:
            ctx: Autograd context used to save values, extrema, and the process group.
            value: Local activation shard, possibly empty.
            group: Process group spanning all activation shards.

        Returns:
            A replicated float32 scalar containing the global absolute maximum.
        """
        maximum = value.float().abs().amax() if value.numel() else value.new_zeros((), dtype=torch.float32)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
        mask = value.float().abs() == maximum
        count = mask.sum().to(torch.float32)
        dist.all_reduce(count, group=group)
        ctx.group = group
        ctx.save_for_backward(value, mask, count)
        return maximum

    @staticmethod
    def backward(ctx: FunctionCtx, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Distribute the summed maximum gradient equally among global extrema.

        Args:
            ctx: Autograd context containing saved local values and extrema statistics.
            *grad_outputs: Output gradients; the first belongs to the global maximum.

        Returns:
            A tuple of the local activation gradient and None for the process group.
        """
        value, mask, count = ctx.saved_tensors
        grad = grad_outputs[0].clone()
        dist.all_reduce(grad, group=ctx.group)
        return (value.sign() * mask * (grad / count.clamp_min(1))).to(value.dtype), None


class TensorParallelActivationQuantizer:
    """Adapt a quant function to local tensors and replicated coefficients.

    Attributes:
        quant_func: Underlying shard-local activation quantization callable.
        mesh: Weight mesh defining the activation collective group.
        input_width: Global input-channel width used to detect local channel shards.
        local_forward: Whether local expert forwards require coefficient-gradient reduction.
    """

    def __init__(
        self,
        quant_func: Callable[..., tuple[torch.Tensor, Any, Any]],
        weight: DTensor,
        local_forward: bool = False,
    ) -> None:
        """Bind activation quantization to a distributed weight layout.

        Args:
            quant_func: Callable returning quantized values, scale, and zero point.
            weight: DTensor weight defining the mesh and global input width.
            local_forward: Whether the projection bypasses native child TP forwards.
        """
        self.quant_func = quant_func
        self.mesh = weight.device_mesh
        self.input_width = weight.shape[1]
        self.local_forward = local_forward

    def __call__(self, x: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, Any, Any]:
        """Quantize local activations while preserving layout and shared-scale gradients.

        Args:
            x: Local activation tensor or DTensor on the weight mesh.
            **kwargs: Quantization options, including group size, data type, and scales.

        Returns:
            A tuple of quantized activations, scale, and zero point. Empty shards retain
            backward dependencies and return two None values; DTensors retain their layout.

        Raises:
            NotImplementedError: The activation mesh differs or its layout is partial.
            ValueError: A channel shard splits an activation quantization group.
        """
        local = x.to_local() if isinstance(x, DTensor) else x
        if isinstance(x, DTensor):
            if x.device_mesh != self.mesh or any(p.is_partial() for p in x.placements):
                raise NotImplementedError("Activation QDQ requires a matching TP mesh and non-partial input.")
            sharded = any(isinstance(p, Shard) for p in x.placements)
            channel_shard = any(isinstance(p, Shard) and p.dim % x.ndim == x.ndim - 1 for p in x.placements)
        else:
            sharded = channel_shard = local.shape[-1] != self.input_width
        if channel_shard and local.shape[-1] % kwargs["group_size"]:
            raise ValueError("Row-wise TP must not split an activation quantization group.")

        # A scalar shared by all ranks needs the sum of shard-local gradients.
        # Replicated inputs already receive complete gradients through native TP.
        if sharded or self.local_forward:
            for name in ("min_scale", "max_scale"):
                scale = kwargs.get(name)
                if isinstance(scale, torch.Tensor) and scale.requires_grad:
                    kwargs[name] = _ReplicatedScale.apply(scale, self.mesh.get_group())

        data_type = kwargs["data_type"]
        if needs_nvfp_global_scale(data_type) and kwargs.get("global_scale") is None:
            from auto_round.data_type.nvfp import calculate_gparam_by_dtype

            maximum = kwargs.get("tensor_max")
            if maximum is None:
                if sharded:
                    maximum = _GlobalAmax.apply(local, self.mesh.get_group())
                else:
                    maximum = local.float().abs().amax() if local.numel() else local.new_zeros((), dtype=torch.float32)
            kwargs["global_scale"] = calculate_gparam_by_dtype(maximum, data_type, kwargs["group_size"]).to(
                local.device
            )
        if local.numel() == 0:
            # Traverse the same QDQ graph as nonempty peers, but contribute zero
            # gradients. Returning x directly drops scale/amax backward reductions.
            width = local.shape[-1] or max(1, kwargs["group_size"])
            probe = local.new_zeros((1, width)) + local.sum() * 0
            probe, _, _ = self.quant_func(probe, **kwargs)
            quantized, scale, zp = local + probe.sum() * 0, None, None
        else:
            quantized, scale, zp = self.quant_func(local, **kwargs)
        if isinstance(x, DTensor):
            quantized = DTensor.from_local(
                quantized, self.mesh, x.placements, run_check=False, shape=x.shape, stride=x.stride()
            )
        return quantized, scale, zp
