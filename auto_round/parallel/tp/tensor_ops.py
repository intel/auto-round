# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shard-local weight operations for algorithm-specific statistics/transforms.

Only channel statistics are gathered/reduced. No operation materializes a full
weight matrix, and writes preserve native DTensor parameters and forward hooks.
"""

from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist

from auto_round.parallel.tp.runtime import is_dtensor


class ParallelTensorOps:
    """Provide local weight transforms and collective channel statistics.

    Operations preserve native DTensor layouts and never gather full weight matrices.
    Ranks sharing a weight mesh must invoke collective operations in the same order.
    """

    @staticmethod
    def local(value: Any) -> Any:
        """Extract local storage without gathering distributed tensors.

        Args:
            value: DTensor or value to leave unchanged.

        Returns:
            The rank-local tensor for DTensor input, otherwise the original value.
        """
        return value.to_local() if is_dtensor(value) else value

    @staticmethod
    def shard_axis(value: torch.Tensor) -> int | None:
        """Find the sharded axis for a tensor on a one-dimensional mesh.

        Args:
            value: Tensor whose first placement is inspected.

        Returns:
            The non-negative shard axis, or None for local or unsharded tensors.
        """
        if not is_dtensor(value):
            return None
        placement = value.placements[0]
        return placement.dim % value.ndim if placement.is_shard() else None

    @classmethod
    def channel_scale(cls, weight: torch.Tensor, scales: torch.Tensor, axis: int) -> torch.Tensor:
        """Slice a replicated channel vector to the weight's local channel span.

        Args:
            weight: Local or distributed weight defining the channel layout.
            scales: Global channel scales or an already-local scale vector.
            axis: Weight dimension corresponding to the scale vector.

        Returns:
            Channel scales on the weight device, sliced when the axis is sharded.
        """
        scales = scales.to(weight.device)
        if cls.shard_axis(weight) == axis:
            width = cls.local(weight).shape[axis]
            if scales.shape[-1] == width and width != weight.shape[axis]:
                return scales
            start = weight.device_mesh.get_local_rank() * width
            scales = scales[start : start + width]
        return scales

    @classmethod
    def input_activation(cls, layer: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
        """Recover global channels from an explicitly row-sharded local input.

        Args:
            layer: Projection whose weight defines the input-channel layout.
            value: DTensor activation, local channel shard, or complete activation.

        Returns:
            The full activation when a distributed layout is identifiable, else the input.
        """
        from torch.distributed.tensor import DTensor, Shard

        if is_dtensor(value):
            return value.full_tensor()
        weight = getattr(layer, "weight", None)
        if (
            isinstance(value, torch.Tensor)
            and weight is not None
            and cls.shard_axis(weight) == 1
            and value.shape[-1] == cls.local(weight).shape[1]
            and value.shape[-1] != weight.shape[1]
        ):
            shape = torch.Size((*value.shape[:-1], weight.shape[1]))
            stride = torch.empty(shape, device="meta").stride()
            return DTensor.from_local(
                value, weight.device_mesh, [Shard(value.ndim - 1)], run_check=False, shape=shape, stride=stride
            ).full_tensor()
        return value

    @classmethod
    @torch.no_grad()
    def copy_weight(cls, layer: torch.nn.Module, local_weight: torch.Tensor) -> None:
        """Overwrite local weight storage without replacing its distributed parameter.

        Args:
            layer: Module whose weight is updated in place.
            local_weight: Replacement tensor with the same shape as the local shard.

        Raises:
            ValueError: The replacement changes the local weight shape.
        """
        target = cls.local(layer.weight)
        if target.shape != local_weight.shape:
            raise ValueError("A shard-local weight update must preserve the local weight shape.")
        target.copy_(local_weight)

    @classmethod
    def weight_sum(cls, weight: torch.Tensor, local_sum: torch.Tensor) -> torch.Tensor:
        """Sum disjoint weight errors, without multiplying replicated errors.

        Args:
            weight: Weight whose placement determines whether reduction is required.
            local_sum: Local error sum, reduced in place for sharded weights.

        Returns:
            The error sum across unique weight elements.
        """
        if cls.shard_axis(weight) is not None:
            dist.all_reduce(local_sum, group=weight.device_mesh.get_group())
        return local_sum

    @classmethod
    @torch.no_grad()
    def input_channel_means(cls, layers: Sequence[torch.nn.Module], group_size: int | None) -> torch.Tensor:
        """Compute global input-channel means of group-normalized absolute weights.

        Args:
            layers: Nonempty collection of projections sharing an input-channel width.
            group_size: Quantization group size; non-positive values use the full width.

        Returns:
            A replicated vector of normalized weight means for each input channel.

        Raises:
            NotImplementedError: A row-wise shard splits a quantization group.
        """
        from torch.distributed.tensor import DTensor, Shard

        from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad

        total, count = None, 0
        for layer in layers:
            original = layer.weight
            weight = cls.local(original).detach().float()
            shape = weight.shape
            size = group_size if group_size and group_size > 0 else original.shape[1]
            if cls.shard_axis(original) == 1 and shape[1] % size:
                raise NotImplementedError("A TP input shard must contain complete quantization groups.")
            grouped, _, padding = reshape_pad_tensor_by_group_size(weight, size)
            normalized = grouped.abs() / (grouped.abs().amax(dim=1, keepdim=True) + 1e-6)
            normalized = revert_tensor_by_pad(normalized, orig_shape=shape, pad_len=padding)
            channel_sum = normalized.sum(0)
            rows = shape[0]
            if cls.shard_axis(original) == 0:
                dist.all_reduce(channel_sum, group=original.device_mesh.get_group())
                rows = original.shape[0]
            elif cls.shard_axis(original) == 1:
                channel_sum = DTensor.from_local(
                    channel_sum,
                    original.device_mesh,
                    [Shard(0)],
                    run_check=False,
                    shape=torch.Size((original.shape[1],)),
                    stride=(1,),
                ).full_tensor()
            total = channel_sum if total is None else total + channel_sum
            count += rows
        return total / count

    @classmethod
    @torch.no_grad()
    def scale_inputs(cls, layer: torch.nn.Module, scales: torch.Tensor) -> None:
        """Multiply local weight columns by their input-channel scales in place.

        Args:
            layer: Projection whose local weight is modified.
            scales: Global or already-local input-channel scale vector.
        """
        local = cls.local(layer.weight)
        local.mul_(cls.channel_scale(layer.weight, scales, 1).view(1, -1))

    @classmethod
    @torch.no_grad()
    def divide_outputs(cls, layer: torch.nn.Module, scales: torch.Tensor, *, unit_offset: bool = False) -> None:
        """Divide output weights and optional bias by channel scales in place.

        Args:
            layer: Projection or normalization layer to modify.
            scales: Output scales, optionally covering only trailing projection rows.
            unit_offset: Whether one-dimensional weights represent a scale offset from one.
        """
        weight = layer.weight
        local = cls.local(weight)
        if weight.ndim == 1:
            scale = cls.channel_scale(weight, scales, 0)
            local.copy_((1 + local) / scale - 1 if unit_offset else local / scale)
        else:
            # Some smooth mappings operate only on the trailing output rows.
            width = local.shape[0]
            start = weight.device_mesh.get_local_rank() * width if cls.shard_axis(weight) == 0 else 0
            first = weight.shape[0] - scales.numel()
            left, right = max(start, first), min(start + width, weight.shape[0])
            if left < right:
                local[left - start : right - start].div_(
                    scales[left - first : right - first].to(local.device).view(-1, 1)
                )
        bias = getattr(layer, "bias", None)
        if bias is not None:
            cls.local(bias).div_(cls.channel_scale(bias, scales, 0))
