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

"""Shard-local activation QDQ without replacing native TP layout transforms."""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from auto_round.schemes import needs_nvfp_global_scale


class _ReplicatedScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        return value.clone()

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad = grad_outputs[0].clone()
        dist.all_reduce(grad, group=ctx.group)
        return grad, None


class _GlobalAmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        maximum = value.float().abs().amax() if value.numel() else value.new_zeros((), dtype=torch.float32)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
        mask = value.float().abs() == maximum
        count = mask.sum().to(torch.float32)
        dist.all_reduce(count, group=group)
        ctx.group = group
        ctx.save_for_backward(value, mask, count)
        return maximum

    @staticmethod
    def backward(ctx, *grad_outputs):
        value, mask, count = ctx.saved_tensors
        grad = grad_outputs[0].clone()
        dist.all_reduce(grad, group=ctx.group)
        return (value.sign() * mask * (grad / count.clamp_min(1))).to(value.dtype), None


class TensorParallelActivationQuantizer:
    """Adapt an existing quant function to local tensors and replicated coefficients."""

    def __init__(self, quant_func, weight, local_forward=False):
        self.quant_func = quant_func
        self.mesh = weight.device_mesh
        self.input_width = weight.shape[1]
        self.local_forward = local_forward

    def __call__(self, x, **kwargs):
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
            return x, None, None
        quantized, scale, zp = self.quant_func(local, **kwargs)
        if isinstance(x, DTensor):
            quantized = DTensor.from_local(
                quantized, self.mesh, x.placements, run_check=False, shape=x.shape, stride=x.stride()
            )
        return quantized, scale, zp
