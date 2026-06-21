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

import os
from functools import lru_cache

import torch

from auto_round.logger import logger


@lru_cache(maxsize=None)
def is_distributed():
    import torch.distributed as dist

    return dist.is_initialized() and dist.get_world_size() > 1


def setup_ddp_if_needed_(ar, block: torch.nn.Module, device_list: list[int]):
    """Prepare ``block`` for distributed training and return a gradient-sync hook.

    Returns ``(block, sync_fn)`` where ``sync_fn()`` must be called after gradient
    accumulation and before ``optimizer.step()``.

    * **Non-distributed**: returns ``(block, noop)``.
    * **Single GPU per rank**: wraps ``block`` with DDP, which synchronizes
      gradients automatically during the backward pass.  ``sync_fn`` is a no-op.
    * **Multi-GPU per rank**: the block's submodules are sharded across GPUs
      and cannot be DDP-wrapped.  ``sync_fn`` performs a manual ``all_reduce``
      (AVG) on every parameter that has a gradient, so that each rank sees the
      same averaged gradient before taking a step.

    .. note::
       ``ReduceOp.AVG`` is safe for SignSGD because
       ``sign(avg(g)) == sign(sum(g))`` for any non-zero sum.

    Args:
        ar: AutoRound instance invoking the helper.
        block: Model block that may need DDP wrapping.
        device_list: Device identifiers assigned to the current rank.

    Returns:
        ``(block, sync_fn)``
    """
    import torch.distributed as dist

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")

    if not is_distributed():
        return block, _noop_sync

    num_devices = len(device_list)

    if num_devices == 1:
        from torch.nn.parallel import DistributedDataParallel as DDP

        logger.warning_once("AutoRound DDP is an experimental feature, please use with caution.")
        logger.trace(
            "[Rank: %d] Wrapping block with DDP on device_list=%s, CUDA_VISIBLE_DEVICES=%s",
            dist.get_rank(),
            device_list,
            visible_devices,
        )
        # Ensure all block parameters are on the DDP device before wrapping.
        _move_block_to_device(block, device_list[0])
        block = DDP(block, device_ids=device_list, find_unused_parameters=True)
        return block, _noop_sync

    # Multi-GPU per rank: block is sharded, can't DDP-wrap.
    # Use manual all_reduce on gradients before optimizer step.
    logger.warning_once("AutoRound multi-GPU DDP is an experimental feature, " "please use with caution.")
    logger.trace(
        (
            "[Rank: %d] Multi-GPU(device_list=%s, CUDA_VISIBLE_DEVICES=%s) per rank (%d GPUs),"
            " using manual all_reduce sync"
        ),
        dist.get_rank(),
        str(device_list),
        visible_devices,
        num_devices,
    )

    def _sync_gradients():
        _all_reduce_model_grads(block)

    return block, _sync_gradients


def _noop_sync():
    pass


def _move_block_to_device(module: torch.nn.Module, device):
    """Move *module* to *device*, used before DDP wrapping."""
    module.to(device)


def _all_reduce_model_grads(module: torch.nn.Module):
    """All-reduce (AVG) gradients of all parameters in *module* across ranks."""
    import torch.distributed as dist

    comm_device = torch.cuda.current_device() if torch.cuda.is_available() else None
    if comm_device is not None:
        comm_device = torch.device("cuda", comm_device)
    for param in module.parameters():
        if param.grad is not None:
            grad = param.grad
            if comm_device is not None and grad.is_cuda and grad.device != comm_device:
                synced = grad.to(comm_device)
                dist.all_reduce(synced, op=dist.ReduceOp.AVG)
                grad.copy_(synced.to(grad.device))
            else:
                dist.all_reduce(grad, op=dist.ReduceOp.AVG)
