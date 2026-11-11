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

"""Naive pipeline-parallel layer partitioning without overlapping microbatches."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def resolve_device_map(size: int) -> str:
    """Resolve visible CUDA devices for single-process pipeline placement.

    Args:
        size: Number of pipeline-placement devices to select.

    Returns:
        A comma-separated string of device indices starting at zero.

    Raises:
        ValueError: CUDA is unavailable or fewer than ``size`` devices are visible.
    """
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < size:
        raise ValueError(f"pp_size={size} requires that many visible CUDA devices.")
    return ",".join(str(index) for index in range(size))


def dispatch_auto_block(
    block: torch.nn.Module,
    inputs: list[torch.Tensor] | dict[str, Any],
    *,
    device_list: Sequence[str | torch.device],
    device: str | torch.device,
    low_gpu_mem_usage: bool,
    batch_size: int,
) -> tuple[torch.nn.Module, str | torch.device, bool]:
    """Place a tuning block automatically and install leaf-device alignment hooks.

    Args:
        block: Block to dispatch in place.
        inputs: Cached calibration inputs used for memory estimates.
        device_list: Candidate devices for automatic placement.
        device: Primary execution device.
        low_gpu_mem_usage: Whether placement should conserve accelerator memory.
        batch_size: Tuning batch size used for placement estimates.

    Returns:
        A tuple of the dispatched block, loss device, and primary-device memory-risk flag.
    """
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    from auto_round.utils.device import set_auto_device_map_for_block_with_tuning

    at_risk, loss_device = set_auto_device_map_for_block_with_tuning(
        block, device_list, inputs, low_gpu_mem_usage, batch_size, device
    )
    for _, module in block.named_modules():
        if not list(module.children()) and hasattr(module, "tuning_device"):
            add_hook_to_module(module, AlignDevicesHook(module.tuning_device, io_same_device=True), True)
    return block, loss_device, at_risk
