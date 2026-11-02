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

"""Public parallelism policy; process groups remain owned by the caller/Transformers."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelConfig:
    """Configure independent tensor, pipeline-placement and data parallel axes.

    TP uses Transformers' native automatic plan. ``pp_size`` selects the existing
    single-process multi-device placement path, not a pipeline training schedule.
    Explicit DP and hybrid execution are reserved and currently rejected.
    """

    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1

    def __post_init__(self) -> None:
        for name in ("tp_size", "pp_size", "dp_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if self.dp_size != 1:
            raise NotImplementedError("Explicit DP configuration is not supported yet; dp_size must be 1.")
        if self.tp_size > 1 and self.pp_size > 1:
            raise NotImplementedError("Hybrid TP/PP is not supported yet.")

    @classmethod
    def from_value(cls, value: "ParallelConfig | Mapping | None") -> "ParallelConfig":
        """Normalize the public API without mutating caller-owned mappings."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError("parallel_config must be a ParallelConfig, mapping, or None.")

    def resolve_device_map(self, device_map):
        """Resolve rank-local TP placement or the existing multi-device PP path."""
        import os

        import torch
        import torch.distributed as dist

        if self.tp_size == self.pp_size == 1:
            return device_map
        if device_map not in (None, 0, "0"):
            raise ValueError("parallel_config cannot be combined with an explicit device_map.")
        if self.tp_size > 1:
            world_size = dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", "1"))
            if world_size != self.tp_size:
                raise ValueError(f"TP requires WORLD_SIZE={self.tp_size}, got {world_size}. Launch with torchrun.")
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                if local_rank >= torch.cuda.device_count():
                    raise ValueError("LOCAL_RANK exceeds the number of visible CUDA devices.")
                torch.cuda.set_device(local_rank)
                return f"cuda:{local_rank}"
            return "cpu"
        if not torch.cuda.is_available() or torch.cuda.device_count() < self.pp_size:
            raise ValueError(f"pp_size={self.pp_size} requires that many visible CUDA devices.")
        return ",".join(str(index) for index in range(self.pp_size))

    def to_transformers_config(self):
        """Create the native loading config, without deprecated tp_plan kwargs."""
        if self.tp_size == 1:
            return None
        try:
            from transformers import DistributedConfig
        except ImportError as exc:
            raise ImportError("Native TP requires a Transformers version exposing DistributedConfig.") from exc
        return DistributedConfig(tp_size=self.tp_size, tp_plan="auto")
