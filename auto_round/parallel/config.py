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


def _reject_parallel_kwargs(kwargs):
    if "dp_size" in kwargs:
        raise TypeError("dp_size is not supported; configure DP replicas through parallel_config with data_parallel.")
    misplaced = {"tensor_parallel", "pipeline_parallel", "data_parallel", "parallel_batch"}.intersection(kwargs)
    if misplaced:
        names = ", ".join(sorted(misplaced))
        raise TypeError(f"Set {names} through parallel_config=ParallelConfig(...) or a mapping, not top-level kwargs.")


@dataclass(frozen=True, init=False)
class ParallelConfig:
    """Configure tensor, naive pipeline and data parallelism.

    TP uses Transformers' native automatic plan. ``pp_size`` selects naive
    pipeline parallelism: single-process layer partitioning without overlapping
    microbatch scheduling.
    ``data_parallel`` selects the single-process replica engine. Hybrid execution
    is not implemented.
    """

    tp_size: int = 1
    pp_size: int = 1
    data_parallel: str = "off"
    parallel_batch: str = "split"

    def __init__(
        self,
        tp_size=None,
        pp_size=None,
        *,
        tensor_parallel=None,
        pipeline_parallel=None,
        data_parallel="off",
        parallel_batch="split",
    ):
        for short, long, old, new in (
            ("tp_size", "tensor_parallel", tp_size, tensor_parallel),
            ("pp_size", "pipeline_parallel", pp_size, pipeline_parallel),
        ):
            if old is not None and new is not None and old != new:
                raise ValueError(f"Conflicting {short} and {long} values.")
            object.__setattr__(self, short, new if new is not None else (old if old is not None else 1))
        if isinstance(data_parallel, bool):
            raise ValueError("data_parallel accepts off|auto|N, not a boolean.")
        dp = str(data_parallel).strip().lower()
        if dp == "1":
            dp = "off"
        if dp not in ("off", "auto") and (not dp.isdigit() or int(dp) < 2):
            raise ValueError("data_parallel accepts off|auto|N (N >= 2); 1 disables DP.")
        if dp.isdigit():
            dp = str(int(dp))
        object.__setattr__(self, "data_parallel", dp)
        object.__setattr__(self, "parallel_batch", parallel_batch)
        self._validate()

    def _validate(self) -> None:
        for name in ("tp_size", "pp_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if self.tp_size > 1 and self.pp_size > 1:
            raise NotImplementedError("Hybrid TP/PP is not supported yet.")
        if self.data_parallel != "off" and (self.tp_size > 1 or self.pp_size > 1):
            raise NotImplementedError("data_parallel cannot be combined with TP/PP; hybrid execution is not supported.")
        if self.parallel_batch not in ("split", "per_replica"):
            raise ValueError("parallel_batch must be split or per_replica.")

    @property
    def tensor_parallel(self):
        return self.tp_size

    @property
    def pipeline_parallel(self):
        return self.pp_size

    @classmethod
    def from_value(cls, value: "ParallelConfig | Mapping | None") -> "ParallelConfig":
        """Normalize the public API without mutating caller-owned mappings."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            if "dp_size" in value:
                raise TypeError("dp_size is not supported; use data_parallel to select the DP replica count.")
            return cls(**dict(value))
        raise TypeError("parallel_config must be a ParallelConfig, mapping, or None.")

    def resolve_data_parallel_policy(self, device_map, *, iters=0):
        """Build the same replica policy for CLI and API before loading weights."""
        import os

        import torch
        import torch.distributed as dist

        from auto_round import envs
        from auto_round.logger import logger
        from auto_round.parallel.dp.engine import (
            ParallelPolicy,
            check_parallel_device_map_compat_,
            resolve_parallel_world,
        )

        if self.data_parallel == "off":
            return None
        world = dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", "1"))
        if world > 1:
            raise NotImplementedError("data_parallel is single-process; do not launch it with torchrun.")
        visible = torch.cuda.device_count()
        replicas = resolve_parallel_world(self.data_parallel, visible, iters=iters)
        check_parallel_device_map_compat_(replicas, device_map, iters)
        if self.data_parallel == "auto" and replicas < visible:
            logger.info("[tune-ddp] data_parallel auto -> world=%d (%d visible devices)", replicas, visible)
        return ParallelPolicy(
            world=replicas,
            source="auto" if self.data_parallel == "auto" else "explicit",
            collect_forward_cap=int(getattr(envs, "AR_TUNE_DDP_MAX_COLLECT_FORWARD_DEVICES", 0) or 0),
            batch_mode=self.parallel_batch,
        )

    def resolve_device_map(self, device_map):
        """Resolve rank-local TP devices or naive pipeline-parallel stages."""
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
                raise ValueError(
                    f"TP requires WORLD_SIZE={self.tp_size}, got {world_size}. "
                    "Use auto_round.quantize_and_save for automatic local launch, or launch AutoRound with torchrun."
                )
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                if not 0 <= local_rank < torch.cuda.device_count():
                    raise ValueError("LOCAL_RANK exceeds the number of visible CUDA devices.")
                torch.cuda.set_device(local_rank)
                return f"cuda:{local_rank}"
            return "cpu"
        from auto_round.parallel.pp.placement import resolve_device_map

        return resolve_device_map(self.pp_size)

    def to_transformers_config(self):
        """Create the native loading config, without deprecated tp_plan kwargs."""
        if self.tp_size == 1:
            return None
        try:
            from transformers import DistributedConfig
        except ImportError as exc:
            raise ImportError("Native TP requires a Transformers version exposing DistributedConfig.") from exc
        return DistributedConfig(tp_size=self.tp_size, tp_plan="auto")
