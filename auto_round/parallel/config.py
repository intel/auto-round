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

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import DistributedConfig

    from auto_round.parallel.dp.engine import ParallelPolicy


def _reject_parallel_kwargs(kwargs: Mapping[str, Any]) -> None:
    """Reject parallel options passed outside the public configuration.

    Args:
        kwargs (Mapping[str, Any]): Constructor keyword arguments to inspect without modifying.

    Raises:
        TypeError: A removed or misplaced parallel option is present.
    """
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

    Attributes:
        tp_size (int): Number of tensor-parallel ranks.
        pp_size (int): Number of devices used for pipeline placement.
        data_parallel (str): Replica policy, specified as ``off``, ``auto``, or a count.
        parallel_batch (str): Whether replicas split a batch or each receive a full batch.
    """

    tp_size: int = 1
    pp_size: int = 1
    data_parallel: str = "off"
    parallel_batch: str = "split"

    def __init__(
        self,
        tp_size: int | None = None,
        pp_size: int | None = None,
        *,
        tensor_parallel: int | None = None,
        pipeline_parallel: int | None = None,
        data_parallel: str | int = "off",
        parallel_batch: str = "split",
    ) -> None:
        """Initialize and validate a non-hybrid parallel policy.

        Args:
            tp_size (int | None): Tensor-parallel degree, defaulting to one.
            pp_size (int | None): Pipeline-parallel degree, defaulting to one.
            tensor_parallel (int | None): Alias for ``tp_size``.
            pipeline_parallel (int | None): Alias for ``pp_size``.
            data_parallel (str | int): ``off``, ``auto``, or a replica count; one disables DP.
            parallel_batch (str): Batch policy, either ``split`` or ``per_replica``.

        Raises:
            ValueError: Aliases conflict or a degree or policy is invalid.
            NotImplementedError: Multiple parallel modes are requested together.
        """
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
        """Validate degrees and reject unsupported parallel combinations.

        Raises:
            ValueError: A parallel degree or batch policy is invalid.
            NotImplementedError: A hybrid parallel topology is requested.
        """
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
    def tensor_parallel(self) -> int:
        """int: Number of tensor-parallel ranks."""
        return self.tp_size

    @property
    def pipeline_parallel(self) -> int:
        """int: Number of pipeline-placement devices."""
        return self.pp_size

    @classmethod
    def from_value(cls, value: ParallelConfig | Mapping | None) -> ParallelConfig:
        """Normalize the public API without mutating caller-owned mappings.

        Args:
            value (ParallelConfig | Mapping[str, Any] | None): Configuration, option mapping, or None.

        Returns:
            ParallelConfig: The existing configuration or a newly constructed configuration.

        Raises:
            TypeError: The value has an unsupported type or contains ``dp_size``.
            ValueError: Configuration options are invalid or conflicting.
            NotImplementedError: Configuration options request hybrid execution.
        """
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            if "dp_size" in value:
                raise TypeError("dp_size is not supported; use data_parallel to select the DP replica count.")
            return cls(**dict(value))
        raise TypeError("parallel_config must be a ParallelConfig, mapping, or None.")

    def resolve_data_parallel_policy(
        self, device_map: str | int | dict[str, Any] | None, *, iters: int = 0
    ) -> ParallelPolicy | None:
        """Build the same replica policy for CLI and API before loading weights.

        Args:
            device_map (str | int | dict | None): Requested placement used to validate DP compatibility.
            iters (int): Number of tuning iterations, including zero-shot execution.

        Returns:
            ParallelPolicy | None: Resolved replica policy, or None when DP is disabled.

        Raises:
            NotImplementedError: DP is requested in a multi-process environment.
            ValueError: The replica count or device mapping is incompatible.
        """
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

    def resolve_device_map(self, device_map: str | int | dict[str, Any] | None) -> str | int | dict[str, Any] | None:
        """Resolve rank-local TP devices or naive pipeline-parallel stages.

        Args:
            device_map (str | int | dict | None): Existing placement; explicit maps cannot accompany TP/PP.

        Returns:
            str | int | dict | None: Unchanged map, rank-local device, or pipeline device list string.

        Raises:
            ValueError: Placement, world size, rank, or available devices are invalid.
            NotImplementedError: PP is requested in a multi-process environment.
        """
        import os

        import torch
        import torch.distributed as dist

        if self.tp_size == self.pp_size == 1:
            return device_map
        if device_map not in (None, 0, "0"):
            raise ValueError("parallel_config cannot be combined with an explicit device_map.")
        world_size = dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", "1"))
        if self.tp_size > 1:
            if world_size != self.tp_size:
                raise ValueError(
                    f"TP requires WORLD_SIZE={self.tp_size}, got {world_size}. "
                    "Use auto_round.quantize_and_save for automatic local launch, or launch AutoRound with torchrun."
                )
            if torch.cuda.is_available():
                local_rank = (
                    int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else torch.cuda.current_device()
                )
                if not 0 <= local_rank < torch.cuda.device_count():
                    raise ValueError("LOCAL_RANK exceeds the number of visible CUDA devices.")
                if "LOCAL_RANK" in os.environ:
                    torch.cuda.set_device(local_rank)
                return f"cuda:{local_rank}"
            return "cpu"
        if world_size > 1 or int(os.environ.get("WORLD_SIZE", "1")) > 1:
            raise NotImplementedError("Pipeline parallelism is single-process; do not launch it with torchrun.")
        from auto_round.parallel.pp.placement import resolve_device_map

        return resolve_device_map(self.pp_size)

    def to_transformers_config(self) -> DistributedConfig | None:
        """Create the native loading config, without deprecated tp_plan kwargs.

        Returns:
            transformers.DistributedConfig | None: Native loading configuration, or None without TP.

        Raises:
            ImportError: Transformers does not expose ``DistributedConfig``.
        """
        if self.tp_size == 1:
            return None
        try:
            from transformers import DistributedConfig
        except ImportError as exc:
            raise ImportError("Native TP requires a Transformers version exposing DistributedConfig.") from exc
        return DistributedConfig(tp_size=self.tp_size, tp_plan="auto")
