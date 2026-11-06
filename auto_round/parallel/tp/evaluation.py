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

"""Transition from CLI-owned tensor-parallel quantization to local evaluation."""

import os
from collections.abc import Iterator
from contextlib import contextmanager

import torch.distributed as dist

from auto_round.logger import logger
from auto_round.parallel.distributed import is_distributed

_DISTRIBUTED_ENV_VARS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "GROUP_WORLD_SIZE",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "ACCELERATE_USE_DEEPSPEED",
    "ACCELERATE_USE_FSDP",
    "ACCELERATE_USE_MEGATRON_LM",
    "ACCELERATE_USE_TP",
    "ACCELERATE_USE_PARALLELISM_CONFIG",
)


@contextmanager
def rank_zero_evaluation() -> Iterator[bool]:
    """Finish CLI TP execution and yield whether this rank should evaluate.

    Every rank must enter after export completes. This permanently destroys the
    process groups, so it must not be used by APIs that retain a live TP model.
    Both fake-model and checkpoint-reload evaluation then run without training
    collectives or Accelerate's cached multi-process state.

    Yields:
        True on rank zero, which may evaluate without distributed launcher state;
        False on other ranks, which must skip evaluation.
    """
    from accelerate.state import AcceleratorState

    rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0"))
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    is_distributed.cache_clear()
    AcceleratorState._reset_state(reset_partial_state=True)
    if rank != 0:
        yield False
        return

    saved_env = {key: os.environ.pop(key) for key in _DISTRIBUTED_ENV_VARS if key in os.environ}
    try:
        logger.info("TP process groups closed; running evaluation on rank 0 in single-process mode.")
        yield True
    finally:
        # Do not keep evaluation's singleton state across launcher environments.
        AcceleratorState._reset_state(reset_partial_state=True)
        is_distributed.cache_clear()
        for key in _DISTRIBUTED_ENV_VARS:
            os.environ.pop(key, None)
        os.environ.update(saved_env)
