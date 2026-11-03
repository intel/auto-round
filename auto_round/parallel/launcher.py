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

"""Local TP process orchestration, shared by CLI and task-oriented Python APIs."""

import os

import torch
import torch.distributed as dist

from auto_round.parallel.config import ParallelConfig, _reject_parallel_kwargs


def needs_local_launch(config):
    """Never recursively launch from torchrun or a caller-owned process group."""
    config = ParallelConfig.from_value(config)
    world = dist.get_world_size() if dist.is_initialized() else int(os.environ.get("WORLD_SIZE", "1"))
    if config.tp_size <= 1:
        return False
    if dist.is_initialized() or "RANK" in os.environ or world > 1:
        if world != config.tp_size:
            raise ValueError(f"TP requires WORLD_SIZE={config.tp_size}, got {world}.")
        return False
    return True


def _worker(function, args):
    """Groups created in our disposable workers belong to this launcher."""
    try:
        return function(*args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def launch(config, function, *args):
    """Run directly or start one local worker per TP shard; propagate failures.

    ``function`` must be a module-level callable and its arguments must support
    multiprocessing serialization. Existing distributed groups remain caller-owned.
    """
    config = ParallelConfig.from_value(config)
    if not needs_local_launch(config):
        return function(*args)
    if not torch.cuda.is_available() or torch.cuda.device_count() < config.tp_size:
        raise ValueError(f"tensor_parallel={config.tp_size} requires that many visible CUDA devices.")
    from torch.distributed.launcher.api import LaunchConfig, elastic_launch

    launch_config = LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=config.tp_size,
        rdzv_backend="c10d",
        rdzv_endpoint="localhost:0",
        max_restarts=0,
        start_method="spawn",
    )
    results = elastic_launch(launch_config, _worker)(function, args)
    return results[0]


def _quantize_task(model, output_dir, constructor_kwargs, save_kwargs):
    from auto_round import AutoRound

    compressor = AutoRound(model, **constructor_kwargs)
    _, folders = compressor.quantize_and_save(output_dir, **save_kwargs)
    return [folders] if isinstance(folders, str) else folders


def quantize_and_save(
    model,
    output_dir,
    *,
    parallel_config=None,
    save_kwargs=None,
    **kwargs,
):
    """Quantize a model path and return checkpoint folders, automatically launching TP.

    Call from an ``if __name__ == "__main__":`` guard when TP workers are started.
    Pass serialization-safe AutoRound constructor arguments in ``kwargs`` and save
    options in ``save_kwargs``. Preloaded models/tokenizers are intentionally excluded
    from this task API; use rank-local ``AutoRound`` under torchrun for those cases.
    This function does not return a distributed model to the parent process.
    """
    _reject_parallel_kwargs(kwargs)
    if not isinstance(model, str):
        raise TypeError("The task API requires a model name/path; use AutoRound for a preloaded model.")
    if kwargs.get("tokenizer") is not None:
        raise ValueError("The task API loads its tokenizer in each worker; use AutoRound for a preloaded tokenizer.")
    config = ParallelConfig.from_value(parallel_config)
    options = dict(save_kwargs or {})
    if "return_folders" in options:
        raise ValueError("The task API always returns checkpoint folders; do not set return_folders.")
    constructor_kwargs = dict(kwargs, parallel_config=config)
    return launch(config, _quantize_task, model, output_dir, constructor_kwargs, options)
