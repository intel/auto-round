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

"""Managed execution boundaries for algorithms with their own tuning loops."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self, TypeVar

import torch
import torch.distributed as dist

from auto_round.parallel.tp import runtime
from auto_round.utils.device_manager import device_manager

if TYPE_CHECKING:
    from auto_round.algorithms.base import BaseAlgorithm
    from auto_round.compressors.utils import IndexSampler
    from auto_round.parallel.execution import ParallelExecution

_ResultT = TypeVar("_ResultT")


class RunCache:
    """Rank-local storage with collective cache-miss decisions for model forwards.

    All model-parallel participants must call ``ensure`` in the same order.
    The build callback runs on every participant when *any* participant misses;
    it must not contain rank-local branches around collective model forwards.
    Data-parallel replicas have independent caches and independent decisions.

    Attributes:
        run: Execution scope that owns the cache and its collective group.
    """

    def __init__(self, run: PreparedRun) -> None:
        """Bind cache operations to an execution scope.

        Args:
            run: PreparedRun providing model state, device, and process-group ownership.
        """
        self.run = run

    def workspace(self, path: str | os.PathLike[str]) -> str:
        """Keep independently owned cache writers out of the same directory.

        Args:
            path: Base cache directory, optionally implementing the path protocol.

        Returns:
            The created rank-local directory as a string.

        Raises:
            RuntimeError: The owning run has already exited.
        """
        self.run._check_open()
        if self.run._tensor_parallel:
            path = os.path.join(path, f"tp_rank_{dist.get_rank()}")
        elif dist.is_initialized() and dist.get_world_size() > 1:
            path = os.path.join(path, f"dp_rank_{dist.get_rank()}")
        os.makedirs(path, exist_ok=True)
        return os.fspath(path)

    def ensure(self, valid: bool, build: Callable[[], None]) -> bool:
        """Build on a coordinated miss; return whether the callback was run.

        ``valid`` describes local storage only. Never branch on it before calling
        this method: other ranks may still need a collective model forward.

        Args:
            valid: Whether this rank's local cache is usable.
            build: No-argument callback that rebuilds this rank's cache on a miss.

        Returns:
            True if the build callback ran, otherwise False.

        Raises:
            RuntimeError: The owning run has already exited.
        """
        self.run._check_open()
        if self.run._tensor_parallel:
            hit = torch.tensor(int(valid), device=self.run.device)
            dist.all_reduce(hit, op=dist.ReduceOp.MIN, group=self.run._group)
            valid = bool(hit.item())
        if valid:
            return False
        build()
        return True

    def signature(self, samples: Sequence[Any] | None) -> dict[str, Any]:
        """Fingerprint model state and inputs without gathering full weights.

        Include every shard's digest so changing a remote shard invalidates all
        participants' caches, even if a participant's own tensors are unchanged.

        Args:
            samples: Input samples to fingerprint, or None when no data is available.

        Returns:
            A dictionary of model metadata, weight/input digests, and sample dimensions.

        Raises:
            RuntimeError: The owning run has already exited.
        """
        self.run._check_open()
        model = self.run.model
        cfg = getattr(model, "config", None)

        def update_tensor(digest: Any, name: str, tensor: torch.Tensor) -> None:
            """Update a digest from tensor metadata and bounded CPU byte chunks.

            Args:
                digest: Hash object to update in place.
                name: Tensor name included in the fingerprint.
                tensor: Local or distributed tensor whose local storage is hashed.
            """
            tensor = tensor.detach()
            if runtime.is_dtensor(tensor):
                digest.update(f"{tensor.shape}:{tensor.placements}".encode())
                tensor = tensor.to_local()
            digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
            flat = tensor.reshape(-1)
            chunk_size = max(1, (16 * 1024 * 1024) // tensor.element_size())
            for start in range(0, flat.numel(), chunk_size):
                chunk = flat[start : start + chunk_size].cpu().contiguous().view(torch.uint8)
                digest.update(chunk.numpy().tobytes())

        weights = hashlib.sha256()
        for name, tensor in model.state_dict().items():
            update_tensor(weights, name, tensor)
        weights_hash = weights.hexdigest()
        if self.run._tensor_parallel:
            shards = [None] * dist.get_world_size(self.run._group)
            dist.all_gather_object(shards, weights_hash, group=self.run._group)
            weights_hash = hashlib.sha256(json.dumps(shards).encode()).hexdigest()
        data_hash = hashlib.sha256()
        for index, sample in enumerate(samples if samples is not None else []):
            fields = sample if isinstance(sample, dict) else {"labels": sample}
            for key, tensor in sorted(fields.items()):
                update_tensor(data_hash, f"{index}:{key}", torch.as_tensor(tensor))
        seqlen = None
        if samples is not None and len(samples):
            first = samples[0]
            if isinstance(first, dict):
                first = first["input_ids"]
            if torch.is_tensor(first):
                seqlen = int(first.reshape(-1).shape[0])
            else:
                try:
                    seqlen = len(first)
                except TypeError:
                    pass
        return {
            "model_type": getattr(cfg, "model_type", None),
            "vocab_size": getattr(cfg, "vocab_size", None),
            "name_or_path": getattr(cfg, "_name_or_path", None),
            "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "config": json.dumps(cfg.to_dict(), sort_keys=True, default=str) if hasattr(cfg, "to_dict") else str(cfg),
            "weights_sha256": weights_hash,
            "inputs_sha256": data_hash.hexdigest(),
            "nsamples": None if samples is None else len(samples),
            "seqlen": seqlen,
        }


class PreparedRun:
    """A model execution scope, not a training-loop implementation.

    Enter before calibration/teacher construction. Install trainable wrappers and
    configure checkpointing, then call ``prepare_training`` before the first
    student forward. Use ``forward``, ``next_batch`` and ``step`` in a custom
    loop. Loss math, optimizer choice and best-parameter snapshots stay with the
    algorithm; no arbitrary tensor operation is automatically made TP-safe.

    Whole-model execution supports local/native TP, not pipeline scheduling.
    Block scopes retain the composer's existing placement. The caller owns model
    residency after exit; this scope only releases its temporary training wrapper.

    Attributes:
        execution: Parallel execution service managing placement and synchronization.
        model: Original model or block executed by the scope.
        samples: Samples synchronized across the TP mesh on entry.
        device: Execution device for local model tensors.
        whole_model: Whether entry restores residency for the complete model.
        cache: Rank-aware cache operations bound to this scope.
    """

    def __init__(
        self,
        execution: ParallelExecution,
        model: torch.nn.Module,
        samples: Sequence[Any] | None = None,
        device: str | torch.device | None = None,
        *,
        whole_model: bool = False,
    ) -> None:
        """Initialize an execution scope without entering or moving the model.

        Args:
            execution: Parallel execution service for the loaded topology.
            model: Model or block to execute.
            samples: Optional input samples shared by TP participants on entry.
            device: Target device, defaulting to the current managed device.
            whole_model: Whether to restore whole-model residency on entry.

        Raises:
            NotImplementedError: Parameters use multiple or non-one-dimensional TP meshes.
            ValueError: A TP execution service receives a model without DTensor parameters.
        """
        self.execution = execution
        self.model = model
        self.samples = samples
        self.device = device if device is not None else device_manager.device
        self.whole_model = whole_model
        self._tensor_parallel = runtime.is_tp_model(model)
        self._group = None
        self._source = 0
        mesh = None
        for parameter in model.parameters():
            if runtime.is_dtensor(parameter):
                candidate = parameter.device_mesh
                if candidate.ndim != 1 or (mesh is not None and candidate != mesh):
                    raise NotImplementedError("PreparedRun requires a single one-dimensional TP mesh.")
                mesh = candidate
        if mesh is not None:
            self._group = mesh.get_group()
            self._source = int(mesh.mesh.reshape(-1)[0])
        if execution.is_tp and not self._tensor_parallel:
            raise ValueError("A TP execution scope requires a model with native DTensor parameters.")
        self.cache = RunCache(self)
        self._training_model = model
        self._sync_gradients: Callable[[], None] | None = None
        self._entered = False
        self._closed = False

    def _check_open(self) -> None:
        """Reject operations after the scope has exited.

        Raises:
            RuntimeError: The scope has already exited.
        """
        if self._closed:
            raise RuntimeError("PreparedRun has already exited.")

    def __enter__(self) -> Self:
        """Synchronize samples and prepare model residency for execution.

        Returns:
            This execution scope, ready for calibration and training preparation.

        Raises:
            RuntimeError: The scope was already entered or closed.
            NotImplementedError: Whole-model execution requests pipeline placement.
        """
        self._check_open()
        if self._entered:
            raise RuntimeError("PreparedRun cannot be entered twice.")
        if self.whole_model and self.execution.config.pp_size > 1:
            raise NotImplementedError("Whole-model PreparedRun does not support naive pipeline parallelism (PP).")
        if self._tensor_parallel:
            samples = [self.samples if dist.get_rank() == self._source else None]
            dist.broadcast_object_list(samples, src=self._source, group=self._group)
            self.samples = samples[0]
        if self.whole_model:
            if not self._tensor_parallel:
                self._strip_dispatch_hooks(self.model)
            self.execution.prepare_whole_model(self.model, self.device)
            for module in self.model.modules():
                if hasattr(module, "tuning_device"):
                    del module.tuning_device
        self._entered = True
        return self

    @staticmethod
    def _strip_dispatch_hooks(model: torch.nn.Module) -> None:
        """Remove stale Accelerate placement hooks before whole-model execution.

        Args:
            model: Model whose existing device map may have installed dispatch hooks.
        """
        if getattr(model, "hf_device_map", None):
            from accelerate.hooks import remove_hook_from_submodules

            remove_hook_from_submodules(model)

    def prepare_training(self, algorithm: BaseAlgorithm) -> None:
        """Bind once, AFTER installing all trainable parameters and placement.

        TP retains native autograd; legacy DP uses the existing preparation
        policy. In particular, do not DDP-wrap before creating quantizer wrappers.

        Args:
            algorithm: Algorithm providing the gradient-synchronization policy.

        Raises:
            RuntimeError: The scope is closed, not entered, or already bound for training.
        """
        self._check_open()
        if not self._entered or self._sync_gradients is not None:
            raise RuntimeError("Call prepare_training once inside an entered PreparedRun.")
        if self._tensor_parallel:
            self._sync_gradients = lambda: None
        else:
            self._training_model, self._sync_gradients = self.execution.prepare_gradients(algorithm, self.model)

    def forward(self, *args: Any, output: str | None = None, **kwargs: Any) -> Any:
        """Return replicated outputs with autograd intact, never gather weights.

        Select an attribute/key (e.g. ``output='logits'``) before normalization to
        avoid materializing unused model outputs. The default suits block tensors.
        DTensor outputs carry layout metadata; ordinary tensors are assumed to be
        complete already. Unlabelled local-shard outputs need an explicit adapter.

        Args:
            *args: Positional arguments forwarded to the training model.
            output: Optional output key or attribute to select before replication.
            **kwargs: Keyword arguments forwarded to the training model.

        Returns:
            The selected output with DTensors materialized and autograd preserved.

        Raises:
            RuntimeError: The scope has already exited.
        """
        self._check_open()
        result = self._training_model(*args, **kwargs)
        if output is not None:
            result = result[output] if isinstance(result, dict) else getattr(result, output)
        return self.execution.replicated(result)

    def next_batch(self, sampler: IndexSampler) -> list[int]:
        """Sample once per TP mesh; DP retains its existing sampler semantics.

        Args:
            sampler: Sampler exposing a ``next_batch`` method.

        Returns:
            Local sample indices, or indices broadcast from the TP mesh source.

        Raises:
            RuntimeError: The scope has already exited.
        """
        self._check_open()
        if not self._tensor_parallel:
            return sampler.next_batch()
        indices = [sampler.next_batch() if dist.get_rank() == self._source else None]
        dist.broadcast_object_list(indices, src=self._source, group=self._group)
        return indices[0]

    def step(self, update: Callable[..., _ResultT], *args: Any, **kwargs: Any) -> _ResultT:
        """Synchronize accumulated grads, then run the algorithm's update callback.

        The callback may keep existing scaler/scheduler logic. Copy gradients to
        FP optimizer masters *inside* the callback, after synchronization. Closure
        optimizers that perform backward inside step require a separate policy.

        Args:
            update: Callback performing optimizer, scaler, and scheduler updates.
            *args: Positional arguments passed to the update callback.
            **kwargs: Keyword arguments passed to the update callback.

        Returns:
            The value returned by the update callback.

        Raises:
            RuntimeError: The scope is closed or training has not been prepared.
        """
        self._check_open()
        if self._sync_gradients is None:
            raise RuntimeError("Call prepare_training before step.")
        self._sync_gradients()
        return update(*args, **kwargs)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release temporary training state without suppressing exceptions.

        Args:
            exc_type: Exception type raised inside the scope, or None.
            exc_value: Exception raised inside the scope, or None.
            traceback: Traceback associated with the exception, or None.
        """
        self._training_model = self.model
        self._sync_gradients = None
        self._closed = True
