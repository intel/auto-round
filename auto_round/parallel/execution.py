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

"""Algorithm-facing services shared by local, naive pipeline-parallel and TP runs."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from auto_round.parallel.config import ParallelConfig
from auto_round.parallel.tp.runtime import is_dtensor
from auto_round.utils.device_manager import device_manager

if TYPE_CHECKING:
    from auto_round.algorithms.base import BaseAlgorithm
    from auto_round.algorithms.composer import AlgorithmComposer, BlockContext
    from auto_round.algorithms.config import AlgorithmConfig
    from auto_round.algorithms.quantization.base import BaseQuantizer
    from auto_round.compressors import BaseOrchestrator
    from auto_round.compressors.utils import IndexSampler
    from auto_round.parallel.dp.engine import ParallelPolicy
    from auto_round.parallel.dp.tune_parallel import TuneParallelContext
    from auto_round.parallel.prepared import PreparedRun
    from auto_round.parallel.tp.tensor_ops import ParallelTensorOps


@dataclass(frozen=True)
class ParallelCapabilities:
    """Declare supported execution modes for an algorithm.

    Attributes:
        tensor_parallel: Whether custom tensor math satisfies the TP contract.
        pipeline_placement: Whether naive pipeline placement is supported.
    """

    tensor_parallel: bool = False
    pipeline_placement: bool = True


@dataclass(frozen=True)
class BlockPlacement:
    """Describe a dispatched block and its loss-computation device.

    Attributes:
        block: Module after placement and any dispatch-hook installation.
        loss_device: Device on which to compute the block loss.
        primary_device_at_risk: Whether placement risks exhausting the primary device.
    """

    block: torch.nn.Module
    loss_device: str | torch.device
    primary_device_at_risk: bool = False


class ParallelExecution:
    """Place shards, normalize activations, and synchronize *data*-parallel grads.

    TP never gathers weights here. CPU residency retains DTensor placements and
    moves only each rank's local shard. PP is naive pipeline parallelism, with
    layers partitioned across devices but no overlapping microbatch schedule.
    Hybrid TP/PP remains unsupported.

    Attributes:
        config: Normalized parallel topology shared by execution services.
    """

    def __init__(self, config: ParallelConfig | Mapping[str, Any] | None = None) -> None:
        """Initialize services for a parallel topology.

        Args:
            config: A ParallelConfig, an option mapping, or None for local execution.
        """
        self.config = ParallelConfig.from_value(config)

    def prepare_run(
        self,
        model: torch.nn.Module,
        samples: Sequence[Any] | None = None,
        device: str | torch.device | None = None,
        *,
        whole_model: bool = False,
    ) -> PreparedRun:
        """Create a managed model scope; entering it prepares residency and data.

        The algorithm retains its own loop. After installing trainable wrappers,
        bind training once and use the scope's forward/next_batch/step methods.
        Cache coordination uses the model's actual mesh, not the global DP group.

        Args:
            model: Model or block to execute within the managed scope.
            samples: Optional calibration or training samples.
            device: Target device, defaulting to the current device manager's device.
            whole_model: Whether to restore whole-model residency on entry.

        Returns:
            A PreparedRun context manager owned by the caller.
        """
        from auto_round.parallel.prepared import PreparedRun

        return PreparedRun(self, model, samples, device, whole_model=whole_model)

    def for_algorithm(self, algorithm_class: type[BaseAlgorithm], config: AlgorithmConfig) -> ParallelExecution:
        """Select and validate an algorithm's execution policy before tuning.

        Policies can specialize placement/streaming and stage-local hooks, but
        cannot reinterpret shards loaded for a different process topology.

        Args:
            algorithm_class: Algorithm implementation selecting its execution policy.
            config: Algorithm configuration to validate against the policy.

        Returns:
            A validated execution service for the same loaded topology.

        Raises:
            TypeError: The algorithm does not return a ParallelExecution instance.
            NotImplementedError: The topology or algorithm options are unsupported.
        """
        execution = algorithm_class.create_parallel_execution(config, self)
        if not isinstance(execution, ParallelExecution):
            raise TypeError("create_parallel_execution() must return a ParallelExecution instance.")
        if execution.config != self.config:
            raise NotImplementedError(
                "Algorithm execution policies must retain the loaded parallel topology. "
                "Changing TP/PP degrees requires explicit model and state resharding."
            )
        execution.validate(algorithm_class, config)
        return execution

    @contextmanager
    def algorithm_stage(self, algorithm: BaseAlgorithm, model: torch.nn.Module) -> Iterator[None]:
        """Override to apply/remove an algorithm-specific policy with try/finally.

        Native Transformers forwards and DTensor layouts are transport, not
        stage-local hooks: leave them intact unless state is explicitly resharded.

        Args:
            algorithm: Algorithm whose stage-specific policy may be installed.
            model: Model used by the stage.

        Yields:
            None while the algorithm-specific policy is active.
        """
        yield

    @contextmanager
    def stage(self, algorithm: BaseAlgorithm, orchestrator: BaseOrchestrator) -> Iterator[None]:
        """Route shared infrastructure through this policy for one stage only.

        Args:
            algorithm: Active algorithm whose hooks run inside the scope.
            orchestrator: Compressor owning calibration, model, and runner services.

        Yields:
            None while stage-local execution services are installed.

        Raises:
            NotImplementedError: The stage combines incompatible parallel modes.
        """
        compress_context = getattr(orchestrator, "compress_context", None)
        self.validate_data_parallel_policy(getattr(compress_context, "parallel_policy", None))
        runner = getattr(algorithm, "block_forward", None)
        calibration = getattr(orchestrator, "calibration", None)
        with ExitStack() as cleanup:
            for owner, name in (
                (orchestrator, "parallel_execution"),
                (calibration, "parallel"),
                (runner, "parallel_execution"),
            ):
                if owner is not None and hasattr(owner, name):
                    cleanup.callback(setattr, owner, name, getattr(owner, name))
                    setattr(owner, name, self)
            if runner is not None:
                # Cached placement from the previous stage must not survive a
                # policy change, including when policy setup or tuning raises.
                runner.pool_placement = None
                cleanup.callback(setattr, runner, "pool_placement", None)
            with self.algorithm_stage(algorithm, orchestrator.model_context.model):
                yield

    @property
    def is_tp(self) -> bool:
        """bool: Whether this policy uses tensor parallelism."""
        return self.config.tp_size > 1

    @property
    def tensor_ops(self) -> type[ParallelTensorOps]:
        """type: Shard-aware tensor operations available to algorithms."""
        from auto_round.parallel.tp.tensor_ops import ParallelTensorOps

        return ParallelTensorOps

    @staticmethod
    def replicated(value: Any) -> Any:
        """Materialize replicated activations/statistics, preserving autograd.

        Args:
            value: A tensor, nested dict/list/tuple, or non-tensor value.

        Returns:
            The same structure with DTensors replaced by full tensors.
        """
        if is_dtensor(value):
            return value.full_tensor()
        if isinstance(value, dict):
            return {key: ParallelExecution.replicated(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(ParallelExecution.replicated(item) for item in value)
        if isinstance(value, list):
            return [ParallelExecution.replicated(item) for item in value]
        return value

    def validate(self, algorithm_class: type[BaseAlgorithm], config: AlgorithmConfig) -> None:
        """Check algorithm capabilities and options before parallel execution.

        Args:
            algorithm_class: Implementation declaring its parallel capabilities.
            config: Algorithm-specific configuration to validate.

        Raises:
            NotImplementedError: Capabilities, custom hooks, or options are unsupported.
        """
        capabilities = algorithm_class.parallel_capabilities()
        if self.is_tp and not capabilities.tensor_parallel:
            raise NotImplementedError(
                f"{algorithm_class.__name__} has not implemented the tensor-parallel contract. "
                "Implement parallel_capabilities() and use the shared parallel execution/tensor services."
            )
        if self.is_tp:
            custom_hooks = {
                "quantize_block",
                "quantize_model",
                "dispatch_block",
                "prepare_run",
                "finalize_run",
                "register_fp_input_forward_hooks",
                "register_qinput_forward_hooks",
                "pre_quantize_block",
                "post_quantize_block",
                "get_calibration_token_ids",
                "quantize_layer_outside_block",
                "quantize_embedding_layer",
                "finalize_block",
            }
            for implementation in algorithm_class.__mro__:
                if "parallel_capabilities" in implementation.__dict__:
                    break
                if custom_hooks.intersection(implementation.__dict__):
                    raise NotImplementedError(
                        f"{algorithm_class.__name__} overrides parallel-sensitive algorithm hooks. "
                        "Implement parallel_capabilities() explicitly after adapting those hooks for TP."
                    )
        if self.config.pp_size > 1 and not capabilities.pipeline_placement:
            raise NotImplementedError(f"{algorithm_class.__name__} does not support naive pipeline parallelism (PP).")
        if self.is_tp:
            unsupported = ["enable_norm_bias_tuning", "tune_non_quant_params"]
            if not getattr(algorithm_class, "whole_model_tuning", False):
                unsupported.extend(("enable_lfq", "enable_qad"))
            for name in unsupported:
                if getattr(config, name, False):
                    raise NotImplementedError(f"TP does not support {name} for {algorithm_class.__name__} yet.")
        algorithm_class.validate_parallel_config(config, self)

    def dispatch_block(
        self,
        block: torch.nn.Module,
        inputs: list[torch.Tensor] | dict[str, Any] | None,
        input_others: dict[str, Any],
        *,
        tuning: bool = False,
        is_diffusion: bool = False,
        low_gpu_mem_usage: bool = False,
        batch_size: int = 1,
    ) -> BlockPlacement:
        """Place a block and return a shared placement result for algorithms.

        Args:
            block: Module to place for collection or tuning.
            inputs: Cached inputs used to estimate automatic placement.
            input_others: Auxiliary inputs retained by the dispatch interface.
            tuning: Whether placement is for a tuning pass.
            is_diffusion: Whether the block belongs to a diffusion model.
            low_gpu_mem_usage: Whether automatic placement should conserve GPU memory.
            batch_size: Batch size used for placement estimates.

        Returns:
            A BlockPlacement containing the module, loss device, and memory-risk flag.
        """
        from auto_round.utils import is_auto_device_mapping
        from auto_round.utils.model import move_to_device_preserving_cpu_pinned, place_ngram_embeddings_for_tuning_

        if (
            not self.is_tp
            and tuning
            and is_auto_device_mapping(device_manager.device_map)
            and len(device_manager.device_list) > 1
            and not is_diffusion
        ):
            from auto_round.parallel.pp.placement import dispatch_auto_block

            return BlockPlacement(
                *dispatch_auto_block(
                    block,
                    inputs,
                    device_list=device_manager.device_list,
                    device=device_manager.device,
                    low_gpu_mem_usage=low_gpu_mem_usage,
                    batch_size=batch_size,
                )
            )
        if self.is_tp:
            block = self.move_block(block, device_manager.device)
        else:
            place_ngram_embeddings_for_tuning_(block)
            block = move_to_device_preserving_cpu_pinned(block, device_manager.device)
        return BlockPlacement(block, device_manager.device)

    @contextmanager
    def streamed_model(
        self, model: torch.nn.Module, block_names: Sequence[str | list[str] | tuple[str, ...]]
    ) -> Iterator[torch.nn.Module]:
        """Forward one TP block at a time; all other block shards stay on CPU.

        Args:
            model: Model whose block shards are streamed between CPU and accelerator.
            block_names: Block paths, optionally grouped in lists or tuples.

        Yields:
            The model with temporary streaming hooks, or unchanged for non-TP runs.
        """
        if not self.is_tp:
            yield model
            return
        names = {name for group in block_names for name in (group if isinstance(group, (list, tuple)) else [group])}
        blocks = [model.get_submodule(name) for name in sorted(names)]
        handles = []
        self.move_block(model, "cpu")
        try:
            # Move only each module's own tensors; recursive .to() on the root
            # would also move every inactive block. _apply preserves parameter
            # aliases (e.g. tied embeddings/output weights) like Module.to().
            for name, module in model.named_modules():
                if any(name == prefix or name.startswith(prefix + ".") for prefix in names):
                    continue
                module._apply(lambda value: value.to(device_manager.device), recurse=False)

            def activate(module: torch.nn.Module, args: tuple[Any, ...]) -> None:
                """Move a block onto the execution device before its forward.

                Args:
                    module: Block receiving the forward call.
                    args: Positional forward inputs supplied by the hook interface.
                """
                self.move_block(module, device_manager.device)

            def offload(module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> None:
                """Return a block to CPU after its forward, including failed calls.

                Args:
                    module: Block whose forward has completed.
                    args: Positional forward inputs supplied by the hook interface.
                    output: Forward result supplied by the hook interface.
                """
                self.move_block(module, "cpu")

            for block in blocks:
                handles.append(block.register_forward_pre_hook(activate))
                handles.append(block.register_forward_hook(offload, always_call=True))
            yield model
        finally:
            for handle in handles:
                handle.remove()
            self.move_block(model, "cpu")

    def prepare_gradients(
        self, algorithm: BaseAlgorithm, block: torch.nn.Module
    ) -> tuple[torch.nn.Module, Callable[[], None]]:
        """Prepare gradient synchronization without duplicating native TP work.

        Args:
            algorithm: Algorithm providing data-parallel preparation options.
            block: Module containing the trainable parameters.

        Returns:
            A tuple of the training module and a gradient-synchronization callback.
        """
        if self.is_tp:
            return block, lambda: None
        from auto_round.parallel.distributed import setup_ddp_if_needed_

        return setup_ddp_if_needed_(algorithm, block, device_manager.device_list)

    def collection_context(
        self,
        composer: AlgorithmComposer,
        block: torch.nn.Module,
        inputs: list[torch.Tensor] | dict[str, Any],
    ) -> TuneParallelContext:
        """Select collection transport without replicating native TP weight shards.

        Args:
            composer: Composer exposing the active quantizer and replica policy.
            block: Module whose calibration inputs are being collected.
            inputs: Cached inputs used to construct the collection context.

        Returns:
            A TuneParallelContext for local or replica-based collection.

        Raises:
            NotImplementedError: A TP run also requests data-parallel replicas.
        """
        from auto_round.parallel.dp.tune_parallel import TuneParallelContext

        if self.is_tp:
            context = getattr(composer.active_quantizer, "compress_context", None)
            self.validate_data_parallel_policy(getattr(context, "parallel_policy", None))
            return TuneParallelContext()
        return TuneParallelContext.for_collection(composer, block, inputs)

    def quantize_block(
        self,
        quantizer: BaseQuantizer,
        block: torch.nn.Module,
        fp_inputs: list[torch.Tensor] | dict[str, Any],
        input_others: dict[str, Any],
        fp_outputs: list[torch.Tensor],
        q_inputs: list[torch.Tensor] | None,
        block_ctx: BlockContext,
        input_ids: list[torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Run the same algorithm on local/TP shards or on data-parallel replicas.

        Args:
            quantizer: Algorithm implementing block quantization.
            block: Module to quantize.
            fp_inputs: Full-precision reference inputs.
            input_others: Auxiliary forward arguments.
            fp_outputs: Full-precision reference outputs.
            q_inputs: Inputs produced by previously quantized blocks.
            block_ctx: Block metadata and compression context.
            input_ids: Optional token IDs associated with the inputs.

        Returns:
            The result produced by the selected block-quantization path.
        """
        if not self.is_tp:
            from auto_round.parallel.dp.spmd import quantize_block_on_ranks

            engaged, result = quantize_block_on_ranks(
                quantizer, block, fp_inputs, input_others, fp_outputs, q_inputs, block_ctx, input_ids=input_ids
            )
            if engaged:
                return result
        return quantizer.quantize_block(
            block, fp_inputs, input_others, fp_outputs, q_inputs, block_ctx, input_ids=input_ids
        )

    def validate_data_parallel_policy(self, policy: ParallelPolicy | None) -> None:
        """Reject mixing thread-based DP mirrors with native TP or explicit PP.

        Args:
            policy: Replica policy to validate, or None when replicas are disabled.

        Raises:
            NotImplementedError: Multiple replicas are combined with TP or PP.
        """
        if policy is not None and policy.world > 1 and (self.is_tp or self.config.pp_size > 1):
            raise NotImplementedError(
                "--data_parallel cannot be combined with TP/PP; hybrid execution is not supported."
            )

    def index_sampler(self, nsamples: int, batch_size: int) -> IndexSampler:
        """Create a sampler with DP sharding and TP-global batch semantics.

        Args:
            nsamples: Number of available calibration samples.
            batch_size: Number of samples requested per batch.

        Returns:
            An index sampler for the current distributed execution mode.
        """
        from auto_round.parallel.distributed import make_index_sampler

        return make_index_sampler(nsamples, batch_size)

    def reduce_loss(self, loss: float, weight: float = 1.0) -> float:
        """Reduce independently computed DP losses without reducing TP losses twice.

        Args:
            loss: Local loss to reduce.
            weight: Contribution weight used by data-parallel reduction.

        Returns:
            The unchanged TP loss or the data-parallel reduced loss.
        """
        if self.is_tp:
            return loss
        from auto_round.parallel.distributed import all_reduce_loss

        return all_reduce_loss(loss, weight)

    def sample_indices(self, sampler: IndexSampler) -> list[int]:
        """Select identical examples on TP ranks despite rank-local RNG use.

        Args:
            sampler: Sampler exposing a ``next_batch`` method.

        Returns:
            Sample indices drawn locally or broadcast from TP rank zero.
        """
        if not self.is_tp:
            return sampler.next_batch()
        import torch.distributed as dist

        indices = [sampler.next_batch() if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(indices, src=0)
        return indices[0]

    def prepare_blockwise_model(self, model: torch.nn.Module) -> None:
        """Keep inactive TP shards in host memory, including zero-shot runs.

        Args:
            model: Model to offload when tensor parallelism is enabled.
        """
        if self.is_tp:
            self.move_block(model, "cpu")

    def prepare_whole_model(self, model: torch.nn.Module, device: str | torch.device) -> torch.nn.Module:
        """Restore all local shards and metadata before whole-model forwards.

        Block tuning/calibration may leave embeddings and inactive blocks on
        CPU. Module.to() preserves native DTensor layouts and forwards; moving
        a model here must not depend on a teacher-cache miss or calibration.

        Args:
            model: Model to prepare for whole-model execution.
            device: Target device for local shards and quantization metadata.

        Returns:
            The model after moving its tensors and metadata.
        """
        return self.move_block(model, device)

    @staticmethod
    def move_block(block: torch.nn.Module, device: str | torch.device) -> torch.nn.Module:
        """Move registered shards and unregistered quantization tensor metadata.

        Args:
            block: Module to move in place.
            device: Target device accepted by PyTorch tensor conversion.

        Returns:
            The input module on the requested device.
        """
        block.to(device)
        for module in block.modules():
            for name in (
                "scale",
                "zp",
                "weight_global_scale",
                "imatrix",
                "act_max",
                "act_scale",
                "act_min_scale",
                "act_max_scale",
                "input_global_scale",
            ):
                if name in module._parameters or name in module._buffers:
                    continue
                value = getattr(module, name, None)
                if isinstance(value, torch.Tensor):
                    setattr(module, name, value.to(device))
        return block

    def offload_block(self, block: torch.nn.Module) -> torch.nn.Module:
        """Release accelerator residency using the topology-appropriate path.

        Args:
            block: Module to offload.

        Returns:
            The module after TP CPU placement or ordinary device offloading.
        """
        if self.is_tp:
            return self.move_block(block, "cpu")
        from auto_round.utils import mv_module_from_gpu

        return mv_module_from_gpu(block)
