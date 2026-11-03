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

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

import torch

from auto_round.parallel.config import ParallelConfig
from auto_round.parallel.tp.runtime import is_dtensor
from auto_round.utils.device_manager import device_manager


@dataclass(frozen=True)
class ParallelCapabilities:
    """Explicit algorithm contract; custom tensor math is not TP-safe by default."""

    tensor_parallel: bool = False
    pipeline_placement: bool = True


@dataclass(frozen=True)
class BlockPlacement:
    block: torch.nn.Module
    loss_device: str | torch.device
    primary_device_at_risk: bool = False


class ParallelExecution:
    """Place shards, normalize activations, and synchronize *data*-parallel grads.

    TP never gathers weights here. CPU residency retains DTensor placements and
    moves only each rank's local shard. PP is naive pipeline parallelism, with
    layers partitioned across devices but no overlapping microbatch schedule.
    Hybrid TP/PP remains unsupported.
    """

    def __init__(self, config=None):
        self.config = ParallelConfig.from_value(config)

    def prepare_run(self, model, samples=None, device=None, *, whole_model=False):
        """Create a managed model scope; entering it prepares residency and data.

        The algorithm retains its own loop. After installing trainable wrappers,
        bind training once and use the scope's forward/next_batch/step methods.
        Cache coordination uses the model's actual mesh, not the global DP group.
        """
        from auto_round.parallel.prepared import PreparedRun

        return PreparedRun(self, model, samples, device, whole_model=whole_model)

    def for_algorithm(self, algorithm_class, config):
        """Select and validate an algorithm's execution policy before tuning.

        Policies can specialize placement/streaming and stage-local hooks, but
        cannot reinterpret shards loaded for a different process topology.
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
    def algorithm_stage(self, algorithm, model):
        """Override to apply/remove an algorithm-specific policy with try/finally.

        Native Transformers forwards and DTensor layouts are transport, not
        stage-local hooks: leave them intact unless state is explicitly resharded.
        """
        yield

    @contextmanager
    def stage(self, algorithm, orchestrator):
        """Route shared infrastructure through this policy for one stage only."""
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
    def is_tp(self):
        return self.config.tp_size > 1

    def log_memory_summary(self, msg="", level="info"):
        """Report memory at a successful boundary reached by every model-parallel rank."""
        if self.is_tp:
            from auto_round.parallel.tp.runtime import log_memory_summary

            return log_memory_summary(msg, level)
        from auto_round.utils.device import memory_monitor

        return memory_monitor.log_summary(msg, level)

    @property
    def tensor_ops(self):
        from auto_round.parallel.tp.tensor_ops import ParallelTensorOps

        return ParallelTensorOps

    @staticmethod
    def replicated(value):
        """Materialize replicated activations/statistics, preserving autograd."""
        if is_dtensor(value):
            return value.full_tensor()
        if isinstance(value, dict):
            return {key: ParallelExecution.replicated(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(ParallelExecution.replicated(item) for item in value)
        if isinstance(value, list):
            return [ParallelExecution.replicated(item) for item in value]
        return value

    def validate(self, algorithm_class, config):
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
        self, block, inputs, input_others, *, tuning=False, is_diffusion=False, low_gpu_mem_usage=False, batch_size=1
    ):
        """Single placement entry point; algorithms receive a placement result."""
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
    def streamed_model(self, model, block_names):
        """Forward one TP block at a time; all other block shards stay on CPU."""
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

            def activate(module, args):
                self.move_block(module, device_manager.device)

            def offload(module, args, output):
                self.move_block(module, "cpu")

            for block in blocks:
                handles.append(block.register_forward_pre_hook(activate))
                handles.append(block.register_forward_hook(offload, always_call=True))
            yield model
        finally:
            for handle in handles:
                handle.remove()
            self.move_block(model, "cpu")

    def prepare_gradients(self, algorithm, block):
        """TP autograd already communicates; only DP uses gradient all-reduce."""
        if self.is_tp:
            return block, lambda: None
        from auto_round.parallel.distributed import setup_ddp_if_needed_

        return setup_ddp_if_needed_(algorithm, block, device_manager.device_list)

    def collection_context(self, composer, block, inputs):
        """Select collection transport without replicating native TP weight shards."""
        from auto_round.parallel.dp.tune_parallel import TuneParallelContext

        if self.is_tp:
            context = getattr(composer.block_quantizer, "compress_context", None)
            self.validate_data_parallel_policy(getattr(context, "parallel_policy", None))
            return TuneParallelContext()
        return TuneParallelContext.for_collection(composer, block, inputs)

    def quantize_block(
        self, quantizer, block, fp_inputs, input_others, fp_outputs, q_inputs, block_ctx, input_ids=None
    ):
        """Run the same algorithm on local/TP shards or on data-parallel replicas."""
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

    def validate_data_parallel_policy(self, policy):
        """Thread-based DP mirrors and native TP/explicit PP are separate modes."""
        if policy is not None and policy.world > 1 and (self.is_tp or self.config.pp_size > 1):
            raise NotImplementedError(
                "--data_parallel cannot be combined with TP/PP; hybrid execution is not supported."
            )

    def index_sampler(self, nsamples, batch_size):
        """DP ranks draw disjoint shards; TP ranks draw identical global batches."""
        from auto_round.parallel.distributed import make_index_sampler

        return make_index_sampler(nsamples, batch_size)

    def reduce_loss(self, loss, weight=1.0):
        """Only data-parallel replicas reduce independently computed losses."""
        if self.is_tp:
            return loss
        from auto_round.parallel.distributed import all_reduce_loss

        return all_reduce_loss(loss, weight)

    def sample_indices(self, sampler):
        """TP ranks optimize identical examples even after rank-local RNG use."""
        if not self.is_tp:
            return sampler.next_batch()
        import torch.distributed as dist

        indices = [sampler.next_batch() if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(indices, src=0)
        return indices[0]

    def prepare_blockwise_model(self, model):
        """Keep inactive TP shards in host memory, including zero-shot runs."""
        if self.is_tp:
            self.move_block(model, "cpu")

    def prepare_whole_model(self, model, device):
        """Restore all local shards and metadata before whole-model forwards.

        Block tuning/calibration may leave embeddings and inactive blocks on
        CPU. Module.to() preserves native DTensor layouts and forwards; moving
        a model here must not depend on a teacher-cache miss or calibration.
        """
        return self.move_block(model, device)

    @staticmethod
    def move_block(block, device):
        """Move registered shards and unregistered quantization tensor metadata."""
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

    def offload_block(self, block):
        if self.is_tp:
            return self.move_block(block, "cpu")
        from auto_round.utils import mv_module_from_gpu

        return mv_module_from_gpu(block)
