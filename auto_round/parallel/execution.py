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

"""Algorithm-facing execution services shared by local, PP-placement and TP runs."""

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from auto_round.parallel.config import ParallelConfig
from auto_round.parallel.runtime import is_dtensor
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
    moves only each rank's local shard. PP means the existing multi-device block
    placement, not a pipeline training schedule. Hybrid TP/PP remains unsupported.
    """

    def __init__(self, config=None):
        self.config = ParallelConfig.from_value(config)

    @property
    def is_tp(self):
        return self.config.tp_size > 1

    @property
    def tensor_ops(self):
        from auto_round.parallel.tensor_ops import ParallelTensorOps

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
            raise NotImplementedError(f"{algorithm_class.__name__} does not support PP (pipeline device placement).")
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
            from accelerate.hooks import AlignDevicesHook, add_hook_to_module

            from auto_round.utils.device import set_auto_device_map_for_block_with_tuning

            at_risk, loss_device = set_auto_device_map_for_block_with_tuning(
                block,
                device_manager.device_list,
                inputs,
                low_gpu_mem_usage,
                batch_size,
                device_manager.device,
            )
            for _, module in block.named_modules():
                if not list(module.children()) and hasattr(module, "tuning_device"):
                    add_hook_to_module(module, AlignDevicesHook(module.tuning_device, io_same_device=True), True)
            return BlockPlacement(block, loss_device, at_risk)
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
        from auto_round.utils.distributed import setup_ddp_if_needed_

        return setup_ddp_if_needed_(algorithm, block, device_manager.device_list)

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

    @staticmethod
    def move_block(block, device):
        """Move registered shards and unregistered quantization tensor metadata."""
        block.to(device)
        for module in block.modules():
            for name in ("scale", "zp", "weight_global_scale", "imatrix", "act_max"):
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
