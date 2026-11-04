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

"""Small runtime operations shared by distributed model consumers."""

import copy
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist


def is_dtensor(value: object) -> bool:
    """Check for a DTensor without requiring distributed tensor support.

    Args:
        value: Object to inspect.

    Returns:
        True if DTensor is available and the value is a DTensor instance.
    """
    # Keep DTensor optional for non-distributed installations.
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return False
    return isinstance(value, DTensor)


def is_tp_model(model: torch.nn.Module) -> bool:
    """Check whether a model contains native distributed parameters.

    Args:
        model: Module whose parameters are inspected recursively.

    Returns:
        True if at least one parameter is a DTensor.
    """
    return any(is_dtensor(parameter) for parameter in model.parameters())


def replicated_tensor(value: Any) -> Any:
    """Gather a distributed result while preserving its autograd connection.

    Args:
        value: Distributed tensor or value to leave unchanged.

    Returns:
        A full tensor for DTensor input, otherwise the original value.
    """
    return value.full_tensor() if is_dtensor(value) else value


@contextmanager
def _export_step() -> Iterator[None]:
    """Agree on local work before any rank enters the next export collective.

    Only local operations belong here, never a tensor collective. Transport
    failures propagate directly without attempting a different collective.

    Yields:
        None while ranks perform local export preparation.

    Raises:
        RuntimeError: Any rank reports a failure in the local export step.
    """
    failure = None
    try:
        yield
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    failures = [None] * dist.get_world_size()
    dist.all_gather_object(failures, failure)
    for rank, error in enumerate(failures):
        if error is not None:
            raise RuntimeError(f"TP checkpoint gathering failed on rank {rank}: {error}")


@contextmanager
def synchronized_export() -> Iterator[None]:
    """Share writer-only completion/failure, strictly after checkpoint gathering.

    Every rank must enter; only rank zero should perform checkpoint writing.
    The writer's original exception propagates unchanged on rank zero.

    Yields:
        None while the writer saves the gathered checkpoint.

    Raises:
        RuntimeError: Rank zero reports a writer failure to a non-writer rank.
    """
    failure = None
    try:
        yield
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        status = [failure]
        dist.broadcast_object_list(status, src=0)
        if dist.get_rank() != 0 and status[0] is not None:
            raise RuntimeError(f"TP checkpoint writer failed: {status[0]}")


def validate_tp_model(model: torch.nn.Module, tp_size: int) -> None:
    """Reject unsupported plans before calibration or wrapper mutation.

    Args:
        model: Model loaded with native tensor-parallel parameters.
        tp_size: Required process-group and one-dimensional mesh size.

    Raises:
        ValueError: The process group is incompatible or no DTensor parameters exist.
        NotImplementedError: A mesh, expert layout, or native TP style is unsupported.
    """
    if not dist.is_initialized() or dist.get_world_size() != tp_size:
        raise ValueError("The initialized process group must match parallel_config.tp_size.")
    if not is_tp_model(model):
        raise ValueError("Native TP did not produce DTensor parameters; check the model's base_model_tp_plan.")
    for name, parameter in model.named_parameters():
        if is_dtensor(parameter) and (parameter.device_mesh.ndim != 1 or parameter.device_mesh.size() != tp_size):
            raise NotImplementedError(f"Unsupported TP mesh for parameter {name!r}.")
    for name, module in model.named_modules():
        if "experts" in name.split(".") and isinstance(module, torch.nn.Linear):
            if not is_dtensor(module.weight) or not any(p.is_shard() for p in module.weight.placements):
                raise NotImplementedError(
                    f"Expert Linear {name!r} must be natively sharded; reload without legacy MoE monkey patches."
                )
    # Packed styles are permitted only at the standard MoE boundary. The MoE
    # adapter validates actual tensor placements before changing the structure.
    plan = getattr(model, "tp_plan", None) or getattr(model, "_tp_plan", {})
    supported = {
        "colwise",
        "rowwise",
        "colwise_rep",
        "rowwise_rep",
        "colwise_gather_output",
        "rowwise_split_input",
        "embedding_rowwise",
        "embedding_colwise",
        "replicated_with_grad_allreduce",
        "replicate",
    }
    unsupported = {
        style
        for name, style in plan.items()
        if style not in supported
        and not (
            (style == "moe_tp_experts" and name.endswith(".experts"))
            or (style == "packed_colwise" and name.endswith(".experts.gate_up_proj"))
        )
    }
    if unsupported:
        raise NotImplementedError(f"Unsupported native TP styles for QAD: {sorted(unsupported)}.")


@torch.no_grad()
def _gather_cpu_tensor(value: torch.Tensor, chunk_bytes: int = 4 * 1024 * 1024) -> torch.Tensor | None:
    """Gather a tensor on a 1-D TP mesh to writer CPU memory with bounded staging.

    Broadcast byte chunks so NCCL does not require a separate Gloo group and
    communication does not depend on support for the tensor's storage dtype.
    No rank ever materializes a full distributed tensor on its accelerator.

    Args:
        value: Local tensor or DTensor to gather; every rank must participate.
        chunk_bytes: Maximum byte count staged for each broadcast.

    Returns:
        The complete detached CPU tensor on rank zero, or None on other ranks.

    Raises:
        RuntimeError: A rank reports unsupported placement or a local export failure.
    """
    from torch.distributed.tensor import Replicate, Shard

    writer = dist.get_rank() == 0
    distributed = is_dtensor(value)
    with _export_step():
        if not distributed:
            local = value.detach().cpu() if writer else None
        else:
            if value.device_mesh.ndim != 1 or len(value.placements) != 1:
                raise NotImplementedError("CPU export requires a one-dimensional TP mesh.")
            placement = value.placements[0]
            if not isinstance(placement, (Replicate, Shard)):
                raise NotImplementedError("CPU export supports only replicated or sharded tensors.")
            local = value.to_local().detach().cpu().contiguous()
    if not distributed:
        return local
    if isinstance(placement, Replicate):
        return local if writer else None
    group = value.device_mesh.get_group()
    size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    axis = placement.dim % value.ndim
    width = (value.shape[axis] + size - 1) // size
    with _export_step():
        output = torch.empty(value.shape, dtype=value.dtype, device="cpu") if writer else None
    # Gloo stages on CPU; NCCL stages at most chunk_bytes on the local GPU.
    device = (
        torch.device("cuda", torch.cuda.current_device()) if dist.get_backend(group) == "nccl" else torch.device("cpu")
    )
    for source in range(size):
        shape = list(value.shape)
        start = min(source * width, value.shape[axis])
        shape[axis] = min(width, value.shape[axis] - start)
        with _export_step():
            shard = local if source == rank else (torch.empty(shape, dtype=value.dtype) if writer else None)
        byte_count = value.element_size()
        for dimension in shape:
            byte_count *= dimension
        for offset in range(0, byte_count, chunk_bytes):
            count = min(chunk_bytes, byte_count - offset)
            with _export_step():
                staging = torch.empty(count, dtype=torch.uint8, device=device)
                if source == rank:
                    staging.copy_(local.reshape(-1).view(torch.uint8)[offset : offset + count])
            dist.broadcast(staging, src=dist.get_global_rank(group, source), group=group)
            with _export_step():
                if writer:
                    shard.reshape(-1).view(torch.uint8)[offset : offset + count].copy_(staging.cpu())
        with _export_step():
            if writer:
                output.narrow(axis, start, shape[axis]).copy_(shard)
    return output


def gather_for_export(model: torch.nn.Module) -> torch.nn.Module | None:
    """Return a non-TP CPU model on rank zero; all TP ranks must participate.

    Build a clean model instead of trying to undo Transformers' private forward
    closures. The caller retains its original TP model for further inference.
    Shards are copied to CPU before bounded transfers; full tensors exist only
    in writer host memory, never temporarily on a GPU.

    Args:
        model: Native TP model containing quantized weights and optional wrappers.

    Returns:
        A standalone CPU evaluation model on rank zero, or None on other ranks.

    Raises:
        RuntimeError: A rank fails during local gathering or model reconstruction.
    """
    from auto_round.parallel.tp.activation import TensorParallelActivationQuantizer
    from auto_round.wrapper import WrapperWALayer

    writer = dist.get_rank() == 0
    with _export_step():
        activation_wrappers = {name for name, module in model.named_modules() if isinstance(module, WrapperWALayer)}
        moe_names = [name for name, module in model.named_modules() if getattr(module, "_auto_round_tp_moe", False)]
        aliases = {f"{name}.{key}" for name in moe_names for key in ("gate_up_proj", "down_proj")}
        state_items = list(model.state_dict().items())
        buffer_items = list(model.named_buffers())
        modules = list(model.named_modules())

    def export_name(name: str) -> str:
        """Remove activation-wrapper path components from an exported name.

        Args:
            name: State, buffer, or module path in the wrapped model.

        Returns:
            The equivalent path in the standalone export model.
        """
        for wrapper_name in sorted(activation_wrappers, key=len, reverse=True):
            prefix = f"{wrapper_name}.orig_layer"
            if name == prefix or name.startswith(prefix + "."):
                name = wrapper_name + name[len(prefix) :]
        return name

    def gather(value: Any) -> Any:
        """Gather tensor values while passing non-tensor metadata through unchanged.

        Args:
            value: Tensor or metadata value to prepare for export.

        Returns:
            The gathered tensor on the writer, None on peers, or unchanged metadata.
        """
        if isinstance(value, torch.Tensor):
            value = _gather_cpu_tensor(value)
        return value

    state = {}
    for name, value in state_items:
        name = export_name(name)
        if name in aliases:
            continue
        full = gather(value)
        with _export_step():
            if writer:
                state[name] = full
    buffers = {}
    for name, value in buffer_items:
        full = gather(value)
        with _export_step():
            if writer:
                buffers[export_name(name)] = full
    metadata = {}
    names = (
        "bits",
        "group_size",
        "sym",
        "data_type",
        "scale_dtype",
        "act_bits",
        "scale",
        "zp",
        "q_scale_thresh",
        "weight_global_scale",
        "act_data_type",
        "act_group_size",
        "act_sym",
        "act_dynamic",
        "act_max",
        "act_scale",
        "act_min_scale",
        "act_max_scale",
        "input_global_scale",
        "act_quant_func",
    )
    for name, module in modules:
        attributes = {}
        for key in names:
            if hasattr(module, key):
                full = gather(getattr(module, key))
                with _export_step():
                    if isinstance(full, TensorParallelActivationQuantizer):
                        full = full.quant_func
                    if writer:
                        attributes[key] = full
        with _export_step():
            if writer and attributes:
                metadata.setdefault(export_name(name), {}).update(attributes)

    def build_exported() -> torch.nn.Module:
        """Reconstruct the writer's standalone model from gathered CPU state.

        Returns:
            An evaluation model with restored buffers, metadata, and activation wrappers.
        """
        export_config = copy.deepcopy(model.config)
        # A full checkpoint is deliberately not a distributed training checkpoint.
        # Retaining this policy makes native save_pretrained enter TP barriers even
        # though its tensors have already been gathered and only rank zero writes.
        export_config.distributed_config = None
        with torch.device("meta"):
            exported = type(model)(export_config)
            if moe_names:
                from auto_round.modeling.fused_moe.moe_experts_interface import prepare_model_for_moe_quantization

                prepare_model_for_moe_quantization(exported, implementation="linear_loop")
        exported.load_state_dict(state, strict=True, assign=True)
        exported.tie_weights()
        for name, value in buffers.items():
            parent, _, key = name.rpartition(".")
            exported.get_submodule(parent)._buffers[key] = value
        for name, attributes in metadata.items():
            module = exported.get_submodule(name)
            for key, value in attributes.items():
                setattr(module, key, value)
        for name in sorted(activation_wrappers):
            parent, _, key = name.rpartition(".")
            exported.get_submodule(parent).add_module(
                key, WrapperWALayer(exported.get_submodule(name), enable_torch_compile=False, device="cpu")
            )
        if hasattr(model, "generation_config"):
            exported.generation_config = copy.deepcopy(model.generation_config)
        return exported.eval()

    with _export_step():
        result = build_exported() if writer else None
    return result
