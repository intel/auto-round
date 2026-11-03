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
from contextlib import contextmanager

import torch
import torch.distributed as dist


def is_dtensor(value) -> bool:
    # Keep DTensor optional for non-distributed installations.
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return False
    return isinstance(value, DTensor)


def is_tp_model(model) -> bool:
    return any(is_dtensor(parameter) for parameter in model.parameters())


def replicated_tensor(value):
    """Gather a distributed result while preserving its autograd connection."""
    return value.full_tensor() if is_dtensor(value) else value


@contextmanager
def synchronized_export():
    """Share writer completion/failure after collective checkpoint gathering."""
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


def validate_tp_model(model, tp_size: int) -> None:
    """Reject unsupported plans before calibration or wrapper mutation."""
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
def _gather_cpu_tensor(value, chunk_bytes=4 * 1024 * 1024):
    """Gather a 1-D DTensor to writer CPU memory with bounded device staging.

    Broadcast byte chunks so NCCL does not require a separate Gloo group and
    communication does not depend on support for the tensor's storage dtype.
    No rank ever materializes a full distributed tensor on its accelerator.
    """
    from torch.distributed.tensor import Replicate, Shard

    writer = dist.get_rank() == 0
    if not is_dtensor(value):
        return value.detach().cpu() if writer else None
    if value.device_mesh.ndim != 1 or len(value.placements) != 1:
        raise NotImplementedError("CPU export requires a one-dimensional TP mesh.")
    placement = value.placements[0]
    local = value.to_local().detach().cpu().contiguous()
    if isinstance(placement, Replicate):
        return local if writer else None
    if not isinstance(placement, Shard):
        raise NotImplementedError("CPU export supports only replicated or sharded tensors.")
    group = value.device_mesh.get_group()
    size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    axis = placement.dim % value.ndim
    width = (value.shape[axis] + size - 1) // size
    output = torch.empty(value.shape, dtype=value.dtype, device="cpu") if writer else None
    # Gloo stages on CPU; NCCL stages at most chunk_bytes on the local GPU.
    device = (
        torch.device("cuda", torch.cuda.current_device()) if dist.get_backend(group) == "nccl" else torch.device("cpu")
    )
    for source in range(size):
        shape = list(value.shape)
        start = min(source * width, value.shape[axis])
        shape[axis] = min(width, value.shape[axis] - start)
        shard = local if source == rank else (torch.empty(shape, dtype=value.dtype) if writer else None)
        byte_count = value.element_size()
        for dimension in shape:
            byte_count *= dimension
        for offset in range(0, byte_count, chunk_bytes):
            count = min(chunk_bytes, byte_count - offset)
            staging = torch.empty(count, dtype=torch.uint8, device=device)
            if source == rank:
                staging.copy_(local.reshape(-1).view(torch.uint8)[offset : offset + count])
            dist.broadcast(staging, src=dist.get_global_rank(group, source), group=group)
            if writer:
                shard.reshape(-1).view(torch.uint8)[offset : offset + count].copy_(staging.cpu())
        if writer:
            output.narrow(axis, start, shape[axis]).copy_(shard)
    return output


def gather_for_export(model):
    """Return a non-TP CPU model on rank zero; all TP ranks must participate.

    Build a clean model instead of trying to undo Transformers' private forward
    closures. The caller retains its original TP model for further inference.
    Shards are copied to CPU before bounded transfers; full tensors exist only
    in writer host memory, never temporarily on a GPU.
    """
    from auto_round.parallel.activation import TensorParallelActivationQuantizer
    from auto_round.wrapper import WrapperWALayer

    writer = dist.get_rank() == 0
    activation_wrappers = {name for name, module in model.named_modules() if isinstance(module, WrapperWALayer)}

    def export_name(name):
        for wrapper_name in sorted(activation_wrappers, key=len, reverse=True):
            prefix = f"{wrapper_name}.orig_layer"
            if name == prefix or name.startswith(prefix + "."):
                name = wrapper_name + name[len(prefix) :]
        return name

    def gather(value):
        if isinstance(value, torch.Tensor):
            value = _gather_cpu_tensor(value)
        return value

    moe_names = [name for name, module in model.named_modules() if getattr(module, "_auto_round_tp_moe", False)]
    aliases = {f"{name}.{key}" for name in moe_names for key in ("gate_up_proj", "down_proj")}
    state = {}
    for name, value in model.state_dict().items():
        name = export_name(name)
        if name in aliases:
            continue
        full = gather(value)
        if writer:
            state[name] = full
    buffers = {}
    for name, value in model.named_buffers():
        full = gather(value)
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
    for name, module in model.named_modules():
        attributes = {}
        for key in names:
            if hasattr(module, key):
                full = gather(getattr(module, key))
                if isinstance(full, TensorParallelActivationQuantizer):
                    full = full.quant_func
                if writer:
                    attributes[key] = full
        if writer and attributes:
            metadata.setdefault(export_name(name), {}).update(attributes)
    if not writer:
        return None
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
