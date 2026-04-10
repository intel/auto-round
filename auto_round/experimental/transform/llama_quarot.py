# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import math
from contextlib import contextmanager

import torch
import tqdm

from auto_round.experimental.transform.hadamard_config import HadamardConfig
from auto_round.experimental.transform.utils.hadamard import _fetch_hadamard_divisor, deterministic_hadamard_matrix, is_pow2

__all__ = [
    "LLAMA_QUAROT_STRATEGY",
    "apply_llama_quarot_weight_transform",
    "apply_llama_quarot_layer_config_overrides",
    "llama_quarot_down_proj_groupsize",
    "llama_quarot_online_transform",
    "register_llama_quarot_online_transforms",
]


LLAMA_QUAROT_STRATEGY = "llama_quarot"
LLAMA_QUAROT_ONLINE_KIND_ATTR = "_autoround_llama_quarot_online_kind"
LLAMA_QUAROT_ONLINE_NUM_HEADS_ATTR = "_autoround_llama_quarot_online_num_heads"
LLAMA_QUAROT_ONLINE_FORCE_FP32_ATTR = "_autoround_llama_quarot_online_force_fp32"
LLAMA_QUAROT_ONLINE_HOOK_ATTR = "_autoround_llama_quarot_hook_handle"
LLAMA_QUAROT_ONLINE_HOOK_BYPASS_ATTR = "_autoround_llama_quarot_bypass_hook"
LLAMA_QUAROT_WEIGHT_APPLIED_ATTR = "_autoround_llama_quarot_weight_applied"
LLAMA_QUAROT_PROMOTED_DTYPES_ATTR = "_autoround_llama_quarot_promoted_dtypes"
LLAMA_QUAROT_HADAMARD_PRIORITY = [172, 156, 140, 108, 60, 52, 36, 28, 40, 20, 12]


def _get_llama_backbone(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "model") and hasattr(model.model, "layers") and hasattr(model.model, "embed_tokens"):
        return model.model
    if hasattr(model, "layers") and hasattr(model, "embed_tokens"):
        return model
    raise ValueError("llama_quarot currently expects a Llama-style model with `layers` and `embed_tokens`.")


def _get_lm_head(model: torch.nn.Module) -> torch.nn.Module | None:
    if hasattr(model, "get_output_embeddings"):
        return model.get_output_embeddings()
    return getattr(model, "lm_head", None)


def _get_model_config(model: torch.nn.Module):
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("llama_quarot requires `model.config` to be available.")
    return config


def _get_llama_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    return list(_get_llama_backbone(model).layers)


def _get_module_by_name(model: torch.nn.Module, module_name: str) -> torch.nn.Module | None:
    module = model
    for part in module_name.split("."):
        if not hasattr(module, part):
            return None
        module = getattr(module, part)
    return module


def llama_quarot_down_proj_groupsize(hidden_size: int, intermediate_size: int, groupsize: int) -> int:
    if groupsize <= 1:
        raise ValueError("groupsize should be greater than 1 for llama_quarot down_proj activation remapping.")

    if intermediate_size % groupsize == 0:
        return groupsize

    group_num = hidden_size // groupsize
    if group_num * groupsize != hidden_size:
        raise ValueError(
            f"Invalid llama_quarot act_group_size={groupsize}: hidden_size={hidden_size} is not divisible by it."
        )

    down_proj_groupsize = intermediate_size // group_num
    if down_proj_groupsize * group_num != intermediate_size:
        raise ValueError(
            f"Invalid llama_quarot act_group_size={groupsize}: intermediate_size={intermediate_size} "
            f"cannot be evenly remapped for down_proj."
        )
    return down_proj_groupsize


def apply_llama_quarot_layer_config_overrides(
    model: torch.nn.Module,
    layer_config: dict[str, dict],
    warn_fn=None,
) -> dict[str, dict]:
    config = _get_model_config(model)
    for layer_name, scheme in layer_config.items():
        if not layer_name.endswith("down_proj"):
            continue

        act_group_size = scheme.get("act_group_size", None)
        if not isinstance(act_group_size, int) or act_group_size <= 0:
            continue

        group_size = scheme.get("group_size", None)
        if (
            warn_fn is not None
            and isinstance(group_size, int)
            and group_size > 0
            and group_size != act_group_size
        ):
            warn_fn(
                "llama_quarot follows QuaRot's Llama activation grouping best when "
                f"`group_size` matches `act_group_size`; got group_size={group_size}, "
                f"act_group_size={act_group_size} for {layer_name}."
            )

        remapped_group_size = llama_quarot_down_proj_groupsize(
            config.hidden_size, config.intermediate_size, act_group_size
        )
        if remapped_group_size == act_group_size:
            continue

        scheme["act_group_size"] = remapped_group_size
        module = _get_module_by_name(model, layer_name)
        if module is not None:
            setattr(module, "act_group_size", remapped_group_size)

    return layer_config


def _matmul_hadamard_last_dim(x: torch.Tensor, transpose: bool = False, force_fp32: bool = False) -> torch.Tensor:
    size = x.shape[-1]
    original_dtype = x.dtype
    if force_fp32 and x.dtype in (torch.float16, torch.bfloat16, torch.float32):
        compute_dtype = torch.float32
    else:
        compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x = x.to(dtype=compute_dtype)
    # QuaRot relies on Sylvester Hadamards for power-of-two dimensions so
    # attention-side factorizations like H_hidden = H_heads \otimes H_head_dim
    # remain exact.
    if is_pow2(size):
        hadamard_divisor = deterministic_hadamard_matrix(size, dtype=compute_dtype, device=x.device)
    else:
        hadamard_divisor = None
        for divisor in LLAMA_QUAROT_HADAMARD_PRIORITY:
            if size % divisor != 0 or not is_pow2(size // divisor):
                continue

            # Match QuaRot's get_hadK priority instead of picking the largest
            # divisible matrix from a generic table.
            preferred_divisor = _fetch_hadamard_divisor(divisor, compute_dtype, device=x.device)
            if preferred_divisor is not None and preferred_divisor.shape[0] == divisor:
                hadamard_divisor = preferred_divisor
                break

        if hadamard_divisor is None:
            hadamard_divisor = _fetch_hadamard_divisor(size, compute_dtype, device=x.device)
        if hadamard_divisor is None:
            raise ValueError(f"Cannot construct Hadamard transform for size {size}")

    if transpose:
        hadamard_divisor = hadamard_divisor.T

    divisor_size = hadamard_divisor.shape[0]
    input_tensor = x.contiguous().reshape(-1, size, 1)
    output_tensor = input_tensor.clone()

    while input_tensor.shape[1] > divisor_size:
        input_tensor = input_tensor.reshape(input_tensor.shape[0], input_tensor.shape[1] // 2, 2, input_tensor.shape[2])
        output_tensor = output_tensor.reshape(input_tensor.shape)
        output_tensor[:, :, 0, :] = input_tensor[:, :, 0, :] + input_tensor[:, :, 1, :]
        output_tensor[:, :, 1, :] = input_tensor[:, :, 0, :] - input_tensor[:, :, 1, :]
        output_tensor = output_tensor.reshape(input_tensor.shape[0], input_tensor.shape[1], -1)
        input_tensor, output_tensor = output_tensor, input_tensor

    if divisor_size > 1:
        input_tensor = hadamard_divisor.view(1, divisor_size, divisor_size).to(input_tensor) @ input_tensor

    output = input_tensor.reshape(x.shape) / math.sqrt(size)
    return output.to(dtype=original_dtype)


def _make_rotation_signs(size: int, device: torch.device, config: HadamardConfig) -> torch.Tensor:
    if config.hadamard_type == "hadamard":
        return torch.ones(size, dtype=torch.float64, device=device)

    signs = torch.randint(0, 2, (size,), device=device, dtype=torch.int64)
    return signs.mul_(2).sub_(1).to(torch.float64)


def _apply_random_hadamard_right(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    return _matmul_hadamard_last_dim(x * signs.to(device=x.device, dtype=x.dtype))


def _apply_random_hadamard_left_transpose(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    return _apply_random_hadamard_right(x.transpose(-1, -2).contiguous(), signs).transpose(-1, -2).contiguous()


def _apply_exact_hadamard_right(x: torch.Tensor) -> torch.Tensor:
    return _matmul_hadamard_last_dim(x)


def _apply_exact_hadamard_to_output_chunks(weight: torch.Tensor, hadamard_dim: int) -> torch.Tensor:
    if weight.shape[0] % hadamard_dim != 0:
        raise ValueError(
            f"Output dimension {weight.shape[0]} is not divisible by the Hadamard chunk size {hadamard_dim}."
        )

    transposed_weight = weight.transpose(0, 1).contiguous()
    initial_shape = transposed_weight.shape
    transposed_weight = transposed_weight.reshape(-1, initial_shape[-1] // hadamard_dim, hadamard_dim)
    transposed_weight = _matmul_hadamard_last_dim(transposed_weight)
    return transposed_weight.reshape(initial_shape).transpose(0, 1).contiguous()


def _promote_module_params_to_fp32(module: torch.nn.Module) -> None:
    if getattr(module, LLAMA_QUAROT_PROMOTED_DTYPES_ATTR, None) is not None:
        return

    original_dtypes: dict[str, torch.dtype] = {}
    for name, parameter in module.named_parameters(recurse=False):
        original_dtypes[name] = parameter.dtype
        if parameter.dtype in (torch.float16, torch.bfloat16):
            parameter.data = parameter.data.to(dtype=torch.float32)

    setattr(module, LLAMA_QUAROT_PROMOTED_DTYPES_ATTR, original_dtypes)


def _restore_module_params_dtype(module: torch.nn.Module) -> None:
    original_dtypes = getattr(module, LLAMA_QUAROT_PROMOTED_DTYPES_ATTR, None)
    if original_dtypes is None:
        return

    for name, original_dtype in original_dtypes.items():
        parameter = getattr(module, name, None)
        if parameter is None:
            continue
        if parameter.dtype != original_dtype:
            parameter.data = parameter.data.to(dtype=original_dtype)

    delattr(module, LLAMA_QUAROT_PROMOTED_DTYPES_ATTR)


def _collect_llama_quarot_precision_modules(model: torch.nn.Module) -> list[torch.nn.Module]:
    backbone = _get_llama_backbone(model)
    modules: list[torch.nn.Module] = [backbone.embed_tokens]

    final_norm = getattr(backbone, "norm", None)
    if final_norm is not None:
        modules.append(final_norm)

    lm_head = _get_lm_head(model)
    if lm_head is not None:
        modules.append(lm_head)

    for layer in _get_llama_layers(model):
        modules.extend(
            [
                layer.input_layernorm,
                layer.post_attention_layernorm,
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
                layer.mlp.gate_proj,
                layer.mlp.up_proj,
                layer.mlp.down_proj,
            ]
        )

    return modules


def _resolve_rotation_device(target_device: str | torch.device | None) -> torch.device | None:
    """Resolve execution device for offline weight rotation."""
    if target_device is None:
        return None

    resolved = torch.device(target_device)
    if resolved.type != "cuda":
        return None

    if not torch.cuda.is_available():
        return None

    if resolved.index is None:
        return torch.device("cuda:0")
    return resolved


def _module_has_tensors(module: torch.nn.Module) -> bool:
    for _ in module.parameters():
        return True
    for _ in module.buffers():
        return True
    return False


def _module_device(module: torch.nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


@contextmanager
def _temporary_move_modules(modules: list[torch.nn.Module], target_device: torch.device | None):
    if target_device is None:
        yield
        return

    unique_modules: list[torch.nn.Module] = []
    seen = set()
    for module in modules:
        if module is None:
            continue
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)
        unique_modules.append(module)

    original_devices: dict[int, torch.device] = {}
    moved_modules: list[torch.nn.Module] = []

    for module in unique_modules:
        if not _module_has_tensors(module):
            continue
        source_device = _module_device(module)
        original_devices[id(module)] = source_device
        if source_device != target_device:
            module.to(target_device)
            moved_modules.append(module)

    try:
        yield
    finally:
        for module in reversed(moved_modules):
            module.to(original_devices[id(module)])


def _update_linear_weight(linear: torch.nn.Linear, new_weight: torch.Tensor) -> None:
    linear.weight.data.copy_(new_weight.to(device=linear.weight.device, dtype=linear.weight.dtype))


def _update_linear_bias(linear: torch.nn.Linear, signs: torch.Tensor) -> None:
    if linear.bias is None:
        return

    bias = linear.bias.data.to(dtype=torch.float64)
    rotated_bias = _apply_random_hadamard_right(bias.unsqueeze(0), signs).squeeze(0)
    linear.bias.data.copy_(rotated_bias.to(device=linear.bias.device, dtype=linear.bias.dtype))


def _rotate_linear_right(linear: torch.nn.Linear, signs: torch.Tensor) -> None:
    weight = linear.weight.data.to(dtype=torch.float64)
    rotated_weight = _apply_random_hadamard_right(weight, signs)
    _update_linear_weight(linear, rotated_weight)


def _rotate_linear_left(linear: torch.nn.Linear, signs: torch.Tensor) -> None:
    weight = linear.weight.data.to(dtype=torch.float64)
    rotated_weight = _apply_random_hadamard_left_transpose(weight, signs)
    _update_linear_weight(linear, rotated_weight)
    _update_linear_bias(linear, signs)


def _apply_exact_hadamard_to_linear_input(linear: torch.nn.Linear) -> None:
    weight = linear.weight.data.to(dtype=torch.float64)
    rotated_weight = _apply_exact_hadamard_right(weight)
    _update_linear_weight(linear, rotated_weight)


def _apply_exact_hadamard_to_linear_output(linear: torch.nn.Linear, hadamard_dim: int) -> None:
    weight = linear.weight.data.to(dtype=torch.float64)
    rotated_weight = _apply_exact_hadamard_to_output_chunks(weight, hadamard_dim)
    _update_linear_weight(linear, rotated_weight)


def _rotate_bias_left_in_float64(bias: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    return _apply_random_hadamard_right(bias.unsqueeze(0), signs).squeeze(0)


def _apply_strict_v_proj_transform(linear: torch.nn.Linear, signs: torch.Tensor, head_dim: int) -> None:
    weight = linear.weight.data.to(dtype=torch.float64)
    weight = _apply_random_hadamard_right(weight, signs)
    weight = _apply_exact_hadamard_to_output_chunks(weight, head_dim)
    _update_linear_weight(linear, weight)


def _apply_strict_o_proj_transform(linear: torch.nn.Linear, signs: torch.Tensor) -> None:
    weight = linear.weight.data.to(dtype=torch.float64)
    weight = _apply_random_hadamard_left_transpose(weight, signs)
    weight = _apply_exact_hadamard_right(weight)
    _update_linear_weight(linear, weight)
    if linear.bias is not None:
        bias = linear.bias.data.to(dtype=torch.float64)
        bias = _rotate_bias_left_in_float64(bias, signs)
        linear.bias.data.copy_(bias.to(device=linear.bias.device, dtype=linear.bias.dtype))


def _apply_strict_down_proj_transform(linear: torch.nn.Linear, signs: torch.Tensor) -> None:
    weight = linear.weight.data.to(dtype=torch.float64)
    weight = _apply_random_hadamard_left_transpose(weight, signs)
    weight = _apply_exact_hadamard_right(weight)
    _update_linear_weight(linear, weight)
    if linear.bias is not None:
        bias = linear.bias.data.to(dtype=torch.float64)
        bias = _rotate_bias_left_in_float64(bias, signs)
        linear.bias.data.copy_(bias.to(device=linear.bias.device, dtype=linear.bias.dtype))


def _fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: list[torch.nn.Linear]) -> None:
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        weight = linear.weight.data.to(dtype=torch.float64)
        linear.weight.data = (weight * layernorm.weight.data.to(dtype=torch.float64)).to(
            device=linear.weight.device, dtype=linear_dtype
        )

        if getattr(layernorm, "bias", None) is not None:
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64, device=linear.weight.device)
                )
            linear.bias.data = (
                linear.bias.data.to(dtype=torch.float64) + torch.matmul(weight, layernorm.bias.data.to(dtype=torch.float64))
            ).to(device=linear.bias.device, dtype=linear_dtype)


def _neutralize_norm(layernorm: torch.nn.Module) -> None:
    if getattr(layernorm, "weight", None) is not None:
        layernorm.weight.data.fill_(1.0)
    if getattr(layernorm, "bias", None) is not None:
        layernorm.bias.data.zero_()


def _center_embedding_weights(embedding: torch.nn.Module) -> None:
    weight = embedding.weight.data.to(dtype=torch.float64)
    centered_weight = weight - weight.mean(dim=-1, keepdim=True)
    embedding.weight.data.copy_(centered_weight.to(device=embedding.weight.device, dtype=embedding.weight.dtype))


def _mark_online_module(
    module: torch.nn.Module, kind: str, num_heads: int | None = None, force_fp32: bool = False
) -> None:
    setattr(module, LLAMA_QUAROT_ONLINE_KIND_ATTR, kind)
    if num_heads is not None:
        setattr(module, LLAMA_QUAROT_ONLINE_NUM_HEADS_ATTR, num_heads)
    setattr(module, LLAMA_QUAROT_ONLINE_FORCE_FP32_ATTR, force_fp32)


def llama_quarot_online_transform(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    online_kind = getattr(module, LLAMA_QUAROT_ONLINE_KIND_ATTR, None)
    force_fp32 = getattr(module, LLAMA_QUAROT_ONLINE_FORCE_FP32_ATTR, False)
    if online_kind is None:
        return x

    if online_kind == "full":
        return _matmul_hadamard_last_dim(x, force_fp32=force_fp32)

    if online_kind == "partial":
        num_heads = getattr(module, LLAMA_QUAROT_ONLINE_NUM_HEADS_ATTR, None)
        if num_heads is None:
            raise ValueError("Missing num_heads metadata for Llama QuaRot partial online transform.")
        if x.shape[-1] % num_heads != 0:
            raise ValueError(f"Hidden size {x.shape[-1]} is not divisible by num_heads={num_heads}.")

        head_dim = x.shape[-1] // num_heads
        original_shape = x.shape
        x = x.reshape(*original_shape[:-1], num_heads, head_dim).transpose(-1, -2).contiguous()
        x = _matmul_hadamard_last_dim(x, force_fp32=force_fp32)
        return x.transpose(-1, -2).reshape(original_shape)

    raise ValueError(f"Unsupported Llama QuaRot online transform kind: {online_kind}")


def _build_online_hook():
    def input_hook(module: torch.nn.Module, args):
        if getattr(module, LLAMA_QUAROT_ONLINE_HOOK_BYPASS_ATTR, False):
            return None

        input_tensor = args[0]
        transformed_input = llama_quarot_online_transform(module, input_tensor)
        if len(args) == 1:
            return transformed_input
        return (transformed_input,) + args[1:]

    return input_hook


def register_llama_quarot_online_transforms(
    model: torch.nn.Module,
    use_tqdm: bool = True,
    desc: str | None = None,
    force_fp32: bool = True,
) -> None:
    config = _get_model_config(model)
    layers = _get_llama_layers(model)
    num_heads = config.num_attention_heads
    desc = "Register Llama QuaRot online transforms" if desc is None else desc
    hook = _build_online_hook()

    for layer in tqdm.tqdm(layers, desc=desc, disable=(not use_tqdm)):
        for module, online_kind in ((layer.self_attn.o_proj, "partial"), (layer.mlp.down_proj, "full")):
            _mark_online_module(
                module,
                online_kind,
                num_heads=num_heads if online_kind == "partial" else None,
                force_fp32=force_fp32,
            )
            if getattr(module, LLAMA_QUAROT_ONLINE_HOOK_ATTR, None) is None:
                handle = module.register_forward_pre_hook(hook, prepend=True)
                setattr(module, LLAMA_QUAROT_ONLINE_HOOK_ATTR, handle)


@torch.no_grad()
def apply_llama_quarot_weight_transform(
    model: torch.nn.Module,
    config: HadamardConfig,
    use_tqdm: bool = True,
    desc: str | None = None,
    target_device: str | torch.device | None = None,
) -> torch.nn.Module:
    if getattr(model, LLAMA_QUAROT_WEIGHT_APPLIED_ATTR, False):
        return model

    model_config = _get_model_config(model)
    if model_config.hidden_size % model_config.num_attention_heads != 0:
        raise ValueError(
            f"hidden_size={model_config.hidden_size} is not divisible by num_attention_heads={model_config.num_attention_heads}."
        )

    backbone = _get_llama_backbone(model)
    layers = _get_llama_layers(model)
    embed_tokens = backbone.embed_tokens
    final_norm = getattr(backbone, "norm", None)
    lm_head = _get_lm_head(model)
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    rotation_device = _resolve_rotation_device(target_device)
    signs_device = rotation_device if rotation_device is not None else embed_tokens.weight.device
    setattr(model, "_autoround_llama_quarot_rotation_device", str(signs_device))
    rotation_signs = _make_rotation_signs(model_config.hidden_size, signs_device, config)
    strict_quarot = config.llama_quarot_strict
    precision_modules = _collect_llama_quarot_precision_modules(model)

    for module in precision_modules:
        _promote_module_params_to_fp32(module)
    try:
        if lm_head is not None and final_norm is not None:
            with _temporary_move_modules([final_norm, lm_head], rotation_device):
                _fuse_ln_linear(final_norm, [lm_head])
                _neutralize_norm(final_norm)

        with _temporary_move_modules([embed_tokens], rotation_device):
            if config.llama_quarot_center_embeddings:
                _center_embedding_weights(embed_tokens)
            _rotate_linear_right(embed_tokens, rotation_signs)

        if lm_head is not None:
            with _temporary_move_modules([lm_head], rotation_device):
                _rotate_linear_right(lm_head, rotation_signs)

        desc = f"Applying {config.hadamard_type} Llama QuaRot transforms" if desc is None else desc
        for layer in tqdm.tqdm(layers, desc=desc, disable=(not use_tqdm)):
            with _temporary_move_modules([layer], rotation_device):
                _fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])
                _neutralize_norm(layer.post_attention_layernorm)
                _fuse_ln_linear(
                    layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]
                )
                _neutralize_norm(layer.input_layernorm)

                _rotate_linear_right(layer.self_attn.q_proj, rotation_signs)
                _rotate_linear_right(layer.self_attn.k_proj, rotation_signs)
                _rotate_linear_right(layer.mlp.up_proj, rotation_signs)
                _rotate_linear_right(layer.mlp.gate_proj, rotation_signs)

                if strict_quarot:
                    _apply_strict_v_proj_transform(layer.self_attn.v_proj, rotation_signs, head_dim)
                    _apply_strict_o_proj_transform(layer.self_attn.o_proj, rotation_signs)
                    _apply_strict_down_proj_transform(layer.mlp.down_proj, rotation_signs)
                else:
                    _rotate_linear_right(layer.self_attn.v_proj, rotation_signs)
                    _rotate_linear_left(layer.self_attn.o_proj, rotation_signs)
                    _apply_exact_hadamard_to_linear_output(layer.self_attn.v_proj, head_dim)
                    _apply_exact_hadamard_to_linear_input(layer.self_attn.o_proj)
                    _rotate_linear_left(layer.mlp.down_proj, rotation_signs)
                    _apply_exact_hadamard_to_linear_input(layer.mlp.down_proj)
    finally:
        for module in precision_modules:
            _restore_module_params_dtype(module)

    setattr(model, LLAMA_QUAROT_WEIGHT_APPLIED_ATTR, True)
    return model
