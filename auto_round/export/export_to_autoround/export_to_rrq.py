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
"""RRQ residual model export: save the 3 residual planes to ``auto_round:rrq``.

Storage strategy (Phase 1, packed INT2):
    The three residual planes (k = 1..K-1) are each stored in the *standard*
    single-plane INT2 AutoRound layout, packed into a single ``auto_round:rrq``
    artifact -- i.e. three INT2 AutoRound planes stored together::

        {layer}.qweight_k   # int32, (in_features // 8, out_features)  [2-bit]
        {layer}.scales_k    # float16, (num_groups, out_features)
        {layer}.qzeros_k    # int32,   (num_groups, out_features // 8)

    All three planes use the same ``group_size`` / ``sym`` as the base model, so
    each ``qweight_k``/``scales_k``/``qzeros_k`` triple is bit-identical to what
    a standalone INT2 AutoRound model would produce and dequantizes through the
    stock W2A16 ``QuantLinear.forward``.  The base plane (``qweight_0``) is NOT
    included -- it lives in the separately-exported base model.

    Packing is done by the quantizer (``RRQRTNQuantizer``) via the W2A16
    ``QuantLinear.pack`` path; this module only renames ``rrq_*`` buffers to the
    on-disk ABI names and serializes.  The ``quantization_config`` uses
    ``quant_method="auto-round-rrq"`` so loaders can distinguish the artifact.
"""

from typing import Union

import torch
import torch.nn as nn

from auto_round.logger import logger


RRQ_QUANT_METHOD = "auto-round-rrq"


def build_rrq_quantization_config(num_planes: int, group_size: int, sym: bool) -> dict:
    """Build the ``quantization_config`` dict for an RRQ residual model.

    Args:
        num_planes: Total number of planes (base + residual).
        group_size: Quantization group size.
        sym: Whether quantization is symmetric.

    Returns:
        A JSON-serializable dict describing the RRQ residual model.
    """
    return {
        "quant_method": RRQ_QUANT_METHOD,
        "format_version": 1,
        "bits": 2,
        "base_bits": 2,
        "data_type": "int",
        "act_bits": 16,
        "residual_planes": [2] * (num_planes - 1),
        "supported_effective_bits": [4, 6, 8],
        "total_planes": num_planes,
        "group_size": group_size,
        "sym": sym,
        "packing_format": "auto_round:rrq",
    }


def _first_layer_params(model: nn.Module):
    """Return ``(group_size, sym, num_planes)`` read from the first RRQ layer.

    Falls back to defaults if no RRQ layer is found.
    """
    for module in model.modules():
        if hasattr(module, "rrq_total_planes"):
            return (
                getattr(module, "rrq_group_size", 128),
                getattr(module, "rrq_sym", False),
                module.rrq_total_planes,
            )
    return 128, False, 4


def _save_state_dict_sharded(state: dict, output_dir: str, safe_serialization: bool) -> None:
    """Write a (possibly large) state dict to ``output_dir`` as sharded files.

    Uses ``safetensors`` when ``safe_serialization`` is True, otherwise the
    torch pickle format.  A single file is written when the state is small
    enough; otherwise it is split into shards with an index file.
    """
    import os

    if safe_serialization:
        from safetensors.torch import save_file

        # For small models a single ``model.safetensors`` is simplest; large
        # models use sharded files.  The sharding threshold is intentionally low
        # so the residual artifact (3 planes of INT2) stays a handful of files.
        total_bytes = sum(t.numel() * t.element_size() for t in state.values())
        if total_bytes <= (5 * 1024**3):
            save_file(state, os.path.join(output_dir, "model.safetensors"))
            return
        _shard_safetensors(state, output_dir)
    else:
        import torch

        torch.save(state, os.path.join(output_dir, "pytorch_model.bin"))


def _shard_safetensors(state: dict, output_dir: str, max_shard_bytes: int = 5 * 1024**3) -> None:
    """Split a state dict into ``model-00001-of-NNNNN.safetensors`` shards + index."""
    import os

    from safetensors.torch import save_file

    weights_map = {}
    shard = {}
    shard_size = 0
    shard_count = 0
    for name, tensor in state.items():
        tensor_bytes = tensor.numel() * tensor.element_size()
        if shard and shard_size + tensor_bytes > max_shard_bytes:
            shard_count += 1
            fname = f"model-{shard_count:05d}.safetensors"
            save_file(shard, os.path.join(output_dir, fname))
            for k in shard:
                weights_map[k] = fname
            shard = {}
            shard_size = 0
        shard[name] = tensor
        shard_size += tensor_bytes
    if shard:
        shard_count += 1
        fname = f"model-{shard_count:05d}.safetensors"
        save_file(shard, os.path.join(output_dir, fname))
        for k in shard:
            weights_map[k] = fname

    index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in state.values())},
        "weight_map": weights_map,
    }
    import json

    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def _write_quantization_config(quantization_config: dict, output_dir: str) -> None:
    """Write the ``quantization_config`` to ``output_dir``.

    Mirrors :func:`auto_round.export.utils.save_model`, which writes
    ``quantization_config.json`` and, when a model ``config`` exists, merges it
    into ``config.json``.  Here we only persist the standalone JSON file, which
    is what ``load_rrq_model`` reads (it loads the base model's ``config.json``
    for architecture and reads the residual's ``quantization_config`` from the
    config.json / quantization_config.json in this directory).
    """
    import json
    import os

    with open(os.path.join(output_dir, "quantization_config.json"), "w", encoding="utf-8") as f:
        json.dump(quantization_config, f, indent=2)

    # Also ensure a ``config.json`` exists so ``_load_config`` (which reads
    # config.json) can find the quantization_config.  If an existing config.json
    # is present we preserve it and just add the quantization_config; otherwise
    # we create a minimal stand-in describing the residual artifact.
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["quantization_config"] = quantization_config
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    else:
        cfg = {
            "quantization_config": quantization_config,
            "architectures": ["RRQResidualModel"],
            "model_type": "rrq-residual",
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)


def save_quantized_rrq(
    output_dir: str,
    model: nn.Module,
    group_size: int = None,
    sym: bool = None,
    num_planes: int = None,
    device: Union[str, torch.device] = "cpu",
    safe_serialization: bool = True,
    **kwargs,
):
    """Save the RRQ residual model to disk in ``auto_round:rrq`` format.

    The residual model contains *only* the residual planes (k = 1..K-1), each in
    the standard single-plane INT2 AutoRound layout::

        {layer}.qweight_k   int32   (in_features // 8, out_features)   [2-bit]
        {layer}.scales_k    float16 (num_groups, out_features)
        {layer}.qzeros_k    int32   (num_groups, out_features // 8)

    The three planes are packed into a *single* artifact.  The base plane
    (``qweight_0``) and all non-RRQ weights (embeddings, layernorms, ...) are
    **not** included -- they live in the separately-exported base model -- so
    the residual artifact is compact.

    The in-memory buffers are renamed from ``rrq_*_k`` to the on-disk ``*_k``
    names so ``RRQLinear``/loaders find them by convention; the packed INT2
    tensors are stored as-is (no dequant/round-trip).

    Args:
        output_dir: Output directory for the residual model.
        model: The model with RRQ-quantized layers (after ``quantize()``).
        group_size: Quantization group size (default: read from the model).
        sym: Symmetric quantization flag (default: read from the model).
        num_planes: Total planes (default: read from the model).
        device: Device for computation (unused for the save itself).
        safe_serialization: Use safetensors format (default True).
        **kwargs: Additional keyword args (e.g. ``max_shard_size``).
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Parameters carried by the model (set during quantize()) are authoritative;
    # explicit arguments only override when provided.
    model_group, model_sym, model_planes = _first_layer_params(model)
    if group_size is None:
        group_size = model_group
    if sym is None:
        sym = model_sym
    if num_planes is None:
        num_planes = model_planes

    logger.info(f"Saving RRQ residual model ({num_planes - 1} planes) to {output_dir}")

    # Build a state dict containing *only* the packed residual planes, and
    # rename the in-memory ``rrq_*_k`` buffers to the on-disk ``*_k`` names.
    residual_state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not hasattr(module, "rrq_total_planes"):
            continue
        for k in range(1, module.rrq_total_planes):
            for src, dst in (
                (f"rrq_qweight_{k}", f"qweight_{k}"),
                (f"rrq_scales_{k}", f"scales_{k}"),
                (f"rrq_qzeros_{k}", f"qzeros_{k}"),
            ):
                if src in module._buffers:
                    module._buffers[dst] = module._buffers.pop(src)
            for dst in (f"qweight_{k}", f"scales_{k}", f"qzeros_{k}"):
                if dst in module._buffers:
                    residual_state[f"{name}.{dst}"] = module._buffers[dst].detach().cpu()

    # Serialize the residual state dict (safetensors by default, torch otherwise).
    _save_state_dict_sharded(residual_state, output_dir, safe_serialization)

    # Build quantization_config and attach to model config (for config.json).
    quantization_config = build_rrq_quantization_config(num_planes, group_size, sym)
    if hasattr(model, "config"):
        model.config.quantization_config = quantization_config
    _write_quantization_config(quantization_config, output_dir)

    logger.info(
        f"RRQ residual model saved: {len(residual_state) // 3} layers, "
        f"{num_planes - 1} planes each, to {output_dir}"
    )


def save_rrq_base_model(
    output_dir: str,
    model: nn.Module,
    layer_config: dict = None,
    device: Union[str, torch.device] = "cpu",
    serialization_dict: dict = None,
    safe_serialization: bool = True,
    **kwargs,
):
    """Save the RRQ base model (standard INT2) to disk.

    The base model is a standard INT2 quantized model compatible with existing
    runtimes.  It uses the regular ``auto_round`` export path.

    IMPORTANT: The standard ``auto_round`` export packs base layers in-place
    (replacing ``nn.Linear`` with ``QuantLinear``), which drops the residual
    buffers.  Export the residual model (``save_quantized_rrq``) *first*, then
    call this to export the base model.  The base plane is stored in
    ``layer.weight``/``layer.scale``/``layer.zp`` (standard format).

    Args:
        output_dir: Output directory for the base model.
        model: The model with RRQ-quantized layers (after ``quantize()``).
        layer_config: Per-layer configuration dict.
        device: Device for computation.
        serialization_dict: Serialization config dict (from the compressor).
        safe_serialization: Use safetensors format (default True).
        **kwargs: Additional keyword arguments.
    """
    from auto_round.export.export_to_autoround.export import save_quantized_as_autoround

    save_quantized_as_autoround(
        output_dir=output_dir,
        model=model,
        layer_config=layer_config,
        device=str(device),
        backend="auto_round",
        serialization_dict=serialization_dict,
        safe_serialization=safe_serialization,
        **kwargs,
    )
