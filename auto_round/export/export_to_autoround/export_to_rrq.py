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


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Generate residual from existing base model + FP weights
# ─────────────────────────────────────────────────────────────────────────────


def _generate_residual_for_layer(
    W_fp: torch.Tensor,
    W_dequant_base: torch.Tensor,
    bits: int,
    group_size: int,
    sym: bool,
    num_planes: int,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Generate 3 packed INT2 residual planes for a single layer.

    Args:
        W_fp: Original FP weight tensor, shape ``(out_features, in_features)``.
        W_dequant_base: Dequantized base weight, shape ``(out_features, in_features)``.
        bits: Bits per plane (always 2 for RRQ).
        group_size: Quantization group size.
        sym: Symmetric quantization flag.
        num_planes: Total planes (4 = 1 base + 3 residual).

    Returns:
        List of ``(qweight, scales, qzeros)`` tuples for planes 1..K-1.
    """
    from auto_round.algorithms.quantization.rrq.quantizer import _rrq_quant_linear_class
    from auto_round.data_type.utils import get_quant_func

    QuantLinear = _rrq_quant_linear_class(sym)
    quant_func, _ = get_quant_func(
        dtype="int",
        bits=bits,
        sym=sym,
        disable_opt_rtn=True,
        group_size=group_size,
        iters=0,
    )

    out_features, in_features = W_fp.shape
    scale_dtype = torch.float16

    def _normalize(scale, zp):
        if isinstance(scale, torch.Tensor):
            scale = scale.reshape(out_features, -1).to(scale_dtype)
        else:
            scale = torch.tensor(scale, dtype=scale_dtype)
        if isinstance(zp, torch.Tensor):
            zp = zp.reshape(out_features, -1)
        return scale, zp

    def _pack(dequant, scale, zp):
        scale_n, zp_n = _normalize(scale, zp)
        ql = QuantLinear(bits, group_size, in_features, out_features, bias=False)
        plane_linear = nn.Linear(in_features, out_features, bias=False)
        plane_linear.weight.data = dequant.detach().clone().to(torch.float32)
        ql.to("cpu")
        ql.pack(plane_linear, scale_n, zp_n, None, device="cpu")
        ql.to("cpu")
        return ql.qweight.detach(), ql.scales.detach(), ql.qzeros.detach()

    # Start from the residual after the base plane
    residual = W_fp.to(torch.float32) - W_dequant_base.to(torch.float32)
    accumulated = torch.zeros_like(residual)
    planes = []

    for plane_idx in range(1, num_planes):
        quantized, scale, zp = quant_func(
            residual,
            bits=bits,
            group_size=group_size,
            scale_dtype=scale_dtype,
            q_scale_thresh=1e-5,
        )
        quantized = quantized.to(torch.float32)
        accumulated = accumulated + quantized
        planes.append(_pack(quantized, scale, zp))
        residual = residual - quantized

    return planes


def generate_rrq_residual(
    base_model_dir: str,
    raw_model: str,
    output_dir: str,
    group_size: int = 128,
    sym: bool = False,
    device: Union[str, torch.device] = "cpu",
    safe_serialization: bool = True,
):
    """Generate an RRQ residual model from an existing INT2 base model + FP weights.

    This is the Phase 2 entry point: for users who already have an INT2 quantized
    model, generate the 3 residual planes without re-quantizing the base.

    For each eligible layer:
        1. Dequantize the base plane: ``W_dequant_base`` (from packed ``qweight``/``scales``/``qzeros``)
        2. Load the original FP weight: ``W_fp``
        3. Compute ``E_1 = W_fp - W_dequant_base``
        4. Run 3 rounds of RTN INT2 quantization on the residual
        5. Pack the results as ``auto_round:rrq``

    Args:
        base_model_dir: Path to the exported INT2 base model (standard ``auto_round`` format).
        raw_model: Path (or HF name) to the original FP model weights.
        output_dir: Output directory for the residual model.
        group_size: Quantization group size (must match the base model).
        sym: Symmetric quantization flag (must match the base model).
        device: Device for computation.
        safe_serialization: Use safetensors format (default True).
    """
    import glob
    import json
    import os

    from auto_round.algorithms.quantization.rrq.quantizer import _rrq_quant_linear_class

    bits = 2  # fixed for RRQ
    num_planes = 4  # 1 base + 3 residual

    # Load base config to validate group_size/sym
    base_config_path = os.path.join(base_model_dir, "config.json")
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"config.json not found in {base_model_dir}")
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_config = json.load(f)
    base_quant = base_config.get("quantization_config", {})
    base_bits = base_quant.get("bits", 2)
    base_group_size = base_quant.get("group_size", group_size)
    base_sym = base_quant.get("sym", sym)

    if base_bits != bits:
        raise ValueError(f"Base model bits={base_bits}, but RRQ requires bits={bits}.")
    if base_group_size != group_size:
        raise ValueError(f"Base model group_size={base_group_size}, but got group_size={group_size}.")
    if base_sym != sym:
        raise ValueError(f"Base model sym={base_sym}, but got sym={sym}.")

    logger.info(
        f"Generating RRQ residual from base={base_model_dir} + raw={raw_model} "
        f"(group_size={group_size}, sym={sym}) -> {output_dir}"
    )

    # Load base state dict (packed INT2)
    base_state: dict[str, torch.Tensor] = {}
    for f in sorted(glob.glob(os.path.join(base_model_dir, "*.safetensors"))):
        from safetensors.torch import load_file as _load_st
        base_state.update(_load_st(f))
    if not base_state:
        for f in sorted(glob.glob(os.path.join(base_model_dir, "*.bin"))):
            base_state.update(torch.load(f, map_location="cpu", weights_only=False))
    if not base_state:
        raise FileNotFoundError(f"No weight files found in {base_model_dir}")

    # Load raw (FP) state dict
    raw_state: dict[str, torch.Tensor] = {}
    raw_path = os.path.expanduser(raw_model)
    if os.path.isdir(raw_path):
        for f in sorted(glob.glob(os.path.join(raw_path, "*.safetensors"))):
            from safetensors.torch import load_file as _load_st
            raw_state.update(_load_st(f))
        if not raw_state:
            for f in sorted(glob.glob(os.path.join(raw_path, "*.bin"))):
                raw_state.update(torch.load(f, map_location="cpu", weights_only=False))
    else:
        # HF model name -- download via transformers
        import transformers
        from safetensors.torch import load_file as _load_st

        hf_dir = transformers.AutoConfig.from_pretrained(raw_model)
        # Use snapshot_download to get the local path
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(raw_model)
        for f in sorted(glob.glob(os.path.join(local_dir, "*.safetensors"))):
            raw_state.update(_load_st(f))
        if not raw_state:
            for f in sorted(glob.glob(os.path.join(local_dir, "*.bin"))):
                raw_state.update(torch.load(f, map_location="cpu", weights_only=False))

    if not raw_state:
        raise FileNotFoundError(f"No weight files found for raw model: {raw_model}")

    # Enumerate base layers (packed INT2 with .qweight keys)
    base_layers = {}
    for key, value in base_state.items():
        if not key.endswith(".qweight") or not isinstance(value, torch.Tensor):
            continue
        if key.rsplit(".", 1)[1] != "qweight":
            continue
        layer_name = key.rsplit(".qweight", 1)[0]
        if f"{layer_name}.scales" not in base_state:
            continue
        in_features = value.shape[0] * 32 // bits
        out_features = value.shape[1]
        base_layers[layer_name] = (in_features, out_features)

    if not base_layers:
        raise ValueError(f"No packed INT2 layers found in base model {base_model_dir}")

    logger.info(f"Found {len(base_layers)} base layers to generate residual for.")

    # Build residual state dict
    residual_state: dict[str, torch.Tensor] = {}
    processed = 0
    for layer_name, (in_features, out_features) in base_layers.items():
        # Check that the raw model has this layer's weight
        if f"{layer_name}.weight" not in raw_state:
            logger.warning(f"Layer {layer_name!r} not found in raw model; skipping.")
            continue

        W_fp = raw_state[f"{layer_name}.weight"].to(torch.float32)

        # Dequantize the base plane to get W_dequant_base
        qweight = base_state[f"{layer_name}.qweight"].to(torch.int32)
        scales = base_state[f"{layer_name}.scales"].to(torch.float16)
        qzeros = base_state.get(f"{layer_name}.qzeros")
        if qzeros is None:
            num_groups = (in_features + group_size - 1) // group_size
            pack_factor = 32 // bits
            qzeros = torch.zeros((num_groups, max(1, out_features // pack_factor)), dtype=torch.int32)
        else:
            qzeros = qzeros.to(torch.int32)

        # Build QuantLinear and dequant
        QuantLinear = _rrq_quant_linear_class(sym)
        ql = QuantLinear(bits, group_size, in_features, out_features, bias=False)
        ql.qweight.data = qweight
        ql.scales.data = scales
        ql.qzeros.data = qzeros
        ql.to("cpu")

        # Dequant by running forward on identity
        identity = torch.eye(in_features, dtype=torch.float32)
        with torch.no_grad():
            out = ql.forward(identity)  # (in_features, out_features)
        W_dequant_base = out.T.to(torch.float32)  # (out_features, in_features)

        # Generate 3 residual planes
        planes = _generate_residual_for_layer(
            W_fp, W_dequant_base, bits, group_size, sym, num_planes
        )

        for k, (qw, sc, qz) in enumerate(planes, start=1):
            residual_state[f"{layer_name}.qweight_{k}"] = qw
            residual_state[f"{layer_name}.scales_{k}"] = sc
            residual_state[f"{layer_name}.qzeros_{k}"] = qz
        processed += 1

        if processed % 50 == 0:
            logger.info(f"  processed {processed}/{len(base_layers)} layers...")

    if processed == 0:
        raise ValueError(
            f"No layers could be processed: no common layers found between "
            f"base model ({len(base_layers)} layers) and raw model."
        )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    _save_state_dict_sharded(residual_state, output_dir, safe_serialization)

    # Write quantization config
    quantization_config = build_rrq_quantization_config(num_planes, group_size, sym)
    _write_quantization_config(quantization_config, output_dir)

    logger.info(
        f"RRQ residual model saved: {processed} layers, {num_planes - 1} planes each, "
        f"to {output_dir}"
    )
    return residual_state
