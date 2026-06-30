# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

_LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")
_MOE_PACKED_RE = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(w13_weight|w2_weight)(?:_scale)?$")


def _resolve_model_dir(path: Path) -> Path:
    if (path / "config.json").exists():
        return path
    subs = [p for p in path.iterdir() if p.is_dir() and (p / "config.json").exists()]
    if len(subs) == 1:
        return subs[0]
    raise ValueError(f"Cannot resolve model directory from: {path}")


def _load_weight_map(model_dir: Path) -> dict[str, str]:
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists():
        obj = json.loads(idx.read_text(encoding="utf-8"))
        return dict(obj.get("weight_map", {}))

    single = model_dir / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt", device="cpu") as f:
            return {k: single.name for k in f.keys()}

    raise FileNotFoundError(f"No safetensors files found in: {model_dir}")


def _load_quant_group_size(model_dir: Path, default: int = 32) -> int:
    """Read weight group_size from quantization config if available."""
    qcfg_path = model_dir / "quantization_config.json"
    cfg_path = model_dir / "config.json"

    def _extract_group_size(obj: dict) -> int | None:
        qc = obj.get("quantization_config", obj)
        config_groups = qc.get("config_groups", {}) if isinstance(qc, dict) else {}
        for group in config_groups.values():
            weights = group.get("weights", {}) if isinstance(group, dict) else {}
            group_size = weights.get("group_size") if isinstance(weights, dict) else None
            if isinstance(group_size, int) and group_size > 0:
                return group_size
        return None

    for path in (qcfg_path, cfg_path):
        if not path.exists():
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            value = _extract_group_size(obj)
            if value is not None:
                return value
        except Exception:
            continue

    return default


def _layer_id(key: str) -> int | None:
    match = _LAYER_RE.search(key)
    if not match:
        return None
    return int(match.group(1))


def _split_rows(tensor: torch.Tensor, row_sizes: list[int]) -> list[torch.Tensor]:
    if tensor.shape[0] != sum(row_sizes):
        raise ValueError(f"Row mismatch: tensor={tensor.shape[0]}, expected={sum(row_sizes)}")
    return list(torch.split(tensor, row_sizes, dim=0))


def _collect_expert_ids(orig_weight_map: dict[str, str]) -> dict[int, list[int]]:
    layer_to_experts: dict[int, set[int]] = {}
    for key in orig_weight_map:
        match = _EXPERT_RE.search(key)
        if match is None:
            continue
        # Use gate_proj as canonical expert list source to avoid duplicates.
        if ".gate_proj.weight" not in key:
            continue
        layer_id = int(match.group(1))
        expert_id = int(match.group(2))
        layer_to_experts.setdefault(layer_id, set()).add(expert_id)
    return {lid: sorted(eids) for lid, eids in layer_to_experts.items()}


def _plan_moe_packed_split(
    key: str,
    layer_to_experts: dict[int, list[int]],
    get_orig_shape: Callable[[str], tuple[int, ...]],
) -> dict | None:
    match = _MOE_PACKED_RE.match(key)
    if match is None:
        return None

    layer_id = int(match.group(1))
    packed_name = match.group(2)
    is_scale = key.endswith("_scale")
    out_suffix = ".weight_scale" if is_scale else ".weight"
    expert_ids = layer_to_experts.get(layer_id)
    if not expert_ids:
        return None

    if packed_name == "w13_weight":
        sample_gate = f"model.layers.{layer_id}.mlp.experts.{expert_ids[0]}.gate_proj.weight"
        sample_up = f"model.layers.{layer_id}.mlp.experts.{expert_ids[0]}.up_proj.weight"
        gate_rows = get_orig_shape(sample_gate)[0]
        up_rows = get_orig_shape(sample_up)[0]
        out_keys: list[str] = []
        for expert_id in expert_ids:
            out_keys.append(f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj{out_suffix}")
            out_keys.append(f"model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj{out_suffix}")
        return {
            "kind": "moe_w13",
            "layer_id": layer_id,
            "expert_ids": expert_ids,
            "row_sizes": [gate_rows, up_rows],
            "out_keys": out_keys,
        }

    out_keys = [f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj{out_suffix}" for expert_id in expert_ids]
    return {
        "kind": "moe_w2",
        "layer_id": layer_id,
        "expert_ids": expert_ids,
        "out_keys": out_keys,
    }


def _build_shape_getter(orig_dir: Path) -> Callable[[str], tuple[int, ...]]:
    weight_map = _load_weight_map(orig_dir)
    cache: dict[tuple[str, str], tuple[int, ...]] = {}

    def get_shape(key: str) -> tuple[int, ...]:
        shard = weight_map.get(key)
        if shard is None:
            raise KeyError(f"Original key missing: {key}")
        cache_key = (shard, key)
        if cache_key in cache:
            return cache[cache_key]
        with safe_open(str(orig_dir / shard), framework="pt", device="cpu") as f:
            shape = tuple(f.get_slice(key).get_shape())
        cache[cache_key] = shape
        return shape

    return get_shape


def _plan_split(
    key: str,
    all_quant_keys: set[str],
    get_orig_shape: Callable[[str], tuple[int, ...]],
) -> tuple[list[str], list[int]] | None:
    lid = _layer_id(key)
    if lid is None:
        return None

    if ".self_attn.qkv_proj." in key:
        q = f"model.layers.{lid}.self_attn.q_proj.weight"
        k = f"model.layers.{lid}.self_attn.k_proj.weight"
        v = f"model.layers.{lid}.self_attn.v_proj.weight"
        rows = [get_orig_shape(q)[0], get_orig_shape(k)[0], get_orig_shape(v)[0]]
        names = [
            key.replace(".self_attn.qkv_proj.", ".self_attn.q_proj."),
            key.replace(".self_attn.qkv_proj.", ".self_attn.k_proj."),
            key.replace(".self_attn.qkv_proj.", ".self_attn.v_proj."),
        ]
        return names, rows

    for fused in (".mlp.gate_up_proj.", ".mlp.gate_gate_up_proj."):
        if fused in key:
            g = f"model.layers.{lid}.mlp.gate_proj.weight"
            u = f"model.layers.{lid}.mlp.up_proj.weight"
            rows = [get_orig_shape(g)[0], get_orig_shape(u)[0]]
            names = [
                key.replace(fused, ".mlp.gate_proj."),
                key.replace(fused, ".mlp.up_proj."),
            ]
            return names, rows

    # MoE expert MLP fusion: experts.{i}.gate_up_proj -> gate_proj + up_proj
    expert_match = _EXPERT_RE.search(key)
    if expert_match is not None:
        layer_id, expert_id = expert_match.group(1), expert_match.group(2)
        for fused in (".gate_up_proj.", ".gate_gate_up_proj."):
            if fused in key:
                g = f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight"
                u = f"model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight"
                rows = [get_orig_shape(g)[0], get_orig_shape(u)[0]]
                names = [
                    key.replace(fused, ".gate_proj."),
                    key.replace(fused, ".up_proj."),
                ]
                return names, rows

    # If earlier name-fix renamed fused tensor to up_proj, split it only when
    # corresponding gate_proj peer is absent.
    if ".mlp.up_proj." in key and ".mlp.shared_expert." not in key:
        gate_peer = key.replace(".mlp.up_proj.", ".mlp.gate_proj.")
        if gate_peer not in all_quant_keys:
            g = f"model.layers.{lid}.mlp.gate_proj.weight"
            u = f"model.layers.{lid}.mlp.up_proj.weight"
            rows = [get_orig_shape(g)[0], get_orig_shape(u)[0]]
            return [gate_peer, key], rows

    # MoE expert fallback: if fused tensor was renamed to up_proj earlier,
    # split only when its gate_proj peer is absent.
    if ".mlp.experts." in key and ".up_proj." in key:
        gate_peer = key.replace(".up_proj.", ".gate_proj.")
        if gate_peer not in all_quant_keys:
            expert_match = _EXPERT_RE.search(key)
            if expert_match is not None:
                layer_id, expert_id = expert_match.group(1), expert_match.group(2)
                g = f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight"
                u = f"model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight"
                rows = [get_orig_shape(g)[0], get_orig_shape(u)[0]]
                return [gate_peer, key], rows

    # vLLM expert projection naming: experts.{i}.w13_weight -> gate_proj + up_proj
    # and experts.{i}.w2_weight -> down_proj.
    if expert_match is not None:
        layer_id, expert_id = expert_match.group(1), expert_match.group(2)

        if ".w13_weight." in key:
            g = f"model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight"
            u = f"model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight"
            rows = [get_orig_shape(g)[0], get_orig_shape(u)[0]]
            names = [
                key.replace(".w13_weight.", ".gate_proj."),
                key.replace(".w13_weight.", ".up_proj."),
            ]
            return names, rows

        if ".w2_weight." in key:
            d = f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight"
            rows = [get_orig_shape(d)[0]]
            names = [key.replace(".w2_weight.", ".down_proj.")]
            return names, rows

        # Some traces/logs may use w2weight spelling; support it for compatibility.
        if ".w2weight." in key:
            d = f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight"
            rows = [get_orig_shape(d)[0]]
            names = [key.replace(".w2weight.", ".down_proj.")]
            return names, rows

    return None


def remap_fused_quantized_names(
    quant_model_dir: str | Path,
    orig_model_dir: str | Path,
    logger,
) -> tuple[int, int, int]:
    """Rewrite fused quantized tensor names to HF split names in-place.

    Returns:
        (planned_splits, index_rewrites, tensor_splits)
    """
    quant_dir = _resolve_model_dir(Path(quant_model_dir))
    orig_dir = _resolve_model_dir(Path(orig_model_dir))

    quant_weight_map = _load_weight_map(quant_dir)
    quant_group_size = _load_quant_group_size(quant_dir, default=32)
    orig_weight_map = _load_weight_map(orig_dir)
    quant_keys = set(quant_weight_map.keys())
    get_shape = _build_shape_getter(orig_dir)
    layer_to_experts = _collect_expert_ids(orig_weight_map)

    split_plan: dict[str, dict] = {}
    for key in sorted(quant_keys):
        # Skip unrelated metadata tensors; keep .weight_shape for name remap.
        if key.endswith("_shape") and not key.endswith(".weight_shape"):
            continue

        moe_plan = _plan_moe_packed_split(key, layer_to_experts, get_shape)
        if moe_plan is not None:
            split_plan[key] = moe_plan
            continue

        plan = _plan_split(key, quant_keys, get_shape)
        if plan is not None:
            out_keys, row_sizes = plan
            split_plan[key] = {
                "kind": "row",
                "out_keys": out_keys,
                "row_sizes": row_sizes,
            }

    if not split_plan:
        logger.info("No fused tensors detected for remapping under %s", quant_dir)
        return 0, 0, 0

    idx_rewrites = 0
    idx_path = quant_dir / "model.safetensors.index.json"
    if idx_path.exists():
        idx_obj = json.loads(idx_path.read_text(encoding="utf-8"))
        old_map = idx_obj.get("weight_map", {})
        new_map: dict[str, str] = {}
        for key, shard in old_map.items():
            if key in split_plan:
                out_keys = split_plan[key]["out_keys"]
                for out_key in out_keys:
                    new_map[out_key] = shard
                idx_rewrites += 1
            else:
                new_map[key] = shard
        idx_obj["weight_map"] = new_map
        idx_path.write_text(json.dumps(idx_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    shard_names = sorted(set(quant_weight_map.values()))
    tensor_splits = 0
    for shard_name in shard_names:
        shard_path = quant_dir / shard_name
        if not shard_path.exists():
            continue

        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            metadata = f.metadata()

        state_dict = load_file(str(shard_path))
        new_state: dict[str, torch.Tensor] = {}

        for key, tensor in state_dict.items():
            if key not in split_plan:
                new_state[key] = tensor
                continue

            plan = split_plan[key]
            kind = plan["kind"]

            if kind == "row":
                out_keys = plan["out_keys"]
                row_sizes = plan["row_sizes"]
                is_scale_key = key.endswith(".weight_scale")
                is_shape_key = key.endswith(".weight_shape")
                grouped_sizes = [((r + quant_group_size - 1) // quant_group_size) for r in row_sizes]

                if is_shape_key:
                    # shape metadata is not row-splittable data; rebuild from
                    # original split weight shapes to avoid fused-name leftovers.
                    for out_key in out_keys:
                        weight_key = out_key.replace(".weight_shape", ".weight")
                        expected_shape = get_shape(weight_key)
                        new_state[out_key] = torch.tensor(list(expected_shape), dtype=tensor.dtype)
                    tensor_splits += 1
                    continue


                if tensor.ndim < 2:
                    # Keep non-matrix metadata tensors unchanged.
                    new_state[key] = tensor
                    continue
                if tensor.shape[0] == sum(row_sizes):
                    parts = _split_rows(tensor, row_sizes)
                elif tensor.shape[1] == sum(row_sizes):
                    parts = list(torch.split(tensor, row_sizes, dim=1))
                elif is_scale_key and tensor.shape[0] == sum(grouped_sizes):
                    parts = _split_rows(tensor, grouped_sizes)
                elif is_scale_key and tensor.shape[1] == sum(grouped_sizes):
                    parts = list(torch.split(tensor, grouped_sizes, dim=1))
                else:
                    raise ValueError(
                        f"Row mismatch for {key}: tensor_rows={tensor.shape[0]}, tensor_cols={tensor.shape[1]}, "
                        f"expected_rows={sum(row_sizes)}"
                        + (f", expected_grouped={sum(grouped_sizes)}" if is_scale_key else "")
                    )

                for out_key, part in zip(out_keys, parts):
                    if out_key.endswith(".weight"):
                        expected_shape = get_shape(out_key)
                        if tuple(part.shape) != tuple(expected_shape):
                            raise ValueError(
                                "Remap produced unexpected weight shape without transpose fallback: "
                                f"key={out_key}, got={tuple(part.shape)}, expected={tuple(expected_shape)}"
                            )
                    elif out_key.endswith(".weight_scale"):
                        expected_weight_shape = get_shape(out_key.replace(".weight_scale", ".weight"))
                        expected_scale_shape = (
                            expected_weight_shape[0],
                            (expected_weight_shape[1] + quant_group_size - 1) // quant_group_size,
                        )
                        if tuple(part.shape) != tuple(expected_scale_shape):
                            raise ValueError(
                                "Remap produced unexpected scale shape without recompute fallback: "
                                f"key={out_key}, got={tuple(part.shape)}, expected={tuple(expected_scale_shape)}"
                            )

                    new_state[out_key] = part.contiguous()
            elif kind == "moe_w13":
                expert_ids = plan["expert_ids"]
                gate_rows, up_rows = plan["row_sizes"]
                out_keys = plan["out_keys"]
                if tensor.ndim != 3:
                    raise ValueError(f"Expected 3D MoE w13 tensor for {key}, got shape {tuple(tensor.shape)}")
                if tensor.shape[0] != len(expert_ids):
                    raise ValueError(
                        f"Expert mismatch for {key}: tensor has {tensor.shape[0]}, expected {len(expert_ids)}"
                    )
                if len(out_keys) != 2 * len(expert_ids):
                    raise ValueError(
                        f"Split-plan mismatch for {key}: out_keys={len(out_keys)}, expected={2 * len(expert_ids)}"
                    )
                for idx, _expert_id in enumerate(expert_ids):
                    expert_tensor = tensor[idx]
                    gate, up = _split_rows(expert_tensor, [gate_rows, up_rows])
                    gate_key = out_keys[2 * idx]
                    up_key = out_keys[2 * idx + 1]

                    for out_key, part in ((gate_key, gate), (up_key, up)):
                        if out_key.endswith(".weight"):
                            expected_shape = get_shape(out_key)
                            if tuple(part.shape) != tuple(expected_shape):
                                raise ValueError(
                                    "Unexpected MoE w13 weight shape without transpose fallback: "
                                    f"key={out_key}, got={tuple(part.shape)}, expected={tuple(expected_shape)}"
                                )
                        elif out_key.endswith(".weight_scale"):
                            expected_weight_shape = get_shape(out_key.replace(".weight_scale", ".weight"))
                            expected_scale_shape = (
                                expected_weight_shape[0],
                                (expected_weight_shape[1] + quant_group_size - 1) // quant_group_size,
                            )
                            if tuple(part.shape) != tuple(expected_scale_shape):
                                raise ValueError(
                                    "Unexpected MoE w13 scale shape without recompute fallback: "
                                    f"key={out_key}, got={tuple(part.shape)}, expected={tuple(expected_scale_shape)}"
                                )

                        new_state[out_key] = part.contiguous()

            elif kind == "moe_w2":
                expert_ids = plan["expert_ids"]
                out_keys = plan["out_keys"]
                if tensor.ndim < 2:
                    raise ValueError(f"Expected at least 2D MoE w2 tensor for {key}, got shape {tuple(tensor.shape)}")
                if tensor.shape[0] != len(expert_ids):
                    raise ValueError(
                        f"Expert mismatch for {key}: tensor has {tensor.shape[0]}, expected {len(expert_ids)}"
                    )
                if len(out_keys) != len(expert_ids):
                    raise ValueError(
                        f"Split-plan mismatch for {key}: out_keys={len(out_keys)}, expected={len(expert_ids)}"
                    )
                for idx, _expert_id in enumerate(expert_ids):
                    out_key = out_keys[idx]
                    part = tensor[idx]
                    if out_key.endswith(".weight"):
                        expected_shape = get_shape(out_key)
                        if tuple(part.shape) != tuple(expected_shape):
                            raise ValueError(
                                "Unexpected MoE w2 weight shape without transpose fallback: "
                                f"key={out_key}, got={tuple(part.shape)}, expected={tuple(expected_shape)}"
                            )
                    elif out_key.endswith(".weight_scale"):
                        expected_weight_shape = get_shape(out_key.replace(".weight_scale", ".weight"))
                        expected_scale_shape = (
                            expected_weight_shape[0],
                            (expected_weight_shape[1] + quant_group_size - 1) // quant_group_size,
                        )
                        if tuple(part.shape) != tuple(expected_scale_shape):
                            raise ValueError(
                                "Unexpected MoE w2 scale shape without recompute fallback: "
                                f"key={out_key}, got={tuple(part.shape)}, expected={tuple(expected_scale_shape)}"
                            )

                    new_state[out_key] = part.contiguous()

            else:
                raise ValueError(f"Unknown split kind for {key}: {kind}")

            tensor_splits += 1

        save_kwargs = {"metadata": metadata} if metadata else {}
        save_file(new_state, str(shard_path), **save_kwargs)

    logger.info(
        "Remapped fused quantized names in-place: planned=%d index_rewrites=%d tensor_splits=%d",
        len(split_plan),
        idx_rewrites,
        tensor_splits,
    )

    # For pack-quantized (W4A16 / MXFP4) MoE models, vLLM's WNA16 Marlin MoE
    # method registers parameters as ``w2_weight_packed`` / ``w13_weight_packed``.
    # The universal ``fused_moe_make_expert_params_mapping`` maps checkpoint
    # suffix ``down_proj.weight`` → runtime ``w2_weight``, so the key lookup
    # fails.  Rename the expert weight keys in the checkpoint from
    # ``{proj}.weight`` to ``{proj}.weight_packed`` so the mapping produces the
    # correct ``w2_weight_packed`` / ``w13_weight_packed`` runtime key.
    _rename_moe_expert_weight_to_packed(quant_dir, logger)

    return len(split_plan), idx_rewrites, tensor_splits


_PACK_QUANTIZED_FORMAT = "pack-quantized"
_EXPERT_PROJ_WEIGHT_RE = re.compile(
    r"(model\.layers\.\d+\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj))\.weight$"
)


def _rename_moe_expert_weight_to_packed(quant_dir: Path, logger) -> int:
    """Rename ``experts.{i}.{proj}.weight`` → ``{proj}.weight_packed`` in-place.

    Only applied for ``pack-quantized`` format checkpoints (W4A16, MXFP4).
    MXFP8 uses ``mxfp8-quantized`` format and registers ``w2_weight``
    directly, so it does not need this rename.

    Returns the number of renamed keys.
    """
    cfg_path = quant_dir / "config.json"
    if not cfg_path.exists():
        return 0
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return 0

    qcfg = cfg.get("quantization_config") if isinstance(cfg, dict) else None
    fmt = (qcfg or {}).get("format", "") if isinstance(qcfg, dict) else ""
    if fmt != _PACK_QUANTIZED_FORMAT:
        return 0

    idx_path = quant_dir / "model.safetensors.index.json"
    if not idx_path.exists():
        return 0

    idx_obj = json.loads(idx_path.read_text(encoding="utf-8"))
    old_map: dict[str, str] = idx_obj.get("weight_map", {})

    rename_map: dict[str, str] = {}  # old_key → new_key
    for key in list(old_map.keys()):
        m = _EXPERT_PROJ_WEIGHT_RE.match(key)
        if m:
            new_key = m.group(1) + ".weight_packed"
            rename_map[key] = new_key

    if not rename_map:
        return 0

    # Update index
    new_map: dict[str, str] = {}
    for key, shard in old_map.items():
        new_map[rename_map.get(key, key)] = shard
    idx_obj["weight_map"] = new_map
    idx_path.write_text(json.dumps(idx_obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Update safetensors files
    shard_names = sorted(set(old_map[k] for k in rename_map))
    for shard_name in shard_names:
        shard_path = quant_dir / shard_name
        if not shard_path.exists():
            continue
        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            metadata = f.metadata()
        state_dict = load_file(str(shard_path))
        new_state: dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            new_state[rename_map.get(key, key)] = tensor
        save_kwargs = {"metadata": metadata} if metadata else {}
        save_file(new_state, str(shard_path), **save_kwargs)

    logger.info(
        "Renamed %d MoE expert weight keys to weight_packed for pack-quantized format",
        len(rename_map),
    )
    return len(rename_map)


def synthesize_missing_moe_expert_scales(
    quant_model_dir: str | Path,
    logger,
) -> tuple[int, int]:
    """Compatibility wrapper: strict mode no longer synthesizes missing scales.

    Missing MoE expert scales are now treated as fatal export errors.
    Returns the same tuple shape for compatibility with legacy callers.
    """
    logger.warning(
        "synthesize_missing_moe_expert_scales is disabled in strict mode; "
        "running coverage validation instead."
    )
    return validate_moe_expert_scale_coverage(quant_model_dir=quant_model_dir, logger=logger)


def validate_moe_expert_scale_coverage(
    quant_model_dir: str | Path,
    logger,
) -> tuple[int, int]:
    """Validate MoE MXFP export integrity.

    For MoE models exported in MXFP format, each expert projection weight
    (gate/up/down) must have a matching ``weight_scale`` key.

    Returns:
        (expert_weight_count, expert_scale_count)
    """
    quant_dir = _resolve_model_dir(Path(quant_model_dir))

    config_path = quant_dir / "config.json"
    if not config_path.exists():
        logger.warning("Skip MoE scale coverage check: missing config.json under %s", quant_dir)
        return 0, 0

    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Skip MoE scale coverage check: cannot parse config.json under %s (%s)", quant_dir, exc)
        return 0, 0

    qcfg = cfg.get("quantization_config") if isinstance(cfg, dict) else None
    qformat = ""
    if isinstance(qcfg, dict):
        qformat = str(qcfg.get("format", "")).lower()

    if "mxfp" not in qformat:
        return 0, 0

    model_type = str(cfg.get("model_type", "")).lower()
    archs = cfg.get("architectures") if isinstance(cfg, dict) else None
    arch_text = " ".join(map(str, archs)) if isinstance(archs, list) else ""
    is_moe_model = ("moe" in model_type) or ("moe" in arch_text.lower())

    weight_map = _load_weight_map(quant_dir)
    keys = set(weight_map.keys())

    if not is_moe_model:
        # Fallback on tensor-key based detection when model metadata is ambiguous.
        is_moe_model = any(".mlp.experts." in key for key in keys)

    if not is_moe_model:
        return 0, 0

    def _is_expert_weight_key(key: str) -> bool:
        if ".mlp.experts." not in key:
            return False

        # Canonical split expert projections (e.g. MXFP8): gate/up/down .weight
        if key.endswith(".weight") and any(
            proj in key for proj in (".gate_proj.weight", ".up_proj.weight", ".down_proj.weight")
        ):
            return True

        # Packed expert projections (e.g. MXFP4 pack): gate/up/down .weight_packed
        if key.endswith(".weight_packed") and any(
            proj in key for proj in (".gate_proj.weight_packed", ".up_proj.weight_packed", ".down_proj.weight_packed")
        ):
            return True

        # vLLM fused expert projections in packed export paths.
        if key.endswith("w13_weight") or key.endswith("w2_weight"):
            return True
        if key.endswith("w13_weight_packed") or key.endswith("w2_weight_packed"):
            return True

        return False

    def _expected_scale_key(weight_key: str) -> str | None:
        if weight_key.endswith(".weight"):
            return weight_key.replace(".weight", ".weight_scale")
        if weight_key.endswith(".weight_packed"):
            return weight_key.replace(".weight_packed", ".weight_scale")
        if weight_key.endswith("w13_weight") or weight_key.endswith("w2_weight"):
            return f"{weight_key}_scale"
        if weight_key.endswith("w13_weight_packed"):
            return weight_key.replace("w13_weight_packed", "w13_weight_scale")
        if weight_key.endswith("w2_weight_packed"):
            return weight_key.replace("w2_weight_packed", "w2_weight_scale")
        return None

    expert_weight_keys = [key for key in keys if _is_expert_weight_key(key)]

    if not expert_weight_keys:
        logger.warning("MoE MXFP model detected but no expert weight tensors found under %s", quant_dir)
        return 0, 0

    missing_scales = []
    for key in expert_weight_keys:
        scale_key = _expected_scale_key(key)
        if scale_key is None or scale_key not in keys:
            missing_scales.append(scale_key)

    expert_scale_count = len(expert_weight_keys) - len(missing_scales)
    if missing_scales:
        sample = ", ".join(missing_scales[:5])
        raise RuntimeError(
            "Invalid MoE MXFP export: missing expert weight_scale tensors "
            f"(found {expert_scale_count}/{len(expert_weight_keys)}). "
            f"Example missing keys: {sample}"
        )

    logger.info(
        "MoE MXFP expert scale coverage check passed: %d/%d",
        expert_scale_count,
        len(expert_weight_keys),
    )
    return len(expert_weight_keys), expert_scale_count
