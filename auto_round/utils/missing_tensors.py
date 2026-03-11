# Copyright (c) 2025 Intel Corporation
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

# Background
# ----------
# When ``transformers`` loads a model, it only instantiates the sub-modules
# declared in the model class.  Some architectures include auxiliary
# parameters — most notably **MTP (Multi-Token Prediction)** layers — that are
# stored in the original checkpoint but are *not* part of the default module
# graph.  As a result, ``model.save_pretrained`` silently omits those tensors
# from the quantized output.
#
# This module provides :func:`copy_missing_tensors_from_source`, which detects
# and copies such tensors from the original checkpoint into the saved output
# directory.  Detection relies on a user-supplied list of name-component
# prefixes (``missing_param_prefix``, default ``["mtp"]``): a source tensor is
# considered "missing" when it is absent from the saved output *and* at least
# one dot-separated segment of its name starts with one of those prefixes.
#
# Before writing, the function optionally:
#   - **dequantizes FP8 weights** to BF16 (when a matching ``weight_scale_inv``
#     tensor is present in the source checkpoint), and
#   - **re-quantizes with RTN** into the packed WOQ format used by the rest of
#     the saved model (when ``quant_method == "auto-round"`` and
#     ``packing_format == "auto_round:auto_gptq"`` are detected in
#     ``config.json``).

import json
import os
from typing import Optional, Tuple

import torch

from auto_round.logger import logger
from auto_round.utils.weight_handler import _dequant_fp8_linear_weight

MISSING_PARAM_PREFIX_DEFAULT = ["mtp"]


def copy_missing_tensors_from_source(
    source_dir: str,
    target_dir: str,
    missing_param_prefix: list[str] | None = MISSING_PARAM_PREFIX_DEFAULT,
) -> None:
    """Copy tensors from the source checkpoint that are absent from the saved output.

    Some parameters (e.g., MTP layers) are not loaded by ``transformers`` and
    therefore not written by ``model.save_pretrained``.  This function copies
    those tensors from the original checkpoint into the quantized output dir.

    A tensor is considered "missing" when:
      1. its name is **not** already present in the saved output, **and**
      2. at least one dot-separated component of its name *starts with* one of
         the strings in ``missing_param_prefix``.

    FP8 handling:
        Missing weight tensors in FP8 dtype that have a corresponding
        ``weight_scale_inv`` tensor are dequantized to BF16 before saving.

    WOQ quantization:
        If the saved ``config.json`` contains a ``quantization_config`` with
        ``quant_method == "auto-round"`` and
        ``packing_format == "auto_round:auto_gptq"``, missing 2-D weight tensors
        will be quantized with RTN and packed into qweight/qzeros/scales format.

    Args:
        source_dir: Path to the original (pre-quantization) checkpoint directory
            or a HuggingFace repo id.
        target_dir: Directory to which the quantized model was saved.  Must
            contain a ``config.json``.
        missing_param_prefix: List of name-component prefixes that identify
            "extra" parameters not handled by ``transformers``.  A source tensor
            matches when any of its dot-separated path segments *starts with*
            one of the given strings.  Defaults to ``["mtp"]``.
    """
    if missing_param_prefix is None:
        missing_param_prefix = ["mtp"]

    if not missing_param_prefix:
        return

    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError:
        logger.warning("safetensors not available, skipping copy of missing tensors from source checkpoint")
        return

    # ------------------------------------------------------------------ #
    # Resolve source directory                                             #
    # ------------------------------------------------------------------ #
    config_path = os.path.join(target_dir, "config.json")
    if not os.path.exists(config_path):
        return

    if not os.path.isdir(source_dir):
        try:
            from auto_round.utils.model import download_hf_model

            source_dir = download_hf_model(source_dir)
        except Exception as e:
            logger.debug(f"Could not resolve source model path to check for missing tensors: {e}")
            return

    if not source_dir or not os.path.isdir(source_dir):
        return

    # ------------------------------------------------------------------ #
    # Build a mapping: tensor_name -> source shard file path               #
    # ------------------------------------------------------------------ #
    source_index_file = os.path.join(source_dir, "model.safetensors.index.json")
    source_single_file = os.path.join(source_dir, "model.safetensors")

    source_tensor_to_file: dict = {}
    if os.path.exists(source_index_file):
        with open(source_index_file) as f:
            src_index = json.load(f)
        for tensor_name, shard_file in src_index["weight_map"].items():
            source_tensor_to_file[tensor_name] = os.path.join(source_dir, shard_file)
    elif os.path.exists(source_single_file):
        with safe_open(source_single_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                source_tensor_to_file[key] = source_single_file
    else:
        return

    # ------------------------------------------------------------------ #
    # Collect tensor names already present in the saved output              #
    # ------------------------------------------------------------------ #
    saved_tensor_names: set = set()
    saved_index_file = os.path.join(target_dir, "model.safetensors.index.json")
    saved_single_file = os.path.join(target_dir, "model.safetensors")

    if os.path.exists(saved_index_file):
        with open(saved_index_file) as f:
            saved_idx = json.load(f)
        saved_tensor_names = set(saved_idx["weight_map"].keys())
    elif os.path.exists(saved_single_file):
        with safe_open(saved_single_file, framework="pt", device="cpu") as f:
            saved_tensor_names = set(f.keys())
    else:
        return

    # ------------------------------------------------------------------ #
    # Identify missing tensors via explicit prefix list                    #
    # ------------------------------------------------------------------ #
    # A tensor is "missing" when it is absent from the saved output AND
    # at least one of its dot-separated name components starts with a
    # user-supplied prefix (e.g. "mtp" matches "mtp_block", "mtp_layers").
    def _matches_prefix(name: str) -> bool:
        return any(part.startswith(pfx) for part in name.split(".") for pfx in missing_param_prefix)

    missing_tensor_names: list = [
        name for name in source_tensor_to_file if name not in saved_tensor_names and _matches_prefix(name)
    ]

    if not missing_tensor_names:
        return

    logger.info(
        f"Found {len(missing_tensor_names)} tensor(s) in the source checkpoint that are "
        f"absent from the saved output (e.g., MTP parameters). Copying them now..."
    )

    # ------------------------------------------------------------------ #
    # Load missing tensors from source shards                              #
    # ------------------------------------------------------------------ #
    shard_to_missing: dict = {}
    for tensor_name in missing_tensor_names:
        shard_file = source_tensor_to_file[tensor_name]
        shard_to_missing.setdefault(shard_file, []).append(tensor_name)

    missing_tensors_dict: dict = {}
    for shard_file, tensor_names in shard_to_missing.items():
        with safe_open(shard_file, framework="pt", device="cpu") as f:
            for tensor_name in tensor_names:
                missing_tensors_dict[tensor_name] = f.get_tensor(tensor_name)

    # ------------------------------------------------------------------ #
    # FP8 dequantization: if a weight is FP8 and its scale_inv is present  #
    # ------------------------------------------------------------------ #

    fp8_dtypes = {"torch.float8_e4m3fn", "torch.float8_e5m2", "torch.float8_e4m3fnuz", "torch.float8_e5m2fnuz"}

    dequantized_keys: list = []
    keys_to_remove: list = []
    for tensor_name, tensor in list(missing_tensors_dict.items()):
        if str(tensor.dtype) not in fp8_dtypes:
            continue
        if not tensor_name.endswith(".weight"):
            continue

        # Look for the corresponding scale_inv tensor
        base = tensor_name[: -len(".weight")]
        scale_inv_name = base + ".weight_scale_inv"
        scale_name = base + ".weight_scale"

        weight_scale = missing_tensors_dict.get(scale_inv_name)
        if weight_scale is None:
            weight_scale = missing_tensors_dict.get(scale_name)

        if weight_scale is None:
            continue

        # Try to determine block_size from config or fall back to per-tensor
        block_size = None
        if weight_scale.dim() >= 2 and weight_scale.shape != tensor.shape:
            block_m = tensor.shape[-2] // weight_scale.shape[-2] if weight_scale.shape[-2] > 0 else 1
            block_n = tensor.shape[-1] // weight_scale.shape[-1] if weight_scale.shape[-1] > 0 else 1
            if block_m > 1 or block_n > 1:
                block_size = [block_m, block_n]

        dq_weight = _dequant_fp8_linear_weight(tensor, weight_scale, block_size)
        missing_tensors_dict[tensor_name] = dq_weight.to(torch.bfloat16)
        dequantized_keys.append(tensor_name)

        for s_name in [scale_inv_name, scale_name]:
            if s_name in missing_tensors_dict:
                keys_to_remove.append(s_name)

    for k in keys_to_remove:
        missing_tensors_dict.pop(k, None)

    if dequantized_keys:
        logger.info(
            f"Dequantized {len(dequantized_keys)} FP8 weight(s) to BF16 before saving: "
            f"{dequantized_keys[:5]}{'...' if len(dequantized_keys) > 5 else ''}"
        )

    if not missing_tensors_dict:
        return

    # ------------------------------------------------------------------ #
    # WOQ quantization: quantize missing Linear weights if model is WOQ    #
    # ------------------------------------------------------------------ #
    missing_tensors_dict = _woq_quantize_missing_tensors(target_dir, missing_tensors_dict)

    if not missing_tensors_dict:
        return

    # ------------------------------------------------------------------ #
    # Write missing tensors to a new shard and update the index             #
    # ------------------------------------------------------------------ #
    is_sharded = os.path.exists(saved_index_file)

    new_shard_name = "model_extra_tensors.safetensors"
    new_shard_path = os.path.join(target_dir, new_shard_name)
    missing_tensors_dict = {k: v.contiguous() for k, v in missing_tensors_dict.items()}
    save_file(missing_tensors_dict, new_shard_path)

    if is_sharded:
        with open(saved_index_file) as f:
            saved_index = json.load(f)
        for tensor_name in missing_tensors_dict:
            saved_index["weight_map"][tensor_name] = new_shard_name
        with open(saved_index_file, "w") as f:
            json.dump(saved_index, f, indent=2)
    elif os.path.exists(saved_single_file):
        weight_map: dict = {}
        with safe_open(saved_single_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = "model.safetensors"
        for tensor_name in missing_tensors_dict:
            weight_map[tensor_name] = new_shard_name
        new_index = {"metadata": {"total_size": 0}, "weight_map": weight_map}
        with open(saved_index_file, "w") as f:
            json.dump(new_index, f, indent=2)

    logger.info(
        f"Successfully wrote {len(missing_tensors_dict)} missing tensor(s) to " f"'{new_shard_name}' in {target_dir}."
    )


def quantize_weight_rtn(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
    sym: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2-D weight tensor and pack into auto_gptq format.

    Parameters
    ----------
    weight : Tensor [out_features, in_features]
    bits   : target bit-width (e.g. 4, 8)
    group_size : quantization group size along in_features
    sym    : use symmetric quantisation
    device : compute device (cuda / cpu). Results are always returned on CPU.

    Returns
    -------
    qweight : [in_features // pack_factor, out_features]  int32
    qzeros  : [num_groups,  out_features // pack_factor]   int32
    scales  : [num_groups,  out_features]                   float16
    """
    assert weight.dim() == 2, f"Expected 2-D weight, got {weight.dim()}-D"
    out_features, in_features = weight.shape
    if device is None:
        device = weight.device
    weight = weight.to(device).float()

    # --- pad in_features to multiple of group_size ---
    if in_features % group_size != 0:
        pad = group_size - (in_features % group_size)
        weight = torch.nn.functional.pad(weight, (0, pad))
        in_features = weight.shape[1]

    num_groups = in_features // group_size
    pack_factor = 32 // bits  # values per int32

    # --- pad out_features to multiple of pack_factor (needed for qzeros) ---
    out_pad = 0
    if out_features % pack_factor != 0:
        out_pad = pack_factor - (out_features % pack_factor)
        weight = torch.nn.functional.pad(weight, (0, 0, 0, out_pad))
    padded_out = weight.shape[0]

    # Reshape for group-wise ops: [padded_out, num_groups, group_size]
    w_grouped = weight.reshape(padded_out, num_groups, group_size)

    if sym:
        half_range = (1 << (bits - 1)) - 1  # e.g. 7 for 4-bit
        zero_point = 1 << (bits - 1)  # e.g. 8 for 4-bit

        w_absmax = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
        scales_full = w_absmax / half_range  # [padded_out, num_groups, 1]

        q = torch.round(w_grouped / scales_full).clamp(-half_range - 1, half_range)
        q = (q + zero_point).to(torch.int32)  # shift to unsigned [0, 2^bits-1]

        zp = torch.full((num_groups, padded_out), zero_point, dtype=torch.int32, device=device)
    else:
        max_int = (1 << bits) - 1
        w_min = w_grouped.amin(dim=-1, keepdim=True)
        w_max = w_grouped.amax(dim=-1, keepdim=True)

        scales_full = ((w_max - w_min).clamp(min=1e-10)) / max_int
        zp_full = torch.round(-w_min / scales_full).clamp(0, max_int).to(torch.int32)

        q = torch.round(w_grouped / scales_full + zp_full.float()).clamp(0, max_int).to(torch.int32)

        zp = zp_full.squeeze(-1).t().contiguous()  # [num_groups, padded_out]

    # scales → [num_groups, padded_out] (float16)
    scales_out = scales_full.squeeze(-1).t().contiguous().to(torch.float16)

    # q → [in_features, padded_out]
    q = q.reshape(padded_out, in_features).t().contiguous()

    # ---- Pack qweight: [in_features // pack_factor, padded_out] ----
    qweight = torch.zeros(in_features // pack_factor, padded_out, dtype=torch.int32, device=device)
    for k in range(pack_factor):
        qweight |= q[k::pack_factor, :] << (bits * k)

    # ---- Pack qzeros: [num_groups, padded_out // pack_factor] ----
    qzeros = torch.zeros(num_groups, padded_out // pack_factor, dtype=torch.int32, device=device)
    for k in range(pack_factor):
        qzeros |= zp[:, k::pack_factor] << (bits * k)

    # Remove output padding from qweight / scales (qzeros stays in pack units)
    if out_pad > 0:
        qweight = qweight[:, :out_features]
        scales_out = scales_out[:, :out_features]

    # Always return CPU tensors (safetensors requires CPU)
    return qweight.cpu(), qzeros.cpu(), scales_out.cpu()


def _woq_quantize_missing_tensors(target_dir: str, missing_tensors_dict: dict) -> dict:
    """Apply WOQ (Weight-Only Quantization) to missing Linear weight tensors.

    Reads ``config.json`` from *target_dir* to obtain ``quantization_config``.
    Only activates when ``quant_method == "auto-round"`` and
    ``packing_format == "auto_round:auto_gptq"``.

    Uses :func:`quantize_weight_rtn` for RTN quantisation + packing so that
    there is **no dependency on the model object** or on QuantLinear classes.

    Non-weight tensors (bias, norms, embeddings, etc.) are kept as-is.

    Per-layer resolution:
        For each weight, ``extra_config`` is checked first (exact layer name,
        then regex pattern match), and the result is merged with global
        defaults.  This means entries like ``".*mtp.*": {"bits": 8}`` or
        ``"mtp.fc": {"bits": 16, "data_type": "fp"}`` are honoured, while
        layers absent from ``extra_config`` fall back to global ``bits`` /
        ``group_size`` / ``sym``.

    Args:
        target_dir: Output directory that contains ``config.json``.
        missing_tensors_dict: Dict mapping tensor names to tensor values.

    Returns:
        Updated dict with quantized+packed tensors replacing original weight tensors.
    """
    import re as _re

    BLOCK_NAME_TO_IGNORE = ["mtp.fc", "shared_expert_gate", "mlp.gate"]
    qconfig = _get_woq_config_from_dir(target_dir)
    if qconfig is None:
        return missing_tensors_dict

    global_bits = qconfig["bits"]
    global_group_size = qconfig["group_size"]
    global_sym = qconfig["sym"]
    block_name_to_quantize = qconfig.get("block_name_to_quantize", None)
    extra_config: dict = qconfig.get("extra_config", {}) or {}

    def _resolve_layer_cfg(layer_name: str) -> dict:
        """Return effective {bits, group_size, sym, data_type} for *layer_name*.

        Lookup order:
          1. Exact key match in extra_config.
          2. Among all regex keys that match layer_name, pick the longest pattern
             (longer pattern == more specific, e.g. ``.*mtp\\.fc.*`` beats ``.*mtp.*``).
          3. Global defaults.
        """
        override: dict = {}
        # 1. exact match
        if layer_name in extra_config:
            override = extra_config[layer_name]
        else:
            # 2. collect all matching regex patterns, keep the most specific (longest)
            best_pattern: str | None = None
            for pattern, cfg in extra_config.items():
                if pattern == layer_name:
                    continue  # already handled above
                try:
                    if _re.search(pattern, layer_name):
                        if best_pattern is None or len(pattern) > len(best_pattern):
                            best_pattern = pattern
                            override = cfg
                except _re.error:
                    pass  # not a valid regex → skip
        return {
            "bits": override.get("bits", global_bits),
            "group_size": override.get("group_size", global_group_size),
            "sym": override.get("sym", global_sym),
            "data_type": override.get("data_type", "int"),
        }

    def _is_fp_layer(layer_cfg: dict) -> bool:
        """Return True when the resolved config indicates full-precision (no quantization)."""
        dt = layer_cfg.get("data_type", "int")
        return layer_cfg["bits"] >= 16 or dt in ("fp", "float", "float16", "bfloat16", "float32")

    # Identify weight tensors eligible for quantization (2D Linear weights)
    def _is_eligible(k: str) -> bool:
        if not k.endswith(".weight"):
            return False
        if missing_tensors_dict[k].dim() != 2:
            return False
        layer_name = k[: -len(".weight")]
        layer_cfg = _resolve_layer_cfg(layer_name)
        # If extra_config explicitly covers this layer, trust its decision
        if layer_name in extra_config or any(
            _re.search(p, layer_name) for p in extra_config if p != layer_name and _is_valid_regex(p)
        ):
            return not _is_fp_layer(layer_cfg)
        # Fall back to BLOCK_NAME_TO_IGNORE heuristic
        if any(block in k for block in BLOCK_NAME_TO_IGNORE):
            return False
        return not _is_fp_layer(layer_cfg)

    weight_keys = [k for k in missing_tensors_dict if _is_eligible(k)]

    # Collect 2-D weight tensors that will NOT be quantized (for extra_config bookkeeping).
    ignored_weight_keys = [
        k
        for k in missing_tensors_dict
        if k.endswith(".weight") and missing_tensors_dict[k].dim() == 2 and k not in weight_keys
    ]

    # Update quantization config: extra_config for ignored layers + block_name_to_quantize for new blocks
    if ignored_weight_keys or weight_keys:

        def _update_qcfg(qcfg):
            if ignored_weight_keys:
                ec = qcfg.get("extra_config", {})
                for k in ignored_weight_keys:
                    layer_name = k[: -len(".weight")]
                    ec.setdefault(layer_name, {"bits": 16, "data_type": "fp"})
                qcfg["extra_config"] = ec
                logger.info(
                    f"Updated extra_config for {len(ignored_weight_keys)} ignored layer(s): "
                    f"{[k[: -len('.weight')] for k in ignored_weight_keys]}"
                )
            if weight_keys:
                existing = qcfg.get("block_name_to_quantize") or []
                if isinstance(existing, str):
                    existing = [b.strip() for b in existing.split(",") if b.strip()]
                existing_set = set(existing)
                new_prefixes = {
                    ".".join(k.split(".")[:i])
                    for k in weight_keys
                    for i, p in enumerate(k.split("."))
                    if p.isdigit() and ".".join(k.split(".")[:i])
                }
                added = sorted(new_prefixes - existing_set)
                if added:
                    merged = existing + added
                    merged = [
                        b for b in merged if not any(b != other and b.startswith(other + ".") for other in merged)
                    ]
                    qcfg["block_name_to_quantize"] = merged
                    logger.info(f"Updated block_name_to_quantize: {merged}")
            return qcfg

        for cfg_file in ["config.json", "quantization_config.json"]:
            cfg_path = os.path.join(target_dir, cfg_file)
            if not os.path.exists(cfg_path):
                continue
            logger.info(f"Processing {cfg_file} to update quantization_config for missing tensors...")
            try:
                with open(cfg_path) as f:
                    cfg_data = json.load(f)
                if cfg_file == "config.json":
                    cfg_data["quantization_config"] = _update_qcfg(cfg_data.get("quantization_config", {}))
                else:
                    cfg_data = _update_qcfg(cfg_data)
                with open(cfg_path, "w") as f:
                    json.dump(cfg_data, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to update {cfg_file}: {e}")

    if not weight_keys:
        return missing_tensors_dict

    logger.info(
        f"Applying WOQ to "
        f"{len(weight_keys)} missing Linear weight(s) (per-layer overrides from extra_config applied)..."
    )

    new_tensors: dict = {}
    packed_weight_keys: set = set()

    for weight_key in weight_keys:
        weight = missing_tensors_dict[weight_key]
        out_features, in_features = weight.shape

        layer_name = weight_key[: -len(".weight")]
        layer_cfg = _resolve_layer_cfg(layer_name)
        bits = layer_cfg["bits"]
        group_size = layer_cfg["group_size"]
        sym = layer_cfg["sym"]

        effective_gs = group_size if group_size != -1 else in_features
        if effective_gs <= 0:
            continue

        logger.debug(
            f"WOQ [{layer_name}]: bits={bits}, group_size={effective_gs}, sym={sym}, " f"shape={list(weight.shape)}"
        )

        base_name = layer_name

        try:
            qweight, qzeros, scales = quantize_weight_rtn(weight, bits=bits, group_size=effective_gs, sym=sym)
        except Exception as e:
            logger.warning(f"Failed to quantize {weight_key}: {e}, keeping original weight")
            continue

        new_tensors[base_name + ".qweight"] = qweight
        new_tensors[base_name + ".qzeros"] = qzeros
        new_tensors[base_name + ".scales"] = scales

        packed_weight_keys.add(weight_key)

    if not packed_weight_keys:
        return missing_tensors_dict

    # Replace original weights with packed versions
    result = {}
    for k, v in missing_tensors_dict.items():
        if k not in packed_weight_keys:
            result[k] = v
    result.update(new_tensors)

    logger.info(
        f"Successfully packed {len(packed_weight_keys)} weight(s) into WOQ format "
        f"({len(new_tensors)} packed tensor(s) created)."
    )
    return result


def _is_valid_regex(pattern: str) -> bool:
    """Return True if *pattern* is a valid regular expression."""
    import re as _re

    try:
        _re.compile(pattern)
        return True
    except _re.error:
        return False


def _get_woq_config_from_dir(target_dir: str) -> dict | None:
    """Read ``quantization_config`` from ``config.json`` in *target_dir*.

    Only supports ``quant_method == "auto-round"`` with
    ``packing_format == "auto_round:auto_gptq"``.  Returns ``None`` (skip
    quantization) for any other combination or if any required parameter
    (``bits``, ``group_size``, ``sym``) is missing.

    Returns a dict with keys ``bits``, ``group_size``, ``sym``
    or ``None``.
    """
    config_path = os.path.join(target_dir, "config.json")
    if not os.path.exists(config_path):
        return None

    with open(config_path) as f:
        config_data = json.load(f)

    qcfg = config_data.get("quantization_config", None)
    if not isinstance(qcfg, dict):
        return None

    quant_method = qcfg.get("quant_method", "")
    packing_format = qcfg.get("packing_format", "")
    bits = qcfg.get("bits", None)
    group_size = qcfg.get("group_size", None)
    sym = qcfg.get("sym", None)

    if quant_method != "auto-round":
        logger.debug(
            f"Skipping WOQ quantization of missing tensors: "
            f"quant_method='{quant_method}' (only 'auto-round' is supported)"
        )
        return None

    if packing_format != "auto_round:auto_gptq":
        logger.debug(
            f"Skipping WOQ quantization of missing tensors: "
            f"packing_format='{packing_format}' (only 'auto_round:auto_gptq' is supported)"
        )
        return None

    if bits is None or group_size is None or sym is None:
        logger.debug(
            f"Skipping WOQ quantization of missing tensors: "
            f"incomplete quantization_config (bits={bits}, group_size={group_size}, sym={sym})"
        )
        return None

    if bits > 8:
        return None

    block_name_to_quantize = qcfg.get("block_name_to_quantize", None)
    extra_config = qcfg.get("extra_config", {})
    return {
        "bits": bits,
        "group_size": group_size,
        "sym": sym,
        "block_name_to_quantize": block_name_to_quantize,
        "extra_config": extra_config,
    }
