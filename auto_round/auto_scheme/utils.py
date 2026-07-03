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
import logging
import math
import re
from dataclasses import asdict, fields
from typing import Iterable, Optional, Union

import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

from auto_round.logger import logger
from auto_round.schemes import QuantizationScheme, preset_name_to_scheme, scheme_to_preset_name
from auto_round.utils import (
    DEVICE_ENVIRON_VARIABLE_MAPPING,
    SUPPORTED_LAYER_TYPES,
    check_to_quantized,
    get_block_names,
    get_layer_features,
    get_lm_head_name,
    get_module,
    is_hpex_available,
    normalize_no_split_modules,
    parse_available_devices,
)

_EXPERT_ID_PATTERN = re.compile(r"^(.*?\.experts\.\d+)(?:\.|$)")
_ZERO_EPS = 1e-12


def apply_quant_scheme(
    model: torch.nn.Module,
    quant_layer_names: Iterable[str],
    fixed_layer_scheme: dict[str, dict],
    scheme: Union[str, dict],  # TODO add scale_dtype
) -> None:
    """Apply a quantization scheme to each quantized layer.

    Args:
        model: The model whose layers are to be updated.
        scheme: The scheme preset name or dictionary to apply.
        quant_layer_names: Iterable of layer names to quantize.
        fixed_layer_scheme: Dictionary of fixed per-layer quantization schemes.
    """
    for name in quant_layer_names:
        layer_scheme = fixed_layer_scheme.get(name, scheme)
        if isinstance(layer_scheme, str):
            layer_scheme = asdict(preset_name_to_scheme(layer_scheme))

        module = get_module(model, name)
        for key, value in layer_scheme.items():
            setattr(module, key, value)


def remove_quant_scheme(
    model: torch.nn.Module,
) -> None:
    """Remove attributes corresponding to the applied quantization scheme.

    Args:
        model: The model whose layers are to be cleared.
    """
    scheme_keys = [f.name for f in fields(QuantizationScheme)] + ["scale_dtype"]
    # `rotation_config` is a QuantizationScheme field but on the model root
    # it carries the active Hadamard rotation state (weights + hooks). Deleting
    # it there would silently corrupt the rotation transform, so we never strip
    # `rotation_config` from the top-level model object.
    root_preserve = {"rotation_config"}
    for n, m in model.named_modules():
        is_root = n == ""
        for key in scheme_keys:
            if is_root and key in root_preserve:
                continue
            if hasattr(m, key):
                delattr(m, key)


def compute_avg_bits_for_scheme(
    model: torch.nn.Module,
    quant_layer_names: Iterable[str],
    fixed_layer_scheme: dict[str, dict],
    scheme: Union[str, dict, None] = None,
    ignore_scale_zp_bits: bool = False,
    clean_scheme: bool = True,
) -> tuple[float, float]:
    """Compute the average and total bit usage for the given quantization scheme.

    Args:
        model: The model to analyze.
        quant_layer_names: Iterable of layer names to include.
        fixed_layer_scheme: Dictionary of fixed per-layer quantization schemes.
        scheme: Optional scheme to temporarily apply before measuring.
        ignore_scale_zp_bits: If True, ignores overhead from scale and zero-points.
        clean_scheme: If True, removes the applied quantization scheme after computation.

    Returns:
        A tuple (avg_bits, total_quantized_bits):
            avg_bits: Average bitwidth per parameter.
            total_quantized_bits: Total quantized bit count.
    """
    if scheme is not None:
        apply_quant_scheme(model, quant_layer_names, fixed_layer_scheme, scheme)

    total_params = 0
    total_quantized_bits = 0

    for name in quant_layer_names:
        module = get_module(model, name)
        # if isinstance(module,torch.nn.Embedding):
        #     continue
        if not hasattr(module, "weight"):
            continue
        n_param = module.weight.numel()
        if n_param == 0 and hasattr(module, "_cached_weight_numel"):
            n_param = module._cached_weight_numel
        total_params += n_param
        layer_bits, _ = compute_layer_bits(module, ignore_scale_zp_bits)
        total_quantized_bits += layer_bits
    avg_bits = float(total_quantized_bits) / total_params

    if scheme is not None and clean_scheme:
        remove_quant_scheme(model)

    return avg_bits, total_quantized_bits


def compute_avg_bits_for_model(model: torch.nn.Module, ignore_scale_zp_bits: bool = False):
    """Compute the average and total bit usage for the entire model.

    Args:
        model: The model to analyze.
        ignore_scale_zp_bits: If True, ignores overhead from scale and zero-points.
        if scheme is not None:
    """

    total_params = 0
    total_quantized_bits = 0

    for n, module in model.named_modules():
        if not hasattr(module, "bits"):
            continue
        if not hasattr(module, "weight"):
            continue
        n_param = module.weight.numel()
        if n_param == 0 and hasattr(module, "_cached_weight_numel"):
            n_param = module._cached_weight_numel
        total_params += n_param
        layer_bits, _ = compute_layer_bits(module, ignore_scale_zp_bits)
        total_quantized_bits += layer_bits

    avg_bits = float(total_quantized_bits) / total_params

    return avg_bits, total_quantized_bits


def compute_layer_bits(
    layer: torch.nn.Module,
    ignore_scale_zp_bits: bool = False,
) -> tuple[int, float]:
    """Compute total and average bitwidth for a single quantized layer.

    Args:
        layer: A PyTorch layer with quantization attributes.
        ignore_scale_zp_bits: Whether to ignore scale/zero-point overhead.

    Returns:
        A tuple (total_bits, avg_bits) representing bit usage.
    """
    weight = layer.weight
    n_param = weight.numel()
    # Use cached numel when weight has been cleared to an empty tensor (low_cpu_mem_usage offload)
    if n_param == 0 and hasattr(layer, "_cached_weight_numel"):
        n_param = layer._cached_weight_numel
    weight_bits = getattr(layer, "bits", 16)
    group_size = getattr(layer, "group_size", 128)
    data_type = getattr(layer, "data_type", "int")
    is_sym = getattr(layer, "sym", False)
    super_group_size = getattr(layer, "super_group_size", None)
    super_weight_bits = getattr(layer, "super_bits", None)

    # Unquantized layer or ignoring scale/zp overhead
    if weight_bits >= 16 or ignore_scale_zp_bits:
        if super_weight_bits is not None:  # reset gguf 16 bits to 32 bits, TODO gguf q4_0, q4_1 have bug (wenhua)
            if weight_bits >= 16:
                return 32 * n_param, 32

        return weight_bits * n_param, min(16, weight_bits)

    in_features, out_features = get_layer_features(layer)

    # Determine number of groups based on group size
    if group_size > 0:
        n_group = out_features * ((in_features + group_size - 1) // group_size)
    elif group_size == 0:
        n_group = 1
    elif group_size == -1:
        n_group = out_features
    else:
        raise ValueError(f"Invalid group_size {group_size}")

    # Compute auxiliary bits (scales, zero-points, or double quantization)
    aux_total_bits = 0
    if "mx_fp" in data_type or "nv_fp" in data_type or "fp4" in data_type:
        scale_bits = 8
    else:
        scale_bits = 16
    zp_bits = weight_bits if not is_sym or "int" in data_type else 0
    if not super_group_size:
        aux_total_bits = n_group * (scale_bits + zp_bits)
    else:
        aux_total_bits += n_group * super_weight_bits * 2
        n_super_group = (n_group + super_group_size - 1) // super_group_size
        aux_total_bits += n_super_group * 32 * 2  # 32-bit scale and min_v

    total_bits = weight_bits * n_param + aux_total_bits
    avg_bits = total_bits / n_param
    return total_bits, avg_bits


def merge_lists_unionfind(list_of_lists):
    """Merge lists that share at least one common element using union-find, returning the
    resulting disjoint groups (each input list's elements are unioned together).
    """
    parent = {}

    def find(x):
        """Return the representative (root) element of ``x``'s union-find set, with path
        compression.
        """
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        """Merge the sets containing ``x`` and ``y``."""
        root_x, root_y = find(x), find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    # Initialize Union-Find
    for lst in list_of_lists:
        for item in lst:
            if item not in parent:
                parent[item] = item
        for i in range(1, len(lst)):
            union(lst[0], lst[i])

    # Collect results
    groups = {}
    for item in parent:
        root = find(item)
        groups.setdefault(root, []).append(item)
    return list(groups.values())


def parse_shared_layers(model: torch.nn.Module, shared_patterns: Iterable[Iterable[str]]) -> list[list[str]]:
    """
    Parse shared layer groups based on regex or substring matches.

    Args:
        model (torch.nn.Module): The model whose modules will be analyzed.
        shared_patterns (Iterable[Iterable[str]]):
            Each inner iterable defines one shared group. Each element can be:
              - a string: checked by full-name or substring match
              - a regex pattern: checked by re.fullmatch or re.search

    Returns:
        list[list[str]]: A list of matched shared layer groups.
    """
    if not shared_patterns:
        return []
    # Retrieve all high-level block names (for example, transformer blocks)
    for n, m in model.named_modules():
        m.global_name = n  # attach global name

    block_names = get_block_names(model, quant_vision=True)
    block_names = [item for sublist in block_names for item in sublist]

    # Collect all supported layer names from the model
    supported_layer_names = [name for name, module in model.named_modules() if type(module) in SUPPORTED_LAYER_TYPES]

    # Separate groups into those already fully matched and those requiring pattern matching
    direct_match_groups = []
    fuzzy_match_groups = []
    for group in shared_patterns:
        match_status = {name: (name in supported_layer_names) for name in group}
        if all(match_status.values()):
            direct_match_groups.append(list(match_status.keys()))
        else:
            fuzzy_match_groups.append(match_status)

    matched_groups = list(direct_match_groups)

    # Search each block for modules matching remaining patterns
    for block_name in block_names:
        block_module = get_module(model, block_name)
        block_layer_local_names = [
            name for name, module in block_module.named_modules() if type(module) in SUPPORTED_LAYER_TYPES
        ]
        block_layer_names = []
        for name in block_layer_local_names:
            module = get_module(block_module, name)
            block_layer_names.append(module.global_name)

        for group in fuzzy_match_groups:
            matched_layers = set()
            for pattern, is_direct in group.items():
                if is_direct:
                    matched_layers.add(pattern)
                    continue

                for layer_name in block_layer_names:
                    # Try regex match first
                    try:
                        if re.fullmatch(pattern, layer_name) or re.search(pattern, layer_name):
                            matched_layers.add(layer_name)
                            continue
                    except re.error:
                        pass  # Not a valid regex, fallback to substring matching

                    # Substring or partial match
                    if pattern in layer_name:
                        matched_layers.add(layer_name)

            if matched_layers:
                matched_groups.append(sorted(matched_layers))
    matched_groups = merge_lists_unionfind(matched_groups)
    return matched_groups


def _expert_key_from_layer_name(layer_name: str) -> Optional[str]:
    """Map one MoE-related linear layer to a unique expert key.

    Gate/up/down projections belonging to the same expert should map to one key.
    """
    match = _EXPERT_ID_PATTERN.search(layer_name)
    if match:
        return match.group(1)

    if ".moe." in layer_name:
        # Fallback for models that expose MoE expert layers without
        # `.experts.<id>` naming. Best-effort collapse by dropping the last
        # projection suffix (gate/up/down/etc.).
        parts = layer_name.split(".")
        if len(parts) > 1:
            return ".".join(parts[:-1])
        return layer_name

    return None


def _short_summary_name(layer_name: str) -> str:
    """Shorten ``layer_name`` to its last two dotted segments when the final segment is a
    numeric index (e.g. expert id), otherwise return it unchanged.
    """
    parts = layer_name.rsplit(".", 2)
    if len(parts) >= 2 and parts[-1].isdigit():
        return ".".join(parts[-2:])
    return layer_name


def _scheme_short_name(scheme) -> str:
    """Concise human-readable label for a quantization scheme, e.g. 'MXFP4' or 'W4A16'."""
    if isinstance(scheme, str):
        return scheme
    if isinstance(scheme, QuantizationScheme):
        preset_name = scheme_to_preset_name(scheme)
        if preset_name:
            return preset_name
        scheme = asdict(scheme)
    if isinstance(scheme, dict):
        try:
            preset_name = scheme_to_preset_name(QuantizationScheme.from_dict(scheme))
            if preset_name:
                return preset_name
        except Exception:  # noqa: BLE001
            pass
        data_type = str(scheme.get("data_type", "")).lower()
        bits = scheme.get("bits", 16)
        act_bits = scheme.get("act_bits", 16)
        if data_type.startswith("mx_fp"):
            return f"MXFP{bits}"
        if data_type.startswith("nv_fp"):
            return f"NVFP{bits}"
        if data_type.startswith("gguf"):
            return data_type.upper()
        if act_bits >= 16:
            return f"W{bits}"
        return f"W{bits}A{act_bits}"
    return str(scheme)


def build_expert_groups(
    model: torch.nn.Module,
    quant_layer_names: list[str],
    fixed_layer_scheme: dict[str, dict],
) -> list[list[str]]:
    """Auto-detect MoE expert layers and group all experts in the same block together.

    All expert projection layers (gate/up/down etc.) across all experts within a single
    transformer block are grouped into one list, so that DP treats them as a single entity
    with summed loss and summed numel.

    Returns a list of groups, each group is a list of layer names.
    """
    block_names = get_block_names(model, quant_vision=False)
    block_names = [item for sublist in block_names for item in sublist]
    block_prefixes = [(name, name + ".") for name in sorted(block_names, key=len, reverse=True)]

    dp_names = set(quant_layer_names) - set(fixed_layer_scheme.keys())
    # Collect expert layers per block
    block_expert_layers: dict[str, list[str]] = {}
    for layer_name in dp_names:
        if _expert_key_from_layer_name(layer_name) is None:
            continue
        for block_name, block_prefix in block_prefixes:
            if layer_name.startswith(block_prefix):
                block_expert_layers.setdefault(block_name, []).append(layer_name)
                break

    groups = []
    for block_name in block_names:
        layers = block_expert_layers.get(block_name)
        if layers and len(layers) > 1:
            groups.append(sorted(layers))
    return groups


def _fill_inactive_expert_scores(scores_dict: dict[str, list[float]], block_names: list[str]):
    """Fill inactive experts with the average loss of active experts in each block.

    Inactive expert means all its tracked projection losses are zero.
    """
    block_prefixes = [(name, name + ".") for name in sorted(block_names, key=len, reverse=True)]
    block_expert_stats: dict[str, dict[str, dict[str, object]]] = {}

    for layer_name, values in scores_dict.items():
        layer_loss = float(values[1])
        if not math.isfinite(layer_loss):
            continue

        matched_block = None
        for block_name, block_prefix in block_prefixes:
            if layer_name == block_name or layer_name.startswith(block_prefix):
                matched_block = block_name
                break
        if matched_block is None:
            continue

        expert_key = _expert_key_from_layer_name(layer_name)
        if expert_key is None:
            continue

        expert_stats = block_expert_stats.setdefault(matched_block, {}).setdefault(
            expert_key,
            {"layers": [], "has_active": False},
        )
        expert_stats["layers"].append(layer_name)

        if abs(layer_loss) >= _ZERO_EPS:
            expert_stats["has_active"] = True

    for block_name, expert_stats_map in block_expert_stats.items():
        active_expert_avg_losses = []
        for expert_stats in expert_stats_map.values():
            if not expert_stats["has_active"]:
                continue
            active_losses = [
                float(scores_dict[layer_name][1])
                for layer_name in expert_stats["layers"]
                if abs(float(scores_dict[layer_name][1])) >= _ZERO_EPS
            ]
            if not active_losses:
                continue
            active_expert_avg_losses.append(sum(active_losses) / len(active_losses))
        if not active_expert_avg_losses:
            continue
        fill_value = sum(active_expert_avg_losses) / len(active_expert_avg_losses)
        for _, expert_stats in expert_stats_map.items():
            if expert_stats["has_active"]:
                continue
            for layer_name in expert_stats["layers"]:
                scores_dict[layer_name][1] = fill_value


def _log_score_summary_by_block_and_nonblock(
    scores_dict: dict[str, list[float]],
    block_names: list[str],
    model=None,
    scheme_tag: Optional[str] = None,
    summary_stage: Optional[str] = None,
):
    """Log a per-block (and non-block) breakdown of ``scores_dict`` losses at debug level."""
    if not scores_dict:
        logger.info("AutoScheme score summary: empty.")
        return

    head_name = get_lm_head_name(model) if model is not None else None
    if head_name is not None and head_name.endswith(".orig_layer"):
        head_name = head_name[: -len(".orig_layer")]
    if head_name is None and "lm_head" in scores_dict:
        head_name = "lm_head"

    block_prefixes = [(name, name + ".") for name in sorted(block_names, key=len, reverse=True)]
    block_stats: dict[str, list[float]] = {name: [0.0, 0.0] for name in block_names}
    non_block_items: list[tuple[str, float]] = []

    for layer_name, values in scores_dict.items():
        layer_loss = float(values[1])
        if not math.isfinite(layer_loss):
            continue
        matched_block = None
        for block_name, block_prefix in block_prefixes:
            if layer_name == block_name or layer_name.startswith(block_prefix):
                matched_block = block_name
                break
        if matched_block is None:
            if layer_name != head_name:
                non_block_items.append((layer_name, layer_loss))
        else:
            block_stats[matched_block][0] += layer_loss
            block_stats[matched_block][1] += 1

    tag = f"[{scheme_tag}] " if scheme_tag else ""
    stage_str = f" ({summary_stage})" if summary_stage else ""
    logger.debug("AutoScheme %sblock loss summary%s:", tag, stage_str)
    logger.debug("AutoScheme | block | avg_loss |")

    for block_name in block_names:
        total_loss, cnt = block_stats.get(block_name, [0.0, 0.0])
        avg_loss = 0.0 if cnt <= 0 else total_loss / cnt
        logger.debug("AutoScheme | %s | %.6f |", _short_summary_name(block_name), avg_loss)

    if head_name is not None:
        head_loss = None
        if head_name in scores_dict:
            head_loss = float(scores_dict[head_name][1])
        head_avg = "N/A" if head_loss is None or not math.isfinite(head_loss) else f"{head_loss:.6f}"
        logger.debug("AutoScheme | %s | %s |", head_name, head_avg)

    if non_block_items:
        non_block_items.sort(key=lambda x: x[0])
        for layer_name, layer_loss in non_block_items:
            logger.info("AutoScheme non_block=%s loss=%.6f", layer_name, layer_loss)
    else:
        logger.info("AutoScheme non_block loss: none")


def _collect_current_scores(model):
    """Snapshot the current ``mix_score`` of every wrapped module into a
    ``{name: [0.0, loss]}`` dict.
    """
    scores_dict = {}
    for name, module in model.named_modules():
        if not hasattr(module, "mix_score"):
            continue
        loss = float(module.mix_score)
        if not math.isfinite(loss):
            continue
        scores_dict[name] = [0.0, loss]
    return scores_dict


_BATCH_SUMMARY_LOG_INTERVAL = 10


def _log_batch_avg_loss(model, batch_idx: int, pbar=None, block_names=None, total_batches=None, scheme_tag=None):
    """Log the running average ``mix_score`` after processing one calibration batch."""
    is_last_batch = total_batches is not None and batch_idx == total_batches
    should_log = logger.isEnabledFor(logging.DEBUG) and (is_last_batch or batch_idx % _BATCH_SUMMARY_LOG_INTERVAL == 0)
    if not should_log:
        return

    total_loss = 0.0
    layer_cnt = 0
    for _, module in model.named_modules():
        if not hasattr(module, "mix_score"):
            continue
        loss = float(module.mix_score)
        if not math.isfinite(loss):
            continue
        total_loss += loss
        layer_cnt += 1

    avg_loss = 0.0 if layer_cnt == 0 else total_loss / layer_cnt
    tag = f"[{scheme_tag}] " if scheme_tag else ""
    batch_str = f"{batch_idx}/{total_batches}" if total_batches is not None else str(batch_idx)
    msg = f"AutoScheme {tag}cumulative batch {batch_str}  avg_loss={avg_loss:.6f} layers={layer_cnt}"
    if pbar is not None:
        pbar.write(msg)
    logger.debug(msg)

    if block_names:
        scores_dict = _collect_current_scores(model)
        if scores_dict:
            tag = f"[{scheme_tag}] " if scheme_tag else ""
            batch_str = f"{batch_idx}/{total_batches}" if total_batches is not None else str(batch_idx)
            if is_last_batch:
                logger.debug("AutoScheme %scumulative batch %s block summary skipped (same as final)", tag, batch_str)
            else:
                logger.debug("AutoScheme %scumulative batch %s block summary:", tag, batch_str)
                _log_score_summary_by_block_and_nonblock(
                    scores_dict,
                    block_names,
                    model=model,
                    scheme_tag=scheme_tag,
                    summary_stage="cumulative",
                )


def _build_layer_config_header_rows(columns: list[str]) -> list[list[str]]:
    """Build a compact two-row header for the layer-config matrix.

    The first row keeps a shared prefix for each grouped set of columns (for
    example ``mlp`` or ``self_attn``), while the second row keeps the leaf
    suffix (for example ``down_proj``).
    """
    if not columns:
        return [["block"], [""]]

    leaves = []
    first_row = []
    prev_prefix = None
    for column in columns:
        parts = column.split(".")
        if len(parts) > 1:
            prefix = ".".join(parts[:-1])
            leaves.append(parts[-1])
            if prefix == prev_prefix:
                first_row.append("")
            else:
                first_row.append(prefix)
                prev_prefix = prefix
        else:
            leaves.append(column)
            first_row.append("")
            prev_prefix = None

    header_rows = [["block"] + first_row]
    header_rows.append([""] + leaves)
    return header_rows


def _log_scheme_loss_matrix(total_scores, options, block_names, model=None, layer_numel=None):
    """For every scheme in *options* log a block×layer matrix of per-element losses.

    Cells show ``loss / weight.numel()`` in scientific notation with three
    decimal places (e.g. ``1.234E-05``).  Expert layers are aggregated into an
    ``experts`` column that shows the average per-element loss across all active
    expert layers in each block.
    """
    if not total_scores or not options:
        return

    head_name = get_lm_head_name(model) if model is not None else None
    if head_name is not None and head_name.endswith(".orig_layer"):
        head_name = head_name[: -len(".orig_layer")]
    block_prefixes = [(name, name + ".") for name in sorted(block_names, key=len, reverse=True)]

    # Classify every key in total_scores into a (block, short_name) pair or expert bucket.
    # lm_head is treated as a regular non-block row (not skipped).
    block_col_keys: dict[str, dict[str, str]] = {name: {} for name in block_names}
    block_expert_keys: dict[str, list[str]] = {name: [] for name in block_names}
    other_keys: list[str] = []  # keys not belonging to any block (e.g. lm_head)
    for key in sorted(total_scores):
        matched_block = None
        for block_name, block_prefix in block_prefixes:
            if key == block_name or key.startswith(block_prefix):
                matched_block = block_name
                break
        if matched_block is None:
            other_keys.append(key)
            continue
        if _expert_key_from_layer_name(key) is not None:
            block_expert_keys[matched_block].append(key)
        else:
            short_name = key[len(matched_block) + 1 :] if key.startswith(matched_block + ".") else key
            block_col_keys[matched_block][short_name] = key

    columns = sorted({col for row in block_col_keys.values() for col in row.keys()})
    if not columns:
        columns = ["layer"]
    has_expert_layers = any(block_expert_keys.get(b) for b in block_names)

    block_display_names = [_short_summary_name(b) for b in block_names]
    header_rows = _build_layer_config_header_rows(columns)

    # Fixed cell width for scientific-notation values: "1.234E-05" = 9 chars.
    _CELL_W = 9
    widths: dict[str, int] = {"block": max(8, max((len(b) for b in block_display_names), default=5))}
    for col in columns:
        leaf = col.split(".")[-1]
        widths[col] = max(len(leaf), _CELL_W)
    if has_expert_layers:
        widths["experts"] = max(len("experts"), _CELL_W)
    # Ensure the first column of each prefix group is wide enough for the prefix text in header row 1.
    for col, prefix in zip(columns, header_rows[0][1:]):
        if prefix:
            widths[col] = max(widths[col], len(prefix))

    header_keys = ["block"] + columns + (["experts"] if has_expert_layers else [])
    header_vals = [header_rows[0][0]] + header_rows[0][1:]
    header_vals_row2 = ([""]) + (header_rows[1][1:] if len(header_rows) > 1 else [""] * len(columns))
    if has_expert_layers:
        header_vals.append("experts")
        header_vals_row2.append("")

    def _fmt_row(values: list[str], keys: list[str]) -> str:
        return "|".join(v.ljust(widths[k]) for v, k in zip(values, keys))

    sep = "|".join("-" * widths[k] for k in header_keys)

    for scheme_idx, scheme in enumerate(options):
        scheme_name = _scheme_short_name(scheme)
        block_loss_cells: dict[str, dict[str, str]] = {name: {} for name in block_names}
        block_expert_avg: dict[str, str] = {}

        _numel = layer_numel or {}
        for block_name in block_names:
            for short_name, key in block_col_keys[block_name].items():
                loss = next((item[2] for item in total_scores.get(key, []) if item[0] == scheme_idx), None)
                if loss is not None:
                    n = _numel.get(key, 0)
                    block_loss_cells[block_name][short_name] = f"{loss / n:.3E}" if n > 0 else f"{loss:.3E}"
                else:
                    block_loss_cells[block_name][short_name] = "-"

            expert_losses_per_elem: list[float] = []
            for ek in block_expert_keys.get(block_name, []):
                el = next((item[2] for item in total_scores.get(ek, []) if item[0] == scheme_idx), None)
                if el is not None:
                    n = _numel.get(ek, 0)
                    expert_losses_per_elem.append(el / n if n > 0 else el)
            if expert_losses_per_elem:
                block_expert_avg[block_name] = f"{sum(expert_losses_per_elem) / len(expert_losses_per_elem):.3E}"
            else:
                block_expert_avg[block_name] = "-"

        logger.debug("AutoScheme [%s] per-op loss/elem matrix:", scheme_name)
        logger.debug("  %s", _fmt_row(header_vals, header_keys))
        logger.debug("  %s", _fmt_row(header_vals_row2, header_keys))
        logger.debug("  %s", sep)
        for block_name, block_display_name in zip(block_names, block_display_names):
            row_vals = [block_display_name]
            for col in columns:
                row_vals.append(block_loss_cells.get(block_name, {}).get(col, "-"))
            if has_expert_layers:
                row_vals.append(block_expert_avg.get(block_name, "-"))
            logger.debug("  %s", _fmt_row(row_vals, header_keys))
        if other_keys:
            for key in other_keys:
                loss = next((item[2] for item in total_scores.get(key, []) if item[0] == scheme_idx), None)
                if loss is not None:
                    n = _numel.get(key, 0)
                    loss_str = f"{loss / n:.3E}" if n > 0 else f"{loss:.3E}"
                else:
                    loss_str = "-"
                logger.debug("  %s -> %s", _short_summary_name(key), loss_str)


def _describe_layer_config(layer_config, total_scores, options, block_names, model=None):
    """Log final ``layer_config`` as a block-row / layer-column matrix.

    Cells show the selected scheme name only; the detailed loss delta is no
    longer printed to keep the table compact.
    """

    if not layer_config:
        logger.info("AutoScheme final layer_config: empty.")
        return

    head_name = get_lm_head_name(model) if model is not None else None
    if head_name is not None and head_name.endswith(".orig_layer"):
        head_name = head_name[: -len(".orig_layer")]
    block_prefixes = [(name, name + ".") for name in sorted(block_names, key=len, reverse=True)]

    block_cells: dict[str, dict[str, str]] = {name: {} for name in block_names}
    block_expert_scheme_counts: dict[str, dict[str, int]] = {name: {} for name in block_names}
    block_expert_layers: dict[str, dict[str, list[str]]] = {name: {} for name in block_names}
    other_rows: list[tuple[str, str]] = []

    for layer_name in sorted(layer_config):
        cell = _scheme_short_name(layer_config[layer_name])

        if head_name is not None and layer_name == head_name:
            continue

        matched_block = None
        for block_name, block_prefix in block_prefixes:
            if layer_name == block_name or layer_name.startswith(block_prefix):
                matched_block = block_name
                break

        expert_key = _expert_key_from_layer_name(layer_name)
        if matched_block is None:
            other_rows.append((layer_name, cell))
            continue

        if expert_key is not None:
            block_expert_layers[matched_block].setdefault(expert_key, []).append(layer_name)
            continue

        short_name = layer_name[len(matched_block) + 1 :] if layer_name.startswith(matched_block + ".") else layer_name
        block_cells[matched_block][short_name] = cell

    for block_name, expert_map in block_expert_layers.items():
        if not expert_map:
            continue
        counts: dict[str, int] = {}
        for layer_names in expert_map.values():
            scheme_set = {_scheme_short_name(layer_config[layer_name]) for layer_name in layer_names}
            rep_scheme = (
                next(iter(sorted(scheme_set))) if len(scheme_set) == 1 else f"MIXED({', '.join(sorted(scheme_set))})"
            )
            counts[rep_scheme] = counts.get(rep_scheme, 0) + 1
        block_expert_scheme_counts[block_name] = counts

    columns = sorted({col for row in block_cells.values() for col in row.keys()})
    if not columns:
        columns = ["layer"]

    expert_text_by_block: dict[str, str] = {}
    for block_name in block_names:
        expert_counts = block_expert_scheme_counts.get(block_name, {})
        if not expert_counts:
            expert_text_by_block[block_name] = "-"
            continue
        expert_total = len(block_expert_layers.get(block_name, {}))
        expert_text_by_block[block_name] = ", ".join(
            f"{count}/{expert_total} {scheme_name}"
            for scheme_name, count in sorted(expert_counts.items(), key=lambda item: (-item[1], item[0]))
        )

    has_expert_layers = any(block_expert_layers.get(block_name) for block_name in block_names)

    block_display_names = [_short_summary_name(block_name) for block_name in block_names]
    header_rows = _build_layer_config_header_rows(columns)

    # Width is driven by the leaf name and cell content — NOT the full dotted column name,
    # so columns stay tight (e.g. "down_proj" width, not "mlp.down_proj" width).
    widths: dict[str, int] = {"block": max(8, max((len(b) for b in block_display_names), default=5))}
    for col in columns:
        leaf = col.split(".")[-1]
        max_cell_len = max((len(block_cells.get(b, {}).get(col, "-")) for b in block_names), default=1)
        widths[col] = max(len(leaf), max_cell_len)
    if has_expert_layers:
        widths["experts"] = max(len("experts"), max((len(v) for v in expert_text_by_block.values()), default=1))
    # Ensure the first column of each prefix group is wide enough for the prefix text in header row 1.
    for col, prefix in zip(columns, header_rows[0][1:]):
        if prefix:
            widths[col] = max(widths[col], len(prefix))

    def _fmt_row(values: list[str], keys: list[str]) -> str:
        return "|".join(v.ljust(widths[k]) for v, k in zip(values, keys))

    header_keys = ["block"] + columns + (["experts"] if has_expert_layers else [])
    header_vals = [header_rows[0][0]] + header_rows[0][1:]
    if len(header_rows) > 1:
        header_vals_row2 = [header_rows[1][0]] + header_rows[1][1:]
    else:
        header_vals_row2 = [""] + [""] * len(columns)

    logger.info("AutoScheme final layer_config matrix:")
    logger.info("AutoScheme note: cell=`scheme`.")
    logger.info("  %s", _fmt_row(header_vals, header_keys))
    logger.info("  %s", _fmt_row(header_vals_row2, header_keys))
    logger.info("  %s", "|".join("-" * widths[k] for k in header_keys))

    for block_name, block_display_name in zip(block_names, block_display_names):
        row_vals = [block_display_name]
        for col in columns:
            row_vals.append(block_cells.get(block_name, {}).get(col, "-"))
        if has_expert_layers:
            row_vals.append(expert_text_by_block.get(block_name, "-"))
        logger.info("  %s", _fmt_row(row_vals, header_keys))

    if head_name is not None and head_name in layer_config:
        logger.info("  %s is using %s", head_name, _scheme_short_name(layer_config[head_name]))

    if other_rows:
        logger.info("  other:")
        for layer_name, cell in other_rows:
            logger.info("    %s -> %s", layer_name, cell)
