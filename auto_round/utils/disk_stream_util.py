# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Lazy, mmap-backed reads of individual tensors by name straight from a
# checkpoint's safetensors shards, plus meta<->real materialize/free for a
# whole module. Used by auto_round/auto_scheme/delta_loss.py's streaming
# scoring path (get_score_for_scheme_streaming) to materialize one decoder
# block's real tensors right before scoring it and release them back to meta
# right after -- instead of loading the entire model onto CPU up front, which
# doesn't fit when the checkpoint is larger than available RAM+VRAM combined.
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open

logger = logging.getLogger(__name__)


class SafetensorsIndex:
    """Lazy, mmap-backed access to a checkpoint's tensors by name.

    Deliberately does NOT cache open safe_open() handles across calls: mmap'd pages
    stay resident (counted in RSS) for as long as the mapping is open, even after
    the torch tensor copied out of them is freed. For a checkpoint far larger than
    available RAM, caching handles indefinitely would silently re-create the exact
    problem this module exists to avoid (RSS creeping up by however much of the
    file has been touched so far, instead of staying bounded to one block at a
    time). Every read (or batch of reads via read_tensors) opens a shard, reads,
    and closes/unmaps before returning.
    """

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        index_path = self.checkpoint_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                self.weight_map: Dict[str, str] = json.load(f)["weight_map"]
        else:
            # Small, unsharded checkpoint: one model.safetensors file.
            single_file = self.checkpoint_dir / "model.safetensors"
            with safe_open(str(single_file), framework="pt") as f:
                self.weight_map = {name: single_file.name for name in f.keys()}

    def has_tensor(self, name: str) -> bool:
        return name in self.weight_map

    def read_tensor(self, name: str, device: str = "cpu") -> torch.Tensor:
        return self.read_tensors([name], device=device)[name]

    def read_tensors(self, names: list[str], device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Read several tensors, grouped by shard file so each shard is opened and
        closed (unmapped) once regardless of how many tensors are pulled from it."""
        by_shard: Dict[str, list[str]] = {}
        for name in names:
            by_shard.setdefault(self.weight_map[name], []).append(name)

        result: Dict[str, torch.Tensor] = {}
        for shard_name, shard_tensor_names in by_shard.items():
            with safe_open(str(self.checkpoint_dir / shard_name), framework="pt") as f:
                for name in shard_tensor_names:
                    tensor = f.get_tensor(name)
                    if device != "cpu":
                        tensor = tensor.to(device)
                    result[name] = tensor
        return result

    def tensor_names_with_prefix(self, prefix: str) -> list[str]:
        dotted = prefix if prefix.endswith(".") else prefix + "."
        return [n for n in self.weight_map if n == prefix or n.startswith(dotted)]


@lru_cache(maxsize=8)
def get_safetensors_index(checkpoint_dir: str) -> "SafetensorsIndex":
    """Shared ``SafetensorsIndex`` per checkpoint directory.

    Only the (cheap) name->shard map is cached; no ``safe_open`` handle is kept, so this
    does not reintroduce the mmap-residency problem ``SafetensorsIndex`` avoids. Callers
    used to build a fresh index for every block, re-reading the index JSON 40+ times.
    """
    return SafetensorsIndex(str(checkpoint_dir))


# Model-side module names sometimes differ from the checkpoint-side tensor
# names.  Transformers already maintains the authoritative per-family renames
# (``transformers/conversion_mapping.py``, checkpoint -> model direction);
# the disk-stream materializers know only model-side names, so those entries
# are inverted here.  A rename is applied only when the model-side name is
# absent from the index AND the renamed candidate exists in it, so checkpoints
# whose names already match the model are untouched.


@lru_cache(maxsize=None)
def _reverse_renamings_for(model_type):
    """Invert the checkpoint-conversion WeightRenaming entries for one family.

    ``transformers`` owns the checkpoint->model renames, so reversing them is best left to
    ``WeightTransform.reverse_transform()``: it handles capturing groups and the
    ``PrefixChange`` special cases correctly. Naively reusing a ``target_pattern`` as a
    regex pattern does not -- the targets are *replacement* strings and a ``\\1``
    backreference in one (e.g. ``model.language_model.\\1``) raises
    ``re.PatternError: invalid group reference``.

    Returns reversed ``WeightTransform`` objects; call ``rename_source_key(model_side_name)``
    on them to get the checkpoint-side name.
    """
    if not model_type:
        return ()
    try:
        from transformers.conversion_mapping import WeightRenaming, get_checkpoint_conversion_mapping
    except ImportError:  # pragma: no cover - transformers is a hard dep, guarded for safety
        return ()
    mapping = get_checkpoint_conversion_mapping(model_type)
    if not mapping:
        return ()
    reversed_transforms = []
    for entry in mapping:
        if not isinstance(entry, WeightRenaming):
            # Other converter types (expert fusions) are handled separately, by
            # `_renamed_expert_candidates` and `materialize_module`'s fused lookup.
            continue
        try:
            reversed_transforms.append(entry.reverse_transform())
        except Exception as exc:  # pragma: no cover - not every transform is reversible
            logger.debug("skipping non-reversible rename %s: %s", entry, exc)
    return tuple(reversed_transforms)


@lru_cache(maxsize=None)
def _model_types_for_dir(checkpoint_dir: str):
    """Every model_type in the checkpoint's config, including nested sub-configs.

    A VLM config nests the text model (``text_config.model_type = 'qwen3_5_moe_text'``)
    and that is where the MoE expert conversion rules are registered, so the top-level
    ``model_type`` alone is not enough to resolve expert tensor names.

    Keyed by directory rather than by ``SafetensorsIndex`` instance: callers build a fresh
    index per block, so caching on the object would both miss every time and pin every
    index (and its ``weight_map``) in memory for the whole run.
    """
    try:
        with open(Path(checkpoint_dir) / "config.json") as f:
            config = json.load(f)
    except OSError:
        return ()

    found: list[str] = []

    def _walk(node):
        if not isinstance(node, dict):
            return
        model_type = node.get("model_type")
        if isinstance(model_type, str) and model_type not in found:
            found.append(model_type)
        for value in node.values():
            if isinstance(value, dict):
                _walk(value)

    _walk(config)
    return tuple(found)


def _index_model_type(index):
    model_types = _model_types_for_dir(str(getattr(index, "checkpoint_dir", None)))
    return model_types[0] if model_types else None


def _index_model_types(index):
    # Some lightweight/test indices carry no ``checkpoint_dir``; degrade to no
    # known model types rather than raising AttributeError.
    return _model_types_for_dir(str(getattr(index, "checkpoint_dir", None)))


# Model-side per-expert projection -> (fused checkpoint projection, index within it).
# ``gate_proj``/``up_proj`` come from splitting a fused ``gate_up_proj`` in that order,
# which matches the order of the corresponding WeightConverter ``source_patterns``.
_MODEL_PROJ_TO_FUSED = {
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
    "down_proj": ("down_proj", 0),
}

_MODEL_SIDE_EXPERT_RE = re.compile(
    r"^(?P<prefix>.*\.experts)\.(?P<expert>\d+)\.(?P<proj>[A-Za-z0-9_]+)\.(?P<attr>weight|bias)$"
)


@lru_cache(maxsize=None)
def _expert_projection_renames_for(model_type):
    """Map a fused projection to the checkpoint-side per-expert projection names.

    ``transformers`` describes MoE experts with a ``WeightConverter`` whose
    ``source_patterns`` are the per-expert checkpoint keys and whose single
    ``target_pattern`` is the fused model-side parameter, e.g. for phimoe::

        source_patterns = [".experts.*.w1.weight", ".experts.*.w3.weight"]
        target_patterns = ".experts.gate_up_proj"

    After AutoRound unfuses the experts the model-side names are
    ``experts.<i>.gate_proj.weight`` / ``experts.<i>.up_proj.weight``, so we need
    ``(gate_up_proj, 0) -> "w1.weight"`` and ``(gate_up_proj, 1) -> "w3.weight"`` to find
    the tensors back on disk. Returns a tuple of ``((fused, index), suffix)`` pairs.
    """
    if not model_type:
        return ()
    try:
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
        from transformers.core_model_loading import MergeModulelist, WeightConverter
    except ImportError:  # pragma: no cover - transformers < 5
        return ()

    mapping = get_checkpoint_conversion_mapping(model_type)
    if not mapping:
        return ()

    renames: dict[tuple[str, int], str] = {}
    for entry in mapping:
        if not isinstance(entry, WeightConverter):
            continue
        # Only expert merges (`experts.*.<proj>` -> fused 3D) are relevant here.
        if not any(isinstance(op, MergeModulelist) for op in entry.operations):
            continue
        if len(entry.target_patterns) != 1:
            continue
        fused = entry.target_patterns[0].rstrip("$").rsplit(".", 1)[-1]
        for position, source_pattern in enumerate(entry.source_patterns):
            # ".experts.*.w1.weight" -> "w1.weight"
            _, star, suffix = source_pattern.partition("*.")
            if not star:
                continue
            renames.setdefault((fused, position), suffix.rstrip("$"))
    return tuple(renames.items())


def _renamed_expert_candidates(index, full_name: str):
    """Checkpoint-name candidates for an unfused per-expert parameter."""
    match = _MODEL_SIDE_EXPERT_RE.match(full_name)
    if match is None:
        return

    fused_position = _MODEL_PROJ_TO_FUSED.get(match["proj"])
    if fused_position is None:
        return
    prefix, expert = match["prefix"], match["expert"]

    for model_type in _index_model_types(index):
        for key, suffix in _expert_projection_renames_for(model_type):
            if key != fused_position:
                continue
            # "w1.weight" keeps the checkpoint's own attribute name; drop it for a bias.
            suffix_attr = suffix.rsplit(".", 1)
            if len(suffix_attr) == 2:
                suffix = f"{suffix_attr[0]}.{match['attr']}"
            yield f"{prefix}.{expert}.{suffix}"


# On-disk layout of fused expert tensors, keyed by checkpoint ``model_type``.
# Two axes vary between families:
#   * ``checkpoint_transposed`` -- the per-expert 2-D slice is stored as
#     ``[in_features, out_features]`` (matmul layout, ``x @ W``) instead of the
#     ``nn.Linear`` ``[out_features, in_features]`` weight layout, so it must be
#     transposed before being loaded into the unfused per-expert Linear.
#   * ``gate_up_interleaved`` -- the fused ``gate_up_proj`` interleaves the gate
#     and up columns (``gate = fused[..., 0::2]``, ``up = fused[..., 1::2]``)
#     instead of storing them as two contiguous halves.
# The transpose axis is normally inferred from the *target* parameter shape and
# does not need this table; the table is only consulted (a) to pick the
# interleave axis, which is invisible to shapes, and (b) as a tiebreak for the
# rare square/ambiguous shapes where transposed and non-transposed slices share
# the same shape. Families not listed here default to the non-transposed,
# contiguous ``[N, 2*inter, hidden]`` layout (qwen3_moe / phimoe / mixtral-style).
_FUSED_EXPERT_LAYOUTS = {
    # model_type: (checkpoint_transposed, gate_up_interleaved)
    "gpt_oss": (True, True),
    "llama4": (True, False),
}


def _fused_expert_layout(index) -> tuple[bool, bool]:
    """Return the ``(checkpoint_transposed, gate_up_interleaved)`` hint for ``index``.

    The transpose flag is only a *hint* (shape inference takes precedence); the
    interleave flag is authoritative since it cannot be recovered from shapes.
    Unknown model types default to ``(False, False)``.
    """
    for model_type in _index_model_types(index):
        if model_type in _FUSED_EXPERT_LAYOUTS:
            return _FUSED_EXPERT_LAYOUTS[model_type]
    return (False, False)


def _slice_fused_expert(fused, proj, attr, is_gate_up, target_shape, transposed_hint, interleaved):
    """Slice one unfused per-expert tensor out of a fused per-expert tensor.

    ``fused`` is a single expert's slice of the fused checkpoint tensor (2-D for a
    weight, 1-D for a bias). ``target_shape`` is the shape the unfused
    ``nn.Linear`` parameter expects. Whether the checkpoint stores the weight
    transposed is inferred from ``target_shape`` first; ``transposed_hint`` only
    breaks ambiguous (square) ties, and ``interleaved`` selects the gate/up split
    order. Returns ``None`` when no slice can produce ``target_shape`` (the caller
    then leaves the parameter on meta and reports an actionable error).
    """
    if attr == "bias":
        # 1-D per-expert bias; only gate/up need splitting.
        if not is_gate_up:
            return fused.contiguous()
        if interleaved:
            return (fused[0::2] if proj == "gate_proj" else fused[1::2]).contiguous()
        half = fused.shape[0] // 2
        return (fused[:half] if proj == "gate_proj" else fused[half:]).contiguous()

    if fused.ndim != 2:
        return None

    if not is_gate_up:  # down_proj weight, target (out, in)
        reversed_shape = (target_shape[1], target_shape[0])
        if target_shape == reversed_shape:
            transposed = transposed_hint  # square: shape can't decide
        elif tuple(fused.shape) == target_shape:
            transposed = False
        elif tuple(fused.shape) == reversed_shape:
            transposed = True
        else:
            return None
        return fused.t().contiguous() if transposed else fused.contiguous()

    # gate/up weight, target (out, in) = (inter, hidden)
    out, in_features = target_shape
    non_transposed = tuple(fused.shape) == (2 * out, in_features)
    transposed = tuple(fused.shape) == (in_features, 2 * out)
    if non_transposed and transposed:
        transposed = transposed_hint  # square-ish (2*out == in): shape can't decide
        non_transposed = not transposed
    if non_transposed:
        return (fused[:out] if proj == "gate_proj" else fused[out:]).contiguous()
    if transposed:
        # fused is [in_features, 2*out]; split along the 2*out dim, then transpose
        # back to the [out, in_features] nn.Linear weight layout.
        if interleaved:
            sliced = fused[:, 0::2] if proj == "gate_proj" else fused[:, 1::2]
        else:
            sliced = fused[:, :out] if proj == "gate_proj" else fused[:, out:]
        return sliced.t().contiguous()
    return None


_MODEL_SIDE_EXPERT_RE = re.compile(
    r"^(?P<prefix>.*\.experts)\.(?P<expert>\d+)\.(?P<proj>[A-Za-z0-9_]+)\.(?P<attr>weight|bias)$"
)


@lru_cache(maxsize=None)
def _expert_projection_renames_for(model_type):
    """Map a fused projection to the checkpoint-side per-expert projection names.

    ``transformers`` describes MoE experts with a ``WeightConverter`` whose
    ``source_patterns`` are the per-expert checkpoint keys and whose single
    ``target_pattern`` is the fused model-side parameter, e.g. for phimoe::

        source_patterns = [".experts.*.w1.weight", ".experts.*.w3.weight"]
        target_patterns = ".experts.gate_up_proj"

    After AutoRound unfuses the experts the model-side names are
    ``experts.<i>.gate_proj.weight`` / ``experts.<i>.up_proj.weight``, so we need
    ``(gate_up_proj, 0) -> "w1.weight"`` and ``(gate_up_proj, 1) -> "w3.weight"`` to find
    the tensors back on disk. Returns a tuple of ``((fused, index), suffix)`` pairs.
    """
    if not model_type:
        return ()
    try:
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
        from transformers.core_model_loading import MergeModulelist, WeightConverter
    except ImportError:  # pragma: no cover - transformers < 5
        return ()

    mapping = get_checkpoint_conversion_mapping(model_type)
    if not mapping:
        return ()

    renames: dict[tuple[str, int], str] = {}
    for entry in mapping:
        if not isinstance(entry, WeightConverter):
            continue
        # Only expert merges (`experts.*.<proj>` -> fused 3D) are relevant here.
        if not any(isinstance(op, MergeModulelist) for op in entry.operations):
            continue
        if len(entry.target_patterns) != 1:
            continue
        fused = entry.target_patterns[0].rstrip("$").rsplit(".", 1)[-1]
        for position, source_pattern in enumerate(entry.source_patterns):
            # ".experts.*.w1.weight" -> "w1.weight"
            _, star, suffix = source_pattern.partition("*.")
            if not star:
                continue
            renames.setdefault((fused, position), suffix.rstrip("$"))
    return tuple(renames.items())


@lru_cache(maxsize=None)
def _concat_converters_for(model_type):
    """Model-side params assembled by concatenating several checkpoint tensors.

    Some families describe a single model-side parameter as the concatenation of
    several separately-stored checkpoint tensors, via a ``WeightConverter`` whose
    ``operations`` contain a ``Concatenate`` (and *no* ``MergeModulelist`` -- that is the
    expert-fusion case, handled elsewhere). GLM-5.3-Flash (``glm5_next``) is the motivating
    example: ``self_attn.conv1d.weight`` is ``cat([q_conv1d.weight, k_conv1d.weight,
    v_conv1d.weight], dim=0)``. The meta skeleton only knows the fused model-side name, so
    map it back to its checkpoint-side sources to materialize it.

    Returns a tuple of ``(target_suffix, (source_suffixes...), concat_dim)`` triples, using
    only literal (wildcard-free) patterns -- wildcard/expert converters are handled by the
    expert-specific helpers.
    """
    if not model_type:
        return ()
    try:
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
        from transformers.core_model_loading import Concatenate, MergeModulelist, WeightConverter
    except ImportError:  # pragma: no cover - transformers < 5
        return ()

    mapping = get_checkpoint_conversion_mapping(model_type)
    if not mapping:
        return ()

    converters = []
    for entry in mapping:
        if not isinstance(entry, WeightConverter):
            continue
        # Expert fusions (MergeModulelist) are handled by `_renamed_expert_candidates`
        # and `materialize_module`'s fused lookup.
        if any(isinstance(op, MergeModulelist) for op in entry.operations):
            continue
        concat = next((op for op in entry.operations if isinstance(op, Concatenate)), None)
        if concat is None or len(entry.target_patterns) != 1:
            continue
        target = entry.target_patterns[0].rstrip("$")
        sources = tuple(src.rstrip("$") for src in entry.source_patterns)
        # Only literal patterns here; wildcard (`*`) converters are per-expert.
        if "*" in target or any("*" in src for src in sources):
            continue
        converters.append((target, sources, getattr(concat, "dim", 0)))
    return tuple(converters)


def _renamed_expert_candidates(index, full_name: str):
    """Checkpoint-name candidates for an unfused per-expert parameter."""
    match = _MODEL_SIDE_EXPERT_RE.match(full_name)
    if match is None:
        return

    fused_position = _MODEL_PROJ_TO_FUSED.get(match["proj"])
    if fused_position is None:
        return
    prefix, expert = match["prefix"], match["expert"]

    for model_type in _index_model_types(index):
        for key, suffix in _expert_projection_renames_for(model_type):
            if key != fused_position:
                continue
            # "w1.weight" keeps the checkpoint's own attribute name; drop it for a bias.
            suffix_attr = suffix.rsplit(".", 1)
            if len(suffix_attr) == 2:
                suffix = f"{suffix_attr[0]}.{match['attr']}"
            yield f"{prefix}.{expert}.{suffix}"


def _resolve_checkpoint_name(index, full_name: str):
    """Map a model-side parameter name to its checkpoint-side tensor name."""
    if index.has_tensor(full_name):
        return full_name
    # Unfused MoE experts whose checkpoint spells the projections differently
    # (e.g. phimoe's w1/w2/w3, mixtral-style checkpoints).
    for candidate in _renamed_expert_candidates(index, full_name):
        if index.has_tensor(candidate):
            return candidate
    for model_type in _index_model_types(index):
        for transform in _reverse_renamings_for(model_type):
            try:
                candidate, _ = transform.rename_source_key(full_name)
            except Exception as exc:  # pragma: no cover - defensive, never fail a load here
                logger.debug("rename %s via %s failed: %s", full_name, transform, exc)
                continue
            if candidate != full_name and index.has_tensor(candidate):
                return candidate
    return None


def materialize_module(module: nn.Module, module_name: str, index: SafetensorsIndex, device: str) -> None:
    """Populate `module`'s (currently meta) parameters/buffers with real data read
    directly from the checkpoint, onto `device`. `module_name` is `module`'s dotted
    path in the full model (used as the tensor-name prefix in the checkpoint).

    AutoScheme's scoring
    wraps quantized layers in ``AutoSchemeWrapperLinear``, which replaces a plain
    ``nn.Linear`` with a wrapper holding the real layer at ``.orig_layer`` --
    inserting an extra ``.orig_layer`` path segment that doesn't exist in the
    checkpoint's own tensor names. Strip it back out before looking up the name.
    """
    import re as _re

    # Fused-MoE replacement modules
    # (SequentialQwen3_5MoeExperts and friends) expose UNFUSED per-expert
    # parameter names (experts.{i}.gate_proj.weight ...) that don't exist in a
    # checkpoint whose on-disk layout is the fused 3D one
    # (experts.gate_up_proj [N, 2*inter, hidden] / experts.down_proj). The
    # compressor's own tuning loop handles this via OffloadManager.reload +
    # materialize_model_, but every bare materialize_module() consumer
    # (AutoScheme's delta_loss scoring streams, stream_block_forward for the
    # calibration/eval forwards) previously left those params on meta -- the
    # meta-ness then propagated silently until a crash far downstream. Map
    # each unfused name onto its fused on-disk tensor and slice.
    #
    # The exact slicing depends on the family's fused layout: qwen3_moe stores
    # ``[N, 2*inter, hidden]`` (contiguous halves, no transpose), while gpt_oss /
    # llama4 store ``[N, hidden, 2*inter]`` (matmul layout, needs transpose) and
    # gpt_oss additionally interleaves the gate/up columns. Rather than key the
    # whole decision off ``model_type`` (which does not generalise to new
    # families), the transpose is inferred from the *target* parameter shape and
    # only the interleave axis (invisible to shapes) falls back to a small
    # model_type hint table (see ``_slice_fused_expert``). Biases (gpt_oss) follow
    # the same split as their weight.
    _FUSED_RE = _re.compile(r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|bias)$")
    _fused_cache: dict = {}
    _layout_cache: list = []  # lazily-resolved (transposed_hint, interleaved)

    def _fused_lookup(full_name: str, target_shape):
        m = _FUSED_RE.match(full_name)
        if not m:
            return None
        prefix, expert_idx, proj, attr = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        is_gate_up = proj in ("gate_proj", "up_proj")
        if is_gate_up:
            fused_name = f"{prefix}.gate_up_proj" if attr == "weight" else f"{prefix}.gate_up_proj_bias"
        else:
            fused_name = f"{prefix}.down_proj" if attr == "weight" else f"{prefix}.down_proj_bias"
        if not index.has_tensor(fused_name):
            return None
        if fused_name not in _fused_cache:
            _fused_cache[fused_name] = index.read_tensors([fused_name], device=device)[fused_name]
        fused = _fused_cache[fused_name][expert_idx]
        # Resolve the model_type layout hint lazily and only once, and only when a
        # fused expert tensor is actually being sliced -- reading the model type
        # touches ``index.checkpoint_dir``/config, which cheap/fake indices used by
        # non-fused (dense) materialization do not populate.
        if not _layout_cache:
            _layout_cache.append(_fused_expert_layout(index))
        transposed_hint, interleaved = _layout_cache[0]
        value = _slice_fused_expert(fused, proj, attr, is_gate_up, tuple(target_shape), transposed_hint, interleaved)
        # Guard against a slice that does not match the destination parameter: it
        # is safer to leave the param on meta (the caller then raises an
        # actionable "fall back to a full CPU load" error) than to hand accelerate
        # a mismatched tensor and crash with a cryptic ValueError.
        if value is not None and tuple(value.shape) != tuple(target_shape):
            return None
        return value

    def _concat_lookup(full_name: str):
        """Assemble a model-side param that the checkpoint stores as several tensors
        concatenated by a ``WeightConverter`` (e.g. GLM-5.3 ``q/k/v_conv1d`` ->
        ``conv1d``). Returns the concatenated tensor, or ``None`` when no converter
        applies or a source tensor is missing."""
        for model_type in _index_model_types(index):
            for target_suffix, source_suffixes, dim in _concat_converters_for(model_type):
                if not full_name.endswith(target_suffix):
                    continue
                prefix = full_name[: -len(target_suffix)]
                # Guard against a spurious mid-token match (require a clean boundary).
                if prefix and not prefix.endswith("."):
                    continue
                source_names = [f"{prefix}{suffix}" for suffix in source_suffixes]
                if not all(index.has_tensor(name) for name in source_names):
                    continue
                read = index.read_tensors(source_names, device=device)
                return torch.cat([read[name] for name in source_names], dim=dim).contiguous()
        return None

    targets = []  # (param_name, full_checkpoint_name, declared_meta_dtype)
    fused_targets = []  # (param_name, sliced_value)
    for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        if str(tensor.device) != "meta":
            continue  # already materialized (e.g. shared/tied weights)
        full_name = f"{module_name}.{name}".replace(".orig_layer.", ".")
        resolved_name = _resolve_checkpoint_name(index, full_name)
        if resolved_name is None:
            sliced = _fused_lookup(full_name, tensor.shape)
            if sliced is None:
                sliced = _concat_lookup(full_name)
            if sliced is not None:
                fused_targets.append((name, sliced))
                continue
            logger.warning("No checkpoint tensor found for %s, leaving on meta", full_name)
            continue
        targets.append((name, resolved_name, tensor.dtype))

    for name, value in fused_targets:
        try:
            set_module_tensor_to_device(module, name, device, value=value, dtype=value.dtype)
        except (ValueError, RuntimeError) as exc:
            # A fused-expert slice whose shape does not match its destination
            # cannot be reconciled from the checkpoint; rather than surface the
            # raw accelerate error, point the user at the non-meta fallback.
            raise RuntimeError(
                f"Failed to materialize fused MoE parameter '{module_name}.{name}' from the checkpoint "
                f"(value shape {tuple(value.shape)}). This usually means the model's fused-expert layout "
                "is not yet recognised by the meta-skeleton loader. Set AR_DISABLE_AUTO_META_LOAD=1 to load "
                "the whole model on CPU instead (uses more RAM)."
            ) from exc
    _fused_cache.clear()

    if not targets:
        return
    values = index.read_tensors([full_name for _, full_name, _ in targets], device=device)
    for name, full_name, declared_dtype in targets:
        # Prefer the meta parameter's already-declared dtype: it reflects
        # whatever compute dtype the caller already promoted the (still-meta)
        # model to (e.g. ModelContext._set_amp_dtype()'s `model.to(amp_dtype)`),
        # and materializing to a different dtype than sibling non-block params
        # that were promoted while still real breaks ops mixing the two (e.g.
        # LayerNorm on bf16 activations with fp16 weight/bias). The one
        # exception: a meta skeleton built without an enclosing dtype context
        # (e.g. Qwen3_5MoeExperts' per-expert Linears, built under
        # `torch.device("meta")` alone) defaults the declared dtype to
        # float32 regardless of the checkpoint's real dtype -- found via a
        # real 8-layer MoE fixture crashing with "expected m1 and m2 to have
        # the same dtype" inside AutoScheme scoring. Detect that case (declared
        # float32 but checkpoint isn't) and fall back to the checkpoint's own
        # dtype instead.
        target_dtype = declared_dtype
        if declared_dtype == torch.float32 and values[full_name].dtype != torch.float32:
            target_dtype = values[full_name].dtype
        set_module_tensor_to_device(module, name, device, value=values[full_name], dtype=target_dtype)


def free_module(module: nn.Module) -> None:
    """Release a module's real tensors back to the meta device, freeing memory."""
    for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        if str(tensor.device) == "meta":
            continue
        set_module_tensor_to_device(module, name, "meta")


def total_resident_bytes(model: nn.Module) -> int:
    """Debug helper: sum the byte size of every non-meta parameter/buffer in
    `model`. Used to diagnose whether blocks are genuinely returning to meta
    after free_module(), or something else is holding real memory."""
    total = 0
    for _, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if str(tensor.device) != "meta":
            total += tensor.numel() * tensor.element_size()
    return total


def unfuse_meta_moe_(model: nn.Module) -> list[str]:
    """Turn ``transformers>=5`` fused 3D expert parameters into per-expert ``nn.Linear``.

    ``transformers>=5`` declares MoE experts as a single fused ``nn.Parameter`` of shape
    ``(num_experts, ...)``, which AutoRound cannot quantize (it is not an ``nn.Linear``).
    The regular load pipeline fixes this *after* a full CPU load; on a meta skeleton we
    can do it up-front for free -- meta tensors own no storage, so unfusing here only
    rewires ``nn.Module`` objects and costs ~0 RAM.

    Doing it before any weight is read is also what keeps the later per-block
    materialization cheap: the model-side names become ``experts.<i>.<proj>.weight``,
    which is exactly the layout the checkpoints use, so ``materialize_module`` reads one
    small 2D tensor per expert instead of forcing a whole fused ``(num_experts, ...)``
    tensor to be assembled in RAM.

    No modeling class is monkey-patched: the per-expert forward is dispatched through
    ``transformers.integrations.moe.ALL_EXPERTS_FUNCTIONS`` (the public experts-interface
    registry) via ``config._experts_implementation``.

    Returns the list of unfused module names (empty when there is nothing to do).
    """
    try:
        from auto_round.modeling.fused_moe.replace_modules import (
            _handle_moe_modules,
            is_custom_model,
            log_moe_block_transition,
        )
    except ImportError:  # pragma: no cover - transformers < 5 or partial install
        return []

    # Model families with a dedicated replacement (llama4, qwen3_5_moe, step3p5, ...) must
    # go through `apply_replacements`, which runs its custom `from_original` against the
    # *fused* module. Unfusing them here with the generic linear_loop path would leave
    # that replacement facing a module whose fused params no longer exist.
    if is_custom_model(model):
        logger.debug(
            "meta skeleton: %s has a dedicated MoE replacement, deferring the unfuse to apply_replacements",
            getattr(getattr(model, "config", None), "model_type", None),
        )
        return []

    try:
        # This is where the structural change actually happens for a meta-built model, so
        # report the before/after here rather than in `apply_replacements` (which by then
        # has nothing left to do).
        with log_moe_block_transition(model, "unfuse"):
            unfused = _handle_moe_modules(model)
    except Exception:  # pragma: no cover - never block the meta build on this
        logger.warning("Structural MoE unfuse on the meta skeleton failed", exc_info=True)
        return []

    if unfused:
        logger.info("meta skeleton: structural MoE unfuse produced %d unfused experts modules", len(unfused))
    return unfused


def build_meta_model(model_name: str, trust_remote_code: bool = True, unfuse_moe: bool = True):
    """Build a meta-device model skeleton (~0 RAM) plus its tokenizer and a
    SafetensorsIndex for on-demand materialization, instead of AutoRound's own
    ``llm_load_model(model_name, device_map="cpu")`` which fully materializes the
    checkpoint on CPU RAM in one shot. Deliberately narrower than ``llm_load_model``:
    only covers the common local-directory ``AutoModelForCausalLM`` case (no
    bagel/glm/mxfp4/HPU special-casing) -- callers should fall back to
    ``llm_load_model`` for anything this doesn't handle.

    When ``unfuse_moe`` is set (the default) the fused ``transformers>=5`` MoE experts
    are split into per-expert ``nn.Linear`` while still on meta, so the returned skeleton
    is immediately quantizable and every later materialization is per-expert.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    # Prefer the exact class named in config.architectures: AutoModelForCausalLM
    # cannot resolve multimodal architectures (e.g.
    # Qwen3_5MoeForConditionalGeneration), which previously forced VLM
    # checkpoints down the full-CPU-load path. Same resolution strategy as
    # reap/layerwise_prune.py's disk-streamed model builder.
    import transformers as _transformers

    archs = getattr(config, "architectures", None) or []
    model_cls = next((getattr(_transformers, a) for a in archs if hasattr(_transformers, a)), None)
    with init_empty_weights():
        if model_cls is not None:
            model = model_cls(config)
        else:
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)

    # Unfuse while everything is still meta: free, and it makes the model-side
    # parameter names line up 1:1 with the per-expert checkpoint tensors.
    if unfuse_moe:
        unfuse_meta_moe_(model)

    # Constructing from config does not tie weights (that happens in
    # from_pretrained), so tied params such as lm_head.weight -- absent from
    # the checkpoint by design -- stay separate meta tensors and later crash
    # with "Cannot copy out of meta tensor" on the first device move. Tie
    # here: the shared tensor materializes once from the checkpoint side it
    # is stored under and both names become real together.
    if getattr(model.config, "tie_word_embeddings", False):
        model.tie_weights()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    index = SafetensorsIndex(model_name)
    return model, tokenizer, index


def materialize_non_block_params(
    model: nn.Module, block_prefixes: list[str], index: SafetensorsIndex, device: str
) -> None:
    """Materialize every real (non-meta) parameter/buffer NOT under one of
    ``block_prefixes`` -- i.e. embeddings, final norm, lm_head, and similar small
    top-level modules -- leaving the (typically 100+GB combined) decoder blocks on
    meta for later per-block materialize/free. These non-block modules are needed
    continuously throughout scoring and are comparatively small even for large
    vocabularies, so it's simplest to load them once, for real, up front.
    """

    def _in_block(name: str) -> bool:
        return any(name == p or name.startswith(p + ".") for p in block_prefixes)

    # Tied output embeddings (lm_head.weight tied to embed_tokens.weight) are absent from
    # the checkpoint by design -- the tie is re-established below, which makes them real.
    # Do not warn about those, or every tied model reports a scary "leaving on meta" line
    # for a parameter that is fine a few statements later.
    tied_keys = set(getattr(model, "_tied_weights_keys", None) or ())

    def _is_tied(name: str) -> bool:
        return name in tied_keys or any(name.startswith(key.rstrip("*")) for key in tied_keys if key.endswith("*"))

    targets = []  # (param_name, full_checkpoint_name)
    deferred_tied = []
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if str(tensor.device) != "meta" or _in_block(name):
            continue
        full_name = name.replace(".orig_layer.", ".")
        resolved_name = _resolve_checkpoint_name(index, full_name)
        if resolved_name is None:
            if _is_tied(name):
                deferred_tied.append(name)
            else:
                logger.warning("No checkpoint tensor found for %s, leaving on meta", full_name)
            continue
        targets.append((name, resolved_name))

    if targets:
        values = index.read_tensors([full_name for _, full_name in targets], device=device)
        for name, full_name in targets:
            # See the matching comment in materialize_module() -- explicit dtype=
            # is required so the checkpoint's real dtype wins over whatever the
            # meta skeleton happened to declare.
            set_module_tensor_to_device(model, name, device, value=values[full_name], dtype=values[full_name].dtype)

    # set_module_tensor_to_device replaces parameter objects, which breaks
    # weight tying: a tied lm_head still references the old (meta) parameter
    # while the embedding points at the new real one. Re-tie so every tied
    # name follows its checkpoint-backed source tensor.
    if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        model.tie_weights()

    # Anything declared tied that the re-tie did not actually make real is a genuine
    # problem (quantizing it would fail on a meta tensor), so surface it now.
    still_meta = [name for name in deferred_tied if _param_or_buffer(model, name) is None]
    if still_meta:
        logger.warning("No checkpoint tensor found for tied %s, leaving on meta", ", ".join(still_meta))


def _param_or_buffer(model: nn.Module, name: str):
    """Return the named parameter/buffer when it is materialized, else ``None``."""
    for candidate_name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if candidate_name == name:
            return None if tensor.device.type == "meta" else tensor
    return None


class stream_block_forward:
    """Context manager: wrap every top-level decoder block's ``forward`` so it
    materializes its own real weights from ``index`` right before running and
    frees them back to meta right after -- letting a plain ``model(...)`` call
    (e.g. for computing held-out loss) drive the model exactly as normal while
    only ever one block's weights are resident at a time.

    Deliberately much lighter than the auto_scheme delta_loss.py streaming
    forward it's modeled on (``prepare_model_low_gpu``/``model_forward_low_gpu``):
    no input-caching for later backward replay, no grad-mode bookkeeping -- this
    is for a plain inference-only forward pass (e.g. eval loss), not tuning.
    """

    def __init__(self, model: nn.Module, index: SafetensorsIndex, device: str, block_names: list[str] = None):
        self.model = model
        self.index = index
        self.device = device
        self.block_names = block_names if block_names is not None else _default_block_names(model)
        self._originals: Dict[str, "callable"] = {}

    def __enter__(self):
        for block_name in self.block_names:
            module = _get_module(self.model, block_name)
            self._originals[block_name] = module.forward

            def make_wrapped(module=module, block_name=block_name, original_forward=module.forward):
                def wrapped(*args, **kwargs):
                    materialize_module(module, block_name, self.index, device=self.device)
                    try:
                        return original_forward(*args, **kwargs)
                    finally:
                        free_module(module)

                return wrapped

            module.forward = make_wrapped()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for block_name, original_forward in self._originals.items():
            _get_module(self.model, block_name).forward = original_forward
        return False


def _default_block_names(model: nn.Module) -> list[str]:
    from auto_round.utils import get_block_names

    return get_block_names(model)[0]


def _get_module(model: nn.Module, name: str) -> nn.Module:
    from auto_round.utils import get_module

    return get_module(model, name)
