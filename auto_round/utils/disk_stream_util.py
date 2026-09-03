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
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from accelerate.utils import set_module_tensor_to_device
from safetensors import safe_open

from auto_round import envs

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
        closed (unmapped) once regardless of how many tensors are pulled from it.

        Reads are issued from a small thread pool. Unfused MoE blocks ask for hundreds of
        small per-expert tensors at a time (256 experts x 3 projections), and issuing those
        copies one after another leaves the device queue idle between requests -- the
        checkpoint bytes are the same either way, but a cold block takes far longer. The
        safetensors reader releases the GIL during ``get_tensor``, so this actually
        overlaps. Each worker opens its own handle and closes it before returning, so the
        no-cached-mmap guarantee above still holds.
        """
        if not names:
            return {}

        by_shard: Dict[str, list[str]] = {}
        for name in names:
            by_shard.setdefault(self.weight_map[name], []).append(name)

        workers = max(1, int(envs.AR_DISK_STREAM_WORKERS))
        result: Dict[str, torch.Tensor] = {}

        for shard_name, shard_tensor_names in by_shard.items():
            # One handle per shard: opening is not free (the header of a large shard lists
            # every tensor in it), so re-opening per worker would cost more than the
            # overlap buys. The handle is closed before moving on, keeping the
            # no-cached-mmap guarantee documented above.
            with safe_open(str(self.checkpoint_dir / shard_name), framework="pt") as f:

                def _read(name: str):
                    tensor = f.get_tensor(name)
                    return name, (tensor.to(device) if device != "cpu" else tensor)

                if workers <= 1 or len(shard_tensor_names) <= 1:
                    result.update(dict(_read(name) for name in shard_tensor_names))
                else:
                    with ThreadPoolExecutor(max_workers=min(workers, len(shard_tensor_names))) as pool:
                        result.update(dict(pool.map(_read, shard_tensor_names)))
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
    ``PrefixChange`` special cases correctly. Naively re-using a ``target_pattern`` as a
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
    model_types = _model_types_for_dir(str(index.checkpoint_dir))
    return model_types[0] if model_types else None


def _index_model_types(index):
    return _model_types_for_dir(str(index.checkpoint_dir))


# Model-side per-expert projection -> (fused checkpoint projection, index within it).
# ``gate_proj``/``up_proj`` come from splitting a fused ``gate_up_proj`` in that order,
# which matches the order of the corresponding WeightConverter ``source_patterns``.
_MODEL_PROJ_TO_FUSED = {
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
    "down_proj": ("down_proj", 0),
}

_MODEL_SIDE_EXPERT_RE = re.compile(r"^(?P<prefix>.*\.experts)\.(?P<expert>\d+)\.(?P<proj>[A-Za-z0-9_]+)\.(?P<attr>weight|bias)$")

# Report the first block materialization at INFO (see `materialize_module`).
_logged_first_materialize = False


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
    _FUSED_RE = _re.compile(r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
    _fused_cache: dict = {}

    def _fused_lookup(full_name: str):
        m = _FUSED_RE.match(full_name)
        if not m:
            return None
        prefix, expert_idx, proj = m.group(1), int(m.group(2)), m.group(3)
        fused_name = f"{prefix}.gate_up_proj" if proj in ("gate_proj", "up_proj") else f"{prefix}.down_proj"
        if not index.has_tensor(fused_name):
            return None
        if fused_name not in _fused_cache:
            _fused_cache[fused_name] = index.read_tensors([fused_name], device=device)[fused_name]
        fused = _fused_cache[fused_name][expert_idx]
        if proj == "down_proj":
            return fused.contiguous()
        inter = fused.shape[0] // 2
        return (fused[:inter] if proj == "gate_proj" else fused[inter:]).contiguous()

    targets = []  # (param_name, full_checkpoint_name, declared_meta_dtype)
    fused_targets = []  # (param_name, sliced_value)
    _t_start = time.perf_counter()
    for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        if str(tensor.device) != "meta":
            continue  # already materialized (e.g. shared/tied weights)
        full_name = f"{module_name}.{name}".replace(".orig_layer.", ".")
        resolved_name = _resolve_checkpoint_name(index, full_name)
        if resolved_name is None:
            sliced = _fused_lookup(full_name)
            if sliced is not None:
                fused_targets.append((name, sliced))
                continue
            logger.warning("No checkpoint tensor found for %s, leaving on meta", full_name)
            continue
        targets.append((name, resolved_name, tensor.dtype))
    _t_resolve = time.perf_counter()

    for name, value in fused_targets:
        set_module_tensor_to_device(module, name, device, value=value, dtype=value.dtype)
    _fused_cache.clear()

    if not targets:
        return
    values = index.read_tensors([full_name for _, full_name, _ in targets], device=device)
    _t_read = time.perf_counter()
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

    _t_assign = time.perf_counter()
    total_bytes = sum(t.numel() * t.element_size() for t in values.values())
    # The first block is the interesting one: it pays the cold-cache read plus any
    # one-off setup, and is what makes the first `Quantizing layers.0` step look slow
    # compared to a fully-resident model. Report it at INFO so it needs no debug logging,
    # then drop to DEBUG for the remaining blocks.
    global _logged_first_materialize
    log = logger.debug if _logged_first_materialize else logger.info
    _logged_first_materialize = True
    log(
        "materialize %s: %d tensors / %.2f GB in %.2fs "
        "(resolve %.2fs, read %.2fs @ %.0f MB/s, assign %.2fs)",
        module_name,
        len(targets),
        total_bytes / 1024**3,
        _t_assign - _t_start,
        _t_resolve - _t_start,
        _t_read - _t_resolve,
        (total_bytes / 1024**2) / max(_t_read - _t_resolve, 1e-6),
        _t_assign - _t_read,
    )


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
        from auto_round.modeling.fused_moe.replace_modules import _handle_moe_modules, is_custom_model
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

    targets = []  # (param_name, full_checkpoint_name)
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if str(tensor.device) != "meta" or _in_block(name):
            continue
        full_name = name.replace(".orig_layer.", ".")
        resolved_name = _resolve_checkpoint_name(index, full_name)
        if resolved_name is None:
            logger.warning("No checkpoint tensor found for %s, leaving on meta", full_name)
            continue
        targets.append((name, resolved_name))

    if not targets:
        return
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
