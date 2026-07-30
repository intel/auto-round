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
    for name, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        if str(tensor.device) != "meta":
            continue  # already materialized (e.g. shared/tied weights)
        full_name = f"{module_name}.{name}".replace(".orig_layer.", ".")
        if not index.has_tensor(full_name):
            sliced = _fused_lookup(full_name)
            if sliced is not None:
                fused_targets.append((name, sliced))
                continue
            logger.warning("No checkpoint tensor found for %s, leaving on meta", full_name)
            continue
        targets.append((name, full_name, tensor.dtype))

    for name, value in fused_targets:
        set_module_tensor_to_device(module, name, device, value=value, dtype=value.dtype)
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


def build_meta_model(model_name: str, trust_remote_code: bool = True):
    """Build a meta-device model skeleton (~0 RAM) plus its tokenizer and a
    SafetensorsIndex for on-demand materialization, instead of AutoRound's own
    ``llm_load_model(model_name, device_map="cpu")`` which fully materializes the
    checkpoint on CPU RAM in one shot. Deliberately narrower than ``llm_load_model``:
    only covers the common local-directory ``AutoModelForCausalLM`` case (no
    bagel/glm/mxfp4/HPU special-casing) -- callers should fall back to
    ``llm_load_model`` for anything this doesn't handle.
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
        if not index.has_tensor(full_name):
            logger.warning("No checkpoint tensor found for %s, leaving on meta", full_name)
            continue
        targets.append((name, full_name))

    if not targets:
        return
    values = index.read_tensors([full_name for _, full_name in targets], device=device)
    for name, full_name in targets:
        # See the matching comment in materialize_module() -- explicit dtype=
        # is required so the checkpoint's real dtype wins over whatever the
        # meta skeleton happened to declare.
        set_module_tensor_to_device(model, name, device, value=values[full_name], dtype=values[full_name].dtype)


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
