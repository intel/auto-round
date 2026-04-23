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

"""
MLX Format Exporter for AutoRound

Exports quantized models to MLX-compatible format that can be loaded
directly by mlx-lm for inference on Apple Silicon.

MLX QuantizedLinear dequantization (affine mode):
    w_float = scale * w_int + bias

where w_int is packed into uint32, each holding (32 // bits) elements,
and scale/bias (called "biases" in MLX) have shape [out_features, num_groups].

Supported schemes: W2A16, W3A16, W4A16, W8A16
"""

import copy
import json
import os
from typing import Callable, Union

import torch
import torch.nn as nn
import transformers

from auto_round.export.utils import unsupported_meta_device
from auto_round.logger import logger
from auto_round.utils import (
    check_to_quantized,
    copy_python_files_from_model_cache,
    get_module,
    get_packing_device,
    set_module,
)

SUPPORTED_LAYER_TYPES = [nn.Linear, nn.Conv2d, transformers.pytorch_utils.Conv1D]


def _is_mlx_quantizable(module: nn.Module, group_size: int) -> bool:
    """Predicate matching mlx-lm's default quantization predicate.

    A layer is considered quantizable by mlx-lm if its inner dim is a multiple
    of ``group_size`` and its outer dim is a multiple of 64. Embeddings are also
    eligible. Layers that fail this predicate are skipped at load time, so we do
    NOT need to emit ``false`` entries for them.
    """
    if isinstance(module, nn.Linear):
        in_dim, out_dim = module.in_features, module.out_features
    elif isinstance(module, nn.Embedding):
        in_dim, out_dim = module.embedding_dim, module.num_embeddings
    else:
        return False
    return (in_dim % group_size == 0) and (out_dim % 64 == 0)


# Sub-modules whose layers we never want to auto-quantize on the mlx side.
# Vision / audio towers in VLMs are conventionally kept in fp16 by mlx-vlm,
# so emitting per-layer ``false`` for hundreds of vision linears just bloats
# the config. We simply skip these subtrees.
_DEFAULT_SKIP_PREFIXES = (
    "vision_tower",
    "vision_model",
    "visual",
    "audio_tower",
    "audio_model",
    "image_newline",
    "multi_modal_projector",
)


def _build_mlx_quantization_config(
    model: nn.Module,
    default_bits: int,
    default_group_size: int,
    skip_prefixes: tuple = _DEFAULT_SKIP_PREFIXES,
) -> dict:
    """Build the ``quantization`` dict for an MLX ``config.json``.

    Mirrors the mlx-community mixed-bit format (see e.g.
    ``mlx-community/GLM-4.7-REAP-50-mixed-3-4-bits``)::

        "quantization": {
            "group_size": 64,
            "bits": 4,
            "model.layers.0.self_attn.q_proj": {"group_size": 64, "bits": 3},
            "model.layers.0.mlp.down_proj": false,
            ...
        }

    * Top-level ``group_size`` / ``bits`` give the default scheme.
    * Per-layer ``{group_size, bits}`` overrides differing layers.
    * Per-layer ``false`` marks an mlx-quantizable layer that should be kept
      in fp16 (e.g. ``lm_head`` left at 16 bits in our ``layer_config``).

    For VLMs, ``skip_prefixes`` lists sub-trees (vision/audio towers, projectors)
    whose un-quantized layers we do NOT emit ``false`` entries for. Quantized
    layers inside those sub-trees are still recorded faithfully.

    Note: AutoRound may leave non-quantized layers wrapped in ``WrapperLinear``
    (``.orig_layer`` points to the real ``nn.Linear``); we unwrap before testing
    so that e.g. ``lm_head`` with ``bits=16`` is correctly emitted as ``false``.
    """
    quant_cfg: dict = {"group_size": default_group_size, "bits": default_bits}

    def _is_skipped(layer_name: str) -> bool:
        return any(
            layer_name == p or layer_name.startswith(p + ".") for p in skip_prefixes
        )

    for name, module in model.named_modules():
        if isinstance(module, _MLXPackedLayer):
            # Always record packed layers, even if inside a skipped sub-tree
            # (user explicitly asked for them to be quantized).
            quant_cfg[name] = {
                "group_size": int(module.group_size) if module.group_size is not None else default_group_size,
                "bits": int(module.bits) if module.bits is not None else default_bits,
            }
            continue

        if _is_skipped(name):
            continue

        # Unwrap auto-round WrapperLinear (and similar) before the predicate test.
        target = module
        while hasattr(target, "orig_layer") and target.orig_layer is not None:
            target = target.orig_layer

        if _is_mlx_quantizable(target, default_group_size):
            # mlx-lm would auto-quantize this layer using the default scheme,
            # but since we deliberately left it un-quantized, mark it explicitly.
            quant_cfg[name] = False
    return quant_cfg


# Sub-config keys commonly used by HF multi-modal / dual-tower configs that
# carry their own ``rope_parameters`` and may want a ``quantization`` block.
_TEXT_SUB_CONFIG_KEYS = ("text_config", "language_config", "thinker_config")


def _flatten_rope_parameters_recursive(cfg: dict) -> None:
    """Pop ``rope_parameters`` and surface its keys to the same level.

    mlx-lm / mlx-vlm both expect ``rope_theta`` (and friends) at the same level
    as the rest of the model config — i.e. on the top-level config for LLMs and
    on ``text_config`` for VLMs. We walk every nested dict so both cases work.

    Two ``rope_parameters`` shapes appear in the wild and both are handled:
      1. ``{"rope_theta": 1e6, "rope_type": "default", ...}`` (flat)
      2. ``{"default": {"rope_theta": 1e6, "rope_type": "default", ...}, ...}``
         (newer HF "by-mode" layout — we surface keys from the ``"default"``
         entry, falling back to the first sub-dict if absent).
    """
    if not isinstance(cfg, dict):
        return
    rope_params = cfg.pop("rope_parameters", None)
    if isinstance(rope_params, dict):
        # If the dict is nested by mode, prefer "default", else any sub-dict.
        sub_dicts = {k: v for k, v in rope_params.items() if isinstance(v, dict)}
        if sub_dicts:
            chosen = sub_dicts.get("default") or next(iter(sub_dicts.values()))
            for k, v in chosen.items():
                cfg.setdefault(k, v)
        else:
            for k, v in rope_params.items():
                cfg.setdefault(k, v)
    for v in cfg.values():
        if isinstance(v, dict):
            _flatten_rope_parameters_recursive(v)


def _extract_rope_theta_from_obj(config_obj) -> "float | None":
    """Best-effort extraction of ``rope_theta`` from a HF config object.

    Newer transformers versions stop exposing ``rope_theta`` as a direct
    attribute on the config and only expose ``rope_parameters`` (which can be
    a dict, a typed object, or a by-mode mapping like
    ``{"default": <obj-or-dict>}``). This helper tries every shape we've
    encountered in the wild.
    """
    if config_obj is None:
        return None
    direct = getattr(config_obj, "rope_theta", None)
    if direct is not None:
        return direct

    rope_params = getattr(config_obj, "rope_parameters", None)
    if rope_params is None:
        return None

    # Case A: dict-like
    if isinstance(rope_params, dict):
        if "rope_theta" in rope_params:
            return rope_params["rope_theta"]
        # by-mode: {"default": {...}, ...}
        sub_dicts = [v for v in rope_params.values() if isinstance(v, (dict,)) or hasattr(v, "rope_theta")]
        chosen = None
        if "default" in rope_params and (
            isinstance(rope_params["default"], dict) or hasattr(rope_params["default"], "rope_theta")
        ):
            chosen = rope_params["default"]
        elif sub_dicts:
            chosen = sub_dicts[0]
        if chosen is not None:
            return _extract_rope_theta_from_obj(chosen) if not isinstance(chosen, dict) else chosen.get("rope_theta")
        return None

    # Case B: object with attributes
    inner = getattr(rope_params, "rope_theta", None)
    if inner is not None:
        return inner
    # by-mode object: try .default
    inner_default = getattr(rope_params, "default", None)
    if inner_default is not None:
        return _extract_rope_theta_from_obj(inner_default)
    return None


def _ensure_rope_theta_from_config_obj(cfg: dict, config_obj) -> None:
    """Backfill ``rope_theta`` on every level by probing the live HF config object.

    HF model configs may carry ``rope_theta`` only inside ``rope_parameters`` (or
    not at all on the serialized JSON when ``rope_parameters`` is nested by mode
    and the JSON-side flatten couldn't reach a usable value). Walk the config
    object alongside the JSON dict and copy ``rope_theta`` over wherever it's
    still missing — at top level for LLMs, and on
    ``text_config`` / ``language_config`` / ``thinker_config`` for VLMs.
    """
    if not isinstance(cfg, dict) or config_obj is None:
        return
    if "rope_theta" not in cfg:
        rope_theta = _extract_rope_theta_from_obj(config_obj)
        if rope_theta is not None:
            cfg["rope_theta"] = rope_theta
    for sub_key in _TEXT_SUB_CONFIG_KEYS + ("vision_config", "audio_config"):
        sub_cfg = cfg.get(sub_key)
        sub_obj = getattr(config_obj, sub_key, None)
        if isinstance(sub_cfg, dict) and sub_obj is not None:
            _ensure_rope_theta_from_config_obj(sub_cfg, sub_obj)


def _strip_prefix(name: str, prefix: str) -> str:
    """Return ``name`` with ``prefix.`` removed, or ``name`` if no match."""
    if name == prefix:
        return name
    pref = prefix + "."
    return name[len(pref):] if name.startswith(pref) else name


def _build_text_subconfig_quantization(quant_cfg: dict, text_prefix: str) -> dict:
    """Re-key a top-level quantization dict for placement under ``text_config``.

    mlx-vlm's language-model loader queries ``text_config["quantization"]`` with
    keys *relative to the language model* (e.g. ``model.layers.0...``), so we
    strip the outer ``language_model.`` / ``text_model.`` prefix from each entry.
    Entries that don't belong to the language model are dropped.
    """
    sub: dict = {"group_size": quant_cfg["group_size"], "bits": quant_cfg["bits"]}
    pref = text_prefix + "."
    for k, v in quant_cfg.items():
        if k in ("group_size", "bits"):
            continue
        if k == text_prefix or k.startswith(pref):
            sub[_strip_prefix(k, text_prefix)] = v
    return sub


def _detect_text_module_prefix(model: nn.Module) -> str:
    """Best-effort detection of the language-model sub-module name in a VLM.

    Returns an empty string if the model is text-only.
    """
    for cand in ("language_model", "text_model", "thinker"):
        if hasattr(model, cand):
            return cand
    return ""


def _pack_weight_mlx(intweight, bits):
    """Pack integer weights into uint32 in MLX format.

    MLX packs elements as a contiguous bit stream across uint32 boundaries.
    For bits that evenly divide 32 (2, 4, 8), each uint32 holds 32//bits elements.
    For other bit widths (e.g. 3, 5, 6, 7), every 32 elements are packed into
    `bits` uint32s (32 * bits = bits * 32 bits total).

    Args:
        intweight: shape [out_features, in_features], values in [0, 2^bits - 1]
        bits: quantization bits (2, 3, 4, 5, 6, 7, 8)

    Returns:
        packed: uint32 tensor, shape [out_features, in_features * bits / 32]
    """
    out_features, in_features = intweight.shape

    if 32 % bits == 0:
        # Simple case: bits evenly divides 32 (2, 4, 8)
        elems_per_int = 32 // bits
        assert (
            in_features % elems_per_int == 0
        ), f"in_features ({in_features}) must be divisible by {elems_per_int} for {bits}-bit packing"

        intweight = intweight.to(torch.int32)
        intweight = intweight.reshape(out_features, -1, elems_per_int)
        shifts = torch.arange(elems_per_int, device=intweight.device, dtype=torch.int32) * bits
        packed = (intweight << shifts).sum(dim=-1).to(torch.int32)
        return packed.view(torch.uint32)
    else:
        # Cross-word packing: 32 elements → `bits` uint32s
        # MLX packs as a contiguous bit stream across uint32 boundaries
        assert in_features % 32 == 0, f"in_features ({in_features}) must be divisible by 32 for {bits}-bit packing"

        intweight = intweight.to(torch.int64)
        num_groups = in_features // 32
        # Reshape to [out_features, num_groups, 32]
        elems = intweight.reshape(out_features, num_groups, 32)

        # For each element i in [0..31], it contributes `bits` bits starting at bit position i*bits
        # in a 32*bits bit stream packed into `bits` uint32s.
        # We process each bit b of each element i:
        #   absolute_bit = i * bits + b
        #   word_idx = absolute_bit // 32
        #   bit_pos  = absolute_bit % 32
        packed = torch.zeros(out_features, num_groups, bits, dtype=torch.int64, device=intweight.device)

        for b in range(bits):
            # Extract bit b from all elements: shape [out, num_groups, 32]
            bit_vals = (elems >> b) & 1
            for i in range(32):
                abs_bit = i * bits + b
                word_idx = abs_bit // 32
                bit_pos = abs_bit % 32
                packed[:, :, word_idx] |= bit_vals[:, :, i] << bit_pos

        packed = packed.to(torch.int32).reshape(out_features, num_groups * bits)
        return packed.view(torch.uint32)


class _MLXPackedLayer(nn.Module):
    """Holds MLX-packed quantized tensors for serialization.

    Tensor names match MLX convention: weight, scales, biases, bias.
    The ``bits`` / ``group_size`` attributes are kept so that mixed-bit
    quantization can be reflected in the exported ``config.json``.
    """

    def __init__(self, weight, scales, biases, bias, bits=None, group_size=None):
        super().__init__()
        self.register_buffer("weight", weight)  # uint32
        self.register_buffer("scales", scales)  # float16
        self.register_buffer("biases", biases)  # float16
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        self.bits = bits
        self.group_size = group_size


def pack_layer(name, model, device=None, **kwargs):
    """Pack a single layer into MLX quantized format.

    Reads layer attributes set by auto-round quantization:
        layer.weight  - float weight [out_features, in_features]
        layer.scale   - [out_features, num_groups]
        layer.zp      - [out_features, num_groups] or scalar
        layer.bits, layer.group_size, layer.sym

    Replaces the layer with _MLXPackedLayer containing:
        weight  - uint32 packed ints [out_features, in_features * bits / 32]
        scales  - float16 [out_features, num_groups]
        biases  - float16 [out_features, num_groups]
        bias    - original linear bias (if any)
    """
    layer = get_module(model, name)
    if hasattr(layer, "orig_layer"):
        layer = layer.orig_layer

    if type(layer) not in SUPPORTED_LAYER_TYPES:
        return

    if not check_to_quantized(layer):
        return

    device = get_packing_device(device)

    bits = int(layer.bits)
    group_size = int(layer.group_size)
    sym = bool(getattr(layer, "sym", False))
    scale = layer.scale  # [out_features, num_groups]
    zp = layer.zp  # [out_features, num_groups] or scalar (asym only)

    # Get weight in [out_features, in_features] layout
    W = layer.weight.data.to(device).clone().float()
    if type(layer) == nn.Conv2d:
        W = W.flatten(1)
    if type(layer) == transformers.pytorch_utils.Conv1D:
        W = W.t()

    out_features, in_features = W.shape
    if group_size == -1:
        group_size = in_features

    maxq = 2**bits - 1

    # Quantize: w_int = round(W / scale + zp), clamped to [0, maxq].
    # For symmetric mode we deliberately IGNORE layer.zp (it may be a tensor /
    # carry rounding noise / be device-dependent) and use the fixed integer
    # zero point ``2**(bits-1)`` (e.g. 8 for 4-bit, 4 for 3-bit, 128 for 8-bit).
    # This matches QuantLinearMLX.from_gptq's sym branch and the GPTQ sym
    # convention used everywhere else in auto-round.
    scale_dev = scale.to(device).float()
    repeat_scales = scale_dev.repeat_interleave(group_size, dim=1)[:, :in_features]

    if sym:
        zp_dev = float(2 ** (bits - 1))
        repeat_zp = zp_dev
    elif isinstance(zp, torch.Tensor):
        zp_dev = zp.to(device).float()
        repeat_zp = zp_dev.repeat_interleave(group_size, dim=1)[:, :in_features]
    else:
        zp_dev = float(zp)
        repeat_zp = zp_dev

    intweight = torch.round(W / repeat_scales + repeat_zp).clamp(0, maxq).to(torch.int32)

    # Pack weights into uint32
    packed_weight = _pack_weight_mlx(intweight, bits)

    # MLX dequant: w_float = mlx_scale * w_int + mlx_bias
    # auto-round:  w_float = scale * (w_int - zp) = scale * w_int - scale * zp
    # So: mlx_scale = scale, mlx_bias = -scale * zp
    mlx_scales = scale_dev.contiguous().to(torch.float16)
    mlx_biases = (-scale_dev * zp_dev).contiguous().to(torch.float16)

    # Preserve original bias
    orig_bias = None
    if layer.bias is not None:
        orig_bias = layer.bias.clone().to(torch.float16).cpu()

    new_layer = _MLXPackedLayer(
        packed_weight.cpu(),
        mlx_scales.cpu(),
        mlx_biases.cpu(),
        orig_bias,
        bits=bits,
        group_size=group_size,
    )
    set_module(model, name, new_layer)
    logger.debug(f"Packed layer {name} for MLX format (bits={bits}, group_size={group_size})")


def save_quantized_as_mlx(
    output_dir: str,
    model: nn.Module = None,
    tokenizer: Callable = None,
    layer_config: dict = None,
    inplace: bool = True,
    device: Union[str, torch.device] = "cpu",
    serialization_dict: dict = None,
    **kwargs,
) -> nn.Module:
    """Save quantized model in MLX-compatible format.

    The output can be loaded by mlx-lm::

        from mlx_lm import load, generate
        model, tokenizer = load("output_dir")
        response = generate(model, tokenizer, prompt="Hello", max_tokens=100)
    """
    if not inplace:
        model = copy.deepcopy(model.to("cpu"))

    if not unsupported_meta_device(model):
        model = model.to("cpu")

    os.makedirs(output_dir, exist_ok=True)

    bits = serialization_dict.get("bits", 4) if serialization_dict else 4
    group_size = serialization_dict.get("group_size", 128) if serialization_dict else 128

    # Pack all quantized layers (skip if already packed by immediate_pack)
    if layer_config:
        for layer_name in layer_config:
            pack_layer(layer_name, model, device=device)

    # Save model weights (uint32 packed weights are saved directly by safetensors)
    if not unsupported_meta_device(model):
        model.save_pretrained(output_dir, safe_serialization=True)

    # Write config.json with MLX-compatible quantization info
    config_path = os.path.join(output_dir, "config.json")
    if hasattr(model, "config") and not os.path.exists(config_path):
        model.config.save_pretrained(output_dir)

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        # Build mixed-bit aware quantization dict (mlx-community format).
        quant_cfg = _build_mlx_quantization_config(model, default_bits=bits, default_group_size=group_size)
        config["quantization"] = quant_cfg
        autoround_format = kwargs.get("autoround_format", False)
        quant_cfg_full = None
        if autoround_format:
            quant_cfg_full = {
                "quant_method": "auto-round",
                "packing_format": "mlx",
                "bits": bits,
                "group_size": group_size,
                "sym": serialization_dict.get("sym", True) if serialization_dict else True,
                "data_type": serialization_dict.get("data_type", "int") if serialization_dict else "int",
            }
            config["quantization_config"] = quant_cfg_full
        else:
            config["quantization_config"] = quant_cfg

        # ---- VLM support ------------------------------------------------ #
        # If the model has a language-model sub-module (e.g. Qwen-VL family),
        # mlx-vlm's language loader queries the quantization dict under
        # ``text_config`` (or its alias) with keys relative to that sub-module.
        # Mirror our config there so both code paths work.
        text_prefix = _detect_text_module_prefix(model)
        if text_prefix:
            sub_quant = _build_text_subconfig_quantization(quant_cfg, text_prefix)
            for sub_key in _TEXT_SUB_CONFIG_KEYS:
                sub_cfg = config.get(sub_key)
                if isinstance(sub_cfg, dict):
                    sub_cfg["quantization"] = sub_quant
                    if autoround_format and quant_cfg_full is not None:
                        sub_cfg["quantization_config"] = quant_cfg_full

        # Flatten rope_parameters recursively so mlx-lm/mlx-vlm find rope_theta
        # at the level it expects (top-level for LLMs, text_config for VLMs).
        _flatten_rope_parameters_recursive(config)

        # Backfill rope_theta from the live model.config object — required by
        # HF's model __init__ for the ``auto_round:mlx`` packing format (the
        # checkpoint is loaded through transformers.AutoModelForCausalLM, which
        # raises KeyError for missing rope_theta on Qwen-family models).
        _ensure_rope_theta_from_config_obj(config, getattr(model, "config", None))

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    # Save tokenizer
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_dir)

    # Copy processor if available
    processor = kwargs.get("processor", None)
    if processor is not None and hasattr(processor, "save_pretrained"):
        processor.save_pretrained(output_dir)

    # Try to copy Python files from model cache
    try:
        copy_python_files_from_model_cache(model, output_dir)
    except Exception as e:
        logger.warning(f"Failed to copy Python files from model cache: {e}")

    logger.info(f"Model saved to {output_dir} in MLX format (bits={bits}, group_size={group_size})")
    logger.info(f"Load with: from mlx_lm import load, generate; " f"model, tokenizer = load('{output_dir}')")

    return model
