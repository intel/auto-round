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
"""Combined RRQ loading: base model + residual model -> RRQ-enabled model.

Phase 1 entry point :func:`load_rrq_model` combines an already-exported INT2
base model (standard ``auto_round`` format) with its ``auto_round:rrq``
residual model (three packed INT2 planes stored together) into a single model
whose eligible linear layers are replaced with
:class:`~auto_round.inference.rrq_linear.RRQLinear` modules.

Plane storage (packed INT2):
    - plane 0 (base): the base model's standard W2A16 packed tensors
      (``qweight`` / ``scales`` / ``qzeros`` / ``bias``).  Because the base
      checkpoint is *packed*, ``AutoModelForCausalLM.from_pretrained`` cannot
      reconstruct the ``QuantLinear`` modules, so each base plane is rebuilt
      from its packed tensors (see :func:`_build_quant_plane`).  ``from_
      pretrained`` is still used to load the model *architecture* and the
      non-quant weights (embeddings, layernorms, ...).
    - planes 1..K-1 (residual): loaded from the residual model's
      ``{layer}.qweight_{k}`` / ``scales_{k}`` / ``qzeros_{k}`` buffers -- each
      a standard single-plane INT2 AutoRound layout -- and wrapped in a stock
      W2A16 ``QuantLinear``.

Both planes are built with the ``QuantLinear`` class matching the model's
``sym`` flag (``qlinear_torch_zp`` for symmetric, ``qlinear_torch`` for
asymmetric), exactly as the export path does, so dequantization
(:meth:`QuantLinear.forward`) matches bit-for-bit.

``forward`` reuses the existing W2A16 ``QuantLinear`` code path: the base
result is computed first and each active residual's result is accumulated on
top of it.  The base plane owns the bias, so :class:`RRQLinear` does not add a
second one.

Use :func:`~auto_round.inference.rrq_linear.set_rrq_bits` to switch the active
precision (2/4/6/8-bit) across all RRQ layers.
"""

import glob
import json
import os
import re
from typing import Optional

import torch
import torch.nn as nn

from auto_round.logger import logger

from auto_round.export.export_to_autoround.export_to_rrq import RRQ_QUANT_METHOD

__all__ = ["load_rrq_model"]


def _quant_linear_class(sym: bool) -> type:
    """The W2A16 ``QuantLinear`` class for the given ``sym`` flag.

    Mirrors ``dynamic_import_quant_linear_for_packing`` for ``act_bits=16``:
    symmetric -> ``qlinear_torch_zp.QuantLinear``, asymmetric ->
    ``qlinear_torch.QuantLinear``.  The two classes store/dequantize the zero
    point differently, so the class must match the model's ``sym`` flag.
    """
    if sym:
        from auto_round_extension.torch.qlinear_torch_zp import QuantLinear
    else:
        from auto_round_extension.torch.qlinear_torch import QuantLinear

    return QuantLinear


def _load_config(config_dir: str) -> dict:
    path = os.path.join(config_dir, "config.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"config.json not found in {config_dir}")
    with open(path) as f:
        return json.load(f)


def _load_state_dict(model_dir: str) -> dict:
    """Load a (possibly sharded) safetensors / pytorch state dict from ``model_dir``."""
    state_dict = {}
    st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if st_files:
        from safetensors.torch import load_file as _load_st

        for f in st_files:
            state_dict.update(_load_st(f))
        return state_dict

    pt_files = sorted(glob.glob(os.path.join(model_dir, "*.pt"))) + sorted(
        glob.glob(os.path.join(model_dir, "*.bin"))
    )
    if pt_files:
        for f in pt_files:
            state_dict.update(torch.load(f, map_location="cpu", weights_only=False))
        return state_dict

    raise FileNotFoundError(f"No model weight files found in {model_dir}")


def _in_out_from_qweight(qweight: torch.Tensor, bits: int):
    """Recover ``(in_features, out_features)`` from a packed ``qweight`` tensor.

    ``qweight`` has shape ``(in_features // 32 * bits, out_features)``.
    """
    in_features = qweight.shape[0] * 32 // bits
    out_features = qweight.shape[1]
    return in_features, out_features


def _enumerate_base_layers(state: dict, bits: int) -> dict:
    """Map ``layer_name -> (in, out, has_bias)`` for packed base planes.

    Matches keys ending in an exact ``.qweight`` (not ``.qweight_k``) that also
    have a ``.scales`` tensor.
    """
    layers = {}
    for key, value in state.items():
        if not key.endswith(".qweight") or not isinstance(value, torch.Tensor):
            continue
        if key.rsplit(".", 1)[1] != "qweight":
            continue
        layer_name = key.rsplit(".qweight", 1)[0]
        if f"{layer_name}.scales" not in state:
            continue
        in_features, out_features = _in_out_from_qweight(value, bits)
        has_bias = f"{layer_name}.bias" in state
        layers[layer_name] = (in_features, out_features, has_bias)
    return layers


def _enumerate_residual_planes(state: dict) -> dict:
    """Map ``layer_name -> [k, ...]`` for residual planes (from ``qweight_k`` keys)."""
    layers = {}
    pattern = re.compile(r"^(.*)\.qweight_(\d+)$")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        match = pattern.match(key)
        if match is None:
            continue
        layer_name = match.group(1)
        k = int(match.group(2))
        layers.setdefault(layer_name, set()).add(k)
    return {layer: sorted(ks) for layer, ks in layers.items()}


def _build_quant_plane(
    QuantLinear: type,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    bits: int,
    group_size: int,
    in_features: int,
    out_features: int,
    has_bias: bool,
    bias: Optional[torch.Tensor],
    device: torch.device,
) -> nn.Module:
    """Build one ``QuantLinear`` plane from its packed tensors (on ``device``)."""
    ql = QuantLinear(bits, group_size, in_features, out_features, bias=has_bias)
    ql.qweight.data = qweight.to(torch.int32).cpu()
    ql.scales.data = scales.to(torch.float16).cpu()
    ql.qzeros.data = qzeros.to(torch.int32).cpu()
    if has_bias and bias is not None:
        ql.bias.data = bias.to(torch.float16).cpu()
    ql.to(device)
    return ql


def _synthesize_qzeros(num_groups: int, out_features: int, bits: int) -> torch.Tensor:
    """Build an all-zero ``qzeros`` tensor of the expected packed shape."""
    pack_factor = 32 // bits
    return torch.zeros((num_groups, max(1, out_features // pack_factor)), dtype=torch.int32)


def _validate_base_matches_residual(base_config: dict, residual_config: dict) -> None:
    """Fail fast if the base model's quant settings do not match the residual model."""
    base_q = base_config.get("quantization_config", {})
    res_q = residual_config.get("quantization_config", {})

    checks = [
        ("bits", base_q.get("bits", 2), res_q.get("bits", 2)),
        ("group_size", base_q.get("group_size", 128), res_q.get("group_size", 128)),
        ("sym", base_q.get("sym"), res_q.get("sym")),
    ]
    for field, base_val, res_val in checks:
        if base_val != res_val:
            raise ValueError(
                f"Base/residual quant settings mismatch on '{field}': "
                f"base={base_val!r}, residual={res_val!r}. "
                "The residual model must match the base model's bits/group_size/sym."
            )

    if res_q.get("quant_method") != RRQ_QUANT_METHOD:
        raise ValueError(
            f"Residual model has quant_method={res_q.get('quant_method')!r}, "
            f"expected {RRQ_QUANT_METHOD!r}."
        )


def load_rrq_model(
    base_model_dir: str,
    residual_model_dir: str,
    active_bits: int = 8,
    device: Optional[str] = None,
    **load_kwargs,
):
    """Load a base model + residual model and return an RRQ-enabled model.

    Args:
        base_model_dir: Path to the exported INT2 base model (standard format).
        residual_model_dir: Path to the exported ``auto_round:rrq`` residual model
            (three packed INT2 planes).
        active_bits: Effective bit-width to start with (2, 4, 6, or 8).
        device: Target device (e.g. ``"cpu"``, ``"cuda"``).
        **load_kwargs: Extra keyword args forwarded to
            ``transformers.AutoModelForCausalLM.from_pretrained`` for the base
            model.

    Returns:
        The combined model. Eligible linear layers are replaced with
        :class:`~auto_round.inference.rrq_linear.RRQLinear` modules; use
        :func:`auto_round.inference.rrq_linear.set_rrq_bits` to switch
        precision.

    Raises:
        FileNotFoundError: If either directory is missing or has no weights.
        ValueError: If base/residual quant settings do not match, the residual
            model is not an ``auto-round-rrq`` model, or no layers are found.
    """
    import transformers

    device_obj = torch.device(device or "cpu")
    if active_bits not in {2, 4, 6, 8}:
        raise ValueError(f"active_bits must be one of 2/4/6/8, got {active_bits}")

    for d in (base_model_dir, residual_model_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Model directory not found: {d}")

    base_config = _load_config(base_model_dir)
    residual_config = _load_config(residual_model_dir)
    _validate_base_matches_residual(base_config, residual_config)

    residual_q = residual_config.get("quantization_config", {})
    total_planes = residual_q.get("total_planes", 4)
    bits = residual_q.get("bits", 2)
    group_size = residual_q.get("group_size", 128)
    sym = bool(residual_q.get("sym", False))
    QuantLinear = _quant_linear_class(sym)

    # Collect packed planes from both checkpoints (state dicts live on CPU).
    base_state = _load_state_dict(base_model_dir)
    base_layers = _enumerate_base_layers(base_state, bits)
    residual_state = _load_state_dict(residual_model_dir)
    residual_planes_by_layer = _enumerate_residual_planes(residual_state)

    eligible = set(base_layers) & set(residual_planes_by_layer)
    if not eligible:
        raise ValueError(
            "No common packed-INT2 layers found between base and residual models; "
            "cannot build any RRQ layer."
        )

    # Load the base model architecture + non-quant weights.  The packed ``qweight``
    # keys do not match the standard ``nn.Linear.weight`` shape, so those layers
    # load with uninitialised weights here -- we rebuild them as ``QuantLinear``
    # from the packed tensors below, so the uninitialised values are discarded.
    logger.info(f"Loading base model architecture from {base_model_dir}")
    try:
        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            **load_kwargs,
        )
    except Exception:  # pragma: no cover - fall back to architecture-only load
        logger.warning(
            "from_pretrained(%s) failed; loading architecture without weights.",
            base_model_dir,
            exc_info=True,
        )
        base_model = transformers.AutoModelForCausalLM.from_config(
            transformers.AutoConfig.from_pretrained(base_model_dir)
        )
    base_model.to(device_obj)

    from auto_round.inference.rrq_linear import RRQLinear
    from auto_round.utils import set_module

    replaced = 0
    for layer_name in sorted(eligible):
        in_features, out_features, has_bias = base_layers[layer_name]

        try:
            base_model.get_submodule(layer_name)
        except AttributeError:
            logger.warning(f"Base module {layer_name!r} not found; skipping.")
            continue

        # --- Base plane (plane 0) ---
        base_qweight = base_state[f"{layer_name}.qweight"]
        base_scales = base_state[f"{layer_name}.scales"]
        base_qzeros = base_state.get(f"{layer_name}.qzeros")
        base_bias = base_state.get(f"{layer_name}.bias")
        if base_qzeros is None:
            num_groups = (in_features + group_size - 1) // group_size
            base_qzeros = _synthesize_qzeros(num_groups, out_features, bits)
        base_plane = _build_quant_plane(
            QuantLinear,
            base_qweight,
            base_scales,
            base_qzeros,
            bits,
            group_size,
            in_features,
            out_features,
            has_bias,
            base_bias,
            device_obj,
        )

        # --- Residual planes (1..K-1) ---
        present = set(residual_planes_by_layer[layer_name])
        residual_planes = []
        complete = True
        for k in range(1, total_planes):
            if k not in present:
                complete = False
                break
            rw = residual_state[f"{layer_name}.qweight_{k}"]
            rs = residual_state[f"{layer_name}.scales_{k}"]
            rz = residual_state[f"{layer_name}.qzeros_{k}"]
            residual_planes.append(
                _build_quant_plane(
                    QuantLinear, rw, rs, rz, bits, group_size, in_features,
                    out_features, False, None, device_obj,
                )
            )
        if not complete or len(residual_planes) != total_planes - 1:
            logger.warning(f"Layer {layer_name!r} missing residual planes; skipping.")
            continue

        # The base plane owns the bias; RRQLinear must not add a second one.
        rrq_linear = RRQLinear(base=base_plane, residual_planes=residual_planes, bias=None)
        set_module(base_model, layer_name, rrq_linear)
        replaced += 1

    if replaced == 0:
        raise ValueError(
            "No RRQ layers could be built from base + residual models. "
            "Check that base_model_dir and residual_model_dir are compatible."
        )

    from auto_round.inference.rrq_linear import set_rrq_bits

    set_rrq_bits(base_model, active_bits)

    logger.info(f"Built {replaced} RRQ layers from base + residual (active={active_bits}-bit).")
    logger.info(
        "Switch precision via auto_round.inference.rrq_linear.set_rrq_bits(model, bits). "
        "Each RRQ layer computes the base result first, then accumulates each active "
        "residual's result (stock W2A16 dequant; correctness reference, not a fused kernel)."
    )
    return base_model
