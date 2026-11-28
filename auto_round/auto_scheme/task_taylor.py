# Copyright (c) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Task-loss directional Taylor scores at isolated quantized layer points.

Other layers keep their original weights and activations. For the selected layer,
restore quantized weights and (for MXFP) activations along a scalar t. At t=0,
T1=-dL/dt and T2=-0.5*d2L/dt2 predict L(quantized)-L(original). The score is the
sample-weighted absolute signed prediction, not a sum of elementwise magnitudes.
All three scores share the same derivatives and incremental on-disk cache.
"""

import hashlib
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from auto_round.auto_scheme.hessian import _validate_layer, input_fingerprint
from auto_round.auto_scheme.utils import compute_layer_bits
from auto_round.calib_dataset import get_dataloader
from auto_round.data_type.int import quant_tensor_asym, quant_tensor_sym
from auto_round.data_type.mxfp import quant_mx
from auto_round.logger import logger
from auto_round.utils import get_block_names, get_module, set_module, to_device

TASK_METHODS = ("task_first", "task_second", "task_first_second")


def directional_terms(loss, position):
    """Differentiate the continuous restoration path, never a rounding graph."""
    derivative = torch.autograd.grad(loss, position, create_graph=True)[0]
    curvature = torch.autograd.grad(derivative, position)[0] if derivative.requires_grad else torch.zeros_like(position)
    first, second = -derivative.detach().item(), -0.5 * curvature.detach().item()
    prediction_scores(first, second)
    return first, second


def prediction_scores(first, second):
    if not math.isfinite(first) or not math.isfinite(second):
        raise ValueError("Task Taylor terms must be finite")
    return dict(zip(TASK_METHODS, (abs(first), abs(second), abs(first + second))))


class RestorationLinear(torch.nn.Module):
    """Quantized endpoint with smooth joint weight/activation restoration."""

    def __init__(self, original, quantized_weight, activation_quantizer=None):
        super().__init__()
        self.original = original
        self.register_buffer("quantized_weight", quantized_weight.detach())
        self.position = torch.nn.Parameter(
            torch.zeros((), device=quantized_weight.device, dtype=quantized_weight.dtype)
        )
        self.activation_quantizer = activation_quantizer

    def forward(self, inputs):
        if self.activation_quantizer is not None:
            with torch.no_grad():
                quantized = self.activation_quantizer(inputs)
            inputs = quantized + self.position * (inputs - quantized)
        weight = self.quantized_weight + self.position * (self.original.weight - self.quantized_weight)
        return torch.nn.functional.linear(inputs, weight, self.original.bias)


@contextmanager
def candidate_layer(model, name, quantized_weight, activation_quantizer=None):
    original = get_module(model, name)
    replacement = RestorationLinear(original, quantized_weight, activation_quantizer)
    set_module(model, name, replacement)
    try:
        yield replacement
    finally:
        set_module(model, name, original)


@contextmanager
def checkpoint_blocks(model):
    """Checkpoint eval-mode blocks without enabling dropout or training behavior."""
    saved = []
    try:
        for name in get_block_names(model)[0]:
            block = get_module(model, name)
            had_override = "forward" in block.__dict__
            old_override = block.__dict__.get("forward")
            forward = block.forward

            def checked(*args, _forward=forward, **kwargs):
                return checkpoint(_forward, *args, use_reentrant=False, **kwargs)

            saved.append((block, had_override, old_override))
            block.forward = checked
        yield
    finally:
        for block, had_override, old_override in saved:
            if had_override:
                block.forward = old_override
            else:
                del block.forward


def candidate_quantization(layer):
    _validate_layer(layer)
    with torch.no_grad():
        if layer.bits >= 16:
            return layer.weight.detach(), None
        if layer.data_type == "mx_fp":
            weight = quant_mx(layer.weight, bits=layer.bits, group_size=layer.group_size)[0]
            if layer.act_bits < 16:
                bits, group = layer.act_bits, layer.act_group_size
                return weight, lambda x: quant_mx(x, bits=bits, group_size=group)[0]
            return weight, None
        quantizer = quant_tensor_sym if layer.sym else quant_tensor_asym
        weight = quantizer(layer.weight, bits=layer.bits, group_size=layer.group_size, scale_dtype=torch.float16, v=0)[
            0
        ]
        return weight, None


def _cache_path(model, data, names, fixed_layer_scheme):
    """Cache all methods together; include actual token batches and implementation."""
    root = os.environ.get("AR_AUTO_SCHEME_CACHE")
    source = getattr(model.config, "_name_or_path", "")
    if not root or not source:
        return None, None
    tokens = hashlib.sha256()
    for batch in data:
        for key, tensor in sorted(batch.items()):
            tokens.update(key.encode())
            tokens.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            tokens.update(tensor.contiguous().numpy().tobytes())
    specs = {}
    for name in names:
        layer = get_module(model, name)
        specs[name] = {
            key: getattr(layer, key, None)
            for key in ("bits", "group_size", "sym", "data_type", "act_bits", "act_group_size", "act_data_type")
        }
    files = [Path(__file__), Path(__file__).with_name("delta_loss.py")]
    files += [Path(__file__).parents[1] / "data_type" / f"{name}.py" for name in ("int", "mxfp")]
    identity = {
        "version": 1,
        "checkpoint": input_fingerprint(source),
        "tokens_sha256": tokens.hexdigest(),
        "specs": specs,
        "fixed_layer_scheme": fixed_layer_scheme,
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "background": "isolated_candidate_other_layers_original",
        "dtype": "float32",
        "normalization": "valid_token_weighted_microbatch_absolute_prediction",
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    return Path(root) / "task_taylor" / f"{key}.json", identity


def _save_cache(path, identity, statistics):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"identity": identity, "layers": statistics}, indent=2) + "\n")
    temporary.replace(path)


@torch.enable_grad()
def get_task_taylor_score_for_scheme(
    model,
    tokenizer,
    quant_layer_names,
    fixed_layer_scheme,
    dataset,
    ignore_scale_zp_bits=False,
    nsamples=16,
    seqlen=256,
    batch_size=1,
    major_device="cpu",
    method="task_first_second",
    pbar=None,
    disk_index=None,
    **kwargs,
):
    """Score dense causal LMs; full FP32 model resides on a single device.

    This experimental scorer deliberately does not use block offload: its second
    derivatives require a continuous end-to-end graph. Each completed layer is
    checkpointed; interrupting one candidate does not discard previous layers.
    """
    if method not in TASK_METHODS:
        raise ValueError(f"Unknown task Taylor method: {method}")
    if disk_index is not None or kwargs.get("is_vlm", False):
        raise ValueError("Task Taylor scoring requires a resident dense text model; disk streaming/VLM unsupported")
    if getattr(model.config, "_attn_implementation", None) != "eager":
        raise ValueError("Task Taylor second derivatives require attn_implementation=eager")
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Task Taylor scoring requires FP32 model weights")
    names = sorted(name for name in quant_layer_names if name not in fixed_layer_scheme)
    for name in names:
        _validate_layer(get_module(model, name))
    data = [
        {key: value.cpu() for key, value in batch.items()}
        for batch in get_dataloader(tokenizer, seqlen, dataset_name=dataset, seed=42, bs=batch_size, nsamples=nsamples)
    ]
    if not data:
        raise ValueError("No calibration batches for task Taylor scoring")
    path, identity = _cache_path(model, data, names, fixed_layer_scheme)
    statistics = {}
    if path is not None and path.exists():
        saved = json.loads(path.read_text())
        if saved["identity"] != identity:
            raise ValueError(f"Task Taylor cache identity mismatch: {path}")
        statistics = saved["layers"]
    costs = {name: compute_layer_bits(get_module(model, name), ignore_scale_zp_bits)[0] for name in names}
    training = {module: module.training for module in model.modules()}
    gradients = {parameter: parameter.requires_grad for parameter in model.parameters()}
    try:
        model.eval().requires_grad_(False).to(major_device)
        with checkpoint_blocks(model):
            for index, name in enumerate(names):
                if name in statistics:
                    logger.info("Task Taylor: reuse %s (%d/%d)", name, index + 1, len(names))
                    continue
                layer = get_module(model, name)
                qw, activation_quantizer = candidate_quantization(layer)
                totals = {key: 0.0 for key in (*TASK_METHODS, "first_signed", "second_signed", "quantized_loss")}
                valid_tokens = 0
                with candidate_layer(model, name, qw, activation_quantizer) as candidate:
                    for batch in data:
                        batch = to_device(batch, major_device)
                        labels = batch.get("labels", batch["input_ids"]).clone()
                        if "attention_mask" in batch:
                            labels.masked_fill_(batch["attention_mask"] == 0, -100)
                        count = (labels[:, 1:] != -100).sum().item()
                        if not count:
                            raise ValueError("No valid next-token labels for task Taylor scoring")
                        feed = {key: value for key, value in batch.items() if key != "labels"}
                        loss = model(**feed, labels=labels, use_cache=False).loss
                        first, second = directional_terms(loss, candidate.position)
                        if not torch.isfinite(loss):
                            raise ValueError("Non-finite task loss")
                        for key, value in prediction_scores(first, second).items():
                            totals[key] += value * count
                        totals["first_signed"] += first * count
                        totals["second_signed"] += second * count
                        totals["quantized_loss"] += loss.detach().item() * count
                        valid_tokens += count
                        del loss
                statistics[name] = {key: value / valid_tokens for key, value in totals.items()}
                statistics[name]["valid_tokens"] = valid_tokens
                _save_cache(path, identity, statistics)
                logger.info("Task Taylor: completed %s (%d/%d): %s", name, index + 1, len(names), statistics[name])
        if pbar is not None:
            pbar.update(kwargs.get("progress_steps", 2 * len(get_block_names(model)[0])))
        scores = {name: [costs[name], statistics[name][method]] for name in names}
        if any(not math.isfinite(score) or score < 0 for _, score in scores.values()):
            raise ValueError("Invalid task Taylor allocation score")
        return scores
    finally:
        model.to("cpu")
        for parameter, required in gradients.items():
            parameter.requires_grad_(required)
        for module, was_training in training.items():
            module.training = was_training
