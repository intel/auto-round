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

"""GPTQ local reconstruction Hessian scores, evaluated without storing H.

For token rows X and weight perturbations D, H = 2 X.T X / n. Therefore
0.5 * sum_r D[r] H D[r].T = ||X D.T||_F**2 / n. This is the *local output
reconstruction* Hessian, not the end-to-end language-model loss Hessian.
Calibration runs at the original full-precision background for every candidate.
For MXFP weight/activation perturbations the joint local Hessian gives
||X deltaW.T + deltaX W.T||_F**2 / n, including mixed curvature.
"""

import hashlib
from functools import partial
from pathlib import Path

import torch

from auto_round.auto_scheme.utils import compute_layer_bits
from auto_round.calib_dataset import get_dataloader
from auto_round.data_type.int import quant_tensor_asym, quant_tensor_sym
from auto_round.data_type.mxfp import quant_mx
from auto_round.utils import get_block_names, get_module, to_device


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(value):
    """Identify local inputs without reading every byte of multi-GB checkpoints.

    Direct data files are content-hashed. Checkpoint directories record canonical
    path and top-level file size/mtime, plus content hashes for files <= 1 MiB.
    This detects ordinary checkpoint replacement, not deliberate preservation of
    both weight-file size and timestamps. Remote IDs retain their full namespace.
    """
    path = Path(str(value)).expanduser()
    if not path.exists():
        return {"source": str(value), "kind": "remote_or_named_dataset"}
    path = path.resolve()
    if path.is_file():
        return {"source": str(path), "kind": "file", "sha256": _file_sha256(path)}
    files = []
    for item in sorted(path.iterdir()):
        if not item.is_file():
            continue
        stat = item.stat()
        row = {"name": item.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if stat.st_size <= 1024 * 1024:
            row["sha256"] = _file_sha256(item)
        files.append(row)
    return {"source": str(path), "kind": "checkpoint_directory", "files": files}


def scoring_cache_identity(model_name, dataset):
    """Separate local-Hessian caches by actual input and scoring implementation."""
    source = Path(__file__)
    files = [
        source,
        source.with_name("delta_loss.py"),
        source.parents[1] / "data_type" / "int.py",
        source.parents[1] / "data_type" / "mxfp.py",
    ]
    return {
        "checkpoint": input_fingerprint(model_name),
        "calibration": input_fingerprint(dataset),
        "implementation": {path.name: _file_sha256(path) for path in files},
    }


class LocalHessianScore:
    """Accumulate a token-normalized full or diagonal local quadratic score."""

    def __init__(self, diagonal=False):
        self.diagonal = diagonal
        self.total = 0.0
        self.tokens = 0

    @torch.no_grad()
    def add(self, delta, inputs, attention_mask=None, *, weight=None, activation_delta=None):
        """Sum the local quadratic; optional activation perturbations use joint curvature."""
        dx = activation_delta.detach().reshape(-1, inputs.shape[-1]).float() if activation_delta is not None else None
        if dx is not None and weight is None:
            raise ValueError("Joint Hessian scoring requires the original weight")
        x = inputs.detach().reshape(-1, inputs.shape[-1]).float()
        if attention_mask is not None:
            if attention_mask.numel() != x.shape[0]:
                raise ValueError("Attention mask does not match linear input token dimensions")
            mask = attention_mask.reshape(-1).to(device=x.device, dtype=torch.bool)
            x = x[mask]
            if dx is not None:
                dx = dx[mask]
        if not x.shape[0]:
            return
        delta = delta.detach().float()
        if self.diagonal:
            value = (delta.square().sum(0) * x.square().sum(0)).sum()
            if dx is not None:
                value += (weight.detach().float().square().sum(0) * dx.square().sum(0)).sum()
        else:
            # Bound the temporary output error matrix, including large FFN projections.
            value = torch.zeros((), device=x.device)
            for start in range(0, delta.shape[0], 256):
                error = x @ delta[start : start + 256].T
                if dx is not None:
                    error += dx @ weight.detach()[start : start + 256].float().T
                value += error.square().sum()
        if not torch.isfinite(value):
            raise ValueError("Non-finite local Hessian score")
        self.total += value.item()
        self.tokens += x.shape[0]

    @property
    def score(self):
        if self.tokens == 0:
            raise ValueError("No valid calibration tokens observed for local Hessian scoring")
        return self.total / self.tokens


def _validate_layer(layer):
    if not isinstance(layer, torch.nn.Linear):
        raise ValueError("Hessian AutoScheme currently supports dense torch.nn.Linear layers only")
    data_type = getattr(layer, "data_type", "int")
    if data_type == "mx_fp":
        if layer.bits not in (4, 8) or layer.group_size != 32:
            raise ValueError("Hessian MXFP requires native MXFP4/MXFP8 G32")
        if getattr(layer, "act_bits", 16) < 16 and (
            layer.act_data_type != "mx_fp"
            or layer.act_bits not in (4, 8)
            or layer.act_group_size != 32
            or not layer.act_dynamic
        ):
            raise ValueError("Hessian MXFP activations require dynamic native MXFP4/MXFP8 G32")
    elif data_type == "int":
        if getattr(layer, "act_bits", 16) < 16:
            raise ValueError("INT Hessian AutoScheme supports weight-only (A16) schemes only")
    else:
        raise ValueError("Hessian AutoScheme supports native INT and MXFP4/MXFP8 schemes only")
    if getattr(layer, "super_bits", None) is not None:
        raise ValueError("Hessian AutoScheme does not support GGUF super blocks")


@torch.no_grad()
def get_hessian_score_for_scheme(
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
    low_gpu_mem_usage=True,
    disk_index=None,
    pbar=None,
    diagonal=False,
    **kwargs,
):
    """Collect exact local quadratic scores using forward hooks on FP layers.

    Existing AutoScheme owns model loading, offload hooks, bit accounting and
    DP. Hooks here only measure errors; no weight updates or backward occurs.
    Disk-streamed blocks are materialized before hooks inspect their weights.
    """
    if kwargs.get("is_vlm", False):
        raise ValueError("Hessian AutoScheme currently supports text-only dense LLMs")
    names = [name for name in quant_layer_names if name not in fixed_layer_scheme]
    for name in names:
        _validate_layer(get_module(model, name))
    stats = {name: LocalHessianScore(diagonal=diagonal) for name in names}
    costs = {}
    current_mask = None
    handles = []
    training = {m: m.training for m in model.modules()}
    blocks = get_block_names(model)[0] if low_gpu_mem_usage else []
    loaded = set()

    def score_layer(module, args, name):
        if name not in costs:
            costs[name] = compute_layer_bits(module, ignore_scale_zp_bits)[0]
        if module.bits >= 16:
            # Record token counts even for a non-quantized candidate.
            stats[name].tokens += args[0].numel() // args[0].shape[-1]
            return
        if module.data_type == "mx_fp":
            qw, _, _ = quant_mx(module.weight, bits=module.bits, group_size=module.group_size)
            dx = None
            if module.act_bits < 16:
                qx, _, _ = quant_mx(args[0], bits=module.act_bits, group_size=module.act_group_size)
                dx = qx.float() - args[0].float()
            stats[name].add(
                qw.float() - module.weight.float(), args[0], current_mask, weight=module.weight, activation_delta=dx
            )
        else:
            quant_func = quant_tensor_sym if module.sym else quant_tensor_asym
            qw, _, _ = quant_func(
                module.weight, bits=module.bits, group_size=module.group_size, scale_dtype=torch.float16, v=0
            )
            delta = qw.float() - module.weight.float()
            stats[name].add(delta, args[0], current_mask)

    def enter_block(module, args, kwargs, name):
        if disk_index is not None:
            from auto_round.utils.disk_stream_util import materialize_module

            materialize_module(module, name, disk_index, device=major_device)
        module.to(major_device)
        loaded.add(name)
        return to_device(args, major_device), to_device(kwargs, major_device)

    def leave_block(module, args, output, name):
        if name in loaded:
            if disk_index is not None:
                from auto_round.utils.disk_stream_util import free_module

                free_module(module)
            else:
                module.to("cpu")
            loaded.remove(name)
        if pbar is not None:
            pbar.update(2)

    try:
        model.eval()
        for name in names:
            handles.append(get_module(model, name).register_forward_pre_hook(partial(score_layer, name=name)))
        for name in blocks:
            block = get_module(model, name)
            handles.append(block.register_forward_pre_hook(partial(enter_block, name=name), with_kwargs=True))
            handles.append(block.register_forward_hook(partial(leave_block, name=name), always_call=True))
        dataloader = get_dataloader(tokenizer, seqlen, dataset_name=dataset, seed=42, bs=batch_size, nsamples=nsamples)
        for data in dataloader:
            data = to_device(data, major_device)
            current_mask = data.get("attention_mask")
            model(**{k: v for k, v in data.items() if k != "labels"}, use_cache=False)
        return {name: [costs[name], score.score] for name, score in stats.items()}
    finally:
        for handle in handles:
            handle.remove()
        for module, was_training in training.items():
            module.training = was_training
