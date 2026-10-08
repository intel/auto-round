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

"""Convert logical SVDQuant records to vLLM-Omni's NVFP4 tensor layout.

Model adapters own naming and fusion. The default adapter preserves names.
Tuned NVFP4 residuals retain their scales; untuned NVFP4 residuals use RTN.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from auto_round.data_type.nvfp import calculate_gparam, get_reciprocal, nv_fp4
from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear, pack_fp4_to_uint8
from auto_round.export.svdquant_nunchaku import (
    IdentitySVDQuantModelAdapter,
    SVDQuantModelAdapter,
    _source_records,
    _validate_adapter_provenance,
)
from auto_round.wrapper import WrapperWALayer


def validate_nvfp4_scheme(scheme) -> bool:
    """Require the NVFP4 W4A4 scheme supported by Omni's SVDQuant runtime."""
    aliases = frozenset({"nv_fp", "nv_fp4", "nv_fp4_with_static_gs"})
    rules = dict(
        data_type=aliases,
        bits=4,
        group_size=16,
        sym=True,
        act_data_type=aliases,
        act_bits=4,
        act_group_size=16,
        act_sym=True,
        act_dynamic=True,
    )
    for name, expected in rules.items():
        actual = getattr(scheme, name, None)
        if isinstance(expected, frozenset):
            valid = isinstance(actual, str) and actual in expected
        else:
            valid = type(actual) is type(expected) and actual == expected
        if not valid:
            raise ValueError(
                f"svdquant_omni only supports NVFP4 W4A4 group16: " f"got {name}={actual!r}, expected {expected!r}"
            )
    return True


@torch.inference_mode()
def _pack_residual(weight: torch.Tensor, *, scale=None, global_scale=None) -> dict[str, torch.Tensor]:
    """Reuse the existing NVFP4 quantizer and FP4 encoder; only adapt layout."""
    if weight.ndim != 2 or weight.shape[1] % 16 or 0 in weight.shape:
        raise ValueError("NVFP4 residual must be a nonempty matrix with input width divisible by 16")
    if not torch.isfinite(weight).all():
        raise ValueError("NVFP4 residual must contain only finite values")
    weight = weight.float()
    n, k = weight.shape
    if scale is not None:
        if global_scale is None:
            raise ValueError("tuned NVFP4 export requires weight_global_scale")
        packed = QuantLinear(4, 16, k, n, False, data_type="nv_fp4", act_bits=16)
        linear = torch.nn.Module()
        linear.weight = weight
        packed.pack(linear, scale, global_scale=global_scale, device=weight.device)
        return {
            "qweight": packed.weight_packed.view(torch.int8),
            "wscales": packed.weight_scale.reshape(n, k // 16).T,
            "wtscale": global_scale.float().reciprocal().reshape(1).bfloat16(),
            "wcscales": torch.ones(n, dtype=torch.bfloat16, device=weight.device),
        }
    alpha = get_reciprocal(calculate_gparam(weight)).reshape(1).bfloat16()
    if weight.count_nonzero() == 0:
        alpha.fill_(1)
    if not torch.isfinite(alpha.float()).all() or not (alpha > 0).all():
        raise ValueError("NVFP4 tensor scale must be representable as positive finite BF16")
    # Bound the temporary allocation in the existing FP4 encoder.
    chunks, scales = [], []
    for chunk in weight.split(128):
        quantized, scale, _ = nv_fp4(chunk, global_scale=alpha.float().reciprocal())
        denominator = scale.reshape(chunk.shape[0], k // 16, 1) * alpha.float()
        normalized = quantized.reshape(chunk.shape[0], k // 16, 16) * get_reciprocal(denominator)
        chunks.append(pack_fp4_to_uint8(normalized.reshape_as(chunk)))
        scales.append(scale.reshape(chunk.shape[0], k // 16).to(torch.float8_e4m3fn))
    return {
        "qweight": torch.cat(chunks).view(torch.int8),
        "wscales": torch.cat(scales).T,
        "wtscale": alpha,
        "wcscales": torch.ones(n, dtype=torch.bfloat16, device=weight.device),
    }


@torch.inference_mode()
def collect_svdquant_omni_tensors(
    model: torch.nn.Module, *, device: str = "cpu", adapter: SVDQuantModelAdapter | None = None
) -> tuple[dict, dict]:
    """Serialize adapter-selected projections without assuming a model architecture."""
    adapter = adapter or IdentitySVDQuantModelAdapter()
    sources = _source_records(model)
    for source in sources:
        validate_nvfp4_scheme(source.scheme)
    records = tuple(adapter.map_modules(model, sources))
    if not records:
        raise ValueError("model adapter produced no SVDQuant export records")
    _validate_adapter_provenance(sources, records)
    adapter.validate_records(sources, records)
    tensors, ranks = {}, set()
    for record in records:
        validate_nvfp4_scheme(record.scheme)
        weight, down, up = record.residual_weight, record.lora_down, record.lora_up
        smooth = record.smooth.float()
        n, k = weight.shape
        rank = down.shape[0]
        if rank <= 0 or down.shape != (rank, k) or up.shape != (n, rank) or smooth.shape != (k,):
            raise ValueError(f"{record.prefix}: incompatible low-rank or smoothing dimensions")
        if not torch.isfinite(smooth).all() or not (smooth > 0).all():
            raise ValueError(f"{record.prefix}: smooth factors must be positive and finite")
        if record.bias is not None and record.bias.shape != (n,):
            raise ValueError(f"{record.prefix}: incompatible bias dimensions")
        ranks.add(rank)
        residuals = []
        for source in record.sources:
            residual = model.get_submodule(source.name).residual_linear
            while isinstance(residual, WrapperWALayer):
                residual = residual.orig_layer
            residuals.append(residual)
        scale = global_scale = None
        if any(hasattr(residual, "scale") for residual in residuals):
            if not all(
                hasattr(residual, "scale") and hasattr(residual, "weight_global_scale") for residual in residuals
            ):
                raise ValueError("tuned Omni residuals require NVFP4 scale and weight_global_scale")
            global_scale = residuals[0].weight_global_scale.to(device).float().reshape(1)
            if not all(
                torch.equal(global_scale, residual.weight_global_scale.to(device).float().reshape(1))
                for residual in residuals
            ) or not torch.equal(weight, torch.cat([source.residual_weight for source in record.sources])):
                raise ValueError("adapter must preserve tuned residuals and their shared global scale")
            scale = torch.cat([residual.scale.reshape(residual.weight.shape[0], -1) for residual in residuals]).to(
                device
            )
        values = _pack_residual(weight.to(device), scale=scale, global_scale=global_scale)
        values.update(
            proj_down=(down.float() * smooth).T.bfloat16(),
            proj_up=up.bfloat16(),
            smooth_factor=smooth.reciprocal().bfloat16(),
            bias=torch.zeros(n, dtype=torch.bfloat16) if record.bias is None else record.bias.bfloat16(),
        )
        for suffix, value in values.items():
            key = f"{record.prefix}.{suffix}"
            if key in tensors:
                raise ValueError(f"duplicate Omni tensor key {key!r}")
            value = value.detach().cpu().contiguous()
            if value.is_floating_point() and not torch.isfinite(value.float()).all():
                raise ValueError(f"{key}: nonfinite export tensor")
            tensors[key] = value
    if len(ranks) != 1:
        raise ValueError("all exported projections must have the same rank")
    prefixes = tuple(f"{source.name}." for source in sources)
    extras = adapter.extra_tensors(model)
    if not extras:
        extras = {name: value for name, value in model.state_dict().items() if not name.startswith(prefixes)}
    for name, value in extras.items():
        if name in tensors:
            raise ValueError(f"duplicate Omni tensor key {name!r}")
        tensors[name] = value.detach().cpu().contiguous()
    quantization = dict(
        quant_method="svdquant",
        precision="nvfp4",
        rank=ranks.pop(),
        act_unsigned=False,
        modules_to_not_convert=[
            name
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and not f"{name}.".startswith(prefixes)
        ],
    )
    return tensors, quantization


def save_svdquant_omni(
    model: torch.nn.Module, output_dir: str | Path, *, device: str = "cpu", adapter: SVDQuantModelAdapter | None = None
) -> str:
    """Save a Diffusers-style component using the caller-selected model adapter."""
    from safetensors.torch import save_file

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    filename = output / "diffusion_pytorch_model.safetensors"
    if filename.exists():
        raise FileExistsError(filename)
    tensors, quantization = collect_svdquant_omni_tensors(model, device=device, adapter=adapter)
    config = dict(model.config)
    config["quantization_config"] = quantization
    temporary = output / ".svdquant.tmp.safetensors"
    try:
        save_file(tensors, str(temporary), metadata={"format": "pt"})
        os.replace(temporary, filename)
    finally:
        temporary.unlink(missing_ok=True)
    for name, value in (("config.json", config), ("quantization_config.json", quantization)):
        (output / name).write_text(json.dumps(value, indent=2, default=os.fspath) + "\n", encoding="utf-8")
    return str(filename)
