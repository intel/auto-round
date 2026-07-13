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

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Protocol

import torch

from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear
from auto_round.export.svdquant_mxfp4 import NunchakuMXFP4Packer, pack_lowrank_weight


class ResidualTensorProvider(Protocol):
    """Provides packed MXFP4 residual tensors for one logical export record."""

    def tensors_for(self, record: SVDQuantExportRecord) -> Mapping[str, torch.Tensor]:
        """Return tensors keyed by suffix, for example ``qweight`` and ``wscales``."""


@dataclass(frozen=True)
class SourceLinearRecord:
    """Logical tensors from one source ``SVDQuantLinear`` before adapter mapping."""

    name: str
    residual_weight: torch.Tensor
    lora_down: torch.Tensor
    lora_up: torch.Tensor
    smooth: torch.Tensor
    smooth_orig: torch.Tensor
    bias: torch.Tensor | None


@dataclass(frozen=True)
class SVDQuantExportRecord:
    """Adapter-selected logical tensors and their exported key mapping."""

    prefix: str
    residual_weight: torch.Tensor
    lora_down: torch.Tensor
    lora_up: torch.Tensor
    smooth: torch.Tensor
    smooth_orig: torch.Tensor
    bias: torch.Tensor | None
    key_mapping: Mapping[str, str] = field(default_factory=dict)


class SVDQuantModelAdapter(Protocol):
    """Boundary for architecture-specific module mapping and runtime metadata."""

    def map_modules(
        self, model: torch.nn.Module, records: Iterable[SourceLinearRecord]
    ) -> Iterable[SVDQuantExportRecord]: ...

    def metadata(self, model: torch.nn.Module, rank: int) -> Mapping[str, str]: ...

    def validate(self, tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, str]) -> None: ...


class IdentitySVDQuantModelAdapter:
    """Map modules unchanged and explicitly identify a generic intermediate."""

    def map_modules(
        self, model: torch.nn.Module, records: Iterable[SourceLinearRecord]
    ) -> Iterable[SVDQuantExportRecord]:
        return (
            SVDQuantExportRecord(
                prefix=record.name,
                residual_weight=record.residual_weight,
                lora_down=record.lora_down,
                lora_up=record.lora_up,
                smooth=record.smooth,
                smooth_orig=record.smooth_orig,
                bias=record.bias,
            )
            for record in records
        )

    def metadata(self, model: torch.nn.Module, rank: int) -> Mapping[str, str]:
        return {"artifact_type": "generic_intermediate"}

    def validate(self, tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, str]) -> None:
        if metadata.get("artifact_type") != "generic_intermediate":
            raise ValueError("identity adapter output must be marked as a generic intermediate")


@dataclass
class SVDQuantExportConfig:
    """Strict configuration for SVDQuant Nunchaku serialization."""

    weight_dtype: str = "fp4_e2m1_all"
    activation_dtype: str = "fp4_e2m1_all"
    scale_dtype: str = "ue8m0"
    group_size: int = 32
    low_rank_dtype: torch.dtype = torch.bfloat16
    debug_unpacked: bool = False
    runtime_loadable: bool = False

    def __post_init__(self) -> None:
        if self.weight_dtype != "fp4_e2m1_all":
            raise ValueError("weight_dtype must be 'fp4_e2m1_all'")
        if self.activation_dtype != "fp4_e2m1_all":
            raise ValueError("activation_dtype must be 'fp4_e2m1_all'")
        if self.scale_dtype != "ue8m0":
            raise ValueError("scale_dtype must be 'ue8m0'")
        if self.group_size != 32:
            raise ValueError("group_size must be 32")
        if self.low_rank_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("low_rank_dtype must be torch.bfloat16 or torch.float16")
        if not isinstance(self.debug_unpacked, bool):
            raise ValueError("debug_unpacked must be a bool")
        if not isinstance(self.runtime_loadable, bool):
            raise ValueError("runtime_loadable must be a bool")

    def to_quantization_config(self) -> dict:
        config = {
            "method": "svdquant",
            "weight": {
                "dtype": self.weight_dtype,
                "scale_dtype": self.scale_dtype,
                "group_size": self.group_size,
            },
            "activation": {
                "dtype": self.activation_dtype,
                "scale_dtype": self.scale_dtype,
                "group_size": self.group_size,
            },
        }
        return config


def pack_nunchaku_16bit_vector(vector: torch.Tensor) -> torch.Tensor:
    """Pack and pad one BF16/FP16 vector like Nunchaku's scalar scale path."""

    if not isinstance(vector, torch.Tensor) or vector.ndim != 1:
        raise ValueError("vector must be a 1D torch.Tensor")
    if vector.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("vector dtype must be torch.bfloat16 or torch.float16")
    if vector.numel() == 0:
        raise ValueError("vector must be non-empty")
    if not bool(torch.isfinite(vector).all()):
        raise ValueError("vector must contain only finite values")
    padded_size = NunchakuMXFP4Packer._ceil_to(vector.numel(), 128)
    padded = torch.ones(padded_size, dtype=vector.dtype, device=vector.device)
    padded[: vector.numel()] = vector
    packed = padded.reshape(padded_size // 128, 1, 8, 2, 4, 2, 1)
    return packed.permute(0, 6, 1, 2, 4, 3, 5).contiguous().view(-1)


def unpack_nunchaku_16bit_vector(vector: torch.Tensor) -> torch.Tensor:
    """Invert :func:`pack_nunchaku_16bit_vector`, retaining padding."""

    if not isinstance(vector, torch.Tensor) or vector.ndim != 1:
        raise ValueError("vector must be a 1D torch.Tensor")
    if vector.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("vector dtype must be torch.bfloat16 or torch.float16")
    if vector.numel() == 0 or vector.numel() % 128:
        raise ValueError("packed vector length must be a non-zero multiple of 128")
    unpacked = vector.reshape(vector.numel() // 128, 1, 1, 8, 4, 2, 2)
    return unpacked.permute(0, 2, 3, 5, 4, 6, 1).contiguous().view(-1)


class MXFP4ResidualTensorProvider:
    """Quantize and physically pack residual weights as E2M1 with UE8M0 scales."""

    def __init__(self, group_size: int = 32) -> None:
        if group_size != 32:
            raise ValueError("group_size must be 32")
        self.group_size = group_size
        self.packer = NunchakuMXFP4Packer()

    def tensors_for(self, record: SVDQuantExportRecord) -> Mapping[str, torch.Tensor]:
        weight = record.residual_weight
        if not bool(torch.isfinite(weight).all()):
            raise ValueError(f"{record.prefix or '<root>'} residual weight must contain only finite values")
        packed = self.packer.pack_residual(weight, group_size=self.group_size)
        return {"qweight": packed.qweight, "wscales": packed.wscales}


def _source_records(model: torch.nn.Module) -> tuple[SourceLinearRecord, ...]:
    records = []
    for name, module in model.named_modules():
        if not isinstance(module, SVDQuantLinear):
            continue
        records.append(
            SourceLinearRecord(
                name=name,
                residual_weight=module.residual_linear.weight.detach().cpu().contiguous(),
                lora_down=module.lora_down.weight.detach().cpu().contiguous(),
                lora_up=module.lora_up.weight.detach().cpu().contiguous(),
                smooth=module.smooth.detach().cpu().contiguous(),
                smooth_orig=getattr(module, "smooth_orig", module.smooth).detach().cpu().contiguous(),
                bias=None
                if module.residual_linear.bias is None
                else module.residual_linear.bias.detach().cpu().contiguous(),
            )
        )
    if not records:
        raise ValueError("No SVDQuantLinear modules found to export.")
    return tuple(records)


def _validate_export_record(record: SVDQuantExportRecord) -> int:
    tensors = (record.residual_weight, record.lora_down, record.lora_up, record.smooth, record.smooth_orig)
    if not record.prefix or not isinstance(record.prefix, str):
        raise ValueError("export record prefix must be a non-empty string")
    if any(not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point() for tensor in tensors):
        raise ValueError(f"{record.prefix} logical tensors must be floating-point torch.Tensor values")
    if record.residual_weight.ndim != 2 or record.lora_down.ndim != 2 or record.lora_up.ndim != 2:
        raise ValueError(f"{record.prefix} residual and low-rank tensors must be 2D")
    out_features, in_features = record.residual_weight.shape
    rank = record.lora_down.shape[0]
    if (
        record.lora_down.shape[1] != in_features
        or record.lora_up.shape != (out_features, rank)
        or record.smooth.shape != (in_features,)
        or record.smooth_orig.shape != (in_features,)
        or (record.bias is not None and record.bias.shape != (out_features,))
    ):
        raise ValueError(f"{record.prefix} logical tensor shapes are inconsistent")
    finite_tensors = tensors + (() if record.bias is None else (record.bias,))
    if any(not bool(torch.isfinite(tensor).all()) for tensor in finite_tensors):
        raise ValueError(f"{record.prefix} logical tensors must contain only finite values")
    if rank <= 0:
        raise ValueError(f"{record.prefix} rank must be positive")
    return rank


def _export_key(record: SVDQuantExportRecord, suffix: str) -> str:
    mapped_suffix = record.key_mapping.get(suffix, suffix)
    if not isinstance(mapped_suffix, str) or not mapped_suffix:
        raise ValueError(f"{record.prefix} key mapping for {suffix!r} must be a non-empty string")
    return f"{record.prefix}.{mapped_suffix}"


def _validate_packed_residual(payload: Mapping[str, torch.Tensor], prefix: str) -> None:
    if set(payload) != {"qweight", "wscales"}:
        raise ValueError(f"{prefix} packed residual must contain exactly qweight and wscales")
    qweight, wscales = payload["qweight"], payload["wscales"]
    if not isinstance(qweight, torch.Tensor) or qweight.dtype != torch.int8 or qweight.ndim != 2:
        raise ValueError(f"{prefix} qweight must be a 2D torch.int8 tensor")
    if qweight.shape[0] == 0 or qweight.shape[0] % 128 or qweight.shape[1] == 0 or qweight.shape[1] % 64:
        raise ValueError(f"{prefix} qweight shape must encode N and K dimensions divisible by 128")
    expected_scale_shape = (qweight.shape[1] * 2 // 32, qweight.shape[0])
    if not isinstance(wscales, torch.Tensor) or wscales.dtype != torch.uint8:
        raise ValueError(f"{prefix} wscales must be a torch.uint8 tensor")
    if tuple(wscales.shape) != expected_scale_shape:
        raise ValueError(f"{prefix} wscales shape must be {expected_scale_shape}")


def _collect_svdquant_export(
    model: torch.nn.Module,
    config: SVDQuantExportConfig,
    residual_provider: ResidualTensorProvider,
    adapter: SVDQuantModelAdapter,
) -> tuple[dict[str, torch.Tensor], int]:
    records = tuple(adapter.map_modules(model, _source_records(model)))
    if not records:
        raise ValueError("model adapter produced no SVDQuant export records")
    ranks = {_validate_export_record(record) for record in records}
    if len(ranks) != 1:
        raise ValueError(f"mixed SVDQuant ranks are not supported: {sorted(ranks)}")
    tensors: dict[str, torch.Tensor] = {}
    for record in records:
        high_precision = {
            "smooth": pack_nunchaku_16bit_vector(record.smooth.to(config.low_rank_dtype)),
            "smooth_orig": pack_nunchaku_16bit_vector(record.smooth_orig.to(config.low_rank_dtype)),
            "lora_down": pack_lowrank_weight(record.lora_down.to(config.low_rank_dtype), down=True),
            "lora_up": pack_lowrank_weight(record.lora_up.to(config.low_rank_dtype), down=False),
            "bias": pack_nunchaku_16bit_vector(
                torch.zeros(record.residual_weight.shape[0], dtype=config.low_rank_dtype)
                if record.bias is None
                else record.bias.to(config.low_rank_dtype)
            ),
        }
        residual_payload = residual_provider.tensors_for(record)
        _validate_packed_residual(residual_payload, record.prefix)
        serialized = {**high_precision, **residual_payload}
        if config.debug_unpacked:
            serialized["residual.weight"] = record.residual_weight
        for suffix, tensor in serialized.items():
            key = _export_key(record, suffix)
            if key in tensors:
                raise ValueError(f"model adapter produced duplicate tensor key {key!r}")
            tensors[key] = tensor.detach().cpu().contiguous()
    return tensors, ranks.pop()


def collect_svdquant_tensors(
    model: torch.nn.Module,
    *,
    config: SVDQuantExportConfig | None = None,
    residual_provider: ResidualTensorProvider | None = None,
    adapter: SVDQuantModelAdapter | None = None,
) -> dict[str, torch.Tensor]:
    """Collect generic SVDQuant tensors from all ``SVDQuantLinear`` modules."""

    config = config or SVDQuantExportConfig()
    residual_provider = residual_provider or MXFP4ResidualTensorProvider(config.group_size)
    adapter = adapter or IdentitySVDQuantModelAdapter()
    tensors, _ = _collect_svdquant_export(model, config, residual_provider, adapter)
    return tensors


def build_svdquant_metadata(
    model: torch.nn.Module,
    *,
    config: SVDQuantExportConfig | None = None,
    adapter: SVDQuantModelAdapter | None = None,
    rank: int | None = None,
) -> dict[str, str]:
    """Build validated string metadata for one SVDQuant artifact."""

    config = config or SVDQuantExportConfig()
    adapter = adapter or IdentitySVDQuantModelAdapter()
    if rank is None:
        source_ranks = {record.lora_down.shape[0] for record in _source_records(model)}
        if len(source_ranks) != 1:
            raise ValueError(f"mixed SVDQuant ranks are not supported: {sorted(source_ranks)}")
        rank = source_ranks.pop()
    quantization_config = config.to_quantization_config()
    quantization_config["rank"] = rank
    metadata = dict(adapter.metadata(model, rank))
    metadata["quantization_config"] = json.dumps(quantization_config, sort_keys=True)
    if not metadata or any(not isinstance(key, str) or not isinstance(value, str) for key, value in metadata.items()):
        raise ValueError("safetensors metadata keys and values must be strings")
    if config.runtime_loadable:
        if config.debug_unpacked:
            raise ValueError("debug_unpacked output is not runtime-loadable")
        if not metadata.get("model_class") or not metadata.get("config"):
            raise ValueError("runtime-loadable export requires adapter metadata 'model_class' and serialized 'config'")
        try:
            serialized_config = json.loads(metadata["config"])
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("adapter metadata 'config' must be serialized JSON") from exc
        if not isinstance(serialized_config, dict):
            raise ValueError("adapter metadata 'config' must serialize a JSON object")
    return metadata


def save_svdquant_nunchaku_safetensors(
    model: torch.nn.Module,
    output_path: str,
    *,
    config: SVDQuantExportConfig | None = None,
    residual_provider: ResidualTensorProvider | None = None,
    adapter: SVDQuantModelAdapter | None = None,
) -> str:
    """Save decomposed SVDQuant tensors to one safetensors file.

    This function is model-family agnostic. It exports every ``SVDQuantLinear``
    it finds and does not import inference runtimes.
    """

    from safetensors.torch import save_file

    config = config or SVDQuantExportConfig()
    adapter = adapter or IdentitySVDQuantModelAdapter()
    residual_provider = residual_provider or MXFP4ResidualTensorProvider(config.group_size)
    tensors, rank = _collect_svdquant_export(model, config, residual_provider, adapter)
    metadata = build_svdquant_metadata(model, config=config, adapter=adapter, rank=rank)
    adapter.validate(tensors, metadata)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_file(tensors, output_path, metadata=metadata)
    return output_path
