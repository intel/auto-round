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


_DEPLOYABLE_E2M1_ALIASES = frozenset({"mx_fp", "mx_fp4", "mx_fp4e2m1"})


class ResidualTensorProvider(Protocol):
    """Provides packed MXFP4 residual tensors for one logical export record."""

    def tensors_for(self, record: SVDQuantExportRecord) -> Mapping[str, torch.Tensor]:
        """Return tensors keyed by suffix, for example ``qweight`` and ``wscales``."""


@dataclass(frozen=True)
class SVDQuantLinearScheme:
    """AutoRound-selected weight and activation scheme values."""

    data_type: str | None
    bits: int | None
    group_size: int | tuple[int, int] | None
    sym: bool | None
    act_data_type: str | None
    act_bits: int | None
    act_group_size: int | tuple[int, int] | None
    act_sym: bool | None
    act_dynamic: bool | None


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
    scheme: SVDQuantLinearScheme


@dataclass(frozen=True)
class SVDQuantExportRecord:
    """Adapter-selected logical tensors and their exported key mapping.

    Fusion adapters must reconstruct effective source weights, fuse or split those
    weights, and recompute the low-rank decomposition at the configured output
    rank. Nunchaku metadata stores one rank, so exact rank-sum LoRA fusion is not
    representable by this export schema.
    """

    prefix: str
    residual_weight: torch.Tensor
    lora_down: torch.Tensor
    lora_up: torch.Tensor
    smooth: torch.Tensor
    smooth_orig: torch.Tensor
    bias: torch.Tensor | None
    scheme: SVDQuantLinearScheme
    sources: tuple[SourceLinearRecord, ...]
    key_mapping: Mapping[str, str] = field(default_factory=dict)


class SVDQuantModelAdapter(Protocol):
    """Boundary for model-level mapping and runtime metadata.

    ``map_modules`` receives every logical source together so architecture adapters
    can recompose effective weights before sibling fusion or splitting. Its output
    must retain source provenance and use the configured source rank.
    """

    def map_modules(
        self, model: torch.nn.Module, records: Iterable[SourceLinearRecord]
    ) -> Iterable[SVDQuantExportRecord]: ...

    def metadata(self, model: torch.nn.Module, rank: int) -> Mapping[str, str]: ...

    def validate_records(
        self, sources: tuple[SourceLinearRecord, ...], records: tuple[SVDQuantExportRecord, ...]
    ) -> None: ...

    def validate(self, tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, str]) -> None: ...


class IdentitySVDQuantModelAdapter:
    """Map modules unchanged and explicitly identify a generic intermediate."""

    def map_modules(
        self, model: torch.nn.Module, records: Iterable[SourceLinearRecord]
    ) -> Iterable[SVDQuantExportRecord]:
        return (
            SVDQuantExportRecord(
                prefix=record.name or "model",
                residual_weight=record.residual_weight,
                lora_down=record.lora_down,
                lora_up=record.lora_up,
                smooth=record.smooth,
                smooth_orig=record.smooth_orig,
                bias=record.bias,
                scheme=record.scheme,
                sources=(record,),
            )
            for record in records
        )

    def metadata(self, model: torch.nn.Module, rank: int) -> Mapping[str, str]:
        return {"artifact_type": "generic_intermediate"}

    def validate_records(
        self, sources: tuple[SourceLinearRecord, ...], records: tuple[SVDQuantExportRecord, ...]
    ) -> None:
        return

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
        if isinstance(self.group_size, bool) or not isinstance(self.group_size, int) or self.group_size != 32:
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
        if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size != 32:
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
                residual_weight=module.residual_linear.weight.detach(),
                lora_down=module.lora_down.weight.detach(),
                lora_up=module.lora_up.weight.detach(),
                smooth=module.smooth.detach(),
                smooth_orig=getattr(module, "smooth_orig", module.smooth).detach(),
                bias=None
                if module.residual_linear.bias is None
                else module.residual_linear.bias.detach(),
                scheme=SVDQuantLinearScheme(
                    data_type=getattr(module.residual_linear, "data_type", None),
                    bits=getattr(module.residual_linear, "bits", None),
                    group_size=getattr(module.residual_linear, "group_size", None),
                    sym=getattr(module.residual_linear, "sym", None),
                    act_data_type=getattr(module.residual_linear, "act_data_type", None),
                    act_bits=getattr(module.residual_linear, "act_bits", None),
                    act_group_size=getattr(module.residual_linear, "act_group_size", None),
                    act_sym=getattr(module.residual_linear, "act_sym", None),
                    act_dynamic=getattr(module.residual_linear, "act_dynamic", None),
                ),
            )
        )
    if not records:
        raise ValueError("No SVDQuantLinear modules found to export.")
    return tuple(records)


def _validate_selected_scheme(scheme: SVDQuantLinearScheme, prefix: str) -> tuple:
    required_weight = {
        "data_type": scheme.data_type,
        "bits": scheme.bits,
        "group_size": scheme.group_size,
        "sym": scheme.sym,
    }
    missing_weight = [name for name, value in required_weight.items() if value is None]
    if missing_weight:
        raise ValueError(f"{prefix} selected residual scheme is missing required {missing_weight[0]}")
    if not isinstance(scheme.data_type, str) or scheme.data_type not in _DEPLOYABLE_E2M1_ALIASES:
        raise ValueError(
            f"{prefix} residual data_type must be one of {sorted(_DEPLOYABLE_E2M1_ALIASES)}, got {scheme.data_type!r}"
        )
    if isinstance(scheme.bits, bool) or not isinstance(scheme.bits, int) or scheme.bits != 4:
        raise ValueError(f"{prefix} residual scheme requires bits=4, got {scheme.bits!r}")
    if (
        isinstance(scheme.group_size, bool)
        or not isinstance(scheme.group_size, int)
        or scheme.group_size != 32
    ):
        raise ValueError(f"{prefix} residual scheme requires scalar group_size=32, got {scheme.group_size!r}")
    if scheme.sym is not True:
        raise ValueError(f"{prefix} residual scheme requires sym=True, got {scheme.sym!r}")

    required_activation = {
        "act_data_type": scheme.act_data_type,
        "act_bits": scheme.act_bits,
        "act_group_size": scheme.act_group_size,
        "act_sym": scheme.act_sym,
        "act_dynamic": scheme.act_dynamic,
    }
    missing_activation = [name for name, value in required_activation.items() if value is None]
    if missing_activation:
        raise ValueError(f"{prefix} selected scheme is missing required activation value {missing_activation[0]}")
    if not isinstance(scheme.act_data_type, str) or scheme.act_data_type not in _DEPLOYABLE_E2M1_ALIASES:
        raise ValueError(
            f"{prefix} activation data_type must be one of {sorted(_DEPLOYABLE_E2M1_ALIASES)}, "
            f"got {scheme.act_data_type!r}"
        )
    if isinstance(scheme.act_bits, bool) or not isinstance(scheme.act_bits, int) or scheme.act_bits != 4:
        raise ValueError(f"{prefix} activation scheme requires activation bits=4, got {scheme.act_bits!r}")
    if (
        isinstance(scheme.act_group_size, bool)
        or not isinstance(scheme.act_group_size, int)
        or scheme.act_group_size != 32
    ):
        raise ValueError(
            f"{prefix} activation scheme requires activation scalar group_size=32, got {scheme.act_group_size!r}"
        )
    if scheme.act_sym is not True:
        raise ValueError(f"{prefix} activation scheme requires activation sym=True, got {scheme.act_sym!r}")
    if scheme.act_dynamic is not True:
        raise ValueError(f"{prefix} activation scheme requires act_dynamic=True, got {scheme.act_dynamic!r}")
    return ("mx_fp4e2m1", 4, 32, True, "mx_fp4e2m1", 4, 32, True, True)


def _validate_export_record(record: SVDQuantExportRecord, config: SVDQuantExportConfig) -> int:
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
    selected_scheme = _validate_selected_scheme(record.scheme, record.prefix)
    if not record.sources or any(not isinstance(source, SourceLinearRecord) for source in record.sources):
        raise ValueError(f"{record.prefix} export record must retain at least one logical source record")
    source_schemes = {_validate_selected_scheme(source.scheme, source.name) for source in record.sources}
    if len(source_schemes) != 1 or selected_scheme not in source_schemes:
        raise ValueError(f"{record.prefix} adapter sources have incompatible selected quantization schemes")
    if config.group_size != selected_scheme[2]:
        raise ValueError(
            f"{record.prefix} export group_size={config.group_size} disagrees with "
            f"selected group_size={selected_scheme[2]}"
        )
    return rank


def _export_key(record: SVDQuantExportRecord, suffix: str) -> str:
    mapped_suffix = record.key_mapping.get(suffix, suffix)
    if not isinstance(mapped_suffix, str) or not mapped_suffix:
        raise ValueError(f"{record.prefix} key mapping for {suffix!r} must be a non-empty string")
    return f"{record.prefix}.{mapped_suffix}"


def _validate_packed_residual(payload: Mapping[str, torch.Tensor], record: SVDQuantExportRecord) -> None:
    prefix = record.prefix
    if set(payload) != {"qweight", "wscales"}:
        raise ValueError(f"{prefix} packed residual must contain exactly qweight and wscales")
    qweight, wscales = payload["qweight"], payload["wscales"]
    if not isinstance(qweight, torch.Tensor) or qweight.dtype != torch.int8 or qweight.ndim != 2:
        raise ValueError(f"{prefix} qweight must be a 2D torch.int8 tensor")
    out_features, in_features = record.residual_weight.shape
    padded_out = NunchakuMXFP4Packer._ceil_to(out_features, 128)
    padded_in = NunchakuMXFP4Packer._ceil_to(in_features, 128)
    expected_weight_shape = (padded_out, padded_in // 2)
    if tuple(qweight.shape) != expected_weight_shape:
        raise ValueError(f"{prefix} qweight shape must be {expected_weight_shape}")
    expected_scale_shape = (padded_in // 32, padded_out)
    if not isinstance(wscales, torch.Tensor) or wscales.dtype != torch.uint8:
        raise ValueError(f"{prefix} wscales must be a torch.uint8 tensor")
    if tuple(wscales.shape) != expected_scale_shape:
        raise ValueError(f"{prefix} wscales shape must be {expected_scale_shape}")


def _validate_adapter_provenance(
    sources: tuple[SourceLinearRecord, ...], records: tuple[SVDQuantExportRecord, ...]
) -> None:
    """Require complete source identity coverage and configured-rank outputs."""

    source_ids = {id(source) for source in sources}
    referenced_ids: set[int] = set()
    for record in records:
        output_rank = record.lora_down.shape[0]
        for source in record.sources:
            if id(source) not in source_ids:
                raise ValueError(f"{record.prefix} adapter record references a foreign logical source")
            if source.lora_down.shape[0] != output_rank:
                raise ValueError(
                    f"{record.prefix} source rank={source.lora_down.shape[0]} must equal output rank={output_rank}; "
                    "the Nunchaku schema stores one configured rank, so adapters must recompose effective source "
                    "weights and decompose them at that configured rank; exact rank-sum fusion is unsupported"
                )
            referenced_ids.add(id(source))
    missing = [source.name or "<root>" for source in sources if id(source) not in referenced_ids]
    if missing:
        raise ValueError(f"model adapter dropped logical sources: {missing}")


def _prepare_export_records(
    model: torch.nn.Module,
    config: SVDQuantExportConfig,
    adapter: SVDQuantModelAdapter,
) -> tuple[tuple[SourceLinearRecord, ...], tuple[SVDQuantExportRecord, ...], int]:
    source_records = _source_records(model)
    for source in source_records:
        _validate_selected_scheme(source.scheme, source.name or "<root>")
    records = tuple(adapter.map_modules(model, source_records))
    if not records:
        raise ValueError("model adapter produced no SVDQuant export records")
    ranks = {_validate_export_record(record, config) for record in records}
    if len(ranks) != 1:
        raise ValueError(f"mixed SVDQuant ranks are not supported: {sorted(ranks)}")
    _validate_adapter_provenance(source_records, records)
    validate_records = getattr(adapter, "validate_records", None)
    if validate_records is not None:
        validate_records(source_records, records)
    return source_records, records, ranks.pop()


def _serialize_export_records(
    records: tuple[SVDQuantExportRecord, ...],
    config: SVDQuantExportConfig,
    residual_provider: ResidualTensorProvider,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for record in records:
        high_precision = {
            "smooth": pack_nunchaku_16bit_vector(record.smooth.to(config.low_rank_dtype)),
            "smooth_orig": pack_nunchaku_16bit_vector(record.smooth_orig.to(config.low_rank_dtype)),
            "lora_down": pack_lowrank_weight(record.lora_down.to(config.low_rank_dtype), down=True),
            "lora_up": pack_lowrank_weight(record.lora_up.to(config.low_rank_dtype), down=False),
            "bias": pack_nunchaku_16bit_vector(
                torch.zeros(
                    record.residual_weight.shape[0],
                    dtype=config.low_rank_dtype,
                    device=record.residual_weight.device,
                )
                if record.bias is None
                else record.bias.to(config.low_rank_dtype)
            ),
        }
        residual_payload = residual_provider.tensors_for(record)
        _validate_packed_residual(residual_payload, record)
        serialized = {**high_precision, **residual_payload}
        if config.debug_unpacked:
            serialized["residual.weight"] = record.residual_weight
        for suffix, tensor in serialized.items():
            key = _export_key(record, suffix)
            if key in tensors:
                raise ValueError(f"model adapter produced duplicate tensor key {key!r}")
            tensors[key] = tensor.detach().cpu().contiguous()
    return tensors


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
    _, records, rank = _prepare_export_records(model, config, adapter)
    if config.runtime_loadable:
        build_svdquant_metadata(model, config=config, adapter=adapter, rank=rank)
    return _serialize_export_records(records, config, residual_provider)


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
        _, _, rank = _prepare_export_records(model, config, adapter)
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
    _, records, rank = _prepare_export_records(model, config, adapter)
    metadata = build_svdquant_metadata(model, config=config, adapter=adapter, rank=rank)
    tensors = _serialize_export_records(records, config, residual_provider)
    adapter.validate(tensors, metadata)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    save_file(tensors, output_path, metadata=metadata)
    return output_path
