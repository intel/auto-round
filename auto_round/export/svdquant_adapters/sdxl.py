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

"""SDXL model mapping for runtime-loadable SVDQuant Nunchaku artifacts."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from auto_round.export.svdquant_nunchaku import SourceLinearRecord, SVDQuantExportRecord

_BLOCK_RE = re.compile(r"^(.*\.transformer_blocks\.\d+)\.(.+)$")
_QKV_SOURCES = ("attn1.to_q", "attn1.to_k", "attn1.to_v")
_DIRECT_SOURCES = (
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)

SDXL_SVDQUANT_TARGET_MODULES = (
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)


def _config_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    if hasattr(value, "__dict__"):
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
    raise ValueError("SDXL config must be a mapping or serialize to a JSON object")


def _normalize_config_paths(value: Any) -> Any:
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {key: _normalize_config_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_config_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_config_paths(item) for item in value)
    return value


def is_sdxl_unet_config(config: Mapping[str, Any], class_name: str = "") -> bool:
    """Return whether a Diffusers config matches the SDXL UNet runtime contract."""

    resolved_class = str(config.get("_class_name", class_name)).lower()
    return (
        resolved_class == "unet2dconditionmodel"
        and config.get("addition_embed_type") == "text_time"
        and config.get("cross_attention_dim") == 2048
        and config.get("projection_class_embeddings_input_dim") == 2816
    )


def _effective_weight(source: SourceLinearRecord, device: torch.device) -> torch.Tensor:
    out_features, in_features = source.residual_weight.shape
    rank = source.lora_down.shape[0]
    if (
        source.residual_weight.ndim != 2
        or source.lora_down.shape != (rank, in_features)
        or source.lora_up.shape != (out_features, rank)
        or source.smooth.shape != (in_features,)
    ):
        raise ValueError(f"{source.name} source dimensions are inconsistent")
    residual = source.residual_weight.detach().to(device=device, dtype=torch.float32)
    up = source.lora_up.detach().to(device=device, dtype=torch.float32)
    down = source.lora_down.detach().to(device=device, dtype=torch.float32)
    smooth = source.smooth.detach().to(device=device, dtype=torch.float32)
    return (residual + up @ down) * smooth.reshape(1, -1)


def _has_shared_input_decomposition(sources: tuple[SourceLinearRecord, ...]) -> bool:
    first = sources[0]
    return all(
        source.scheme == first.scheme
        and torch.equal(source.lora_down, first.lora_down)
        and torch.equal(source.smooth, first.smooth)
        and torch.equal(source.smooth_orig, first.smooth_orig)
        for source in sources[1:]
    )


def _fused_bias(prefix: str, sources: tuple[SourceLinearRecord, ...], device: torch.device | None = None):
    biases = [source.bias for source in sources]
    if any(bias is None for bias in biases) and not all(bias is None for bias in biases):
        raise ValueError(f"{prefix} fused sources must either all have bias or all omit bias")
    if biases[0] is None:
        return None
    values = biases if device is None else [bias.to(device) for bias in biases]
    return torch.cat(values).detach().cpu().contiguous() if device is None else torch.cat(values)


@dataclass
class SDXLSVDQuantNunchakuAdapter:
    """Map Diffusers SDXL UNet projections to the Nunchaku SDXL tensor schema."""

    config: Mapping[str, Any] | None = None
    decomposition_device: str | torch.device = "cpu"
    require_complete_model: bool = True

    def __post_init__(self) -> None:
        try:
            self.decomposition_device = torch.device(self.decomposition_device)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"invalid SDXL decomposition_device {self.decomposition_device!r}") from exc
        if self.decomposition_device.type not in ("cpu", "cuda"):
            raise ValueError("SDXL decomposition_device must be CPU or CUDA")
        if self.decomposition_device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("SDXL decomposition_device requests CUDA, but CUDA is not available")
        if self.config is not None:
            self.config = _config_dict(self.config)

    def _resolved_config(self, model: torch.nn.Module) -> dict[str, Any]:
        value = self.config if self.config is not None else getattr(model, "config", None)
        if value is None:
            raise ValueError("SDXL export requires explicit config or model.config")
        config = _normalize_config_paths(_config_dict(value))
        if not is_sdxl_unet_config(config, type(model).__name__):
            raise ValueError("SDXL adapter requires an SDXL UNet2DConditionModel config")
        try:
            json.dumps(config)
        except (TypeError, ValueError) as exc:
            raise ValueError("SDXL config must be JSON serializable") from exc
        return config

    def metadata(self, model: torch.nn.Module, rank: int) -> Mapping[str, str]:
        return {
            "model_class": "NunchakuSDXLUNet2DConditionModel",
            "config": json.dumps(self._resolved_config(model), sort_keys=True),
            "format": "pt",
            "comfy_config": "{}",
        }

    @staticmethod
    def _direct(source: SourceLinearRecord) -> SVDQuantExportRecord:
        return SVDQuantExportRecord(
            prefix=source.name,
            residual_weight=source.residual_weight,
            lora_down=source.lora_down,
            lora_up=source.lora_up,
            smooth=source.smooth,
            smooth_orig=source.smooth_orig,
            bias=source.bias,
            scheme=source.scheme,
            sources=(source,),
        )

    def _fuse_qkv(self, prefix: str, sources: tuple[SourceLinearRecord, ...], rank: int) -> SVDQuantExportRecord:
        if _has_shared_input_decomposition(sources):
            first = sources[0]
            return SVDQuantExportRecord(
                prefix=prefix,
                residual_weight=torch.cat([source.residual_weight for source in sources]).detach().cpu().contiguous(),
                lora_down=first.lora_down.detach().cpu().contiguous(),
                lora_up=torch.cat([source.lora_up for source in sources]).detach().cpu().contiguous(),
                smooth=first.smooth.detach().cpu().contiguous(),
                smooth_orig=first.smooth_orig.detach().cpu().contiguous(),
                bias=_fused_bias(prefix, sources),
                scheme=first.scheme,
                sources=sources,
            )
        try:
            effective = [_effective_weight(source, self.decomposition_device) for source in sources]
            input_dims = {weight.shape[1] for weight in effective}
            if len(input_dims) != 1:
                raise ValueError(f"{prefix} fused sources have incompatible input dimensions {sorted(input_dims)}")
            weight = torch.cat(effective, dim=0)
            if rank > min(weight.shape):
                raise ValueError(f"{prefix} configured rank={rank} exceeds fused dimensions {tuple(weight.shape)}")
            if not bool(torch.isfinite(weight).all()):
                raise ValueError(f"{prefix} effective weight contains non-finite values")
            u, singular_values, vh = torch.linalg.svd(weight, full_matrices=False)
            up = u[:, :rank] * singular_values[:rank]
            down = vh[:rank]
            residual = weight - up @ down
            template = sources[0]
            return SVDQuantExportRecord(
                prefix=prefix,
                residual_weight=residual.to(dtype=template.residual_weight.dtype).cpu().contiguous(),
                lora_down=down.to(dtype=template.lora_down.dtype).cpu().contiguous(),
                lora_up=up.to(dtype=template.lora_up.dtype).cpu().contiguous(),
                smooth=torch.ones(weight.shape[1], dtype=template.smooth.dtype).cpu(),
                smooth_orig=torch.ones(weight.shape[1], dtype=template.smooth_orig.dtype).cpu(),
                bias=(
                    _fused_bias(prefix, sources, self.decomposition_device).cpu().contiguous()
                    if sources[0].bias is not None
                    else None
                ),
                scheme=template.scheme,
                sources=sources,
            )
        finally:
            if self.decomposition_device.type == "cuda":
                torch.cuda.empty_cache()

    @staticmethod
    def _expected_blocks(model: torch.nn.Module) -> set[str]:
        return {name for name, module in model.named_modules() if type(module).__name__ == "BasicTransformerBlock"}

    def map_modules(
        self, model: torch.nn.Module, records: Iterable[SourceLinearRecord]
    ) -> Iterable[SVDQuantExportRecord]:
        self._resolved_config(model)
        by_block: dict[str, dict[str, SourceLinearRecord]] = {}
        for record in records:
            match = _BLOCK_RE.match(record.name)
            if match is None:
                raise ValueError(f"unrecognized SDXL SVDQuant source {record.name!r}")
            block_prefix, local_name = match.groups()
            local = by_block.setdefault(block_prefix, {})
            if local_name in local:
                raise ValueError(f"duplicate SDXL source {record.name!r}")
            local[local_name] = record

        required = set(SDXL_SVDQUANT_TARGET_MODULES)
        if self.require_complete_model:
            expected_blocks = self._expected_blocks(model)
            if set(by_block) != expected_blocks:
                raise ValueError(
                    f"complete SDXL block coverage mismatch: expected {sorted(expected_blocks)}, got {sorted(by_block)}"
                )
        output: list[SVDQuantExportRecord] = []
        for block_prefix, local in sorted(by_block.items()):
            names = set(local)
            if self.require_complete_model and names != required:
                raise ValueError(
                    f"complete SDXL {block_prefix} coverage: missing={sorted(required - names)}, "
                    f"extras={sorted(names - required)}"
                )
            present_qkv = tuple(local[name] for name in _QKV_SOURCES if name in local)
            if present_qkv and len(present_qkv) != len(_QKV_SOURCES):
                missing = [name for name in _QKV_SOURCES if name not in local]
                raise ValueError(f"{block_prefix}.attn1.to_qkv missing fused sources {missing}")
            if present_qkv:
                rank = present_qkv[0].lora_down.shape[0]
                output.append(self._fuse_qkv(f"{block_prefix}.attn1.to_qkv", present_qkv, rank))
            output.extend(self._direct(local[name]) for name in _DIRECT_SOURCES if name in local)
        return output

    def validate_records(
        self, sources: tuple[SourceLinearRecord, ...], records: tuple[SVDQuantExportRecord, ...]
    ) -> None:
        ranks = {source.lora_down.shape[0] for source in sources}
        if len(ranks) != 1:
            raise ValueError(f"SDXL source ranks must agree, got {sorted(ranks)}")
        prefixes = [record.prefix for record in records]
        if len(prefixes) != len(set(prefixes)):
            raise ValueError("SDXL adapter produced duplicate logical record prefixes")

    def extra_tensors(self, model: torch.nn.Module) -> Mapping[str, torch.Tensor]:
        """Copy every runtime tensor not replaced by an SVDQW4A4Linear."""

        from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear

        quantized_prefixes = tuple(
            f"{name}." for name, module in model.named_modules() if name and isinstance(module, SVDQuantLinear)
        )
        tensors: dict[str, torch.Tensor] = {}
        for name, tensor in model.state_dict().items():
            if name.startswith(quantized_prefixes):
                continue
            value = tensor.detach()
            if value.is_floating_point():
                value = value.to(torch.bfloat16)
            tensors[name] = value.cpu().contiguous()
        self._passthrough_keys = frozenset(tensors)
        return tensors

    def validate(self, tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, str]) -> None:
        if metadata.get("model_class") != "NunchakuSDXLUNet2DConditionModel":
            raise ValueError("SDXL metadata has incorrect model_class")
        if metadata.get("format") != "pt" or metadata.get("comfy_config") != "{}":
            raise ValueError("SDXL metadata requires format='pt' and empty comfy_config")
        try:
            config = json.loads(metadata["config"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError("SDXL metadata config must be a JSON object") from exc
        if not isinstance(config, dict) or not is_sdxl_unet_config(config):
            raise ValueError("SDXL metadata config must describe an SDXL UNet")

        passthrough_keys = getattr(self, "_passthrough_keys", frozenset())
        missing_passthrough = passthrough_keys - tensors.keys()
        if missing_passthrough:
            raise ValueError(f"SDXL artifact is missing passthrough tensors: {sorted(missing_passthrough)[:5]}")
        internal_markers = (".residual_linear.", ".lora_down.", ".lora_up.")
        for key, tensor in tensors.items():
            if any(marker in key for marker in internal_markers):
                raise ValueError(f"SDXL artifact exposes wrapper-internal tensor {key!r}")
            if tensor.device.type != "cpu" or not tensor.is_contiguous():
                raise ValueError(f"SDXL tensor {key!r} must be contiguous on CPU")
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"SDXL tensor {key!r} must be finite")
            if key in passthrough_keys:
                continue
            if key.endswith(".qweight"):
                if tensor.dtype != torch.int8 or tensor.ndim != 2:
                    raise ValueError(f"SDXL qweight {key!r} must be 2D int8")
            elif key.endswith(".wscales"):
                if tensor.dtype != torch.uint8 or tensor.ndim != 2:
                    raise ValueError(f"SDXL wscales {key!r} must be 2D uint8")
            elif key.endswith((".lora_down", ".lora_up")):
                if tensor.dtype != torch.bfloat16 or tensor.ndim != 2:
                    raise ValueError(f"SDXL low-rank tensor {key!r} must be 2D BF16")
            elif key.endswith((".smooth", ".smooth_orig", ".bias")):
                if tensor.dtype != torch.bfloat16 or tensor.ndim != 1:
                    raise ValueError(f"SDXL vector tensor {key!r} must be 1D BF16")
            else:
                raise ValueError(f"SDXL artifact contains unknown packed tensor {key!r}")


__all__ = ["SDXL_SVDQUANT_TARGET_MODULES", "SDXLSVDQuantNunchakuAdapter", "is_sdxl_unet_config"]
