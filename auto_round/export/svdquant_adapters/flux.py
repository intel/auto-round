# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""FLUX model mapping for runtime-loadable SVDQuant Nunchaku artifacts."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from auto_round.export.svdquant_nunchaku import SourceLinearRecord, SVDQuantExportRecord
from auto_round.export.svdquant_w4a16 import quantize_adanorm_w4a16_rtn

_BLOCK_RE = re.compile(r"^(transformer_blocks|single_transformer_blocks)\.(\d+)\.(.+)$")
_DOUBLE_DIRECT = {
    "attn.to_out.0": "out_proj",
    "attn.to_add_out": "out_proj_context",
    "ff.net.0.proj": "mlp_fc1",
    "ff.net.2": "mlp_fc2",
    "ff.net.2.linear": "mlp_fc2",
    "ff_context.net.0.proj": "mlp_context_fc1",
    "ff_context.net.2": "mlp_context_fc2",
    "ff_context.net.2.linear": "mlp_context_fc2",
}
_DOUBLE_FUSED = {
    "qkv_proj": ("attn.to_q", "attn.to_k", "attn.to_v"),
    "qkv_proj_context": ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"),
}
_SINGLE_FUSED = {"qkv_proj": ("attn.to_q", "attn.to_k", "attn.to_v")}
_RMS_MAP = {
    "attn.norm_q": "norm_q",
    "attn.norm_k": "norm_k",
    "attn.norm_added_q": "norm_added_q",
    "attn.norm_added_k": "norm_added_k",
}
_TOP_LEVEL_PREFIXES = ("x_embedder.", "context_embedder.", "time_text_embed.", "norm_out.linear.", "proj_out.")


def flux_onefile_tensor_count(num_layers: int, num_single_layers: int, top_level_tensors: int = 20) -> int:
    """Return the key count without constructing model-sized tensors."""

    return num_layers * (8 * 7 + 2 * 4 + 4) + num_single_layers * (4 * 7 + 4 + 2) + top_level_tensors


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
    raise ValueError("FLUX config must be a mapping or serialize to a JSON object")


def _effective_weight(source: SourceLinearRecord, device: torch.device) -> torch.Tensor:
    if source.residual_weight.ndim != 2 or source.lora_down.ndim != 2 or source.lora_up.ndim != 2:
        raise ValueError(f"{source.name} residual and low-rank weights must be 2D")
    out_features, in_features = source.residual_weight.shape
    rank = source.lora_down.shape[0]
    if (
        source.lora_down.shape != (rank, in_features)
        or source.lora_up.shape != (out_features, rank)
        or source.smooth.shape != (in_features,)
    ):
        raise ValueError(f"{source.name} source dimensions are inconsistent")
    residual = source.residual_weight.detach().to(device=device, dtype=torch.float32)
    up = source.lora_up.detach().to(device=device, dtype=torch.float32)
    down = source.lora_down.detach().to(device=device, dtype=torch.float32)
    smooth = source.smooth.detach().to(device=device, dtype=torch.float32)
    return (residual + up @ down) * smooth.reshape(1, -1)


def _decompose(
    weight: torch.Tensor,
    *,
    rank: int,
    template: SourceLinearRecord,
    prefix: str,
    sources: tuple[SourceLinearRecord, ...],
    bias: torch.Tensor | None,
) -> SVDQuantExportRecord:
    if rank > min(weight.shape):
        raise ValueError(f"{prefix} configured rank={rank} exceeds fused dimensions {tuple(weight.shape)}")
    if not bool(torch.isfinite(weight).all()):
        raise ValueError(f"{prefix} effective weight contains non-finite values")
    u, singular_values, vh = torch.linalg.svd(weight, full_matrices=False)
    up = u[:, :rank] * singular_values[:rank]
    down = vh[:rank]
    residual = weight - up @ down
    weight_dtype = template.residual_weight.dtype
    low_rank_dtype = template.lora_down.dtype
    in_features = weight.shape[1]
    record = SVDQuantExportRecord(
        prefix=prefix,
        residual_weight=residual.to(dtype=weight_dtype).cpu().contiguous(),
        lora_down=down.to(dtype=low_rank_dtype).cpu().contiguous(),
        lora_up=up.to(dtype=template.lora_up.dtype).cpu().contiguous(),
        smooth=torch.ones(in_features, dtype=template.smooth.dtype).cpu(),
        smooth_orig=torch.ones(in_features, dtype=template.smooth_orig.dtype).cpu(),
        bias=None if bias is None else bias.detach().to(dtype=weight_dtype).cpu().contiguous(),
        scheme=template.scheme,
        sources=sources,
    )
    del u, singular_values, vh, up, down, residual, weight
    return record


@dataclass
class FluxSVDQuantNunchakuAdapter:
    """Map Diffusers FLUX modules to the Nunchaku one-file tensor schema."""

    config: Mapping[str, Any] | None = None
    decomposition_device: str | torch.device = "cpu"
    require_complete_model: bool = True

    def __post_init__(self) -> None:
        self.decomposition_device = torch.device(self.decomposition_device)
        if self.decomposition_device.type != "cpu":
            raise ValueError("FLUX decomposition_device must be cpu")
        if self.config is not None:
            self.config = _config_dict(self.config)
        if not isinstance(self.require_complete_model, bool):
            raise ValueError("require_complete_model must be a bool")

    def _resolved_config(self, model: torch.nn.Module) -> dict[str, Any]:
        value = self.config if self.config is not None else getattr(model, "config", None)
        if value is None:
            raise ValueError("FLUX export requires explicit config or model.config")
        config = _config_dict(value)
        try:
            json.dumps(config)
        except (TypeError, ValueError) as exc:
            raise ValueError("FLUX config must be JSON serializable") from exc
        return config

    def metadata(self, model: torch.nn.Module, rank: int) -> Mapping[str, str]:
        return {
            "model_class": "NunchakuFluxTransformer2dModel",
            "config": json.dumps(self._resolved_config(model), sort_keys=True),
            "format": "pt",
            "comfy_config": "{}",
        }

    @staticmethod
    def _direct(source: SourceLinearRecord, prefix: str) -> SVDQuantExportRecord:
        return SVDQuantExportRecord(
            prefix=prefix,
            residual_weight=source.residual_weight,
            lora_down=source.lora_down,
            lora_up=source.lora_up,
            smooth=source.smooth,
            smooth_orig=source.smooth_orig,
            bias=source.bias,
            scheme=source.scheme,
            sources=(source,),
        )

    def _fuse(self, prefix: str, sources: tuple[SourceLinearRecord, ...], rank: int) -> SVDQuantExportRecord:
        effective = [_effective_weight(source, self.decomposition_device) for source in sources]
        input_dims = {weight.shape[1] for weight in effective}
        if len(input_dims) != 1:
            raise ValueError(f"{prefix} fused sources have incompatible input dimensions {sorted(input_dims)}")
        weight = torch.cat(effective, dim=0)
        del effective
        biases = [source.bias for source in sources]
        if any(bias is None for bias in biases) and not all(bias is None for bias in biases):
            raise ValueError(f"{prefix} fused sources must either all have bias or all omit bias")
        bias = None if biases[0] is None else torch.cat([item.to(self.decomposition_device) for item in biases])
        return _decompose(weight, rank=rank, template=sources[0], prefix=prefix, sources=sources, bias=bias)

    def map_modules(
        self, model: torch.nn.Module, records: Iterable[SourceLinearRecord]
    ) -> Iterable[SVDQuantExportRecord]:
        records = tuple(records)
        by_block: dict[tuple[str, int], dict[str, SourceLinearRecord]] = {}
        for record in records:
            match = _BLOCK_RE.match(record.name)
            if match is None:
                raise ValueError(f"unrecognized FLUX SVDQuant source {record.name!r}")
            family, index, local_name = match.groups()
            local = by_block.setdefault((family, int(index)), {})
            if local_name in local:
                raise ValueError(f"duplicate FLUX source {record.name!r}")
            local[local_name] = record

        if self.require_complete_model:
            config = self._resolved_config(model)
            try:
                expected_indices = {
                    "transformer_blocks": set(range(int(config["num_layers"]))),
                    "single_transformer_blocks": set(range(int(config["num_single_layers"]))),
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("complete FLUX export requires integer num_layers and num_single_layers") from exc
            for family, expected in expected_indices.items():
                actual = {index for candidate, index in by_block if candidate == family}
                if actual != expected:
                    raise ValueError(
                        f"complete FLUX export {family} indices mismatch: expected {sorted(expected)}, "
                        f"got {sorted(actual)}"
                    )
            double_fixed = {name for names in _DOUBLE_FUSED.values() for name in names} | {
                "attn.to_out.0",
                "attn.to_add_out",
                "ff.net.0.proj",
                "ff_context.net.0.proj",
            }
            for (family, index), local in by_block.items():
                names = set(local)
                if family == "single_transformer_blocks":
                    required = {"attn.to_q", "attn.to_k", "attn.to_v", "proj_mlp", "proj_out"}
                    missing, extra = required - names, names - required
                else:
                    missing = double_fixed - names
                    extra = (
                        names
                        - set(_DOUBLE_DIRECT)
                        - {name for fused_names in _DOUBLE_FUSED.values() for name in fused_names}
                    )
                    for stem in ("ff.net.2", "ff_context.net.2"):
                        variants = {stem, f"{stem}.linear"}
                        present = names & variants
                        if len(present) != 1:
                            missing.add(f"exactly one of {sorted(variants)}")
                if missing or extra:
                    raise ValueError(
                        f"complete FLUX export rejects {family}.{index} coverage: "
                        f"missing={sorted(missing)}, extras={sorted(extra)}"
                    )

        output: list[SVDQuantExportRecord] = []
        for (family, index), local in sorted(by_block.items()):
            block_prefix = f"{family}.{index}"
            rank = next(iter(local.values())).lora_down.shape[0]
            if family == "transformer_blocks":
                for target, names in _DOUBLE_FUSED.items():
                    present = tuple(local[name] for name in names if name in local)
                    if present and len(present) != len(names):
                        missing = [name for name in names if name not in local]
                        raise ValueError(f"{block_prefix}.{target} missing fused sources {missing}")
                    if present:
                        output.append(self._fuse(f"{block_prefix}.{target}", present, rank))
                for source_name, target in _DOUBLE_DIRECT.items():
                    if source_name in local:
                        output.append(self._direct(local[source_name], f"{block_prefix}.{target}"))
            else:
                names = _SINGLE_FUSED["qkv_proj"]
                present = tuple(local[name] for name in names if name in local)
                if present and len(present) != len(names):
                    missing = [name for name in names if name not in local]
                    raise ValueError(f"{block_prefix}.qkv_proj missing fused sources {missing}")
                if present:
                    output.append(self._fuse(f"{block_prefix}.qkv_proj", present, rank))
                if "proj_mlp" in local:
                    output.append(self._direct(local["proj_mlp"], f"{block_prefix}.mlp_fc1"))
                if "proj_out" in local:
                    source = local["proj_out"]
                    weight = _effective_weight(source, self.decomposition_device)
                    config = self._resolved_config(model)
                    heads, head_dim = config.get("num_attention_heads"), config.get("attention_head_dim")
                    inner_dim = (
                        heads * head_dim if isinstance(heads, int) and isinstance(head_dim, int) else weight.shape[0]
                    )
                    if inner_dim <= 0 or inner_dim >= weight.shape[1]:
                        raise ValueError(
                            f"{block_prefix}.proj_out cannot split input columns at inner_dim={inner_dim} "
                            f"for shape {tuple(weight.shape)}"
                        )
                    output.append(
                        _decompose(
                            weight[:, :inner_dim].clone(),
                            rank=rank,
                            template=source,
                            prefix=f"{block_prefix}.out_proj",
                            sources=(source,),
                            bias=None,
                        )
                    )
                    output.append(
                        _decompose(
                            weight[:, inner_dim:].clone(),
                            rank=rank,
                            template=source,
                            prefix=f"{block_prefix}.mlp_fc2",
                            sources=(source,),
                            bias=source.bias,
                        )
                    )
                    del weight
        return output

    def validate_records(
        self, sources: tuple[SourceLinearRecord, ...], records: tuple[SVDQuantExportRecord, ...]
    ) -> None:
        ranks = {source.lora_down.shape[0] for source in sources}
        if len(ranks) != 1:
            raise ValueError(f"FLUX source ranks must agree, got {sorted(ranks)}")
        if self.require_complete_model:
            # Config was already resolved during mapping for single blocks; retain model-independent
            # structural checks here and perform exact source coverage in map_modules via cached names.
            families: dict[str, set[int]] = {"transformer_blocks": set(), "single_transformer_blocks": set()}
            for source in sources:
                match = _BLOCK_RE.match(source.name)
                assert match is not None
                families[match.group(1)].add(int(match.group(2)))
            for family, indices in families.items():
                if indices and indices != set(range(max(indices) + 1)):
                    raise ValueError(f"complete FLUX export rejects gaps in {family}: {sorted(indices)}")
        prefixes = [record.prefix for record in records]
        if len(prefixes) != len(set(prefixes)):
            raise ValueError("FLUX adapter produced duplicate logical record prefixes")

    @staticmethod
    def _module_map(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
        return dict(model.named_modules())

    def extra_tensors(self, model: torch.nn.Module) -> Mapping[str, torch.Tensor]:
        modules = self._module_map(model)
        tensors: dict[str, torch.Tensor] = {}
        block_indices: dict[str, set[int]] = {"transformer_blocks": set(), "single_transformer_blocks": set()}
        for name in modules:
            match = _BLOCK_RE.match(name)
            if match:
                block_indices[match.group(1)].add(int(match.group(2)))

        if self.require_complete_model:
            config = self._resolved_config(model)
            for family, config_name in (
                ("transformer_blocks", "num_layers"),
                ("single_transformer_blocks", "num_single_layers"),
            ):
                expected = set(range(int(config[config_name])))
                if block_indices[family] != expected:
                    raise ValueError(
                        f"complete FLUX extras {family} indices mismatch: expected {sorted(expected)}, "
                        f"got {sorted(block_indices[family])}"
                    )

        for family, indices in block_indices.items():
            for index in sorted(indices):
                block = f"{family}.{index}"
                adanorms = (
                    (("norm1.linear", 6), ("norm1_context.linear", 6))
                    if family == "transformer_blocks"
                    else (("norm.linear", 3),)
                )
                for local_name, splits in adanorms:
                    module = modules.get(f"{block}.{local_name}")
                    if module is None:
                        if self.require_complete_model:
                            raise ValueError(f"complete FLUX export missing {block}.{local_name}")
                        continue
                    weight = getattr(module, "weight", None)
                    bias = getattr(module, "bias", None)
                    if not isinstance(weight, torch.Tensor):
                        raise ValueError(f"{block}.{local_name}.weight must be a tensor")
                    packed = quantize_adanorm_w4a16_rtn(
                        weight.detach().to(device="cpu", dtype=torch.bfloat16),
                        bias=None if bias is None else bias.detach().to(device="cpu", dtype=torch.bfloat16),
                        splits=splits,
                        group_size=64,
                    )
                    prefix = f"{block}.{local_name}"
                    tensors.update(
                        {
                            f"{prefix}.qweight": packed.qweight,
                            f"{prefix}.wscales": packed.wscales,
                            f"{prefix}.wzeros": packed.wzeros,
                            f"{prefix}.bias": packed.bias,
                        }
                    )
                for source_name, target_name in _RMS_MAP.items():
                    module = modules.get(f"{block}.{source_name}")
                    weight = None if module is None else getattr(module, "weight", None)
                    if weight is None:
                        required_rms = family == "transformer_blocks" or source_name in ("attn.norm_q", "attn.norm_k")
                        if self.require_complete_model and required_rms:
                            raise ValueError(f"complete FLUX export missing {block}.{source_name}.weight")
                        continue
                    tensors[f"{block}.{target_name}.weight"] = weight.detach().to(torch.bfloat16).cpu()

        for name, parameter in model.named_parameters():
            if name.startswith(_TOP_LEVEL_PREFIXES):
                tensors[name] = parameter.detach().to(torch.bfloat16).cpu()
        return {key: value.contiguous() for key, value in tensors.items()}

    def validate(self, tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, str]) -> None:
        if metadata.get("model_class") != "NunchakuFluxTransformer2dModel":
            raise ValueError("FLUX metadata has incorrect model_class")
        if metadata.get("format") != "pt" or metadata.get("comfy_config") != "{}":
            raise ValueError("FLUX metadata requires format='pt' and empty comfy_config")
        try:
            config = json.loads(metadata["config"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError("FLUX metadata config must be a JSON object") from exc
        if not isinstance(config, dict):
            raise ValueError("FLUX metadata config must be a JSON object")
        required: set[str] = set()
        if self.require_complete_model:
            try:
                num_layers = int(config["num_layers"])
                num_single_layers = int(config["num_single_layers"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("complete FLUX metadata requires layer counts") from exc
            linear_suffixes = ("qweight", "wscales", "smooth", "smooth_orig", "lora_down", "lora_up", "bias")
            for index in range(num_layers):
                block = f"transformer_blocks.{index}"
                for linear in (
                    "qkv_proj",
                    "qkv_proj_context",
                    "out_proj",
                    "out_proj_context",
                    "mlp_fc1",
                    "mlp_fc2",
                    "mlp_context_fc1",
                    "mlp_context_fc2",
                ):
                    required.update(f"{block}.{linear}.{suffix}" for suffix in linear_suffixes)
                for norm in ("norm1.linear", "norm1_context.linear"):
                    required.update(f"{block}.{norm}.{suffix}" for suffix in ("qweight", "wscales", "wzeros", "bias"))
                required.update(
                    f"{block}.{norm}.weight" for norm in ("norm_q", "norm_k", "norm_added_q", "norm_added_k")
                )
            for index in range(num_single_layers):
                block = f"single_transformer_blocks.{index}"
                for linear in ("qkv_proj", "out_proj", "mlp_fc1", "mlp_fc2"):
                    required.update(f"{block}.{linear}.{suffix}" for suffix in linear_suffixes)
                required.update(f"{block}.norm.linear.{suffix}" for suffix in ("qweight", "wscales", "wzeros", "bias"))
                required.update(f"{block}.{norm}.weight" for norm in ("norm_q", "norm_k"))
            missing = required - tensors.keys()
            if missing:
                raise ValueError(f"complete FLUX artifact is missing expected tensors: {sorted(missing)[:5]}")
            if num_layers == 19 and num_single_layers == 38 and len(tensors) != flux_onefile_tensor_count(19, 38):
                raise ValueError(f"standard FLUX one-file artifact must contain 2604 tensors, got {len(tensors)}")
        for key, tensor in tensors.items():
            if tensor.device.type != "cpu" or not tensor.is_contiguous():
                raise ValueError(f"FLUX tensor {key!r} must be contiguous on CPU")
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"FLUX tensor {key!r} must be finite")
            if not key.startswith(("transformer_blocks.", "single_transformer_blocks.")):
                if not key.startswith(_TOP_LEVEL_PREFIXES) or tensor.dtype != torch.bfloat16:
                    raise ValueError(f"FLUX passthrough tensor {key!r} must be in an allowed top-level BF16 family")
            is_adanorm = ".norm" in key and any(
                marker in key for marker in (".norm.linear.", ".norm1.linear.", ".norm1_context.linear.")
            )
            if key.endswith(".qweight"):
                expected_dtype = torch.int32 if is_adanorm else torch.int8
                if tensor.dtype != expected_dtype or tensor.ndim != 2:
                    raise ValueError(f"FLUX qweight {key!r} must be 2D {expected_dtype}")
            elif key.endswith(".wscales"):
                expected_dtype = torch.bfloat16 if is_adanorm else torch.uint8
                if tensor.dtype != expected_dtype or tensor.ndim != 2:
                    raise ValueError(f"FLUX wscales {key!r} must be 2D {expected_dtype}")
            elif key.endswith(".wzeros"):
                if not is_adanorm or tensor.dtype != torch.bfloat16 or tensor.ndim != 2:
                    raise ValueError(f"FLUX wzeros {key!r} must be a 2D AdaNorm BF16 tensor")
            elif key.endswith((".lora_down", ".lora_up")):
                if tensor.dtype != torch.bfloat16 or tensor.ndim != 2:
                    raise ValueError(f"FLUX low-rank tensor {key!r} must be 2D BF16")
            elif key.endswith((".smooth", ".smooth_orig", ".bias")):
                if tensor.dtype != torch.bfloat16 or tensor.ndim != 1:
                    raise ValueError(f"FLUX vector tensor {key!r} must be 1D BF16")
            elif key.endswith(".weight") and key.startswith(("transformer_blocks.", "single_transformer_blocks.")):
                if tensor.dtype != torch.bfloat16 or tensor.ndim != 1:
                    raise ValueError(f"FLUX RMSNorm tensor {key!r} must be 1D BF16")


__all__ = ["FluxSVDQuantNunchakuAdapter", "flux_onefile_tensor_count"]
