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

"""TEQ (Trainable Equivalent Transformation) pre-processor."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from auto_round.algorithms.registry import register_pipeline_member
from auto_round.algorithms.transforms.awq.base import (
    AWQTransform,
    _detect_parent_batch,
    _iter_block_names_for_mapping,
    _slice_batch_args_kwargs,
    _truncate_args_kwargs,
)
from auto_round.algorithms.transforms.awq.mappings import (
    ResolvedMapping,
    _extract_block_prefix,
    resolve_mappings,
)
from auto_round.algorithms.transforms.awq.qdq import QDQTool
from auto_round.algorithms.transforms.base import BasePreprocessor
from auto_round.algorithms.transforms.teq.config import TEQConfig
from auto_round.data_type.utils import compute_optimized_init_scale, get_quant_func
from auto_round.logger import logger
from auto_round.utils.model import set_module

if TYPE_CHECKING:
    from auto_round.algorithms.composer import AlgorithmComposer, BlockContext


class _TEQLinearFakeQuant(torch.nn.Module):
    """Differentiable TEQ wrapper used only while training the transform scale."""

    def __init__(
        self,
        orig_layer: torch.nn.Linear,
        log_alpha: torch.nn.Parameter,
        params: dict,
        quant_func,
        qdq_tool: QDQTool | None = None,
        act_quant_func=None,
        activation_second_moment: torch.Tensor | None = None,
        *,
        min_scale: float,
        max_scale: float,
    ) -> None:
        super().__init__()
        self.orig_layer = orig_layer
        self.log_alpha = log_alpha
        self.params = params
        self.quant_func = quant_func
        self.qdq_tool = qdq_tool
        self.act_quant_func = act_quant_func
        self.activation_second_moment = activation_second_moment
        self.min_scale = min_scale
        self.max_scale = max_scale

    @property
    def weight(self):
        return self.orig_layer.weight

    @property
    def bias(self):
        return self.orig_layer.bias

    def _scales(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.log_alpha.exp().clamp(min=self.min_scale, max=self.max_scale).to(device=x.device, dtype=x.dtype)
        return scale.view((1,) * (x.ndim - 1) + (-1,))

    def _qdq_weight(self, weight: torch.Tensor) -> torch.Tensor:
        params = self.params
        # Optimized scale search is a non-differentiable initializer. Recompute
        # it from the current transformed weight, then keep it fixed while STE
        # propagates gradients through the actual QDQ operation.
        quant_kwargs = {
            "bits": params["bits"],
            "group_size": params["group_size"],
            "data_type": params["data_type"],
            "sym": params["sym"],
        }
        if params.get("super_bits") is not None:
            quant_kwargs["super_bits"] = params["super_bits"]
        if params.get("super_group_size") is not None:
            quant_kwargs["super_group_size"] = params["super_group_size"]

        if self.qdq_tool is not None:
            quant_func, opt_quant_func = self.qdq_tool.resolve_quant_funcs(params)
            if opt_quant_func is not None:
                imatrix = self.activation_second_moment
                if imatrix is not None:
                    scales = self.log_alpha.detach().exp().clamp(self.min_scale, self.max_scale)
                    imatrix = imatrix.to(scales.device) / scales.square()
                init_scale = compute_optimized_init_scale(
                    weight.detach(),
                    params["data_type"],
                    params["bits"],
                    params["group_size"],
                    imatrix=imatrix,
                )
                if init_scale is not None:
                    quant_func = opt_quant_func
                    quant_kwargs["init_scale"] = init_scale
                    quant_kwargs["imatrix"] = imatrix if isinstance(imatrix, torch.Tensor) else torch.ones_like(weight)
            qdq_weight, _, _ = quant_func(weight, **quant_kwargs)
            return qdq_weight
        qdq_weight, _, _ = self.quant_func(weight, **quant_kwargs)
        return qdq_weight

    def _qdq_activation(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.act_quant_func is None:
            return tensor
        params = self.params
        quant_kwargs = {
            "bits": params["act_bits"],
            "group_size": params["act_group_size"],
            "data_type": params["act_data_type"],
            "sym": params["act_sym"],
        }
        qdq_tensor, _, _ = self.act_quant_func(tensor, **quant_kwargs)
        return qdq_tensor

    def forward(self, x=None, input=None):
        x = input if x is None else x
        alpha = self._scales(x)
        weight_alpha = alpha.reshape(1, -1).to(device=self.orig_layer.weight.device, dtype=self.orig_layer.weight.dtype)
        weight_qdq = self._qdq_weight(self.orig_layer.weight * weight_alpha)
        x_qdq = self._qdq_activation(x / alpha)
        bias = self.orig_layer.bias
        if bias is not None and bias.dtype != x.dtype:
            bias = bias.to(x.dtype)
        return F.linear(x_qdq, weight_qdq.to(x.dtype), bias)


@register_pipeline_member(TEQConfig)
class TEQTransform(BasePreprocessor):
    """Trainable equivalent transform pre-processor.

    This is the AutoRound pipeline version of TEQ: train alpha with fake
    quantized balance layers, restore the original modules, fold the trained
    scale into weights/norms, and let the downstream block quantizer handle the
    final compression.
    """

    def __init__(self, config: TEQConfig) -> None:
        super().__init__(config)
        self.iters = config.teq_iters
        self.lr = config.teq_lr
        self.min_scale = config.teq_min_scale
        self.max_scale = config.teq_max_scale
        self.sqrt_w_init = config.teq_sqrt_w_init
        self.awq_init = getattr(config, "teq_awq_init", False)
        self.awq_init_n_grid = getattr(config, "teq_awq_init_n_grid", 20)
        self.teq_nsamples = config.teq_nsamples
        self.teq_batch_size = config.teq_batch_size
        sample_seqlen = getattr(config, "teq_sample_seqlen", 512)
        self.teq_sample_seqlen = sample_seqlen if sample_seqlen and sample_seqlen > 0 else None
        self.teq_skip_moe = getattr(config, "teq_skip_moe", True)
        self.optimization_mode = getattr(config, "teq_optimization_mode", "staged")
        self.refine_iters = getattr(config, "teq_refine_iters", 10)
        self.joint_lr = getattr(config, "teq_joint_lr", 1e-4)
        self._user_mappings = config.mappings
        self.layer_config: dict | None = None

        self._resolved_mappings: list[ResolvedMapping] = []
        self._block_mappings: dict[str, list[ResolvedMapping]] = {}
        self._activation_stats: dict[str, list] = {}
        self._parent_args_cache: dict[str, list[tuple[tuple, dict]]] = {}
        self._finalized = False
        self._qdq_tool = QDQTool(
            bits=config.bits,
            group_size=config.group_size,
            sym=config.sym,
            data_type=config.data_type,
        )

    def bind(self, compressor) -> None:
        super().bind(compressor)
        nblocks = getattr(compressor, "nblocks", 1)
        if nblocks > 1:
            raise ValueError(f"TEQ does not support nblocks > 1, got nblocks={nblocks}.")

    def can_compile_block_forward(self) -> bool:
        # TEQ relies on transient Python forward hooks to collect per-block
        # replay samples. torch.compile can bypass those hooks after graph reuse.
        return False

    def prepare_run(self, composer: "AlgorithmComposer" = None) -> None:
        self._composer = composer
        self._qdq_tool.configure(composer)
        if self.optimization_mode != "staged":
            from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer

            quantizer = getattr(composer, "block_quantizer", None)
            if not isinstance(quantizer, SignRoundQuantizer):
                raise ValueError(
                    f"TEQ optimization_mode={self.optimization_mode!r} requires AutoRound/SignRound as the "
                    f"block quantizer, got {type(quantizer).__name__}."
                )
            quantizer.teq_refinement_mode = self.optimization_mode
            quantizer.teq_refine_iters = self.refine_iters
            quantizer.teq_joint_lr = self.joint_lr
        model = self.model
        self._resolved_mappings = resolve_mappings(model, self._user_mappings, skip_moe=self.teq_skip_moe)
        if not self._resolved_mappings:
            raise ValueError(
                "TEQ: no layer mappings were resolved for this model. "
                f"Model class: {type(model).__name__}. "
                "Provide explicit 'mappings' in TEQConfig to add support."
            )

        iter_block_names = _iter_block_names_for_mapping(model)
        self._block_mappings = {}
        for mapping in self._resolved_mappings:
            key = None
            for block_name in iter_block_names:
                if mapping.smooth_name == block_name or mapping.smooth_name.startswith(block_name + "."):
                    key = block_name
                    break
            if key is None:
                key = _extract_block_prefix(mapping.smooth_name)
            self._block_mappings.setdefault(key, []).append(mapping)

        logger.info(
            "TEQ: resolved %d mappings across %d blocks.",
            len(self._resolved_mappings),
            len(self._block_mappings),
        )
        self._finalized = False

    def register_fp_input_forward_hooks(self, block) -> list:
        if self.optimization_mode != "staged":
            # Experimental refinement starts from an exact identity transform
            # and consumes the block quantizer's normal calibration cache.
            return []
        block_name = getattr(block, "global_name", "")
        if block_name not in self._block_mappings:
            return []
        return self._register_parent_arg_hooks(block_name)

    def pre_quantize_block(self, ctx: "BlockContext") -> None:
        if len(ctx.block_names) != 1:
            raise ValueError(f"TEQ requires nblocks=1, got {len(ctx.block_names)} blocks: {ctx.block_names}.")
        block_mappings = self._block_mappings.get(ctx.block_names[0], [])
        self._qdq_tool.layer_config = self.layer_config
        active_mappings = [mapping for mapping in block_mappings if self._mapping_is_smoothable(mapping)]
        skipped = len(block_mappings) - len(active_mappings)
        if skipped:
            logger.warning_once(
                "TEQ: skipped %d mapping(s) in block '%s' that include ignore_layers / full-precision layers "
                "or incompatible per-layer quantization parameters.",
                skipped,
                ctx.block_names[0],
            )
        if self.optimization_mode == "staged":
            for mapping in active_mappings:
                self._train_and_apply_mapping(mapping)
        else:
            for mapping in active_mappings:
                self._prepare_ar_refinement_mapping(mapping)

        seen_parents: set[str] = set()
        for mapping in active_mappings:
            if mapping.parent_name not in seen_parents:
                seen_parents.add(mapping.parent_name)
                self._parent_args_cache.pop(mapping.parent_name, None)
            self._activation_stats.pop(mapping.smooth_name, None)

    def post_quantize_block(self, ctx: "BlockContext") -> None:
        block_mappings = self._block_mappings.get(ctx.block_name, [])
        if self.optimization_mode != "staged":
            for mapping in block_mappings:
                if self._mapping_is_smoothable(mapping):
                    self._finalize_ar_refinement_mapping(mapping)
        for mapping in block_mappings:
            self._parent_args_cache.pop(mapping.parent_name, None)

    def finalize_run(self) -> None:
        if self._finalized:
            return
        self._parent_args_cache.clear()
        self._activation_stats.clear()
        self._finalized = True

    def _register_parent_arg_hooks(self, block_name: str) -> list:
        handles = []
        mappings = self._block_mappings.get(block_name, [])
        module_lookup = dict(self.model.named_modules())

        def _resolve_activation_hook_layer(mapping: ResolvedMapping) -> torch.nn.Module | None:
            if not mapping.balance_layers:
                return None
            hook_target = mapping.activation_hook_target
            if not hook_target:
                return mapping.balance_layers[0]

            target_layer = module_lookup.get(hook_target)
            if target_layer is None and mapping.parent_name:
                target_layer = module_lookup.get(f"{mapping.parent_name}.{hook_target}")
            if target_layer is None:
                try:
                    target_layer = mapping.parent.get_submodule(hook_target)
                except AttributeError:
                    target_layer = None
            if target_layer is None:
                logger.warning(
                    "TEQ: activation_hook_target '%s' for '%s' was not found; using first balance layer '%s'.",
                    hook_target,
                    mapping.smooth_name,
                    mapping.balance_names[0] if mapping.balance_names else "<unknown>",
                )
                return mapping.balance_layers[0]
            return target_layer

        if self.awq_init or self._block_quantizer_is_signroundv2():
            for mapping in mappings:
                target_layer = _resolve_activation_hook_layer(mapping)
                if target_layer is None:
                    continue

                def _make_activation_hook(smooth_name: str):
                    def hook_fn(mod, args):
                        x = args[0] if isinstance(args, tuple) else args
                        if x is None or not isinstance(x, torch.Tensor) or x.numel() == 0:
                            return

                        feat = x.detach()
                        if feat.ndim == 1:
                            feat = feat.view(1, -1)
                        else:
                            feat = feat.flatten(0, -2)

                        feat_float = feat.float()
                        channel_sum = feat_float.abs().sum(dim=0).cpu()
                        channel_sq_sum = feat_float.square().sum(dim=0).cpu()
                        count = feat.shape[0]
                        if smooth_name not in self._activation_stats:
                            self._activation_stats[smooth_name] = [
                                torch.zeros_like(channel_sum),
                                torch.zeros_like(channel_sq_sum),
                                0,
                            ]
                        self._activation_stats[smooth_name][0] += channel_sum
                        self._activation_stats[smooth_name][1] += channel_sq_sum
                        self._activation_stats[smooth_name][2] += count

                    return hook_fn

                handles.append(target_layer.register_forward_pre_hook(_make_activation_hook(mapping.smooth_name)))

        parent_modules_hooked: set[str] = set()
        for mapping in mappings:
            parent_name = mapping.parent_name
            parent = self.model.get_submodule(parent_name) if parent_name else self.model
            if parent_name in parent_modules_hooked:
                continue
            parent_modules_hooked.add(parent_name)
            self._parent_args_cache.setdefault(parent_name, [])

            def _make_parent_hook(cache_key: str):
                def hook_fn(mod, args, kwargs):
                    param = next(mod.parameters(), None)
                    w_dtype = param.dtype if param is not None else None
                    proc_args = tuple(self._detach_forward_arg(value, w_dtype) for value in args)
                    proc_kwargs = {key: self._detach_forward_arg(value, w_dtype) for key, value in kwargs.items()}
                    if self.teq_sample_seqlen is not None:
                        proc_args, proc_kwargs = _truncate_args_kwargs(proc_args, proc_kwargs, self.teq_sample_seqlen)
                    self._parent_args_cache[cache_key].append((proc_args, proc_kwargs))

                return hook_fn

            handles.append(parent.register_forward_pre_hook(_make_parent_hook(parent_name), with_kwargs=True))
        return handles

    def _detach_forward_arg(self, value, w_dtype):
        if isinstance(value, torch.Tensor):
            value = value.detach()
            if w_dtype and value.is_floating_point() and value.dtype != w_dtype:
                value = value.to(w_dtype)
            return value.to("cpu", non_blocking=False)
        if isinstance(value, tuple):
            return tuple(self._detach_forward_arg(item, w_dtype) for item in value)
        if isinstance(value, list):
            return [self._detach_forward_arg(item, w_dtype) for item in value]
        if isinstance(value, dict):
            return {key: self._detach_forward_arg(item, w_dtype) for key, item in value.items()}
        if hasattr(value, "key_cache"):
            return None
        return value

    @staticmethod
    def _normalize_output(output):
        if isinstance(output, tuple):
            return output[0]
        return output

    @staticmethod
    def _move_parent_value_to_device(value: Any, device: torch.device | str) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(TEQTransform._move_parent_value_to_device(item, device) for item in value)
        if isinstance(value, list):
            return [TEQTransform._move_parent_value_to_device(item, device) for item in value]
        if isinstance(value, dict):
            return {key: TEQTransform._move_parent_value_to_device(item, device) for key, item in value.items()}
        return value

    def _iter_parent_calls(self, stored_args: tuple, stored_kwargs: dict):
        actual_batch = _detect_parent_batch(stored_args, stored_kwargs)
        if self.teq_batch_size is None or actual_batch is None or actual_batch <= self.teq_batch_size:
            yield stored_args, stored_kwargs
            return
        for start in range(0, actual_batch, self.teq_batch_size):
            end = min(start + self.teq_batch_size, actual_batch)
            yield _slice_batch_args_kwargs(stored_args, stored_kwargs, actual_batch, start, end)

    def _run_parent_samples(
        self,
        parent: torch.nn.Module,
        samples: list[tuple[tuple, dict]],
        *,
        offload_to_cpu: bool = False,
    ) -> list[torch.Tensor]:
        param = next(parent.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")
        outputs = []
        with torch.no_grad():
            for stored_args, stored_kwargs in samples:
                for micro_args, micro_kwargs in self._iter_parent_calls(stored_args, stored_kwargs):
                    call_args = tuple(self._move_parent_value_to_device(arg, device) for arg in micro_args)
                    call_kwargs = {
                        key: self._move_parent_value_to_device(value, device) for key, value in micro_kwargs.items()
                    }
                    out = self._normalize_output(parent(*call_args, **call_kwargs)).detach()
                    if offload_to_cpu:
                        out = out.to("cpu", non_blocking=False)
                    outputs.append(out)
        return outputs

    def _select_teq_samples(self, samples: list[tuple[tuple, dict]]) -> list[tuple[tuple, dict]]:
        max_samples = self.teq_nsamples
        if max_samples is None:
            return samples

        selected: list[tuple[tuple, dict]] = []
        remaining = max_samples
        for stored_args, stored_kwargs in samples:
            batch_size = _detect_parent_batch(stored_args, stored_kwargs)
            if batch_size is None:
                selected.append((stored_args, stored_kwargs))
                remaining -= 1
                if remaining <= 0:
                    break
                continue

            take = min(batch_size, remaining)
            selected.append(_slice_batch_args_kwargs(stored_args, stored_kwargs, batch_size, 0, take))
            remaining -= take
            if remaining <= 0:
                break
        return selected

    def _current_modules(
        self, mapping: ResolvedMapping
    ) -> tuple[torch.nn.Module, torch.nn.Module, list[torch.nn.Linear]]:
        parent = self.model.get_submodule(mapping.parent_name) if mapping.parent_name else self.model
        smooth_layer = self.model.get_submodule(mapping.smooth_name)
        balance_layers = [self.model.get_submodule(name) for name in mapping.balance_names]
        return parent, smooth_layer, balance_layers

    def _initial_alpha(self, balance_layers: list[torch.nn.Linear]) -> torch.Tensor:
        device = balance_layers[0].weight.device
        in_features = balance_layers[0].in_features
        if not self.sqrt_w_init:
            return torch.ones(in_features, device=device, dtype=torch.float32)

        weight = torch.cat([layer.weight.detach().float().to(device) for layer in balance_layers], dim=0)
        max_value = torch.sqrt(torch.max(torch.abs(weight), dim=0).values)
        max_value = torch.where(max_value == 0, torch.ones_like(max_value), max_value)
        return (1.0 / max_value).clamp(self.min_scale, self.max_scale)

    def _layer_config_for(self, layer: torch.nn.Module) -> dict:
        name = getattr(layer, "global_name", None) or ""
        return (self.layer_config or {}).get(name, {})

    def _resolve_params(self, layer: torch.nn.Module) -> dict:
        cfg = self._layer_config_for(layer)
        return {
            "bits": cfg.get("bits", self.config.bits),
            "group_size": cfg.get("group_size", self.config.group_size),
            "sym": cfg.get("sym", self.config.sym),
            "data_type": cfg.get("data_type", self.config.data_type),
            "act_bits": cfg.get("act_bits", self.config.act_bits),
            "act_group_size": cfg.get("act_group_size", self.config.act_group_size),
            "act_sym": cfg.get("act_sym", self.config.act_sym),
            "act_data_type": cfg.get("act_data_type", self.config.act_data_type),
            "act_dynamic": cfg.get("act_dynamic", self.config.act_dynamic),
            "disable_opt_rtn": cfg.get("disable_opt_rtn", False),
            "super_bits": cfg.get("super_bits", None),
            "super_group_size": cfg.get("super_group_size", None),
        }

    def _mapping_has_ignored_layer(self, mapping: ResolvedMapping) -> bool:
        """Return True if the smooth or balance layer is intentionally left full precision."""
        if not self.layer_config:
            return False

        def _is_fp(layer: torch.nn.Module) -> bool:
            cfg = self._layer_config_for(layer)
            bits = cfg.get("bits")
            return bits is not None and bits >= 16

        if _is_fp(mapping.smooth_layer):
            return True
        return any(_is_fp(layer) for layer in mapping.balance_layers)

    @staticmethod
    def _freeze_quant_param(value):
        if isinstance(value, list):
            return tuple(TEQTransform._freeze_quant_param(item) for item in value)
        if isinstance(value, tuple):
            return tuple(TEQTransform._freeze_quant_param(item) for item in value)
        return value

    def _balance_quant_signature(self, layer: torch.nn.Module) -> tuple:
        params = self._resolve_params(layer)
        keys = (
            "bits",
            "group_size",
            "sym",
            "data_type",
            "act_bits",
            "act_group_size",
            "act_sym",
            "act_data_type",
            "act_dynamic",
            "super_bits",
            "super_group_size",
        )
        return tuple((key, self._freeze_quant_param(params.get(key))) for key in keys)

    def _mapping_has_mixed_quant_params(self, mapping: ResolvedMapping) -> bool:
        """Return True when balance layers sharing one TEQ alpha have incompatible quant params."""
        if len(mapping.balance_layers) <= 1:
            return False

        signatures = [self._balance_quant_signature(layer) for layer in mapping.balance_layers]
        first = signatures[0]
        if all(signature == first for signature in signatures[1:]):
            return False

        details = {name: dict(signature) for name, signature in zip(mapping.balance_names, signatures)}
        logger.warning(
            "TEQ: skipping transform for '%s' because balance layers in the same mapping "
            "have different quantization parameters: %s.",
            mapping.smooth_name,
            details,
        )
        return True

    def _mapping_is_smoothable(self, mapping: ResolvedMapping) -> bool:
        """TEQ transform is all-or-nothing for layers sharing one equivalent scale."""
        if self._mapping_has_ignored_layer(mapping):
            return False
        if self._mapping_has_mixed_quant_params(mapping):
            return False
        return True

    def _make_wrapper(self, layer: torch.nn.Linear, log_alpha: torch.nn.Parameter) -> _TEQLinearFakeQuant:
        params = self._resolve_params(layer)
        quant_func, _ = get_quant_func(
            params["data_type"],
            params["bits"],
            params["sym"],
            disable_opt_rtn=True,
            group_size=params["group_size"],
            iters=max(1, self.iters),
        )
        act_quant_func = None
        activation_second_moment = None
        if self._block_quantizer_is_signroundv2():
            for mapping in self._resolved_mappings:
                if layer not in mapping.balance_layers:
                    continue
                stats = self._activation_stats.get(mapping.smooth_name)
                if stats is not None and stats[2] > 0:
                    activation_second_moment = stats[1] / stats[2]
                break
        act_bits = params.get("act_bits") or 16
        act_dynamic = True if params.get("act_dynamic") is None else params.get("act_dynamic")
        if act_bits <= 8 and not act_dynamic:
            raise NotImplementedError("TEQ does not support static activation quantization during scale training.")
        if act_bits <= 8 and params.get("act_data_type") is not None and act_dynamic:
            if params.get("act_group_size") is None:
                weight_group_size = params.get("group_size")
                params["act_group_size"] = weight_group_size if isinstance(weight_group_size, int) else -1
            act_quant_func, act_data_type = get_quant_func(
                params["act_data_type"],
                act_bits,
                params.get("act_sym", True),
                disable_opt_rtn=True,
                group_size=params.get("act_group_size"),
                iters=max(1, self.iters),
            )
            params["act_data_type"] = act_data_type
        return _TEQLinearFakeQuant(
            layer,
            log_alpha,
            params,
            quant_func,
            self._qdq_tool,
            act_quant_func=act_quant_func,
            activation_second_moment=activation_second_moment,
            min_scale=self.min_scale,
            max_scale=self.max_scale,
        )

    def _block_quantizer_is_signroundv2(self) -> bool:
        from auto_round.algorithms.quantization.sign_round.config import SignRoundV2Config

        composer = getattr(self, "_composer", None)
        quantizer = getattr(composer, "block_quantizer", None)
        return isinstance(getattr(quantizer, "config", None), SignRoundV2Config)

    def _scale_summary(self, scales: torch.Tensor) -> dict[str, float]:
        scales = scales.detach().float()
        if scales.numel() == 0:
            return {
                "alpha_min": float("nan"),
                "alpha_max": float("nan"),
                "alpha_mean": float("nan"),
                "alpha_std": float("nan"),
                "clamp_min_pct": float("nan"),
                "clamp_max_pct": float("nan"),
            }

        eps = 1e-6
        return {
            "alpha_min": scales.min().item(),
            "alpha_max": scales.max().item(),
            "alpha_mean": scales.mean().item(),
            "alpha_std": scales.std(unbiased=False).item(),
            "clamp_min_pct": (scales <= self.min_scale * (1.0 + eps)).float().mean().item() * 100.0,
            "clamp_max_pct": (scales >= self.max_scale * (1.0 - eps)).float().mean().item() * 100.0,
        }

    def _log_scale_summary(self, phase: str, mapping: ResolvedMapping, best_error: float, scales: torch.Tensor) -> None:
        stats = self._scale_summary(scales)
        logger.info(
            "TEQ %s '%s': best_error=%.3e, alpha_min=%.6g, alpha_max=%.6g, "
            "alpha_mean=%.6g, alpha_std=%.6g, clamp_min_pct=%.2f, clamp_max_pct=%.2f",
            phase,
            mapping.smooth_name,
            best_error,
            stats["alpha_min"],
            stats["alpha_max"],
            stats["alpha_mean"],
            stats["alpha_std"],
            stats["clamp_min_pct"],
            stats["clamp_max_pct"],
        )

    @torch.no_grad()
    def _compute_parent_loss(
        self,
        parent: torch.nn.Module,
        samples: list[tuple[tuple, dict]],
        fp_outputs: list[torch.Tensor],
    ) -> float:
        param = next(parent.parameters(), None)
        device = param.device if param is not None else torch.device("cpu")

        loss = torch.tensor(0.0, device=device)
        num_elements = torch.tensor(0, device=device, dtype=torch.long)
        output_idx = 0
        for stored_args, stored_kwargs in samples:
            for micro_args, micro_kwargs in self._iter_parent_calls(stored_args, stored_kwargs):
                if output_idx >= len(fp_outputs):
                    return float("inf")
                call_args = tuple(self._move_parent_value_to_device(arg, device) for arg in micro_args)
                call_kwargs = {
                    key: self._move_parent_value_to_device(value, device) for key, value in micro_kwargs.items()
                }
                out = self._normalize_output(parent(*call_args, **call_kwargs))
                ref = fp_outputs[output_idx].to(device, non_blocking=False)
                loss += F.mse_loss(out.float(), ref.float(), reduction="sum")
                num_elements += ref.numel()
                output_idx += 1
                del out, ref

        if output_idx != len(fp_outputs) or num_elements == 0:
            return float("inf")
        return (loss / num_elements).item()

    @torch.no_grad()
    def _awq_initial_alpha(
        self,
        mapping: ResolvedMapping,
        samples: list[tuple[tuple, dict]],
        fp_outputs: list[torch.Tensor],
    ) -> torch.Tensor | None:
        stats = self._activation_stats.get(mapping.smooth_name)
        if stats is None:
            logger.warning("TEQ: no AWQ-init activation stats for '%s'; using default alpha.", mapping.smooth_name)
            return None
        act_sum, _, act_count = stats
        if act_count == 0:
            logger.warning("TEQ: zero AWQ-init activation count for '%s'; using default alpha.", mapping.smooth_name)
            return None

        device = mapping.balance_layers[0].weight.device
        x_mean = (act_sum / act_count).to(device=device, dtype=torch.float32)
        params = self._resolve_params(mapping.balance_layers[0])
        group_size = AWQTransform._normalize_group_size(params["group_size"], -1)
        w_mean = AWQTransform._compute_layer_means(mapping.balance_layers, group_size).to(device)

        orig_layers = dict(zip(mapping.balance_names, mapping.balance_layers))
        parent = self.model.get_submodule(mapping.parent_name) if mapping.parent_name else self.model
        best_error = float("inf")
        best_scales = None
        n_grid = max(2, int(self.awq_init_n_grid))

        try:
            for idx in range(n_grid):
                ratio = idx / (n_grid - 1)
                scales = (x_mean.pow(ratio) / (w_mean.pow(1 - ratio) + 1e-4)).clamp(min=1e-4)
                scales = scales / (scales.max() * scales.min()).sqrt()
                scales[torch.isinf(scales)] = 1
                scales[torch.isnan(scales)] = 1
                scales = scales.clamp(self.min_scale, self.max_scale)
                log_alpha = torch.nn.Parameter(scales.detach().clone().log())
                for layer_name, layer in orig_layers.items():
                    set_module(self.model, layer_name, self._make_wrapper(layer, log_alpha))
                train_parent = self.model.get_submodule(mapping.parent_name) if mapping.parent_name else self.model
                total_loss = self._compute_parent_loss(train_parent, samples, fp_outputs)
                for layer_name, layer in orig_layers.items():
                    set_module(self.model, layer_name, layer)
                if total_loss < best_error:
                    best_error = total_loss
                    best_scales = scales.detach().clone()
        finally:
            for layer_name, layer in orig_layers.items():
                set_module(self.model, layer_name, layer)

        if best_scales is None:
            logger.warning("TEQ: AWQ-init grid search failed for '%s'; using default alpha.", mapping.smooth_name)
            return None
        self._log_scale_summary("awq_init", mapping, best_error, best_scales)
        return best_scales

    def _train_and_apply_mapping(self, mapping: ResolvedMapping) -> None:
        parent_kwargs_list = self._select_teq_samples(self._parent_args_cache.get(mapping.parent_name, []))
        if not parent_kwargs_list:
            logger.warning("TEQ: no calibration samples for '%s'; skipping.", mapping.smooth_name)
            return

        parent, _, balance_layers = self._current_modules(mapping)
        fp_outputs = self._run_parent_samples(
            parent, parent_kwargs_list, offload_to_cpu=self.teq_batch_size is not None
        )
        initial_alpha = self._initial_alpha(balance_layers)
        if self.awq_init:
            awq_alpha = self._awq_initial_alpha(mapping, parent_kwargs_list, fp_outputs)
            if awq_alpha is not None:
                initial_alpha = awq_alpha
        initial_alpha = initial_alpha.clamp(self.min_scale, self.max_scale)
        log_alpha = torch.nn.Parameter(initial_alpha.log())
        optimizer = torch.optim.Adam([log_alpha], lr=self.lr, betas=(0.9, 0.9), weight_decay=0.0)
        orig_layers = dict(zip(mapping.balance_names, balance_layers))
        best_loss = float("inf")
        best_alpha = initial_alpha.detach().clone()

        try:
            for layer_name, layer in orig_layers.items():
                set_module(self.model, layer_name, self._make_wrapper(layer, log_alpha))
            train_parent = self.model.get_submodule(mapping.parent_name) if mapping.parent_name else self.model

            # Evaluate iteration zero and every post-step candidate. This keeps
            # the scale that actually produced the measured loss.
            for step in range(self.iters + 1):
                optimizer.zero_grad(set_to_none=True)
                has_grad_loss = False
                finite_loss = True
                output_idx = 0
                param = next(train_parent.parameters(), None)
                device = param.device if param is not None else torch.device("cpu")
                loss_sum = torch.zeros((), device=device, dtype=torch.float32)
                num_elements = 0
                for stored_args, stored_kwargs in parent_kwargs_list:
                    for micro_args, micro_kwargs in self._iter_parent_calls(stored_args, stored_kwargs):
                        if output_idx >= len(fp_outputs):
                            finite_loss = False
                            break
                        call_args = tuple(self._move_parent_value_to_device(arg, device) for arg in micro_args)
                        call_kwargs = {
                            key: self._move_parent_value_to_device(value, device) for key, value in micro_kwargs.items()
                        }
                        ref_output = fp_outputs[output_idx]
                        output_idx += 1
                        with torch.enable_grad():
                            output = self._normalize_output(train_parent(*call_args, **call_kwargs))
                            ref = ref_output.to(output.device).float()
                            loss = F.mse_loss(output.float(), ref, reduction="sum")
                        if loss.requires_grad and step < self.iters:
                            has_grad_loss = True
                        if not torch.isfinite(loss):
                            logger.warning("TEQ: non-finite loss for '%s'; keeping best scale.", mapping.smooth_name)
                            finite_loss = False
                            break
                        loss_sum = loss_sum + loss
                        num_elements += ref.numel()
                        del output, loss, ref, ref_output
                    if not finite_loss:
                        break
                if output_idx != len(fp_outputs):
                    finite_loss = False
                if not finite_loss:
                    break
                total_loss = loss_sum / max(1, num_elements)
                total_loss_value = total_loss.detach().item()
                if total_loss_value < best_loss:
                    best_loss = total_loss_value
                    best_alpha = log_alpha.detach().exp().clamp(self.min_scale, self.max_scale).clone()
                if step == self.iters:
                    break
                if not has_grad_loss or not total_loss.requires_grad:
                    logger.warning(
                        "TEQ: loss for '%s' is not differentiable; keeping the current scale.",
                        mapping.smooth_name,
                    )
                    break
                total_loss.backward()
                optimizer.step()
                with torch.no_grad():
                    log_alpha.clamp_(math.log(self.min_scale), math.log(self.max_scale))
        finally:
            for layer_name, layer in orig_layers.items():
                set_module(self.model, layer_name, layer)

        self._log_scale_summary("train", mapping, best_loss, best_alpha)
        self._apply_scales(mapping, best_alpha)

    @staticmethod
    def _core_layer(layer: torch.nn.Module) -> torch.nn.Module:
        """Return the underlying layer after AutoRound activation wrapping."""
        while hasattr(layer, "orig_layer"):
            layer = layer.orig_layer
        return layer

    def _prepare_ar_refinement_mapping(self, mapping: ResolvedMapping) -> None:
        """Attach an identity TEQ transform for guarded AutoRound refinement.

        Linear smooth layers need an output transform while their consumers
        need the matching input transform. Keeping both in the quantizer's QDQ
        path ensures the final stored weights and scales describe the selected
        transform, including overlapping ``norm -> up -> down`` mappings.
        """
        params = self._resolve_params(mapping.balance_layers[0])
        act_bits = params.get("act_bits") or 16
        act_dynamic = True if params.get("act_dynamic") is None else params.get("act_dynamic")
        if act_bits <= 8 and not act_dynamic:
            raise NotImplementedError(
                "Experimental TEQ refinement does not support static activation quantization because "
                "the activation range changes with the trainable scale."
            )

        device = mapping.balance_layers[0].weight.device
        log_alpha = torch.nn.Parameter(torch.zeros(mapping.balance_layers[0].in_features, device=device))
        log_alpha._teq_min_scale = self.min_scale
        log_alpha._teq_max_scale = self.max_scale
        context = {
            "log_alpha": log_alpha,
            "min_log": math.log(self.min_scale),
            "max_log": math.log(self.max_scale),
            "smooth_name": mapping.smooth_name,
            "apply_input_transform": not isinstance(self._core_layer(mapping.smooth_layer), torch.nn.Linear),
            "wrappers": [],
        }
        for layer in mapping.balance_layers:
            core = self._core_layer(layer)
            if hasattr(core, "_teq_input_context"):
                raise ValueError(f"TEQ: layer {getattr(core, 'global_name', '<unknown>')!r} has multiple input scales.")
            core._teq_input_context = context

        smooth = self._core_layer(mapping.smooth_layer)
        if isinstance(smooth, torch.nn.Linear):
            output_contexts = list(getattr(smooth, "_teq_output_contexts", []))
            output_contexts.append(context)
            smooth._teq_output_contexts = output_contexts

    @torch.no_grad()
    def _finalize_ar_refinement_mapping(self, mapping: ResolvedMapping) -> None:
        """Commit the best guarded refinement scale and remove transient state."""
        balance_layers = [self._core_layer(self.model.get_submodule(name)) for name in mapping.balance_names]
        context = getattr(balance_layers[0], "_teq_input_context", None)
        if context is None:
            return
        scales = context["log_alpha"].detach().exp().clamp(self.min_scale, self.max_scale)
        smooth = self._core_layer(self.model.get_submodule(mapping.smooth_name))

        # Linear weights were transformed before QDQ by WrapperLinear. Only
        # their unquantized bias remains to be folded here. Norms are not block
        # quantized, so fold the complete output transform into them now.
        if isinstance(smooth, torch.nn.Linear):
            if smooth.bias is not None:
                smooth.bias.data.div_(scales.to(smooth.bias.device, smooth.bias.dtype))
        else:
            AWQTransform._fold_scales_into_smooth_layer(smooth, scales)
        for wrapper in context["wrappers"]:
            if hasattr(wrapper, "commit_teq_scale"):
                wrapper.commit_teq_scale(context)

        self._log_scale_summary("ar_refine", mapping, float("nan"), scales)
        for layer in balance_layers:
            if hasattr(layer, "_teq_input_context"):
                delattr(layer, "_teq_input_context")
        if hasattr(smooth, "_teq_output_contexts"):
            remaining = [item for item in smooth._teq_output_contexts if item is not context]
            if remaining:
                smooth._teq_output_contexts = remaining
            else:
                delattr(smooth, "_teq_output_contexts")

    @torch.no_grad()
    def _apply_scales(self, mapping: ResolvedMapping, scales: torch.Tensor) -> None:
        _, smooth_layer, balance_layers = self._current_modules(mapping)
        scales = scales.clamp(self.min_scale, self.max_scale)
        for layer in balance_layers:
            layer.weight.data.mul_(scales.to(layer.weight.device, layer.weight.dtype).view(1, -1))
        AWQTransform._fold_scales_into_smooth_layer(smooth_layer, scales)
