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
"""BlockForwardRunner — stateless block-forward execution engine.

This module owns:

* ``_DIFFUSION_OUTPUT_REGISTRY`` — a global map from block class name → output key order.
* :func:`register_diffusion_output` — public API to register new diffusion architectures.
* :class:`BlockForwardRunner` — the shared, stateless forward engine used by both the
  compressor and quantizers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import torch

from auto_round.compressors.utils import block_forward
from auto_round.utils.device_manager import device_manager

if TYPE_CHECKING:
    from auto_round.compressors.base import BaseOrchestrator


# ---------------------------------------------------------------------------
# Diffusion block output registry
# ---------------------------------------------------------------------------

#: Maps block class name → ordered list of output tensor keys.
#: Register new diffusion architectures with :func:`register_diffusion_output`.
_DIFFUSION_OUTPUT_REGISTRY: dict[str, list[str]] = {}


def register_diffusion_output(block_cls_name: str, output_keys: list[str]) -> None:
    """Register the output key order for a diffusion transformer block class.

    Args:
        block_cls_name: The ``__class__.__name__`` of the diffusion block
            (e.g. ``"FluxTransformerBlock"``).
        output_keys: Ordered list of tensor keys returned by the block's
            forward pass (e.g. ``["encoder_hidden_states", "hidden_states"]``).
            ``"hidden_states"`` must be present.

    Example::

        register_diffusion_output("MyDiTBlock", ["hidden_states"])
    """
    _DIFFUSION_OUTPUT_REGISTRY[block_cls_name] = output_keys


# Built-in diffusion block registrations.
# Add new architectures here instead of editing BlockRunner internals.
register_diffusion_output("FluxTransformerBlock", ["encoder_hidden_states", "hidden_states"])
register_diffusion_output("FluxSingleTransformerBlock", ["encoder_hidden_states", "hidden_states"])
register_diffusion_output("OvisImageTransformerBlock", ["encoder_hidden_states", "hidden_states"])
register_diffusion_output("OvisImageSingleTransformerBlock", ["encoder_hidden_states", "hidden_states"])
register_diffusion_output("StableAudioDiTBlock", ["hidden_states"])
register_diffusion_output("WanTransformerBlock", ["hidden_states"])


# ---------------------------------------------------------------------------
# BlockForwardRunner
# ---------------------------------------------------------------------------


# TODO wenhuach better follow heng's imp to decouple llm/diffusion
class BlockForwardRunner:
    """Stateless block-forward execution engine shared across quantizer & compressor.

    Created **once** by the compressor at init time and shared with quantizers
    via :class:`QuantizationRunContext`.

    Usage::

        # Orchestrator creates once:
        self.block_forward = BlockForwardRunner.from_orchestrator(self)

        # Quantizer (via _run_ctx):
        output = self._run_ctx.block_forward_runner(block, inputs, others, indices)

    To register a new diffusion block output layout::

        from auto_round.algorithms.block_runner import register_diffusion_output
        register_diffusion_output("MyDiTBlock", ["hidden_states"])
    """

    # Class-level reference to the module-level registry — read-only view for
    # tests and introspection (e.g. ``BlockForwardRunner.DIFFUSION_OUTPUT_CONFIGS``).
    DIFFUSION_OUTPUT_CONFIGS = _DIFFUSION_OUTPUT_REGISTRY

    def __init__(
        self,
        batch_dim: int = 0,
        batch_size: int = 8,
        device: Union[str, "torch.device"] = "cpu",
        cache_device: Union[str, "torch.device"] = "cpu",
        amp: bool = True,
        amp_dtype: torch.dtype | None = None,
        is_diffusion: bool = False,
        shared_cache_keys: tuple = (),
        output_config: list[str] | None = None,
        enable_torch_compile: bool = False,
    ) -> None:
        self.batch_dim = batch_dim
        self.batch_size = batch_size
        self.device = device
        self.cache_device = cache_device
        self.amp = amp
        self.amp_dtype = amp_dtype if amp_dtype is not None else torch.bfloat16
        self.is_diffusion = is_diffusion
        self.shared_cache_keys = shared_cache_keys
        self.output_config = output_config if output_config is not None else ["hidden_states"]
        self.enable_torch_compile = enable_torch_compile
        self.last_output_dict = None
        self.block_forward = block_forward
        if self.enable_torch_compile:
            from auto_round.utils import compile_func

            self.block_forward = compile_func(self.block_forward, device)

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_orchestrator(cls, orchestrator: "BaseOrchestrator", enable_torch_compile=False) -> "BlockForwardRunner":
        """Create from an orchestrator instance (called once at orchestrator init)."""
        model_ctx = getattr(orchestrator, "model_context", None)
        is_diffusion = getattr(model_ctx, "is_diffusion", False) if model_ctx else False
        output_config = getattr(model_ctx, "output_config", None) if model_ctx else None

        return cls(
            batch_dim=getattr(orchestrator, "batch_dim", 0),
            batch_size=getattr(orchestrator, "batch_size", 8),
            device=device_manager.device,
            cache_device=getattr(orchestrator, "cache_device", "cpu"),
            amp=getattr(orchestrator, "amp", True),
            amp_dtype=getattr(orchestrator, "amp_dtype", torch.bfloat16),
            is_diffusion=is_diffusion,
            shared_cache_keys=getattr(orchestrator, "shared_cache_keys", ()),
            output_config=output_config,
            enable_torch_compile=enable_torch_compile,
        )

    # ── Core forward ─────────────────────────────────────────────────────────

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.forward(*args, **kwargs)

    def forward(
        self,
        block: "torch.nn.Module",
        inputs: list[torch.Tensor] | dict,
        input_others: dict,
        indices: torch.Tensor | None = None,
        cache_device=None,
    ) -> list[torch.Tensor] | torch.Tensor:
        """Run block forward with batching, output normalization, and cache transfer.

        Args:
            block:        The transformer block.
            inputs:       Cached inputs (list[Tensor] for LLM/MLLM, dict for diffusion).
            input_others: Auxiliary kwargs (attention_mask, position_ids, etc.).
            indices:      Sample indices to forward. None = all samples.
            cache_device: Device for the returned tensor(s).  When ``None`` (default)
                          ``self.cache_device`` is used.  Pass an explicit device to
                          override for a single call without mutating shared state.

        Returns:
            if indices is not None, this func returns tensor, otherwise list
            Normalized output tensor on ``cache_device`` (or ``self.cache_device``).
        """
        out_device = cache_device if cache_device is not None else self.cache_device
        is_returned_list = True
        if indices is not None:
            is_returned_list = False
        num_samples = self._count_samples(inputs)
        if isinstance(inputs, list):
            device = inputs[0].device
        elif isinstance(inputs, dict):
            first_val = next(iter(inputs.values()))
            device = first_val[0].device if isinstance(first_val, list) else first_val.device
        else:
            device = inputs.device

        self.last_output_dict = None
        output_dict = {}
        if indices is None:
            indices = torch.arange(num_samples, dtype=torch.long, device=device)
        elif not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.long, device=device)
        else:
            indices = indices.to(device=device)

        outputs = []

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch_inputs, batch_others = self._select_batch(inputs, input_others, batch_indices)
            raw_output = self._forward_one_batch(block, batch_inputs, batch_others)
            batch_output_dict = self._get_diffusion_output_dict(raw_output, block)
            output = self._normalize_output(raw_output, block)
            if is_returned_list and self.batch_size != 1:  # split  it to 1
                if batch_output_dict:
                    for key, value in batch_output_dict.items():
                        output_dict.setdefault(key, []).extend(self.split_outputs(value))
                output = self.split_outputs(output)
            else:
                if batch_output_dict:
                    for key, value in batch_output_dict.items():
                        output_dict.setdefault(key, []).append(value)
                output = [output]
            outputs.extend(output)

        if not outputs:
            raise RuntimeError("BlockForwardRunner.forward: no outputs collected.")

        if is_returned_list:
            result = [o.to(out_device) for o in outputs]
            if output_dict:
                self.last_output_dict = {key: [o.to(out_device) for o in values] for key, values in output_dict.items()}
                self.last_output_dict["hidden_states"] = result
            return result
        else:
            if self.batch_size == 1:
                outputs = [output.unsqueeze(dim=self.batch_dim).to(out_device) for output in outputs]
                if output_dict:
                    output_dict = {
                        key: [value.unsqueeze(dim=self.batch_dim).to(out_device) for value in values]
                        for key, values in output_dict.items()
                    }

            outputs = torch.cat(outputs, dim=self.batch_dim).to(out_device)
            if output_dict:
                self.last_output_dict = {
                    key: torch.cat(values, dim=self.batch_dim).to(out_device) for key, values in output_dict.items()
                }
                self.last_output_dict["hidden_states"] = outputs

        return outputs.to(out_device)

    # ── Input selection ──────────────────────────────────────────────────────

    def select_batch(
        self,
        inputs: Any,
        input_others: dict,
        indices: torch.Tensor,
    ) -> tuple[Any, dict]:
        """Slice inputs and others by sample indices (public for custom loops)."""
        return self._select_batch(inputs, input_others, indices)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def split_outputs(self, output: torch.Tensor) -> list[torch.Tensor]:
        """Split a batched output back into per-sample tensors."""
        return list(torch.split(output, 1, dim=self.batch_dim))

    # ── Private ──────────────────────────────────────────────────────────────

    def _forward_one_batch(self, block, batch_inputs, batch_others) -> Any:
        """Forward one already-selected batch through the block (raw output)."""
        if isinstance(batch_inputs, dict):
            batch_inputs = dict(batch_inputs)
            batch_others = dict(batch_others)
            hidden_states = batch_inputs.pop("hidden_states")
            batch_others.update(batch_inputs)
            return self.block_forward(
                block,
                hidden_states,
                batch_others,
                self.amp,
                self.amp_dtype,
                self.device,
                None,
            )
        else:
            return self.block_forward(
                block,
                batch_inputs,
                batch_others,
                self.amp,
                self.amp_dtype,
                self.device,
                0,
            )

    def _count_samples(self, inputs: Any) -> int:
        if isinstance(inputs, dict):
            hs = inputs.get("hidden_states")
            return len(hs) if isinstance(hs, list) else hs.shape[self.batch_dim]
        elif isinstance(inputs, list):
            return len(inputs)
        else:
            return inputs.shape[self.batch_dim]

    def _normalize_output(self, output: Any, block: "torch.nn.Module" = None) -> torch.Tensor:
        """Normalize block output to a single tensor."""
        if isinstance(output, torch.Tensor):
            return output

        if not isinstance(output, (tuple, list)):
            raise TypeError(f"Block output must be tensor or tuple/list, got {type(output).__name__}.")

        if len(output) == 0:
            raise ValueError("Block output is an empty tuple/list.")

        if self.is_diffusion:
            # Look up per-block-type output config from the module-level registry;
            # fall back to instance-level output_config.
            block_cls_name = block.__class__.__name__ if block is not None else None
            oc = (
                _DIFFUSION_OUTPUT_REGISTRY.get(block_cls_name, self.output_config)
                if block_cls_name
                else self.output_config
            )
            idx = oc.index("hidden_states")
            if idx >= len(output):
                raise ValueError(f"Diffusion output has {len(output)} elements, but hidden_states index is {idx}.")
            hs = output[idx]
            if not isinstance(hs, torch.Tensor):
                raise TypeError(f"Expected hidden_states tensor, got {type(hs).__name__}.")
            return hs

        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
        raise TypeError(f"Block output[0] must be tensor, got {type(first).__name__}.")

    def _get_diffusion_output_dict(self, output: Any, block: "torch.nn.Module" = None) -> dict[str, torch.Tensor] | None:
        if not self.is_diffusion or isinstance(output, torch.Tensor) or not isinstance(output, (tuple, list)):
            return None
        block_cls_name = block.__class__.__name__ if block is not None else None
        output_config = (
            _DIFFUSION_OUTPUT_REGISTRY.get(block_cls_name, self.output_config) if block_cls_name else self.output_config
        )
        output_dict = {}
        for idx, key in enumerate(output_config):
            if idx >= len(output):
                break
            if isinstance(output[idx], torch.Tensor):
                output_dict[key] = output[idx]
        return output_dict or None

    def _select_batch(self, inputs, input_others, indices):
        """Select a subset of inputs by indices."""
        batch_dim = self.batch_dim
        shared_cache_keys = self.shared_cache_keys

        if isinstance(inputs, dict):
            selected_inputs = {}
            for key, val in inputs.items():
                if key in shared_cache_keys:
                    if isinstance(val, list) and len(val) == 1:
                        selected_inputs[key] = val[0]
                    elif isinstance(val, list) and len(val) > 1:
                        idx = int(indices[0]) if len(indices) == 1 else 0
                        selected_inputs[key] = val[idx] if idx < len(val) else val[0]
                    else:
                        selected_inputs[key] = val
                else:
                    if isinstance(val, list):
                        selected_inputs[key] = torch.cat([val[i] for i in indices], dim=batch_dim)
                    elif isinstance(val, torch.Tensor):
                        selected_inputs[key] = torch.index_select(val, batch_dim, indices)
                    else:
                        selected_inputs[key] = val
        else:
            if isinstance(inputs, list):
                selected_inputs = torch.cat([inputs[i] for i in indices], dim=batch_dim)
            else:
                selected_inputs = torch.index_select(inputs, batch_dim, indices)

        selected_others = {"positional_inputs": input_others.get("positional_inputs")}

        for key, val in input_others.items():
            if "positional_inputs" in key:
                continue
            if key in shared_cache_keys:
                if isinstance(val, list) and len(val) == 1:
                    selected_others[key] = val[0]
                elif isinstance(val, list) and len(val) > 1:
                    idx = int(indices[0]) if len(indices) == 1 else 0
                    selected_others[key] = val[idx] if idx < len(val) else val[0]
                else:
                    selected_others[key] = val
            elif isinstance(val, list):
                batch_vals = [val[i] for i in indices]
                if len(batch_vals) == 1:
                    selected_others[key] = batch_vals[0]
                else:
                    selected_others[key] = torch.cat(batch_vals, dim=batch_dim)
            elif isinstance(val, torch.Tensor):
                selected_others[key] = torch.index_select(val, batch_dim, indices)
            elif isinstance(val, (str, bool, type(None))):
                selected_others[key] = val
            else:
                selected_others[key] = val
        return selected_inputs, selected_others
