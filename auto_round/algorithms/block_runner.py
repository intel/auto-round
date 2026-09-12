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

import logging
from typing import TYPE_CHECKING, Any, Union

import torch

from auto_round.compressors.utils import block_forward
from auto_round.logger import logger
from auto_round.utils.device_manager import device_manager

if TYPE_CHECKING:
    from auto_round.compressors.base import BaseOrchestrator


# ---------------------------------------------------------------------------
# Diffusion block output registry
# ---------------------------------------------------------------------------

#: Maps block class name → ordered list of output tensor keys.
#: Register new diffusion architectures with :func:`register_diffusion_output`.
_DIFFUSION_OUTPUT_REGISTRY: dict[str, list[str]] = {}

#: Maps decoder block class name to output keys that become the next block's inputs.
_BLOCK_OUTPUT_REGISTRY: dict[str, list[str]] = {}


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


def register_block_output(block_cls_name: str, output_keys: list[str]) -> None:
    """Register tuple output keys that should be forwarded to the next block."""
    _BLOCK_OUTPUT_REGISTRY[block_cls_name] = output_keys


# Built-in diffusion block registrations.
# Add new architectures here instead of editing BlockRunner internals.
register_diffusion_output("FluxTransformerBlock", ["encoder_hidden_states", "hidden_states"])
register_diffusion_output("FluxSingleTransformerBlock", ["encoder_hidden_states", "hidden_states"])
register_diffusion_output("OvisImageTransformerBlock", ["encoder_hidden_states", "hidden_states"])
register_diffusion_output("OvisImageSingleTransformerBlock", ["encoder_hidden_states", "hidden_states"])
register_diffusion_output("StableAudioDiTBlock", ["hidden_states"])
register_diffusion_output("WanTransformerBlock", ["hidden_states"])

# GLM DSA full indexer layers return top-k indices for subsequent shared layers.
register_block_output("GlmMoeDsaDecoderLayer", ["hidden_states", "prev_topk_indices"])


# ---------------------------------------------------------------------------
# BlockForwardRunner
# ---------------------------------------------------------------------------


# TODO wenhuach better follow heng's imp to decouple llm/diffusion
def _cat_device_safe(tensors: list, dim: int) -> "torch.Tensor":
    """Concatenate per-sample tensors that may live on different devices.

    Distributed calibration pools keep each sample on its owning DDP device;
    batch selection that crosses owners moves the minority to the first
    piece's device (same-device selections -- the common DDP case -- stay
    copy-free).
    """
    if not tensors:
        raise ValueError("_cat_device_safe: empty selection")
    dev = tensors[0].device
    if any(t.device != dev for t in tensors):
        logging.getLogger(__name__).debug(
            "_cat_device_safe: moved %d/%d pieces to %s",
            sum(1 for t in tensors if t.device != dev),
            len(tensors),
            dev,
        )
        tensors = [t.to(dev) for t in tensors]
    return torch.cat(tensors, dim=dim)


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
        enable_torch_compile: bool = True,
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
    def from_orchestrator(cls, orchestrator: "BaseOrchestrator", enable_torch_compile=True) -> "BlockForwardRunner":
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
            batch_output_dict = self._get_output_dict(raw_output, block)
            output = self._normalize_output(raw_output, block)
            if is_returned_list and self.batch_size != 1:  # split  it to 1
                if batch_output_dict:
                    for key, value in batch_output_dict.items():
                        output_dict.setdefault(key, []).extend(
                            item.to(out_device) for item in self.split_outputs(value)
                        )
                output = self.split_outputs(output)
            else:
                if batch_output_dict:
                    for key, value in batch_output_dict.items():
                        output_dict.setdefault(key, []).append(value.to(out_device))
                output = [output]
            outputs.extend(item.to(out_device) for item in output)
            del raw_output, batch_output_dict, output, batch_inputs, batch_others

        if not outputs:
            raise RuntimeError("BlockForwardRunner.forward: no outputs collected.")

        if is_returned_list:
            result = outputs
            if output_dict:
                self.last_output_dict = output_dict
                self.last_output_dict["hidden_states"] = result
            return result
        else:
            outputs = torch.cat(outputs, dim=self.batch_dim)
            if output_dict:
                self.last_output_dict = {
                    key: torch.cat(values, dim=self.batch_dim) for key, values in output_dict.items()
                }
                self.last_output_dict["hidden_states"] = outputs

        return outputs

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

    def _forward_device(self, block) -> torch.device:
        """Staging device for a forward pass through ``block``.

        Normally the runner's global device. The one exception is a block
        that sits WHOLE on a single accelerator device other than the runner
        device -- a DDP mirror replica: staging its inputs to the global
        (primary) device would put the hidden states on a different GPU than
        the replica's weights. Blocks that span several accelerator devices
        (upstream sharded placement, routed by their align hooks) and
        mixed CPU/GPU blocks (CPU-pinned tables) keep the global device, so
        serial behavior is unchanged.
        """
        try:
            devs = {pp.device for pp in block.parameters()}
        except Exception as e:  # pragma: no cover - exotic modules
            logger.warning("block device sniff failed (%s); assuming the runner device", e)
            return self.device
        acc = {d for d in devs if d.type in ("cuda", "xpu", "hpu")}
        if len(acc) == 1 and devs <= (acc | {torch.device("cpu")}):
            (d,) = acc
            if d != self.device:
                return d
        return self.device

    def _forward_one_batch(self, block, batch_inputs, batch_others) -> Any:
        """Forward one already-selected batch through the block (raw output)."""
        # Stage on the block's actual device (see _forward_device). The shared
        # kwargs follow the hidden states so masks/position ids reach the same
        # GPU as the weights; for the serial path both are already there and
        # the moves are no-ops.
        from auto_round.utils.model import to_device

        fwd_device = self._forward_device(block)
        if isinstance(batch_inputs, dict):
            batch_inputs = dict(batch_inputs)
            batch_others = dict(batch_others)
            hidden_states = batch_inputs.pop("hidden_states")
            batch_others.update(batch_inputs)
        else:
            hidden_states = batch_inputs
        if torch.is_tensor(hidden_states) and hidden_states.device != fwd_device:
            hidden_states = hidden_states.to(fwd_device)
        batch_others = to_device(batch_others, fwd_device)
        return self.block_forward(
            block,
            hidden_states,
            batch_others,
            self.amp,
            self.amp_dtype,
            fwd_device,
            None,
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

    def _get_output_dict(self, output: Any, block: "torch.nn.Module" = None) -> dict[str, torch.Tensor] | None:
        if isinstance(output, torch.Tensor) or not isinstance(output, (tuple, list)):
            return None
        block_cls_name = block.__class__.__name__ if block is not None else None
        if self.is_diffusion:
            output_config = (
                _DIFFUSION_OUTPUT_REGISTRY.get(block_cls_name, self.output_config)
                if block_cls_name
                else self.output_config
            )
        else:
            output_config = _BLOCK_OUTPUT_REGISTRY.get(block_cls_name)
            if output_config is None:
                return None
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
                        selected_inputs[key] = _cat_device_safe([val[i] for i in indices], dim=batch_dim)
                    elif isinstance(val, torch.Tensor):
                        selected_inputs[key] = torch.index_select(val, batch_dim, indices)
                    else:
                        selected_inputs[key] = val
        else:
            if isinstance(inputs, list):
                selected_inputs = _cat_device_safe([inputs[i] for i in indices], dim=batch_dim)
            else:
                selected_inputs = torch.index_select(inputs, batch_dim, indices)

        selected_others = {"positional_inputs": input_others.get("positional_inputs")}

        def _slice_shared_entry(entry, rows):
            """Within-batch row slice of one shared-cache entry (batch pick already done)."""
            if isinstance(entry, torch.Tensor):
                try:
                    return entry.index_select(batch_dim, rows.to(device=entry.device))
                except (RuntimeError, IndexError):
                    logger.warning_once(
                        "shared-cache kwarg slicing fell back to the unsliced entry; "
                        "sub-batch draws may see the wrong batch's entry"
                    )
                    return entry
            if isinstance(entry, tuple) and entry and all(torch.is_tensor(t) for t in entry):
                parts = []
                for t in entry:
                    try:
                        parts.append(t.index_select(batch_dim, rows.to(device=t.device)))
                    except (RuntimeError, IndexError):
                        logger.warning_once(
                            "shared-cache kwarg element slicing fell back to the unsliced tensor; "
                            "sub-batch draws may see the wrong batch's entry"
                        )
                        parts.append(t)
                return tuple(parts)
            return entry

        for key, val in input_others.items():
            if "positional_inputs" in key:
                continue
            if key in shared_cache_keys:
                if isinstance(val, list) and len(val) == 1:
                    selected_others[key] = val[0]
                elif (
                    isinstance(val, list)
                    and len(val) > 1
                    and 1 <= len(indices) < self.batch_size
                    and inputs is not None
                ):
                    # sub-batch draw (DDP shard). Two cached-list layouts exist:
                    # per-SAMPLE (len == pool size; legacy single-index pick indexes
                    # it directly) and per-BATCH (len == pool / batch_size; entries
                    # are whole-batch tensors/tuples). Disambiguate by length.
                    _n = self._count_samples(inputs)
                    if len(val) == _n and _n != self.batch_size:
                        # per-sample list: pick the drawn samples' entries
                        _picks = [val[int(i)] if int(i) < len(val) else val[0] for i in indices]
                        if len(_picks) == 1:
                            selected_others[key] = _picks[0]
                        elif all(isinstance(pp, tuple) and pp and all(torch.is_tensor(t) for t in pp) for pp in _picks):
                            selected_others[key] = tuple(
                                _cat_device_safe([pp[slot] for pp in _picks], dim=batch_dim)
                                for slot in range(len(_picks[0]))
                            )
                        elif all(torch.is_tensor(pp) for pp in _picks):
                            selected_others[key] = _cat_device_safe(_picks, dim=batch_dim)
                        else:
                            selected_others[key] = val[0]
                    else:
                        # per-batch list: pick the owning batch's entry, slice rows
                        _b = int(indices[0]) // self.batch_size
                        _entry = val[_b] if 0 <= _b < len(val) else val[0]
                        _rows = torch.as_tensor([int(i) % self.batch_size for i in indices], dtype=torch.long)
                        selected_others[key] = _slice_shared_entry(_entry, _rows)
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
                    selected_others[key] = _cat_device_safe(batch_vals, dim=batch_dim)
            elif isinstance(val, torch.Tensor):
                # ``batch_indices`` are created on CPU by the sampler.  XPU
                # (and other accelerator backends) require index tensors on
                # the same device as the indexed value.
                selected_others[key] = torch.index_select(val, batch_dim, indices.to(device=val.device))
            elif isinstance(val, tuple) and val and all(torch.is_tensor(t) for t in val):
                # tuple-of-tensors kwargs (transformers v5 passes rope as
                # ``position_embeddings=(cos, sin)`` with a per-sample batch
                # dim). Slice each element like the tensor branch; elements
                # that are NOT per-sample (e.g. broadcast [1, S, D] tables)
                # fail the select and pass through unsliced, preserving their
                # broadcast behavior. Without this, shard/batch forwards
                # smaller than the cached batch crash in apply_rotary_pos_emb
                # (hidden batch N vs cos batch full).
                parts = []
                for t in val:
                    try:
                        _idx = torch.as_tensor(indices, device=t.device)
                        parts.append(t.index_select(batch_dim, _idx))
                    except (RuntimeError, IndexError):
                        logger.warning_once(
                            "shared-cache kwarg element slicing fell back to the unsliced tensor; "
                            "sub-batch draws may see the wrong batch's entry"
                        )
                        parts.append(t)
                selected_others[key] = tuple(parts)
            elif isinstance(val, (str, bool, type(None))):
                selected_others[key] = val
            else:
                selected_others[key] = val
        return selected_inputs, selected_others
