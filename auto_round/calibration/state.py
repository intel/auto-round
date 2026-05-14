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
"""``CalibrationState`` — single source of truth for calibration-time state.

This dataclass owns every per-run calibration field shared between
:class:`~auto_round.compressors.base.BaseCompressor` and
:class:`~auto_round.algorithms.quantization.base.BaseQuantizers`:

- Cache state ``(inputs, to_cached_layers, last_cache_name, blocks_requiring_input_ids)``
- Per-batch shape state ``(attention_mask, batch_dim)``
- Calibration parameters ``(batch_size, nsamples, seqlen, dataset, dataloader)``

Both the compressor and the quantizer hold a reference to the same
``CalibrationState`` instance (wired in ``BaseCompressor._resolve_scheme``).
All legacy attribute reads/writes are routed here through ``@property``
forwarders, so existing call sites need no changes.

The dataclass also provides two behavioural helpers that previously lived
inline in :class:`~auto_round.compressors.data_driven.DataDrivenCompressor`:

- :meth:`clamp_seqlen` — clamps ``self.seqlen`` to model / tokenizer limits.
- :meth:`ensure_dataloader` — builds ``self.dataloader`` from
  ``self.dataset`` (string name or pre-built loader).
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from auto_round.logger import logger

__all__ = ["CalibrationState"]


@dataclass
class CalibrationState:
    """Authoritative runtime store for calibration state.

    See module docstring for field semantics and design rationale.
    """

    # ── Capture buffers ────────────────────────────────────────────────────
    inputs: dict = field(default_factory=dict)
    to_cached_layers: list = field(default_factory=list)
    last_cache_name: Optional[str] = None
    blocks_requiring_input_ids: list = field(default_factory=list)

    # ── Per-batch shape state ──────────────────────────────────────────────
    attention_mask: list = field(default_factory=list)
    batch_dim: Optional[int] = None

    # ── Calibration parameters ─────────────────────────────────────────────
    batch_size: int = 8
    nsamples: int = 128
    seqlen: int = 2048
    dataset: Any = None
    dataloader: Any = None

    # ── Compressor / quantizer round-tripping ──────────────────────────────

    @classmethod
    def from_compressor(cls, compressor: Any) -> "CalibrationState":
        """Return the live shared instance held by ``compressor``.

        The compressor always owns a ``_calibration_state`` after
        ``BaseCompressor.__init__``, so the legacy "snapshot" fallback is
        no longer required.  We still allow it for safety in case a custom
        subclass forgets to call ``super().__init__``.
        """
        live = getattr(compressor, "_calibration_state", None)
        if isinstance(live, cls):
            return live
        # Legacy fallback (no shared instance wired yet).
        return cls(
            inputs=getattr(compressor, "inputs", {}) or {},
            to_cached_layers=getattr(compressor, "to_cached_layers", []) or [],
            last_cache_name=getattr(compressor, "last_cache_name", None),
            blocks_requiring_input_ids=getattr(compressor, "blocks_requiring_input_ids", []) or [],
            batch_size=getattr(compressor, "batch_size", 8) or 8,
            nsamples=getattr(compressor, "nsamples", 128) or 128,
            seqlen=getattr(compressor, "seqlen", 2048) or 2048,
            dataset=getattr(compressor, "dataset", None),
            dataloader=getattr(compressor, "dataloader", None),
        )

    # ── Behavioural helpers ────────────────────────────────────────────────

    def clamp_seqlen(self, model_context: Any) -> None:
        """Clamp :attr:`seqlen` to model / tokenizer maximum lengths.

        Migrated verbatim from ``DataDrivenCompressor._check_compatibility``.
        Safe to call multiple times; warns on each clamp.
        """
        if self.seqlen is None:
            return
        model = getattr(model_context, "model", None)
        max_pos = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        if max_pos is not None and max_pos < self.seqlen:
            logger.warning(f"Change sequence length to {max_pos} due to the limitation of max_position_embeddings")
            self.seqlen = min(self.seqlen, max_pos)

        tokenizer = getattr(model_context, "tokenizer", None)
        tok_max = getattr(tokenizer, "model_max_length", None)
        if tok_max is not None and tok_max < self.seqlen:
            logger.warning(
                f"Change sequence length to {tok_max} due to the limitation of model_max_length. "
                "You can also try to increase the model_max_length to avoid this issue."
            )
            self.seqlen = min(self.seqlen, tok_max)

    def ensure_dataloader(self, model_context: Any, seed: int) -> Any:
        """Resolve :attr:`dataset` into :attr:`dataloader` and return it.

        - If ``self.dataset`` is a string, builds a tokenized dataloader via
          :func:`auto_round.calib_dataset.get_dataloader`.
        - Otherwise, treats ``self.dataset`` as an already-iterable loader.

        Mirrors the inline logic that previously lived in
        ``DataDrivenCompressor._compute_imatrix`` and the calibrator subclasses.
        """
        if isinstance(self.dataset, str):
            tokenizer = getattr(model_context, "tokenizer", None)
            if tokenizer is None:
                raise ValueError("A tokenizer must be set for the model when using a dataset string.")
            from auto_round.calib_dataset import get_dataloader

            dataset_name = self.dataset.replace(" ", "")
            self.dataloader = get_dataloader(
                tokenizer,
                self.seqlen,
                dataset_name,
                seed,
                self.batch_size,
                self.nsamples,
            )
        else:
            self.dataloader = self.dataset
        return self.dataloader
