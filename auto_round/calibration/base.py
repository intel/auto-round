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
"""Calibrator abstract base class.

A ``Calibrator`` owns the *block-input collection* phase of quantization.
It holds a back-reference to the owning ``BaseCompressor`` so it can
read/write shared state (``model_context`` / ``compress_context`` /
``quantizer`` / ``inputs`` / ``dataloader`` / ...).

Extension points for new calibration strategies:

- :meth:`calib` — how the model is driven so hooks fire.
- :meth:`should_stop` — early-stop policy (e.g. diffusion never stops).
- :meth:`wrap_block_forward` — block-forward wrapping (e.g. positional → kwargs).
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from auto_round.compressors.base import BaseOrchestrator as BaseCompressor


class Calibrator(ABC):
    """Abstract base for all calibration strategies."""

    def __init__(self, compressor: "BaseCompressor", **kwargs) -> None:
        self.model = compressor.model_context.model
        self.tokenizer = compressor.model_context.tokenizer
        self.dataset = compressor.dataset
        self.seed = compressor.seed
        self.low_gpu_mem_usage = compressor.low_gpu_mem_usage
        self.batch_size = compressor.calibration_context.batch_size
        self.batch_dim = compressor.calibration_context.batch_dim
        self.has_variable_block_shape = compressor.has_variable_block_shape
        self.shared_cache_keys = compressor.model_context.shared_cache_keys
        self.is_only_supported_bs1 = False
        self.seqlen = compressor.calibration_context.seqlen
        self.hook_handles = []
        self.inputs = {}

    # ── Public API ──────────────────────────────────────────────────────────
    def __call__(self, block_names, nsamples, layer_names=None, last_cache_name=None):
        """Alias for :meth:`calibration` so the calibrator can be called directly.

        Allows ``self.calibration(...)`` instead of
        ``self.calibration.calibration(...)``.
        """
        return self.calibration(block_names, nsamples, layer_names=layer_names, last_cache_name=last_cache_name)

    @abstractmethod
    def calibration(self, block_names, nsamples, layer_names=None, last_cache_name=None):
        """Run calibration and return ``inputs`` dict for the given blocks/layers.

        Drives the model with calibration data, installs forward hooks on the
        specified blocks and layers to capture their inputs, and returns the
        accumulated ``inputs`` mapping used by the quantization algorithm.

        Args:
            block_names (list[str | list[str]]): Fully-qualified module names of
                the transformer blocks (e.g. ``["model.layers.0",
                "model.layers.1"]``) whose inputs should be cached.  Nested
                lists are flattened before use, so grouped blocks may be passed
                as sublists.
            nsamples (int): Target number of calibration samples to accumulate.
                The calibrator iterates the dataloader and stops once
                ``nsamples`` token sequences have been processed.  If fewer
                valid sequences exist in the dataset a warning is emitted and
                calibration continues with the available data.
            layer_names (list[str] | None): Fully-qualified names of individual
                weight layers (e.g. ``nn.Linear``, ``nn.Conv1d``) whose inputs
                should also be cached via forward hooks, in addition to the
                block-level captures.  When ``None`` or an empty list, only
                block inputs are collected.  Defaults to ``None``.
            last_cache_name (str | None): Name of the block or layer at which
                forward-hook collection should stop.  Once the hook for this
                module fires, a ``NotImplementedError`` is raised internally to
                short-circuit the remainder of the forward pass and avoid
                unnecessary computation.  When ``None`` the value is inferred
                automatically as the last entry in the union of ``block_names``
                and ``layer_names``.  Defaults to ``None``.

        Returns:
            dict[str, dict | list]: Nested mapping of
                ``{module_name: {kwarg_name: [tensor, ...]}}`` for blocks, and
                ``{layer_name: [tensor, ...]}`` for individual layers.  Consumed
                directly by the quantization algorithm as per-block input data.
        """

    @abstractmethod
    def _make_block_forward_func(self, name: str) -> Callable:
        """Build and return a forward-replacement ``Callable`` that captures block inputs.

        This is the primary extension point for calibration strategies.  Each
        concrete ``Calibrator`` subclass **must** implement this method.  The
        returned callable is installed as a monkey-patched replacement for the
        block's original ``forward`` method during the input-collection phase.

        Responsibilities of the returned callable
        -----------------------------------------
        1. **Input caching** — accumulate every kwargs/positional tensor passed
           to the block into ``self.inputs[name]``.  Tensors are moved to CPU
           immediately to avoid holding GPU memory across samples.
        2. **Batch splitting** — when ``batch_size > 1`` the per-batch-dim slices
           are stored individually (via ``torch.split``) so downstream quantization
           can iterate sample-by-sample.  ``self.batch_dim`` is auto-detected on
           the first call if it is ``None``.
        3. **Shared-cache keys** — kwargs listed in ``self.shared_cache_keys`` are
           stored once as a scalar value rather than accumulated as a list.
           VLMs that vary per image (e.g. ``position_embeddings``) are transparently
           upgraded from scalar to list storage on the second distinct value.
        4. **Early-stop signalling** — after caching, the callable checks
           ``self._should_stop_cache_forward(name)``.  When ``True`` it raises
           ``NotImplementedError`` to short-circuit the remainder of the forward
           pass and avoid unnecessary computation.  When ``False`` it delegates
           to the block's original forward (stored as ``m.orig_forward``).
        5. **Variable-block-shape models** — when ``self.has_variable_block_shape``
           is set, ``hidden_states`` is conditionally skipped for blocks that do
           not require input-ids, preventing shape-mismatch errors.

        The caller (``_get_block_forward_func``) will optionally wrap the
        result through ``_wrap_block_forward`` before installation, which
        subclasses may override to adapt positional → keyword arguments
        (e.g. diffusion blocks).

        Args:
            name (str): Fully-qualified module name of the transformer block
                whose inputs should be captured (e.g. ``"model.layers.3"``).
                Used as the key in ``self.inputs``.

        Returns:
            Callable: A function with signature
                ``(m, hidden_states=None, *positional_inputs, **kwargs) -> Any``
                intended to replace ``module.forward``.  It must write captured
                data into ``self.inputs[name]`` and either raise
                ``NotImplementedError`` (stop-signal) or return the result of
                ``m.orig_forward(...)``.

        Raises:
            NotImplementedError: Implementations **must not** raise this
                themselves; it is reserved as the stop-signal raised by the
                *returned* callable when ``_should_stop_cache_forward`` is
                ``True``.
        """

    @torch.no_grad()
    def _get_block_forward_func(self, name: str) -> Callable:
        """Build the block-forward replacement, then let the calibrator wrap it.

        ``Calibrator.wrap_block_forward`` defaults to passthrough; the
        Diffusion calibrator overrides it to convert positional → kwargs.
        """
        fn = self._make_block_forward_func(name)  # TODO have a double check wenhuach

        fn = self._wrap_block_forward(fn)
        return fn

    def _wrap_block_forward(self, forward_fn):
        """Optionally wrap the block-forward function.  Default: passthrough.

        Subclasses (e.g. ``DiffusionCalibrator``) may override to convert
        positional → kwargs so diffusion blocks can be captured uniformly.
        """
        return forward_fn

    def _should_stop_cache_forward(self, name: str) -> bool:
        """Bridge hook stop checks to the calibrator's stop policy."""
        return False
