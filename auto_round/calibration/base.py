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
from typing import TYPE_CHECKING, Any, Callable

import torch

if TYPE_CHECKING:
    from auto_round.compressors.base import BaseCompressor


class Calibrator(ABC):
    """Abstract base for all calibration strategies."""

    # TODO wenhuach remove compressor in init
    def __init__(self, compressor: "BaseCompressor", **kwargs) -> None:
        self.model = compressor.model_context.model
        self.tokenizer = compressor.model_context.tokenizer
        self.dataset = compressor.dataset
        self.seed = compressor.seed
        self.low_gpu_mem_usage = compressor.low_gpu_mem_usage
        self.batch_size = compressor.batch_size
        self.batch_dim = compressor.calibration_state.batch_dim
        self.has_variable_block_shape = compressor.has_variable_block_shape
        self.shared_cache_keys = compressor.model_context.shared_cache_keys
        self.is_only_supported_bs1 = False
        self.seqlen = compressor.seqlen
        self.hook_handles = []
        self.inputs = {}

    # ── Public API ──────────────────────────────────────────────────────────

    @abstractmethod
    def collect(self, block_names, nsamples, layer_names=None, last_cache_name=None):
        """Run calibration and return ``inputs`` dict for the given blocks/layers."""

    @abstractmethod
    def cache_inter_data(self, block_names, nsamples, layer_names=None, last_cache_name=None):
        """Replace forward, run :meth:`calib`, return cached ``inputs``."""

    @abstractmethod
    def calib(self, nsamples: int, bs: int) -> None:
        """Drive the model so block-forward hooks fire.

        Subclasses (LLM / MLLM / Diffusion / ...) implement their own data
        loading and forward driver here.
        """

    # ── Optional hooks (sane defaults) ─────────────────────────────────────

    def should_stop(self, name: str) -> bool:
        """Default early-stop policy: delegate to the module helper.

        Subclasses (e.g. ``DiffusionCalibrator``) may override to always
        return ``False`` so the pipeline runs every denoising step.
        """
        return self.should_stop_cache_forward(name)

    def wrap_block_forward(self, forward_fn):
        """Optionally wrap the block-forward function.  Default: passthrough.

        Subclasses (e.g. ``DiffusionCalibrator``) may override to convert
        positional → kwargs so diffusion blocks can be captured uniformly.
        """
        return forward_fn

    @torch.no_grad()
    def _get_block_forward_func(self, name: str) -> Callable:
        """Build the block-forward replacement, then let the calibrator wrap it.

        ``Calibrator.wrap_block_forward`` defaults to passthrough; the
        Diffusion calibrator overrides it to convert positional → kwargs.
        """
        fn = self.make_block_forward_func(name)  # TODO have a double check wenhuach

        fn = self.wrap_block_forward(fn)
        return fn

    def _should_stop_cache_forward(self, name: str) -> bool:
        """Bridge hook stop checks to the calibrator's stop policy."""
        return self.should_stop(name)

    # def __getattr__(self, name: str) -> Any:
    #     # Anything not defined on the calibrator is read off the compressor.
    #     # Calibrator code can use ``self.model_context`` / ``self.quantizer`` /
    #     # ``self.dataset`` / etc. as if it were still a method on DataDrivenCompressor.
    #     compressor = self.__dict__.get("compressor", None)
    #     if compressor is None:
    #         raise AttributeError(name)
    #     return getattr(compressor, name)
