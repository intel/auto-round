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


from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from auto_round.algorithms.registry import resolve_pipeline_member

if TYPE_CHECKING:
    from auto_round.algorithms.block_runner import BlockForwardRunner
    from auto_round.algorithms.composer import AlgorithmComposer
    from auto_round.calibration.state import CalibrationContext
    from auto_round.compressors import BaseOrchestrator
    from auto_round.context.compress import CompressContext
    from auto_round.context.model import ModelContext
    from auto_round.schemes import QuantizationScheme


@dataclass(frozen=True)
class QuantizationRunContext:
    """Compressor-owned state shared with an algorithm for one compression run.
    Created once in :meth:`BaseAlgorithm.bind` and stored as ``self.__run_ctx``
    (name-mangled so subclasses cannot accidentally overwrite it).
    All fields are exposed as read-only properties on :class:`BaseAlgorithm` so
    subclass code (``self.model_context``, ``self.block_forward``, …) continues
    to work without modification.
    This dataclass is **frozen**: the pointers to context objects do not change
    after :meth:`~BaseAlgorithm.bind` is called.  Mutable run-time state lives
    *inside* the referenced objects (e.g. ``calibration_context.batch_size``).
    Attributes:
        model_context:        Model-level metadata (architecture, amp, dtype, …).
        compress_context:     Compression settings (torch compile, packing, …).
        calibration_context:  Calibration state; ``None`` for non-quantizer algorithms.
        block_forward_runner: Shared :class:`~auto_round.algorithms.pipeline.BlockForwardRunner`;
                              ``None`` for non-quantizer algorithms.
        scale_dtype:          Dtype for scales/zero-points; ``None`` for non-quantizer algorithms.
        scheme:               Active :class:`~auto_round.schemes.QuantizationScheme`,
                              or ``None`` before scheme resolution.
    """

    model_context: "ModelContext"
    compress_context: "CompressContext"
    calibration_context: "CalibrationContext | None" = None
    scale_dtype: "torch.dtype | None" = None
    scheme: "QuantizationScheme | None" = None


class BaseAlgorithm:
    """Shared interface for all algorithms in a quantization pipeline.

    Subclass either :class:`~auto_round.algorithms.quantization.base.BaseQuantizer`
    (terminal weight-compression) or
    :class:`~auto_round.algorithms.transforms.base.BasePreprocessor`
    (pre-quantization weight/activation transform).
    """

    def __init__(self, config: Any = None) -> None:
        self.config = config
        # Name-mangled so subclasses cannot accidentally overwrite the run context.
        self.__run_ctx: QuantizationRunContext | None = None
        self.__block_forward_runner: "BlockForwardRunner | None" = None

    @classmethod
    def from_config(cls, config: Any) -> "BaseAlgorithm":
        """Instantiate the registered implementation class for ``config``."""
        alg_cls = resolve_pipeline_member(config)
        if cls is alg_cls:
            return cls(config)
        return alg_cls(config)

    def bind(self, orchestrator: "BaseOrchestrator") -> None:
        """Wire compressor-owned state into a frozen :class:`QuantizationRunContext`.
        Called once before block iteration starts.  Fields not present on the
        compressor (e.g. ``calibration_context`` for preprocessors) default to ``None``.
        """
        self.__run_ctx = QuantizationRunContext(
            model_context=orchestrator.model_context,
            compress_context=orchestrator.compress_context,
            calibration_context=getattr(orchestrator, "calibration_context", None),
            scale_dtype=getattr(orchestrator, "scale_dtype", None),
            scheme=getattr(orchestrator, "scheme_context", None),
        )

    def bind_block_forward_runner(self, block_forward_runner: "type[BlockForwardRunner]") -> None:
        """Bind a shared :class:`~auto_round.algorithms.pipeline.BlockForwardRunner`
        class to the algorithm.  Called once before block iteration starts.
        """
        self.__block_forward_runner = block_forward_runner

    # ── Read-only context accessors ───────────────────────────────────────────
    @property
    def model_context(self) -> "ModelContext | None":
        return self.__run_ctx.model_context if self.__run_ctx is not None else None

    @property
    def compress_context(self) -> "CompressContext | None":
        return self.__run_ctx.compress_context if self.__run_ctx is not None else None

    @property
    def calibration_context(self) -> "CalibrationContext | None":
        return self.__run_ctx.calibration_context if self.__run_ctx is not None else None

    @property
    def block_forward(self) -> "BlockForwardRunner | None":
        """The shared :class:`~auto_round.algorithms.pipeline.BlockForwardRunner` instance."""
        return self.__block_forward_runner if self.__block_forward_runner is not None else None

    @property
    def scale_dtype(self) -> "torch.dtype | None":
        return self.__run_ctx.scale_dtype if self.__run_ctx is not None else None

    @property
    def scheme(self) -> "QuantizationScheme | None":
        return self.__run_ctx.scheme if self.__run_ctx is not None else None

    # ── Derived convenience properties ────────────────────────────────────────
    @property
    def model(self) -> "torch.nn.Module | None":
        return self.model_context.model if self.model_context is not None else None

    @property
    def amp(self) -> bool:
        return getattr(self.model_context, "amp", False)

    @property
    def amp_dtype(self) -> torch.dtype:
        return getattr(self.model_context, "amp_dtype", torch.float32)

    def prepare_run(self, composer: "AlgorithmComposer" = None) -> None:
        """Model-level preparation called once before block iteration starts."""
        return

    def finalize_run(self) -> None:
        """Model-level teardown called once after all blocks are processed."""
        return
