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

"""Infrastructure services supplied to a composed quantization run.

Algorithm authors implement BaseQuantizer's stage hooks, not these callbacks.
The composer owns sequencing; the compressor owns caching, devices and storage.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from auto_round.logger import logger


@dataclass
class ModelCompressionContext:
    """Provide compressor-owned services to model-level algorithm stages.

    Attributes:
        model (torch.nn.Module): Model being compressed.
        device (str | torch.device): Device used for model-level execution.
        token_ids (Sequence | None): Cached calibration samples, if available.
        run_blocks (Callable[[], None]): Callback executing the blockwise pipeline.
        reload_model (Callable[[], None]): Callback restoring the original model weights.
        collect_token_ids (Callable[[int], Sequence | None] | None): Optional sample collector.
        nsamples (int): Default number of calibration samples.
        streaming_export (bool): Whether the run writes checkpoints incrementally.
        resuming (bool): Whether compression resumes an existing run.
    """

    model: Any
    device: Any
    token_ids: Any
    run_blocks: Callable[[], None]
    reload_model: Callable[[], None]
    collect_token_ids: Callable[[int], Any] | None = None
    nsamples: int = 128
    streaming_export: bool = False
    resuming: bool = False

    def resolve_token_ids(self, nsamples: int | None) -> Sequence[Any] | None:
        """Reuse a cached prefix or collect raw tokens without block forwards.

        Args:
            nsamples (int | None): Requested sample count, or None to use the stage default.

        Returns:
            Sequence | None: Cached or freshly collected samples, falling back to the
            existing cache if collection is unavailable or empty.
        """
        target = self.nsamples if nsamples is None else nsamples
        cached = self.token_ids
        if cached is not None:
            if nsamples is None or target == len(cached):
                return cached
            if target < len(cached):
                logger.info("Algorithm stage uses the first %d of %d calibration samples.", target, len(cached))
                return list(cached)[:target]
        if self.collect_token_ids is not None:
            logger.info("Collecting %d calibration samples for the algorithm stage.", target)
            token_ids = self.collect_token_ids(target)
            if token_ids is not None and len(token_ids) > 0:
                return token_ids
        logger.warning("Cannot collect %d calibration token ids; falling back to the cached samples.", target)
        return cached
