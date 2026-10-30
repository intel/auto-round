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
"""QAT (quantization-aware training) algorithm config.

QAT is the pure hard-label special case of QAD: all text-decoder blocks are
tuned jointly against the ground-truth next-token cross-entropy, so the
expensive offline teacher Top-K cache is never built. The config reuses QAD's
whole-model tuning machinery and only drops the distillation-specific knobs
(``--topk``, ``--qad_workspace`` and the KD/CE loss weights).
"""

from __future__ import annotations

from auto_round.algorithms.config import AlgorithmParameterRegistry
from auto_round.algorithms.quantization.qad.config import QADConfig
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.algorithms.registry import register_algorithm


class QATConfig(QADConfig):
    """Configuration for the QAT (quantization-aware training) algorithm.

    Equivalent to ``QADConfig(qad_kd_weight=0, qad_ce_weight=1)`` but exposed as a
    first-class algorithm: it forces the pure hard-label objective and never
    constructs the teacher cache. It inherits every whole-model tuning behavior
    from :class:`QADConfig` and resolves to
    :class:`~auto_round.algorithms.quantization.qat.quantizer.QATQuantizer`.
    """

    @classmethod
    def register_args(cls, registry: AlgorithmParameterRegistry) -> None:
        # Skip QAD's teacher-distillation args (topk / workspace / KD-CE weights);
        # QAT only exposes the shared whole-model tuning knobs (lr, nsamples,
        # optimizer, gradient checkpointing) on top of the SignRound base args.
        SignRoundConfig.register_args(registry)
        cls._register_qad_tuning_args(registry)

    def __init__(self, *, qad_ce_weight: float = 1.0, **kwargs) -> None:
        # Pure hard-label training: no KD term -> no teacher model, no cache.
        kwargs["qad_kd_weight"] = 0.0
        if float(qad_ce_weight) <= 0:
            raise ValueError("`qad_ce_weight` must be > 0 for QAT")
        # Drop any teacher-only kwargs that might arrive from a shared config so
        # they never reach the base initializer with surprising values.
        kwargs.pop("topk", None)
        super().__init__(qad_ce_weight=float(qad_ce_weight), **kwargs)


register_algorithm(
    "qat",
    aliases=("qat",),
    config_factory=QATConfig,
    summary="Quantization-aware training on ground-truth labels (no teacher cache).",
)
