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
"""QAT quantizer: whole-network hard-label training (no teacher cache)."""

from __future__ import annotations

from auto_round.algorithms.quantization.qad.quantizer import QADQuantizer
from auto_round.algorithms.quantization.qat.config import QATConfig
from auto_round.algorithms.registry import register_pipeline_member


@register_pipeline_member(QATConfig)
class QATQuantizer(QADQuantizer):
    """Pure quantization-aware training on ground-truth labels.

    QAT is QAD with ``qad_kd_weight == 0``: the whole model is tuned jointly
    against the hard-label next-token cross-entropy, so the offline teacher
    Top-K cache is skipped entirely (``_qad_use_teacher`` is ``False``). All the
    heavy lifting -- wrapping layers, the whole-model forward, the AdamW/SignSGD
    optimizer selection and the best-param bookkeeping -- is inherited unchanged
    from :class:`QADQuantizer`.
    """
