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
"""QAD (quantization-aware distillation) algorithm config.

QAD fuses all text-decoder blocks into a single jointly-optimized group
(whole-network training) and tunes them against the full-precision teacher's
next-token distribution. To keep GPU/CPU memory low, the teacher is forwarded to
the LM head *once*, offline, and only the Top-K ``(index, prob)`` per token is
cached to disk; the student reads those compact tensors during tuning.
"""

from __future__ import annotations

from auto_round.algorithms.config import AlgorithmParameterRegistry
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.algorithms.registry import register_algorithm
from auto_round.logger import logger


class QADConfig(SignRoundConfig):
    """Configuration for the QAD (quantization-aware distillation) algorithm."""

    _warn_enable_qad = False

    @classmethod
    def register_args(cls, registry: AlgorithmParameterRegistry) -> None:
        super().register_args(registry)
        registry.add_argument(
            "--topk",
            field="topk",
            default=32,
            type=int,
            help=(
                "Number of top teacher logits cached per token for QAD distillation. "
                "Use -1 to store the full vocabulary distribution."
            ),
        )
        registry.add_argument(
            "--qad_lr",
            field="qad_lr",
            default=None,
            type=float,
            help=(
                "Learning rate for QAD tuning (rounding params). Lets QAD use a different "
                "lr from block-wise SignRound; falls back to --lr when unset."
            ),
        )
        registry.add_argument(
            "--qad_nsamples",
            field="qad_nsamples",
            default=None,
            type=int,
            help=(
                "Number of calibration samples used by QAD tuning. Lets QAD use a different "
                "sample count from block-wise SignRound (e.g. '--algs sign_round,qad'); "
                "falls back to --nsamples when unset."
            ),
        )
        registry.add_argument(
            "--qad_optimizer",
            field="qad_optimizer",
            default="signround",
            choices=["signround", "adamw"],
            help="Optimizer used for QAD tuning: 'signround' (SignSGD) or 'adamw' (torch.optim.AdamW).",
        )
        registry.add_argument(
            "--qad_workspace",
            field="qad_workspace",
            default=None,
            help=(
                "Directory where the offline teacher Top-K cache is written. "
                "Defaults to <AR_WORK_SPACE>/qad_teacher_cache when unset."
            ),
        )
        registry.add_argument(
            "--gradient_checkpointing",
            field="gradient_checkpointing",
            default=True,
            action="boolean_optional",
            help=(
                "Enable activation (gradient) checkpointing during QAD tuning (default: on): the "
                "whole-model backward recomputes block activations instead of "
                "storing them, trading compute for a large activation-memory saving. "
                "Use --no-gradient_checkpointing to disable."
            ),
        )

    def __init__(
        self,
        *,
        iters: int = 200,
        lr: float | None = None,
        qad_lr: float | None = None,
        qad_nsamples: int | None = None,
        topk: int = 32,
        qad_optimizer: str | None = None,
        optimizer: str | None = None,
        qad_workspace: str | None = None,
        gradient_checkpointing: bool = True,
        **kwargs,
    ) -> None:
        # QAD implies whole-network joint distillation. `enable_qad` forces
        # `enable_lfq` and makes the orchestrator fuse all blocks into one group.
        kwargs.setdefault("enable_qad", True)
        # NOTE: QAD distills against the FP teacher's logits: the loss is the
        # forward KL(p_T || p_S) with p_T = softmax(teacher_logits/τ) (Top-K or
        # full vocab, see `_QADDistillLoss`); hard labels are only used as the
        # ignore mask. The inherited `lfq_loss` field is therefore NOT consumed on
        # the QAD path; it is kept in sync with the SignRound default only to
        # avoid a confusing mismatch.
        kwargs.setdefault("lfq_loss", "tk")
        # Tuning schedule: QAD defaults to more iterations than block-wise tuning
        # since the whole network is optimized jointly. `lr=None` keeps the base
        # heuristic (1/iters); pass an explicit value (e.g. 1e-4 for AdamW).
        # `qad_lr` (QAD-specific) takes precedence over the shared `lr`, so QAD
        # can use a different lr than SignRound when both run in one pipeline.
        kwargs.setdefault("iters", iters)
        kwargs.setdefault("lr", lr)
        self.qad_lr = qad_lr
        if qad_lr is not None:
            if qad_lr <= 0:
                raise ValueError("`qad_lr` must be positive")
            kwargs["lr"] = qad_lr
        # `qad_nsamples` lets QAD use a different calibration sample count than the
        # block-wise warm-up (SignRound); None falls back to the global `nsamples`.
        if qad_nsamples is not None and int(qad_nsamples) <= 0:
            raise ValueError("`qad_nsamples` must be positive")
        self.qad_nsamples = int(qad_nsamples) if qad_nsamples is not None else None
        # Normalize the optimizer choice before handing it to the base config
        # (SignRound keeps it on ``self.optimizer`` for the quantizer to resolve).
        # `qad_optimizer` is the QAD-specific name; `optimizer` is still accepted
        # for API backward compatibility.
        optimizer = (qad_optimizer or optimizer or "signround").lower()
        if optimizer not in ("signround", "adamw"):
            raise ValueError(f"`qad_optimizer` must be 'signround' or 'adamw', got {optimizer!r}")
        self.qad_optimizer = optimizer
        kwargs["optimizer"] = optimizer
        super().__init__(**kwargs)
        # Force QAD semantics regardless of CLI arg defaults.
        self.enable_qad = True
        self.enable_lfq = True
        self.topk = topk
        if self.topk == 0:
            raise ValueError("`topk` must be positive, or -1 to store the full distribution")
        self.qad_workspace = qad_workspace
        self.gradient_checkpointing = bool(gradient_checkpointing)
        logger.warning_once(
            "the `QAD` algorithm is experimental: the full-precision teacher Top-K is cached to disk "
            "and all text-decoder blocks are tuned jointly against it."
        )


register_algorithm(
    "qad",
    aliases=("qad",),
    config_factory=QADConfig,
    summary="Quantization-aware distillation with an offline teacher Top-K cache.",
)

