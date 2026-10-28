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
        cls._register_qad_teacher_args(registry)
        cls._register_qad_tuning_args(registry)

    @classmethod
    def _register_qad_teacher_args(cls, registry: AlgorithmParameterRegistry) -> None:
        """Teacher-distillation-only args (not used by the pure-QAT variant)."""
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
            "--qad_kd_weight",
            field="qad_kd_weight",
            default=1.0,
            type=float,
            help=(
                "Weight of the teacher-distribution KD (soft) loss in the QAD objective: "
                "loss = qad_kd_weight * KL(p_T || p_S) + qad_ce_weight * CE(label, p_S). "
                "Set to 0 to drop distillation and train on hard labels only (pure QAT); "
                "when 0 the offline teacher cache is skipped entirely."
            ),
        )
        registry.add_argument(
            "--qad_ce_weight",
            field="qad_ce_weight",
            default=0.0,
            type=float,
            help=(
                "Weight of the hard-label next-token cross-entropy term added to the QAD loss. "
                "0 (default) keeps pure teacher distillation; a positive value mixes in the "
                "ground-truth label CE, turning QAD into quantization-aware training (QAT). "
                "Use together with --qad_kd_weight to interpolate KD<->QAT."
            ),
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

    @classmethod
    def _register_qad_tuning_args(cls, registry: AlgorithmParameterRegistry) -> None:
        """Whole-model tuning knobs shared by QAD and the pure-QAT variant."""
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
            "--tune_non_quant_params",
            field="tune_non_quant_params",
            default=False,
            action="boolean_optional",
            help=(
                "Also train the model's non-quantized FP parameters (e.g. lm_head, embeddings, "
                "norms, biases) jointly with the quantization params during QAD/QAT whole-model "
                "tuning, instead of freezing them. Useful for QAT where a kept-FP lm_head should "
                "adapt to the quantized backbone."
            ),
        )
        registry.add_argument(
            "--non_quant_params_lr",
            field="non_quant_params_lr",
            default=None,
            type=float,
            help=(
                "Learning rate for the non-quantized FP parameters when --tune_non_quant_params is "
                "set. Falls back to the QAD rounding lr when unset."
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
        qad_workspace: str | None = None,
        gradient_checkpointing: bool = True,
        qad_kd_weight: float = 1.0,
        qad_ce_weight: float = 0.0,
        tune_non_quant_params: bool = False,
        non_quant_params_lr: float | None = None,
        **kwargs,
    ) -> None:        # QAD implies whole-network joint distillation. `enable_qad` forces
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
        # ``qad_optimizer`` is the QAD-specific name; a legacy ``optimizer=`` kwarg
        # is still accepted (pulled from ``kwargs``) for API backward compatibility.
        optimizer = (qad_optimizer or kwargs.pop("optimizer", None) or "signround").lower()
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
        # Optionally fold the non-quantized FP params (lm_head, embeddings, norms,
        # biases) into the joint tuning instead of freezing them (QAT-style).
        self.tune_non_quant_params = bool(tune_non_quant_params)
        if non_quant_params_lr is not None and non_quant_params_lr <= 0:
            raise ValueError("`non_quant_params_lr` must be positive")
        self.non_quant_params_lr = non_quant_params_lr
        if self.tune_non_quant_params and self.qad_optimizer == "signround":
            logger.warning_once(
                "`tune_non_quant_params` trains full-precision params (lm_head/embeddings/norms) with "
                "SignSGD, whose coarse sign-only updates are ill-suited to FP fine-tuning; consider "
                "`--qad_optimizer adamw` (optionally with a small `--non_quant_params_lr`, e.g. 1e-5)."
            )
        # Loss mixing: total = kd_weight * KD(teacher) + ce_weight * CE(label).
        # Pure KD (default) keeps the historical behavior; a positive ce_weight
        # mixes in the hard-label objective (QAT), and kd_weight=0 drops the
        # teacher cache for a pure QAT run.
        self.qad_kd_weight = float(qad_kd_weight)
        self.qad_ce_weight = float(qad_ce_weight)
        if self.qad_kd_weight < 0 or self.qad_ce_weight < 0:
            raise ValueError("`qad_kd_weight` and `qad_ce_weight` must be non-negative")
        if self.qad_kd_weight == 0.0 and self.qad_ce_weight == 0.0:
            raise ValueError("`qad_kd_weight` and `qad_ce_weight` cannot both be 0 (no loss to optimize)")
        if self.qad_kd_weight == 0.0:
            # Pure hard-label objective: no teacher model/cache involved at all.
            logger.warning_once(
                "the `QAT` algorithm is experimental: all text-decoder blocks are tuned jointly "
                "against the ground-truth next-token cross-entropy (no teacher cache)."
            )
        else:
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


