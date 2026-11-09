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
next-token distribution. The teacher is forwarded to the LM head *once*, offline.
The default ``tk`` objective stores compact Top-K ``(index, prob)`` targets;
``kl`` and ``rkl`` instead cache the full vocabulary distribution, increasing
disk and target-memory usage. All objectives use FP32 teacher probabilities by default.
"""

from __future__ import annotations

from auto_round.algorithms.config import AlgorithmParameterRegistry
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.algorithms.registry import register_algorithm
from auto_round.logger import logger


class QADConfig(SignRoundConfig):
    """Configure experimental whole-model quantization-aware distillation (QAD).

    Args:
        qad_nsamples (int | None): QAD/QAT-specific calibration sample count.
            None reuses the common calibration samples; an explicit value overrides
            an inherited ``nsamples`` value without changing other pipeline stages.

    Attributes:
        qad_iters (int | None): QAD/QAT iteration override. None inherits ``iters``;
            the effective count drives the automatic learning rate, scheduler,
            and SignSGD LoRA scaling budget.
        loss (str): Distillation objective: ``tk`` aligns teacher Top-K conditional
            distributions, ``kl`` computes KL(teacher || student), and ``rkl``
            computes KL(student || teacher). The default is ``tk``. Both full KL
            modes cache the entire teacher vocabulary regardless of ``topk``.
    """

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
            "--qad_loss",
            field="loss",
            default="tk",
            choices=["tk", "kl", "rkl"],
            help=(
                "QAD distillation objective: tk (default) aligns teacher Top-K conditional distributions; "
                "kl uses KL(teacher || student); rkl uses KL(student || teacher). "
                "kl/rkl cache the full vocabulary. Teacher probabilities are stored in FP32 for all objectives."
            ),
        )
        registry.add_argument(
            "--topk",
            field="topk",
            default=32,
            type=int,
            help=(
                "Number of top teacher logits cached per token for the QAD tk loss. "
                "Use -1 to store the full vocabulary distribution. kl/rkl always cache the full vocabulary."
            ),
        )
        registry.add_argument(
            "--qad_kd_weight",
            field="qad_kd_weight",
            default=1.0,
            type=float,
            help=(
                "Weight of the teacher-distribution KD (soft) loss in the QAD objective: "
                "loss = qad_kd_weight * KD(--qad_loss) + qad_ce_weight * CE(label, p_S). "
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
                "Defaults to <AR_WORK_SPACE>/qad_teacher_cache, where AR_WORK_SPACE is an environment variable "
                "defaulting to the local directory 'ar_work_space'. Relative paths are resolved against "
                "the current working directory."
            ),
        )

    @classmethod
    def _register_qad_tuning_args(cls, registry: AlgorithmParameterRegistry) -> None:
        """Whole-model tuning knobs shared by QAD and the pure-QAT variant."""
        registry.add_argument(
            "--qad_iters",
            field="qad_iters",
            default=None,
            type=int,
            help=(
                "Number of QAD/QAT optimization iterations; inherits --iters when unset. "
                "The effective count determines the automatic learning rate, linear scheduler, "
                "and automatic SignSGD LoRA scaling independently of block-wise tuning."
            ),
        )
        rounding = registry.add_mutually_exclusive_group()
        rounding.add_argument(
            "--qad_enable_round_tuning",
            field="qad_enable_round_tuning",
            dest="qad_enable_round_tuning",
            default=True,
            action="store_true",
            help="Tune QAD rounding offsets (default).",
        )
        rounding.add_argument(
            "--qad_disable_round_tuning",
            "--no-qad_enable_round_tuning",
            field="qad_enable_round_tuning",
            dest="qad_enable_round_tuning",
            action="store_false",
            help=(
                "Freeze QAD rounding offsets, preserving any warm-start rounding. "
                "LoRA rounding is disabled too; min/max and activation tuning keep their own settings."
            ),
        )
        registry.add_argument(
            "--qad_lora_rank",
            field="qad_lora_rank",
            default=None,
            type=int,
            help=(
                "Replace dense rounding parameters with low-rank B @ A tuning. "
                "0 disables LoRA; when omitted, inherit --lora_rank (default 0). Under TP, rank is per local weight shard; "
                "only rounding is changed, not the FP weights or inference forward."
            ),
        )
        registry.add_argument(
            "--qad_lora_alpha",
            field="qad_lora_alpha",
            default=None,
            type=float,
            help=(
                "Scale rounding LoRA by alpha/rank. When unset, SignSGD with the default linear scheduler "
                "automatically bounds the new offset using the QAD iteration and learning-rate budget; "
                "other optimizers retain unit scaling."
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
                "sample count from block-wise SignRound (e.g. '--alg_configs sign_round,qad'); "
                "falls back to --nsamples when unset."
            ),
        )
        registry.add_argument(
            "--qad_optimizer",
            field="qad_optimizer",
            default="signround",
            choices=["signround", "adamw"],
            help=(
                "Optimizer used for QAD tuning: 'signround' (SignSGD) or 'adamw' (torch.optim.AdamW, weight_decay=0). "
                "AdamW requires substantially more optimizer-state GPU memory because it stores two moment "
                "tensors per trainable parameter."
            ),
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
        qad_iters: int | None = None,
        lr: float | None = None,
        qad_lr: float | None = None,
        qad_nsamples: int | None = None,
        topk: int = 32,
        loss: str = "tk",
        qad_optimizer: str | None = None,
        qad_workspace: str | None = None,
        qad_lora_rank: int | None = None,
        qad_lora_alpha: float | None = None,
        qad_enable_round_tuning: bool = True,
        gradient_checkpointing: bool = True,
        qad_kd_weight: float = 1.0,
        qad_ce_weight: float = 0.0,
        tune_non_quant_params: bool = False,
        non_quant_params_lr: float | None = None,
        **kwargs,
    ) -> None:  # QAD implies whole-network joint distillation. `enable_qad` forces
        # `enable_lfq` and makes the orchestrator fuse all blocks into one group.
        kwargs.setdefault("enable_qad", True)
        # QAD uses its own loss selector; lfq_loss remains a block-wise LFQ option.
        if loss not in ("tk", "kl", "rkl"):
            raise ValueError(f"`loss` must be 'tk', 'kl', or 'rkl', got {loss!r}")
        self.loss = loss
        kwargs.setdefault("lfq_loss", "tk")
        # Resolve the stage-specific count before the base config derives its
        # learning rate. The quantizer also uses this effective `iters` for the
        # scheduler and the automatic SignSGD LoRA scaling budget.
        if qad_iters is not None and (isinstance(qad_iters, bool) or not isinstance(qad_iters, int) or qad_iters < 0):
            raise ValueError("`qad_iters` must be a non-negative integer or None")
        self.qad_iters = qad_iters
        kwargs["iters"] = iters if qad_iters is None else qad_iters
        # `lr=None` keeps the base heuristic; explicit learning rates are not
        # rescaled when overriding the iteration count.
        # `qad_lr` (QAD-specific) takes precedence over the shared `lr`, so QAD
        # can use a different lr than SignRound when both run in one pipeline.
        kwargs.setdefault("lr", lr)
        self.qad_lr = qad_lr
        if qad_lr is not None:
            if qad_lr <= 0:
                raise ValueError("`qad_lr` must be positive")
            kwargs["lr"] = qad_lr
        # Keep a single source of truth in QuantizationConfig.nsamples. None
        # inherits common sampling; an explicit QAD count overrides only this stage.
        if qad_nsamples is not None:
            kwargs["nsamples"] = qad_nsamples
        # Normalize the optimizer choice before handing it to the base config
        # (SignRound keeps it on ``self.optimizer`` for the quantizer to resolve).
        # ``qad_optimizer`` is the QAD-specific name; a legacy ``optimizer=`` kwarg
        # is still accepted (pulled from ``kwargs``) for API backward compatibility.
        optimizer = (qad_optimizer or kwargs.pop("optimizer", None) or "signround").lower()
        if optimizer not in ("signround", "adamw"):
            raise ValueError(f"`qad_optimizer` must be 'signround' or 'adamw', got {optimizer!r}")
        self.qad_optimizer = optimizer
        kwargs["optimizer"] = optimizer
        self.qad_lora_rank = qad_lora_rank
        self.qad_lora_alpha = qad_lora_alpha
        super().__init__(**kwargs)
        self.validate_lora_config()
        self.qad_enable_round_tuning = bool(qad_enable_round_tuning)
        # Force QAD semantics regardless of CLI arg defaults.
        self.enable_qad = True
        self.enable_lfq = True
        self.topk = topk
        if self.topk != -1 and self.topk <= 0:
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
        # Pure KD (default) uses the selected objective; a positive ce_weight
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
                "the `QAD` algorithm is experimental: the full-precision teacher distribution is cached to disk "
                "and all text-decoder blocks are tuned jointly against it."
            )

    @property
    def qad_lora_rank(self) -> int:
        """int: Stage-specific LoRA rank, inheriting the shared rank only when omitted."""
        return self.lora_rank if self._qad_lora_rank is None else self._qad_lora_rank

    @qad_lora_rank.setter
    def qad_lora_rank(self, value: int | None) -> None:
        """Set the QAD-specific rank override without changing the shared rank.

        Args:
            value (int | None): Stage-specific rank, or None to inherit the shared setting.
        """
        self._qad_lora_rank = value

    @property
    def qad_lora_alpha(self) -> float | None:
        """float | None: Stage-specific LoRA alpha, falling back to the shared alpha."""
        return self.lora_alpha if self._qad_lora_alpha is None else self._qad_lora_alpha

    @qad_lora_alpha.setter
    def qad_lora_alpha(self, value: float | None) -> None:
        """Set the QAD-specific alpha override without changing the shared alpha.

        Args:
            value (float | None): Stage-specific alpha, or None to inherit the shared setting.
        """
        self._qad_lora_alpha = value


register_algorithm(
    "qad",
    aliases=("qad",),
    config_factory=QADConfig,
    summary="Quantization-aware distillation with an offline teacher Top-K cache.",
)
