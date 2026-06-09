# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

"""
SpinQuant / QuaRot rotation for Intel AutoRound.

This package provides orthogonal rotation transforms (QuaRot / SpinQuant)
that can be applied before AutoRound quantization to improve accuracy.

Feature Status
--------------
✅ **QuaRot (fixed Hadamard rotation)** — Fully supported.
   Use ``trainable_rotation=False, trainable_smooth=False`` for fixed
   Hadamard rotation (R1–R4) without any training or calibration data.

⚠️  **SpinQuant (trainable rotation)** — Experimental / under development.
   The training loop (``trainable_rotation=True``) has basic infrastructure
   (Cayley SGD optimizer, KL-divergence loss, training hooks) but has NOT
   been validated end-to-end on real models. Use at your own risk.
   Known limitations:
   - Training + R3 (online Q/K rotation) may cause gradient issues
   - ``RotationTrainer.fuse()`` does not handle online R1 mode fully
   - No pre-trained rotation matrices are shipped

⚠️  **Model save/load after rotation**
   Rotated + quantized models are saved with rotation buffers injected into
   QuantLinear modules. Use :func:`inject_spinquant_buffers` before save and
   :func:`rebuild_spinquant_online` after load. See ``serialize.py`` for details.

Main API::

    SpinQuantPreprocessor       – Direct preprocessing (8-step pipeline, recommended)
    SpinQuantConfig              – Configuration dataclass
    apply_spinquant_in_place     – One-shot in-place application
    register_spinquant_hooks     – Online hook registration (R3/R4)

    RotationTrainer              – HuggingFace-Trainer-style SpinQuant trainer (⚠️ experimental)
    RotationTrainerConfig        – Trainer hyperparameters

Example (QuaRot — fixed Hadamard, recommended)::

    from auto_round.algorithms.transforms.spinquant import SpinQuantPreprocessor, SpinQuantConfig

    config = SpinQuantConfig(
        r1=True, r2=True, r3=False, r4=False,
        # Default: trainable_rotation=False, trainable_smooth=False (QuaRot mode)
        online_r1_rotation=True,
    )
    SpinQuantPreprocessor(model, config).preprocess()  # no dataloader needed
    AutoRound(model, tokenizer, bits=4).quantize()

Example (SpinQuant — trainable, ⚠️ experimental)::

    from auto_round.algorithms.transforms.spinquant import (
        RotationTrainer, RotationTrainerConfig
    )

    trainer = RotationTrainer(
        model,
        config=RotationTrainerConfig(
            trainable_rotation=True, iters=200, lr=1e-4,
        ),
    )
    metrics = trainer.train(dataloader)  # requires calibration data
    model = trainer.fuse()

    # Now model is ready for AutoRound
    autoround = AutoRound(model, tokenizer, bits=4, group_size=128)
    autoround.quantize()
"""

from auto_round.algorithms.transforms.spinquant.apply import (
    SpinQuantRotation,
)
from auto_round.algorithms.transforms.spinquant.cayley_optimizer import (
    AdamAndSGDG,
    SGDG,
)
from auto_round.algorithms.transforms.spinquant.inplace.apply import (
    apply_spinquant_in_place,
    register_spinquant_hooks,
    remove_spinquant_hooks,
)
from auto_round.algorithms.transforms.spinquant.preprocessor import (
    SpinQuantConfig,
    SpinQuantPreprocessor,
    TrainableRMSNorm,
)
from auto_round.algorithms.transforms.spinquant.training import (
    LossLogger,
    OrthogonalityMonitor,
    RotationTrainer,
    RotationTrainerCallback,
    RotationTrainerConfig,
    SpinQuantState,
    SpinQuantTrainingHook,
    create_spinquant_optimizer,
    spinquant_loss_fn,
)
from auto_round.algorithms.transforms.spinquant.rotation_utils import (
    InputRotationWrapperHadamard,
)
from auto_round.algorithms.transforms.spinquant.serialize import (
    inject_spinquant_buffers,
    rebuild_spinquant_online,
    save_spinquant_config,
)

__all__ = [
    # -- Registry algorithm (unified apply_rotation() entry) --
    "SpinQuantRotation",
    # -- Preprocessor (QuaRot, recommended) --
    "SpinQuantConfig",
    "SpinQuantPreprocessor",
    "TrainableRMSNorm",
    # -- In-place (AutoRound style) --
    "apply_spinquant_in_place",
    "register_spinquant_hooks",
    "remove_spinquant_hooks",
    # -- Input rotation wrapper (utility, used for rotation-only save/load) --
    "InputRotationWrapperHadamard",
    # -- Serialization (save/load rotated+quantized models) --
    "inject_spinquant_buffers",
    "rebuild_spinquant_online",
    "save_spinquant_config",
    # -- Trainer (⚠️ experimental: training loop not fully validated) --
    "RotationTrainer",
    "RotationTrainerConfig",
    "RotationTrainerCallback",
    "LossLogger",
    "OrthogonalityMonitor",
    # -- Optimiser core (Cayley) --
    "AdamAndSGDG",
    "SGDG",
    # -- Training hooks (⚠️ experimental) --
    "SpinQuantTrainingHook",
    "SpinQuantState",
    "create_spinquant_optimizer",
    "spinquant_loss_fn",
]
