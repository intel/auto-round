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
"""SpinQuant / QuaRot rotation — ``BaseRotation`` + ``SerializerMixin``.

Registers ``"spinquant"`` in the :class:`BaseRotation` registry so that
the unified ``apply_rotation(model, config)`` entry point can dispatch to
the SpinQuant preprocessing pipeline automatically.

Also implements :class:`SerializerMixin` so that the generic dispatch
functions (``inject_rotation_buffers_on_layer``, etc.) route to SpinQuant
serialization logic without export files ever naming "spinquant".

Usage via the unified entry point::

    from auto_round.algorithms.transforms import apply_rotation
    from auto_round.algorithms.transforms.spinquant import SpinQuantConfig

    config = SpinQuantConfig(r1=True, r2=True, r3=False, r4=False,
                             trainable_rotation=False)
    model = apply_rotation(model, config)

Usage via AutoRound pipeline::

    from auto_round.compressors.entry import AutoRound
    from auto_round.algorithms.transforms.spinquant import SpinQuantConfig

    AutoRound(
        alg_configs=[SignRoundConfig(iters=200), SpinQuantConfig(r1=True, r2=True)],
        model=model,
        scheme="W4A16",
    )
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from auto_round.algorithms.transforms.base import BaseRotation, SerializerMixin

logger = logging.getLogger("autoround")


@BaseRotation.register("spinquant")
class SpinQuantRotation(BaseRotation, SerializerMixin):
    """QuaRot / SpinQuant rotation registered as ``"spinquant"`` in
    :class:`BaseRotation`.

    Delegates to :class:`SpinQuantPreprocessor` for the actual rotation
    pipeline (RMSNorm fusion → rotation matrix init → optional training →
    weight fusion → online hook registration).

    Implements :class:`SerializerMixin` for save/load support — all
    serialization calls are delegated to the existing ``serialize.py``
    functions.

    For QuaRot mode (``trainable_rotation=False``), no ``dataloader`` is
    needed. For SpinQuant mode (``trainable_rotation=True``, ⚠️ experimental),
    pass ``dataloader=`` via ``**kwargs``.
    """

    # ------------------------------------------------------------------
    # Preprocessing (existing, unchanged)
    # ------------------------------------------------------------------

    def apply_to_model(
        self,
        model: torch.nn.Module,
        data_type: str = "mx_fp",
        **kwargs: Any,
    ) -> torch.nn.Module:
        """Apply SpinQuant / QuaRot rotation to *model*.

        Args:
            model: The model to transform (modified in-place).
            data_type: Quantization data type (informational; SpinQuant does
                not change behavior based on this).
            **kwargs: Extra arguments.  ``dataloader`` is forwarded to
                :meth:`SpinQuantPreprocessor.preprocess` when the config
                has ``trainable_rotation=True``.

        Returns:
            The transformed model.
        """
        # Lazy import to avoid circular dependencies — SpinQuantPreprocessor
        # imports from this package's __init__ which imports this module.
        from auto_round.algorithms.transforms.spinquant.preprocessor import (
            SpinQuantPreprocessor,
        )

        dataloader = kwargs.get("dataloader")
        preprocessor = SpinQuantPreprocessor(model, self.config)
        return preprocessor.preprocess(dataloader)

    # ------------------------------------------------------------------
    # SerializerMixin — Save side
    # ------------------------------------------------------------------

    def inject_buffers_on_layer(
        self,
        layer_name: str,
        qlayer: nn.Module,
        model: nn.Module,
    ) -> None:
        """Inject SpinQuant rotation buffers onto a single QuantLinear."""
        config = self._get_model_config(model)
        if config is None:
            return

        try:
            from auto_round.algorithms.transforms.spinquant.serialize import (
                _R1_PREFIX,
                _R4_PREFIX,
                _get_hidden_size,
                _get_intermediate_size,
                _get_stored_rotation,
                _inject_rotation_buffers,
            )

            short_name = layer_name.split(".")[-1]
            r1_proj_names = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
            r4_proj_names = ("down_proj",)

            if config.r1 and config.online_r1_rotation and short_name in r1_proj_names:
                hidden_size = _get_hidden_size(model)
                r1_size = config.rotation_size or hidden_size
                _inject_rotation_buffers(
                    qlayer,
                    _R1_PREFIX,
                    r1_size,
                    random=config.random_r1,
                    is_trained=False,
                    rotation_matrix=_get_stored_rotation(model, "spinquant_R1"),
                )

            if config.r4 and short_name in r4_proj_names:
                intermediate_size = _get_intermediate_size(model)
                r4_size = config.rotation_size or intermediate_size
                _inject_rotation_buffers(
                    qlayer,
                    _R4_PREFIX,
                    r4_size,
                    random=config.random_r4,
                    is_trained=False,
                    rotation_matrix=_get_stored_rotation(model, "spinquant_R4_matrix") if config.random_r4 else None,
                )
        except Exception as e:
            logger.warning(f"Failed to inject SpinQuant buffers on {layer_name}: {e}")

    def inject_buffers_bulk(
        self,
        model: nn.Module,
        quantization_config: dict,
    ) -> None:
        """Inject SpinQuant rotation buffers on all QuantLinear modules."""
        config = self._get_model_config(model)
        if config is None:
            return

        try:
            from auto_round.algorithms.transforms.spinquant.serialize import (
                _config_to_serializable,
                inject_spinquant_buffers,
            )

            n = inject_spinquant_buffers(model, config)
            if n > 0:
                cfg_dict = _config_to_serializable(config, model)
                cfg_dict["algorithm"] = "spinquant"
                # Use "spinquant_config" key (not "rotation_config" —
                # that key is used by the Hadamard rotation system).
                quantization_config["spinquant_config"] = cfg_dict
        except Exception as e:
            logger.warning(f"Failed to inject SpinQuant buffers: {e}")

    def save_config(self, model: nn.Module, save_dir: str) -> None:
        """Save SpinQuant config to config.json."""
        config = self._get_model_config(model)
        if config is None:
            return

        try:
            from auto_round.algorithms.transforms.spinquant.serialize import (
                save_spinquant_config,
            )

            save_spinquant_config(model, save_dir, config)
        except Exception as e:
            logger.warning(f"Failed to save SpinQuant config: {e}")

    # ------------------------------------------------------------------
    # SerializerMixin — Load side
    # ------------------------------------------------------------------

    def preregister_buffers(
        self,
        model: nn.Module,
        config_dict: dict,
    ) -> int:
        """Pre-register empty SpinQuant buffers for state_dict loading."""
        from auto_round.algorithms.transforms.spinquant.serialize import (
            preregister_spinquant_buffers,
        )

        return preregister_spinquant_buffers(model, config_dict)

    def rebuild_online(self, model: nn.Module) -> nn.Module:
        """Rebuild online SpinQuant rotations (forward patch + R3 hook)."""
        from auto_round.algorithms.transforms.spinquant.serialize import (
            rebuild_spinquant_online,
        )

        return rebuild_spinquant_online(model)

    def has_rotation_buffers(self, module: nn.Module) -> bool:
        """Check for spinquant-specific buffer prefixes."""
        return hasattr(module, "spinquant_r1_type") or hasattr(module, "spinquant_r4_type")

    @classmethod
    def config_key(cls) -> str:
        return "spinquant_config"

    # ------------------------------------------------------------------
    # Private helper
    # ------------------------------------------------------------------

    @staticmethod
    def _get_model_config(model: nn.Module):
        """Get rotation config from model, with legacy fallback."""
        config = getattr(model, "_rotation_config", None)
        if config is None:
            config = getattr(model, "_spinquant_config", None)
        return config
