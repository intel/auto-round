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
"""Hadamard rotation – concrete ``BaseRotation`` implementation.

Public entry points
-------------------
* :class:`HadamardRotation` – the stateful algorithm object.
* :func:`apply_hadamard_transform` – convenience one-shot function.
"""

from __future__ import annotations

from typing import Any

import torch
import tqdm

from auto_round.algorithms.transforms.base import BaseRotation
from auto_round.algorithms.transforms.hadamard.config import HadamardConfig, normalize_hadamard_config
from auto_round.algorithms.transforms.hadamard.transforms import build_hadamard_transform
from auto_round.experimental.qmodules.mx import MXQuantLinearBase  # optional dep, guarded below

__all__ = ["HadamardRotation", "apply_hadamard_transform"]


# Detect optional Triton path once at import time.
def _triton_available() -> bool:
    try:
        import triton  # noqa: F401  # pylint: disable=E0401

        if not torch.cuda.is_available():
            return False
        from auto_round.algorithms.transforms.hadamard.utils.triton.mxfp4 import (  # noqa: F401
            mxfp4_forward_kernel_wrapper,
        )

        return True
    except Exception:
        return False


@BaseRotation.register("hadamard")
class HadamardRotation(BaseRotation):
    """Hadamard rotation algorithm.

    Registered under ``"hadamard"`` in the
    :class:`~auto_round.algorithms.transforms.base.BaseRotation` registry.

    Typical usage (via the top-level helper)::

        from auto_round.algorithms.transforms import apply_rotation
        model = apply_rotation(model, config={"hadamard_type": "random_hadamard"})

    Or directly::

        from auto_round.algorithms.transforms.hadamard import apply_hadamard_transform
        model = apply_hadamard_transform(model, config=HadamardConfig(), need_calibration=True)
    """

    def __init__(self, config: HadamardConfig) -> None:
        super().__init__(config)

    @classmethod
    def from_config(cls, config: dict | HadamardConfig) -> "HadamardRotation":
        """Build a :class:`HadamardRotation` from a raw dict or :class:`HadamardConfig`."""
        if isinstance(config, dict):
            config = HadamardConfig.model_validate(config)
        return cls(config)

    def apply_to_model(
        self,
        model: torch.nn.Module,
        need_calibration: bool = False,
        location: str = "weight",
        use_tqdm: bool = True,
        desc: str | None = None,
        **kwargs: Any,
    ) -> torch.nn.Module:
        """Apply the Hadamard rotation to *model*.

        Args:
            model:           Target model; modified in-place.
            need_calibration: When ``True``, calibration wrappers
                              (:class:`~auto_round.wrapper.WrapperLinear`,
                              :class:`~auto_round.wrapper.WrapperWALayer`) are
                              monkey-patched so the transform is re-applied each
                              forward pass during AutoRound tuning.
            location:        ``"weight"`` (eager, fused into weights) or
                             ``"input"`` (activation-side, via forward hook).
            use_tqdm:        Show a progress bar while iterating modules.
            desc:            Custom progress-bar description.
            **kwargs:        Reserved for future use.

        Returns:
            The mutated *model* with ``model.hadamard_config`` set to the
            normalised :class:`HadamardConfig`.
        """
        cfg = self.config

        # Collect target modules.
        try:
            target_types = (torch.nn.Linear, MXQuantLinearBase)
        except Exception:
            target_types = (torch.nn.Linear,)

        modules = [(name, module) for name, module in model.named_modules() if isinstance(module, target_types)]

        _desc = desc or f"Applying {cfg.hadamard_type} transforms"
        for name, module in tqdm.tqdm(modules, desc=_desc, disable=not use_tqdm):
            if "lm_head" in name:
                continue
            _apply_to_module(model, module, cfg, need_calibration, location)

        # Store config on model for serialisation / downstream inspection.
        setattr(model, "hadamard_config", cfg)
        return model


# ---------------------------------------------------------------------------
# Module-level application helper
# ---------------------------------------------------------------------------


def _apply_to_module(
    model: torch.nn.Module,
    module: torch.nn.Module,
    config: HadamardConfig,
    need_calibration: bool,
    location: str,
) -> None:
    """Apply the configured Hadamard transform to a single *module*."""
    from auto_round.algorithms.transforms.hadamard.patch import (
        patch_quantlinear,
        patch_wrapperlinear_to_apply_transform,
        patch_wrapperwalayer_forward_to_apply_transform,
    )

    if location == "input":
        _apply_input_transform(module, config)

    elif location == "weight":
        _apply_weight_transform(module, config, need_calibration)

    else:
        raise NotImplementedError(f"Unsupported transform location: {location!r}")


def _apply_input_transform(module: torch.nn.Module, config: HadamardConfig) -> None:
    """Register a forward pre-hook that applies the Hadamard to the input activation."""
    from auto_round.algorithms.transforms.hadamard.utils.matrix import multihead_matmul

    inp_transform = build_hadamard_transform(
        **config.model_dump(),
        location="input",
        inverse=True,
        device="cpu",
        precision=module.dtype if hasattr(module, "dtype") else None,
    )

    if config.hadamard_type != "random_hadamard":
        hadamard_weight = inp_transform.weight
    else:
        hadamard_weight = None

    if _triton_available():
        from auto_round.algorithms.transforms.hadamard.utils.triton.mxfp4 import mxfp4_forward_kernel_wrapper

        def _input_hook(self, args):
            x = args[0]
            orig_shape = x.shape
            x_flat = x.contiguous().flatten(end_dim=-2)
            w = hadamard_weight if hadamard_weight is not None else self.hadamard_matrix.T
            qdq_input, _ = mxfp4_forward_kernel_wrapper(x_flat, w)
            return qdq_input.reshape(orig_shape)

        module.pre_dequantized_input = True
        module.register_forward_pre_hook(_input_hook, prepend=True)
    else:

        def _input_hook(self, args):
            x = args[0]
            ori_shape = x.shape
            if hadamard_weight is not None:
                x = x.view(-1, hadamard_weight.shape[0])
                return multihead_matmul(x, hadamard_weight.to(x.device)).view(ori_shape)
            else:
                x = x.view(-1, self.hadamard_matrix.shape[0])
                return multihead_matmul(x, self.hadamard_matrix.T).view(ori_shape)

        module.pre_dequantized_input = False
        module.register_forward_pre_hook(_input_hook, prepend=True)


def _apply_weight_transform(
    module: torch.nn.Module,
    config: HadamardConfig,
    need_calibration: bool,
) -> None:
    """Fuse or patch the Hadamard rotation into the weight of *module*."""
    from auto_round.algorithms.transforms.hadamard.patch import (
        patch_quantlinear,
        patch_wrapperlinear_to_apply_transform,
        patch_wrapperwalayer_forward_to_apply_transform,
    )

    assert hasattr(module, "weight"), "Weight transform requires module to have a 'weight' attribute"

    w_transform = build_hadamard_transform(
        **config.model_dump(),
        location="weight",
        device=module.weight.device,
        precision=module.weight.dtype,
    )

    # For random Hadamard, save the matrix as a submodule for serialisation.
    if config.hadamard_type == "random_hadamard":
        module.register_module(config.hadamard_type, w_transform)
        patch_quantlinear(config.hadamard_type)

    if need_calibration:
        inp_transform = build_hadamard_transform(
            **config.model_dump(),
            location="input",
            inverse=True,
            device=module.weight.device,
            precision=module.weight.dtype,
        )
        patch_wrapperlinear_to_apply_transform(w_transform, inp_transform)
        patch_wrapperwalayer_forward_to_apply_transform(inp_transform)
    else:
        # Eagerly fuse the transform into the weight tensor.
        with torch.no_grad():
            module.weight.copy_(w_transform(module.weight).to(module.weight.device))


# ---------------------------------------------------------------------------
# Convenience one-shot function
# ---------------------------------------------------------------------------


def apply_hadamard_transform(
    model: torch.nn.Module,
    config: str | dict | HadamardConfig | None,
    need_calibration: bool = False,
    location: str = "weight",
    use_tqdm: bool = True,
    desc: str | None = None,
) -> torch.nn.Module:
    """Apply a Hadamard rotation to *model*.

    This is the main public entry point when you only want Hadamard (rather
    than the polymorphic :func:`~auto_round.algorithms.transforms.apply_rotation`).

    Args:
        model:            Target model.
        config:           One of: :class:`HadamardConfig`, ``dict``, ``str``
                          shorthand, or ``None`` (no-op).
        need_calibration: See :meth:`HadamardRotation.apply_to_model`.
        location:         ``"weight"`` or ``"input"``.
        use_tqdm:         Show progress bar.
        desc:             Custom progress-bar label.

    Returns:
        The transformed model.
    """
    normalised = normalize_hadamard_config(config)
    if not normalised:
        return model
    rotation = HadamardRotation.from_config(normalised)
    return rotation.apply_to_model(
        model,
        need_calibration=need_calibration,
        location=location,
        use_tqdm=use_tqdm,
        desc=desc,
    )
