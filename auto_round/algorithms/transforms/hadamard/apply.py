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
from auto_round.algorithms.transforms.hadamard.config import RotationConfig, normalize_rotation_config
from auto_round.algorithms.transforms.hadamard.transforms import build_hadamard_transform
from auto_round.compressors.utils import is_nv_fp
from auto_round.experimental.qmodules.base import QModuleBase

__all__ = ["HadamardRotation", "apply_rotation_transform", "apply_hadamard_transform"]


def _triton_available(data_type: str = "mx_fp") -> bool:
    """Best-effort check for whether Triton kernel path can be used."""
    if is_nv_fp(data_type):
        return False
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

    def __init__(self, config: RotationConfig) -> None:
        super().__init__(config)

    @classmethod
    def from_config(cls, config: dict | RotationConfig) -> "HadamardRotation":
        """Build a :class:`HadamardRotation` from a raw dict or :class:`RotationConfig`."""
        if isinstance(config, dict):
            config = RotationConfig.model_validate(config)
        return cls(config)

    def apply_to_model(
        self,
        model: torch.nn.Module,
        location: str = "weight",
        use_tqdm: bool = True,
        desc: str | None = None,
        data_type: str = "mx_fp",
        **kwargs: Any,
    ) -> torch.nn.Module:
        """Apply the Hadamard rotation to *model*.

        Args:
            model:           Target model; modified in-place.
            location:        ``"weight"`` (eager, fused into weights) or
                             ``"input"`` (activation-side, via forward hook).
            use_tqdm:        Show a progress bar while iterating modules.
            desc:            Custom progress-bar description.
            data_type:       Quantization data type (e.g. ``"mx_fp"``).
            **kwargs:        Reserved for future use.

        Returns:
            The mutated *model* with ``model.rotation_config`` set to the
            normalised :class:`RotationConfig` dict.
        """
        cfg = self.config

        # Collect target modules.
        target_types = (torch.nn.Linear, QModuleBase)

        modules = [(name, module) for name, module in model.named_modules() if isinstance(module, target_types)]

        _desc = desc or f"Applying {cfg.hadamard_type} transforms"
        for name, module in tqdm.tqdm(modules, desc=_desc, disable=not use_tqdm):
            if "lm_head" in name:
                continue
            _apply_to_module(model, module, cfg, location, data_type)

        # Store config on model for serialisation / downstream inspection.
        setattr(model, "rotation_config", cfg.model_dump())
        return model


# ---------------------------------------------------------------------------
# Module-level application helper
# ---------------------------------------------------------------------------


def _apply_to_module(
    model: torch.nn.Module,
    module: torch.nn.Module,
    config: RotationConfig,
    location: str,
    data_type: str = "mx_fp",
) -> None:
    """Apply the configured Hadamard transform to a single *module*."""
    if location == "input":
        _apply_input_transform(module, config, data_type)

    elif location == "weight":
        _apply_weight_transform(module, config)

    else:
        raise NotImplementedError(f"Unsupported transform location: {location!r}")


def _apply_input_transform(module: torch.nn.Module, config: RotationConfig, data_type: str = "mx_fp") -> None:
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

    if _triton_available(data_type):
        from auto_round.algorithms.transforms.hadamard.utils.triton.mxfp4 import mxfp4_forward_kernel_wrapper

        def _input_hook(self, args):
            x = args[0]
            orig_shape = x.shape
            orig_dtype = x.dtype
            x_flat = x.contiguous().flatten(end_dim=-2)
            w = hadamard_weight.to(orig_dtype) if hadamard_weight is not None else self.hadamard_matrix.T.to(orig_dtype)
            qdq_input, _ = mxfp4_forward_kernel_wrapper(x_flat, w)
            return qdq_input.reshape(orig_shape).to(orig_dtype)

        module.pre_dequantized_input = True
        module.register_forward_pre_hook(_input_hook, prepend=True)
    else:

        def _input_hook(self, args):
            x = args[0]
            ori_shape = x.shape
            orig_dtype = x.dtype
            if hadamard_weight is not None:
                x = x.view(-1, hadamard_weight.shape[0])
                return multihead_matmul(x, hadamard_weight.to(x.device).to(orig_dtype)).view(ori_shape).to(orig_dtype)
            else:
                x = x.view(-1, self.hadamard_matrix.shape[0])
                return multihead_matmul(x, self.hadamard_matrix.T.to(orig_dtype)).view(ori_shape).to(orig_dtype)

        module.pre_dequantized_input = False
        module.register_forward_pre_hook(_input_hook, prepend=True)


def _apply_weight_transform(
    module: torch.nn.Module,
    config: RotationConfig,
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
    )

    # For random Hadamard, save the matrix as a submodule for serialisation.
    if config.hadamard_type == "random_hadamard":
        from auto_round.algorithms.transforms.hadamard.patch import patch_quantlinear as _patch_ql

        _patch_ql(w_transform)

    # Patch WrapperLinear and WrapperWALayer so the transform is applied
    # during calibration tuning.
    inp_transform = build_hadamard_transform(
        **config.model_dump(),
        location="input",
        inverse=True,
        device=module.weight.device,
        precision=module.weight.dtype,
    )

    patch_wrapperlinear_to_apply_transform(w_transform, inp_transform)
    patch_wrapperwalayer_forward_to_apply_transform(inp_transform)


# ---------------------------------------------------------------------------
# Convenience one-shot function
# ---------------------------------------------------------------------------


def apply_rotation_transform(
    model: torch.nn.Module,
    config: str | dict | RotationConfig | None,
    location: str = "weight",
    use_tqdm: bool = True,
    desc: str | None = None,
    data_type: str = "mx_fp",
) -> torch.nn.Module:
    """Apply a Hadamard rotation to *model*.

    This is the main public entry point when you only want Hadamard (rather
    than the polymorphic :func:`~auto_round.algorithms.transforms.apply_rotation`).

    Args:
        model:            Target model.
        config:           One of: :class:`RotationConfig`, ``dict``, ``str``
                          shorthand, or ``None`` (no-op).
        location:         ``"weight"`` or ``"input"``.
        use_tqdm:         Show progress bar.
        desc:             Custom progress-bar label.
        data_type:        Quantization data type (e.g. ``"mx_fp"``).

    Returns:
        The transformed model.
    """
    normalised = normalize_rotation_config(config)
    if not normalised:
        return model
    rotation = HadamardRotation.from_config(normalised)
    return rotation.apply_to_model(
        model,
        location=location,
        use_tqdm=use_tqdm,
        desc=desc,
        data_type=data_type,
    )


# Backward-compatibility alias
apply_hadamard_transform = apply_rotation_transform
