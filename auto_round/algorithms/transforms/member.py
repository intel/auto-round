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
"""Rotation as a first-class pipeline member.

``RotationPreprocessor`` wraps a :class:`~auto_round.algorithms.transforms.base.BaseRotation`
and plugs it into the ordinary :class:`~auto_round.algorithms.composer.AlgorithmComposer`
member lifecycle. It owns the *entire* rotation lifecycle so the composer stays
rotation-agnostic:

* :meth:`rotate_model`   – model-level (Phase 4.5) full-model rotation, or
                           layer-wise preparation (R matrices only).
* :meth:`on_block_ready` – per-block rotation for layer-wise mode, run as the
                           first step of ``compress_block``. Handles ``nblocks``
                           block fusion (``WrapperMultiblock``) so each fused
                           decoder layer receives the correct ``layer_idx``.
* :meth:`finalize_run`   – layer-wise teardown, invoked automatically via the
                           composer's ``members()`` finalize loop.

The composer only ever calls these generic entry points in a thin loop; all
algorithm-specific decisions (does this rotation support layer-wise? which
sub-modules to rotate?) live here / in the concrete ``BaseRotation``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from auto_round.algorithms.transforms.base import BasePreprocessor
from auto_round.utils import logger

if TYPE_CHECKING:  # pragma: no cover
    import torch

    from auto_round.algorithms.composer import BlockContext
    from auto_round.algorithms.transforms.base import BaseRotation


class RotationPreprocessor(BasePreprocessor):
    """Pipeline member that owns a single rotation transform's lifecycle.

    Instances are built by the composer from the rotation configs found in the
    algorithm config list and are added both to ``members()`` (so ``prepare_run``
    / ``finalize_run`` cover them) and to the composer's dedicated rotation list
    (so the model-level and per-block entry points can reach them).
    """

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._rotation: "BaseRotation | None" = None
        # Set once :meth:`rotate_model` decides layer-wise preparation succeeded.
        self._layerwise_active: bool = False

    # ------------------------------------------------------------------
    # Lazy rotation construction
    # ------------------------------------------------------------------
    @property
    def rotation(self) -> "BaseRotation":
        """The concrete :class:`BaseRotation` for this member (built lazily)."""
        if self._rotation is None:
            from auto_round.algorithms.transforms import normalize_rotation_config
            from auto_round.algorithms.transforms.base import BaseRotation

            normalised = normalize_rotation_config(self.config)
            if normalised is None:
                raise ValueError(f"Rotation member received an empty config: {self.config!r}.")
            self._rotation = BaseRotation.from_config(normalised)
        return self._rotation

    @property
    def wants_layerwise(self) -> bool:
        """Whether this member's config requests per-block (layer-wise) rotation."""
        return bool(getattr(self.config, "layerwise", False))

    @property
    def is_layerwise_active(self) -> bool:
        """Whether this member prepared (and will drive) per-block rotation."""
        return self._layerwise_active

    # ------------------------------------------------------------------
    # Model-level entry (Phase 4.5)
    # ------------------------------------------------------------------
    def rotate_model(
        self,
        model: "torch.nn.Module",
        data_type: str = "mx_fp",
        layerwise: "bool | None" = None,
    ) -> "torch.nn.Module":
        """Rotate *model* up-front, or prepare layer-wise rotation matrices.

        Whether to rotate per-block is taken from the rotation config's
        ``layerwise`` field. ``layerwise`` may be passed explicitly to override
        the config (``None`` means "use the config value").

        For full-model rotation the model is rotated immediately and returned.
        For layer-wise rotation (when the underlying algorithm supports it) only
        the rotation matrices are initialised; per-block work is deferred to
        :meth:`on_block_ready`. When layer-wise is requested but unsupported, the
        member transparently falls back to full-model rotation.

        Returns:
            The (possibly mutated) model.
        """
        if layerwise is None:
            layerwise = self.wants_layerwise

        rotation = self.rotation
        if layerwise:
            if rotation.supports_layerwise:
                logger.info(
                    "[Rotation] Layer-wise mode: preparing R matrices only " "(rotation deferred to per-block hook)."
                )
                rotation.prepare_layerwise(model, data_type=data_type)
                self._layerwise_active = True
                return model
            logger.warning(
                f"[Rotation] {rotation.__class__.__name__} does not support "
                "layer-wise mode. Falling back to full-model rotation."
            )

        return rotation.apply_to_model(model, data_type=data_type)

    # ------------------------------------------------------------------
    # Per-block entry (compress_block step 0)
    # ------------------------------------------------------------------
    def on_block_ready(self, block: "torch.nn.Module", ctx: "BlockContext") -> None:
        """Rotate the block about to be quantized (layer-wise mode only).

        No-op unless :meth:`rotate_model` prepared layer-wise rotation. Iterates
        the (possibly fused) decoder layers inside *block* and rotates each with
        its global ``layer_idx``.
        """
        if not self._layerwise_active:
            return

        for layer, layer_idx in self._iter_layers(block, ctx.block_index):
            self.rotation.rotate_layer(layer, layer_idx=layer_idx)

    # ------------------------------------------------------------------
    # Teardown (members() finalize loop)
    # ------------------------------------------------------------------
    def finalize_run(self) -> None:
        """Finalize layer-wise rotation once all blocks have been processed."""
        if not self._layerwise_active:
            return
        model = self.model_context.model if self.model_context is not None else None
        if model is None:
            return
        self.rotation.finalize_layerwise(model)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _iter_layers(block: "torch.nn.Module", block_index: int):
        """Yield ``(decoder_layer, layer_idx)`` for every layer inside *block*.

        ``block_index`` is the global index of the *first* layer in this block.
        When ``nblocks > 1`` several consecutive decoder layers are fused into a
        single :class:`~auto_round.wrapper.WrapperMultiblock`; each fused layer
        ``j`` maps to global index ``block_index + j``. A plain (non-fused) block
        maps directly to ``block_index``.
        """
        from auto_round.wrapper import WrapperMultiblock

        if isinstance(block, WrapperMultiblock):
            sub_layers = list(block.layers)
        else:
            sub_layers = [block]

        for j, layer in enumerate(sub_layers):
            yield layer, block_index + j
