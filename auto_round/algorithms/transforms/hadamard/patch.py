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
"""Monkey-patching helpers to inject Hadamard transforms into calibration wrappers.

During AutoRound calibration (``need_calib=True``) the weight is re-
quantised at every forward pass.  These patches insert the Hadamard rotation
into :class:`~auto_round.wrapper.WrapperLinear` and
:class:`~auto_round.wrapper.WrapperWALayer` so the transform is applied
transparently inside the tuning loop.

Each patch is idempotent: calling it twice has no effect.
"""

from __future__ import annotations

import torch
import transformers

from auto_round.algorithms.transforms.teq.protocol import get_teq_equivalent_transforms
from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear, pack_fp4_to_uint8
from auto_round.wrapper import WrapperLinear, WrapperWALayer

__all__ = [
    "patch_wrapperlinear_to_apply_transform",
    "patch_wrapperwalayer_forward_to_apply_transform",
    "patch_quantlinear",
]


def patch_wrapperlinear_to_apply_transform(
    w_transform: torch.nn.Module,
    inp_transform: torch.nn.Module,
) -> None:
    """Inject *w_transform* and *inp_transform* into :class:`WrapperLinear`.

    After this call, every ``WrapperLinear`` instance will:

    * Apply *w_transform* to the weight before quantisation (``_qdq_weight``).
    * Apply *inp_transform* to the activation before quantisation (``_qdq_act``).

    The patch is written at the **class** level and is therefore global – it
    affects all future instances as well.  A guard flag ``_hadamard_patched``
    prevents double-patching.
    """
    if getattr(WrapperLinear, "_hadamard_patched", False):
        return

    _orig_qdq_weight = WrapperLinear._qdq_weight
    _orig_qdq_act = WrapperLinear._qdq_act

    def _qdq_weight_patched(self, value, min_scale, max_scale):
        # WrapperLinear implements the per-layer protocol natively.  Keep the
        # class-level adapter for backward-compatible patch registration, but
        # never mutate weights here: a mutable transform is incompatible with
        # a trainable TEQ alpha and with independently seeded random layers.
        return _orig_qdq_weight(self, value, min_scale, max_scale)

    def _qdq_act_patched(self, x, act_min_scale=torch.tensor(1.0), act_max_scale=torch.tensor(1.0), act_max=None):
        # Activation-side transforms are applied by WrapperLinear.forward
        # after its TEQ input scale, which gives the equivalent order
        # ``(x / alpha) @ V`` rather than the non-commuting ``(x @ V) / alpha``.
        return _orig_qdq_act(self, x, act_min_scale=act_min_scale, act_max_scale=act_max_scale, act_max=act_max)

    WrapperLinear._qdq_weight = _qdq_weight_patched
    WrapperLinear._qdq_act = _qdq_act_patched
    WrapperLinear._hadamard_patched = True


def patch_wrapperwalayer_forward_to_apply_transform(
    inp_transform: torch.nn.Module,
) -> None:
    """Inject *inp_transform* into :class:`WrapperWALayer`.forward.

    After this call every ``WrapperWALayer`` will rotate its input activation
    before the activation quantisation step.  Idempotent via
    ``_hadamard_forward_patched`` guard.
    """
    if getattr(WrapperWALayer, "_hadamard_forward_patched", False):
        return

    _orig_forward = WrapperWALayer.forward

    def _forward_patched(self, x):
        # The native wrapper reads the per-layer protocol and executes any
        # stolen online hooks exactly once.  Delegating avoids the former
        # class-level transform being applied a second time after a committed
        # rotated weight has installed its matching source pre-hook.
        return _orig_forward(self, x)

    WrapperWALayer.forward = _forward_patched
    WrapperWALayer._hadamard_forward_patched = True


def patch_quantlinear(w_transform) -> None:
    """Patch :class:`QuantLinear` so random Hadamard matrices are saved when packing.

    Only needed for ``random_hadamard`` where the rotation matrix must be
    serialised alongside the quantised weights for correct inference.
    Idempotent via ``_pack_patched`` guard.
    """
    if getattr(QuantLinear, "_pack_patched", False):
        return

    from auto_round.data_type.nvfp import cast_to_fp4, get_reciprocal
    from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad
    from auto_round.utils import get_packing_device

    E8M0_EXPONENT_BIAS = 127
    E8M0_EXPONENT_NAN_VAL = 255

    def _pack_patched(
        self,
        linear,
        scales,
        zeros=None,
        g_idx=None,
        global_scale=None,
        input_global_scale=None,
        device=None,
    ):
        device = get_packing_device(device)
        if getattr(linear, "bias", None) is not None:
            self.bias = linear.bias.detach().to(torch.float16)

        W = linear.weight.data.detach().to(device)
        if type(linear) is torch.nn.Conv2d:
            W = W.flatten(1)
        if type(linear) is transformers.pytorch_utils.Conv1D:
            W = W.t()

        tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(W, self.group_size)
        scales = scales.to(device)
        if self.is_nv:
            assert global_scale is not None and global_scale.numel() == 1
            global_scale = global_scale.reshape([1]).to(device)
            scaled_tensor = tensor.to(global_scale.dtype) * get_reciprocal(
                scales.reshape(tensor.shape[0], -1) * get_reciprocal(global_scale)
            )
            scaled_tensor.clamp_(-6.0, 6.0)
            scaled_tensor = cast_to_fp4(scaled_tensor)
        else:
            scaled_tensor = tensor / (2 ** scales.reshape(tensor.shape[0], -1))
        scaled_tensor = revert_tensor_by_pad(scaled_tensor, orig_shape=orig_shape, pad_len=pad_len)
        if self.is_mx:
            final_scale = (scales + E8M0_EXPONENT_BIAS).clamp(0, E8M0_EXPONENT_NAN_VAL).to(torch.uint8)
        else:
            final_scale = scales.to(torch.float8_e4m3fn)

        self.weight_scale = final_scale
        if self.bits == 8:
            self.weight = scaled_tensor.to(torch.float8_e4m3fn)
        else:
            self.weight_packed = pack_fp4_to_uint8(scaled_tensor)

        if global_scale is not None:
            self.weight_global_scale = global_scale.to(torch.float32).to(device)
        if input_global_scale is not None:
            self.input_global_scale = input_global_scale.to(torch.float32).to(device).reshape([1])

        # Each random-Hadamard layer owns a distinct matrix.  The old
        # class-level closure reused the first layer's matrix for every packed
        # layer, which made reloaded inference disagree with calibration.
        transform = get_teq_equivalent_transforms(linear).get("weight")
        if transform is None:
            transform = w_transform
        self.register_buffer("hadamard_matrix", transform.weight.to(device))
        return

    QuantLinear.pack = _pack_patched
    QuantLinear._pack_patched = True
