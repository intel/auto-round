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
from auto_round.export.formats.autoround.qlinear_fp import QuantLinear, pack_fp4_to_uint8
from auto_round.wrapper import WrapperLinear, WrapperWALayer

__all__ = [
    "patch_quantlinear",
    "patch_wrapperlinear_to_apply_transform",
    "patch_wrapperwalayer_forward_to_apply_transform",
]


def patch_wrapperlinear_to_apply_transform(
    w_transform: torch.nn.Module,
    inp_transform: torch.nn.Module,
) -> None:
    """Install the compatibility adapter for legacy Hadamard callers.

    Transform application now lives in :class:`WrapperLinear` and uses the
    per-layer protocol published by :mod:`auto_round.algorithms.transforms.teq.protocol`.
    The arguments are retained for compatibility, but shared implementations
    must not capture them in a class-level closure because each layer can own a
    different transform.

    The patch is written at the **class** level and is therefore global – it
    affects all future instances as well.  A guard flag ``_hadamard_patched``
    prevents double-patching.
    """
    if getattr(WrapperLinear, "_hadamard_patched", False):
        return

    _orig_qdq_weight = WrapperLinear._qdq_weight

    def _qdq_weight_patched(self, value, min_scale, max_scale):
        # WrapperLinear implements the per-layer protocol natively. Keep the
        # adapter for compatibility, but do not mutate weights in the patch.
        return _orig_qdq_weight(self, value, min_scale, max_scale)

    _orig_qdq_act = WrapperLinear._qdq_act

    def _qdq_act_patched(self, x, act_min_scale=None, act_max_scale=None, act_max=None):
        # WrapperLinear applies the rotation after x / alpha in forward().
        return _orig_qdq_act(self, x, act_min_scale=act_min_scale, act_max_scale=act_max_scale, act_max=act_max)

    WrapperLinear._qdq_weight = _qdq_weight_patched
    WrapperLinear._qdq_act = _qdq_act_patched
    WrapperLinear._hadamard_patched = True


def patch_wrapperwalayer_forward_to_apply_transform(
    inp_transform: torch.nn.Module,
) -> None:
    """Install the compatibility adapter for legacy WrapperWALayer callers.

    :class:`WrapperWALayer` reads the per-layer transform protocol directly.
    The argument is retained for compatibility; capturing it here would apply
    the first layer's transform to every layer. Idempotent via the
    ``_hadamard_forward_patched`` guard.
    """
    if getattr(WrapperWALayer, "_hadamard_forward_patched", False):
        return

    _orig_forward = WrapperWALayer.forward

    def _forward_patched(self, x):
        return _orig_forward(self, x)

    WrapperWALayer.forward = _forward_patched
    WrapperWALayer._hadamard_forward_patched = True


def patch_quantlinear(w_transform) -> None:
    """Patch :class:`QuantLinear` so random Hadamard matrices are saved when packing.

    Only needed for ``random_hadamard`` where the rotation matrix must be
    serialised alongside the quantised weights for correct inference.
    The argument is retained for compatibility; packing requires each layer's
    own matrix-backed protocol transform.
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
        transform = get_teq_equivalent_transforms(linear).get("weight")
        transform_matrix = getattr(transform, "weight", None)
        if not isinstance(transform_matrix, torch.Tensor):
            raise ValueError(
                "Random Hadamard export requires a per-layer matrix-backed weight transform "
                "with a tensor `.weight` attribute."
            )
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

        self.register_buffer("hadamard_matrix", transform_matrix.to(device))

    QuantLinear.pack = _pack_patched
    QuantLinear._pack_patched = True
