# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import torch
import transformers

from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear, pack_fp4_to_uint8
from auto_round.wrapper import WrapperLinear, WrapperWALayer


def patch_wrapperlinear_to_apply_transform(w_transform, inp_transform):
    """
    Globally monkey-patch WrapperLinear._qdq_weight and WrapperLinear._qdq_act so that it applies
    a weight and activation transform before quantization.

    e.g. by apply_transform() before wrapper_block().
    """

    if getattr(WrapperLinear, "_hadamard_patched", False):
        return

    orig_qdq_weight = WrapperLinear._qdq_weight

    def _qdq_weight_patched(self, value, min_scale, max_scale):
        """
        # If no transform attached, fall back to original behavior
        if not hasattr(self.orig_layer, transform_attr):
            return orig_qdq_weight(self, value, min_scale, max_scale)
        """

        if self.orig_layer.bits >= 16:
            # keep original behavior for >=16bit to avoid changing semantics unexpectedly
            return orig_qdq_weight(self, value, min_scale, max_scale)

        if getattr(self, "applied_weight_hadamard", None) is None:
            with torch.no_grad():
                weight = self.orig_layer.weight
                if weight.device.type == "meta":
                    weight = self.orig_layer.get_weight().to(self.device)

                is_conv1d = type(self.orig_layer) == transformers.pytorch_utils.Conv1D
                if is_conv1d:
                    weight = weight.t().continuous()
                new_weight = w_transform(weight)
                if is_conv1d:
                    new_weight = weight.t().continuous()
                self.orig_layer.weight.data.copy_(new_weight)
                self.applied_weight_hadamard = True

        return orig_qdq_weight(self, value, min_scale, max_scale)

    orig_qdq_act = WrapperLinear._qdq_act
    def _qdq_act_patched(self, x, act_max_scale, act_max=None):

        x = inp_transform(x)

        return orig_qdq_act(self, x, act_max_scale, act_max)

    WrapperLinear._qdq_weight = _qdq_weight_patched
    WrapperLinear._qdq_act = _qdq_act_patched
    WrapperLinear._hadamard_patched = True


def patch_wrapperwalayer_forward_to_apply_transform(inp_transform):
    """
    Globally monkey-patch WrapperWALayer.forward so that it applies
    a activation transform before quantization.

    e.g. by apply_transform() before wrapper_block().
    """

    if getattr(WrapperWALayer, "_hadamard_forward_patched", False):
        return

    orig_forward = WrapperWALayer.forward

    def _forward_patched(self, x):
        """
        # If no transform attached, fall back to original behavior
        if not hasattr(self.orig_layer, transform_attr):
            return orig_forward(self, x)
        """

        act_max = self.orig_layer.act_max if hasattr(self.orig_layer, "act_max") else None

        # transform = getattr(self.orig_layer, transform_attr)
        x = inp_transform(x)

        x, _, _ = self.orig_layer.act_quant_func(
            x,
            bits=self.orig_layer.act_bits,
            group_size=self.orig_layer.act_group_size,
            scale_dtype=self.orig_layer.scale_dtype,
            q_scale_thresh=self.orig_layer.q_scale_thresh,
            data_type=self.orig_layer.act_data_type,
            tensor_max=act_max,
        )
        return self.orig_layer.forward(x)

    WrapperWALayer.forward = _forward_patched
    WrapperWALayer._hadamard_forward_patched = True


def patch_quantlinear(hadamard_type):
    """ """

    if getattr(QuantLinear, "_pack_patched", False):
        return

    from auto_round.data_type.nvfp import cast_to_fp4, get_reciprocal
    from auto_round.data_type.utils import reshape_pad_tensor_by_group_size, revert_tensor_by_pad
    from auto_round.utils import get_packing_device

    E8M0_EXPONENT_BIAS = 127
    E8M0_EXPONENT_NAN_VAL = 255

    def _pack_patched(
        self, linear, scales, zeros=None, g_idx=None, global_scale=None, input_global_scale=None, device=None
    ):
        device = get_packing_device(device)
        if getattr(linear, "bias", None) is not None:
            self.bias = linear.bias.detach().to(torch.float16)

        W = linear.weight.data.detach().to(device)
        if type(linear) == torch.nn.Conv2d:
            W = W.flatten(1)
        if type(linear) == transformers.pytorch_utils.Conv1D:
            W = W.t()

        tensor, orig_shape, pad_len = reshape_pad_tensor_by_group_size(W, self.group_size)
        scales = scales.to(device)
        if self.is_nv:
            assert global_scale is not None and global_scale.numel() == 1
            global_scale = global_scale.reshape([1])
            global_scale = global_scale.to(device)
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
        # self.weight =  get_compressed_weight(scaled_tensor, self.bits, self.data_type) ## TODO
        if self.bits == 8:
            compress_dtype = torch.float8_e4m3fn
            self.weight = scaled_tensor.to(compress_dtype)

        else:
            compress_dtype = torch.uint8
            self.weight_packed = pack_fp4_to_uint8(scaled_tensor)

        if global_scale is not None:
            self.weight_global_scale = global_scale.to(torch.float32).to(device)

        if input_global_scale is not None:
            # TODO: the shape of `input_global_scale` is [] in some cases — need to investigate why.
            self.input_global_scale = input_global_scale.to(torch.float32).to(device).reshape([1])

        # add transform weight
        transform = getattr(linear, hadamard_type)
        self.register_buffer("hadamard_matrix", transform.weight.to(device))
        return

    QuantLinear.pack = _pack_patched
    QuantLinear._pack_patched = True
