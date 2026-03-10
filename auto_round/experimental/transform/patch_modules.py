# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import transformers
from ...wrapper import WrapperLinear, WrapperWALayer


#def patch_wrapperlinear_to_apply_transform(transform_attr: str = "forward_hadamard"):
def patch_wrapperlinear_to_apply_transform(transform):
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

        min_scale.data.clamp_(0, 1.0)
        max_scale.data.clamp_(0, 1.0)

        weight = self.orig_layer.weight
        if weight.device.type == "meta":
            weight = self.orig_layer.get_weight().to(self.device)

        is_conv1d = type(self.orig_layer) == transformers.pytorch_utils.Conv1D
        if is_conv1d:
            weight = weight.t()

        weight = weight.to(self.device)

        # transform = getattr(self.orig_layer, transform_attr)
        weight_t = transform(weight)

        quant_kwargs = {}
        if hasattr(self.orig_layer, "super_bits"):
            quant_kwargs["super_bits"] = self.orig_layer.super_bits
            quant_kwargs["super_group_size"] = self.orig_layer.super_group_size

        weight_q, scale, zp = self.weight_quant_func(
            weight_t,
            bits=self.orig_layer.bits,
            group_size=self.orig_layer.group_size,
            v=value,
            min_scale=min_scale,
            max_scale=max_scale,
            scale_dtype=self.orig_layer.scale_dtype,
            tensor_min=self.weight_min,
            tensor_max=self.weight_max,
            data_type=self.data_type,
            q_scale_thresh=self.q_scale_thresh,
            imatrix=self.orig_layer.imatrix.to(self.device) if hasattr(self.orig_layer, "imatrix") else None,
            global_scale=getattr(self, "weight_global_scale", None),
            **quant_kwargs,
        )

        weight_q = weight_q.to(dtype=weight.dtype)

        if is_conv1d:
            weight_q = weight_q.t()

        return weight_q, scale, zp

    def _qdq_act_patched(self, x, act_max_scale, act_max=None):

        # transform = getattr(self.orig_layer, transform_attr)
        x = transform(x)
        act_max_scale.data.clamp_(0, 1.0)
        x, scale, zp = self.act_quant_func(
            x,
            bits=self.orig_layer.act_bits,
            group_size=self.orig_layer.act_group_size,
            scale_dtype=self.orig_layer.scale_dtype,
            q_scale_thresh=self.q_scale_thresh,
            data_type=self.act_data_type,
            max_scale=act_max_scale,
            tensor_max=act_max,
            global_scale=getattr(self, "input_global_scale", None),
        )
        return x, scale, zp

    WrapperLinear._qdq_weight = _qdq_weight_patched
    WrapperLinear._qdq_act = _qdq_act_patched
    WrapperLinear._hadamard_patched = True


#def patch_wrapperwalayer_forward_to_apply_transform(transform_attr: str = "forward_hadamard"):
def patch_wrapperwalayer_forward_to_apply_transform(transform):
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
        x = transform(x)

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
