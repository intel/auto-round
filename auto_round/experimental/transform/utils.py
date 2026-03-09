# # Copyright (C) 2026 Intel Corporation
# # SPDX-License-Identifier: Apache-2.0

import transformers

from ...wrapper import WrapperLinear


def patch_wrapperlinear_qdq_weight_to_apply_transform(transform_attr: str = "forward_hadamard"):
    """
    Globally monkey-patch WrapperLinear._qdq_weight so that it applies
    a weight transform before quantization.

    e.g. by apply_transform() before wrapper_block().
    """

    if getattr(WrapperLinear, "_hadamard_qdq_weight_patched", False):
        return

    orig_qdq_weight = WrapperLinear._qdq_weight

    def _qdq_weight_patched(self, value, min_scale, max_scale):
        # If no transform attached, fall back to original behavior
        if not hasattr(self.orig_layer, transform_attr):
            return orig_qdq_weight(self, value, min_scale, max_scale)

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

        transform = getattr(self.orig_layer, transform_attr)
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

    WrapperLinear._qdq_weight = _qdq_weight_patched
    WrapperLinear._hadamard_qdq_weight_patched = True
