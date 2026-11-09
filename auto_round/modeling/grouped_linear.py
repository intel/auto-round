# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4's block-diagonal output projection (not a dense Linear)."""

import torch

try:
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4GroupedLinear
except ImportError:
    DeepseekV4GroupedLinear = None


def is_grouped_linear(layer):
    return DeepseekV4GroupedLinear is not None and type(layer) is DeepseekV4GroupedLinear


def grouped_linear_forward(x, weight, n_groups):
    """Match Transformers' grouped BMM, including arbitrary leading dimensions."""
    if not isinstance(n_groups, int) or isinstance(n_groups, bool) or n_groups <= 0:
        raise ValueError("Grouped linear requires a positive integer n_groups")
    if x.ndim < 2 or x.shape[-2] != n_groups:
        raise ValueError("Grouped linear input must have shape (..., n_groups, in_features)")
    if weight.ndim != 2 or weight.shape[1] != x.shape[-1] or weight.shape[0] % n_groups:
        raise ValueError("Grouped linear weight must have shape (n_groups * out_per_group, in_features)")
    if x.shape[-1] == 0 or weight.shape[0] == 0:
        raise ValueError("Grouped linear feature dimensions must be positive")
    input_shape = x.shape[:-2]
    hidden_dim = x.shape[-1]
    out_per_group = weight.shape[0] // n_groups
    w = weight.reshape(n_groups, out_per_group, hidden_dim).transpose(1, 2)
    x = x.reshape(-1, n_groups, hidden_dim).transpose(0, 1)
    y = torch.bmm(x, w).transpose(0, 1)
    return y.reshape(*input_shape, n_groups, out_per_group)


def validate_grouped_nvfp4(layer, bits, group_size, data_type):
    if bits != 4 or group_size != 16 or data_type not in ("nv_fp", "nv_fp4"):
        raise ValueError("DeepseekV4GroupedLinear currently supports only NVFP4 with group_size=16")
    if layer.bias is not None:
        raise ValueError("DeepseekV4GroupedLinear NVFP4 requires bias=False")
    weight = getattr(layer, "weight", None)
    weight_shape = (
        tuple(weight.shape) if weight is not None else (layer.weight_packed.shape[0], layer.weight_packed.shape[1] * 2)
    )
    if (
        not isinstance(layer.n_groups, int)
        or isinstance(layer.n_groups, bool)
        or layer.n_groups <= 0
        or layer.out_features <= 0
        or layer.in_features <= 0
        or layer.out_features % layer.n_groups
        or layer.in_features % 16
        or weight_shape != (layer.out_features, layer.in_features)
    ):
        raise ValueError("Invalid DeepseekV4GroupedLinear grouped NVFP4 dimensions")
    validate_grouped_activation(layer)


def validate_grouped_activation(config):
    """Reject activation formats the grouped NVFP4 reference cannot execute."""
    act_bits = getattr(config, "act_bits", 16)
    if act_bits <= 8 and (
        act_bits != 4
        or getattr(config, "act_group_size", 16) != 16
        or getattr(config, "act_data_type", "nv_fp4_with_static_gs")
        not in ("nv_fp4_with_static_gs", "rtn_nv_fp4_with_static_gs")
    ):
        raise ValueError("DeepseekV4GroupedLinear requires static NVFP4 activations with group_size=16 or weight-only")
