# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

__all__ = ["get_fp_scale", "dequant_mx_fp8", "quant_mx_fp8"]


# def get_fp_scale(scale_e8m0):
#     # https://github.com/pytorch/ao/blob/994a4ba6c869854fcaa6ca7e118fcbd75e6c28cc/torchao/prototype/mx_formats/mx_tensor.py#L337
#     assert scale_e8m0.dtype == torch.uint8, f"Expected uint8, got {scale_e8m0.dtype}"
#     E8M0_EXPONENT_BIAS = 127
#     scale_e8m0 = scale_e8m0.view(torch.uint8)
#     s_offset = scale_e8m0.to(torch.int16) - E8M0_EXPONENT_BIAS
#     # TODO(later): it would be nice if there was a way to do the 2^x operation
#     # in PyTorch without creating a tensor of twos
#     two = torch.full(s_offset.size(), 2.0, device=scale_e8m0.device)
#     # pow(two, s_offset) can be out of range of floating point formats.
#     # TODO(later): handle this for float16 if we decide to support float16
#     # scales.
#     s_fp = torch.pow(two, s_offset)

#     return s_fp


def get_fp_scale(scale_e8m0):
    # https://github.com/pytorch/ao/blob/994a4ba6c869854fcaa6ca7e118fcbd75e6c28cc/torchao/prototype/mx_formats/mx_tensor.py#L337
    E8M0_EXPONENT_BIAS = 127

    scale_e8m0 = scale_e8m0.view(torch.uint8)
    s_offset = scale_e8m0.to(torch.int16) - E8M0_EXPONENT_BIAS
    # TODO(later): it would be nice if there was a way to do the 2^x operation
    # in PyTorch without creating a tensor of twos
    # two = torch.full(s_offset.size(), 2.0, device=scale_e8m0.device)
    # pow(two, s_offset) can be out of range of floating point formats.
    # TODO(later): handle this for float16 if we decide to support float16
    # scales.
    # s_fp = torch.pow(two, s_offset)
    # !!!!NOTE Critical: fixed the OoM issue when using HPU graph
    s_fp = torch.pow(2.0, s_offset.to(torch.float))

    return s_fp


def dequant_mx_fp8(weight_fp8, scale_e8m0, block_size, target_dtype):
    scale_float = get_fp_scale(scale_e8m0)
    weight_bf16 = weight_fp8.to(torch.bfloat16)
    weight_original_shape = weight_bf16.shape
    weight_bf16 = weight_bf16.reshape(-1, block_size)
    scale_float = scale_float.reshape(-1, 1)
    dequant_weight = weight_bf16 * scale_float
    dequant_weight = dequant_weight.reshape(weight_original_shape)
    return dequant_weight.to(target_dtype)





def quant_mx_fp8(tensor):
    from auto_round_extension.vllm_ext.utils import to_mx_fp8e4m3

    scale_e8m0_biased, data_lp = to_mx_fp8e4m3(
        data_hp=tensor,
        elem_dtype=torch.float8_e4m3fn,
        block_size=32,
    )
    return scale_e8m0_biased, data_lp
