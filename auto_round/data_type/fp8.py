import torch

from auto_round.data_type import get_quant_func


def quant_fp8_per_tensor(tensor, bits, data_type, v, min_scale, max_scale, **kwargs):
    ##this is mainly for activation, dynamic now, need to support static later
    info = torch.finfo(torch.float8_e4m3fn)
    max_tensor = torch.max(torch.abs(tensor))  ## better train a ratio
    scale = max_tensor.to(torch.float32) / info.max
    min_scaling_factor = 1.0 / (info.max * 512.0)  ##copy from vllm
    scale = torch.clip(scale, min=min_scaling_factor)
    fp8_res = (tensor / scale).to(torch.float8_e4m3fn)
    qdq_res = (fp8_res.to(tensor.dtype) * scale).to(tensor.dtype)
    return qdq_res, scale, None


def progressive_quant_fp8_int4(tensor, bits, group_size, data_type, v, min_scale, max_scale, **kwargs):
    """
    quantize tensor to fp8 by per tensor, then quantize fp8 to w4g128
    """
    info = torch.finfo(torch.float8_e4m3fn)
    max_tensor = torch.max(torch.abs(tensor))
    scale = max_tensor.to(torch.float32) / info.max
    min_scaling_factor = 1.0 / (info.max * 512.0)  ##copy from vllm
    scale_bf16_to_fp8 = torch.clip(scale, min=min_scaling_factor)
    fp8_res = (tensor / scale_bf16_to_fp8).to(torch.float8_e4m3fn)  ##fp8 does not support many ops
    ##convert to bf16
    fp8_res_using_16bit = fp8_res.to(tensor.dtype)
    ##convert to int4
    from auto_round.quantizer import quant_tensor
    quant_func,_ = get_quant_func("int", 4, True)
    qdq_int4_tensor, scale_fp8_to_int4, zp_fp8_to_int4 = quant_tensor(quant_func, fp8_res_using_16bit, bits=bits,
                                                                      group_size=group_size, v=v, min_scale=min_scale,
                                                                      max_scale=max_scale, **kwargs)

    return qdq_int4_tensor * scale_bf16_to_fp8, scale_fp8_to_int4 * scale_bf16_to_fp8, None,
