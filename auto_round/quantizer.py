import torch
def round_ste(x: torch.Tensor):
    """Straight-Through Estimator for rounding.
    This function is adapted from omniquant.

    Args:
        x: torch.Tensor

    Returns:
        torch.Tensor
    """
    return (x.round() - x).detach() + x


def quant_weight_asym(weight, num_bits=4, v=0, min_scale=1.0, max_scale=1.0, scale_dtype=torch.float16):
    """Quantizes and dequantizes weight asymmetrically.

    Args:
        weight: Tensor containing the weight to be quantized
        num_bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for weight
        max_scale: Maximum scale coefficient for weight

    Returns:
        Quantized and dequantized weight, scale, zero-point
    """
    maxq = torch.tensor(2 ** num_bits - 1)
    if isinstance(min_scale, torch.Tensor):
        wmin_tmp = torch.clamp(weight.min(1)[0], max=0)
        wmax_tmp = torch.clamp(weight.max(1)[0], min=0)
        wmin_tmp *= min_scale
        wmax_tmp *= max_scale
        wmax = torch.maximum(wmax_tmp, wmin_tmp)
        wmin = torch.minimum(wmax_tmp, wmin_tmp)
    else:
        wmin = torch.clamp(weight.min(1)[0], max=0)
        wmax = torch.clamp(weight.max(1)[0], min=0)

    tmp = (wmin == 0) & (wmax == 0)
    wmin[tmp] = -1
    wmax[tmp] = +1
    scale = ((wmax - wmin) / maxq).to(scale_dtype)
    zp = round_ste(-wmin / scale)
    scale = scale.unsqueeze(dim=-1)
    zp = zp.unsqueeze(dim=-1)
    int_w = round_ste(weight / scale + v)
    q = torch.clamp(int_w + zp, 0, maxq)
    return scale * (q - zp), scale, zp


def quant_weight_sym(weight, num_bits=4, v=0, min_scale=1.0, max_scale=1.0, scale_dtype=torch.float16):
    """Quantizes and dequantizes weight symmetrically.

    Args:
        weight: Tensor containing the weight to be quantized
        num_bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for weight
        max_scale: Maximum scale coefficient for weight

    Returns:
        Quantized and dequantized weight, scale, zero-point
    """
    maxq = torch.tensor(2 ** num_bits - 1)
    if isinstance(min_scale, torch.Tensor):
        wmin_tmp = torch.clamp(weight.min(1)[0], max=0)
        wmax_tmp = torch.clamp(weight.max(1)[0], min=0)
        wmin_tmp *= min_scale
        wmax_tmp *= max_scale
        wmax = torch.maximum(wmax_tmp, wmin_tmp)
        wmin = torch.minimum(wmax_tmp, wmin_tmp)
    else:
        wmin = torch.clamp(weight.min(1)[0], max=0)
        wmax = torch.clamp(weight.max(1)[0], min=0)
    wmax_new = torch.max(wmin.abs(), wmax)
    tmp = wmin < 0
    wmin_new = wmin.clone()  ##must clone, otherwise inplace backward will occur
    if torch.any(tmp):
        wmin_new[tmp] = -wmax_new[tmp]

    tmp = (wmin_new == 0) & (wmax_new == 0)
    wmin_new[tmp] = -1
    wmax_new[tmp] = +1
    scale = ((wmax_new - wmin_new) / maxq).to(scale_dtype)

    scale = scale.unsqueeze(dim=-1)
    zp = torch.full_like(scale, (maxq + 1) / 2)

    int_w = round_ste(weight / scale + v)
    q = torch.clamp(int_w + zp, 0, maxq)
    return scale * (q - zp), scale, zp


def quant_weight_actor(weight, num_bits, sym, v, min_scale, max_scale, scale_dtype=torch.float16):
    """Quantizes and dequantizes weight symmetrically or asymmetrically .

    Args:
        weight: Tensor containing the weight to be quantized
        num_bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        sym: Sym or asym
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for weight
        max_scale: Maximum scale coefficient for weight

    Returns:
        Quantized and dequantized weight, scale, zero-point
    """
    assert num_bits > 0, "num_bits should be larger than 0"
    if sym:
        return quant_weight_sym(weight, num_bits, v, min_scale, max_scale, scale_dtype)
    else:
        return quant_weight_asym(weight, num_bits, v, min_scale, max_scale, scale_dtype)


def quant_weight(
        weight, num_bits=4, group_size=-1, sym=False, v=0, min_scale=1.0, max_scale=1.0, scale_dtype=torch.float16
):
    """Quantizes and dequantizes weight, handing the group size issue .

    Args:
        weight: Tensor containing the weight to be quantized
        num_bits: Number of bits for quantization (e.g., 2, 3, 4, 8)
        group_size: The number of elements shares scale and zero point
        sym: Sym or asym
        v: Rounding value perturbation
        min_scale: Minimum scale coefficient for weight
        max_scale: Maximum scale coefficient for weight

    Returns:
        Quantized and dequantized weight, scale, zero-point
    """
    if group_size == -1 or weight.shape[1] < group_size:
        return quant_weight_actor(
            weight, num_bits, sym=sym, v=v, min_scale=min_scale, max_scale=max_scale, scale_dtype=scale_dtype
        )
    orig_shape = weight.shape
    if weight.shape[1] % group_size == 0:
        weight = weight.reshape(-1, group_size)
        if isinstance(v, torch.Tensor):
            v = v.reshape(-1, group_size)

        weight, scale, zp = quant_weight_actor(
            weight, num_bits, sym=sym, v=v, min_scale=min_scale, max_scale=max_scale, scale_dtype=scale_dtype
        )
        weight = weight.reshape(orig_shape)
        scale = scale.reshape(weight.shape[0], -1)  ##only for linear, conv1d
        if zp is not None:
            zp = zp.reshape(weight.shape[0], -1)
        return weight, scale, zp

    else:
        pad_len = (weight.shape[1] + group_size - 1) // group_size * group_size - weight.shape[1]
        weight_new = torch.nn.functional.pad(weight, (0, pad_len))
        v = torch.nn.functional.pad(v, (0, pad_len))
        weight_new = weight_new.reshape(-1, group_size)
        if isinstance(v, torch.Tensor):
            v = v.reshape(-1, group_size)
        weight_new, scale, zp = quant_weight_actor(
            weight_new, num_bits, sym=sym, v=v, min_scale=min_scale, max_scale=max_scale, scale_dtype=scale_dtype
        )
        weight_new = weight_new.reshape(orig_shape[0], -1)

        weight_new = weight_new[:, :-pad_len]
        scale = scale.reshape(weight_new.shape[0], -1)  ##only for linear, conv1d
        if zp is not None:
            zp = zp.reshape(weight_new.shape[0], -1)
        return weight_new, scale, zp
