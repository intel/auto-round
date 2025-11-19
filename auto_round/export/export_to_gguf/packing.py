# Copyright (c) 2024 Intel Corporation
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

import numpy as np
import torch

from auto_round.export.export_to_gguf.config import GGML_QUANT_SIZES, K_SCALE_SIZE, QK_K
from auto_round.utils import clear_memory, get_reciprocal

GGML_QUANT_TYPE = {}


def register_qtype(name):
    def register(cls):
        GGML_QUANT_TYPE[name] = cls
        return cls

    return register


def ggml_quant(
    data,
    ggml_type,
    scale=None,
    zp=None,
    wmin=None,
    d_scale=None,
    d_wmin=None,
    imatrix=None,
    device="cuda",
    original=False,
):
    block_size, type_size = GGML_QUANT_SIZES[ggml_type]

    data = data.to(torch.float32).to(device)
    scale = scale.to(device) if scale is not None else scale
    zp = zp.to(device) if zp is not None and isinstance(zp, torch.Tensor) else zp
    wmin = wmin.to(device) if wmin is not None else wmin
    d_scale = d_scale.to(device) if d_scale is not None else d_scale
    d_wmin = d_wmin.to(device) if d_wmin is not None else d_wmin

    shape = data.shape
    n_blocks = data.nelement() // block_size
    split_num = 1
    for dim in data.shape:
        if dim > 100_000:
            split_num = 16
            break

    blocks = data.reshape((n_blocks, block_size))
    quant_func = GGML_QUANT_TYPE[ggml_type]
    try:
        new_data = quant_func(
            blocks,
            scale,
            zp=zp,
            wmin=wmin,
            d_scale=d_scale,
            d_wmin=d_wmin,
            imatrix=imatrix,
            original=original,
            split_num=split_num,
        )
    except torch.OutOfMemoryError:
        orig_device = blocks.device
        device = "cpu"
        blocks = blocks.to(device)
        scale = scale.to(device) if scale is not None else scale
        zp = zp.to(device) if zp is not None and isinstance(zp, torch.Tensor) else zp
        wmin = wmin.to(device) if wmin is not None else wmin
        d_scale = d_scale.to(device) if d_scale is not None else d_scale
        d_wmin = d_wmin.to(device) if d_wmin is not None else d_wmin
        imatrix = imatrix.to(device) if imatrix is not None else imatrix
        clear_memory(device_list=orig_device)
        new_data = quant_func(
            blocks,
            scale,
            zp=zp,
            wmin=wmin,
            d_scale=d_scale,
            d_wmin=d_wmin,
            imatrix=imatrix,
            original=original,
            split_num=split_num,
        )

    assert new_data.shape[-1] == type_size
    new_data = new_data.reshape(*shape[:-1], shape[-1] // block_size * type_size)
    new_data = new_data.reshape(*shape[:-1], -1)
    return new_data


def torch_roundf(n):
    a = torch.abs(n)
    floored = torch.floor(a)
    b = floored + torch.floor((a - floored).mul_(2))
    return torch.sign(n) * b


def make_qx_quants_chunk(data, bits, rmse_type=0, qw=None, split_num=1):
    """
    Extreme VRAM-optimized version of quantization.

    - Processes data in chunks along the batch dimension (dim=0) to reduce peak memory usage.
    - Uses inplace operations to avoid unnecessary tensor copies.
    - Reuses buffers for temporary calculations wherever possible.
    """
    nmax = 2 ** (bits - 1)
    scales_list = []
    L_list = []
    chunk_size = (data.shape[0] + split_num - 1) // split_num
    for start in range(0, data.shape[0], chunk_size):
        end = min(start + chunk_size, data.shape[0])
        chunk = data[start:end]  # Slice a batch chunk to reduce memory footprint

        # Compute absolute values inplace to avoid extra tensor allocation
        chunk_abs = chunk.abs()
        imax = chunk_abs.argmax(dim=-1, keepdim=True)
        group_max = torch.take_along_dim(chunk, imax, dim=-1)

        # Compute scale factors (inverse max) without extra tensor

        iscales = -nmax * get_reciprocal(group_max)

        # L buffer stores quantized values, modified inplace to save memory
        L = (chunk * iscales).round_().clamp_(-nmax, nmax - 1)

        # Simple case: rmse_type == 0
        if rmse_type == 0:
            L.add_(nmax)  # Shift to unsigned representation inplace
            scales = (1 / iscales).reshape(iscales.shape[:2])
            scales_list.append(scales)
            L_list.append(L.to(torch.uint8))
            continue

        return_early = False
        if rmse_type < 0:
            rmse_type = -rmse_type
            return_early = True

        # Compute weighting tensor w based on rmse_type
        if qw is not None:
            w = qw
        elif rmse_type == 1:
            w = chunk * chunk
        elif rmse_type == 2:
            w = torch.ones_like(chunk)
        elif rmse_type == 3:
            w = chunk.abs()
        else:
            w = chunk.abs().sqrt()

        # Compute sumlx and suml2 using the pre-allocated L buffer
        sumlx = (w * chunk * L).sum(dim=-1)
        suml2 = (w * L * L).sum(dim=-1)
        scales = sumlx / suml2

        if return_early:
            iscales_inv = (1 / iscales).reshape(iscales.shape[:2])
            # Mix the current scale with inverse scale if suml2 > 0
            scales = torch.where(suml2 > 0, 0.5 * (scales + iscales_inv), iscales_inv)
            L.add_(nmax)
            scales_list.append(scales)
            L_list.append(L.to(torch.uint8))
            continue

        # Iteratively refine scales and quantized values
        best = scales * sumlx
        for _is in range(-9, 10):
            if _is == 0:
                continue
            iscales_tmp = -(nmax + -0.1 * _is) / group_max
            # Use a temporary L buffer to avoid creating new large tensor
            L_tmp = (chunk * iscales_tmp).round_().clamp_(-nmax, nmax - 1)
            sumlx_tmp = (w * chunk * L_tmp).sum(dim=-1)
            suml2_tmp = (w * L_tmp * L_tmp).sum(dim=-1)
            # Determine which elements should be replaced
            replace_id = (suml2_tmp > 0) & (sumlx_tmp * sumlx_tmp > best * suml2_tmp)
            # Inplace update of L and scales
            L[replace_id] = L_tmp[replace_id]
            scales[replace_id] = sumlx_tmp[replace_id] / suml2_tmp[replace_id]
            best[replace_id] = scales[replace_id] * sumlx_tmp[replace_id]

        L.add_(nmax)  # Final shift to unsigned
        scales_list.append(scales)
        L_list.append(L.to(torch.uint8))

    # Concatenate all chunks along batch dimension
    scales = torch.cat(scales_list, dim=0)
    L = torch.cat(L_list, dim=0)
    return scales, L


def make_qx_quants(data, bits, rmse_type=0, qw=None):
    """
    adapted from llmacpp
    """
    nmax = pow(2, bits - 1)
    imax = abs(data).argmax(axis=-1, keepdims=True)
    group_max = torch.take_along_dim(data, imax, dim=-1)
    iscales = -nmax * get_reciprocal(group_max)

    if rmse_type == 0:
        L = (torch.round(iscales * data).clip(-nmax, nmax - 1) + nmax).to(torch.uint8)
        scales = get_reciprocal(iscales).reshape(iscales.shape[:2])
        return scales, L

    return_early = False
    if rmse_type < 0:
        rmse_type = -rmse_type
        return_early = True

    L = torch.round(iscales * data).clip(-nmax, nmax - 1)
    if qw is not None:
        w = qw
    elif rmse_type == 1:
        w = torch.pow(data, 2)
    elif rmse_type == 2:
        w = 1
    elif rmse_type == 3:
        w = torch.abs(data)
    else:
        w = torch.sqrt(torch.abs(data))
    sumlx = torch.sum(w * data * L, dim=-1)
    suml2 = torch.sum(w * L * L, dim=-1)
    scales = sumlx * get_reciprocal(suml2)
    if return_early:
        iscales_inv = get_reciprocal(iscales).reshape(iscales.shape[:2])
        scales = torch.where(suml2 > 0, 0.5 * (scales + iscales_inv), iscales_inv)
        L = (L + nmax).to(torch.uint8)
        return scales, L
    best = scales * sumlx
    for _is in range(-9, 10):
        if _is == 0:
            continue
        iscales = -(nmax + -0.1 * _is) * get_reciprocal(group_max)
        tmp_L = torch.round(iscales * data).clip(-nmax, nmax - 1)
        sumlx = torch.sum(w * data * L, dim=-1)
        suml2 = torch.sum(w * L * L, dim=-1)

        replace_id = (suml2 > 0) & (sumlx * sumlx > best * suml2)
        L[replace_id] = tmp_L[replace_id]
        scales[replace_id] = sumlx[replace_id] / suml2[replace_id]
        best[replace_id] = scales[replace_id] * sumlx[replace_id]

    L = (L + nmax).to(torch.uint8)
    return scales, L


def make_q3_quants(data, bits, do_rmse=False):
    # Maximum absolute integer value for symmetric quantization
    nmax = 1 << (bits - 1)  # equivalent to pow(2, bits-1)

    # Find per-group max indices along last dim
    imax = abs(data).argmax(axis=-1, keepdims=True)

    # Gather group-wise maximum values
    group_max = torch.take_along_dim(data, imax, dim=-1)

    # Compute inverse scale in-place (multiplying by -nmax)
    iscale = -nmax * get_reciprocal(group_max)

    if do_rmse:
        # Initial quantization L (in-place round and clamp)
        L = torch.empty_like(data)
        torch.round(iscale * data, out=L)
        L.clamp_(-nmax, nmax - 1)

        # Weight for RMSE = x^2 (in-place)
        w = data.clone().pow_(2)

        # Precompute sums
        sumlx = torch.sum(w * data * L, dim=-1)
        suml2 = torch.sum(w * L * L, dim=-1)

        # Iterative RMSE refinement
        for _ in range(5):
            for i in range(sumlx.shape[-1]):
                # Extract current slice
                w_tmp = w[:, :, i]
                data_tmp = data[:, :, i]
                L_tmp = L[:, :, i]

                # Exclude current slice from sums
                slx = sumlx - w_tmp * data_tmp * L_tmp
                replace_idx = slx > 0
                sl2 = suml2 - w_tmp * L_tmp * L_tmp

                # Compute new L candidate (in-place round and clamp)
                new_L = torch.empty_like(L_tmp)
                torch.round(data_tmp * sl2 / slx, out=new_L)
                new_L.clamp_(-nmax, nmax - 1)

                # Identify positions to update
                tmp_replace_idx = replace_idx & (new_L != L_tmp)

                # Update sums where L changes
                slx[tmp_replace_idx] += w_tmp[tmp_replace_idx] * data_tmp[tmp_replace_idx] * new_L[tmp_replace_idx]
                sl2[tmp_replace_idx] += w_tmp[tmp_replace_idx] * new_L[tmp_replace_idx] * new_L[tmp_replace_idx]

                # Further check condition for improvement
                replace_idx &= (sl2 > 0) & (slx * slx * suml2 > sumlx * sumlx * sl2)

                # Update L in-place
                L_tmp[replace_idx] = new_L[replace_idx]

                # Update global sums
                sumlx = slx
                suml2 = sl2

        # Compute final scale and return quantized L
        return sumlx * get_reciprocal(suml2), L.to(torch.uint8)

    # Fast path: quantize without RMSE (in-place round, clamp, shift)
    L = torch.empty_like(data)
    torch.round(iscale * data, out=L)
    L.clamp_(-nmax, nmax - 1)
    L.add_(nmax)

    # Compute scales (reciprocal of iscale)
    scales = get_reciprocal(iscale).reshape(iscale.shape[:2])

    return scales, L.to(torch.uint8)


def make_qkx2_quants(data, bits, weights=None, rmin=-1.0, rdelta=0.1, nstep=20, use_mad=False):
    nmax = pow(2, bits) - 1
    # data shape (nb, 8, 32) for Q4_K, (nb, 16, 16) for Q2_K
    if len(data.shape) == 2:
        if bits in [4, 5]:
            data_shape = (-1, 8, 32)
        elif bits in [2]:
            data_shape = (-1, 16, 16)
        else:
            raise NotImplementedError(f"bits = {bits} is not supported")
        data = data.reshape(data_shape)
    if weights is None:
        sum_x2 = torch.sum(torch.pow(data, 2), dim=-1, keepdim=True)
        if bits == 2:
            av_x = 0
        else:
            av_x = torch.sqrt(sum_x2 / data.shape[-1])
        weights = torch.abs(data) + av_x

    group_min = torch.min(data, dim=-1, keepdim=True)[0]
    group_max = torch.max(data, dim=-1, keepdim=True)[0]

    the_mins = -group_min

    sum_w = torch.sum(weights, dim=-1, keepdim=True)
    sum_x = torch.sum(weights * data, dim=-1, keepdim=True)

    group_min[group_min > 0] = 0

    scale = (group_max - group_min) / nmax
    iscale = get_reciprocal(scale)

    l_values = torch.round(iscale * (data - group_min))
    L = torch.clip(l_values, 0, nmax).to(torch.uint8)

    diffs = scale * L + group_min - data
    diffs = torch.abs(diffs) if use_mad else torch.pow(diffs, 2)
    best_mad = torch.sum(weights * diffs, dim=-1, keepdim=True)

    if nstep < 1:
        return scale.reshape(scale.shape[:2]), L, the_mins.reshape(the_mins.shape[:2])

    for step in range(nstep):
        new_scale = (group_max - group_min) / (rmin + rdelta * step + nmax)
        new_iscale = get_reciprocal(new_scale)

        l_values = torch.round(new_iscale * (data - group_min))
        Laux = torch.clip(l_values, 0, nmax).to(torch.uint8)

        sum_l = torch.sum(weights * Laux, dim=-1, keepdim=True)
        sum_l2 = torch.sum(weights * torch.pow(Laux, 2), dim=-1, keepdim=True)
        sum_xl = torch.sum(weights * Laux * data, dim=-1, keepdim=True)

        D = sum_w * sum_l2 - sum_l * sum_l
        replace_idx = D > 0

        this_scale = (sum_w * sum_xl - sum_x * sum_l) / D
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D
        this_min[this_min > 0] = 0
        this_scale[this_min > 0] = (sum_xl / sum_l2)[this_min > 0]

        diffs = this_scale * Laux + this_min - data
        diffs = torch.abs(diffs) if use_mad else torch.pow(diffs, 2)
        mad = torch.sum(weights * diffs, dim=-1, keepdim=True)

        replace_idx &= mad < best_mad
        best_mad[replace_idx] = mad[replace_idx]
        L[replace_idx.reshape(replace_idx.shape[:2])] = Laux[replace_idx.reshape(replace_idx.shape[:2])]
        scale[replace_idx] = this_scale[replace_idx]
        group_min[replace_idx] = this_min[replace_idx]

    the_mins = -group_min
    return scale.reshape(scale.shape[:2]), L, the_mins.reshape(the_mins.shape[:2])


def make_qp_quants(nmax, data, quant_weights):
    group_max = torch.max(data, dim=-1, keepdim=True)[0]
    scale = group_max / nmax
    iscale = get_reciprocal(scale)

    L = torch.round(iscale * data)
    diffs = data - scale * L
    best_mse = torch.sum(quant_weights * diffs * diffs, dim=-1)

    for _is in range(-4, 5):
        if _is == 0:
            continue
        scale_is = group_max / (0.1 * _is + nmax)
        iscale_is = get_reciprocal(scale_is)

        tmp_L = torch.round(iscale_is * data).clip(max=nmax)
        diffs = data - scale_is * tmp_L
        mse = torch.sum(quant_weights * diffs * diffs, dim=-1)

        replace_idx = mse < best_mse
        best_mse[replace_idx] = mse[replace_idx]
        iscale[replace_idx] = iscale_is[replace_idx]

    L = torch.round(iscale * data).clip(max=nmax)
    sumlx = torch.sum(quant_weights * data * L, dim=-1)
    suml2 = torch.sum(quant_weights * L * L, dim=-1)

    for _ in range(5):
        n_changed = 0
        for i in range(data.shape[-1]):
            slx = sumlx - quant_weights[:, i] * data[:, i] * L[:, i]
            sl2 = suml2 - quant_weights[:, i] * L[:, i] * L[:, i]
            replace_idx = (slx > 0) & (sl2 > 0)
            new_L = torch.round(data[:, i] * sl2 / slx).clip(max=nmax)
            replace_idx &= new_L != L[:, i]
            slx[replace_idx] += quant_weights[:, i][replace_idx] * data[:, i][replace_idx] * new_L[replace_idx]
            sl2[replace_idx] += quant_weights[:, i][replace_idx] * new_L[replace_idx] * new_L[replace_idx]

            replace_idx &= slx * slx * suml2 > sumlx * sumlx * sl2
            L[:, i][replace_idx] = new_L[replace_idx]
            sumlx[replace_idx] = slx[replace_idx]
            suml2[replace_idx] = sl2[replace_idx]
            n_changed = replace_idx.sum()
        if n_changed == 0:
            break

    return sumlx / suml2, L


@register_qtype("bf16")
def bf16_quant_block(blocks, scale=None, zp=None, **kwargs):
    n = blocks.view(torch.uint32)
    # force nan to quiet
    n = torch.where((n & 0x7FFFFFFF) > 0x7F800000, (n & 0xFFFF0000) | (64 << 16), n)
    # round to nearest even
    n = n.to(torch.uint64) + (0x7FFF + ((n >> 16) & 1)) >> 16
    return n.cpu().numpy().astype(np.uint16).view(np.uint8)


@register_qtype("q4_0")
def q4_0_quant_block(blocks, scale=None, zp=None, **kwargs):
    if scale is not None:
        d = scale.reshape((-1, 1))
    else:
        imax = abs(blocks).argmax(axis=-1, keepdims=True)
        max = torch.take_along_dim(blocks, imax, dim=-1)
        d = max / -8
    id = get_reciprocal(d)
    n_blocks = blocks.shape[0]
    qs = torch.trunc(blocks.to(torch.float64).mul_(id.to(torch.float64)).add_(8.5)).clamp_(0, 15).to(torch.uint8)

    block_size = GGML_QUANT_SIZES["q4_0"][0]
    qs = qs.reshape((n_blocks, 2, block_size // 2)).cpu().numpy()
    qs = qs[..., 0, :] | (qs[..., 1, :] << 4)

    d = d.cpu().numpy().astype(np.float16).view(np.uint8)

    return np.concatenate([d, qs], axis=-1)


@register_qtype("q4_1")
def q4_1_quant_block(blocks, scale=None, zp=None, **kwargs):
    if scale is not None:
        d = scale.reshape((-1, 1))
        min = zp.reshape((-1, 1)) * d * -1
    else:
        max = blocks.max(axis=-1, keepdims=True)[0]
        min = blocks.min(axis=-1, keepdims=True)[0]
        d = (max - min) / 15
    id = get_reciprocal(d)
    n_blocks = blocks.shape[0]

    qs = torch.trunc(blocks.sub_(min).mul_(id).add_(0.5)).clamp_(0, 15).to(torch.uint8)

    block_size = GGML_QUANT_SIZES["q4_1"][0]
    qs = qs.reshape((n_blocks, 2, block_size // 2)).cpu().numpy()
    qs = qs[..., 0, :] | (qs[..., 1, :] << np.uint8(4))

    d = d.cpu().numpy().astype(np.float16).view(np.uint8)
    m = min.cpu().numpy().astype(np.float16).view(np.uint8)
    return np.concatenate([d, m, qs], axis=-1)


@register_qtype("q5_0")
def q5_0_quant_block(blocks: np.array, scale=None, zp=None, **kwargs):
    if scale is not None:
        d = scale.reshape((-1, 1))
    else:
        imax = abs(blocks).argmax(axis=-1, keepdims=True)
        max = torch.take_along_dim(blocks, imax, dim=-1)
        d = max / -16

    id = get_reciprocal(d)
    n_blocks = blocks.shape[0]
    block_size = GGML_QUANT_SIZES["q5_0"][0]

    # FIXME: Q5_0's reference rounding is cursed and depends on FMA
    q = (
        torch.trunc(blocks.to(torch.float64).mul_(id.to(torch.float64)).add_(16.5))
        .clamp_(0, 31)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )

    qs = q.reshape((n_blocks, 2, block_size // 2))
    qs = (qs[..., 0, :] & np.uint8(0x0F)) | (qs[..., 1, :] << np.uint8(4))

    qh = np.packbits(q.reshape((n_blocks, 1, 32)) >> np.uint8(4), axis=-1, bitorder="little").reshape(n_blocks, 4)

    d = d.cpu().numpy().astype(np.float16).view(np.uint8)

    return np.concatenate([d, qh, qs], axis=-1)


@register_qtype("q5_1")
def q5_1_quant_block(blocks: np.array, scale=None, zp=None, **kwargs):
    if scale is not None:
        d = scale.reshape((-1, 1))
        min = zp.reshape((-1, 1)) * d * -1
    else:
        max = blocks.max(axis=-1, keepdims=True)[0]
        min = blocks.min(axis=-1, keepdims=True)[0]
        d = (max - min) / 31

    n_blocks = blocks.shape[0]
    block_size = GGML_QUANT_SIZES["q5_1"][0]

    id = get_reciprocal(d)
    q = torch.trunc(blocks.sub_(min).mul_(id).add_(0.5)).clamp_(0, 31).to(torch.uint8).cpu().numpy()

    qs = q.reshape((n_blocks, 2, block_size // 2))
    qs = (qs[..., 0, :] & np.uint8(0x0F)) | (qs[..., 1, :] << np.uint8(4))

    qh = np.packbits(q.reshape((n_blocks, 1, 32)) >> np.uint8(4), axis=-1, bitorder="little").reshape(n_blocks, 4)

    d = d.cpu().numpy().astype(np.float16).view(np.uint8)
    m = min.cpu().numpy().astype(np.float16).view(np.uint8)

    return np.concatenate([d, m, qh, qs], axis=-1)


@register_qtype("q8_0")
def q8_0_quant_block(blocks, scale=None, zp=None, **kwargs) -> np.ndarray:
    if scale is not None:
        d = scale.reshape((-1, 1))
    else:
        d = torch.abs(blocks).max(dim=1, keepdim=True)[0] / 127
    id = get_reciprocal(d)
    blocks = blocks.mul_(id)
    qs = torch_roundf(blocks).clamp_(-128, 127)

    # (n_blocks, 2)
    d = d.cpu().numpy().astype(np.float16).view(np.uint8)
    # (n_blocks, block_size)
    qs = qs.cpu().numpy().astype(np.int8).view(np.uint8)

    return np.concatenate([d, qs], axis=1)


@register_qtype("q2_k")
def q2_k_quant_block(
    blocks, scale=None, wmin=None, d_scale=None, d_wmin=None, imatrix=None, original=False, split_num=None, **kwargs
):
    nb = blocks.shape[0]
    device = blocks.device
    blocks = blocks.reshape((nb, QK_K // 16, 16))  # (nb, 16, 16)

    if scale is not None:
        scales = scale.reshape((-1, QK_K // 16))
        mins = wmin.reshape((-1, QK_K // 16))
        output_d = d_scale.reshape(-1, 1).to(torch.float32)
        output_dmin = d_wmin.reshape(-1, 1).to(torch.float32)
        output_scale = (scales * get_reciprocal(output_d)).round_().clamp_(0, 15).to(torch.uint8)
        output_scale |= (mins * get_reciprocal(output_dmin)).round_().clamp_(0, 15).to(torch.uint8) << 4
        all_L = blocks.add_(mins.unsqueeze(-1)).div_(scales.unsqueeze(-1)).round_().clamp_(0, 3).to(torch.uint8)
    elif original:
        scales, all_L, mins = make_qkx2_quants(blocks, bits=2, rmin=-0.5, rdelta=0.1, nstep=15, use_mad=True)
        max_scales = torch.max(scales, dim=-1, keepdim=True)[0]
        max_mins = torch.max(mins, dim=-1, keepdim=True)[0]
        output_d = (max_scales / 15).clamp(min=0)
        output_dmin = (max_mins / 15).clamp(min=0)
        id_scales = (15 * get_reciprocal(max_scales)).clamp(min=0)
        id_mins = (15 * get_reciprocal(max_mins)).clamp(min=0)

        replace_ids = (max_scales > 0).squeeze()
        output_scale = torch.zeros_like(scales).to(torch.uint8)
        output_scale[replace_ids] = (
            torch.round(id_scales[replace_ids] * scales[replace_ids]).clip(0, 15).to(torch.uint8)
        )

        replace_ids = (max_mins > 0).squeeze()
        output_scale[replace_ids] |= (
            torch.round(id_mins[replace_ids] * mins[replace_ids]).clip(0, 15).to(torch.uint8) << 4
        )

        d_tmp = output_d * (output_scale & 0xF)
        dm_tmp = output_dmin * (output_scale >> 4)

        replace_ids = d_tmp != 0
        all_L[replace_ids] = (
            blocks[replace_ids]
            .add_(dm_tmp[replace_ids].unsqueeze(-1))
            .div_(d_tmp[replace_ids].unsqueeze(-1))
            .round_()
            .clamp_(0, 3)
            .to(torch.uint8)
        )

    else:
        from auto_round.data_type.gguf import quant_tensor_gguf_asym_dq

        blocks.reshape(blocks.shape[0], -1)
        blocks, scales, mins = quant_tensor_gguf_asym_dq(
            blocks, bits=2, scale_dtype=torch.float32, imatrix=imatrix, split_num=split_num
        )
        scales, d_scale = scales["scale"], scales["d_scale"]
        mins, d_wmin = mins["wmin"], mins["d_wmin"]
        if split_num is not None and split_num > 1:
            blocks = blocks.to("cpu")
            scales = scales.to("cpu")
            d_scale = d_scale.to("cpu")
            mins = mins.to("cpu")
            d_wmin = d_wmin.to("cpu")
            clear_memory(device_list=[device])

        blocks = blocks.reshape((nb, QK_K // 16, 16))
        scales = scales.reshape((-1, QK_K // 16))
        mins = mins.reshape((-1, QK_K // 16))
        output_d = d_scale.reshape(-1, 1).to(torch.float32)
        output_dmin = d_wmin.reshape(-1, 1).to(torch.float32)
        output_scale = scales.mul(get_reciprocal(output_d)).round_().clamp_(0, 15).to(torch.uint8)
        output_scale |= (mins * get_reciprocal(output_dmin)).round_().clamp_(0, 15).to(torch.uint8) << 4
        all_L = blocks.add_(mins.unsqueeze(-1)).div_(scales.unsqueeze(-1)).round_().clamp_(0, 3).to(torch.uint8)

    output_scale = output_scale.cpu().numpy()
    all_L = all_L.reshape(-1, 4, 32)
    output_qs = all_L[:, 0, :] | (all_L[:, 1, :] << 2) | (all_L[:, 2, :] << 4) | (all_L[:, 3, :] << 6)
    output_d = output_d.cpu().numpy()
    output_d = output_d.reshape(-1, 1).astype(np.float16).view(np.uint8)
    output_dmin = output_dmin.cpu().numpy()
    output_dmin = output_dmin.reshape(-1, 1).astype(np.float16).view(np.uint8)
    output_qs = output_qs.cpu().numpy().astype(np.uint8).reshape((nb, QK_K // 4))

    # [scale, qs, d, dmin]
    return np.concatenate([output_scale, output_qs, output_d, output_dmin], axis=-1)


@register_qtype("q3_k")
def q3_k_quant_block(
    blocks: np.array, scale=None, d_scale=None, original=False, imatrix=None, split_num=None, **kwargs
):
    nb = blocks.shape[0]
    blocks = blocks.reshape(nb, QK_K // 16, 16)

    if scale is not None:
        qdq_scale = scale.reshape(-1, QK_K // 16).to(torch.float32)
        dq_scale = d_scale.reshape(-1, 1).to(torch.float32)
        all_L = blocks.mul_(get_reciprocal(qdq_scale.unsqueeze(-1))).round_().clamp_(-4, 3).add_(4).to(torch.uint8)
        q_scales_offset = (qdq_scale * get_reciprocal(dq_scale)).round_().clamp_(-32, 31).add_(32)
    elif original:
        scales, _ = make_q3_quants(blocks, bits=3, do_rmse=True)
        scales_abs_max = abs(scales).argmax(dim=-1, keepdim=True)
        max_scales_mag = torch.take_along_dim(scales, scales_abs_max, dim=-1)
        inverse_dq_scale = -32 * get_reciprocal(max_scales_mag)
        dq_scale = get_reciprocal(inverse_dq_scale)
        qscale = (inverse_dq_scale * scales).round_().clamp_(-32, 31)
        qdq_scale = dq_scale.to(torch.float32) * qscale
        reverse_qdq_scale = get_reciprocal(qdq_scale)
        all_L = blocks.mul_(reverse_qdq_scale.unsqueeze(-1)).round_().clamp_(-4, 3).add_(4).to(torch.uint8)
        q_scales_offset = (qdq_scale * inverse_dq_scale).round_().clamp_(-32, 31).add_(32)
    else:
        from auto_round.data_type.gguf import quant_tensor_gguf_sym_dq

        blocks = blocks.reshape(blocks.shape[0], -1)
        blocks, scales, _ = quant_tensor_gguf_sym_dq(
            blocks, bits=3, scale_dtype=torch.float32, imatrix=imatrix, split_num=split_num
        )
        scales, d_scale = scales["scale"], scales["d_scale"]
        if split_num is not None and split_num > 1:
            blocks = blocks.to("cpu")
            scales = scales.to("cpu")
            d_scale = d_scale.to("cpu")

        blocks = blocks.reshape((nb, QK_K // 16, 16))
        qdq_scale = scales.reshape((-1, QK_K // 16)).to(torch.float32)
        dq_scale = d_scale.reshape(-1, 1).to(torch.float32)
        all_L = blocks.mul_(get_reciprocal(qdq_scale.unsqueeze(-1))).round_().clamp_(-4, 3).add_(4).to(torch.uint8)

        q_scales_offset = (qdq_scale * get_reciprocal(dq_scale)).round_().clamp_(-32, 31).add_(32)

    output_scale = np.empty((nb, K_SCALE_SIZE), dtype=np.uint8)
    q_scales_offset = q_scales_offset.cpu().numpy().astype(np.uint8)
    output_scale[:, :8] = (q_scales_offset[:, :8] & 0xF) | ((q_scales_offset[:, 8:] & 0xF) << 4)
    hmask = q_scales_offset >> 4
    output_scale[:, 8:] = hmask[:, :4] | hmask[:, 4:8] << 2 | hmask[:, 8:12] << 4 | hmask[:, 12:] << 6

    output_hmask = all_L.reshape(nb, 8, 32) >> 2 << torch.arange(8, device=all_L.device).reshape(1, 8, 1)
    output_hmask = np.bitwise_or.reduce(output_hmask.cpu().numpy(), axis=1, dtype=np.uint8)  # pylint: disable=E1121
    all_L = torch.where(all_L > 3, all_L - 4, all_L)

    output_qs = all_L.reshape(nb, 2, 4, 32) << torch.tensor([0, 2, 4, 6], device=all_L.device).reshape(1, 1, 4, 1)
    output_qs = np.bitwise_or.reduce(output_qs.cpu().numpy(), axis=2, dtype=np.uint8)  # pylint: disable=E1121

    output_qs = output_qs.reshape(nb, 64).astype(np.uint8)
    dq_scale = dq_scale.cpu().numpy().reshape(-1, 1).astype(np.float16).view(np.uint8)
    # [hmask, qs, scale, d]
    return np.concatenate([output_hmask, output_qs, output_scale, dq_scale], axis=-1)


@register_qtype("q4_k")
def q4_k_quant_block(
    blocks, scale=None, wmin=None, d_scale=None, d_wmin=None, imatrix=None, original=False, split_num=1, **kwargs
):
    nb = blocks.shape[0]
    blocks = blocks.reshape((nb, QK_K // 32, 32))

    if scale is not None:
        scales = scale.reshape(-1, QK_K // 32)
        mins = wmin.reshape(-1, QK_K // 32)
        output_d = d_scale.reshape(-1, 1).to(torch.float32)
        output_dmin = d_wmin.reshape(-1, 1).to(torch.float32)
        q_scales = (scales * get_reciprocal(output_d)).round_().clamp_(0, 63).to(torch.uint8)
        q_mins = (mins * get_reciprocal(output_dmin)).round_().clamp_(0, 63).to(torch.uint8)
        all_L = (
            blocks.add_(mins.unsqueeze(-1))
            .mul_(get_reciprocal(scales.unsqueeze(-1)))
            .round_()
            .clamp_(0, 15)
            .to(torch.uint8)
        )

    elif original:
        scales, all_L, mins = make_qkx2_quants(blocks, bits=4, rmin=-1, rdelta=0.1, nstep=20, use_mad=False)
        max_scales = torch.max(scales, dim=-1, keepdim=True)[0]
        max_mins = torch.max(mins, dim=-1, keepdim=True)[0]
        id_scales = (63 * get_reciprocal(max_scales)).clamp(min=0)
        id_mins = (63 * get_reciprocal(max_mins)).clamp(min=0)
        output_d = max_scales / 63
        output_dmin = max_mins / 63
        q_scales = (id_scales * scales).round_().clamp_(0, 63).to(torch.uint8)
        q_mins = (id_mins * mins).round_().clamp_(0, 63).to(torch.uint8)

        d_tmp = output_d * q_scales
        dm_tmp = output_dmin * q_mins
        replace_ids = d_tmp != 0
        all_L[replace_ids] = (
            blocks[replace_ids]
            .add_(dm_tmp[replace_ids].unsqueeze(-1))
            .div_(d_tmp[replace_ids].unsqueeze(-1))
            .clamp_(0, 15)
            .to(torch.uint8)
        )

    else:
        from auto_round.data_type.gguf import quant_tensor_gguf_asym_dq

        blocks.reshape(blocks.shape[0], -1)
        blocks, scales, mins = quant_tensor_gguf_asym_dq(
            blocks, bits=4, scale_dtype=torch.float32, imatrix=imatrix, split_num=split_num
        )
        scales, d_scale = scales["scale"], scales["d_scale"]
        mins, d_wmin = mins["wmin"], mins["d_wmin"]
        if split_num is not None and split_num > 1:
            blocks = blocks.to("cpu")
            scales = scales.to("cpu")
            d_scale = d_scale.to("cpu")
            mins = mins.to("cpu")
            d_wmin = d_wmin.to("cpu")

        blocks = blocks.reshape((nb, QK_K // 32, 32))
        scales = scales.reshape((-1, QK_K // 32))
        mins = mins.reshape((-1, QK_K // 32))
        output_d = d_scale.reshape(-1, 1).to(torch.float32)
        output_dmin = d_wmin.reshape(-1, 1).to(torch.float32)
        q_scales = (scales * get_reciprocal(output_d)).round_().clamp_(0, 63).to(torch.uint8)
        q_mins = (mins * get_reciprocal(output_dmin)).round_().clamp_(0, 63).to(torch.uint8)
        all_L = (
            blocks.add_(mins.unsqueeze(-1))
            .mul_(get_reciprocal(scales.unsqueeze(-1)))
            .round_()
            .clamp_(0, 15)
            .to(torch.uint8)
        )
    output_scale = torch.empty((nb, K_SCALE_SIZE), dtype=torch.uint8, device=blocks.device)
    output_scale[:, :4] = q_scales[:, :4]
    output_scale[:, 4:8] = q_mins[:, :4]

    output_scale[:, 8:] = (q_scales[:, 4:] & 0xF) | ((q_mins[:, 4:] & 0xF) << 4)
    output_scale[:, :4] |= (q_scales[:, 4:] >> 4) << 6
    output_scale[:, 4:8] |= (q_mins[:, 4:] >> 4) << 6

    output_qs = all_L[:, ::2] | (all_L[:, 1::2] << 4)

    output_d = output_d.cpu().numpy()
    output_d = output_d.reshape(-1, 1).astype(np.float16).view(np.uint8)
    output_dmin = output_dmin.cpu().numpy()
    output_dmin = output_dmin.reshape(-1, 1).astype(np.float16).view(np.uint8)
    output_qs = output_qs.reshape(nb, QK_K // 2).cpu().numpy().astype(np.uint8)
    output_scale = output_scale.cpu().numpy().astype(np.uint8)

    # [d, dmin, scale, qs]
    return np.concatenate([output_d, output_dmin, output_scale, output_qs], axis=-1)


@register_qtype("q5_k")
def q5_k_quant_block(
    blocks,
    scale=None,
    zp=None,
    wmin=None,
    d_scale=None,
    d_wmin=None,
    imatrix=None,
    original=False,
    split_num=1,
    **kwargs,
):
    nb = blocks.shape[0]
    blocks = blocks.reshape((nb, QK_K // 32, 32))

    if scale is not None:
        scales = scale.reshape(-1, QK_K // 32)
        mins = wmin.reshape(-1, QK_K // 32)
        output_d = d_scale.reshape(-1, 1).to(torch.float32)
        output_dmin = d_wmin.reshape(-1, 1).to(torch.float32)
        q_scales = (scales * get_reciprocal(output_d)).round_().clamp_(0, 63).to(torch.uint8)
        q_mins = (mins * get_reciprocal(output_dmin)).round_().clamp_(0, 63).to(torch.uint8)
        all_L = (
            blocks.add_(mins.unsqueeze(-1))
            .mul_(get_reciprocal(scales.unsqueeze(-1)))
            .round_()
            .clamp_(0, 31)
            .to(torch.uint8)
        )

    elif original:
        scales, all_L, mins = make_qkx2_quants(blocks, bits=5, rmin=-0.5, rdelta=0.1, nstep=15, use_mad=False)
        max_scales = torch.max(scales, dim=-1, keepdim=True)[0]
        max_mins = torch.max(mins, dim=-1, keepdim=True)[0]
        id_scales = (63 * get_reciprocal(max_scales)).clamp(min=0)
        id_mins = (63 * get_reciprocal(max_mins)).clamp(min=0)
        output_d = max_scales / 63
        output_dmin = max_mins / 63
        q_scales = (id_scales * scales).round_().clamp_(0, 63).to(torch.uint8)
        q_mins = (id_mins * mins).round_().clamp_(0, 63).to(torch.uint8)

        d_tmp = output_d * q_scales
        dm_tmp = output_dmin * q_mins
        replace_ids = d_tmp != 0
        all_L[replace_ids] = (
            blocks[replace_ids]
            .add_(dm_tmp[replace_ids].unsqueeze(-1))
            .div_(d_tmp[replace_ids].unsqueeze(-1))
            .round_()
            .clamp_(0, 31)
            .to(torch.int8)
        )
    else:
        from auto_round.data_type.gguf import quant_tensor_gguf_asym_dq

        blocks.reshape(blocks.shape[0], -1)
        blocks, scales, mins = quant_tensor_gguf_asym_dq(
            blocks, bits=4, scale_dtype=torch.float32, imatrix=imatrix, split_num=split_num
        )
        scales, d_scale = scales["scale"], scales["d_scale"]
        mins, d_wmin = mins["wmin"], mins["d_wmin"]
        if split_num is not None and split_num > 1:
            blocks = blocks.to("cpu")
            scales = scales.to("cpu")
            d_scale = d_scale.to("cpu")
            mins = mins.to("cpu")
            d_wmin = d_wmin.to("cpu")

        blocks = blocks.reshape((nb, QK_K // 32, 32))
        scales = scales.reshape((-1, QK_K // 32))
        mins = mins.reshape((-1, QK_K // 32))
        output_d = d_scale.reshape(-1, 1).to(torch.float32)
        output_dmin = d_wmin.reshape(-1, 1).to(torch.float32)
        q_scales = (scales * get_reciprocal(output_d)).round_().clamp_(0, 63).to(torch.uint8)
        q_mins = (mins * get_reciprocal(output_dmin)).round_().clamp(0, 63).to(torch.uint8)
        all_L = (
            blocks.add_(mins.unsqueeze(-1))
            .mul_(get_reciprocal(scales.unsqueeze(-1)))
            .round_()
            .clamp_(0, 31)
            .to(torch.uint8)
        )
    output_scale = torch.empty((nb, K_SCALE_SIZE), dtype=torch.uint8, device=blocks.device)

    output_scale[:, :4] = q_scales[:, :4]
    output_scale[:, 4:8] = q_mins[:, :4]

    output_scale[:, 8:] = (q_scales[:, 4:] & 0xF) | ((q_mins[:, 4:] & 0xF) << 4)
    output_scale[:, :4] |= (q_scales[:, 4:] >> 4) << 6
    output_scale[:, 4:8] |= (q_mins[:, 4:] >> 4) << 6

    output_qs = all_L[:, ::2] | (all_L[:, 1::2] << 4)
    output_qh = all_L >> 4 << torch.arange(8, device=all_L.device).reshape(1, 8, 1)
    output_qh = np.bitwise_or.reduce(output_qh.cpu().numpy(), axis=1, dtype=np.uint8).astype(
        np.uint8
    )  # pylint: disable=E1121

    output_d = output_d.cpu().numpy()
    output_d = output_d.reshape(-1, 1).astype(np.float16).view(np.uint8)
    output_dmin = output_dmin.cpu().numpy()
    output_dmin = output_dmin.reshape(-1, 1).astype(np.float16).view(np.uint8)
    output_qs = output_qs.reshape(nb, QK_K // 2).cpu().numpy().astype(np.uint8)
    output_scale = output_scale.cpu().numpy().astype(np.uint8)

    # [d, dmin, scale, qh, qs]
    return np.concatenate([output_d, output_dmin, output_scale, output_qh, output_qs], axis=-1)


@register_qtype("q6_k")
def q6_k_quant_block(
    blocks: np.array, scale=None, d_scale=None, original=False, imatrix=None, split_num=None, **kwargs
):
    nb = blocks.shape[0]
    blocks = blocks.reshape((nb, QK_K // 16, 16))
    device = blocks.device
    if scale is not None:
        scales = scale.reshape(-1, QK_K // 16)
        output_d = d_scale.reshape(-1, 1).to(torch.float32)
        rd = get_reciprocal(output_d)
        output_scale = scales.mul(rd).round_().clamp_(max=127).to(torch.int8)
        rs = get_reciprocal(scales).unsqueeze_(-1)  # inplace unsqueeze
        all_L = blocks.mul(rs).add_(32).round_().clamp_(0, 63).to(torch.uint8)
    elif original:
        scales, all_L = make_qx_quants(blocks, bits=6, rmse_type=1, qw=None)
        imax = abs(scales).argmax(dim=-1, keepdim=True)
        max_scales = torch.take_along_dim(scales, imax, dim=-1)

        iscales = -128 * get_reciprocal(max_scales)
        output_d = get_reciprocal(iscales)
        output_scale = (iscales * scales).round_().clamp_(max=127).to(torch.int8)
        d_tmp = output_d * output_scale.to(torch.float32)
        replace_ids = d_tmp != 0
        all_L[replace_ids] = (
            (blocks[replace_ids] / d_tmp[replace_ids]).reshape(-1, 1).add_(32).round_().clamp_(0, 63).to(torch.uint8)
        )
    else:
        from auto_round.data_type.gguf import quant_tensor_gguf_sym_dq

        blocks = blocks.reshape(blocks.shape[0], -1)
        blocks, scales, _ = quant_tensor_gguf_sym_dq(
            blocks, bits=6, scale_dtype=torch.float32, imatrix=imatrix, split_num=split_num
        )
        scales, d_scale = scales["scale"], scales["d_scale"]
        if split_num is not None and split_num > 1:
            blocks = blocks.to("cpu")
            scales = scales.to("cpu")
            d_scale = d_scale.to("cpu")

        blocks = blocks.reshape((nb, QK_K // 16, 16))
        scales = scales.reshape((-1, QK_K // 16))
        output_d = d_scale.reshape(-1, 1).to(torch.float32)
        output_scale = (scales * get_reciprocal(output_d)).round_().clamp_(max=127).to(torch.int8)
        all_L = blocks.mul_(get_reciprocal(scales.unsqueeze(-1))).add_(32).round_().clamp_(0, 63).to(torch.uint8)

    tmp_L = all_L.reshape(nb, 4, 64) & 0xF
    output_ql = (tmp_L[:, ::2] | (tmp_L[:, 1::2] << 4)).reshape(nb, QK_K // 2).cpu().numpy().astype(np.uint8)
    output_qh = (all_L >> 4).reshape(nb, 2, 4, 32) << torch.tensor([0, 2, 4, 6], device=all_L.device).reshape(
        1, 1, 4, 1
    )
    output_qh = (
        np.bitwise_or.reduce(output_qh.cpu().numpy(), axis=2, dtype=np.uint8)  # pylint: disable=E1121
        .reshape(nb, QK_K // 4)
        .astype(np.uint8)
    )  # pylint: disable=E1121

    output_d = output_d.cpu().numpy().reshape(-1, 1).astype(np.float16).view(np.uint8)

    # [ql, qh, scale, d]
    output_scale = output_scale.cpu().numpy().view(np.uint8)
    return np.concatenate([output_ql, output_qh, output_scale, output_d], axis=-1)
