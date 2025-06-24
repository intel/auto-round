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

import torch
import numpy as np

from auto_round.export.export_to_gguf.config import QK_K, K_SCALE_SIZE, GGML_QUANT_SIZES

GGML_QUANT_TYPE = {}


def register_qtype(name):
    def register(cls):
        GGML_QUANT_TYPE[name] = cls
        return cls

    return register


def ggml_quant_gpu(data, ggml_type, scale=None, zp=None, wmin_m=None, d_scale=None, d_wmin_m=None, imatrix=None,
                   device="cuda"):
    block_size, type_size = GGML_QUANT_SIZES[ggml_type]

    data = data.to(torch.float32).to(device)
    scale = scale.to(device) if scale is not None else scale
    zp = zp.to(device) if zp is not None else zp
    wmin_m = wmin_m.to(device) if wmin_m is not None else wmin_m
    d_scale = d_scale.to(device) if d_scale is not None else d_scale
    d_wmin_m = d_wmin_m.to(device) if d_wmin_m is not None else d_wmin_m

    shape = data.shape
    n_blocks = data.nelement() // block_size
    blocks = data.reshape((n_blocks, block_size))
    quant_func = GGML_QUANT_TYPE[ggml_type]
    new_data = quant_func(blocks, scale, zp, wmin_m=wmin_m, d_scale=d_scale, d_wmin_m=d_wmin_m, imatrix=imatrix)

    assert new_data.shape[-1] == type_size
    new_data = new_data.reshape(*shape[:-1], shape[-1] // block_size * type_size)
    new_data = new_data.reshape(*shape[:-1], -1)
    return new_data


def torch_roundf(n):
    a = torch.abs(n)
    floored = torch.floor(a)
    b = floored + torch.floor(2 * (a - floored))
    return torch.sign(n) * b


def make_qx_quants(data, bits, rmse_type=0, qw=None):
    nmax = pow(2, bits - 1)
    imax = abs(data).argmax(axis=-1, keepdims=True)
    group_max = torch.take_along_dim(data, imax, dim=-1)
    iscales = torch.where(group_max != 0, -nmax / group_max, 0)

    if rmse_type == 0:
        L = (torch.round(iscales * data).clip(-nmax, nmax - 1) + nmax).to(torch.uint8)
        scales = torch.where(iscales != 0, 1 / iscales, 0).reshape(iscales.shape[:2])
        return scales, L

    return_early = False
    if rmse_type < 0:
        rmse_type = -rmse_type
        return_early = True

    L = torch.round(iscales * data).clip(-nmax, nmax - 1)
    if qw is not None:
        w = qw
    elif rmse_type == 1:
        w = data ** 2
    elif rmse_type == 2:
        w = 1
    elif rmse_type == 3:
        w = torch.abs(data)
    else:
        w = torch.sqrt(torch.abs(data))
    sumlx = torch.sum(w * data * L, dim=-1)
    suml2 = torch.sum(w * L * L, dim=-1)
    scales = torch.where(suml2 != 0, sumlx / suml2, 0)
    if return_early:
        iscales_inv = torch.where(iscales != 0, 1 / iscales, 0).reshape(iscales.shape[:2])
        scales = torch.where(suml2 > 0, 0.5 * (scales + iscales_inv), iscales_inv)
        L = (L + nmax).to(torch.uint8)
        return scales, L
    best = scales * sumlx
    for _is in range(-9, 10):
        if _is == 0:
            continue
        iscales = torch.where(group_max != 0, -(nmax + -0.1 * _is) / nmax, 0)
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
    nmax = pow(2, bits - 1)
    imax = abs(data).argmax(axis=-1, keepdims=True)
    group_max = torch.take_along_dim(data, imax, dim=-1)
    iscale = torch.where(group_max != 0, -nmax / group_max, 0)
    if do_rmse:
        L = torch.round(iscale * data).clip(-nmax, nmax - 1)
        w = torch.pow(data, 2)
        sumlx = torch.sum(w * data * L, dim=-1)[0]
        suml2 = torch.sum(w * L * L, dim=-1)[0]

        for itry in range(5):
            # w = np.power(data, 2)
            if len(sumlx.shape) == 2:
                sumlx = sumlx.unsqueeze(-1)
            if len(suml2.shape) == 2:
                suml2 = suml2.unsqueeze(-1)
            slx = sumlx - w * data * L
            replace_idx = slx > 0
            sl2 = suml2 - w * L * L
            new_L = torch.round(data * sl2 / slx).clip(-nmax, nmax - 1)
            tmp_replace_idx = replace_idx & (new_L != L)
            slx[tmp_replace_idx] += w[tmp_replace_idx] * data[tmp_replace_idx] * new_L[tmp_replace_idx]
            sl2[tmp_replace_idx] += w[tmp_replace_idx] * new_L[tmp_replace_idx] * new_L[tmp_replace_idx]
            replace_idx &= (sl2 > 0) & (slx * slx * suml2 > sumlx * sumlx * sl2)

            L[replace_idx] = new_L[replace_idx]
            sumlx = slx
            suml2 = sl2

    L = torch.round(iscale * data).clip(-nmax, nmax - 1) + nmax
    scales = torch.where(iscale != 0, 1 / iscale, 0).reshape(iscale.shape[:2])
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
    iscale = torch.where(scale == 0, 0, 1 / scale)

    l_values = torch.round(iscale * (data - group_min))
    L = torch.clip(l_values, 0, nmax).to(torch.uint8)

    diffs = scale * L + group_min - data
    diffs = torch.abs(diffs) if use_mad else diffs ** 2
    best_mad = torch.sum(weights * diffs, dim=-1, keepdim=True)

    if nstep < 1:
        return scale.reshape(scale.shape[:2]), L, the_mins.reshape(the_mins.shape[:2])

    for step in range(nstep):
        new_scale = (group_max - group_min) / (rmin + rdelta * step + nmax)
        new_iscale = torch.where(new_scale == 0, 0, 1 / new_scale)

        l_values = torch.round(new_iscale * (data - group_min))
        Laux = torch.clip(l_values, 0, nmax).to(torch.uint8)

        sum_l = torch.sum(weights * Laux, dim=-1, keepdim=True)
        sum_l2 = torch.sum(weights * Laux ** 2, dim=-1, keepdim=True)
        sum_xl = torch.sum(weights * Laux * data, dim=-1, keepdim=True)

        D = sum_w * sum_l2 - sum_l * sum_l
        replace_idx = D > 0

        this_scale = (sum_w * sum_xl - sum_x * sum_l) / D
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D
        this_min[this_min > 0] = 0
        this_scale[this_min > 0] = (sum_xl / sum_l2)[this_min > 0]

        diffs = this_scale * Laux + this_min - data
        diffs = torch.abs(diffs) if use_mad else diffs ** 2
        mad = torch.sum(weights * diffs, dim=-1, keepdim=True)

        replace_idx &= mad < best_mad
        best_mad[replace_idx] = mad[replace_idx]
        L[replace_idx.reshape(replace_idx.shape[:2])] = Laux[replace_idx.reshape(replace_idx.shape[:2])]
        scale[replace_idx] = this_scale[replace_idx]
        group_min[replace_idx] = this_min[replace_idx]

    the_mins = -group_min
    return scale.reshape(scale.shape[:2]), L, the_mins.reshape(the_mins.shape[:2])


def make_qkx3_quants(data, bits, weights, rmin=-1.0, rdelta=0.1, nstep=20, use_mad=False):
    return make_qkx2_quants(data, bits, weights, rmin=rmin, rdelta=rdelta, nstep=nstep, use_mad=use_mad)


def make_qp_quants(nmax, data, quant_weights):
    group_max = torch.max(data, dim=-1, keepdim=True)[0]
    scale = group_max / nmax
    iscale = torch.where(scale == 0, 0, 1 / scale)

    L = torch.round(iscale * data)
    diffs = data - scale * L
    best_mse = torch.sum(quant_weights * diffs * diffs)

    for _is in range(-4, 5):
        if _is == 0:
            continue
        scale_is = group_max / (0.1 * _is + nmax)
        iscale_is = torch.where(scale == 0, 0, 1 / scale_is)

        tmp_L = torch.round(iscale_is * data).clip(max=nmax)
        diffs = data - scale_is * tmp_L
        mse = torch.sum(quant_weights * diffs * diffs)

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
    n = torch.where((n & 0x7fffffff) > 0x7f800000, (n & 0xffff0000) | (64 << 16), n)
    # round to nearest even
    n = n.to(torch.uint64) + (0x7fff + ((n >> 16) & 1)) >> 16
    return n.cpu().numpy().astype(np.uint16).view(np.uint8)


@register_qtype("q4_0")
def q4_0_quant_block(blocks, scale=None, zp=None, **kwargs):
    if scale is not None:
        d = scale.reshape((-1, 1))
    else:
        imax = abs(blocks).argmax(axis=-1, keepdims=True)
        max = np.take_along_axis(blocks, imax, axis=-1)
        d = max / -8
    id = torch.where(d == 0, 0, 1 / d)

    qs = torch.trunc(blocks.to(torch.float64) * id.to(torch.float64) + 8.5).clip(0, 15).to(torch.uint8)

    n_blocks = blocks.shape[0]
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
    id = torch.where(d == 0, 0, 1 / d)

    qs = torch.trunc((blocks - min) * id + 0.5).clip(0, 15).to(torch.uint8)

    n_blocks = blocks.shape[0]
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
        max = torch.take_along_dim(blocks, imax, dim=-1)[0]
        d = max / -16

    id = torch.where(d == 0, 0, 1 / d)
    n_blocks = blocks.shape[0]
    block_size = GGML_QUANT_SIZES["q5_0"][0]

    # FIXME: Q5_0's reference rounding is cursed and depends on FMA
    q = torch.trunc(blocks.to(torch.float64) * id.to(torch.float64) + 16.5).clip(0, 31).to(torch.uint8).cpu().numpy()

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

    id = torch.where(d == 0, 0, 1 / d)
    q = torch.trunc((blocks - min) * id + 0.5).clip(0, 31).to(torch.uint8).cpu().numpy()

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
    id = torch.where(d == 0, 0, 1 / d)

    qs = torch.clip(torch_roundf(blocks * id), -128, 127)

    # (n_blocks, 2)
    d = d.cpu().numpy().astype(np.float16).view(np.uint8)
    # (n_blocks, block_size)
    qs = qs.cpu().numpy().astype(np.int8).view(np.uint8)

    return np.concatenate([d, qs], axis=1)


@register_qtype("q2_k")
def q2_k_quant_block(blocks, scale=None, zp=None, wmin_m=None, d_scale=None, d_wmin_m=None, **kwargs):
    nb = blocks.shape[0]

    blocks = blocks.reshape((nb, QK_K // 16, 16))  # (nb, 16, 16)

    if scale is not None:
        scales = scale.reshape((-1, QK_K // 16))
        mins = wmin_m.reshape((-1, QK_K // 16))
        output_d = d_scale.reshape(-1, 1).to(torch.float32)
        output_dmin = d_wmin_m.reshape(-1, 1).to(torch.float32)
        inv_scales = torch.where(output_d == 0, 0, 1 / output_d)
        inv_mins = torch.where(d_wmin_m == 0, 0, 1 / output_dmin)
        max_scales = torch.max(scales, dim=-1, keepdim=True)[0]  # (nb, 1)
        max_mins = torch.max(mins, dim=-1, keepdim=True)[0]  # (nb, 1)
        all_L = torch.zeros_like(blocks, dtype=torch.uint8)
    else:
        scales, all_L, mins = make_qkx2_quants(blocks, bits=2, rmin=-0.5, rdelta=0.1, nstep=15, use_mad=True)
        max_scales = torch.max(scales, dim=-1, keepdim=True)[0]
        max_mins = torch.max(mins, dim=-1, keepdim=True)[0]
        inv_scales = torch.where(max_scales > 0, 15. / max_scales, 0)
        inv_mins = torch.where(max_mins > 0, 15. / max_mins, 0)

    replace_ids = (max_scales > 0).squeeze()
    output_scale = torch.zeros_like(scales).to(torch.uint8)
    output_scale[replace_ids] = torch.round(
        inv_scales[replace_ids] * scales[replace_ids]).clip(0, 15).to(torch.uint8)

    replace_ids = (max_mins > 0).squeeze()
    output_scale[replace_ids] |= torch.round(
        inv_mins[replace_ids] * mins[replace_ids]).clip(0, 15).to(torch.uint8) << 4
    if d_scale is None:
        output_d = torch.where(max_scales > 0, max_scales / 15, 0)
    if d_wmin_m is None:
        output_dmin = torch.where(max_mins > 0, max_mins / 15., 0)

    d_tmp = output_d * (output_scale & 0xF)
    dm_tmp = output_dmin * (output_scale >> 4)

    replace_ids = d_tmp != 0
    all_L[replace_ids] = torch.round(
        (blocks[replace_ids] + dm_tmp[replace_ids].unsqueeze(-1)) / d_tmp[replace_ids].unsqueeze(-1)).clip(0, 3).to(
        torch.uint8)
    all_L = np.clip(all_L.cpu().numpy(), 0, 3).astype(np.uint8)

    output_scale = output_scale.cpu().numpy()
    all_L = all_L.reshape(-1, 4, 32)
    output_qs = all_L[:, 0, :] | (all_L[:, 1, :] << 2) | (all_L[:, 2, :] << 4) | (all_L[:, 3, :] << 6)
    output_d = output_d.cpu().numpy()
    output_d = output_d.reshape(-1, 1).astype(np.float16).view(np.uint8)
    output_dmin = output_dmin.cpu().numpy()
    output_dmin = output_dmin.reshape(-1, 1).astype(np.float16).view(np.uint8)
    output_qs = output_qs.reshape((nb, QK_K // 4))

    # [scale, qs, d, dmin]
    return np.concatenate([output_scale, output_qs, output_d, output_dmin], axis=-1)


@register_qtype("q3_k")
def q3_k_quant_block(blocks: np.array, scale=None, zp=None, wmin_m=None, d_scale=None, d_wmin_m=None, **kwargs):
    nb = blocks.shape[0]
    blocks = blocks.reshape(nb, QK_K // 16, 16)

    output_scale = np.empty((nb, K_SCALE_SIZE), dtype=np.uint8)

    if scale is not None:
        qdq_scale = scale.reshape(-1, QK_K // 16).to(torch.float32)
        dq_scale=d_scale.reshape(-1, 1).to(torch.float32)
        inverse_dq_scale = torch.where(dq_scale==0,0,1/dq_scale)
        # output_d = d_scale.reshape(-1, 1).to(torch.float32)
        # iscales = torch.where(output_d == 0, 0, 1 / output_d)
        # inverse_scale = torch.where(scales==0, 0, 1.0 / scales)
        # all_L = torch.round(blocks*inverse_scale.unsqueeze(-1)).clip(-4, 3) + 4
        # all_L = all_L.to(torch.uint8)
    else: ## this is correct
        scales, _ = make_q3_quants(blocks, bits=3, do_rmse=True)
        scales_abs_max = abs(scales).argmax(dim=-1, keepdim=True)
        max_scales_mag = torch.take_along_dim(scales, scales_abs_max, dim=-1)
        inverse_dq_scale = torch.where(max_scales_mag != 0, -32 / max_scales_mag, 0)
        dq_scale = torch.where(inverse_dq_scale != 0, 1 / inverse_dq_scale, 0)
        qscale = torch.round(inverse_dq_scale * scales).clip(-32, 31)
        qdq_scale = dq_scale.to(torch.float32) * qscale


    # q_scales_offset = torch.round(inverse_dq_scale * scales).clip(-32, 31) + 32
    q_scales_offset = torch.round(qdq_scale*inverse_dq_scale).clip(-32, 31) + 32
    q_scales_offset = q_scales_offset.cpu().numpy().astype(np.uint8)
    output_scale[:, :8] = (q_scales_offset[:, :8] & 0xF) | ((q_scales_offset[:, 8:] & 0xF) << 4)
    hmask = q_scales_offset >> 4
    output_scale[:, 8:] = hmask[:, :4] | hmask[:, 4:8] << 2 | hmask[:, 8:12] << 4 | hmask[:, 12:] << 6

    reverse_qdq_scale = torch.where(qdq_scale == 0, 0, 1 / qdq_scale)
    all_L = (torch.round(blocks * reverse_qdq_scale.unsqueeze(-1)).clip(-4, 3) + 4).to(
        torch.uint8)

    all_L = all_L.cpu().numpy()
    output_hmask = np.bitwise_or.reduce(
        all_L.reshape(nb, 8, 32) >> 2 << np.arange(8, dtype=np.uint8).reshape(1, 8, 1), axis=1)  # pylint: disable=E1121
    all_L = np.where(all_L > 3, all_L - 4, all_L)

    output_qs = np.bitwise_or.reduce(
        all_L.reshape(nb, 2, 4, 32) << np.array([0, 2, 4, 6]).reshape(1, 1, 4, 1), axis=2)  # pylint: disable=E1121

    output_qs = output_qs.reshape(nb, 64).astype(np.uint8)
    dq_scale = dq_scale.cpu().numpy().reshape(-1, 1).astype(np.float16).view(np.uint8)
    # [hmask, qs, scale, d]
    return np.concatenate([output_hmask, output_qs, output_scale, dq_scale], axis=-1)



@register_qtype("q4_k")
def q4_k_quant_block(blocks, scale=None, zp=None, wmin_m=None, d_scale=None, d_wmin_m=None, **kwargs):
    nb = blocks.shape[0]
    blocks = blocks.reshape((nb, QK_K // 32, 32))

    output_scale = np.empty((nb, K_SCALE_SIZE), dtype=np.uint8)
    output_qs = np.empty((nb, QK_K // 64, 32), dtype=np.uint8)

    if scale is not None:
        scales = scale.reshape(-1, QK_K // 32)
        mins = wmin_m.reshape(-1, QK_K // 32)
        output_d = d_scale.reshape(-1, 1).to(torch.float32)
        output_dmin = d_wmin_m.reshape(-1, 1).to(torch.float32)
        inv_scales = torch.where(output_d == 0, 0, 1 / output_d)
        inv_mins = torch.where(d_wmin_m == 0, 0, 1 / output_dmin)
        max_scales = torch.max(scales, dim=-1, keepdim=True)[0]  # (nb, 1)
        max_mins = torch.max(mins, dim=-1, keepdim=True)[0]  # (nb, 1)
        all_L = torch.zeros_like(blocks, dtype=torch.uint8)
        # inverse_scale = torch.where(scales == 0, 0, 1.0 / scales)
        # all_L = torch.round(
        #     (blocks + mins.reshape(*mins.shape, 1)) * inverse_scale.reshape(*scales.shape, 1)).clip(
        #     0,15).to(torch.uint8)
    else:
        scales, all_L, mins = make_qkx2_quants(blocks, bits=4, rmin=-1, rdelta=0.1, nstep=20, use_mad=False)
        max_scales = torch.max(scales, dim=-1, keepdim=True)[0]
        max_mins = torch.max(mins, dim=-1, keepdim=True)[0]
        inv_scales = torch.where(max_scales > 0, 63. / max_scales, 0)
        inv_mins = torch.where(max_mins > 0, 63. / max_mins, 0)

    q_scales = torch.round(inv_scales * scales).clip(0, 63)
    q_mins = torch.round(inv_mins * mins).clip(0, 63)

    if d_scale is None:
        output_d = max_scales / 63
    if d_wmin_m is None:
        output_dmin = max_mins / 63

    d_tmp = output_d * q_scales
    dm_tmp = output_dmin * q_mins
    q_scales = q_scales.cpu().numpy().astype(np.uint8)
    q_mins = q_mins.cpu().numpy().astype(np.uint8)
    output_scale[:, :4] = q_scales[:, :4]
    output_scale[:, 4:8] = q_mins[:, :4]

    output_scale[:, 8:] = (q_scales[:, 4:] & 0xF) | ((q_mins[:, 4:] & 0xF) << 4)
    output_scale[:, :4] |= ((q_scales[:, 4:] >> 4) << 6)
    output_scale[:, 4:8] |= ((q_mins[:, 4:] >> 4) << 6)

    replace_ids = d_tmp != 0
    all_L[replace_ids] = torch.round(
        (blocks[replace_ids] + dm_tmp[replace_ids].unsqueeze(-1)) / d_tmp[replace_ids].unsqueeze(-1)).clip(
        0, 15).to(torch.uint8)
    all_L = np.clip(all_L.cpu().numpy(), 0, 15).astype(np.uint8)
    output_qs = all_L[:, ::2] | (all_L[:, 1::2] << 4)

    output_d = output_d.cpu().numpy()
    output_d = output_d.reshape(-1, 1).astype(np.float16).view(np.uint8)
    output_dmin = output_dmin.cpu().numpy()
    output_dmin = output_dmin.reshape(-1, 1).astype(np.float16).view(np.uint8)
    output_qs = output_qs.reshape(nb, QK_K // 2)

    # [d, dmin, scale, qs]
    return np.concatenate([output_d, output_dmin, output_scale, output_qs], axis=-1)


@register_qtype("q5_k")
def q5_k_quant_block(blocks, scale=None, zp=None, wmin_m=None, d_scale=None, d_wmin_m=None, **kwargs):
    nb = blocks.shape[0]
    blocks = blocks.reshape((nb, QK_K // 32, 32))

    output_scale = np.empty((nb, K_SCALE_SIZE), dtype=np.uint8)
    if scale is not None:
        scales = scale.reshape(-1, QK_K // 32)
        mins = wmin_m.reshape(-1, QK_K // 32)
        output_d = d_scale.reshape(-1, 1).to(torch.float32)
        output_dmin = d_wmin_m.reshape(-1, 1).to(torch.float32)
        inv_scales = torch.where(output_d == 0, 0, 1 / output_d)
        inv_mins = torch.where(d_wmin_m == 0, 0, 1 / output_dmin)
        max_scales = torch.max(scales, dim=-1, keepdim=True)[0]  # (nb, 1)
        max_mins = torch.max(mins, dim=-1, keepdim=True)[0]  # (nb, 1)
        all_L = torch.zeros_like(blocks, dtype=torch.uint8)
    else:
        scales, all_L, mins = make_qkx2_quants(blocks, bits=5, rmin=-0.5, rdelta=0.1, nstep=15, use_mad=False)
        max_scales = torch.max(scales, dim=-1, keepdim=True)[0]
        max_mins = torch.max(mins, dim=-1, keepdim=True)[0]
        inv_scales = torch.where(max_scales > 0, 63. / max_scales, 0)
        inv_mins = torch.where(max_mins > 0, 63. / max_mins, 0)

    q_scales = torch.round(inv_scales * scales).clip(0, 63)
    q_mins = torch.round(inv_mins * mins).clip(0, 63)

    if d_scale is None:
        output_d = max_scales / 63
    if d_wmin_m is None:
        output_dmin = max_mins / 63

    d_tmp = output_d * q_scales
    dm_tmp = output_dmin * q_mins

    q_scales = q_scales.cpu().numpy().astype(np.uint8)
    q_mins = q_mins.cpu().numpy().astype(np.uint8)
    output_scale[:, :4] = q_scales[:, :4]
    output_scale[:, 4:8] = q_mins[:, :4]

    output_scale[:, 8:] = (q_scales[:, 4:] & 0xF) | ((q_mins[:, 4:] & 0xF) << 4)
    output_scale[:, :4] |= ((q_scales[:, 4:] >> 4) << 6)
    output_scale[:, 4:8] |= ((q_mins[:, 4:] >> 4) << 6)

    replace_ids = d_tmp != 0
    all_L[replace_ids] = torch.round(
        (blocks[replace_ids] + dm_tmp[replace_ids].unsqueeze(-1)) \
        / d_tmp[replace_ids].unsqueeze(-1)).clip(0, 31).to(torch.uint8)
    all_L = np.clip(all_L.cpu().numpy().astype(np.uint8), 0, 31)  # (nb, 8, 32)

    output_qs = all_L[:, ::2] | (all_L[:, 1::2] << 4)
    output_qh = np.bitwise_or.reduce(
        all_L >> 4 << np.arange(8, dtype=np.uint8).reshape(1, 8, 1), axis=1).astype(np.uint8)  # pylint: disable=E1121

    output_d = output_d.cpu().numpy()
    output_d = output_d.reshape(-1, 1).astype(np.float16).view(np.uint8)
    output_dmin = output_dmin.cpu().numpy()
    output_dmin = output_dmin.reshape(-1, 1).astype(np.float16).view(np.uint8)
    output_qs = output_qs.reshape(nb, QK_K // 2)
    # [d, dmin, scale, qh, qs]
    return np.concatenate([output_d, output_dmin, output_scale, output_qh, output_qs], axis=-1)


@register_qtype("q6_k")
def q6_k_quant_block(blocks: np.array, scale=None, zp=None, wmin_m=None, d_scale=None, d_wmin_m=None, **kwargs):
    nb = blocks.shape[0]
    blocks = blocks.reshape((nb, QK_K // 16, 16))

    scale = None
    if scale is not None:
        scales = scale.reshape(-1, QK_K // 16)
        output_d = d_scale.reshape(-1, 1).to(torch.float32)
        iscales = torch.where(output_d == 0, 0, 1 / output_d)
        all_L = torch.zeros_like(blocks, dtype=torch.uint8) + 32
    else:
        scales, all_L = make_qx_quants(blocks, bits=6, rmse_type=1, qw=None)
        imax = abs(scales).argmax(dim=-1, keepdim=True)
        max_scales = torch.take_along_dim(scales, imax, dim=-1)
        iscales = torch.where(max_scales != 0, -128 / max_scales, 0)
        output_d = torch.where(iscales != 0, 1 / iscales, -1)

    output_scale = torch.round(iscales * scales).clip(max=127).to(torch.int8)
    d_tmp = output_d * output_scale.to(torch.float32)
    replace_ids = d_tmp != 0
    all_L[replace_ids] = torch.round(blocks[replace_ids] / d_tmp[replace_ids].reshape(-1, 1) + 32) \
        .clip(0, 63).to(torch.uint8)

    all_L = all_L.cpu().numpy()
    tmp_L = all_L.reshape(nb, 4, 64) & 0xF
    output_ql = (tmp_L[:, ::2] | (tmp_L[:, 1::2] << 4)).reshape(nb, QK_K // 2)
    output_qh = np.bitwise_or.reduce(
        (all_L >> 4).reshape(nb, 2, 4, 32) << np.array([0, 2, 4, 6]).reshape(1, 1, 4, 1),  # pylint: disable=E1121
        axis=2).reshape(nb, QK_K // 4).astype(np.uint8)  # pylint: disable=E1121

    output_d = output_d.cpu().numpy().reshape(-1, 1).astype(np.float16).view(np.uint8)

    # [ql, qh, scale, d]
    output_scale = output_scale.cpu().numpy().view(np.uint8)
    return np.concatenate([output_ql, output_qh, output_scale, output_d], axis=-1)
