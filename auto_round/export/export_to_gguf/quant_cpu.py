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
from concurrent.futures import ProcessPoolExecutor
from .utils import QK_K, K_SCALE_SIZE, GGML_QUANT_SIZES

GGML_QUANT_BLOCK = {}


def register_block(name):

    def register(cls):
        GGML_QUANT_BLOCK[name] = cls
        return cls

    return register


def ggml_quant_cpu(data, ggml_type, scale=None, zp=None, wmin_m=None, d_scale=None, d_wmin_m=None):
    import torch
    data = data.squeeze().cpu().numpy() if isinstance(data, torch.Tensor) else data
    scale = scale.numpy() if isinstance(scale, torch.Tensor) else scale
    zp = zp.numpy() if isinstance(zp, torch.Tensor) else zp
    wmin_m = wmin_m.numpy() if isinstance(wmin_m, torch.Tensor) else wmin_m
    d_scale = d_scale.numpy() if isinstance(d_scale, torch.Tensor) else d_scale
    d_wmin_m = d_wmin_m.numpy() if isinstance(d_wmin_m, torch.Tensor) else d_wmin_m

    block_size, type_size = GGML_QUANT_SIZES[ggml_type]

    data = data.astype(np.float32, copy=False)
    shape = data.shape
    n_blocks = data.size // block_size
    blocks = data.reshape((n_blocks, block_size))

    if ggml_type.endswith("_k"):
        worker = 16
    else:
        worker = 0

    worker = 0
    if worker > 0:
        n_groups = (data.shape[0] // worker) or 1
        blocks = np.array_split(blocks, n_groups, axis=0)
        scale = np.array_split(scale, n_groups, axis=0) if scale is not None else [None] * n_groups
        zp = np.array_split(zp, n_groups, axis=0) if zp is not None else [None] * n_groups
        wmin_m = np.array_split(wmin_m, n_groups, axis=0) if wmin_m is not None else [None] * n_groups
        d_scale = np.array_split(d_scale, n_groups, axis=0) if d_scale is not None else [None] * n_groups
        d_wmin_m = np.array_split(d_wmin_m, n_groups, axis=0) if d_wmin_m is not None else [None] * n_groups

        quant_func = GGML_QUANT_BLOCK[ggml_type]
        with ProcessPoolExecutor(worker) as executor:
            result = executor.map(quant_func, blocks, scale, zp, wmin_m, d_scale, d_wmin_m)
        new_data = np.array(list(result), dtype=np.uint8)
    else:
        quant_func = GGML_QUANT_BLOCK[ggml_type]
        new_data = quant_func(blocks, scale, zp, wmin_m=wmin_m, d_scale=d_scale, d_wmin_m=d_wmin_m)

    assert new_data.dtype == np.uint8
    assert new_data.shape[-1] == type_size
    new_data = new_data.reshape(*shape[:-1], shape[-1] // block_size * type_size)
    return new_data


def np_roundf(n: np.ndarray) -> np.ndarray:
    a = abs(n)
    floored = np.floor(a)
    b = floored + np.floor(2 * (a-floored))
    return np.sign(n) * b


@register_block("bf16")
def bf16_quant_block(blocks: np.array, scale=None, zp=None, **kwargs):
    n = blocks.view(np.uint32)
    # force nan to quiet
    n = np.where((n & 0x7fffffff) > 0x7f800000, (n & np.uint32(0xffff0000)) | np.uint32(64 << 16), n)
    # round to nearest even
    n = (np.uint64(n) + (0x7fff + ((n >> 16) & 1))) >> 16
    return n.astype(np.uint16).view(np.uint8)


@register_block("q4_0")
def q4_0_quant_block(blocks: np.array, scale=None, zp=None, **kwargs):
    if scale is not None:
        d = scale.reshape((-1, 1))
    else:
        imax = abs(blocks).argmax(axis=-1, keepdims=True)
        max = np.take_along_axis(blocks, imax, axis=-1)
        d = max / -8
    with np.errstate(divide="ignore"):
        id = np.where(d == 0, 0, 1 / d)

    qs = np.trunc((np.float64(blocks) * np.float64(id)) + np.float64(8.5),
                  dtype=np.float32).astype(np.uint8).clip(0, 15)

    n_blocks = blocks.shape[0]
    block_size = GGML_QUANT_SIZES["q4_0"][0]
    qs = qs.reshape((n_blocks, 2, block_size // 2))
    qs = qs[..., 0, :] | (qs[..., 1, :] << np.uint8(4))

    d = d.astype(np.float16).view(np.uint8)

    return np.concatenate([d, qs], axis=-1)


@register_block("q4_1")
def q4_1_quant_block(blocks: np.array, scale=None, zp=None, **kwargs):
    if scale is not None:
        d = scale.reshape((-1, 1))
        min = zp.reshape((-1, 1)) * d * -1
    else:
        max = blocks.max(axis=-1, keepdims=True)
        min = blocks.min(axis=-1, keepdims=True)
        d = (max-min) / 15
    with np.errstate(divide="ignore"):
        id = np.where(d == 0, 0, 1 / d)

    qs = np.trunc((blocks-min) * id + np.float32(0.5), dtype=np.float32).astype(np.uint8).clip(0, 15)

    n_blocks = blocks.shape[0]
    block_size = GGML_QUANT_SIZES["q4_1"][0]
    qs = qs.reshape((n_blocks, 2, block_size // 2))
    qs = qs[..., 0, :] | (qs[..., 1, :] << np.uint8(4))

    d = d.astype(np.float16).view(np.uint8)
    m = min.astype(np.float16).view(np.uint8)
    return np.concatenate([d, m, qs], axis=-1)


@register_block("q5_0")
def q5_0_quant_block(blocks: np.array, scale=None, zp=None, **kwargs):
    if scale is not None:
        d = scale.reshape((-1, 1))
    else:
        imax = abs(blocks).argmax(axis=-1, keepdims=True)
        max = np.take_along_axis(blocks, imax, axis=-1)
        d = max / -16
        
    n_blocks = blocks.shape[0]
    block_size = GGML_QUANT_SIZES["q5_0"][0]
    with np.errstate(divide="ignore"):
        id = np.where(d == 0, 0, 1 / d)
    # FIXME: Q5_0's reference rounding is cursed and depends on FMA
    q = np.trunc((np.float64(blocks) * np.float64(id)) + np.float64(16.5), dtype=np.float32).astype(np.uint8).clip(0, 31)

    qs = q.reshape((n_blocks, 2, block_size // 2))
    qs = (qs[..., 0, :] & np.uint8(0x0F)) | (qs[..., 1, :] << np.uint8(4))

    qh = np.packbits(q.reshape((n_blocks, 1, 32)) >> np.uint8(4), axis=-1, bitorder="little").reshape(n_blocks, 4)

    d = d.astype(np.float16).view(np.uint8)

    return np.concatenate([d, qh, qs], axis=-1)


@register_block("q5_1")
def q5_1_quant_block(blocks: np.array, scale=None, zp=None, **kwargs):
    if scale is not None:
        d = scale.reshape((-1, 1))
        min = zp.reshape((-1, 1)) * d * -1
    else:
        max = blocks.max(axis=-1, keepdims=True)
        min = blocks.min(axis=-1, keepdims=True)
        d = (max - min) / 31

    n_blocks = blocks.shape[0]
    block_size = GGML_QUANT_SIZES["q5_1"][0]

    with np.errstate(divide="ignore"):
        id = np.where(d == 0, 0, 1 / d)
    q = np.trunc((blocks - min) * id + np.float32(0.5), dtype=np.float32).astype(np.uint8).clip(0, 31)

    qs = q.reshape((n_blocks, 2, block_size // 2))
    qs = (qs[..., 0, :] & np.uint8(0x0F)) | (qs[..., 1, :] << np.uint8(4))

    qh = np.packbits(q.reshape((n_blocks, 1, 32)) >> np.uint8(4), axis=-1, bitorder="little").reshape(n_blocks, 4)

    d = d.astype(np.float16).view(np.uint8)
    m = min.astype(np.float16).view(np.uint8)

    return np.concatenate([d, m, qh, qs], axis=-1)


@register_block("q8_0")
def q8_0_quant_block(blocks: np.array, scale=None, zp=None) -> np.ndarray:
    if scale is not None:
        d = scale.reshape((-1, 1))
    else:
        d = abs(blocks).max(axis=1, keepdims=True) / 127
    with np.errstate(divide="ignore"):
        id = np.where(d == 0, 0, 1 / d)
    qs = np_roundf(blocks * id)

    # (n_blocks, 2)
    d = d.astype(np.float16).view(np.uint8)
    # (n_blocks, block_size)
    qs = qs.astype(np.int8).view(np.uint8)

    return np.concatenate([d, qs], axis=1)

def make_qkx2_quants(data, bits, rmin=-1, rdelta=0.1, nstep=20, use_mad=False):
    assert bits in [2, 4], f"bits={bits} is not supported"
    nmax = bits ** 2 -1
    # data shape (nb, 8, 32) for Q4_K, (nb, 16, 16) for Q2_K
    if len(data.shape) == 2:
        data_shape = (-1, 8, 32) if bits == 4 else (-1, 16, 16)
        data = data.reshape(data_shape)
    sum_x2 = np.sum(np.power(data, 2), axis=-1, keepdims=True)
    av_x = np.sqrt(sum_x2 / 32)
    weight = np.abs(data) + av_x

    group_min = np.min(data, axis=-1, keepdims=True)
    group_max = np.max(data, axis=-1, keepdims=True)

    the_mins = -group_min

    sum_w = np.sum(weight, axis=-1, keepdims=True)
    sum_x = np.sum(weight * data, axis=-1, keepdims=True)

    group_min[group_min > 0] = 0

    scale = (group_max - group_min) / nmax
    iscale = np.where(scale == 0, 0, 1 /scale)

    l_values = np.round(iscale * (data - group_min))
    L = np.clip(l_values, 0, nmax).astype(np.uint8)

    diffs = scale * L + group_min - data
    diffs = np.abs(diffs) if use_mad else diffs**2
    best_mad = np.sum(weight * diffs, axis=-1, keepdims=True)

    if nstep < 1:
        return scale.reshape(scale.shape[:2]), L, the_mins.reshape(the_mins.shape[:2])

    for step in range(nstep):
        new_scale = (group_max - group_min) / (rmin + rdelta * step + nmax)
        new_iscale = np.where(new_scale == 0, 0, 1 /new_scale)

        l_values = np.round(new_iscale * (data - group_min))
        Laux = np.clip(l_values, 0, nmax).astype(np.uint8)

        sum_l = np.sum(weight * Laux, axis=-1, keepdims=True)
        sum_l2 = np.sum(weight * Laux**2, axis=-1, keepdims=True)
        sum_xl = np.sum(weight * Laux * data, axis=-1, keepdims=True)

        D = sum_w * sum_l2 - sum_l * sum_l
        replace_idx = D > 0

        this_scale = (sum_w * sum_xl - sum_x * sum_l) / D
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D
        this_min[this_min > 0] = 0
        this_scale[this_min > 0] = (sum_xl / sum_l2)[this_min > 0]

        diffs = this_scale * Laux + this_min - data
        diffs = np.abs(diffs) if use_mad else diffs**2
        mad = np.sum(weight * diffs, axis=-1, keepdims=True)

        replace_idx &= mad < best_mad
        best_mad[replace_idx] = mad[replace_idx]
        L[replace_idx.reshape(replace_idx.shape[:2])] = Laux[replace_idx.reshape(replace_idx.shape[:2])]
        scale[replace_idx] = this_scale[replace_idx]
        group_min[replace_idx] = this_min[replace_idx]

    the_mins = -group_min
    return scale.reshape(scale.shape[:2]), L, the_mins.reshape(the_mins.shape[:2])


@register_block("q2_k")
def q2_k_quant_block(blocks: np.array, scale=None, zp=None, wmin_m=None, d_scale=None, d_wmin_m=None):
    nb = blocks.shape[0]
    output_scale = np.empty((nb, 16), dtype=np.uint8)
    output_qs = np.empty((nb, QK_K // 16 // 4, 16), dtype=np.uint8)

    blocks = blocks.reshape((nb, QK_K // 16, 16)) # (nb, 16, 16)

    if scale is not None:
        scales = scale.reshape((-1, QK_K // 16))
        mins = wmin_m.reshape((-1, QK_K // 16))
        output_d = d_scale.astype(np.float32) if len(d_scale.shape) == 2 \
            else d_scale.reshape(-1, 1).astype(np.float32)
        output_dmin = d_wmin_m.astype(np.float32) if len(d_wmin_m.shape) == 2 \
            else d_wmin_m.reshape(-1, 1).astype(np.float32)
        inv_scales = np.where(d_scale == 0, 0, 1 / output_d)
        inv_mins = np.where(d_wmin_m == 0, 0, 1 / output_dmin)
        max_scales = np.max(scales, axis=-1, keepdims=True)  # (nb, 1)
        max_mins = np.max(mins, axis=-1, keepdims=True)  # (nb, 1)
        all_L = np.round((blocks + mins.reshape(*mins.shape, 1)) / scales.reshape(*scales.shape, 1)).astype(np.uint8)
    else:
        scales, all_L, mins = make_qkx2_quants(blocks, bits=2, rmin=-0.5, rdelta=0.1, nstep=15, use_mad=True)
        max_scales = np.max(scales, axis=-1, keepdims=True)
        max_mins = np.max(mins, axis=-1, keepdims=True)
        inv_scales = np.where(max_scales > 0, 15. / max_scales, 0)
        inv_mins = np.where(max_mins > 0, 15. / max_mins, 0)
    
    replace_ids = (max_scales > 0).reshape(-1)
    output_scale = np.zeros_like(scales).astype(np.uint8)
    output_scale[replace_ids] = np.round(inv_scales * scales).astype(np.uint8)

    replace_ids = (max_mins > 0).reshape(-1)
    output_scale[replace_ids] |= np.round(inv_mins * mins).astype(np.uint8) << 4

    if d_scale is None:
        output_d = np.where(max_scales > 0, max_scales / 15, 0)
    if d_wmin_m is None:
        output_dmin = np.where(max_mins > 0 ,max_mins / 15., 0)
    
    d_tmp = output_d * (output_scale & 0xF)
    dm_tmp = output_dmin * (output_scale >> 4)

    replace_ids = d_tmp != 0
    all_L[replace_ids] = np.round(
        (blocks[replace_ids] + dm_tmp[replace_ids].reshape(-1, 1)) / d_tmp[replace_ids].reshape(-1, 1)).astype(
            np.uint8)
    all_L = np.clip(all_L, 0, 3)

    output_qs = all_L[:,::4] | (all_L[:,1::4] << 2) | (all_L[:,2::4] << 4) | (all_L[:,3::4] << 6)
    output_d = output_d.reshape(-1, 1).astype(np.float16).view(np.uint8)
    output_dmin = output_dmin.reshape(-1, 1).astype(np.float16).view(np.uint8)
    output_qs = output_qs.reshape((nb, QK_K // 4))

    # [scale, qs, d, dmin]
    return np.concatenate([output_scale, output_qs, output_d, output_dmin], axis=-1)
    

@register_block("q4_k")
def q4_k_quant_block(blocks: np.array, scale=None, zp=None, wmin_m=None, d_scale=None, d_wmin_m=None):
    nb = blocks.shape[0]
    blocks = blocks.reshape((nb, QK_K // 32, 32))  # (nb, 8, 32)

    output_scale = np.empty((nb, K_SCALE_SIZE), dtype=np.uint8)
    output_qs = np.empty((nb, QK_K // 64, 32), dtype=np.uint8)

    if scale is not None:
        scales = scale.reshape(-1, QK_K // 32)
        mins = wmin_m.reshape(-1, QK_K // 32)
        output_d = d_scale.astype(np.float32) if len(d_scale.shape) == 2 \
            else d_scale.reshape(-1, 1).astype(np.float32)
        output_dmin = d_wmin_m.astype(np.float32) if len(d_wmin_m.shape) == 2 \
            else d_wmin_m.reshape(-1, 1).astype(np.float32)
        inv_scales = np.where(d_scale == 0, 0, 1 / output_d)
        inv_mins = np.where(d_wmin_m == 0, 0, 1 / output_dmin)
        max_scales = np.max(scales, axis=-1, keepdims=True)  # (nb, 1)
        max_mins = np.max(mins, axis=-1, keepdims=True)  # (nb, 1)
        all_L = np.round((blocks + mins.reshape(*mins.shape, 1)) / scales.reshape(*scales.shape, 1)).astype(np.uint8)
    else:
        scales, all_L, mins = make_qkx2_quants(blocks, bits=4, rmin=-1, rdelta=0.1, nstep=20, use_mad=False)
        max_scales = np.max(scales, axis=-1, keepdims=True)
        max_mins = np.max(mins, axis=-1, keepdims=True)
        inv_scales = np.where(max_scales > 0, 63. / max_scales, 0)
        inv_mins = np.where(max_mins > 0, 63. / max_mins, 0)

    q_scales = np.round(inv_scales * scales).astype(np.uint8)
    q_mins = np.round(inv_mins * mins).astype(np.uint8)

    output_scale[:,:4] = q_scales[:,:4]
    output_scale[:,4:8] = q_mins[:,:4]

    output_scale[:,8:] = (q_scales[:,4:] & 0xF) | ((q_mins[:,4:] & 0xF) << 4)
    output_scale[:,:4] |= ((q_scales[:,4:] >> 4) << 6)
    output_scale[:,4:8] |= ((q_mins[:,4:] >> 4) << 6)

    if d_scale is None:
        output_d = max_scales / 63
    if d_wmin_m is None:
        output_dmin = max_mins / 63

    d_tmp = output_d * q_scales
    dm_tmp = output_dmin * q_mins

    replace_ids = d_tmp != 0
    all_L[replace_ids] = np.round(
        (blocks[replace_ids] + dm_tmp[replace_ids].reshape(-1, 1)) / d_tmp[replace_ids].reshape(-1, 1)).astype(
            np.uint8)
    all_L = np.clip(all_L, 0, 15)
    output_qs = all_L[:,::2] | (all_L[:,1::2] << 4)

    output_d = output_d.reshape(-1, 1).astype(np.float16).view(np.uint8)
    output_dmin = output_dmin.reshape(-1, 1).astype(np.float16).view(np.uint8)
    output_qs = output_qs.reshape(nb, QK_K // 2)

    # [d, dmin, scale, qs]
    return np.concatenate([output_d, output_dmin, output_scale, output_qs], axis=-1)
