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

QK_K = 256
K_SCALE_SIZE = 12
GGML_QUANT_SIZES = {
    "bf16": (1, 2),
    "q4_0": (32, 2 + 16),
    "q4_1": (32, 2 + 2 + 16),
    "q4_k": (256, 2 + 2 + QK_K//2 + 12),
}

GGML_QUANT_BLOCK = {}


def register_block(name):

    def register(cls):
        GGML_QUANT_BLOCK[name] = cls
        return cls

    return register


def ggml_quant(data: np.array, ggml_type, scale=None, zp=None, **kwargs):
    block_size, type_size = GGML_QUANT_SIZES[ggml_type]

    data = data.astype(np.float32, copy=False)
    shape = data.shape
    n_blocks = data.size // block_size
    blocks = data.reshape((n_blocks, block_size))

    new_data = GGML_QUANT_BLOCK[ggml_type](blocks, scale, zp, **kwargs)
    new_data = new_data.reshape(*shape[:-1], shape[-1] // block_size * type_size)
    return new_data


@register_block("bf16")
def bf16_quant_block(blocks: np.array, scale=None, zp=None):
    n = blocks.view(np.uint32)
    # force nan to quiet
    n = np.where((n & 0x7fffffff) > 0x7f800000, (n & np.uint32(0xffff0000)) | np.uint32(64 << 16), n)
    # round to nearest even
    n = (np.uint64(n) + (0x7fff + ((n >> 16) & 1))) >> 16
    return n.astype(np.uint16).view(np.uint8)


@register_block("q4_0")
def q4_0_quant_block(blocks: np.array, scale=None, zp=None):
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
def q4_1_quant_block(blocks: np.array, scale=None, zp=None):
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


def make_qkx2_quants(data, weight, nmax, group_size, rmin=-1, rdelta=0.1, nstep=20):
    group_min = np.min(data)
    group_max = np.max(data)

    sum_w = np.sum(weight)
    sum_x = np.sum(weight * data)

    if group_min > 0:
        group_min = 0
    if group_min == group_max:
        L = np.zeros(group_size, dtype=np.uint8)
        the_min = -group_min
        return 0.0, L, the_min

    iscale = nmax / (group_max-group_min)
    scale = 1 / iscale
    L = np.zeros(group_size, dtype=np.uint8)

    l_values = np.round(iscale * (data - group_max))
    L = np.clip(l_values, 0, nmax).astype(np.uint8)
    diffs = scale * L + group_min - data
    best_mad = np.sum(weight * (diffs**2))

    if nstep < 1:
        the_min = -group_min
        return scale, L, the_min

    Laux = []
    for step in range(nstep):
        iscale = (rmin + rdelta*step + nmax) / (group_max-group_min)
        l_values = np.round(iscale * (data - group_min))
        Laux = np.clip(l_values, 0, nmax).astype(np.uint8)

        sum_l = np.sum(weight * Laux)
        sum_l2 = np.sum(weight * Laux**2)
        sum_xl = np.sum(weight * Laux * data)

        D = sum_w*sum_l2 - sum_l*sum_l
        if D > 0:
            this_scale = (sum_w*sum_xl - sum_x*sum_l) / D
            this_min = (sum_l2*sum_x - sum_l*sum_xl) / D
            if this_min > 0:
                this_min = 0
                this_scale = sum_xl / sum_l2

            diffs = this_scale * Laux + this_min - data
            mad = np.sum(weight * diffs**2)    

            if mad < best_mad:
                L = Laux.copy()
                best_mad = mad
                scale = this_scale
                group_min = this_min

    the_min = -group_min
    return scale, L, the_min


from tqdm import tqdm
@register_block("q4_k")
def q4_k_quant_block(blocks: np.array, scale=None, zp=None, wmin_m=None, d_scale=None, d_wmin_m=None):
    nb = blocks.shape[0]
    output_scale = np.empty((nb, K_SCALE_SIZE), dtype=np.uint8)
    output_d = np.empty(nb, dtype=np.float32)
    output_dmin = np.empty(nb, dtype=np.float32)
    output_qs = np.empty((nb, QK_K // 64, 32), dtype=np.uint8)

    blocks = blocks.reshape((nb, QK_K // 32, 32))
    sum_x2 = np.sum(np.power(blocks, 2), axis=-1)
    av_x = np.sqrt(sum_x2 / 32)
    weight = blocks + av_x.reshape((*av_x.shape, 1))
    scales = np.empty(QK_K // 32, dtype=np.float32)
    mins = np.empty(QK_K // 32, dtype=np.float32)

    if scale is not None:
        scale = scale.reshape(-1, QK_K // 32)
        wmin_m = wmin_m.reshape(-1, QK_K // 32)
        output_d = d_scale.astype(np.float32)
        output_dmin = d_wmin_m.astype(np.float32)
        inv_scales = np.where(d_scale==0, 0, 1 / output_d)
        inv_mins = np.where(d_wmin_m == 0, 0, 1 / output_dmin)


    for i in tqdm(range(nb), desc="packing layer"):
    # for i in range(nb):
        all_L = np.empty(blocks[i].shape, dtype=np.uint8)

        if scale is not None:
            scales = scale[i]
            mins = wmin_m[i]
            inv_scale = inv_scales[i]
            inv_min = inv_mins[i]
            max_scale = max(scales)
            max_min = max(mins)
            all_L = np.round((blocks[i] + mins.reshape(-1, 1)) / scales.reshape(-1,1)).astype(np.uint8)
        else:
            for j in range(QK_K // 32):
                tmp_scale, tmp_l, the_min = make_qkx2_quants(
                    blocks[i][j], weight[i][j], nmax=15, group_size=32, rmin=-1, rdelta=0.1, nstep=0)
                all_L[j] = tmp_l
                scales[j] = tmp_scale
                mins[j] = the_min

            max_scale = max(scales)
            max_min = max(mins)
            inv_scale = 63. / max_scale if max_scale > 0 else 0.
            inv_min = 63. / max_min if max_min > 0 else 0.

        ls = np.round(inv_scale * scales).astype(np.uint8)
        lm = np.round(inv_min * mins).astype(np.uint8)

        output_scale[i][:4] = ls[:4]
        output_scale[i][4:8] = lm[:4]

        output_scale[i][8:] = (ls[4:] & 0xF) | ((lm[4:] & 0xF) << 4)
        output_scale[i][:4] |= ((ls[4:] >> 4) << 6)
        output_scale[i][4:8] |= ((lm[4:] >> 4) << 6)

        if d_scale is None:
            output_d[i] = max_scale / 63 
        if d_wmin_m is None:
            output_dmin[i] = max_min / 63 
        

        d_tmp = output_d[i] * ls
        dm_tmp = output_dmin[i] * lm

        for j in range(d_tmp.size):
            if d_tmp[j] == 0.:
                continue
            else:
                all_L[j] = np.round((blocks[i][j] + dm_tmp[j]) / d_tmp[j]).astype(np.uint8)

        all_L = np.clip(all_L, 0, 15)

        output_qs[i] = all_L[::2] | (all_L[1::2] << 4)

    output_d = output_d.reshape(-1, 1).astype(np.float16).view(np.uint8)
    output_dmin = output_dmin.reshape(-1, 1).astype(np.float16).view(np.uint8)
    output_qs = output_qs.reshape(nb, QK_K // 2)

    return np.concatenate([output_d, output_dmin, output_scale, output_qs], axis=-1)
