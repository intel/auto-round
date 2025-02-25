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


def ggml_quant(data: np.array, ggml_type, scale=None, zp=None):
    block_size, type_size = GGML_QUANT_SIZES[ggml_type]

    data = data.astype(np.float32, copy=False)
    shape = data.shape
    n_blocks = data.size // block_size
    blocks = data.reshape((n_blocks, block_size))

    new_data = GGML_QUANT_BLOCK[ggml_type](blocks, scale, zp)
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
        L = [0] * group_size
        the_min = -group_min
        return 0.0, L, the_min

    iscale = nmax / (group_max-group_min)
    scale = 1 / iscale
    best_mad = 0
    L = []
    for i in range(group_size):
        l = np.round(iscale * (data[i] - group_min))
        L.append(int(max(0, min(nmax, l))))
        diff = scale * L[-1] + group_min - data[i]
        diff = diff**2
        w = weight[i]
        best_mad += w * diff
    if nstep < 1:
        the_min = -group_min
        return scale, L, the_min

    Laux = []
    for step in range(nstep):
        iscale = (rmin + rdelta*step + nmax) / (group_max-group_min)
        sum_l, sum_l2, sum_xl = 0, 0, 0
        for i in range(group_size):
            l = round(iscale * (data[i] - group_min))
            l = max(0, min(nmax, l))
            Laux.append(l)
            sum_l += weight[i] * l
            sum_l2 += weight[i] * l * l
            sum_xl += weight[i] * l * data[i]
        D = sum_w*sum_l2 - sum_l*sum_l
        if D > 0:
            this_scale = (sum_w*sum_xl - sum_x*sum_l) / D
            this_min = (sum_l2*sum_x - sum_l*sum_xl) / D
            if this_min > 0:
                this_min = 0
                this_scale = sum_xl / sum_l2
            mad = 0
            for i in range(group_size):
                diff = this_scale * Laux[i] + this_min - data[i]
                diff = diff**2
                mad += w * diff
            if mad < best_mad:
                for i in range(group_size):
                    L[i] = Laux[i]
                best_mad = mad
                scale = this_scale
                group_min = this_min

    the_min = -group_min
    return scale, the_min


@register_block("q4_k")
def q4_k_quant_block(blocks: np.array, scale=None, zp=None):

    nb = blocks.shape[0]
    output_scale = np.empty((nb, QK_K//32 + 4), dtype=np.uint8)
    output_d = np.empty(nb, dtype=np.float32)
    output_dmin = np.empty(nb, dtype=np.float32)
    output_qs = np.empty((nb, QK_K //64, 32), dtype=np.uint8)

    if scale is not None:
        #TODO:
        pass
    else:
        ori_shape = blocks.shape
        blocks = blocks.reshape((nb, QK_K // 32, 32))
        sum_x2 = np.sum(np.power(blocks, 2), axis=-1)
        av_x = np.sqrt(sum_x2 / 32)
        weight = blocks + av_x.reshape((*av_x.shape, 1))
        scales = np.empty(QK_K//32, dtype=np.float32)
        mins = np.empty(QK_K//32, dtype=np.float32)
        for i in range(nb):
            for j in range(QK_K // 32):
                scale, the_min = make_qkx2_quants(
                    blocks[i][j], weight[i][j], nmax=15, group_size=32, rmin=-1, rdelta=0.1, nstep=20)
                scales[j] = scale
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
            
            output_d[i] = max_scale / 63
            output_dmin[i] = max_min / 63

            d_tmp = output_d[i] * ls
            dm_tmp = output_dmin[i] * lm

            all_L = np.round((blocks[i] + dm_tmp.reshape(dm_tmp.shape[0], 1)) / d_tmp.reshape(d_tmp.shape[0], 1)).astype(np.uint8)
            all_L = np.clip(all_L, 0, 15)

            for j in range(QK_K // 64):
                output_qs[i][j] = all_L[j] | (all_L[j + 1] << 4)
        return output_d, output_dmin, output_scale, output_qs