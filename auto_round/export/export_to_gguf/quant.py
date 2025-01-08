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
    "q4_k": (256, 2 + 2 + QK_K // 2 + 12),
}

GGML_QUANT_BLOCK = {}

def register_block(name):
    def register(cls):
        GGML_QUANT_BLOCK[name] = cls
        return cls
    return register


def ggml_quant(data: np.array, ggml_type, scale = None, zp = None):
    block_size, type_size = GGML_QUANT_SIZES[ggml_type]

    data = data.astype(np.float32, copy=False)
    shape = data.shape
    n_blocks = data.size // block_size
    blocks = data.reshape((n_blocks, block_size))
    
    new_data = GGML_QUANT_BLOCK[ggml_type](blocks, scale, zp)
    new_data = new_data.reshape(*shape[:-1], shape[-1] // block_size * type_size)
    return new_data

@register_block("bf16")
def bf16_quant_block(blocks: np.array, scale = None, zp = None):
    n = blocks.view(np.uint32)
    # force nan to quiet
    n = np.where((n & 0x7fffffff) > 0x7f800000, (n & np.uint32(0xffff0000)) | np.uint32(64 << 16), n)
    # round to nearest even
    n = (np.uint64(n) + (0x7fff + ((n >> 16) & 1))) >> 16
    return n.astype(np.uint16).view(np.uint8)

@register_block("q4_0")
def q4_0_quant_block(blocks: np.array, scale = None, zp = None):
    if scale is not None:
        d = scale.reshape((-1,1))
    else:
        imax = abs(blocks).argmax(axis=-1, keepdims=True)
        max = np.take_along_axis(blocks, imax, axis=-1)
        d = max / -8
    with np.errstate(divide="ignore"):
        id = np.where(d == 0, 0, 1 / d)

    qs = np.trunc(
        (np.float64(blocks) * np.float64(id)) + np.float64(8.5),
        dtype=np.float32).astype(np.uint8).clip(0, 15)

    n_blocks = blocks.shape[0]
    block_size = GGML_QUANT_SIZES["q4_0"][0]
    qs = qs.reshape((n_blocks, 2, block_size // 2))
    qs = qs[..., 0, :] | (qs[..., 1, :] << np.uint8(4))

    d = d.astype(np.float16).view(np.uint8)

    return np.concatenate([d, qs], axis=-1)


@register_block("q4_1")
def q4_1_quant_block(blocks: np.array, scale = None, zp = None):
    if scale is not None:
        d = scale.reshape((-1,1))
        min = zp.reshape((-1,1)) * d * -1
    else:
        max = blocks.max(axis=-1, keepdims=True)
        min = blocks.min(axis=-1, keepdims=True)
        d = (max - min) / 15
    with np.errstate(divide="ignore"):
        id = np.where(d == 0, 0, 1 / d)

    qs = np.trunc((blocks - min) * id + np.float32(0.5), dtype=np.float32).astype(np.uint8).clip(0, 15)

    n_blocks = blocks.shape[0]
    block_size = GGML_QUANT_SIZES["q4_1"][0]
    qs = qs.reshape((n_blocks, 2, block_size // 2))
    qs = qs[..., 0, :] | (qs[..., 1, :] << np.uint8(4))

    d = d.astype(np.float16).view(np.uint8)
    m = min.astype(np.float16).view(np.uint8)
    return np.concatenate([d, m, qs], axis=-1)
