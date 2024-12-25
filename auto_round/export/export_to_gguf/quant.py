import torch
import numpy as np

QK_K = 256
GGML_QUANT_SIZES = {
    "q4_0": (32, 2 + 16),
    "q4_1": (32, 2 + 2 + 16),
    "q4_k": (256, 2 + 2 + QK_K // 2 + 12),
}

def quantize_q4_0(data: np.array, scale = None):
    block_size, type_size = GGML_QUANT_SIZES['q4_0']

    data = data.astype(np.float32, copy=False)
    shape = data.shape
    n_blocks = data.size // block_size
    blocks = data.reshape((n_blocks, block_size))


    if scale is not None:
        d = scale
    else:
        imax = abs(blocks).argmax(axis=-1, keepdims=True)
        max = np.take_along_axis(blocks, imax, axis=-1)
        d = max / -8
    with np.errstate(divide="ignore"):
        id = np.where(d == 0, 0, 1 / d)

    qs = np.trunc((np.float64(blocks) * np.float64(id)) + np.float64(8.5), dtype=np.float32).astype(np.uint8).clip(0, 15)

    qs = qs.reshape((n_blocks, 2, block_size // 2))
    qs = qs[..., 0, :] | (qs[..., 1, :] << np.uint8(4))

    d = d.astype(np.float16).view(np.uint8)

    out = np.concatenate([d, qs], axis=-1)
    out = out.reshape(*shape[:-1], shape[-1] // block_size * type_size)
    return out


def quantize_q4_1(data: np.array, scale = None):
    block_size, type_size = GGML_QUANT_SIZES['q4_1']

    data = data.astype(np.float32, copy=False)
    shape = data.shape
    n_blocks = data.size // block_size
    blocks = data.reshape((n_blocks, block_size))

    if scale is not None:
        d = scale
    else:
        max = blocks.max(axis=-1, keepdims=True)
        min = blocks.min(axis=-1, keepdims=True)
        d = (max - min) / 15
    with np.errstate(divide="ignore"):
        id = np.where(d == 0, 0, 1 / d)

    qs = np.trunc((blocks - min) * id + np.float32(0.5), dtype=np.float32).astype(np.uint8).clip(0, 15)

    qs = qs.reshape((n_blocks, 2, block_size // 2))
    qs = qs[..., 0, :] | (qs[..., 1, :] << np.uint8(4))

    d = d.astype(np.float16).view(np.uint8)
    m = min.astype(np.float16).view(np.uint8)

    out = np.concatenate([d, m, qs], axis=-1)
    out = out.reshape(*shape[:-1], shape[-1] // block_size * type_size)
    return out