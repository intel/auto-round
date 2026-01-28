# Copyright (c) 2026 Intel Corporation
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

from random import randint

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32 * 32}),
        triton.Config({"BLOCK_SIZE": 64 * 32}),
        triton.Config({"BLOCK_SIZE": 128 * 32}),
        triton.Config({"BLOCK_SIZE": 256 * 32}),
        triton.Config({"BLOCK_SIZE": 512 * 32}),
    ],
    key=[],
)
@triton.jit
def mxfp4_forward_kernel(
    x_ptr,
    hadamard_matrix_ptr,
    output_ptr,
    clip_mask_ptr,
    n_elements: tl.constexpr,
    hadamard_dim: tl.constexpr,
    group_size: tl.constexpr,
    gaussian_scale: tl.constexpr,
    quest: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets_hadamard = tl.arange(0, hadamard_dim * hadamard_dim)
    hadamard_matrix = tl.load(hadamard_matrix_ptr + offsets_hadamard).reshape(hadamard_dim, hadamard_dim)

    # load x
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_flat = tl.load(x_ptr + offsets, mask=mask)

    # hadamard transform
    x = tl.reshape(x_flat, (BLOCK_SIZE // hadamard_dim, hadamard_dim))
    x_had = tl.dot(x, hadamard_matrix)

    # group
    x_had_grouped = tl.reshape(x_had, (BLOCK_SIZE // group_size, group_size))

    # scale
    # quest=True: per-group Gaussian-based scale = gaussian_scale * std
    # quest=False: per-group max-abs-based scale, adjusted to FP4 range
    if quest:
        mean_squared = tl.sum(x_had_grouped * x_had_grouped, axis=-1, keep_dims=True) / group_size
        mean = tl.sum(x_had_grouped, axis=-1, keep_dims=True) / group_size
        std = tl.sqrt(mean_squared - mean * mean)
        scales = gaussian_scale * std + 1e-8
        shared_exps = tl.exp2(tl.floor(tl.log2(scales)))
        x_had_scaled = x_had_grouped / shared_exps
    else:
        scales = tl.max(tl.abs(x_had_grouped), axis=-1, keep_dims=True)
        shared_exps = tl.exp2(tl.floor(tl.log2(scales)) - 2) / (3 / 4)
        x_had_scaled = x_had_grouped / shared_exps

    # quantize
    # Map abs(x) to FP4 levels {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    x_had_scaled_abs = tl.abs(x_had_scaled)
    x_had_scaled_sign = tl.where(
        x_had_scaled > 0,
        1,
        -1,
    )

    x_fp4 = (
        tl.where(
            x_had_scaled_abs > 5,
            6,
            tl.where(
                x_had_scaled_abs > 3.5,
                4,
                tl.where(
                    x_had_scaled_abs > 2.5,
                    3,
                    tl.where(
                        x_had_scaled_abs > 1.75,
                        2,
                        tl.where(
                            x_had_scaled_abs > 1.25,
                            1.5,
                            tl.where(
                                x_had_scaled_abs > 0.75,
                                1,
                                tl.where(
                                    x_had_scaled_abs > 0.25,
                                    0.5,
                                    0,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        * x_had_scaled_sign
    )
    if clip_mask_ptr is not None:
        tl.store(
            clip_mask_ptr + offsets,
            tl.reshape(x_had_scaled_abs < 6, (BLOCK_SIZE,)),
            mask=mask,
        )

    # dequantize
    x_dequantized = x_fp4 * shared_exps

    # Reshape back to flat form for storage
    x_dequantized_flat = tl.reshape(x_dequantized, (BLOCK_SIZE,))

    # store
    tl.store(output_ptr + offsets, x_dequantized_flat, mask=mask)


@torch.compiler.disable()
def mxfp4_forward_kernel_wrapper(
    x,
    hadamard_matrix,
    return_clip_mask=False,
    quest=False,
    gaussian_scale=3 / 4,
):
    """
    Apply Hadamard transform + group-wise FP4 quantize/dequantize on x.

    Note:
    The output is still in the Hadamard-transformed space (no inverse Hadamard is applied).
    """
    # Pick a device — we require CUDA
    device = x.device
    if not device.type == "cuda":
        # Either move to cuda or raise, depending on your design
        device = torch.device("cuda")
        x = x.to(device)

    # Ensure hadamard_matrix is on the same CUDA device
    if hadamard_matrix.device != device:
        hadamard_matrix = hadamard_matrix.to(device)

    # Make sure inputs are contiguous
    x = x.contiguous()
    hadamard_matrix = hadamard_matrix.contiguous()

    # Create output tensors on CUDA
    output = torch.empty_like(x, device=device)
    if return_clip_mask:
        clip_mask = torch.empty_like(x, dtype=torch.bool, device=device).contiguous()
    else:
        clip_mask = None

    # Get total number of elements and calculate grid for launching the kernel
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch kernel – no need for `with torch.device(...)`
    mxfp4_forward_kernel[grid](
        x_ptr=x,
        hadamard_matrix_ptr=hadamard_matrix,
        output_ptr=output,
        clip_mask_ptr=clip_mask,
        n_elements=n_elements,
        hadamard_dim=hadamard_matrix.shape[-1],
        group_size=32,
        gaussian_scale=gaussian_scale,
        quest=quest,
    )

    return output, clip_mask
