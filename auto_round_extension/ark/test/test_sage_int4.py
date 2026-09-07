# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import auto_round_kernel


pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="XPU not available",
)


def _pack_signed_int4(values: torch.Tensor) -> torch.Tensor:
    encoded = values.to(torch.int16) & 0xF
    return (encoded[..., 0::2] | (encoded[..., 1::2] << 4)).to(torch.uint8).contiguous()


def _unpack_signed_int4(packed: torch.Tensor) -> torch.Tensor:
    values = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2).to(torch.int16)
    return torch.where(values >= 8, values - 16, values).to(torch.int8)


def _expand_block_scales(scales: torch.Tensor, seq_len: int, block_size: int) -> torch.Tensor:
    expanded = scales.squeeze(-1).repeat_interleave(block_size, dim=-1)
    return expanded[..., :seq_len].unsqueeze(-1)


@pytest.mark.parametrize("head_dim,seq_len", [(64, 128), (128, 256)])
@pytest.mark.parametrize(
    "batch,heads_q,heads_kv,constant_scales",
    [(1, 2, 2, False), (2, 1, 1, True), (2, 4, 2, False)],
)
def test_sage_s4_matches_unpacked_int4_reference(head_dim, seq_len, batch, heads_q, heads_kv, constant_scales):
    torch.manual_seed(4104 + head_dim)
    block_size = 64
    blocks = (seq_len + block_size - 1) // block_size
    q_logical = torch.randint(-8, 8, (batch, heads_q, seq_len, head_dim), dtype=torch.int8)
    k_logical = torch.randint(-8, 8, (batch, heads_kv, seq_len, head_dim), dtype=torch.int8)
    q_packed = _pack_signed_int4(q_logical).to("xpu")
    k_packed = _pack_signed_int4(k_logical).to("xpu")
    v = torch.randn(batch, heads_kv, seq_len, head_dim, dtype=torch.float16, device="xpu")
    if constant_scales:
        qscale = torch.full((batch, heads_q, blocks, 1), 0.125, dtype=torch.float32, device="xpu")
        kscale = torch.full((batch, heads_kv, blocks, 1), 0.09375, dtype=torch.float32, device="xpu")
    else:
        qscale = torch.linspace(0.0625, 0.125, batch * heads_q * blocks, dtype=torch.float32, device="xpu").reshape(
            batch, heads_q, blocks, 1
        )
        kscale = torch.linspace(
            0.09375, 0.15625, batch * heads_kv * blocks, dtype=torch.float32, device="xpu"
        ).reshape(batch, heads_kv, blocks, 1)
    scale = 1.0 / math.sqrt(head_dim)

    output = auto_round_kernel.sage_s4(
        q_packed,
        k_packed,
        v,
        scale=scale,
        enable_gqa=heads_q != heads_kv,
        quant_block_size=block_size,
        qscale=qscale,
        kscale=kscale,
    )
    torch.xpu.synchronize()

    q_scale_expanded = _expand_block_scales(qscale.cpu(), seq_len, block_size)
    k_scale_expanded = _expand_block_scales(kscale.cpu(), seq_len, block_size)
    q_float = q_logical.float() * q_scale_expanded
    k_float = k_logical.float() * k_scale_expanded
    if heads_q != heads_kv:
        repeat_factor = heads_q // heads_kv
        k_float = k_float.repeat_interleave(repeat_factor, dim=1)
        v_reference = v.cpu().repeat_interleave(repeat_factor, dim=1)
    else:
        v_reference = v.cpu()
    scores = torch.matmul(q_float, k_float.transpose(-2, -1)) * scale
    reference = torch.matmul(torch.softmax(scores, dim=-1), v_reference.float()).to(torch.float16)

    torch.testing.assert_close(output.cpu(), reference, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(_unpack_signed_int4(q_packed.cpu()), q_logical, atol=0, rtol=0)
    torch.testing.assert_close(_unpack_signed_int4(k_packed.cpu()), k_logical, atol=0, rtol=0)
