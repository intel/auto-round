# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auto_round_kernel.sparse_attention import (
    _build_block_causal_mask,
    _build_sparge_preprocess_context,
    _fill_block_map_torch,
)
from auto_round_kernel.sparge_preprocess_triton import _fill_block_map_triton


def test_fill_block_map_torch_skips_preselected_entries():
    final_map = torch.tensor([[[[True, False, True, False, False]]]], dtype=torch.bool)
    num_to_select = torch.tensor([[[2]]], dtype=torch.int64)
    sorted_indices = torch.tensor([[[[0, 2, 4, 1, 3]]]], dtype=torch.int64)

    filled = _fill_block_map_torch(final_map, num_to_select, sorted_indices)

    expected = torch.tensor([[[[True, True, True, False, True]]]], dtype=torch.bool)
    assert torch.equal(filled, expected)


def test_fill_block_map_torch_selects_one_new_entry_when_topk_is_zero():
    final_map = torch.tensor([[[[True, False, False, False]]]], dtype=torch.bool)
    num_to_select = torch.tensor([[[0]]], dtype=torch.int64)
    sorted_indices = torch.tensor([[[[0, 2, 1, 3]]]], dtype=torch.int64)

    filled = _fill_block_map_torch(final_map, num_to_select, sorted_indices)

    expected = torch.tensor([[[[True, False, True, False]]]], dtype=torch.bool)
    assert torch.equal(filled, expected)


@pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="XPU not available",
)
def test_fill_block_map_triton_matches_torch_when_rows_have_preselected_entries():
    final_map = torch.tensor([[[[True, False, True, False, False]]]], dtype=torch.bool, device="xpu")
    num_to_select = torch.tensor([[[2]]], dtype=torch.int64, device="xpu")
    sorted_indices = torch.tensor([[[[0, 2, 4, 1, 3]]]], dtype=torch.int64, device="xpu")

    triton_filled = _fill_block_map_triton(final_map, num_to_select, sorted_indices)
    torch_filled = _fill_block_map_torch(final_map.cpu(), num_to_select.cpu(), sorted_indices.cpu()).to("xpu")
    torch.xpu.synchronize()

    assert torch.equal(triton_filled, torch_filled)


def test_build_block_causal_mask_supports_q64_k128_geometry():
    mask = _build_block_causal_mask(
        num_q_tiles=4,
        num_k_blocks=4,
        q_route_block_tokens=64,
        k_route_block_tokens=128,
        device=torch.device("cpu"),
    )

    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, False, False, False],
            [True, True, False, False],
            [True, True, False, False],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected)


@pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="XPU not available",
)
def test_preprocess_context_uses_wan_head_dim_128_routing_geometry():
    q = torch.randn((1, 2, 256, 128), device="xpu", dtype=torch.float16)
    k = torch.randn((1, 2, 256, 128), device="xpu", dtype=torch.float16)

    ctx = _build_sparge_preprocess_context(
        q,
        k,
        is_causal=False,
        smooth_k=True,
        simthreshd1=-0.1,
        topk=0.5,
        attention_sink=False,
        quant_block_size=64,
        tensor_layout="HND",
    )

    assert ctx.q_route_block_tokens == 64
    assert ctx.k_route_block_tokens == 128
    assert ctx.q_blocks_per_tile == 1
    assert ctx.k_blocks_per_tile == 2
    assert ctx.query_tile_tokens == 64
