# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auto_round_kernel.sparge_preprocess_triton import _fill_block_map_triton
from auto_round_kernel.sparse_attention import (
    _build_block_causal_mask,
    _build_sparge_preprocess_context,
    _fill_block_map_torch,
    _kv_head_index_for_q_heads,
    _validate_gqa_head_config,
)


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


def test_validate_gqa_head_config_accepts_32_over_8() -> None:
    _validate_gqa_head_config(32, 8, op_name="test")


@pytest.mark.parametrize("num_heads_q, num_heads_kv", [(8, 32), (30, 8)])
def test_validate_gqa_head_config_rejects_invalid_ratios(num_heads_q: int, num_heads_kv: int) -> None:
    with pytest.raises(ValueError, match="num_heads_q"):
        _validate_gqa_head_config(num_heads_q, num_heads_kv, op_name="test")


def test_kv_head_index_for_q_heads_maps_32_query_heads_onto_8_kv_heads() -> None:
    indices = _kv_head_index_for_q_heads(32, 8, torch.device("cpu"))

    expected = torch.arange(8, dtype=torch.int64).repeat_interleave(4)
    assert torch.equal(indices.cpu(), expected)


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


@pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="XPU not available",
)
def test_preprocess_context_accepts_explicit_query_tile_tokens_256():
    q = torch.randn((1, 2, 512, 128), device="xpu", dtype=torch.float16)
    k = torch.randn((1, 2, 512, 128), device="xpu", dtype=torch.float16)

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
        query_tile_tokens=256,
    )

    assert ctx.q_route_block_tokens == 256
    assert ctx.q_blocks_per_tile == 4
    assert ctx.num_q_tiles == 2
    assert ctx.query_tile_tokens == 256


@pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="XPU not available",
)
@pytest.mark.parametrize("tensor_layout", ("HND", "NHD"))
def test_preprocess_context_accepts_decoupled_sparse_qtile256_row64k(tensor_layout: str):
    if tensor_layout == "HND":
        q = torch.randn((1, 2, 512, 128), device="xpu", dtype=torch.float16)
        k = torch.randn((1, 2, 512, 128), device="xpu", dtype=torch.float16)
    else:
        q = torch.randn((1, 512, 2, 128), device="xpu", dtype=torch.float16)
        k = torch.randn((1, 512, 2, 128), device="xpu", dtype=torch.float16)

    ctx = _build_sparge_preprocess_context(
        q,
        k,
        is_causal=False,
        smooth_k=True,
        simthreshd1=-0.1,
        topk=0.5,
        attention_sink=False,
        quant_block_size=64,
        tensor_layout=tensor_layout,
        query_tile_tokens=256,
        sparse_q_block_tokens=256,
        sparse_k_block_tokens=64,
    )

    assert ctx.q_route_block_tokens == 256
    assert ctx.k_route_block_tokens == 64
    assert ctx.sparse_q_block_tokens == 256
    assert ctx.sparse_k_block_tokens == 64
    assert ctx.q_blocks_per_tile == 4
    assert ctx.k_blocks_per_tile == 1
    assert ctx.q_sparse_blocks_per_tile == 1
    assert ctx.k_sparse_blocks_per_tile == 1
    assert ctx.num_q_blocks == 8
    assert ctx.num_k_blocks == 8
    assert ctx.num_sparse_q_blocks == 2
    assert ctx.num_sparse_k_blocks == 8


@pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="XPU not available",
)
@pytest.mark.parametrize("tensor_layout", ("HND", "NHD"))
def test_preprocess_context_accepts_gqa_32_8_head_mapping(tensor_layout: str):
    if tensor_layout == "HND":
        q = torch.randn((1, 32, 512, 128), device="xpu", dtype=torch.float16)
        k = torch.randn((1, 8, 512, 128), device="xpu", dtype=torch.float16)
    else:
        q = torch.randn((1, 512, 32, 128), device="xpu", dtype=torch.float16)
        k = torch.randn((1, 512, 8, 128), device="xpu", dtype=torch.float16)

    ctx = _build_sparge_preprocess_context(
        q,
        k,
        is_causal=False,
        smooth_k=True,
        simthreshd1=-0.1,
        topk=0.5,
        attention_sink=False,
        quant_block_size=64,
        tensor_layout=tensor_layout,
        query_tile_tokens=256,
        sparse_q_block_tokens=256,
        sparse_k_block_tokens=64,
    )

    assert ctx.num_heads_q == 32
    assert ctx.num_heads_kv == 8
    assert ctx.q_route_block_tokens == 256
    assert ctx.k_route_block_tokens == 64
    assert ctx.sparse_q_block_tokens == 256
    assert ctx.sparse_k_block_tokens == 64
