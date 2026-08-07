import torch

from auto_round.experimental.utils import fp8_qdq


def test_per_head_fp8_qdq_preserves_mha_head_axis():
    tensor = torch.randn(2, 4, 3, 8)

    qdq_tensor, scale = fp8_qdq(tensor, granularity="head")

    assert qdq_tensor.shape == tensor.shape
    assert scale.shape == torch.Size([4])


def test_per_head_fp8_qdq_preserves_gqa_kv_head_axis():
    tensor = torch.randn(2, 2, 3, 8)

    qdq_tensor, scale = fp8_qdq(tensor, granularity="head")

    assert qdq_tensor.shape == tensor.shape
    assert scale.shape == torch.Size([2])
