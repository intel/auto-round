import pytest
import torch

from auto_round.data_type.nvfp import cast_to_fp4, e5m3_to_float_tensor, nvfp4_v2
from auto_round.experimental.qmodules.fp4_utils import unpack_fp4_from_uint8
from auto_round_extension.cuda.cute_nvfp4_e5m3 import (
    can_use_cute_nvfp4_v2_qdq,
    try_cute_nvfp4_e5m3_weight_dq,
    try_cute_nvfp4_v2_qdq,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("dtype", "atol"),
    [(torch.float32, 1e-6), (torch.float16, 2e-3), (torch.bfloat16, 2e-2)],
)
def test_cute_nvfp4_v2_qdq_matches_reference(dtype, atol):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("CuTe QDQ requires SM80 or newer")

    activation = torch.randn(32, 16, device="cuda", dtype=dtype)
    if not can_use_cute_nvfp4_v2_qdq(activation, 16):
        pytest.skip("CuTe DSL is not available")

    expected, _, _ = nvfp4_v2(activation.float(), bits=4, group_size=16)
    actual = try_cute_nvfp4_v2_qdq(activation, 16)

    assert actual is not None
    torch.testing.assert_close(actual, expected.to(dtype), rtol=0, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cute_nvfp4_e5m3_weight_dq_matches_reference_for_multiple_groups(dtype):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("CuTe weight DQ requires SM80 or newer")

    torch.manual_seed(42)
    weight_packed = torch.randint(0, 256, (37, 64), device="cuda", dtype=torch.uint8)
    weight_scale = torch.randint(0, 120, (37, 8), device="cuda", dtype=torch.uint8)
    actual = try_cute_nvfp4_e5m3_weight_dq(weight_packed, weight_scale, dtype)
    if actual is None:
        pytest.skip("CuTe DSL is not available")

    unpacked = unpack_fp4_from_uint8(weight_packed, 37, 128, dtype=dtype).float()
    expected = (unpacked.reshape(-1, 16) * e5m3_to_float_tensor(weight_scale).reshape(-1, 1)).reshape(37, 128).to(dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nvfp4_v2_uses_direct_scale_at_rounding_boundary():
    activation = torch.tensor([[11.625] + [1.0] * 15], dtype=torch.float32)

    actual, scale, _ = nvfp4_v2(activation, bits=4, group_size=16)
    expected = cast_to_fp4(activation / scale) * scale

    assert scale.item() == 2.0
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cute_nvfp4_v2_qdq_carries_mantissa_round_overflow():
    activation = torch.tensor([[11.625] + [1.0] * 15], device="cuda", dtype=torch.float32)
    if not can_use_cute_nvfp4_v2_qdq(activation, 16):
        pytest.skip("CuTe DSL is not available")

    expected, _, _ = nvfp4_v2(activation, bits=4, group_size=16)
    actual = try_cute_nvfp4_v2_qdq(activation, 16)

    assert actual is not None
    torch.testing.assert_close(actual, expected)


def test_nvfp4_v2_direct_scale_handles_zero_group():
    activation = torch.zeros(1, 16, dtype=torch.float32)

    actual, scale, _ = nvfp4_v2(activation, bits=4, group_size=16)

    assert torch.equal(actual, activation)
    assert torch.equal(scale, torch.zeros_like(scale))
