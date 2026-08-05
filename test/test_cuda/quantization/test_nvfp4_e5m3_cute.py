import pytest
import torch

from auto_round.data_type.nvfp import cast_to_fp4, fp4_v2
from auto_round_extension.cuda.cute_nvfp4_e5m3 import can_use_cute_fp4_v2_qdq, try_cute_fp4_v2_qdq


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("dtype", "atol"),
    [(torch.float32, 1e-6), (torch.float16, 2e-3), (torch.bfloat16, 2e-2)],
)
def test_cute_fp4_v2_qdq_matches_reference(dtype, atol):
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("CuTe QDQ requires SM80 or newer")

    activation = torch.randn(32, 16, device="cuda", dtype=dtype)
    if not can_use_cute_fp4_v2_qdq(activation, 16):
        pytest.skip("CuTe DSL is not available")

    expected, _, _ = fp4_v2(activation.float(), bits=4, group_size=16)
    actual = try_cute_fp4_v2_qdq(activation, 16)

    assert actual is not None
    torch.testing.assert_close(actual, expected.to(dtype), rtol=0, atol=atol)


def test_fp4_v2_uses_direct_scale_at_rounding_boundary():
    activation = torch.tensor([[1.25] + [1.0] * 15], dtype=torch.float32)

    actual, scale, _ = fp4_v2(activation, bits=4, group_size=16)
    expected = cast_to_fp4(activation / scale) * scale

    torch.testing.assert_close(actual, expected)


def test_fp4_v2_direct_scale_handles_zero_group():
    activation = torch.zeros(1, 16, dtype=torch.float32)

    actual, scale, _ = fp4_v2(activation, bits=4, group_size=16)

    assert torch.equal(actual, activation)
    assert torch.equal(scale, torch.zeros_like(scale))
