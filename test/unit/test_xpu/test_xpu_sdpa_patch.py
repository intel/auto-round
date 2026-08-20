import pytest
import torch
import torch.nn.functional as F

from auto_round.utils.device import patch_xpu_sdpa_drop_causal_mask


@pytest.mark.skipif(not (hasattr(torch, "xpu") and torch.xpu.is_available()), reason="XPU not available")
def test_patch_xpu_sdpa_drop_causal_mask():
    """Test that patch_xpu_sdpa_drop_causal_mask correctly identifies and replaces causal masks on XPU."""
    patch_xpu_sdpa_drop_causal_mask()

    s = 128
    query = torch.randn(1, 8, s, 64, device="xpu", dtype=torch.float16)
    key = torch.randn(1, 8, s, 64, device="xpu", dtype=torch.float16)
    value = torch.randn(1, 8, s, 64, device="xpu", dtype=torch.float16)

    # Create a pure causal mask (HF style: small values for non-masked, -inf for masked)
    mask = torch.zeros(1, 1, s, s, device="xpu", dtype=torch.float16)
    tri_up = torch.triu(torch.ones(s, s, dtype=torch.bool, device="xpu"), 1)
    mask.reshape(-1, s, s)[0][tri_up] = float("-inf")

    # We can't easily check internal calls without complex mocking,
    # but we can verify it doesn't crash and produces same results as standard causal=True

    # 1. Test with causal mask (should be patched to is_causal=True internally)
    output_with_mask = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, is_causal=False)

    # 2. Test with is_causal=True directly
    output_is_causal = F.scaled_dot_product_attention(query, key, value, attn_mask=None, is_causal=True)

    torch.xpu.synchronize()
    assert torch.allclose(output_with_mask, output_is_causal, atol=1e-3)

    # 3. Test with a non-causal mask (should NOT be patched)
    non_causal_mask = torch.zeros(1, 1, s, s, device="xpu", dtype=torch.float16)
    non_causal_mask.reshape(-1, s, s)[0][:, 0] = float("-inf")  # Mask first column

    output_non_causal = F.scaled_dot_product_attention(query, key, value, attn_mask=non_causal_mask, is_causal=False)
    # This should be different from a simple causal mask
    assert not torch.allclose(output_with_mask, output_non_causal, atol=1e-3)


if __name__ == "__main__":
    # For manual testing
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        test_patch_xpu_sdpa_drop_causal_mask()
        print("XPU SDPA Patch Test Passed!")
    else:
        print("XPU not available, skipping test.")
