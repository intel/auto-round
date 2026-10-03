"""Regression test for QuantizedKVParameterCache.update (NVFP4 path).

A previous revision passed ``key_states`` to both the KEY and the VALUE
``_nvfp4_quant_dequant`` calls, so ``v_amax`` was a copy of ``k_amax`` and
the exported ``v_global_scale`` silently equaled ``k_global_scale`` for
every layer.  The cache must track K and V magnitudes separately and QDQ
each side from its own tensor.
"""

import pytest
import torch

from auto_round.experimental.kv_cache import QuantizedKVParameterCache


def _reset_singleton():
    # ``reset`` is an instance method; clear the class state directly.
    QuantizedKVParameterCache._instance = None
    QuantizedKVParameterCache._initialized = False


@pytest.fixture(autouse=True)
def _fresh_cache():
    _reset_singleton()
    yield
    _reset_singleton()


def test_update_tracks_k_and_v_amax_separately():
    cache = QuantizedKVParameterCache(dtype="nvfp4")
    assert cache.is_nvfp4

    torch.manual_seed(0)
    # Deliberately different magnitudes per side so a K/V mixup cannot hide.
    key_states = torch.randn(2, 4, 64, 32) * 3.0
    value_states = torch.randn(2, 4, 64, 32) * 11.0
    k_amax = key_states.abs().max().item()
    v_amax = value_states.abs().max().item()
    assert v_amax > 2 * k_amax  # sanity: the two sides are distinguishable

    qk, qv = cache.update(key_states, value_states, layer_idx=0)

    assert cache.k_amax[0] == pytest.approx(k_amax)
    assert cache.v_amax[0] == pytest.approx(v_amax)

    # Calibration is observe-only: K/V pass through unmodified; the exported
    # static scale is derived from the final amax (see the convention test
    # below).  This mirrors llm-compressor's observer-based KV calibration.
    assert torch.equal(qk, key_states)
    assert torch.equal(qv, value_states)


def test_update_running_max_per_side():
    cache = QuantizedKVParameterCache(dtype="nvfp4")

    k_small = torch.full((1, 1, 16, 16), 1.0)
    v_big = torch.full((1, 1, 16, 16), 8.0)
    cache.update(k_small, v_big, layer_idx=2)
    assert cache.k_amax[2] == pytest.approx(1.0)
    assert cache.v_amax[2] == pytest.approx(8.0)

    # A later larger K must not leak into V, and vice versa.
    k_big = torch.full((1, 1, 16, 16), 16.0)
    v_small = torch.full((1, 1, 16, 16), 0.5)
    cache.update(k_big, v_small, layer_idx=2)
    assert cache.k_amax[2] == pytest.approx(16.0)
    assert cache.v_amax[2] == pytest.approx(8.0)


def test_stored_nvfp4_scale_uses_vllm_dequant_multiplier_convention():
    """The checkpoint scale must be amax / 2688 (vLLM dequant multiplier).

    vLLM's store kernel computes ``global_scale = 1 / k_scale`` and
    ``sf = global_scale * block_max / 6``.  Storing the weight-style
    ``2688 / amax`` instead makes the runtime fp8 block scale
    ``block_max * amax / (6 * 2688)`` underflow e4m3 (min subnormal 2**-9)
    for typical activation amax values, zeroing the served KV cache.
    """
    from auto_round.experimental.kv_cache import _nvfp4_global_scale

    device = torch.device("cpu")
    for amax in (0.5, 2.0, 282.0):
        scale = _nvfp4_global_scale(amax, device)
        assert scale.shape == (1,) and scale.dtype == torch.float32
        # Reciprocal of the weight-style global scale (2688 / amax).
        assert scale[0] == pytest.approx(amax / (448.0 * 6.0), rel=1e-6)
        assert 1.0 / scale[0] == pytest.approx(448.0 * 6.0 / amax, rel=1e-5)

        # Every runtime fp8 block scale, from a block at 1% of the calibrated
        # amax up to the worst-case block, must stay representable in e4m3.
        g_rt = 1.0 / scale[0]
        block_maxes = torch.linspace(0.01 * amax, amax, 8)
        sf = g_rt * block_maxes / 6.0
        assert sf.max().item() <= 448.0 + 1e-4
        # e4m3 min subnormal is 2**-9 (~1.95e-3)
        assert sf.min().item() >= 2.0**-9


def _make_fake_attention(head_dim: int, num_heads: int = 4):
    import torch.nn as nn

    class FakeAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_idx = 0
            self.head_dim = head_dim
            self.k_proj = nn.Linear(head_dim, num_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(head_dim, num_heads * head_dim, bias=False)

    return FakeAttention()


def test_nvfp4_kv_incompatible_head_dim_fails_early():
    """A single unsupported layer must fail loudly, not disable all layers."""
    from auto_round.experimental.kv_cache import initialize_quantized_kv_cache

    module = _make_fake_attention(head_dim=88)
    with pytest.raises(ValueError, match="divisible by 16"):
        initialize_quantized_kv_cache(module, dtype="nvfp4")
    assert not hasattr(module, "kv_cache")


def test_nvfp4_kv_incompatible_head_dim_raises_at_update():
    cache = QuantizedKVParameterCache(dtype="nvfp4")
    bad = torch.randn(1, 1, 16, 88)
    with pytest.raises(ValueError, match="divisible by 16"):
        cache.update(bad, bad.clone(), layer_idx=0)
