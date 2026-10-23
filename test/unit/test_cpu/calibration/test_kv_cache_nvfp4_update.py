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


class ObservedAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_idx = 0
        self.head_dim = 16
        self.k_proj = torch.nn.Linear(16, 16, bias=False)
        self.v_proj = torch.nn.Linear(16, 16, bias=False)

    def forward(self, hidden_states, past_key_values=None, use_cache=False):
        key_states = self.k_proj(hidden_states).unsqueeze(1)
        value_states = self.v_proj(hidden_states).unsqueeze(1)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        query_states = hidden_states.unsqueeze(1)
        return torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states)


def test_nvfp4_observer_preserves_optimization_trajectory():
    import copy
    from contextlib import nullcontext

    from auto_round.experimental.kv_cache import kvcache_quant_context

    torch.manual_seed(42)
    baseline = ObservedAttention()
    observed = copy.deepcopy(baseline)
    inputs = [torch.randn(2, 4, 16) for _ in range(3)]
    targets = [torch.randn(2, 1, 4, 16) for _ in inputs]
    trajectories = []
    for model, context in ((baseline, nullcontext()), (observed, kvcache_quant_context(observed, "nvfp4"))):
        trajectory = []
        parameters = [model.k_proj.weight, model.v_proj.weight]
        with context:
            for hidden_states, target in zip(inputs, targets):
                model.zero_grad(set_to_none=True)
                output = model(hidden_states, past_key_values=None)
                loss = torch.nn.functional.mse_loss(output, target)
                loss.backward()
                gradients = [parameter.grad.detach().clone() for parameter in parameters]
                with torch.no_grad():
                    for parameter in parameters:
                        parameter.sub_(0.01 * parameter.grad.sign())
                trajectory.append((output.detach(), loss.detach(), gradients))
        trajectories.append(trajectory)
    for expected, actual in zip(*trajectories):
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        for expected_gradient, actual_gradient in zip(expected[2], actual[2]):
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    for name, expected in baseline.named_parameters():
        torch.testing.assert_close(observed.get_parameter(name), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["nvfp4", "fp8"])
def test_kv_context_removes_its_hooks(dtype):
    from auto_round.experimental.kv_cache import kvcache_quant_context

    model = ObservedAttention()
    inputs = torch.randn(1, 4, 16)
    expected = model(inputs, past_key_values=None)
    with kvcache_quant_context(model, dtype):
        model(inputs, past_key_values=None)
    actual = model(inputs, past_key_values=None)
    torch.testing.assert_close(actual, expected)
    assert not model._forward_pre_hooks
    assert not model._forward_hooks


def test_kv_context_does_not_reuse_previous_amax():
    from auto_round.experimental.kv_cache import kvcache_quant_context

    model = ObservedAttention()
    with kvcache_quant_context(model, "nvfp4"):
        model.kv_cache.update(torch.full((1, 1, 1, 16), 100.0), torch.ones(1, 1, 1, 16), 0)
    with kvcache_quant_context(model, "nvfp4"):
        model.kv_cache.update(torch.ones(1, 1, 1, 16), torch.ones(1, 1, 1, 16), 0)
        assert model.kv_cache.k_amax[0] == 1.0


def test_kv_context_cleans_up_after_exception():
    from auto_round.experimental.kv_cache import kvcache_quant_context

    model = ObservedAttention()
    with pytest.raises(RuntimeError, match="calibration failed"), kvcache_quant_context(model, "nvfp4"):
        raise RuntimeError("calibration failed")
    assert not hasattr(model, "kv_cache")
    assert not model._forward_pre_hooks
    assert not model._forward_hooks
    model(torch.randn(1, 4, 16), past_key_values=None)


def test_kv_context_preserves_unrelated_hooks():
    from auto_round.experimental.kv_cache import kvcache_quant_context

    model = ObservedAttention()
    calls = []
    handle = model.register_forward_hook(lambda module, args, output: calls.append(True))
    with kvcache_quant_context(model, "nvfp4"):
        model(torch.randn(1, 4, 16), past_key_values=None)
    model(torch.randn(1, 4, 16), past_key_values=None)
    assert calls == [True, True]
    assert list(model._forward_hooks) == [handle.id]
    handle.remove()


@pytest.mark.parametrize("granularity", ["tensor", "head"])
def test_fp8_update_quantizes_and_keeps_last_batch_scale(granularity):
    cache = QuantizedKVParameterCache(dtype="fp8", granularity=granularity)
    key_states = torch.linspace(-3.1, 2.7, 64).reshape(1, 2, 2, 16)
    value_states = key_states * 2.7
    actual_keys, actual_values = cache.update(key_states, value_states, 0)
    assert not torch.equal(actual_keys, key_states)
    assert not torch.equal(actual_values, value_states)
    initial_k_scale = cache.k_scales[0].clone()
    initial_v_scale = cache.v_scales[0].clone()
    cache.update(key_states / 4, value_states / 4, 0)
    torch.testing.assert_close(cache.k_scales[0], initial_k_scale / 4)
    torch.testing.assert_close(cache.v_scales[0], initial_v_scale / 4)


@pytest.mark.parametrize("scheme", ["W4A16", "NVFP4"])
def test_nvfp4_observer_preserves_autoround_tuned_weights(scheme):
    import copy
    from contextlib import nullcontext

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from auto_round import AutoRound
    from auto_round.experimental.kv_cache import kvcache_quant_context

    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attn_implementation="eager",
    )
    baseline = LlamaForCausalLM(config).eval()
    observed = copy.deepcopy(baseline)
    original_weight = baseline.model.layers[0].self_attn.k_proj.weight.detach().clone()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({str(index): index for index in range(config.vocab_size)}, unk_token="0")),
        unk_token="0",
        pad_token="0",
        eos_token="1",
    )
    dataset = torch.randint(0, config.vocab_size, (2, 8))
    for model, dtype in ((baseline, None), (observed, "nvfp4")):
        compressor = AutoRound(
            model,
            tokenizer=tokenizer,
            scheme=scheme,
            group_size=16,
            iters=3,
            batch_size=1,
            nsamples=2,
            seqlen=8,
            dataset=[{"input_ids": sample.unsqueeze(0).clone()} for sample in dataset],
            device_map="cpu",
            amp=False,
            seed=42,
            static_kv_dtype=dtype,
        )
        context = nullcontext() if dtype is None else kvcache_quant_context(model, dtype)
        with context:
            compressor.quantize()
    for name, expected in baseline.named_parameters():
        torch.testing.assert_close(observed.get_parameter(name), expected, rtol=0, atol=0)
    assert not torch.equal(baseline.model.layers[0].self_attn.k_proj.weight, original_weight)
    for layer in observed.model.layers:
        assert layer.self_attn.k_global_scale.item() > 0
        assert layer.self_attn.v_global_scale.item() > 0
    with torch.no_grad():
        torch.testing.assert_close(observed(dataset).logits, baseline(dataset).logits, rtol=0, atol=0)
