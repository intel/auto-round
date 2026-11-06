"""Regression tests for the shared KV observer/QDQ lifecycle.

K and V statistics must stay independent. Reference forwards only observe
their magnitudes; tuning forwards QDQ each side using its own fixed amax.
"""

import pytest
import torch

from auto_round.experimental.kv_cache import QuantizedKVParameterCache


def _reset_singleton():
    QuantizedKVParameterCache._instance = None
    QuantizedKVParameterCache._initialized = False


@pytest.fixture(autouse=True)
def _fresh_cache():
    _reset_singleton()
    yield
    _reset_singleton()


def test_update_observes_reference_then_qdqs_with_fixed_per_side_amax():
    from auto_round.data_type.nvfp import nv_fp4_with_static_gs

    cache = QuantizedKVParameterCache(dtype="nvfp4")
    assert cache.is_nvfp4

    torch.manual_seed(0)
    # Deliberately different magnitudes per side so a K/V mixup cannot hide.
    key_states = torch.randn(2, 4, 64, 32) * 3.0
    value_states = torch.randn(2, 4, 64, 32) * 11.0
    k_amax = key_states.abs().max().item()
    v_amax = value_states.abs().max().item()
    assert v_amax > 2 * k_amax  # sanity: the two sides are distinguishable

    ref_keys, ref_values = cache.update(key_states, value_states, layer_idx=0)

    assert cache.k_amax[0] == pytest.approx(k_amax)
    assert cache.v_amax[0] == pytest.approx(v_amax)
    assert isinstance(cache.k_amax[0], torch.Tensor) and not cache.k_amax[0].requires_grad
    assert isinstance(cache.v_amax[0], torch.Tensor) and not cache.v_amax[0].requires_grad
    assert torch.equal(ref_keys, key_states)
    assert torch.equal(ref_values, value_states)

    tune_keys = (key_states * 1.7).requires_grad_()
    tune_values = (value_states * 0.6).requires_grad_()
    with cache.kv_qdq_context():
        qdq_keys, qdq_values = cache.update(tune_keys, tune_values, layer_idx=0)

    expected_keys = nv_fp4_with_static_gs(tune_keys, tensor_max=k_amax)[0]
    expected_values = nv_fp4_with_static_gs(tune_values, tensor_max=v_amax)[0]
    torch.testing.assert_close(qdq_keys, expected_keys)
    torch.testing.assert_close(qdq_values, expected_values)
    assert not torch.equal(qdq_keys, tune_keys)
    assert not torch.equal(qdq_values, tune_values)
    assert cache.k_amax[0] == pytest.approx(k_amax)
    assert cache.v_amax[0] == pytest.approx(v_amax)

    (qdq_keys.float().square().mean() + qdq_values.float().square().mean()).backward()
    assert tune_keys.grad is not None and torch.isfinite(tune_keys.grad).all()
    assert tune_values.grad is not None and torch.isfinite(tune_values.grad).all()
    assert not cache._qdq_enabled


def test_nvfp4_tuning_requires_calibrated_amax():
    cache = QuantizedKVParameterCache(dtype="nvfp4")
    with cache.kv_qdq_context(), pytest.raises(RuntimeError, match="calibrated KEY amax"):
        cache.update(torch.ones(1, 1, 1, 16), torch.ones(1, 1, 1, 16), layer_idx=0)


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


class MultiHeadObservedAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_idx = 0
        self.head_dim = 16
        self.num_heads = 2
        self.q_proj = torch.nn.Linear(32, 32, bias=False)
        self.k_proj = torch.nn.Linear(32, 32, bias=False)
        self.v_proj = torch.nn.Linear(32, 32, bias=False)
        with torch.no_grad():
            self.k_proj.weight[:16].mul_(0.1)
            self.k_proj.weight[16:].mul_(3.0)
            self.v_proj.weight[:16].mul_(0.25)
            self.v_proj.weight[16:].mul_(2.0)

    def forward(self, hidden_states, past_key_values=None, use_cache=False):
        batch, seq_len, _ = hidden_states.shape

        def split_heads(projected):
            return projected.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        query_states = split_heads(self.q_proj(hidden_states))
        key_states = split_heads(self.k_proj(hidden_states))
        value_states = split_heads(self.v_proj(hidden_states))
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        output = torch.nn.functional.scaled_dot_product_attention(query_states, key_states, value_states)
        return output.transpose(1, 2).reshape(batch, seq_len, -1)


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
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
        for expected_gradient, actual_gradient in zip(expected[2], actual[2]):
            torch.testing.assert_close(actual_gradient, expected_gradient)
            torch.testing.assert_close(actual_gradient.sign(), expected_gradient.sign(), rtol=0, atol=0)
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
    handle.remove()


@pytest.mark.parametrize("granularity", ["tensor", "head"])
def test_fp8_granularity_preserves_reference_and_quantizes_tuning_attention(granularity):
    from auto_round.experimental.kv_cache import kvcache_quant_context

    torch.manual_seed(7)
    model = MultiHeadObservedAttention().eval()
    hidden_states = torch.randn(2, 8, 32)
    with torch.no_grad():
        reference_output = model(hidden_states, past_key_values=None)

    with kvcache_quant_context(model, "fp8", static_kv_granularity=granularity):
        with torch.no_grad():
            observed_output = model(hidden_states, past_key_values=None)
        torch.testing.assert_close(observed_output, reference_output, rtol=0, atol=0)

        expected_scale_shape = (model.num_heads,) if granularity == "head" else (1,)
        assert model.k_scale.shape == expected_scale_shape
        assert model.v_scale.shape == expected_scale_shape
        if granularity == "head":
            assert not torch.allclose(model.k_scale[0], model.k_scale[1])
            assert not torch.allclose(model.v_scale[0], model.v_scale[1])

        with model.kv_cache.kv_qdq_context(), torch.no_grad():
            tuned_output = model(hidden_states, past_key_values=None)

        assert torch.isfinite(tuned_output).all()
        assert not torch.allclose(tuned_output, reference_output)

    with torch.no_grad():
        output_after_cleanup = model(hidden_states, past_key_values=None)
    torch.testing.assert_close(output_after_cleanup, reference_output, rtol=0, atol=0)


@pytest.mark.parametrize("scheme", ["W4A16", "NVFP4"])
@pytest.mark.parametrize("kv_dtype", ["fp8", "nvfp4"])
def test_kv_qdq_is_enabled_during_autoround_tuning(scheme, kv_dtype, monkeypatch):

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    from auto_round import AutoRound
    from auto_round.experimental.kv_cache import QuantizedKVParameterCache, kvcache_quant_context

    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        attn_implementation="eager",
    )
    observed = LlamaForCausalLM(config).eval()
    original_weight = observed.model.layers[0].self_attn.k_proj.weight.detach().clone()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({str(index): index for index in range(config.vocab_size)}, unk_token="0")),
        unk_token="0",
        pad_token="0",
        eos_token="1",
    )
    dataset = torch.randint(0, config.vocab_size, (2, 4))
    update_modes = []
    original_update = QuantizedKVParameterCache.update

    def record_qdq_mode(cache, *args, **kwargs):
        update_modes.append(cache._qdq_enabled)
        return original_update(cache, *args, **kwargs)

    monkeypatch.setattr(QuantizedKVParameterCache, "update", record_qdq_mode)
    compressor = AutoRound(
        observed,
        tokenizer=tokenizer,
        scheme=scheme,
        group_size=16,
        iters=2,
        batch_size=1,
        nsamples=2,
        seqlen=4,
        dataset=[{"input_ids": sample.unsqueeze(0).clone()} for sample in dataset],
        device_map="cpu",
        amp=False,
        seed=42,
        static_kv_dtype=kv_dtype,
    )
    with kvcache_quant_context(observed, kv_dtype):
        compressor.quantize()
    assert False in update_modes and True in update_modes
    assert not torch.equal(observed.model.layers[0].self_attn.k_proj.weight, original_weight)
    for layer in observed.model.layers:
        if kv_dtype == "nvfp4":
            assert layer.self_attn.k_global_scale.item() > 0
            assert layer.self_attn.v_global_scale.item() > 0
        else:
            assert torch.all(layer.self_attn.k_scale > 0)
            assert torch.all(layer.self_attn.v_scale > 0)
    with torch.no_grad():
        assert torch.isfinite(observed(dataset).logits).all()
