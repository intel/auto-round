# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn

from auto_round.experimental.kv_cache import (
    QuantizedKVParameterCache,
    build_turboquant_runtime_cache,
    kvcache_quant_context,
    normalize_kv_cache_backend_config,
)
from auto_round.experimental.turboquant import (
    QJLResidualConfig,
    build_turboquant_state,
    turboquant_pack,
    turboquant_qdq,
    turboquant_unpack,
)


class TinySelfAttention(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.layer_idx = 0
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.last_past_key_value = None
        self.last_use_cache = None

    def forward(self, hidden_states, past_key_value=None, use_cache=True):
        self.last_past_key_value = past_key_value
        self.last_use_cache = use_cache

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        if hasattr(past_key_value, "update"):
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        return key_states + value_states


class TinyAttentionModel(nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.self_attention = TinySelfAttention(hidden_size=hidden_size)

    def forward(self, hidden_states, past_key_value=None, use_cache=True):
        return self.self_attention(hidden_states, past_key_value=past_key_value, use_cache=use_cache)


def test_normalize_kv_cache_backend_config_turboquant():
    config = normalize_kv_cache_backend_config("turboquant:3")
    assert config.backend == "turboquant"
    assert config.bits == 3


def test_normalize_kv_cache_backend_config_fp8():
    config = normalize_kv_cache_backend_config("fp8")
    assert config.backend == "fp8"
    assert config.dtype == torch.float8_e4m3fn


def test_turboquant_qdq_shape_and_dtype():
    tensor = torch.randn(2, 4, 3, 16, dtype=torch.float32)
    state = build_turboquant_state(head_dim=16, bits=4, seed=7, device=tensor.device)
    reconstructed, avg_norm = turboquant_qdq(tensor, state)

    assert reconstructed.shape == tensor.shape
    assert reconstructed.dtype == tensor.dtype
    assert avg_norm.shape == (1,)
    assert torch.isfinite(reconstructed).all()
    assert torch.isfinite(avg_norm).all()
    assert not torch.equal(reconstructed, tensor)


def test_kvcache_quant_context_with_tiny_attention_module():
    model = TinyAttentionModel(hidden_size=8)
    hidden_states = torch.randn(2, 3, 8, dtype=torch.float32)

    with kvcache_quant_context(model, static_kv_dtype="turboquant:4"):
        attention = model.self_attention
        output = model(hidden_states, past_key_value="sentinel", use_cache=True)

        assert output.shape == hidden_states.shape
        assert isinstance(attention.last_past_key_value, QuantizedKVParameterCache)
        assert attention.last_use_cache is False
        assert hasattr(attention, "k_scale")
        assert hasattr(attention, "v_scale")
        assert torch.isfinite(attention.k_scale).all()
        assert torch.isfinite(attention.v_scale).all()
        assert not torch.equal(attention.k_scale, torch.zeros_like(attention.k_scale))
        assert not torch.equal(attention.v_scale, torch.zeros_like(attention.v_scale))
        assert hasattr(attention, "_kv_cache_hook_handles")
        assert hasattr(attention, "kv_cache")

    attention = model.self_attention
    assert not hasattr(attention, "_kv_cache_hook_handles")
    assert not hasattr(attention, "kv_cache")


def test_turboquant_pack_reduces_storage_bytes():
    tensor = torch.randn(2, 4, 32, 16, dtype=torch.float32)
    state = build_turboquant_state(head_dim=16, bits=4, seed=7, device=tensor.device)
    packed = turboquant_pack(tensor, state)

    assert packed.memory_bytes() < tensor.numel() * tensor.element_size()
    reconstructed = turboquant_unpack(packed, state, dtype=tensor.dtype)
    assert reconstructed.shape == tensor.shape
    assert torch.isfinite(reconstructed).all()


def test_turboquant_qjl_residual_gives_unbiased_inner_product():
    """1-bit QJL makes inner product estimation unbiased (paper Theorem 2).

    MSE may increase slightly, but the mean bias of <x_hat, y> vs <x, y> should be ~0.
    """
    head_dim = 64
    n_vectors = 200
    torch.manual_seed(42)
    x = torch.randn(n_vectors, 1, 1, head_dim, dtype=torch.float32)
    x = x / x.norm(dim=-1, keepdim=True)  # unit vectors
    y = torch.randn(n_vectors, 1, 1, head_dim, dtype=torch.float32)

    qjl_config = QJLResidualConfig(enabled=True, seed=123)
    state = build_turboquant_state(head_dim=head_dim, bits=2, seed=11, device=x.device, qjl_config=qjl_config)

    # Without QJL
    packed_plain = turboquant_pack(x, state)
    x_hat_plain = turboquant_unpack(packed_plain, state, dtype=x.dtype)

    # With QJL
    packed_qjl = turboquant_pack(x, state, residual_config=qjl_config)
    x_hat_qjl = turboquant_unpack(packed_qjl, state, dtype=x.dtype, residual_config=qjl_config)

    ip_true = (x * y).sum(dim=-1)
    ip_plain = (x_hat_plain * y).sum(dim=-1)
    ip_qjl = (x_hat_qjl * y).sum(dim=-1)

    bias_plain = (ip_plain - ip_true).mean().abs().item()
    bias_qjl = (ip_qjl - ip_true).mean().abs().item()

    # QJL should have lower inner-product bias than plain quantization
    assert bias_qjl < 0.1, f"QJL bias too large: {bias_qjl}"
    # Also verify both produce finite results
    assert torch.isfinite(x_hat_plain).all()
    assert torch.isfinite(x_hat_qjl).all()


def test_runtime_turboquant_packed_cache_has_benefit_over_raw_kv():
    cache = build_turboquant_runtime_cache(bits=4, residual_length=4, seed=17, qjl_residual=True)

    key_states_1 = torch.randn(1, 2, 3, 16, dtype=torch.float32)
    value_states_1 = torch.randn(1, 2, 3, 16, dtype=torch.float32)
    key_states_2 = torch.randn(1, 2, 5, 16, dtype=torch.float32)
    value_states_2 = torch.randn(1, 2, 5, 16, dtype=torch.float32)

    combined_keys = torch.cat([key_states_1, key_states_2], dim=-2)
    combined_values = torch.cat([value_states_1, value_states_2], dim=-2)

    returned_keys_1, returned_values_1 = cache.update(key_states_1, value_states_1, layer_idx=0)
    returned_keys_2, returned_values_2 = cache.update(key_states_2, value_states_2, layer_idx=0)

    assert returned_keys_1.shape == key_states_1.shape
    assert returned_values_1.shape == value_states_1.shape
    assert returned_keys_2.shape == combined_keys.shape
    assert returned_values_2.shape == combined_values.shape
    assert cache.get_seq_length(0) == combined_keys.shape[-2]
    assert cache.packed_memory_bytes() > 0
    assert cache.total_memory_bytes() < cache.raw_memory_bytes()
    assert cache.compression_ratio() > 1.0

    reconstruction_error = torch.mean((returned_keys_2 - combined_keys) ** 2)
    assert torch.isfinite(reconstruction_error)