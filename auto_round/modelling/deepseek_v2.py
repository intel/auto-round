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

import warnings
from functools import partial
from typing import Callable, Optional

import torch
from transformers.cache_utils import Cache
from transformers.modeling_rope_utils import dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import eager_attention_forward

from auto_round.modelling.replace_modules import ReplacementModuleBase


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    real_xq = xq[:, :, :, ::2] * cos - xq[:, :, :, 1::2] * sin
    img_xq = xq[:, :, :, ::2] * sin + xq[:, :, :, 1::2] * cos

    real_xk = xk[:, :, :, ::2] * cos - xk[:, :, :, 1::2] * sin
    img_xk = xk[:, :, :, ::2] * sin + xk[:, :, :, 1::2] * cos

    xq_out = torch.stack([real_xq, img_xq], dim=-1).flatten(3).type_as(xq)
    xk_out = torch.stack([real_xk, img_xk], dim=-1).flatten(3).type_as(xk)

    return xq_out, xk_out


class DeepseekV2RotaryEmbedding(ReplacementModuleBase):
    def __init__(self, original, config):
        super().__init__()
        self.rope_type = original.rope_type

        self.max_seq_len_cached = original.max_seq_len_cached
        self.original_max_seq_len = original.original_max_seq_len

        self.config = original.config
        self.rope_init_fn = original.rope_init_fn

        self.attention_scaling = original.attention_scaling
        self.register_buffer("inv_freq", original.original_inv_freq, persistent=False)
        self.original_inv_freq = original.original_inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(1, 2)
            cos = torch.cos(freqs) * self.attention_scaling
            sin = torch.sin(freqs) * self.attention_scaling
        return cos, sin

    @classmethod
    def original_module_class(cls) -> str:
        """Return the class name of the module this replaces."""
        return "DeepseekV2RotaryEmbedding"

    @classmethod
    def from_original(
        cls,
        original: torch.nn.Module,
        config: DeepseekV2Config,
        **kwargs,
    ) -> "DeepseekV2RotaryEmbedding":
        """Create an instance from the original module."""
        return cls(original, config)


class DeepseekV2Attention(ReplacementModuleBase):
    def __init__(self, original, config):
        super().__init__()
        self.config = original.config
        self.layer_idx = original.layer_idx
        self.attention_dropout = original.attention_dropout
        self.hidden_size = original.hidden_size
        self.num_heads = original.num_heads
        self.head_dim = original.head_dim
        self.max_position_embeddings = original.max_position_embeddings
        self.rope_theta = original.rope_theta
        self.q_lora_rank = original.q_lora_rank
        self.qk_rope_head_dim = original.qk_rope_head_dim
        self.kv_lora_rank = original.kv_lora_rank
        self.v_head_dim = original.v_head_dim
        self.qk_nope_head_dim = original.qk_nope_head_dim
        self.qk_head_dim = original.qk_head_dim
        self.num_key_value_groups = original.num_key_value_groups

        self.is_causal = original.is_causal

        if self.q_lora_rank is None:
            self.q_proj = original.q_proj
        else:
            self.q_a_proj = original.q_a_proj
            self.q_a_layernorm = original.q_a_layernorm
            self.q_b_proj = original.q_b_proj

        self.kv_a_proj_with_mqa = original.kv_a_proj_with_mqa
        self.kv_a_layernorm = original.kv_a_layernorm
        self.kv_b_proj = original.kv_b_proj

        self.o_proj = original.o_proj

        self.scaling = original.scaling

        cur_mod = [name for name, _ in self.named_modules()]
        for name, mod in original.named_modules():
            if name not in cur_mod:
                self.register_module(name, mod)

        cur_param = [name for name, _ in self.named_parameters()]
        for name, param in original.named_parameters():
            if name not in cur_param:
                self.register_parameter(name, param)

        # In case the attn_module is prepared for kv_cache quantization
        if hasattr(original, "kv_cache"):
            self.kv_cache = original.kv_cache

            for _, hook in original._forward_pre_hooks.items():
                self.register_forward_pre_hook(hook, with_kwargs=True)
            for _, hook in original._forward_hooks.items():
                self.register_forward_hook(hook)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(query_shape).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_nope, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_nope = self.kv_b_proj(self.kv_a_layernorm(k_nope)).view(key_shape).transpose(1, 2)
        k_nope, value_states = torch.split(k_nope, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        cos, sin = position_embeddings
        q_pe, k_pe = apply_rotary_emb(q_pe, k_pe, cos.to(q_pe.device), sin.to(q_pe.device))

        k_pe = k_pe.expand(*k_nope.shape[:-1], -1)
        query_states = torch.cat((q_nope, q_pe), dim=-1)
        key_states = torch.cat((k_nope, k_pe), dim=-1)

        past_key_values = None

        # transformers <= 4.55.4 is past_key_value
        if "past_key_values" in kwargs:
            past_key_values = kwargs["past_key_values"]
        elif "past_key_value" in kwargs:
            past_key_values = kwargs["past_key_value"]

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            value_states = torch.nn.functional.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    @classmethod
    def original_module_class(cls) -> str:
        """Return the class name of the module this replaces."""
        return "DeepseekV2Attention"

    @classmethod
    def from_original(
        cls,
        original: torch.nn.Module,
        config: DeepseekV2Config,
        **kwargs,
    ) -> "DeepseekV2Attention":
        """Create an instance from the original module."""
        return cls(original, config)
