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

import inspect
import warnings
from functools import partial
from typing import Callable, Optional

import torch
from transformers.cache_utils import Cache
from transformers.modeling_rope_utils import dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.deepseek_v2.configuration_deepseek_v2 import DeepseekV2Config
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    eager_attention_forward,
)

from auto_round.modeling.replace_modules import ReplacementModuleBase
from auto_round.utils import is_hpex_available


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


@torch.no_grad()
@dynamic_rope_update
def rotary_emb_forward(module, x, position_ids):
    inv_freq_expanded = module.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):  # Force float32
        freqs = (inv_freq_expanded.to(x.device) @ position_ids_expanded).transpose(1, 2)
        cos = torch.cos(freqs) * module.attention_scaling
        sin = torch.sin(freqs) * module.attention_scaling
    return cos, sin


def attn_forward(
    module,
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
    query_shape = (batch_size, seq_length, -1, module.qk_head_dim)
    key_shape = (batch_size, seq_length, -1, module.qk_nope_head_dim + module.v_head_dim)

    if module.q_lora_rank is None:
        q = module.q_proj(hidden_states)
    else:
        q = module.q_b_proj(module.q_a_layernorm(module.q_a_proj(hidden_states)))
    q = q.view(query_shape).transpose(1, 2)
    q_nope, q_pe = torch.split(q, [module.qk_nope_head_dim, module.qk_rope_head_dim], dim=-1)

    compressed_kv = module.kv_a_proj_with_mqa(hidden_states)
    k_nope, k_pe = torch.split(compressed_kv, [module.kv_lora_rank, module.qk_rope_head_dim], dim=-1)
    k_nope = module.kv_b_proj(module.kv_a_layernorm(k_nope)).view(key_shape).transpose(1, 2)
    k_nope, value_states = torch.split(k_nope, [module.qk_nope_head_dim, module.v_head_dim], dim=-1)

    k_pe = k_pe.view(batch_size, 1, seq_length, module.qk_rope_head_dim)
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
        key_states, value_states = past_key_values.update(key_states, value_states, module.layer_idx, cache_kwargs)

    if module.config._attn_implementation == "flash_attention_2" and module.qk_head_dim != module.v_head_dim:
        value_states = torch.nn.functional.pad(value_states, [0, module.qk_head_dim - module.v_head_dim])

    attention_interface: Callable = eager_attention_forward
    if module.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[module.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not module.training else module.attention_dropout,
        scaling=module.scaling,
        **kwargs,
    )

    if module.config._attn_implementation == "flash_attention_2" and module.qk_head_dim != module.v_head_dim:
        attn_output = attn_output[:, :, :, : module.v_head_dim]

    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    attn_output = module.o_proj(attn_output)
    return attn_output, attn_weights


class DeepseekV2RotaryEmbedding(ReplacementModuleBase):
    def __init__(self, original, config):
        super().__init__(original)

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
        original.forward = partial(rotary_emb_forward, original)
        return original

    @classmethod
    def is_to_be_replaced(
        cls,
        original: torch.nn.Module,
    ) -> bool:
        """Determine if the given module should be replaced."""
        return (
            cls.is_registered(original.__class__.__name__)
            and is_hpex_available()
            and inspect.getfile(original.__class__) == inspect.getfile(eager_attention_forward)
        )

    def release_original_module(self) -> None:
        pass


class DeepseekV2Attention(ReplacementModuleBase):
    def __init__(self, original, config):
        super().__init__(original)

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
        original.forward = partial(attn_forward, original)
        return original

    @classmethod
    def is_to_be_replaced(
        cls,
        original: torch.nn.Module,
    ) -> bool:
        """Determine if the given module should be replaced."""
        return (
            cls.is_registered(original.__class__.__name__)
            and is_hpex_available()
            and inspect.getfile(original.__class__) == inspect.getfile(eager_attention_forward)
        )

    def release_original_module(self) -> None:
        pass
