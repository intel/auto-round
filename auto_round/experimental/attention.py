# Copyright (c) 2025 Red Hat AI, vLLM Project and Intel Corporation
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

# NOTICE: The design adapted from:
# https://github.com/vllm-project/compressed-tensors/pull/491


import contextlib
import inspect
from functools import partial
from typing import Callable, Optional
from weakref import ref

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle
from transformers import AttentionInterface, PretrainedConfig, PreTrainedModel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from auto_round.experimental.attn_patches.llama import RuntimeStats
from auto_round.experimental.kv_cache import kvcache_quant_context
from auto_round.experimental.utils import (
    fp8_per_tensor_qdq,
    is_attention_module,
    normalize_static_kv_dtype,
    update_parameter_data,
)
from auto_round.utils import getattr_chain, logger

__all__ = [
    "QuantizedAttentionImpl",
    "initialize_hooked_attention",
    "IMPL_ATTR",
    "attention_quant_ctx",
]


IMPL_ATTR = "impl"
HOOKED_ATTENTION_NAME = "ct_hooked_attention"
QUERY_SCALE_NAME = "q_scale"
QUERY_MAX_NAME = "q_max"


class QuantizedAttentionImpl(torch.nn.Module):
    """
    QuantizedAttentionImpl module which wraps the functionality of the original
    attention implementation. Unlike the original attention function, this
    implementation is a `torch.nn.Module` which can be hooked to trigger
    transforms and calibration hooks.

    This module works by being registered as a submodule to attention modules via
    `initialize_hooked_attention`, registering a new attention implementation function
    which calls this module, then setting the model attention implementation to the new
    function. After triggering hooks and quantization, this module calls the original
    attention implementation function.

    :param attn_module: parent attention module
    """

    _original_impl = "eager"

    def __init__(self, config: PretrainedConfig, attn_module: Module):
        super().__init__()
        self.config = config
        self.attn_module = ref(attn_module)  # avoid circular references
        # register query max
        device = next(attn_module.parameters()).device
        initial_max = torch.tensor([0.0], device=device)
        update_parameter_data(attn_module, initial_max, QUERY_MAX_NAME)
        update_parameter_data(attn_module, initial_max, QUERY_SCALE_NAME)

    def forward(
        self,
        module: Module,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *args,
        **kwargs,
    ):
        # quantization
        # quant_args_attr = "quantization_scheme.input_activations"
        # quant_args = getattr_chain(module, quant_args_attr, None)
        # quant_enabled = getattr(module, "quantization_enabled", True)
        # RuntimeStats.cur_layer_idx = self.attn_module().layer_idx
        # logger.trace(f"Starting quantized attention forward for layer {RuntimeStats.cur_layer_idx}")
        cur_query_max = query.abs().max()
        query_max = torch.max(
            getattr(module, QUERY_MAX_NAME).data,
            cur_query_max.detach().to(getattr(module, QUERY_MAX_NAME).data.device),
        )
        update_parameter_data(module, query_max, QUERY_MAX_NAME)
        query, query_scale = fp8_per_tensor_qdq(query, tensor_max=query_max)
        update_parameter_data(module, query_scale.squeeze(0), QUERY_SCALE_NAME)
        # original attention
        return ALL_ATTENTION_FUNCTIONS[_original_impl](
            module,
            query,
            key,
            value,
            *args,
            **kwargs,
        )


# ----- initialize ----- #


def _ct_hooked_attention(module: Module, *args, **kwargs):
    if hasattr(module, IMPL_ATTR):
        return module.impl(module, *args, **kwargs)
    else:
        return ALL_ATTENTION_FUNCTIONS[_original_impl](module, *args, **kwargs)


def initialize_hooked_attention(module: Module, config):
    """
    Initialize `QuantizedAttentionImpl` and `QuantizedKVCache` instances
    attached to attention

    :param model: parent model of attention module
    :param module: attention module to initialize with
    """
    if not hasattr(module, IMPL_ATTR):
        module.register_module(IMPL_ATTR, QuantizedAttentionImpl(config, module))
        if config._attn_implementation != HOOKED_ATTENTION_NAME:
            # assumes only one model at a time
            global _original_impl
            _original_impl = config._attn_implementation
            # Add new implementation to AttentionInterface(mapping)
            AttentionInterface.register(HOOKED_ATTENTION_NAME, _ct_hooked_attention)
            config._attn_implementation = HOOKED_ATTENTION_NAME

    # initialize_hooked_kv_cache(model, module)


def prep_attention_module_for_calibration(module: torch.nn.Module, config):
    if is_attention_module(module):
        logger.trace(f"Preparing attention module {module.__class__.__name__} for calibration")
        initialize_hooked_attention(module, config)


# # ----- hooks ----- #


# def register_query_hook(module: Module, hook: Callable[[Module, Tensor], Optional[Tensor]]) -> RemovableHandle:
#     """
#     Register a hook which takes post-rope query states as an argument and
#     returns the modified query states or `None`

#     :param module: attention module to add hook to
#     :param hook: query hook function
#     """
#     impl = getattr(module, IMPL_ATTR)

#     def _hook(impl: QuantizedAttentionImpl, args, kwargs):
#         bound = inspect.signature(impl.forward).bind(*args, **kwargs)
#         value = hook(module, bound.arguments["query"])
#         if value is not None:
#             bound.arguments["query"] = value

#         return bound.args, bound.kwargs

#     return impl.register_forward_pre_hook(_hook, with_kwargs=True)


def clean_up_hooked_attention(module, model):
    if is_attention_module(module):
        # Cleanup phase: Restore the original attention implementation
        if hasattr(model.config, "_attn_implementation") and hasattr(model, "_original_impl"):
            model.config._attn_implementation = model._original_impl
            del model._original_impl


@contextlib.contextmanager
def attention_quant_ctx(model: PreTrainedModel, static_attention_dtype=torch.float8_e4m3fn):
    try:
        # Setup phase: Initialize hooked attention
        prepare_fn = partial(prep_attention_module_for_calibration, config=model.config)
        model.apply(prepare_fn)
        with kvcache_quant_context(model, static_kv_dtype=static_attention_dtype):
            yield model
    finally:
        clean_fn = partial(clean_up_hooked_attention, model=model)
        model.apply(clean_fn)
