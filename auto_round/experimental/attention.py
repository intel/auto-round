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

from auto_round.experimental.kv_cache import kvcache_quant_context
from auto_round.experimental.utils import (
    clean_model_parameters_and_buffers_,
    is_attention_module,
    per_tensor_fp8_qdq,
    update_parameter_data,
)
from auto_round.utils import logger

__all__ = [
    "QuantizedAttentionImpl",
    "init_hooked_attention",
    "attention_quant_ctx",
]


ATTN_IMPL_ATTR_NAME = "impl"
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
    `init_hooked_attention`, registering a new attention implementation function
    which calls this module, then setting the model attention implementation to the new
    function. After triggering hooks and quantization, this module calls the original
    attention implementation function.

    :param attn_module: parent attention module
    """

    _original_impl = "sdpa"

    def __init__(self, config: PretrainedConfig, attn_module: Module):
        super().__init__()
        self.config = config
        self.attn_module = ref(attn_module)  # avoid circular references
        # register query max
        device = next(attn_module.parameters()).device
        initial_max = torch.tensor([float("-inf")], device=device)
        update_parameter_data(attn_module, initial_max, QUERY_MAX_NAME)
        initial_scale = torch.tensor([0.0], device=device)
        update_parameter_data(attn_module, initial_scale, QUERY_SCALE_NAME)

    def forward(
        self,
        module: Module,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *args,
        **kwargs,
    ):
        cur_query_max = query.abs().max()
        query_max = torch.max(
            getattr(module, QUERY_MAX_NAME).data,
            cur_query_max.detach().to(getattr(module, QUERY_MAX_NAME).data.device),
        )
        update_parameter_data(module, query_max, QUERY_MAX_NAME)
        _, query_scale = per_tensor_fp8_qdq(query, tensor_max=query_max)
        update_parameter_data(module, query_scale.squeeze(0).detach(), QUERY_SCALE_NAME)
        # original attention
        return ALL_ATTENTION_FUNCTIONS[self._original_impl](
            module,
            query,
            key,
            value,
            *args,
            **kwargs,
        )


# ----- initialize ----- #


def _ct_hooked_attention(module: Module, *args, **kwargs):
    if hasattr(module, ATTN_IMPL_ATTR_NAME):
        return module.impl(module, *args, **kwargs)
    else:
        return ALL_ATTENTION_FUNCTIONS[_original_impl](module, *args, **kwargs)  # pylint: disable=E0601


def init_hooked_attention(module: Module, config):
    """
    Initialize `QuantizedAttentionImpl` and `QuantizedKVCache` instances
    attached to attention

    :param model: parent model of attention module
    :param module: attention module to initialize with
    """
    if not hasattr(module, ATTN_IMPL_ATTR_NAME):
        module.register_module(ATTN_IMPL_ATTR_NAME, QuantizedAttentionImpl(config, module))
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
        init_hooked_attention(module, config)


def clean_up_hooked_attention(module, model):
    if is_attention_module(module):
        clean_model_parameters_and_buffers_(module, (QUERY_MAX_NAME,))
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
