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
from collections.abc import Callable
from functools import partial
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
    fp8_qdq,
    is_attention_module,
    normalize_fp8_granularity,
    update_parameter_data,
)
from auto_round.utils import logger

__all__ = [
    "QuantizedAttentionImpl",
    "attention_quant_ctx",
    "init_hooked_attention",
    "is_attention_calibration_tensor_name",
]


ATTN_IMPL_ATTR_NAME = "impl"
HOOKED_ATTENTION_NAME = "ct_hooked_attention"
QUERY_SCALE_NAME = "q_scale"
QUERY_MAX_NAME = "q_max"


def is_attention_calibration_tensor_name(name: str) -> bool:
    """Return True when a serialized tensor name is the transient attention q_max."""
    return name.rsplit(".", 1)[-1] == QUERY_MAX_NAME


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

    def __init__(self, config: PretrainedConfig, attn_module: Module, granularity: str = "tensor"):
        super().__init__()
        self.config = config
        self.granularity = normalize_fp8_granularity(granularity)
        self.attn_module = ref(attn_module)  # avoid circular references
        self._original_impl = _get_original_attention_impl(config)
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
        self.observe(module, query)
        # original attention
        return ALL_ATTENTION_FUNCTIONS[self._original_impl](
            module,
            query,
            key,
            value,
            *args,
            **kwargs,
        )

    def observe(self, module: Module, query: Tensor):
        """Update q_max/q_scale from a query laid out as [batch, heads, seq, head_dim]."""
        if self.granularity == "head":
            cur_query_max = query.abs().amax(dim=(0, 2, 3))
        else:
            cur_query_max = query.abs().max()
        query_max = torch.max(
            getattr(module, QUERY_MAX_NAME).data,
            cur_query_max.detach().to(getattr(module, QUERY_MAX_NAME).data.device),
        )
        update_parameter_data(module, query_max, QUERY_MAX_NAME)
        _, query_scale = fp8_qdq(query, tensor_max=query_max, granularity=self.granularity)
        update_parameter_data(module, query_scale.reshape(-1).detach(), QUERY_SCALE_NAME)


# ----- legacy (remote-code) attention support ----- #

_LEGACY_PATCHED_ATTRS = ("forward", "_flash_attention_forward")


def _uses_attention_interface(module: Module) -> bool:
    forward = type(module).forward
    if forward is Module.forward:
        return True
    try:
        source = inspect.getsource(forward)
    except (OSError, TypeError):
        return True
    return "ALL_ATTENTION_FUNCTIONS" in source or "attention_interface" in source


class _CaptureFirstMatmul(torch.overrides.TorchFunctionMode):
    """Observe the query of the first 4D matmul (Q @ K^T) inside an eager attention forward."""

    def __init__(self, module: Module):
        super().__init__()
        self.module = module
        self.done = False

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if not self.done and func in (torch.matmul, Tensor.matmul, Tensor.__matmul__):
            query = args[0]
            if isinstance(query, Tensor) and query.dim() == 4:
                self.done = True
                self.module.impl.observe(self.module, query)
        return func(*args, **kwargs)


def _patch_legacy_attention(module: Module):
    """Hook attention modules whose forward bypasses transformers' AttentionInterface."""
    flash_forward = getattr(module, "_flash_attention_forward", None)
    if callable(flash_forward):

        def _observed_flash_forward(query_states, *args, **kwargs):
            # flash-attn layout is [batch, seq, heads, head_dim]
            module.impl.observe(module, query_states.transpose(1, 2))
            return flash_forward(query_states, *args, **kwargs)

        module._flash_attention_forward = _observed_flash_forward
        return

    orig_forward = module.forward

    def _observed_forward(*args, **kwargs):
        with _CaptureFirstMatmul(module):
            return orig_forward(*args, **kwargs)

    module.forward = _observed_forward


def _unpatch_legacy_attention(module: Module):
    for name in _LEGACY_PATCHED_ATTRS:
        if name in module.__dict__:
            delattr(module, name)


# ----- initialize ----- #


def _ct_hooked_attention(module: Module, *args, **kwargs):
    if hasattr(module, ATTN_IMPL_ATTR_NAME):
        return module.impl(module, *args, **kwargs)
    else:
        return ALL_ATTENTION_FUNCTIONS[_original_impl](module, *args, **kwargs)  # pylint: disable=E0601


def _get_attention_config(module: Module, fallback_config: PretrainedConfig) -> PretrainedConfig:
    module_config = getattr(module, "config", None)
    if module_config is not None and getattr(module_config, "_attn_implementation", None) is not None:
        return module_config
    return fallback_config


def _get_original_attention_impl(config: PretrainedConfig) -> str:
    return getattr(config, "_auto_round_original_attn_impl", config._attn_implementation)


def init_hooked_attention(module: Module, config, granularity: str = "tensor"):
    """
    Initialize `QuantizedAttentionImpl` and `QuantizedKVCache` instances
    attached to attention

    :param model: parent model of attention module
    :param module: attention module to initialize with
    """
    config = _get_attention_config(module, config)
    if not hasattr(module, ATTN_IMPL_ATTR_NAME) and not _uses_attention_interface(module):
        logger.info_once(
            f"{module.__class__.__name__} does not dispatch through AttentionInterface, "
            "observing query at the attention kernel boundary instead."
        )
        module.register_module(ATTN_IMPL_ATTR_NAME, QuantizedAttentionImpl(config, module, granularity=granularity))
        _patch_legacy_attention(module)
        return
    if not hasattr(module, ATTN_IMPL_ATTR_NAME):
        if config._attn_implementation != HOOKED_ATTENTION_NAME:
            # assumes only one model at a time
            global _original_impl
            _original_impl = config._attn_implementation
            config._auto_round_original_attn_impl = config._attn_implementation
            # Add new implementation to AttentionInterface(mapping)
            AttentionInterface.register(HOOKED_ATTENTION_NAME, _ct_hooked_attention)
            config._attn_implementation = HOOKED_ATTENTION_NAME
        module.register_module(ATTN_IMPL_ATTR_NAME, QuantizedAttentionImpl(config, module, granularity=granularity))

    # initialize_hooked_kv_cache(model, module)


def prep_attention_module_for_calibration(module: torch.nn.Module, config, granularity: str = "tensor"):
    if is_attention_module(module):
        logger.trace(f"Preparing attention module {module.__class__.__name__} for calibration")
        init_hooked_attention(module, config, granularity=granularity)


def clean_up_hooked_attention(module, model):
    if is_attention_module(module):
        _unpatch_legacy_attention(module)
        query_max = getattr(module, QUERY_MAX_NAME, None)
        if isinstance(query_max, Tensor) and torch.isinf(query_max).all():
            logger.warning_once(
                f"{module.__class__.__name__} was never observed during calibration; its q_scale stays 0."
            )
        clean_model_parameters_and_buffers_(module, (QUERY_MAX_NAME,))
        config = _get_attention_config(module, model.config)
        if hasattr(config, "_auto_round_original_attn_impl"):
            config._attn_implementation = config._auto_round_original_attn_impl
            del config._auto_round_original_attn_impl


@contextlib.contextmanager
def attention_quant_ctx(
    model: PreTrainedModel,
    static_attention_dtype=torch.float8_e4m3fn,
    static_attention_granularity: str = "tensor",
):
    try:
        # Setup phase: Initialize hooked attention
        static_attention_granularity = normalize_fp8_granularity(static_attention_granularity)
        prepare_fn = partial(
            prep_attention_module_for_calibration,
            config=model.config,
            granularity=static_attention_granularity,
        )
        model.apply(prepare_fn)
        with kvcache_quant_context(
            model,
            static_kv_dtype=static_attention_dtype,
            static_kv_granularity=static_attention_granularity,
        ):
            yield model
    finally:
        clean_fn = partial(clean_up_hooked_attention, model=model)
        model.apply(clean_fn)
