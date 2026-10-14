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

from types import MethodType

import torch

from auto_round.modeling.fused_moe.replace_modules import ReplacementModuleBase


def _get_moe_infer_impl(module: torch.nn.Module):
    """Return the undecorated body of the remote ``@torch.no_grad()`` ``moe_infer``, or None."""
    return getattr(getattr(type(module), "moe_infer", None), "__wrapped__", None)


def moe_infer_with_grad(module, x, topk_ids, topk_weight):
    """Differentiable routed-expert inference, used while AutoRound tunes a block."""
    return _get_moe_infer_impl(module)(module, x, topk_ids, topk_weight)


def moe_infer(module, x, topk_ids, topk_weight):
    if torch.is_grad_enabled():
        return moe_infer_with_grad(module, x, topk_ids, topk_weight)
    with torch.no_grad():
        return _get_moe_infer_impl(module)(module, x, topk_ids, topk_weight)


class KimiVLDeepseekV3MoE(ReplacementModuleBase):
    """Expose a grad-aware ``moe_infer`` on Kimi remote MoE blocks (Kimi-VL ``DeepseekV3MoE``).

    The remote ``moe_infer`` is decorated with ``@torch.no_grad()`` and is the only path
    used in eval mode, so routed experts would never receive tuning gradients.
    """

    skip_linearized_experts = False

    def __init__(self, original, config):
        super().__init__(original)

    @classmethod
    def original_module_class(cls) -> str:
        return "DeepseekV3MoE"

    @classmethod
    def is_to_be_replaced(cls, original: torch.nn.Module) -> bool:
        return cls.is_registered(original.__class__.__name__) and callable(_get_moe_infer_impl(original))

    @classmethod
    def from_original(cls, original: torch.nn.Module, config, **kwargs) -> torch.nn.Module:
        original.moe_infer = MethodType(moe_infer, original)
        return original

    def release_original_module(self) -> None:
        pass


class KimiLinearSparseMoeBlock(KimiVLDeepseekV3MoE):
    """Same hook for Kimi-Linear / Kimi-K3 remote ``KimiSparseMoeBlock``."""

    @classmethod
    def original_module_class(cls) -> str:
        return "KimiSparseMoeBlock"
