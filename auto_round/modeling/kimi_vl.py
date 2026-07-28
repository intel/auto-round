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

from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Iterable, Iterator

import torch

_MISSING = object()
_STATE_ATTR = "_auto_round_kimi_vl_moe_grad_state"


@dataclass(frozen=True)
class _KimiVLMoEGradTarget:
    module_name: str
    module: torch.nn.Module


class _KimiVLMoEGradState:
    def __init__(self, previous_method: object, previous_state: object):
        self.previous_method = previous_method
        self.previous_state = previous_state
        self.depth = 1


def _is_kimi_vl_moe_grad_enabled(model: torch.nn.Module | None, iters: int) -> bool:
    config = getattr(model, "config", None)
    return iters > 0 and getattr(config, "model_type", None) == "kimi_vl"


def _find_kimi_vl_moe_grad_targets(root: torch.nn.Module) -> tuple[_KimiVLMoEGradTarget, ...]:
    targets = []
    for module_name, module in root.named_modules():
        if not any(base.__name__ == "DeepseekV3MoE" for base in type(module).__mro__):
            continue
        targets.append(_KimiVLMoEGradTarget(module_name=module_name, module=module))
    return tuple(targets)


def _is_torch_grad_mode_decorator(method: object) -> bool:
    """Return whether ``method`` has PyTorch's grad-mode decorator closure."""
    try:
        decorator_type = torch.autograd.grad_mode._DecoratorContextManager
        wrapper = getattr(method, "__func__", method)
        closure = wrapper.__closure__ or ()
    except (AttributeError, TypeError):
        return False

    for cell in closure:
        try:
            owner = getattr(cell.cell_contents, "__self__", None)
        except ValueError:
            continue
        try:
            if isinstance(owner, decorator_type):
                return True
        except TypeError:
            continue
    return False


def _resolve_kimi_vl_moe_grad_targets(
    targets: Iterable[_KimiVLMoEGradTarget],
) -> tuple[_KimiVLMoEGradTarget, ...]:
    resolved = []
    seen = set()
    for target in targets:
        module_name, module = target.module_name, target.module
        if id(module) in seen:
            continue
        seen.add(id(module))

        display_name = module_name or "<root>"
        ep_size = getattr(module, "ep_size", 1)
        if ep_size > 1:
            raise RuntimeError(
                "Kimi-VL routed-expert tuning does not support "
                f"{module.__class__.__name__} at '{display_name}' with ep_size={ep_size}: "
                "the expert-parallel all_to_all path does not preserve routed-expert gradients. "
                "Quantize with ep_size=1, use iters=0 (RTN), and verify the Kimi-VL remote-code "
                "revision before retrying."
            )

        state = module.__dict__.get(_STATE_ATTR, _MISSING)
        if isinstance(state, _KimiVLMoEGradState):
            resolved.append(target)
            continue

        moe_infer = getattr(module, "moe_infer", None)
        raw_moe_infer = getattr(moe_infer, "__wrapped__", None)
        if not callable(raw_moe_infer):
            raise RuntimeError(
                "Kimi-VL routed-expert tuning cannot unwrap "
                f"{module.__class__.__name__}.moe_infer at '{display_name}': "
                "a callable moe_infer.__wrapped__ is unavailable. Use iters=0 (RTN), or verify "
                "that the Kimi-VL remote-code revision still decorates moe_infer with a supported "
                "torch grad-mode decorator."
            )
        if not _is_torch_grad_mode_decorator(moe_infer):
            raise RuntimeError(
                "Kimi-VL routed-expert tuning refused to unwrap "
                f"{module.__class__.__name__}.moe_infer at '{display_name}': "
                "moe_infer is not a recognized torch grad-mode decorator, or its internal "
                "decorator metadata is unavailable. Use iters=0 (RTN), or verify the Kimi-VL "
                "remote-code revision."
            )
        resolved.append(target)
    return tuple(resolved)


def prepare_kimi_vl_moe_grad(
    model: torch.nn.Module | None,
    iters: int,
) -> int | None:
    """Pre-validate all Kimi-VL routed-expert gradient targets in ``model``."""
    if not _is_kimi_vl_moe_grad_enabled(model, iters):
        return None

    targets = _resolve_kimi_vl_moe_grad_targets(_find_kimi_vl_moe_grad_targets(model))
    return len(targets)


@contextmanager
def enable_kimi_vl_moe_grad(
    model: torch.nn.Module | None,
    block: torch.nn.Module,
    iters: int,
) -> Iterator[None]:
    """Temporarily bypass Kimi-VL's ``@torch.no_grad`` MoE inference wrapper.

    Kimi-VL keeps the model in eval mode during quantization, where
    ``DeepseekV3MoE.forward`` calls a ``moe_infer`` method decorated with
    ``@torch.no_grad``. AutoRound's routed-expert tuning parameters therefore
    cannot receive gradients unless the undecorated method body is used during
    the SignRound optimization step.

    The patch is instance-local and is restored on both normal and exceptional
    exits. Other model types and zero-shot quantization are left unchanged.
    """
    if not _is_kimi_vl_moe_grad_enabled(model, iters):
        yield
        return

    resolved_targets = _resolve_kimi_vl_moe_grad_targets(_find_kimi_vl_moe_grad_targets(block))

    entered = []
    try:
        for target in resolved_targets:
            module = target.module
            previous_state = module.__dict__.get(_STATE_ATTR, _MISSING)
            if isinstance(previous_state, _KimiVLMoEGradState):
                previous_state.depth += 1
                state = previous_state
            else:
                previous_method = module.__dict__.get("moe_infer", _MISSING)
                raw_moe_infer = module.moe_infer.__wrapped__
                module.moe_infer = MethodType(raw_moe_infer, module)
                state = _KimiVLMoEGradState(previous_method, previous_state)
                module.__dict__[_STATE_ATTR] = state
            entered.append((module, state))
        yield
    finally:
        for module, state in reversed(entered):
            state.depth -= 1
            if state.depth != 0:
                continue

            if state.previous_method is _MISSING:
                delattr(module, "moe_infer")
            else:
                module.moe_infer = state.previous_method

            # Restore unrelated data if external code was already using this
            # private attribute instead of silently clobbering the collision.
            if state.previous_state is _MISSING:
                module.__dict__.pop(_STATE_ATTR, None)
            else:
                module.__dict__[_STATE_ATTR] = state.previous_state
