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

from __future__ import annotations

import inspect
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

import torch


def normalize_tensors(output: Any, indices: tuple[int, ...] | None = None) -> tuple[torch.Tensor, ...]:
    """Return floating-point tensors used by the smooth output-error objective."""
    if indices is not None:
        if not isinstance(output, (tuple, list)):
            raise TypeError("Indexed Smooth output must be a tuple or list.")
        output = tuple(output[index] for index in indices)

    tensors = []

    def collect(value):
        if torch.is_tensor(value):
            if value.is_floating_point():
                tensors.append(value)
        elif isinstance(value, Mapping):
            for item in value.values():
                collect(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                collect(item)

    collect(output)
    if not tensors:
        raise ValueError("Smooth evaluation produced no floating-point tensors.")
    return tuple(tensors)


def filter_supported_kwargs(module: torch.nn.Module, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(module.forward).parameters
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(kwargs)
    return {name: value for name, value in kwargs.items() if name in parameters}


def output_squared_error(
    actual: tuple[torch.Tensor, ...],
    reference: tuple[torch.Tensor, ...],
    *,
    accumulator: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sum output errors using FP32 differences and an FP64 CPU accumulator."""
    if len(actual) != len(reference):
        raise ValueError("Smooth output tensor count changed.")
    error = torch.zeros((), dtype=torch.float64) if accumulator is None else accumulator
    for actual_tensor, reference_tensor in zip(actual, reference):
        if actual_tensor.shape != reference_tensor.shape:
            raise ValueError("Smooth output tensor shape changed.")
        error += torch.sum((actual_tensor.float() - reference_tensor.float()).square()).double().cpu()
    return error


@contextmanager
def temporary_modules(root: torch.nn.Module, replacements: list[tuple[str, torch.nn.Module]]):
    """Restore original modules even when installation or evaluation fails."""
    originals = [(name, root.get_submodule(name)) for name, _ in replacements]
    try:
        for name, module in replacements:
            root.set_submodule(name, module)
        yield
    finally:
        for name, module in originals:
            root.set_submodule(name, module)


@contextmanager
def restore_weights(originals: Mapping[torch.nn.Module, torch.Tensor]):
    """Restore candidate weights after successful or failed in-place QDQ."""
    try:
        yield
    finally:
        with torch.no_grad():
            for module, weight in originals.items():
                module.weight.copy_(weight)


@contextmanager
def temporary_hooks():
    """Remove accumulated calibration hooks after registration or replay fails."""
    handles = []
    try:
        yield handles
    finally:
        for handle in handles:
            handle.remove()
