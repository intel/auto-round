"""Regression tests for https://github.com/intel/auto-round/issues/1950.

``forward_capture`` raised ``AttributeError: 'NoneType' object has no attribute
'extend'`` when an optional kwarg was ``None`` on the first calibration sample
but a real ``Tensor`` on a subsequent one.  The fix promotes the stored ``None``
to an empty list before appending.
"""

from functools import partial
from types import SimpleNamespace

import torch

from auto_round.calibration.hooks import make_block_forward_func


def _make_state(batch_size=1):
    """Return a minimal state stub accepted by ``make_block_forward_func``.

    Args:
        batch_size (int): Number of samples per calibration batch.

    Returns:
        SimpleNamespace: State object with ``inputs``, ``quantizer``,
            ``model_context``, ``has_variable_block_shape``,
            ``blocks_requiring_input_ids``, and ``_should_stop_cache_forward``.
    """
    quantizer = SimpleNamespace(batch_size=batch_size, batch_dim=None)
    model_context = SimpleNamespace(
        shared_cache_keys=("position_ids", "cache_position", "position_embeddings", "cu_seqlens"),
    )
    return SimpleNamespace(
        inputs={},
        quantizer=quantizer,
        model_context=model_context,
        has_variable_block_shape=False,
        blocks_requiring_input_ids=[],
        _should_stop_cache_forward=lambda name: False,
    )


class _FakeModule(torch.nn.Module):
    """Minimal decoder-block stub: passes ``hidden_states`` through unchanged."""

    def __init__(self):
        super().__init__()

    def orig_forward(self, hidden_states, **kwargs):
        """Identity forward used as the underlying block implementation."""
        return hidden_states


def _attach_capture(state, name, module):
    """Attach a ``forward_capture`` closure to *module* for block *name*.

    Mirrors the wiring done by ``replace_forward_with_hooks`` so tests
    exercise the same code path as production calibration.

    Args:
        state: Calibration state stub (from ``_make_state``).
        name (str): Block name key used in ``state.inputs``.
        module (_FakeModule): Module to instrument.

    Returns:
        _FakeModule: The same module with ``forward`` replaced.
    """
    fn = make_block_forward_func(state, name)
    module.forward = partial(fn, module)
    return module


def test_none_then_tensor_kwarg_batch_size_1():
    """``batch_size=1``: None-initialized kwarg must not crash on later Tensor sample.

    Sequence:
        1. First forward call — no ``optional_mask`` kwarg at all.
        2. State is mutated to simulate the None-initialization path
           (kwarg was ``None`` on its first appearance).
        3. Second forward call delivers a real Tensor for ``optional_mask``
           — must not raise ``AttributeError``.
    """
    state = _make_state(batch_size=1)
    name = "decoder.layers.0"
    module = _attach_capture(state, name, _FakeModule())

    hidden = torch.randn(1, 4, 8)
    module(hidden)

    # Simulate None stored by the initialization branch.
    state.inputs[name]["optional_mask"] = None

    module(hidden, optional_mask=torch.ones(1, 4, 4))

    stored = state.inputs[name].get("optional_mask")
    assert isinstance(stored, list), f"expected list, got {type(stored)}"
    assert len(stored) == 1


def test_none_then_tensor_kwarg_batch_size_gt1():
    """``batch_size > 1``: None-initialized kwarg must not crash on later Tensor sample.

    The ``batch_size > 1`` path calls ``.extend()`` rather than ``.append()``,
    exercising the second crash site from the original bug.
    """
    state = _make_state(batch_size=2)
    state.quantizer.batch_dim = 0
    name = "decoder.layers.0"
    module = _attach_capture(state, name, _FakeModule())

    hidden = torch.randn(2, 4, 8)
    module(hidden)

    state.inputs[name]["optional_mask"] = None

    module(hidden, optional_mask=torch.ones(2, 4, 4))

    stored = state.inputs[name].get("optional_mask")
    assert isinstance(stored, list), f"expected list, got {type(stored)}"
    # torch.split(..., 1, dim=0) on a batch-2 tensor yields 2 chunks.
    assert len(stored) == 2


def test_normal_tensor_kwarg_unaffected():
    """Kwargs that are always Tensors continue to accumulate correctly after the fix."""
    state = _make_state(batch_size=1)
    name = "decoder.layers.0"
    module = _attach_capture(state, name, _FakeModule())

    hidden = torch.randn(1, 4, 8)
    mask = torch.ones(1, 4, 4)

    module(hidden, attention_mask=mask)
    module(hidden, attention_mask=mask)

    stored = state.inputs[name].get("attention_mask")
    assert isinstance(stored, list)
    assert len(stored) == 2
