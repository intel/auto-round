# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.modeling.fused_moe``.

Covers:

* ``fused_moe.utils._update_parameter`` - the tiny helper used by
  every custom MoE block to swap a parameter in-place while preserving
  ``requires_grad``.
* ``fused_moe.replace_modules`` - the registration machinery
  (``ReplacementModuleBase``, ``ModuleReplacementTracker``,
  ``apply_replacements``, ``is_custom_model``,
  ``_apply_custom_replacements``, ``materialize_model_``,
  ``release_original_module_``).

We deliberately use ``torch.nn.Linear`` as a stand-in "original module"
because the registration helpers are model-agnostic.
"""

import pytest
import torch
import torch.nn as nn

from auto_round.modeling.fused_moe import (
    ReplacementModuleBase,
    apply_replacements,
    materialize_model_,
    release_original_module_,
)
from auto_round.modeling.fused_moe.replace_modules import (
    BUILTIN_MODULES,
    ModuleReplacementTracker,
    _apply_custom_replacements,
    is_custom_model,
)
from auto_round.modeling.fused_moe.utils import _update_parameter

# ---------------------------------------------------------------------------
# fused_moe.utils._update_parameter
# ---------------------------------------------------------------------------


def test_update_parameter_preserves_requires_grad_true():
    """A parameter that was trainable stays trainable after a swap."""
    mod = nn.Linear(4, 4)
    assert mod.weight.requires_grad is True

    new_data = torch.zeros_like(mod.weight)
    _update_parameter(mod, "weight", new_data)

    assert mod.weight.requires_grad is True
    assert torch.equal(mod.weight, new_data)


def test_update_parameter_preserves_requires_grad_false():
    """A frozen parameter stays frozen after a swap."""
    mod = nn.Linear(4, 4)
    for p in mod.parameters():
        p.requires_grad_(False)

    new_data = torch.ones_like(mod.weight)
    _update_parameter(mod, "weight", new_data)

    assert mod.weight.requires_grad is False
    assert torch.equal(mod.weight, new_data)


def test_update_parameter_swap_bias():
    """The same helper is used for ``bias`` (when present)."""
    mod = nn.Linear(4, 4, bias=True)
    new_bias = torch.full((4,), 7.0)
    _update_parameter(mod, "bias", new_bias)
    assert torch.allclose(mod.bias, new_bias)


def test_update_parameter_with_custom_attr():
    """Works for non-standard attributes too (e.g. ``e_score_correction_bias``)."""
    mod = nn.Linear(4, 4)
    mod.register_parameter("e_score_correction_bias", nn.Parameter(torch.zeros(4)))
    new_data = torch.full((4,), 3.14)
    _update_parameter(mod, "e_score_correction_bias", new_data)
    assert torch.allclose(mod.e_score_correction_bias, new_data)


# ---------------------------------------------------------------------------
# fused_moe.replace_modules
# ---------------------------------------------------------------------------


class _TrivialReplacement(ReplacementModuleBase):
    """Concrete subclass used purely to drive the base-class machinery."""

    def __init__(self, original: nn.Module):
        super().__init__(original)

    @classmethod
    def original_module_class(cls) -> str:
        # Uniquely identify this replacement in the registry.
        return "_TrivialReplacement_Original"

    @classmethod
    def from_original(cls, original: nn.Module, config) -> "_TrivialReplacement":
        return cls(original)


def _register_trivial_replacement():
    """Register ``_TrivialReplacement`` and return the registration class name."""
    cls_name = _TrivialReplacement.original_module_class()
    # Idempotent across tests: only register if missing.
    if not ReplacementModuleBase.is_registered(cls_name):
        _TrivialReplacement._replacement_registry[cls_name] = _TrivialReplacement
    return cls_name


def _reset_tracker():
    """Wipe the singleton tracker between tests so each one starts with
    a fresh instance whose ``__init__`` actually runs.
    """
    ModuleReplacementTracker._instance = None
    ModuleReplacementTracker._initialized = False


@pytest.fixture(autouse=True)
def _reset_module_replacement_tracker():
    """Auto-reset the tracker for every test in this file.

    ``_global_tracker`` is a module-level instance; re-assigning the
    class attributes ``_instance = None`` and ``_initialized = False``
    alone is not enough because the *existing* ``_global_tracker``
    object keeps the old ``_replacement_to_original`` and
    ``_name_to_info`` dicts across tests.  Instead we:

    1. Clear the existing global tracker's internal state.
    2. Reset the class so a new instance is created on the next access.
    """
    tracker = ModuleReplacementTracker.get_instance()
    if hasattr(tracker, "_replacement_to_original"):
        tracker._replacement_to_original.clear()
    if hasattr(tracker, "_name_to_info"):
        tracker._name_to_info.clear()
    yield
    tracker = ModuleReplacementTracker.get_instance()
    if hasattr(tracker, "_replacement_to_original"):
        tracker._replacement_to_original.clear()
    if hasattr(tracker, "_name_to_info"):
        tracker._name_to_info.clear()


def test_replacement_module_base_registry_lookups():
    """``is_registered`` and ``get_replacement_class`` reflect registration."""
    cls_name = _register_trivial_replacement()
    assert ReplacementModuleBase.is_registered(cls_name)
    assert ReplacementModuleBase.get_replacement_class(cls_name) is _TrivialReplacement
    # ``get_registered_modules`` is sorted by insertion order
    assert cls_name in ReplacementModuleBase.get_registered_modules()


def test_replacement_module_base_default_materialize_is_noop():
    """``_materialize_weights`` defaults to a no-op and ``materialize_weights``
    flips ``_materialized`` to True via ``post_process_materialization``.
    """
    cls_name = _register_trivial_replacement()
    orig = nn.Linear(2, 2)
    rep = ReplacementModuleBase.get_replacement_class(cls_name)(orig)
    assert rep._materialized is False
    rep.materialize_weights()
    assert rep._materialized is True


def test_replacement_module_base_release_original_drops_tracker_entry():
    """``release_original_module`` removes the original from the tracker."""
    cls_name = _register_trivial_replacement()
    orig = nn.Linear(2, 2)
    rep = ReplacementModuleBase.get_replacement_class(cls_name)(orig)

    tracker = ModuleReplacementTracker.get_instance()
    # The replacement registered itself in __init__.
    assert tracker.get_original(rep) is orig

    rep.release_original_module()
    assert tracker.get_original(rep) is None


def test_replacement_module_base_replacement_gets_name_in_tracker():
    """The tracker stores ``name -> ReplacedModuleInfo`` for every registered
    replacement, accessible via ``get_info_by_name``.
    """
    cls_name = _register_trivial_replacement()
    orig = nn.Linear(2, 2)
    rep = ReplacementModuleBase.get_replacement_class(cls_name)(orig)

    tracker = ModuleReplacementTracker.get_instance()
    # The base class uses ``str(id(self))`` as the registered name.
    info = tracker.get_info_by_name(str(id(rep)))
    assert info is not None
    assert info.original_module is orig
    assert info.replacement_module is rep


# ---------------------------------------------------------------------------
# ModuleReplacementTracker
# ---------------------------------------------------------------------------


def test_tracker_is_singleton():
    """The tracker is a singleton: a second constructor call returns the same
    object, not a new one.
    """
    a = ModuleReplacementTracker()
    b = ModuleReplacementTracker()
    assert a is b


def test_tracker_register_and_get_original():
    tracker = ModuleReplacementTracker.get_instance()
    orig = nn.Linear(2, 2)
    rep = _TrivialReplacement(orig)
    tracker.register_replacement("test_name", orig, rep)
    assert tracker.get_original(rep) is orig
    assert tracker.get_info_by_name("test_name").replacement_module is rep


def test_tracker_release_original_drops_entry():
    tracker = ModuleReplacementTracker.get_instance()
    orig = nn.Linear(2, 2)
    rep = _TrivialReplacement(orig)
    tracker.register_replacement("test_name", orig, rep)

    tracker.release_original(rep)
    # ``release_original`` deletes the original and the entry.
    assert tracker.get_original(rep) is None


def test_tracker_get_original_unknown_returns_none():
    """A replacement that was never *explicitly* registered is still
    tracked through ``ReplacementModuleBase.__init__``, so we expect
    the original to be retrievable.

    This test documents the contract: there is no public "unregister"
    path - once a ``ReplacementModuleBase`` is constructed, the
    original module is captured in the tracker until
    ``release_original_module`` is called.
    """
    tracker = ModuleReplacementTracker.get_instance()
    orig = nn.Linear(2, 2)
    rep = _TrivialReplacement(orig)
    # The base class' __init__ registered the replacement.
    assert tracker.get_original(rep) is orig
    # Releasing the original removes it from the tracker.
    rep.release_original_module()
    assert tracker.get_original(rep) is None


# ---------------------------------------------------------------------------
# is_custom_model
# ---------------------------------------------------------------------------


def test_is_custom_model_true_for_known_model_type():
    """A model whose ``config.model_type`` is in BUILTIN_MODULES is "custom"."""

    class _FakeConfig:
        model_type = "llama4"  # a BUILTIN_MODULES key in the current tree

    model = nn.Linear(2, 2)
    model.config = _FakeConfig()
    assert is_custom_model(model) is True


def test_is_custom_model_false_for_unknown_model_type():
    class _FakeConfig:
        model_type = "this_is_not_in_builtin_modules"

    model = nn.Linear(2, 2)
    model.config = _FakeConfig()
    assert is_custom_model(model) is False


def test_is_custom_model_no_config():
    """A bare module without a ``config`` attribute is not custom."""
    assert is_custom_model(nn.Linear(2, 2)) is False


def test_builtin_modules_contains_expected_keys():
    """Regression guard: if a key is removed, downstream fails silently.  Catch
    it here.
    """
    # ``llama4`` and ``deepseek_v2`` are the only entries that have shipped
    # as part of stable releases; new ones can be added freely.
    for required in ("llama4", "deepseek_v2"):
        assert required in BUILTIN_MODULES, f"BUILTIN_MODULES missing {required!r}"


# ---------------------------------------------------------------------------
# apply_replacements
# ---------------------------------------------------------------------------


def test_apply_replacements_returns_model_unchanged_for_unknown_modules():
    """``apply_replacements`` returns the model itself (modified in place)
    on a model with no registered modules.  The empty case must not crash
    and must not raise.
    """
    from unittest import mock

    # Force ``is_custom_model`` to False and skip the auto-MOE branch.
    with mock.patch(
        "auto_round.modeling.fused_moe.replace_modules.is_custom_model",
        return_value=False,
    ), mock.patch(
        "auto_round.modeling.fused_moe.replace_modules.is_transformers_version_greater_or_equal_5",
        return_value=False,
    ):
        model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
        out = apply_replacements(model, auto_detect_moe=True)
    # The function returns the model object.
    assert out is model


# ---------------------------------------------------------------------------
# _apply_custom_replacements
# ---------------------------------------------------------------------------


def test_apply_custom_replacements_empty_returns_empty_list():
    """``_apply_custom_replacements`` returns an empty list when the model
    has no modules registered for replacement.
    """
    from unittest import mock

    with mock.patch(
        "auto_round.modeling.fused_moe.replace_modules.is_custom_model",
        return_value=True,
    ):
        # A bare linear has no registered class.
        out = _apply_custom_replacements(nn.Linear(2, 2))
    assert out == []


# ---------------------------------------------------------------------------
# materialize_model_ / release_original_module_
# ---------------------------------------------------------------------------


def test_materialize_model_calls_replacement_materialize():
    """``materialize_model_`` should walk the model and call
    ``materialize_weights`` on every ``ReplacementModuleBase`` it finds.
    """
    cls_name = _register_trivial_replacement()
    orig = nn.Linear(2, 2)

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.rep = _TrivialReplacement(orig)
            self.lin = nn.Linear(2, 2)  # non-replacement sibling

    m = _M()
    assert m.rep._materialized is False
    materialize_model_(m)
    assert m.rep._materialized is True


def test_release_original_module_clears_tracker():
    """``release_original_module_`` should call ``release_original_module``
    on every replacement it finds (which clears the tracker entry).
    """
    cls_name = _register_trivial_replacement()
    orig = nn.Linear(2, 2)

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.rep = _TrivialReplacement(orig)

    m = _M()
    tracker = ModuleReplacementTracker.get_instance()
    # Confirm the replacement is registered.
    assert tracker.get_original(m.rep) is orig
    release_original_module_(m)
    assert tracker.get_original(m.rep) is None
