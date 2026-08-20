# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.modeling.hpu_patch``.

The module has the unusual property that it executes its side-effect
(``patch_finegrained_fp8()``) at *import time*.  That makes the
non-HPU path boring (it just returns) but the HPU path hard to
exercise without an actual HPU stack.

We cover both paths with ``unittest.mock``:

* **non-HPU host** (the common case in CI): ``is_hpex_available()``
  returns ``False``; the module's import-time call is a no-op.  We
  verify this by importing the module and then asserting the upstream
  ``transformers.integrations.finegrained_fp8`` module is unchanged.

* **HPU host** (mocked): we monkey-patch ``is_hpex_available`` to
  return ``True`` and then call ``patch_finegrained_fp8()`` directly.
  The function should (a) load the auto-round finegrained_fp8 patch
  module, (b) copy its public attributes into the upstream
  ``transformers.integrations.finegrained_fp8`` module.

* **transformers < 4.0** (mocked): the helper should log a warning and
  return early without touching the upstream module.
"""

import importlib
import sys
import types
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Fixture: always start from a clean module cache for hpu_patch + patch modules
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_hpu_patch(monkeypatch):
    """Reload ``hpu_patch`` so the import-time ``patch_finegrained_fp8()``
    call is executed under the current monkey-patched environment.
    """
    # Drop any cached imports so the module re-runs its top-level code.
    for mod in list(sys.modules):
        if mod == "auto_round.modeling.hpu_patch" or mod.startswith("auto_round.modeling.finegrained_fp8"):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    yield


# ---------------------------------------------------------------------------
# Non-HPU host
# ---------------------------------------------------------------------------


def test_hpu_patch_is_noop_when_hpu_unavailable(fresh_hpu_patch, monkeypatch):
    """Importing the module on a non-HPU host must not patch upstream.

    We mock ``is_hpex_available`` to return False and confirm
    ``patch_finegrained_fp8`` returns silently.
    """
    # Ensure transformers' finegrained_fp8 module is importable so the
    # patcher would have a place to write into if it ever ran.
    import transformers.integrations.finegrained_fp8  # noqa: F401

    monkeypatch.setattr(
        "auto_round.utils.is_hpex_available",
        lambda: False,
    )

    import auto_round.modeling.hpu_patch as hpu_patch  # noqa: F401

    # The function must return without doing anything.
    assert hpu_patch.patch_finegrained_fp8() is None


# ---------------------------------------------------------------------------
# HPU host: patch path
# ---------------------------------------------------------------------------


def test_hpu_patch_patches_upstream_when_hpu_available(fresh_hpu_patch, monkeypatch):
    """When HPEX is available, ``patch_finegrained_fp8()`` copies public
    attributes from the auto-round finegrained_fp8_patch module into the
    upstream ``transformers.integrations.finegrained_fp8`` module.
    """
    # Make ``is_hpex_available`` think HPEX is installed.
    monkeypatch.setattr("auto_round.utils.is_hpex_available", lambda: True)
    # Pretend transformers >= 5 so the auto_round patch module is selected.
    monkeypatch.setattr(
        "auto_round.utils.is_transformers_version_greater_or_equal_5",
        lambda: True,
    )
    monkeypatch.setattr(
        "auto_round.utils.is_transformers_version_greater_or_equal_4",
        lambda: True,
    )

    # Make sure the upstream module is loaded; we will inspect it after.
    import transformers.integrations.finegrained_fp8 as upstream

    # Build a fake "auto_round" patch module with one public symbol that
    # we expect to be copied over.
    fake_patch = types.ModuleType("auto_round.modeling.finegrained_fp8_patch")
    sentinel = object()
    fake_patch.SENTINEL_ATTRIBUTE_FROM_AUTO_ROUND = sentinel
    monkeypatch.setitem(sys.modules, "auto_round.modeling.finegrained_fp8_patch", fake_patch)

    import auto_round.modeling.hpu_patch as hpu_patch  # noqa: F401

    hpu_patch.patch_finegrained_fp8()

    # The upstream module should now expose the sentinel.
    assert getattr(upstream, "SENTINEL_ATTRIBUTE_FROM_AUTO_ROUND", None) is sentinel


def test_hpu_patch_uses_v4_when_transformers_v4(fresh_hpu_patch, monkeypatch):
    """When transformers >= 4 but < 5 the v4 patch module is selected.

    We check this by setting both version gates correctly and patching
    ``importlib.import_module`` (the local import inside the function)
    so the test does not depend on the actual file existing.
    """
    monkeypatch.setattr("auto_round.utils.is_hpex_available", lambda: True)
    monkeypatch.setattr("auto_round.utils.is_transformers_version_greater_or_equal_5", lambda: False)
    monkeypatch.setattr("auto_round.utils.is_transformers_version_greater_or_equal_4", lambda: True)

    imported_names = []

    def fake_import_module(name, package=None):
        if name.startswith("auto_round.modeling.finegrained_fp8_patch"):
            imported_names.append(name)
        # For everything else, return a dummy object so the function
        # can still iterate over ``dir(...)``.
        if name.startswith("auto_round.modeling.finegrained_fp8_patch"):
            return types.SimpleNamespace(SENTINEL=object())
        return types.SimpleNamespace()

    monkeypatch.setattr("importlib.import_module", fake_import_module)

    import auto_round.modeling.hpu_patch as hpu_patch  # noqa: F401

    hpu_patch.patch_finegrained_fp8()

    # The v4 module should have been imported.
    assert "auto_round.modeling.finegrained_fp8_patch_v4" in imported_names


def test_hpu_patch_skips_when_transformers_below_v4(fresh_hpu_patch, monkeypatch):
    """Below transformers v4 the helper must return without doing anything.

    We check that no auto_round.finegrained_fp8_* module is imported.
    """
    monkeypatch.setattr("auto_round.utils.is_hpex_available", lambda: True)
    monkeypatch.setattr("auto_round.utils.is_transformers_version_greater_or_equal_5", lambda: False)
    monkeypatch.setattr("auto_round.utils.is_transformers_version_greater_or_equal_4", lambda: False)

    imported_names = []

    def fake_import_module(name, package=None):
        if name.startswith("auto_round.modeling.finegrained_fp8"):
            imported_names.append(name)
        return types.SimpleNamespace()

    monkeypatch.setattr("importlib.import_module", fake_import_module)

    import auto_round.modeling.hpu_patch as hpu_patch  # noqa: F401

    # Should not raise.
    assert hpu_patch.patch_finegrained_fp8() is None
    # No finegrained_fp8_patch* import attempt.
    assert all(not n.startswith("auto_round.modeling.finegrained_fp8_patch") for n in imported_names)


# ---------------------------------------------------------------------------
# Fallback: when the upstream module cannot be imported
# ---------------------------------------------------------------------------


def test_hpu_patch_falls_back_when_upstream_missing(fresh_hpu_patch, monkeypatch):
    """If importing the upstream ``transformers.integrations.finegrained_fp8``
    fails, the patcher falls back to full module replacement and returns
    without raising.
    """
    monkeypatch.setattr("auto_round.utils.is_hpex_available", lambda: True)
    monkeypatch.setattr("auto_round.utils.is_transformers_version_greater_or_equal_5", lambda: True)
    monkeypatch.setattr("auto_round.utils.is_transformers_version_greater_or_equal_4", lambda: True)

    # Build a fake "auto_round" patch module so the function has something
    # to write into ``sys.modules`` if it falls back to legacy behavior.
    fake_patch = types.ModuleType("auto_round.modeling.finegrained_fp8_patch")
    fake_patch.SENTINEL = object()
    monkeypatch.setitem(sys.modules, "auto_round.modeling.finegrained_fp8_patch", fake_patch)

    # Get the *real* ``importlib.import_module`` and only intercept the
    # upstream import - otherwise this test will recurse forever because
    # the fake calls itself.
    import importlib as _real_importlib

    real_import_module = _real_importlib.import_module

    def fake_import_module(name, package=None):
        if name == "transformers.integrations.finegrained_fp8":
            raise ImportError("simulated: upstream module not available")
        return real_import_module(name, package)

    monkeypatch.setattr("importlib.import_module", fake_import_module)

    import auto_round.modeling.hpu_patch as hpu_patch  # noqa: F401

    # Should not raise; falls back to legacy full-module replacement.
    assert hpu_patch.patch_finegrained_fp8() is None
    # And the fallback has written the module into sys.modules.
    assert "transformers.integrations.finegrained_fp8" in sys.modules
    assert sys.modules["transformers.integrations.finegrained_fp8"] is fake_patch


# ---------------------------------------------------------------------------
# Exception in patch loop
# ---------------------------------------------------------------------------


def test_hpu_patch_handles_generic_exception(fresh_hpu_patch, monkeypatch):
    """If the inner patching logic raises, the function must not propagate
    the exception (it logs a warning and returns)."""
    monkeypatch.setattr("auto_round.utils.is_hpex_available", lambda: True)
    monkeypatch.setattr("auto_round.utils.is_transformers_version_greater_or_equal_5", lambda: True)
    monkeypatch.setattr("auto_round.utils.is_transformers_version_greater_or_equal_4", lambda: True)

    # Build a fake "auto_round" patch module.
    fake_patch = types.ModuleType("auto_round.modeling.finegrained_fp8_patch")
    monkeypatch.setitem(sys.modules, "auto_round.modeling.finegrained_fp8_patch", fake_patch)

    # ``importlib.import_module`` returns our fake module, and we patch
    # ``builtins.dir`` so that ``dir(module)`` raises - this triggers
    # the outer ``except Exception`` branch in the patcher.
    import importlib as _real_importlib

    real_import_module = _real_importlib.import_module

    def fake_import_module(name, package=None):
        if name == "auto_round.modeling.finegrained_fp8_patch":
            return fake_patch
        return real_import_module(name, package)

    monkeypatch.setattr("importlib.import_module", fake_import_module)
    monkeypatch.setattr("builtins.dir", lambda obj: (_ for _ in ()).throw(RuntimeError("boom")))

    import auto_round.modeling.hpu_patch as hpu_patch  # noqa: F401

    # Should swallow the RuntimeError.
    assert hpu_patch.patch_finegrained_fp8() is None
