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

* **solve_triangular fallback**: the module also installs a wrapper around
  ``torch.linalg.solve_triangular`` (``patch_solve_triangular()``) that
  probes for a native kernel at runtime and, for ``(device, dtype)``
  combinations that raise ``NotImplementedError``, falls back to an fp32
  solve cast back to the promoted dtype.  The wrapper factory
  ``_make_solve_triangular_wrapper()`` is tested directly against fakes so
  no real broken kernel is required.
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


# ---------------------------------------------------------------------------
# torch.linalg.solve_triangular runtime-probed fp32 fallback
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_solve_triangular(monkeypatch):
    """Guarantee ``torch.linalg.solve_triangular`` is the pristine torch
    implementation for the duration of the test, and restore the previous
    state afterwards.

    The wrapper installed by ``patch_solve_triangular()`` carries an
    ``_ar_orig_solve_triangular`` back-reference, so any number of wrapping
    layers can be unwound to reach the original builtin.
    """
    import torch

    current = torch.linalg.solve_triangular
    while getattr(current, "_ar_solve_triangular_patched", False):
        current = current._ar_orig_solve_triangular
    monkeypatch.setattr(torch.linalg, "solve_triangular", current)
    yield current


def _make_lower_triangular(n, dtype, device="cpu"):
    """Unit lower-triangular matrix: identity with ones below the diagonal."""
    import torch

    return torch.eye(n, dtype=dtype, device=device) + torch.tril(
        torch.ones(n, n, dtype=dtype, device=device), diagonal=-1
    )


def test_solve_triangular_not_patched_when_hpu_unavailable(fresh_hpu_patch, clean_solve_triangular, monkeypatch):
    """On a non-HPU host the wrapper must not be installed."""
    import torch

    monkeypatch.setattr("auto_round.utils.is_hpex_available", lambda: False)

    import auto_round.modeling.hpu_patch  # noqa: F401

    assert not getattr(torch.linalg.solve_triangular, "_ar_solve_triangular_patched", False)
    assert torch.linalg.solve_triangular is clean_solve_triangular


def test_solve_triangular_patch_is_idempotent(fresh_hpu_patch, clean_solve_triangular, monkeypatch):
    """Repeated ``patch_solve_triangular()`` calls must install exactly one
    wrapper layer whose back-reference reaches the original builtin."""
    import torch

    monkeypatch.setattr("auto_round.utils.is_hpex_available", lambda: True)

    import auto_round.modeling.hpu_patch as hpu_patch

    hpu_patch.patch_solve_triangular()
    wrapper1 = torch.linalg.solve_triangular
    hpu_patch.patch_solve_triangular()
    wrapper2 = torch.linalg.solve_triangular

    assert wrapper1 is wrapper2
    assert wrapper1._ar_solve_triangular_patched is True
    # The import-time call (HPU mocked available above) and the explicit
    # calls must converge on the same single wrapper.
    assert wrapper1._ar_orig_solve_triangular is clean_solve_triangular


def test_solve_triangular_supported_dtype_uses_native_path(fresh_hpu_patch, clean_solve_triangular, monkeypatch):
    """When the native kernel handles the (device, dtype) pair, results and
    dtypes pass through untouched and the pair is cached as supported."""
    import torch

    real = clean_solve_triangular
    seen = []

    def spy(A, B, **kwargs):
        seen.append((A.dtype, B.dtype))
        return real(A, B, **kwargs)

    import auto_round.modeling.hpu_patch as hpu_patch

    wrapper = hpu_patch._make_solve_triangular_wrapper(spy)
    monkeypatch.setattr(torch.linalg, "solve_triangular", wrapper)

    A = _make_lower_triangular(4, torch.float32)
    B = torch.randn(4, 2, dtype=torch.float32)
    result = torch.linalg.solve_triangular(A, B, upper=False, unitriangular=True)
    again = torch.linalg.solve_triangular(A, B, upper=False, unitriangular=True)

    assert result.dtype == torch.float32
    assert torch.allclose(result, real(A, B, upper=False, unitriangular=True))
    assert torch.allclose(again, result)
    # Each call invokes the original exactly once, always in the original dtype.
    assert seen == [(torch.float32, torch.float32)] * 2
    assert wrapper._ar_support_cache[("cpu", torch.float32, torch.float32)] is True


def test_solve_triangular_falls_back_to_fp32_when_kernel_missing(fresh_hpu_patch, clean_solve_triangular, monkeypatch):
    """When the native kernel is missing for a (device, dtype) pair, the
    wrapper solves in fp32 and casts back to the promoted dtype; the negative
    result is cached so later calls skip the probe."""
    import torch

    real = clean_solve_triangular
    seen = []

    def fake_orig(A, B, **kwargs):
        seen.append(A.dtype)
        if A.dtype == torch.bfloat16:
            raise NotImplementedError('triangular_solve_cpu not implemented for "BFloat16"')
        return real(A, B, **kwargs)

    import auto_round.modeling.hpu_patch as hpu_patch

    wrapper = hpu_patch._make_solve_triangular_wrapper(fake_orig)
    monkeypatch.setattr(torch.linalg, "solve_triangular", wrapper)

    A = _make_lower_triangular(4, torch.bfloat16)
    B = torch.randn(4, 2, dtype=torch.float32).to(torch.bfloat16)
    result = torch.linalg.solve_triangular(A, B, upper=False, unitriangular=True)

    expected = real(A.float(), B.float(), upper=False, unitriangular=True).to(torch.bfloat16)
    assert result.dtype == torch.bfloat16
    assert torch.allclose(result, expected, rtol=1e-2, atol=1e-2)
    # First call: the probe with bf16 failed, then the fp32 solve ran.
    assert seen == [torch.bfloat16, torch.float32]
    assert wrapper._ar_support_cache[("cpu", torch.bfloat16, torch.bfloat16)] is False

    # Second call: probe skipped, straight to the fp32 solve.
    seen.clear()
    again = torch.linalg.solve_triangular(A, B, upper=False, unitriangular=True)
    assert torch.allclose(again, expected, rtol=1e-2, atol=1e-2)
    assert seen == [torch.float32]


def test_solve_triangular_out_kwarg_passes_through(fresh_hpu_patch, clean_solve_triangular, monkeypatch):
    """With a caller-provided out= tensor the wrapper must not take the fp32
    detour (it could not write into the caller's tensor); the call is passed
    through untouched."""
    import torch

    captured = {}
    sentinel = object()

    def fake_orig(A, B, **kwargs):
        captured["A"] = A
        captured["B"] = B
        captured["kwargs"] = kwargs
        return sentinel

    import auto_round.modeling.hpu_patch as hpu_patch

    wrapper = hpu_patch._make_solve_triangular_wrapper(fake_orig)

    A = _make_lower_triangular(3, torch.bfloat16)
    B = torch.randn(3, 2, dtype=torch.float32).to(torch.bfloat16)
    out = torch.empty(3, 2, dtype=torch.bfloat16)

    result = wrapper(A, B, upper=False, unitriangular=True, out=out)

    assert result is sentinel
    assert captured["A"] is A
    assert captured["B"] is B
    assert captured["kwargs"]["out"] is out
    assert wrapper._ar_support_cache == {}  # never probed


def test_solve_triangular_runtime_error_propagates_and_is_not_cached(
    fresh_hpu_patch, clean_solve_triangular, monkeypatch
):
    """Errors other than NotImplementedError (e.g. a singular matrix) must
    propagate to the caller and must not poison the capability cache."""
    import torch

    calls = []

    def fake_orig(A, B, **kwargs):
        calls.append(A.dtype)
        raise RuntimeError("simulated singular matrix")

    import auto_round.modeling.hpu_patch as hpu_patch

    wrapper = hpu_patch._make_solve_triangular_wrapper(fake_orig)
    A = _make_lower_triangular(3, torch.float32)
    B = torch.randn(3, 2, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="singular matrix"):
        wrapper(A, B, upper=False, unitriangular=True)

    assert calls == [torch.float32]  # no fp32 retry
    assert wrapper._ar_support_cache == {}
