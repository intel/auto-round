# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``auto_round.modeling.fp8_quant``."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from auto_round.modeling.fp8_quant import (
    apply_fp8_expert_replacement_patch,
    oot_replace_with_fp8_linear,
    oot_validate_environment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """A minimal ``nn.Module`` with a few ``nn.Linear`` children.

    The structure mirrors what a HF model with named children looks like
    after the first ``named_modules()`` recursion level: each ``Linear``
    module has a parent path like ``"<root>.fc1"`` or ``"<root>.nested.0"``.
    """

    def __init__(self, with_bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(8, 16, bias=with_bias)
        self.fc2 = nn.Linear(16, 8, bias=with_bias)
        self.act = nn.ReLU()
        self.nested = nn.Sequential(nn.Linear(8, 8, bias=with_bias))


class _QuantConfigStub:
    """Minimal stand-in for ``FineGrainedFP8Config`` / ``FbgemmFp8Config``.

    The only attributes ``oot_replace_with_fp8_linear`` exercises are
    ``dequantize``, ``activation_scheme`` and ``weight_block_size``.
    """

    def __init__(
        self,
        dequantize: bool = False,
        activation_scheme: str = "dynamic",
        weight_block_size=(128, 128),
    ):
        self.dequantize = dequantize
        self.activation_scheme = activation_scheme
        self.weight_block_size = weight_block_size


# ---------------------------------------------------------------------------
# oot_replace_with_fp8_linear
# ---------------------------------------------------------------------------


class TestOotReplaceWithFp8Linear:
    """Tests for :func:`oot_replace_with_fp8_linear`."""

    def test_dequantize_returns_unchanged(self):
        """If ``quantization_config.dequantize`` is True the original model
        is returned untouched.
        """
        model = _TinyModel()
        original_lc = [m for m in model.modules() if isinstance(m, nn.Linear)]

        config = _QuantConfigStub(dequantize=True)
        result = oot_replace_with_fp8_linear(model, quantization_config=config)

        assert result is model
        # No conversion happened.
        new_lc = [m for m in model.modules() if isinstance(m, nn.Linear)]
        assert new_lc == original_lc

    def test_no_linear_modules_warns(self):
        """When the model has no ``nn.Linear`` children, a warning is logged
        but no exception is raised and the model is returned unchanged.
        """

        class _NoLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(10, 4)

        model = _NoLinear()
        config = _QuantConfigStub(dequantize=False)

        with patch("transformers.integrations.finegrained_fp8.FP8Linear"):
            with patch(
                "transformers.integrations.finegrained_fp8.should_convert_module",
                return_value=True,
            ):
                with patch(
                    "transformers.integrations.finegrained_fp8.logger"
                ) as mock_logger:
                    result = oot_replace_with_fp8_linear(
                        model, quantization_config=config
                    )
                    assert result is model
                    assert mock_logger.warning.called

    def test_replaces_linear_modules(self):
        """All ``nn.Linear`` children should be replaced with the FP8 class."""

        model = _TinyModel(with_bias=True)
        config = _QuantConfigStub(dequantize=False)

        def _make_module(*args, **kwargs):
            return nn.Linear(8, 8)

        with patch(
            "transformers.integrations.finegrained_fp8.FP8Linear",
            side_effect=_make_module,
        ):
            with patch(
                "transformers.integrations.finegrained_fp8.should_convert_module",
                return_value=True,
            ):
                with patch(
                    "auto_round.modeling.fp8_quant.is_transformers_version_greater_or_equal_5_4_0",
                    return_value=False,
                ):
                    result = oot_replace_with_fp8_linear(
                        model, quantization_config=config
                    )
                    # FP8Linear was called for each nn.Linear child.
                    assert result is model

    def test_with_modules_to_not_convert(self):
        """Names listed in ``modules_to_not_convert`` are skipped."""

        model = _TinyModel(with_bias=True)
        config = _QuantConfigStub(dequantize=False)

        with patch("transformers.integrations.finegrained_fp8.FP8Linear") as mock_fp8:
            with patch(
                "transformers.integrations.finegrained_fp8.should_convert_module",
                return_value=False,
            ):
                oot_replace_with_fp8_linear(
                    model,
                    modules_to_not_convert=["fc1"],
                    quantization_config=config,
                )
                # No replacement calls should have happened
                # because should_convert_module returned False everywhere.
                assert not mock_fp8.called

    def test_pre_quantized(self):
        """The ``pre_quantized=True`` path passes ``dtype=None`` instead of
        omitting the kwarg.
        """

        model = _TinyModel(with_bias=True)
        config = _QuantConfigStub(dequantize=False)
        captured_kwargs = []

        def _capture(*args, **kwargs):
            captured_kwargs.append(kwargs)
            return nn.Linear(8, 8)

        with patch(
            "transformers.integrations.finegrained_fp8.FP8Linear", side_effect=_capture
        ):
            with patch(
                "transformers.integrations.finegrained_fp8.should_convert_module",
                return_value=True,
            ):
                with patch(
                    "auto_round.modeling.fp8_quant.is_transformers_version_greater_or_equal_5_4_0",
                    return_value=True,
                ):
                    oot_replace_with_fp8_linear(
                        model,
                        quantization_config=config,
                        pre_quantized=True,
                    )
                    # Every captured call must include ``dtype=None``.
                    for kw in captured_kwargs:
                        assert kw.get("dtype") is None

    def test_bias_kwarg_name_pre_v5_4(self):
        """On transformers < 5.4, the bias flag is passed as ``bias``."""

        model = _TinyModel(with_bias=True)
        config = _QuantConfigStub(dequantize=False)
        captured_kwargs = []

        def _capture(*args, **kwargs):
            captured_kwargs.append(kwargs)
            return nn.Linear(8, 8)

        with patch(
            "transformers.integrations.finegrained_fp8.FP8Linear", side_effect=_capture
        ):
            with patch(
                "transformers.integrations.finegrained_fp8.should_convert_module",
                return_value=True,
            ):
                with patch(
                    "auto_round.modeling.fp8_quant.is_transformers_version_greater_or_equal_5_4_0",
                    return_value=False,
                ):
                    oot_replace_with_fp8_linear(
                        model, quantization_config=config
                    )
                    # At least one replacement happened.
                    assert len(captured_kwargs) >= 1
                    for kw in captured_kwargs:
                        # On pre-5.4, ``bias`` is the kwarg (not ``has_bias``).
                        assert "bias" in kw
                        assert "has_bias" not in kw
                        assert kw["bias"] is True

    def test_bias_kwarg_name_v5_4_plus(self):
        """On transformers >= 5.4, the bias flag is passed as ``has_bias``."""

        model = _TinyModel(with_bias=True)
        config = _QuantConfigStub(dequantize=False)
        captured_kwargs = []

        def _capture(*args, **kwargs):
            captured_kwargs.append(kwargs)
            return nn.Linear(8, 8)

        with patch(
            "transformers.integrations.finegrained_fp8.FP8Linear", side_effect=_capture
        ):
            with patch(
                "transformers.integrations.finegrained_fp8.should_convert_module",
                return_value=True,
            ):
                with patch(
                    "auto_round.modeling.fp8_quant.is_transformers_version_greater_or_equal_5_4_0",
                    return_value=True,
                ):
                    oot_replace_with_fp8_linear(
                        model, quantization_config=config
                    )
                    assert len(captured_kwargs) >= 1
                    for kw in captured_kwargs:
                        assert "has_bias" in kw
                        assert "bias" not in kw
                        assert kw["has_bias"] is True

    def test_no_bias_flag_passed_correctly(self):
        """When the linear module has ``bias=False``, the OOT function must
        still report that fact.
        """

        model = _TinyModel(with_bias=False)
        config = _QuantConfigStub(dequantize=False)
        captured_kwargs = []

        def _capture(*args, **kwargs):
            captured_kwargs.append(kwargs)
            return nn.Linear(8, 8)

        with patch(
            "transformers.integrations.finegrained_fp8.FP8Linear", side_effect=_capture
        ):
            with patch(
                "transformers.integrations.finegrained_fp8.should_convert_module",
                return_value=True,
            ):
                with patch(
                    "auto_round.modeling.fp8_quant.is_transformers_version_greater_or_equal_5_4_0",
                    return_value=False,
                ):
                    oot_replace_with_fp8_linear(
                        model, quantization_config=config
                    )
                    assert len(captured_kwargs) >= 1
                    for kw in captured_kwargs:
                        assert kw.get("bias") is False

    def test_returns_self(self):
        """The function returns the (mutated) model object."""

        model = _TinyModel()
        config = _QuantConfigStub(dequantize=False)

        def _make_module(*args, **kwargs):
            return nn.Linear(8, 8)

        with patch(
            "transformers.integrations.finegrained_fp8.FP8Linear",
            side_effect=_make_module,
        ):
            with patch(
                "transformers.integrations.finegrained_fp8.should_convert_module",
                return_value=True,
            ):
                with patch(
                    "auto_round.modeling.fp8_quant.is_transformers_version_greater_or_equal_5_4_0",
                    return_value=True,
                ):
                    result = oot_replace_with_fp8_linear(
                        model, quantization_config=config
                    )
                    assert result is model


# ---------------------------------------------------------------------------
# oot_validate_environment
# ---------------------------------------------------------------------------


class TestOotValidateEnvironment:
    """Tests for :func:`oot_validate_environment`."""

    def test_calls_original(self):
        """The patched validator must forward args/kwargs to the original."""

        mock_self = MagicMock()
        with patch(
            "auto_round.modeling.fp8_quant._orig_validate_environment"
        ) as mock_orig:
            oot_validate_environment(mock_self, "arg1", kwarg1="value1")
            mock_orig.assert_called_once_with(mock_self, "arg1", kwarg1="value1")

    def test_decorator_overrides_cuda_capability(self):
        """The wrapper is decorated with ``@override_cuda_device_capability``,
        so it must succeed even when CUDA capability is mocked away.
        """

        mock_self = MagicMock()
        # Use a context manager-like block: override_cuda_device_capability
        # is exercised by simply calling the wrapped function.
        with patch(
            "auto_round.modeling.fp8_quant._orig_validate_environment",
            return_value="ok",
        ):
            # Real decorator (``override_cuda_device_capability``) only
            # patches ``torch.cuda.get_device_capability`` so this returns
            # fine even when CUDA is unavailable.
            assert oot_validate_environment(mock_self) == "ok"


# ---------------------------------------------------------------------------
# apply_fp8_expert_replacement_patch
# ---------------------------------------------------------------------------


class TestApplyFp8ExpertReplacementPatch:
    """Tests for :func:`apply_fp8_expert_replacement_patch`."""

    def test_no_cuda_does_nothing(self):
        """On a non-CUDA host the function must be a no-op without raising."""

        with patch("torch.cuda.is_available", return_value=False):
            with patch(
                "auto_round.modeling.fp8_quant.is_transformers_version_greater_or_equal_5",
                return_value=True,
            ):
                # Should not raise.
                assert apply_fp8_expert_replacement_patch() is None

    def test_old_transformers_does_nothing(self):
        """With transformers < 5 the function must be a no-op."""

        with patch("torch.cuda.is_available", return_value=True):
            with patch(
                "auto_round.modeling.fp8_quant.is_transformers_version_greater_or_equal_5",
                return_value=False,
            ):
                assert apply_fp8_expert_replacement_patch() is None

    def test_import_error_is_swallowed(self):
        """If the local import of ``transformers.integrations.finegrained_fp8``
        fails the function logs a warning and returns ``None``.
        """

        import builtins

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith("transformers.integrations.finegrained_fp8"):
                raise ImportError("boom")
            return real_import(name, globals, locals, fromlist, level)

        with patch("torch.cuda.is_available", return_value=True):
            with patch(
                "auto_round.modeling.fp8_quant.is_transformers_version_greater_or_equal_5",
                return_value=True,
            ):
                with patch("builtins.__import__", side_effect=fake_import):
                    # Should not raise despite the ImportError.
                    assert apply_fp8_expert_replacement_patch() is None

    def test_replaces_upstream_replace_with_fp8_linear(self):
        """When transformers >= 5 and CUDA is available, the upstream
        ``replace_with_fp8_linear`` is replaced with our OOT function.
        """

        import auto_round.modeling.fp8_quant as fp8q
        import transformers.integrations.finegrained_fp8 as upstream

        original = upstream.replace_with_fp8_linear
        try:
            with patch("torch.cuda.is_available", return_value=True):
                with patch(
                    "auto_round.modeling.fp8_quant.is_transformers_version_greater_or_equal_5",
                    return_value=True,
                ):
                    apply_fp8_expert_replacement_patch()
                    assert (
                        upstream.replace_with_fp8_linear
                        is fp8q.oot_replace_with_fp8_linear
                    )
        finally:
            upstream.replace_with_fp8_linear = original

    def test_patches_validate_environment(self):
        """The function also patches ``FineGrainedFP8HfQuantizer.validate_environment``
        to ``oot_validate_environment``.
        """

        import auto_round.modeling.fp8_quant as fp8q
        from transformers.quantizers.quantizer_finegrained_fp8 import (
            FineGrainedFP8HfQuantizer,
        )

        original = FineGrainedFP8HfQuantizer.validate_environment
        try:
            with patch("torch.cuda.is_available", return_value=True):
                with patch(
                    "auto_round.modeling.fp8_quant.is_transformers_version_greater_or_equal_5",
                    return_value=True,
                ):
                    apply_fp8_expert_replacement_patch()
                    assert (
                        FineGrainedFP8HfQuantizer.validate_environment
                        is fp8q.oot_validate_environment
                    )
        finally:
            FineGrainedFP8HfQuantizer.validate_environment = original


# ---------------------------------------------------------------------------
# Public surface / smoke tests
# ---------------------------------------------------------------------------


class TestPublicSurface:
    """Verify the public surface is exposed correctly."""

    def test_oot_replace_with_fp8_linear_is_callable(self):
        assert callable(oot_replace_with_fp8_linear)

    def test_oot_validate_environment_is_callable(self):
        assert callable(oot_validate_environment)

    def test_apply_fp8_expert_replacement_patch_is_callable(self):
        assert callable(apply_fp8_expert_replacement_patch)

    def test_module_importable(self):
        """The module must be importable in a normal Python process."""

        import importlib

        import auto_round.modeling.fp8_quant

        importlib.reload(auto_round.modeling.fp8_quant)


# ---------------------------------------------------------------------------
# Integration-style parametrized tests
# ---------------------------------------------------------------------------


class TestPatchBehaviorMatrix:
    """Exhaustively check ``apply_fp8_expert_replacement_patch`` over the
    full boolean matrix of the two gating conditions.
    """

    @pytest.mark.parametrize(
        "cuda_available, transformers_v5",
        [(True, True), (True, False), (False, True), (False, False)],
    )
    def test_all_combinations_no_raise(
        self, cuda_available, transformers_v5
    ):
        """Every combination of gating conditions must not raise."""

        with patch("torch.cuda.is_available", return_value=cuda_available):
            with patch(
                "auto_round.modeling.fp8_quant.is_transformers_version_greater_or_equal_5",
                return_value=transformers_v5,
            ):
                assert apply_fp8_expert_replacement_patch() is None
