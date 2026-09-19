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
"""Unit tests for ``auto_round.algorithms.transforms.hadamard.dispatcher``."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from auto_round.algorithms.transforms.hadamard.config import RotationConfig
from auto_round.algorithms.transforms.hadamard.dispatcher import (
    _to_config,
    apply_hadamard_rotation,
    resolve_hadamard_backend,
)


class TestToConfig:
    """Test _to_config helper that normalises rotation_config input."""

    def test_none_returns_default_rotation_config(self):
        """``None`` input becomes a default :class:`RotationConfig` (with no inferred block_size)."""
        cfg = _to_config(None, "mx_fp")
        assert isinstance(cfg, RotationConfig)
        assert cfg.backend == "auto"
        # Note: RotationConfig.model_validate({}) returns the bare default.
        # block_size inference happens in apply_hadamard_rotation via normalize_rotation_config.
        assert cfg.hadamard_type == "hadamard"

    def test_dict_input_returns_rotation_config(self):
        """Dict input is converted to a :class:`RotationConfig`."""
        cfg = _to_config({"backend": "inplace", "hadamard_type": "hadamard"}, "mx_fp")
        assert isinstance(cfg, RotationConfig)
        assert cfg.backend == "inplace"
        assert cfg.hadamard_type == "hadamard"

    def test_str_input_returns_rotation_config(self):
        """String shorthand is normalised and validated."""
        cfg = _to_config("hadamard", "mx_fp")
        assert isinstance(cfg, RotationConfig)
        assert cfg.hadamard_type == "hadamard"

    def test_existing_rotation_config_is_validated_and_returned(self):
        """An existing :class:`RotationConfig` is validated and re-emitted as a fresh one."""
        original = RotationConfig(backend="transform", hadamard_type="random_hadamard", block_size=16)
        result = _to_config(original, "nv_fp")
        # _to_config normalizes through pydantic validation, producing a fresh
        # but equal-valued config object.
        assert isinstance(result, RotationConfig)
        assert result.backend == "transform"
        assert result.hadamard_type == "random_hadamard"
        assert result.block_size == 16

    def test_invalid_dict_raises_value_error(self):
        """Invalid data in dict raises ``ValueError`` from pydantic validator."""
        with pytest.raises(ValueError):
            _to_config({"backend": "bogus_backend"}, "mx_fp")


class TestResolveHadamardBackend:
    """Test resolve_hadamard_backend function — the central routing logic."""

    def test_backend_inplace_returns_inplace(self):
        """Explicit ``backend='inplace'`` returns ``'inplace'``."""
        cfg = RotationConfig(backend="inplace", hadamard_type="hadamard")
        assert resolve_hadamard_backend(cfg, "mx_fp") == "inplace"

    def test_backend_inplace_strips_prefix_from_hadamard_type(self):
        """``backend='inplace'`` strips the ``inplace_`` prefix from hadamard_type."""
        cfg = RotationConfig(backend="inplace", hadamard_type="inplace_hadamard")
        assert resolve_hadamard_backend(cfg, "mx_fp") == "inplace"
        assert cfg.hadamard_type == "hadamard"

    def test_backend_inplace_with_inplace_quarot_hadamard_strips_prefix(self):
        """``backend='inplace'`` strips ``inplace_`` prefix even with ``quarot`` type."""
        cfg = RotationConfig(backend="inplace", hadamard_type="inplace_quarot_hadamard")
        assert resolve_hadamard_backend(cfg, "mx_fp") == "inplace"
        assert cfg.hadamard_type == "quarot_hadamard"

    def test_inplace_hadamard_type_without_explicit_backend_routes_to_inplace(self):
        """``hadamard_type`` containing ``inplace`` keyword routes to inplace."""
        cfg = RotationConfig(backend="auto", hadamard_type="inplace_random")
        assert resolve_hadamard_backend(cfg, "mx_fp") == "inplace"
        # The prefix should be stripped
        assert cfg.hadamard_type == "random"

    def test_backend_transform_returns_transform_for_mx_fp(self):
        """Explicit ``backend='transform'`` with mx_fp returns ``'transform'``."""
        cfg = RotationConfig(backend="transform", hadamard_type="hadamard")
        assert resolve_hadamard_backend(cfg, "mx_fp") == "transform"

    def test_backend_transform_returns_transform_for_nv_fp(self):
        """Explicit ``backend='transform'`` with nv_fp returns ``'transform'``."""
        cfg = RotationConfig(backend="transform", hadamard_type="hadamard")
        assert resolve_hadamard_backend(cfg, "nv_fp") == "transform"

    def test_backend_transform_with_fuse_raises(self):
        """Fuse + transform raises ValueError (transform cannot fuse)."""
        cfg = RotationConfig(
            backend="transform",
            hadamard_type="hadamard",
            fuse_online_to_weight=True,
            allow_online_rotation=True,
        )
        with pytest.raises(ValueError, match="does not support fuse_online_to_weight=True"):
            resolve_hadamard_backend(cfg, "mx_fp")

    def test_backend_transform_with_non_fp_data_type_raises(self):
        """``backend='transform'`` requires MXFP4/NVFP4 — other dtypes raise."""
        cfg = RotationConfig(backend="transform", hadamard_type="hadamard", allow_online_rotation=True)
        with pytest.raises(ValueError, match="only supports MXFP4 / NVFP4"):
            resolve_hadamard_backend(cfg, "int")

    def test_backend_transform_with_no_online_rotation_raises(self):
        """``backend='transform'`` requires ``allow_online_rotation=True``."""
        cfg = RotationConfig(backend="transform", hadamard_type="hadamard", allow_online_rotation=False)
        with pytest.raises(ValueError, match="only supports `allow_online_rotation`=True"):
            resolve_hadamard_backend(cfg, "mx_fp")

    def test_auto_backend_with_fuse_returns_inplace(self):
        """``auto`` + ``fuse_online_to_weight=True`` → ``inplace``."""
        cfg = RotationConfig(backend="auto", hadamard_type="hadamard", fuse_online_to_weight=True)
        assert resolve_hadamard_backend(cfg, "mx_fp") == "inplace"

    def test_auto_backend_with_mx_fp_returns_transform(self):
        """``auto`` + MXFP data type → ``transform``."""
        cfg = RotationConfig(backend="auto", hadamard_type="hadamard")
        assert resolve_hadamard_backend(cfg, "mx_fp") == "transform"

    def test_auto_backend_with_nv_fp_returns_transform(self):
        """``auto`` + NVFP data type → ``transform``."""
        cfg = RotationConfig(backend="auto", hadamard_type="hadamard")
        assert resolve_hadamard_backend(cfg, "nv_fp") == "transform"

    def test_auto_backend_with_int_returns_inplace(self):
        """``auto`` + non-FP data type → ``inplace``."""
        cfg = RotationConfig(backend="auto", hadamard_type="hadamard")
        assert resolve_hadamard_backend(cfg, "int") == "inplace"

    def test_auto_backend_with_fp_returns_inplace(self):
        """``auto`` + generic ``fp`` data type → ``inplace``."""
        cfg = RotationConfig(backend="auto", hadamard_type="hadamard")
        assert resolve_hadamard_backend(cfg, "fp") == "inplace"

    def test_auto_fuse_takes_priority_over_fp_dtype(self):
        """``auto`` + fuse overrides any data type routing."""
        cfg = RotationConfig(backend="auto", hadamard_type="hadamard", fuse_online_to_weight=True)
        assert resolve_hadamard_backend(cfg, "nv_fp") == "inplace"


class TestApplyHadamardRotation:
    """Test apply_hadamard_rotation dispatcher entry point."""

    def test_inplace_backend_dispatches_to_inplace_module(self):
        """``backend='inplace'`` routes to the inplace apply function."""
        model = nn.Linear(8, 8)
        cfg = RotationConfig(backend="inplace", hadamard_type="hadamard", block_size=None)

        mock_apply = MagicMock(return_value=(model, []))

        with patch(
            "auto_round.algorithms.transforms.hadamard.inplace.apply_rotation_transform",
            mock_apply,
        ):
            result_model, hooks = apply_hadamard_rotation(model, cfg, "mx_fp", compute_device="cpu")

        assert result_model is model
        mock_apply.assert_called_once()
        assert hooks == []
        # After dispatch, _rotation_config is set on the model (a normalized
        # version of the input cfg).
        assert hasattr(model, "_rotation_config")
        assert isinstance(model._rotation_config, RotationConfig)
        assert model._rotation_config.backend == "inplace"

    def test_inplace_with_block_size_passes_as_group_size(self):
        """``block_size > 0`` is forwarded as ``group_size`` to inplace."""
        model = nn.Linear(8, 8)
        cfg = RotationConfig(backend="inplace", hadamard_type="hadamard", block_size=64)
        mock_apply = MagicMock(return_value=(model, []))

        with patch(
            "auto_round.algorithms.transforms.hadamard.inplace.apply_rotation_transform",
            mock_apply,
        ):
            apply_hadamard_rotation(model, cfg, "mx_fp", compute_device="cpu")

        _, kwargs = mock_apply.call_args
        assert kwargs["group_size"] == 64

    def test_inplace_with_zero_block_size_passes_none_group_size(self):
        """``block_size <= 0`` is forwarded as ``group_size=None``."""
        model = nn.Linear(8, 8)
        cfg = RotationConfig(backend="inplace", hadamard_type="hadamard", block_size=0)
        mock_apply = MagicMock(return_value=(model, []))

        with patch(
            "auto_round.algorithms.transforms.hadamard.inplace.apply_rotation_transform",
            mock_apply,
        ):
            apply_hadamard_rotation(model, cfg, "mx_fp", compute_device="cpu")

        _, kwargs = mock_apply.call_args
        assert kwargs["group_size"] is None

    def test_inplace_with_none_block_size_passes_default_mx_group_size(self):
        """``block_size=None`` triggers mx_fp default 32 in normalization."""
        model = nn.Linear(8, 8)
        cfg = RotationConfig(backend="inplace", hadamard_type="hadamard", block_size=None)
        mock_apply = MagicMock(return_value=(model, []))

        with patch(
            "auto_round.algorithms.transforms.hadamard.inplace.apply_rotation_transform",
            mock_apply,
        ):
            apply_hadamard_rotation(model, cfg, "mx_fp", compute_device="cpu")

        _, kwargs = mock_apply.call_args
        # mx_fp default block_size = 32 → forwarded as group_size
        assert kwargs["group_size"] == 32

    def test_inplace_fuse_flag_forwarded(self):
        """``fuse_online_to_weight`` is forwarded to the inplace backend."""
        model = nn.Linear(8, 8)
        cfg = RotationConfig(
            backend="inplace",
            hadamard_type="hadamard",
            block_size=None,
            fuse_online_to_weight=True,
        )
        mock_apply = MagicMock(return_value=(model, []))

        with patch(
            "auto_round.algorithms.transforms.hadamard.inplace.apply_rotation_transform",
            mock_apply,
        ):
            apply_hadamard_rotation(model, cfg, "mx_fp", compute_device="cpu")

        _, kwargs = mock_apply.call_args
        assert kwargs["fuse_online_to_weight"] is True

    def test_transform_backend_dispatches_to_apply_module(self):
        """``backend='transform'`` routes to the apply module."""
        model = nn.Linear(8, 8)
        cfg = RotationConfig(backend="transform", hadamard_type="hadamard")
        mock_apply = MagicMock(return_value=model)

        with patch(
            "auto_round.algorithms.transforms.hadamard.apply.apply_rotation_transform",
            mock_apply,
        ):
            result = apply_hadamard_rotation(model, cfg, "mx_fp", compute_device="cpu")

        assert result is model
        mock_apply.assert_called_once()
        # The dispatcher passes cfg positionally as the 2nd arg.
        args, kwargs = mock_apply.call_args
        assert args[0] is model
        # The cfg is normalised through _to_config, so it is a *new* RotationConfig
        # with the same content.
        assert isinstance(args[1], RotationConfig)
        assert args[1].backend == "transform"
        assert args[1].hadamard_type == "hadamard"
        assert kwargs.get("data_type") == "mx_fp"

    def test_transform_with_unsupported_hadamard_type_raises(self):
        """``backend='transform'`` only supports ``hadamard`` or ``random_hadamard``."""
        # We cannot construct such a RotationConfig directly (pydantic validator
        # rejects it), so we build a SimpleNamespace mimicking the dispatcher's
        # expected attributes and patch resolve_hadamard_backend to return it.
        fake_cfg = SimpleNamespace(
            backend="transform",
            hadamard_type="inplace_hadamard",
            allow_online_rotation=True,
            block_size=32,
            fuse_online_to_weight=None,
        )
        model = nn.Linear(8, 8)
        with patch(
            "auto_round.algorithms.transforms.hadamard.dispatcher.resolve_hadamard_backend",
            return_value="transform",
        ):
            with patch(
                "auto_round.algorithms.transforms.hadamard.dispatcher._to_config",
                return_value=fake_cfg,
            ):
                with pytest.raises(ValueError, match="only supports hadamard or random_hadamard"):
                    apply_hadamard_rotation(model, fake_cfg, "mx_fp", compute_device="cpu")

    def test_rotation_config_stored_on_model(self):
        """After apply, ``_rotation_config`` is set on the model (inplace path)."""
        model = nn.Linear(8, 8)
        cfg = RotationConfig(backend="inplace", hadamard_type="hadamard", block_size=None)

        with patch(
            "auto_round.algorithms.transforms.hadamard.inplace.apply_rotation_transform",
            MagicMock(return_value=(model, [])),
        ):
            apply_hadamard_rotation(model, cfg, "mx_fp", compute_device="cpu")

        assert hasattr(model, "_rotation_config")
        assert isinstance(model._rotation_config, RotationConfig)
        assert model._rotation_config.backend == "inplace"

    def test_string_rotation_config_routes_correctly(self):
        """String rotation_config shorthand is normalised and dispatched."""
        model = nn.Linear(8, 8)

        # "hadamard" string is dispatched to "transform" backend for mx_fp.
        # Mock the apply module so we don't actually rotate weights.
        with patch(
            "auto_round.algorithms.transforms.hadamard.apply.apply_rotation_transform",
            MagicMock(return_value=model),
        ):
            result = apply_hadamard_rotation(model, "hadamard", "mx_fp", compute_device="cpu")

        # For "hadamard" string with mx_fp, dispatcher routes to transform backend
        # which returns just the model (not a tuple).
        assert result is model

    def test_dict_rotation_config_accepted(self):
        """Dict rotation_config shorthand is normalised."""
        model = nn.Linear(8, 8)

        with patch(
            "auto_round.algorithms.transforms.hadamard.inplace.apply_rotation_transform",
            MagicMock(return_value=(model, [])),
        ):
            result_model, _ = apply_hadamard_rotation(
                model,
                {"backend": "inplace", "hadamard_type": "hadamard"},
                "mx_fp",
                compute_device="cpu",
            )

        assert result_model is model

    def test_fuse_flag_with_no_env_returns_none(self):
        """When ``fuse_online_to_weight`` is None and env unset, the original None is forwarded."""
        model = nn.Linear(8, 8)
        # We override the field to None explicitly to bypass default behavior
        cfg = RotationConfig(backend="inplace", hadamard_type="hadamard", block_size=None, fuse_online_to_weight=None)
        assert cfg.fuse_online_to_weight is None

        mock_apply = MagicMock(return_value=(model, []))

        with patch(
            "auto_round.algorithms.transforms.hadamard.inplace.apply_rotation_transform",
            mock_apply,
        ):
            apply_hadamard_rotation(model, cfg, "mx_fp", compute_device="cpu")

        _, kwargs = mock_apply.call_args
        # When env var is False (default) and config is None, the dispatcher
        # forwards None — the inplace module decides based on model class.
        assert kwargs["fuse_online_to_weight"] is None

    def test_fuse_flag_false_in_config(self):
        """When ``fuse_online_to_weight=False``, it is forwarded as ``False``."""
        model = nn.Linear(8, 8)
        cfg = RotationConfig(
            backend="inplace",
            hadamard_type="hadamard",
            block_size=None,
            fuse_online_to_weight=False,
        )
        mock_apply = MagicMock(return_value=(model, []))

        with patch(
            "auto_round.algorithms.transforms.hadamard.inplace.apply_rotation_transform",
            mock_apply,
        ):
            apply_hadamard_rotation(model, cfg, "mx_fp", compute_device="cpu")

        _, kwargs = mock_apply.call_args
        assert kwargs["fuse_online_to_weight"] is False

    def test_fuse_flag_true_in_config(self):
        """When ``fuse_online_to_weight=True``, it is forwarded as ``True``."""
        model = nn.Linear(8, 8)
        cfg = RotationConfig(
            backend="inplace",
            hadamard_type="hadamard",
            block_size=None,
            fuse_online_to_weight=True,
        )
        mock_apply = MagicMock(return_value=(model, []))

        with patch(
            "auto_round.algorithms.transforms.hadamard.inplace.apply_rotation_transform",
            mock_apply,
        ):
            apply_hadamard_rotation(model, cfg, "mx_fp", compute_device="cpu")

        _, kwargs = mock_apply.call_args
        assert kwargs["fuse_online_to_weight"] is True

    def test_compute_device_forwarded(self):
        """The ``compute_device`` argument is forwarded to the inplace backend."""
        model = nn.Linear(8, 8)
        cfg = RotationConfig(backend="inplace", hadamard_type="hadamard", block_size=None)
        mock_apply = MagicMock(return_value=(model, []))

        with patch(
            "auto_round.algorithms.transforms.hadamard.inplace.apply_rotation_transform",
            mock_apply,
        ):
            apply_hadamard_rotation(model, cfg, "mx_fp", compute_device="cuda:0")

        _, kwargs = mock_apply.call_args
        assert kwargs["compute_device"] == "cuda:0"

    def test_allow_online_rotation_forwarded(self):
        """The ``allow_online_rotation`` flag is forwarded to the inplace backend."""
        model = nn.Linear(8, 8)
        cfg = RotationConfig(backend="inplace", hadamard_type="hadamard", block_size=None, allow_online_rotation=False)
        mock_apply = MagicMock(return_value=(model, []))

        with patch(
            "auto_round.algorithms.transforms.hadamard.inplace.apply_rotation_transform",
            mock_apply,
        ):
            apply_hadamard_rotation(model, cfg, "mx_fp", compute_device="cpu")

        _, kwargs = mock_apply.call_args
        assert kwargs["allow_online_rotation"] is False

    def test_rotation_matrix_forwarded(self):
        """The ``hadamard_type`` is forwarded as ``rotation_matrix`` to the inplace backend.

        Note: when ``backend='inplace'`` is explicitly set, the dispatcher
        strips an ``inplace_`` prefix from hadamard_type. We use
        ``inplace_random`` so the dispatcher correctly yields ``random``.
        """
        model = nn.Linear(8, 8)
        cfg = RotationConfig(backend="inplace", hadamard_type="inplace_random", block_size=None)
        mock_apply = MagicMock(return_value=(model, []))

        with patch(
            "auto_round.algorithms.transforms.hadamard.inplace.apply_rotation_transform",
            mock_apply,
        ):
            apply_hadamard_rotation(model, cfg, "mx_fp", compute_device="cpu")

        _, kwargs = mock_apply.call_args
        assert kwargs["rotation_matrix"] == "random"
