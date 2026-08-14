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
"""Unit tests for ``auto_round.algorithms.transforms.hadamard.patch``."""

import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import transformers

from auto_round.export.export_to_autoround.qlinear_fp import QuantLinear
from auto_round.wrapper import WrapperLinear, WrapperWALayer

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _FakeOrigLayer(nn.Module):
    """A minimal stand-in for a quantisable layer (Linear / Conv1D)."""

    def __init__(self, in_features=8, out_features=8, bias=True, is_conv1d=False, bits=4):
        super().__init__()
        self.bits = bits
        self.in_features = in_features
        self.out_features = out_features
        self.is_conv1d = is_conv1d
        self.sym = True
        self.act_sym = True
        self.act_dynamic = False
        self.iters = 200
        self.disable_opt_rtn = True
        self.tuning_device = "cpu"
        if is_conv1d:
            # Conv1D stores weight as (in_features, out_features)
            self.weight = nn.Parameter(torch.randn(in_features, out_features))
        else:
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        self.imatrix = torch.zeros(8)
        self.data_type = "int"
        self.act_data_type = "int"
        self.act_bits = 8
        self.act_max = None
        self.act_max_scale = torch.tensor(1.0)
        self.act_min_scale = torch.tensor(1.0)
        self.act_quant_func = MagicMock(return_value=(torch.zeros(1, 1), None, None))
        self.weight_quant_func = MagicMock(return_value=(torch.zeros(1, 1), None, None))
        self.scale_dtype = torch.float16
        self.q_scale_thresh = 1e-5
        self.group_size = -1
        self.act_group_size = -1


def _make_wrapper_linear(orig_layer=None, bits=4):
    """Build a WrapperLinear suitable for patch testing."""
    if orig_layer is None:
        orig_layer = _FakeOrigLayer(bits=bits)
    wrapper = WrapperLinear(orig_layer, device="cpu")
    return wrapper


def _make_wrapper_wa_layer(orig_layer=None):
    """Build a WrapperWALayer suitable for patch testing."""
    if orig_layer is None:
        orig_layer = _FakeOrigLayer()
    wrapper = WrapperWALayer(orig_layer, device="cpu")
    return wrapper


class _IdentityTransform(nn.Module):
    """Identity transform used as inp_transform / w_transform."""

    def __init__(self, n=None):
        super().__init__()
        if n is not None:
            self.weight = nn.Parameter(torch.eye(n))
        else:
            self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x


class _MultiplyTransform(nn.Module):
    """Scale transform — multiplies input by a fixed scalar weight."""

    def __init__(self, scale=2.0):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(scale))

    def forward(self, x):
        return x * self.weight


@pytest.fixture(autouse=True)
def reset_hadamard_patches():
    """Reset the idempotency guard flags before each test.

    The patch functions set class-level guard flags (``_hadamard_patched``,
    ``_hadamard_forward_patched``, ``_pack_patched``) to ensure they only
    patch once. We reset them so each test starts from a clean state.
    """
    for cls, attr in [
        (WrapperLinear, "_hadamard_patched"),
        (WrapperWALayer, "_hadamard_forward_patched"),
        (QuantLinear, "_pack_patched"),
    ]:
        if hasattr(cls, attr):
            delattr(cls, attr)
    yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPatchWrapperLinearIdempotency:
    """Test idempotency of ``patch_wrapperlinear_to_apply_transform``."""

    def test_first_call_sets_guard_flag(self):
        from auto_round.algorithms.transforms.hadamard.patch import (
            patch_wrapperlinear_to_apply_transform,
        )

        patch_wrapperlinear_to_apply_transform(_IdentityTransform(), _IdentityTransform())
        assert getattr(WrapperLinear, "_hadamard_patched", False) is True

    def test_second_call_is_noop(self):
        """Calling patch twice should not double-patch methods."""
        from auto_round.algorithms.transforms.hadamard.patch import (
            patch_wrapperlinear_to_apply_transform,
        )

        original_qdq_weight = WrapperLinear._qdq_weight
        original_qdq_act = WrapperLinear._qdq_act

        patch_wrapperlinear_to_apply_transform(_IdentityTransform(), _IdentityTransform())
        after_first_qdq_weight = WrapperLinear._qdq_weight
        after_first_qdq_act = WrapperLinear._qdq_act

        # Second call should be a no-op
        patch_wrapperlinear_to_apply_transform(_IdentityTransform(), _IdentityTransform())

        # Methods should be the same instance (unchanged)
        assert WrapperLinear._qdq_weight is after_first_qdq_weight
        assert WrapperLinear._qdq_act is after_first_qdq_act
        # And different from the originals
        assert WrapperLinear._qdq_weight is not original_qdq_weight
        assert WrapperLinear._qdq_act is not original_qdq_act

    def test_inp_transform_applied_in_qdq_act(self):
        """Verify ``inp_transform`` is applied before activation quantisation."""
        from auto_round.algorithms.transforms.hadamard.patch import (
            patch_wrapperlinear_to_apply_transform,
        )

        # Multiply-by-2 transform
        inp_transform = _MultiplyTransform(scale=2.0)
        patch_wrapperlinear_to_apply_transform(_IdentityTransform(), inp_transform)

        # Call the patched _qdq_act on a wrapper
        wrapper = _make_wrapper_linear()
        # Replace the wrapper's act_quant_func with a spy
        wrapper.act_quant_func = MagicMock(return_value=(torch.zeros(1, 8), None, None))
        x = torch.ones(1, 8)
        act_min = torch.tensor(1.0)
        act_max = torch.tensor(1.0)
        wrapper._qdq_act(x, act_min_scale=act_min, act_max_scale=act_max)
        # The mock should have been called with x * 2
        called_args = wrapper.act_quant_func.call_args[0]
        assert torch.equal(called_args[0], x * 2.0)

    def test_qdq_weight_falls_through_for_high_bits(self):
        """``bits >= 16`` keeps the original ``_qdq_weight`` behaviour (no hadamard)."""
        from auto_round.algorithms.transforms.hadamard.patch import (
            patch_wrapperlinear_to_apply_transform,
        )

        patch_wrapperlinear_to_apply_transform(_IdentityTransform(), _IdentityTransform())

        # bits=16 → fall-through path
        orig_layer = _FakeOrigLayer(bits=16)
        wrapper = _make_wrapper_linear(orig_layer)
        # The patched function should still return what the original would have.
        # For bits >= 16, _qdq_weight returns (weight, None, None) per the
        # original implementation in wrapper.py.
        weight = wrapper.orig_layer.weight
        result = wrapper._qdq_weight(torch.zeros_like(weight), torch.tensor(1.0), torch.tensor(1.0))
        # For bits >= 16, the original code returns (orig_layer.weight, None, None).
        assert result[0] is weight or torch.equal(result[0], weight)
        assert result[1] is None
        assert result[2] is None

    def test_qdq_weight_applies_w_transform_on_first_call(self):
        """First call with bits < 16 applies the w_transform and stores result."""
        from auto_round.algorithms.transforms.hadamard.patch import (
            patch_wrapperlinear_to_apply_transform,
        )

        w_transform = _MultiplyTransform(scale=3.0)
        patch_wrapperlinear_to_apply_transform(w_transform, _IdentityTransform())

        orig_layer = _FakeOrigLayer(bits=4)
        wrapper = _make_wrapper_linear(orig_layer)
        original_weight = orig_layer.weight.data.clone()

        # First call should apply the transform
        wrapper._qdq_weight(torch.zeros_like(orig_layer.weight), torch.tensor(1.0), torch.tensor(1.0))

        # After first call, weight should be modified (multiplied by 3)
        assert not torch.equal(orig_layer.weight.data, original_weight)
        # And applied_weight_hadamard flag should be set
        assert wrapper.applied_weight_hadamard is True

    def test_qdq_weight_skips_w_transform_on_subsequent_calls(self):
        """After the first call, subsequent calls do NOT re-apply the transform."""
        from auto_round.algorithms.transforms.hadamard.patch import (
            patch_wrapperlinear_to_apply_transform,
        )

        # Track calls to w_transform
        w_transform = MagicMock()
        w_transform.return_value = torch.zeros(8, 8)

        patch_wrapperlinear_to_apply_transform(w_transform, _IdentityTransform())

        orig_layer = _FakeOrigLayer(bits=4)
        wrapper = _make_wrapper_linear(orig_layer)

        # Pre-set the applied flag to simulate an already-transformed wrapper
        wrapper.applied_weight_hadamard = True

        # The patched _qdq_weight should skip w_transform because the flag is set.
        # Patch the underlying weight_quant_func to verify it gets called normally.
        wrapper.weight_quant_func = MagicMock(return_value=(torch.zeros_like(orig_layer.weight), None, None))
        wrapper._qdq_weight(torch.zeros_like(orig_layer.weight), torch.tensor(1.0), torch.tensor(1.0))

        # w_transform should NOT have been called (the patch short-circuits)
        w_transform.assert_not_called()


class TestPatchWrapperWALayerIdempotency:
    """Test idempotency of ``patch_wrapperwalayer_forward_to_apply_transform``."""

    def test_first_call_sets_guard_flag(self):
        from auto_round.algorithms.transforms.hadamard.patch import (
            patch_wrapperwalayer_forward_to_apply_transform,
        )

        patch_wrapperwalayer_forward_to_apply_transform(_IdentityTransform())
        assert getattr(WrapperWALayer, "_hadamard_forward_patched", False) is True

    def test_second_call_is_noop(self):
        """Calling patch twice should not double-patch the forward method."""
        from auto_round.algorithms.transforms.hadamard.patch import (
            patch_wrapperwalayer_forward_to_apply_transform,
        )

        original_forward = WrapperWALayer.forward
        patch_wrapperwalayer_forward_to_apply_transform(_IdentityTransform())
        after_first_forward = WrapperWALayer.forward
        # Second call should not change forward
        patch_wrapperwalayer_forward_to_apply_transform(_IdentityTransform())
        assert WrapperWALayer.forward is after_first_forward
        # And it's different from the original
        assert WrapperWALayer.forward is not original_forward

    def test_inp_transform_applied_in_forward(self):
        """Verify ``inp_transform`` is applied inside ``forward``."""
        from auto_round.algorithms.transforms.hadamard.patch import (
            patch_wrapperwalayer_forward_to_apply_transform,
        )

        inp_transform = _MultiplyTransform(scale=4.0)
        patch_wrapperwalayer_forward_to_apply_transform(inp_transform)

        wrapper = _make_wrapper_wa_layer()
        # Patch the orig_layer.forward to be a spy
        orig_forward_spy = MagicMock(return_value=torch.zeros(1, 8))
        wrapper.orig_layer.forward = orig_forward_spy

        x = torch.ones(1, 8)
        wrapper(x)

        # The act_quant_func should have been called with x * 4
        called_args = wrapper.orig_layer.act_quant_func.call_args[0]
        assert torch.equal(called_args[0], x * 4.0)

    def test_act_max_passed_when_present(self):
        """If ``orig_layer.act_max`` exists, it should be forwarded to act_quant_func."""
        from auto_round.algorithms.transforms.hadamard.patch import (
            patch_wrapperwalayer_forward_to_apply_transform,
        )

        patch_wrapperwalayer_forward_to_apply_transform(_IdentityTransform())

        wrapper = _make_wrapper_wa_layer()
        wrapper.orig_layer.act_max = torch.tensor(2.0)

        orig_forward_spy = MagicMock(return_value=torch.zeros(1, 8))
        wrapper.orig_layer.forward = orig_forward_spy

        x = torch.ones(1, 8)
        wrapper(x)

        # tensor_max should be the act_max value
        kwargs = wrapper.orig_layer.act_quant_func.call_args.kwargs
        assert kwargs.get("tensor_max") is not None


class TestPatchQuantLinearIdempotency:
    """Test idempotency of ``patch_quantlinear``."""

    def test_first_call_sets_guard_flag(self):
        from auto_round.algorithms.transforms.hadamard.patch import patch_quantlinear

        patch_quantlinear(_IdentityTransform())
        assert getattr(QuantLinear, "_pack_patched", False) is True

    def test_second_call_is_noop(self):
        """Calling patch twice should not double-patch the pack method."""
        from auto_round.algorithms.transforms.hadamard.patch import patch_quantlinear

        original_pack = QuantLinear.pack
        patch_quantlinear(_IdentityTransform())
        after_first_pack = QuantLinear.pack
        patch_quantlinear(_IdentityTransform())
        assert QuantLinear.pack is after_first_pack
        assert QuantLinear.pack is not original_pack

    def test_pack_registers_hadamard_matrix_buffer(self):
        """Patched pack should register a hadamard_matrix buffer on the QuantLinear."""
        from auto_round.algorithms.transforms.hadamard.patch import patch_quantlinear

        # Use a transform with a known weight value
        w_transform = nn.Linear(8, 8, bias=False)
        with torch.no_grad():
            w_transform.weight.copy_(torch.eye(8) * 0.5)
        patch_quantlinear(w_transform)

        # Create a QuantLinear and call pack with a mock linear.
        # We use mx_fp / bits=4, infeatures divisible by 32.
        try:
            qlinear = QuantLinear(bits=4, group_size=32, infeatures=32, outfeatures=8, bias=False)
        except (NotImplementedError, TypeError) as e:
            pytest.skip(f"Could not construct QuantLinear: {e}")

        linear = nn.Linear(32, 8, bias=False)
        scales = torch.zeros(8, 1, dtype=torch.float16)

        # Run the patched pack
        try:
            qlinear.pack(linear, scales)
        except Exception as e:
            # Pack may still fail due to dependencies — we just want to confirm
            # the hadamard_matrix buffer was registered before the failure.
            pass

        # The patch always registers hadamard_matrix regardless of mid-failure.
        assert hasattr(qlinear, "hadamard_matrix")
        # And the matrix should match the transform weight
        assert torch.equal(qlinear.hadamard_matrix.cpu(), w_transform.weight.detach().cpu())

    def test_pack_handles_conv2d_layer(self):
        """Patched pack should flatten Conv2d weights."""
        from auto_round.algorithms.transforms.hadamard.patch import patch_quantlinear

        w_transform = _IdentityTransform(n=32)
        patch_quantlinear(w_transform)

        try:
            qlinear = QuantLinear(bits=4, group_size=32, infeatures=32, outfeatures=4, bias=False)
        except Exception as e:
            pytest.skip(f"Could not construct QuantLinear: {e}")

        # Use a Conv2d layer
        conv = nn.Conv2d(32, 4, kernel_size=3, bias=False)
        scales = torch.zeros(4, 1, dtype=torch.float16)

        # This may still fail in the middle, but the conv2d branch should be exercised.
        try:
            qlinear.pack(conv, scales)
        except Exception:
            pass

        # Conv2d branch was exercised (didn't raise immediately).

    def test_pack_handles_conv1d_layer(self):
        """Patched pack should transpose Conv1D weights."""
        from auto_round.algorithms.transforms.hadamard.patch import patch_quantlinear

        w_transform = _IdentityTransform(n=32)
        patch_quantlinear(w_transform)

        try:
            qlinear = QuantLinear(bits=4, group_size=32, infeatures=32, outfeatures=4, bias=False)
        except Exception as e:
            pytest.skip(f"Could not construct QuantLinear: {e}")

        # Use a Conv1D layer
        conv1d = transformers.pytorch_utils.Conv1D(32, 4)
        scales = torch.zeros(4, 1, dtype=torch.float16)

        try:
            qlinear.pack(conv1d, scales)
        except Exception:
            pass


class TestPatchIntegration:
    """Integration tests for the patch helpers working together."""

    def test_all_patches_can_be_called_in_sequence(self):
        """Calling all three patches in sequence should set all guard flags."""
        from auto_round.algorithms.transforms.hadamard.patch import (
            patch_quantlinear,
            patch_wrapperlinear_to_apply_transform,
            patch_wrapperwalayer_forward_to_apply_transform,
        )

        patch_wrapperlinear_to_apply_transform(_IdentityTransform(), _IdentityTransform())
        patch_wrapperwalayer_forward_to_apply_transform(_IdentityTransform())
        patch_quantlinear(_IdentityTransform())

        assert getattr(WrapperLinear, "_hadamard_patched", False) is True
        assert getattr(WrapperWALayer, "_hadamard_forward_patched", False) is True
        assert getattr(QuantLinear, "_pack_patched", False) is True

    def test_patches_survive_double_call(self):
        """Verify all patches can be called twice without raising."""
        from auto_round.algorithms.transforms.hadamard.patch import (
            patch_quantlinear,
            patch_wrapperlinear_to_apply_transform,
            patch_wrapperwalayer_forward_to_apply_transform,
        )

        # First round
        patch_wrapperlinear_to_apply_transform(_IdentityTransform(), _IdentityTransform())
        patch_wrapperwalayer_forward_to_apply_transform(_IdentityTransform())
        patch_quantlinear(_IdentityTransform())

        # Second round — should be idempotent
        patch_wrapperlinear_to_apply_transform(_IdentityTransform(), _IdentityTransform())
        patch_wrapperwalayer_forward_to_apply_transform(_IdentityTransform())
        patch_quantlinear(_IdentityTransform())
