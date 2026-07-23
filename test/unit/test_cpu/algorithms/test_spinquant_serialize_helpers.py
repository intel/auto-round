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
"""Tests for small pure helpers in ``spinquant/serialize.py`` and
``spinquant/training.py`` that are reachable on CPU without a real model
checkpoint or a GPU.
"""

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# _is_quantlinear / _has_spinquant_buffers
# ---------------------------------------------------------------------------
class TestIsQuantLinear:
    def test_plain_linear_returns_false(self):
        from auto_round.algorithms.transforms.spinquant.serialize import (
            _is_quantlinear,
        )

        assert _is_quantlinear(nn.Linear(4, 4)) is False

    def test_named_quant_linear_returns_true(self):
        """Class name with the suffix ``QuantLinear`` should be detected."""
        from auto_round.algorithms.transforms.spinquant.serialize import (
            _is_quantlinear,
        )

        class MyQuantLinear(nn.Linear):
            pass

        # Class name "MyQuantLinear" contains "QuantLinear"
        assert _is_quantlinear(MyQuantLinear(4, 4)) is True

    def test_quant_linear_subclass_name(self):
        from auto_round.algorithms.transforms.spinquant.serialize import (
            _is_quantlinear,
        )

        class QuantLinear(nn.Linear):
            pass

        assert _is_quantlinear(QuantLinear(4, 4)) is True

    def test_nvfp4_quant_linear_name(self):
        from auto_round.algorithms.transforms.spinquant.serialize import (
            _is_quantlinear,
        )

        class NVFP4QuantLinear(nn.Linear):
            pass

        # Class name contains "QuantLinear" -> True
        assert _is_quantlinear(NVFP4QuantLinear(4, 4)) is True

    def test_qmodule_base_subclass(self):
        from auto_round.algorithms.transforms.spinquant.serialize import (
            _is_quantlinear,
        )

        class QModuleBase(nn.Module):
            pass

        class _MyQ(QModuleBase):
            pass

        assert _is_quantlinear(_MyQ()) is True


class TestHasSpinquantBuffers:
    def test_plain_linear_returns_false(self):
        from auto_round.algorithms.transforms.spinquant.serialize import (
            _has_spinquant_buffers,
        )

        assert _has_spinquant_buffers(nn.Linear(4, 4)) is False

    def test_module_with_r1_type_attribute_returns_true(self):
        from auto_round.algorithms.transforms.spinquant.serialize import (
            _has_spinquant_buffers,
        )

        m = nn.Linear(4, 4)
        m.spinquant_r1_type = "online"
        assert _has_spinquant_buffers(m) is True

    def test_module_with_r4_type_attribute_returns_true(self):
        from auto_round.algorithms.transforms.spinquant.serialize import (
            _has_spinquant_buffers,
        )

        m = nn.Linear(4, 4)
        m.spinquant_r4_type = "block"
        assert _has_spinquant_buffers(m) is True


# ---------------------------------------------------------------------------
# _get_online_r1_target_names / _get_r4_target_names
# ---------------------------------------------------------------------------
class TestTargetNames:
    def _build_mini_lm(self):
        """Build a tiny model with attn and mlp projections."""

        class _Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(4, 4)
                self.k_proj = nn.Linear(4, 4)
                self.v_proj = nn.Linear(4, 4)
                self.o_proj = nn.Linear(4, 4)  # not in R1

        class _MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(4, 4)
                self.up_proj = nn.Linear(4, 4)
                self.down_proj = nn.Linear(4, 4)

        class _Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = _Attn()
                self.mlp = _MLP()

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([_Block()])

        return _Model()

    def test_r1_targets_qkv_gate_up(self):
        from auto_round.algorithms.transforms.spinquant.serialize import (
            _get_online_r1_target_names,
        )

        m = self._build_mini_lm()
        targets = _get_online_r1_target_names(m)
        # q/k/v/o_proj of attn + gate/up/down_proj of mlp, but o_proj is excluded
        assert "layers.0.attn.q_proj" in targets
        assert "layers.0.attn.k_proj" in targets
        assert "layers.0.attn.v_proj" in targets
        assert "layers.0.mlp.gate_proj" in targets
        assert "layers.0.mlp.up_proj" in targets
        # o_proj is NOT in the R1 list
        assert "layers.0.attn.o_proj" not in targets
        # down_proj is NOT in the R1 list
        assert "layers.0.mlp.down_proj" not in targets

    def test_r4_targets_only_down_proj(self):
        from auto_round.algorithms.transforms.spinquant.serialize import (
            _get_r4_target_names,
        )

        m = self._build_mini_lm()
        targets = _get_r4_target_names(m)
        assert targets == {"layers.0.mlp.down_proj"}


# ---------------------------------------------------------------------------
# _get_stored_rotation
# ---------------------------------------------------------------------------
class TestGetStoredRotation:
    def test_returns_none_when_missing(self):
        from auto_round.algorithms.transforms.spinquant.serialize import (
            _get_stored_rotation,
        )

        model = nn.Linear(4, 4)
        assert _get_stored_rotation(model, "spinquant_R1") is None

    def test_returns_tensor_when_present(self):
        from auto_round.algorithms.transforms.spinquant.serialize import (
            _get_stored_rotation,
        )

        model = nn.Linear(4, 4)
        model.spinquant_R1 = nn.Parameter(torch.eye(4))
        result = _get_stored_rotation(model, "spinquant_R1")
        assert result is not None
        assert torch.equal(result, torch.eye(4))

    def test_ignores_non_tensor_attribute(self):
        from auto_round.algorithms.transforms.spinquant.serialize import (
            _get_stored_rotation,
        )

        model = nn.Linear(4, 4)
        model.spinquant_R1 = "not_a_tensor"
        # Non-tensor attributes should return None
        assert _get_stored_rotation(model, "spinquant_R1") is None


# ---------------------------------------------------------------------------
# _get_hidden_size / _get_head_dim / _get_intermediate_size
# ---------------------------------------------------------------------------
class TestConfigExtractors:
    def test_hidden_size_from_config(self):
        from auto_round.algorithms.transforms.spinquant.serialize import (
            _get_hidden_size,
        )

        class _Config:
            hidden_size = 128

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = _Config()

        assert _get_hidden_size(_Model()) == 128

    def test_hidden_size_missing_config(self):
        from auto_round.algorithms.transforms.spinquant.serialize import (
            _get_hidden_size,
        )

        assert _get_hidden_size(nn.Linear(4, 4)) == 0

    def test_head_dim_explicit(self):
        from auto_round.algorithms.transforms.spinquant.serialize import (
            _get_head_dim,
        )

        class _Config:
            head_dim = 32

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = _Config()

        assert _get_head_dim(_Model()) == 32

    def test_head_dim_computed(self):
        from auto_round.algorithms.transforms.spinquant.serialize import (
            _get_head_dim,
        )

        class _Config:
            hidden_size = 128
            num_attention_heads = 4

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = _Config()

        assert _get_head_dim(_Model()) == 32  # 128 / 4

    def test_head_dim_no_config(self):
        from auto_round.algorithms.transforms.spinquant.serialize import (
            _get_head_dim,
        )

        assert _get_head_dim(nn.Linear(4, 4)) == 0

    def test_intermediate_size(self):
        from auto_round.algorithms.transforms.spinquant.serialize import (
            _get_intermediate_size,
        )

        class _Config:
            intermediate_size = 256

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = _Config()

        assert _get_intermediate_size(_Model()) == 256

    def test_intermediate_size_missing(self):
        from auto_round.algorithms.transforms.spinquant.serialize import (
            _get_intermediate_size,
        )

        assert _get_intermediate_size(nn.Linear(4, 4)) == 0


# ---------------------------------------------------------------------------
# Training helpers: move_batch_to_device
# ---------------------------------------------------------------------------
class TestMoveBatchToDevice:
    def test_moves_tensor_to_cpu(self):
        from auto_round.algorithms.transforms.spinquant.training import (
            move_batch_to_device,
        )

        t = torch.zeros(2, 2)
        moved = move_batch_to_device(t, torch.device("cpu"))
        assert moved.device.type == "cpu"

    def test_dict_of_tensors(self):
        from auto_round.algorithms.transforms.spinquant.training import (
            move_batch_to_device,
        )

        batch = {"input_ids": torch.zeros(2, 2), "labels": torch.ones(2)}
        moved = move_batch_to_device(batch, torch.device("cpu"))
        assert isinstance(moved, dict)
        assert moved["input_ids"].device.type == "cpu"
        assert moved["labels"].device.type == "cpu"


# ---------------------------------------------------------------------------
# check_orthogonality
# ---------------------------------------------------------------------------
class TestCheckOrthogonality:
    def test_identity_is_orthogonal(self):
        from auto_round.algorithms.transforms.spinquant.training import (
            check_orthogonality,
        )

        class _Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.eye(4))

        layer = _Layer()
        # Identity matrix is exactly orthogonal -> deviation 0
        err = check_orthogonality(layer, threshold=1e-3)
        assert err == pytest.approx(0.0, abs=1e-5)

    def test_random_matrix_not_orthogonal(self):
        from auto_round.algorithms.transforms.spinquant.training import (
            check_orthogonality,
        )

        class _Layer(nn.Module):
            def __init__(self):
                super().__init__()
                # Parameter must be named spinquant_R* and require grad
                self.spinquant_R1 = nn.Parameter(torch.randn(4, 4))

        layer = _Layer()
        err = check_orthogonality(layer, threshold=1e-3)
        # A random matrix will have non-zero orthogonality error
        assert err > 1e-3

    def test_skips_non_spinquant_params(self):
        from auto_round.algorithms.transforms.spinquant.training import (
            check_orthogonality,
        )

        class _Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 4))

        layer = _Layer()
        err = check_orthogonality(layer)
        # No spinquant_R* parameters -> max_dev stays 0
        assert err == 0.0
