# Copyright (c) 2025 Intel Corporation
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

import inspect

import pytest
import torch

from auto_round import envs
from auto_round.compressors.model_free import _configure_model_free_int_search_ratio
from auto_round.data_type.int import _symmetric_search_offsets, search_scales
from auto_round.utils.model_free_utils import quantize_weight_rtn


class TestSymmetricSearchOffsets:
    def test_offsets_expand_symmetrically_from_center(self):
        assert _symmetric_search_offsets(0) == []
        assert _symmetric_search_offsets(1) == [-1, 1]
        assert _symmetric_search_offsets(3) == [-1, 1, -2, 2, -3, 3]


class TestSearchScales:
    def test_model_free_int_respects_explicit_disable_opt_rtn(self, monkeypatch):
        monkeypatch.delenv("AR_SEARCH_SCALE_RATIO", raising=False)
        messages = []

        def capture_info(message, *args):
            messages.append(message % args if args else message)

        monkeypatch.setattr("auto_round.compressors.model_free.logger.info", capture_info)
        _configure_model_free_int_search_ratio({"data_type": "int", "bits": 4}, {}, disable_opt_rtn=True)

        assert envs.AR_SEARCH_SCALE_RATIO is None
        assert not any("default empirical search ratio" in msg for msg in messages)

    def test_model_free_int_defaults_enable_opt_rtn(self):
        assert inspect.signature(quantize_weight_rtn).parameters["disable_opt_rtn"].default is False

    def test_model_free_int_enables_empirical_search_ratio_default(self, monkeypatch):
        monkeypatch.delenv("AR_SEARCH_SCALE_RATIO", raising=False)
        messages = []

        def capture_info(message, *args):
            messages.append(message % args if args else message)

        monkeypatch.setattr("auto_round.compressors.model_free.logger.info", capture_info)
        _configure_model_free_int_search_ratio({"data_type": "int", "bits": 4}, {}, disable_opt_rtn=False)

        assert envs.AR_SEARCH_SCALE_RATIO == pytest.approx(0.05)
        assert any("default empirical search ratio 0.050" in msg for msg in messages)

    def test_returns_finite_scales(self):
        data = torch.randn(8, 16, dtype=torch.float32)
        scales = search_scales(data, bits=4, qw=torch.ones_like(data))

        assert scales.shape == (8, 1)
        assert torch.isfinite(scales).all()

    def test_weighted_loss_does_not_increase(self):
        torch.manual_seed(42)
        data = torch.randn(8, 16, dtype=torch.float32)
        qw = torch.exp(torch.randn_like(data))
        bits = 4
        nmax = int(2.0 ** (bits - 1))
        group_max = torch.take_along_dim(data, torch.abs(data).argmax(dim=-1, keepdim=True), dim=-1)
        baseline_scale = group_max / -nmax
        baseline_q = torch.round(data / baseline_scale).clamp(-nmax, nmax - 1)
        baseline_loss = (((baseline_q * baseline_scale) - data).float().square() * qw).sum()

        scales = search_scales(data, bits=bits, qw=qw)
        searched_q = torch.round(data / scales).clamp(-nmax, nmax - 1)
        searched_loss = (((searched_q * scales) - data).float().square() * qw).sum()

        assert searched_loss <= baseline_loss

    def test_prefers_centered_order_over_monotonic_bias(self, monkeypatch):
        data = torch.tensor([[1.0, -1.0, 0.49, -0.49]], dtype=torch.float32)
        bits = 4
        calls = []

        original_round = torch.round

        def tracking_round(input_tensor, *args, **kwargs):
            calls.append(input_tensor.detach().clone())
            return original_round(input_tensor, *args, **kwargs)

        monkeypatch.setattr(torch, "round", tracking_round)
        search_scales(data, bits=bits, qw=torch.ones_like(data))

        assert len(calls) > 2
        # After the baseline call, the first two search probes should be the
        # nearest smaller and larger candidates, rather than sweeping one side.
        first_probe = calls[1]
        second_probe = calls[2]
        assert not torch.allclose(first_probe, second_probe)
        assert torch.sign(first_probe.abs().mean() - calls[0].abs().mean()) != torch.sign(
            second_probe.abs().mean() - calls[0].abs().mean()
        )

    @pytest.mark.parametrize("bits", [2, 4])
    def test_search_offsets_are_used_for_multiple_bitwidths(self, bits):
        data = torch.randn(4, 16, dtype=torch.float32)
        scales = search_scales(data, bits=bits, qw=torch.ones_like(data))
        assert torch.isfinite(scales).all()

    def test_scale_search_uses_uniform_scale_ratio_steps(self, monkeypatch):
        monkeypatch.setattr("auto_round.data_type.int.envs.AR_SEARCH_SCALE_RATIO", 0.9)
        data = torch.tensor([[1.0, -1.0, 0.49, -0.49]], dtype=torch.float32)
        probes = []
        original_round = torch.round

        def tracking_round(input_tensor, *args, **kwargs):
            probes.append(input_tensor.detach().abs().amax().item())
            return original_round(input_tensor, *args, **kwargs)

        monkeypatch.setattr(torch, "round", tracking_round)
        search_scales(data, bits=4, qw=torch.ones_like(data))

        # With a 0.005 step and a 0.9 global search span, the first two
        # symmetric probes land at 1±0.005 around the baseline.
        assert probes[:3] == pytest.approx([8.0, 8.0 / 0.995, 8.0 / 1.005])
