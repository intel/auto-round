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
"""Tests for the ``AR_NVFP_GS_ALPHA`` amax multiplier used when computing NVFP global scales."""

import pytest
import torch

from auto_round import envs
from auto_round.data_type.nvfp import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    FLOAT8_UE5M3_MAX,
    calculate_gparam,
    calculate_gparam_by_dtype,
    calculate_gparam_e5m3,
    get_nvfp_gs_alpha,
    nv_fp4,
    nv_fp4_rtn,
    nv_fp4_with_static_gs,
    nvfp4_v2_with_global_scale,
    nvfp4_v2_with_static_gs,
    opt_rtn_fast_nvfp4,
    rtn_nv_fp4_with_static_gs,
)

ENV_NAME = "AR_NVFP_GS_ALPHA"


def _act_gs_nvfp4(amax):
    """Activation-side NVFP4 global scale (alpha = AR_NVFP_GS_ALPHA or 4)."""
    return calculate_gparam_by_dtype(amax, "nv_fp4", activation=True)


def _act_gs_e5m3(amax):
    """Activation-side NVFP4_E5M3_GS global scale (alpha = AR_NVFP_GS_ALPHA or 8)."""
    return calculate_gparam_by_dtype(amax, "nvfp4_v2_gs", activation=True)


@pytest.fixture(autouse=True)
def _clear_alpha_env(monkeypatch):
    monkeypatch.delenv(ENV_NAME, raising=False)


class TestDefaultAlpha:
    def test_nvfp4_default_alpha_is_4(self):
        assert get_nvfp_gs_alpha(e5m3=False) == 4.0

    def test_e5m3_default_alpha_is_8(self):
        assert get_nvfp_gs_alpha(e5m3=True) == 8.0

    def test_weight_calculate_gparam_is_alpha_free(self):
        t = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        expected = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / 4.0
        assert calculate_gparam(t).item() == pytest.approx(expected, rel=1e-6)

    def test_weight_calculate_gparam_e5m3_is_alpha_free(self):
        t = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        expected = FLOAT8_UE5M3_MAX * FLOAT4_E2M1_MAX / 4.0
        assert calculate_gparam_e5m3(t).item() == pytest.approx(expected, rel=1e-6)
        assert calculate_gparam_by_dtype(t, "opt_rtn_nvfp4_v2_gs").item() == pytest.approx(expected, rel=1e-6)

    def test_weight_by_dtype_default_is_alpha_free(self):
        t = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        assert calculate_gparam_by_dtype(t, "nv_fp4").item() == pytest.approx(
            FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / 4.0, rel=1e-6
        )

    def test_activation_by_dtype_applies_alpha(self):
        t = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        assert calculate_gparam_by_dtype(t, "nv_fp4", activation=True).item() == pytest.approx(
            FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / (4.0 * 4.0), rel=1e-6
        )
        assert calculate_gparam_by_dtype(t, "nvfp4_v2_gs", activation=True).item() == pytest.approx(
            FLOAT8_UE5M3_MAX * FLOAT4_E2M1_MAX / (8.0 * 4.0), rel=1e-6
        )

    def test_zero_amax_gives_zero_global_scale(self):
        assert calculate_gparam(torch.zeros(4, 16)).item() == 0.0
        assert calculate_gparam_e5m3(torch.zeros(4, 16)).item() == 0.0


class TestEnvOverride:
    @pytest.mark.parametrize("raw", ["2.5", "1", "0.5"])
    def test_override_applies_to_activation_only(self, monkeypatch, raw):
        monkeypatch.setenv(ENV_NAME, raw)
        alpha = float(raw)
        assert envs.AR_NVFP_GS_ALPHA == alpha
        assert get_nvfp_gs_alpha(e5m3=False) == alpha
        assert get_nvfp_gs_alpha(e5m3=True) == alpha

        t = torch.tensor([2.0, -8.0])
        # Weight-side global scales ignore the override.
        assert calculate_gparam(t).item() == pytest.approx(FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / 8.0, rel=1e-6)
        assert calculate_gparam_e5m3(t).item() == pytest.approx(FLOAT8_UE5M3_MAX * FLOAT4_E2M1_MAX / 8.0, rel=1e-6)
        # Activation-side global scales honor the override.
        assert calculate_gparam_by_dtype(t, "nv_fp4", activation=True).item() == pytest.approx(
            FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / (alpha * 8.0), rel=1e-6
        )
        assert calculate_gparam_by_dtype(t, "nvfp4_v2_gs", activation=True).item() == pytest.approx(
            FLOAT8_UE5M3_MAX * FLOAT4_E2M1_MAX / (alpha * 8.0), rel=1e-6
        )

    @pytest.mark.parametrize("raw", ["auto", "AUTO", " auto ", ""])
    def test_auto_or_empty_uses_family_default(self, monkeypatch, raw):
        monkeypatch.setenv(ENV_NAME, raw)
        assert envs.AR_NVFP_GS_ALPHA is None
        assert get_nvfp_gs_alpha(e5m3=False) == 4.0
        assert get_nvfp_gs_alpha(e5m3=True) == 8.0

    def test_unset_uses_family_default(self):
        assert envs.AR_NVFP_GS_ALPHA is None
        assert get_nvfp_gs_alpha(e5m3=False) == 4.0
        assert get_nvfp_gs_alpha(e5m3=True) == 8.0

    @pytest.mark.parametrize("raw", ["0", "-1", "-0.5", "nan", "inf", "abc"])
    def test_invalid_value_raises(self, monkeypatch, raw):
        monkeypatch.setenv(ENV_NAME, raw)
        with pytest.raises(ValueError, match=ENV_NAME):
            get_nvfp_gs_alpha(e5m3=False)


class TestInlineSitesFollowAlpha:
    """Inline activation global-scale formulas must match the activation-side helpers (same alpha)."""

    @pytest.mark.parametrize("quant_func", [nv_fp4, nv_fp4_rtn, opt_rtn_fast_nvfp4])
    def test_weight_dtype_fallback_is_alpha_free(self, monkeypatch, quant_func):
        # RTN/AWQ/TEQ call weight dtypes without global_scale; QDQ must match the exported weight global scale.
        monkeypatch.setenv(ENV_NAME, "3")
        torch.manual_seed(0)
        t = torch.randn(4, 32, dtype=torch.float32)
        q_dyn, s_dyn, _ = quant_func(t)
        q_ref, s_ref, _ = quant_func(t, global_scale=calculate_gparam(t))
        torch.testing.assert_close(q_dyn, q_ref)
        torch.testing.assert_close(s_dyn, s_ref)

    def test_nv_fp4_with_static_gs_matches_explicit_global_scale(self):
        torch.manual_seed(0)
        t = torch.randn(4, 32, dtype=torch.float32)
        tensor_max = t.abs().max()
        q_static, s_static, _ = nv_fp4_with_static_gs(t, tensor_max=tensor_max)
        q_ref, s_ref, _ = nv_fp4_with_static_gs(t, global_scale=_act_gs_nvfp4(tensor_max))
        torch.testing.assert_close(q_static, q_ref)
        torch.testing.assert_close(s_static, s_ref)

    def test_rtn_nv_fp4_with_static_gs_uses_activation_alpha(self):
        torch.manual_seed(0)
        t = torch.randn(4, 32, dtype=torch.float32)
        tensor_max = t.abs().max()
        q_rtn, s_rtn, _ = rtn_nv_fp4_with_static_gs(t, tensor_max=tensor_max)
        q_ref, s_ref, _ = nv_fp4_with_static_gs(t, global_scale=_act_gs_nvfp4(tensor_max))
        torch.testing.assert_close(q_rtn, q_ref)
        torch.testing.assert_close(s_rtn, s_ref)

    def test_nvfp4_v2_e5m3_matches_explicit_global_scale(self):
        torch.manual_seed(0)
        t = torch.randn(4, 32, dtype=torch.float32)
        tensor_max = t.abs().max()
        q_dyn, _, _ = nvfp4_v2_with_global_scale(t, group_size=16, tensor_max=tensor_max)
        q_ref, _, _ = nvfp4_v2_with_static_gs(t, group_size=16, global_scale=_act_gs_e5m3(tensor_max))
        torch.testing.assert_close(q_dyn, q_ref)

    def test_override_changes_inline_global_scale(self, monkeypatch):
        torch.manual_seed(0)
        t = torch.randn(4, 32, dtype=torch.float32)
        tensor_max = t.abs().max()
        default_gs = _act_gs_nvfp4(tensor_max).item()
        weight_gs = calculate_gparam(tensor_max).item()
        monkeypatch.setenv(ENV_NAME, "1")
        q_alpha1, _, _ = nv_fp4_with_static_gs(t, tensor_max=tensor_max)
        # Activation global scale is inversely proportional to alpha: 4 -> 1 multiplies it by 4.
        assert _act_gs_nvfp4(tensor_max).item() == pytest.approx(default_gs * 4.0, rel=1e-6)
        # Weight global scale is untouched by the override.
        assert calculate_gparam(tensor_max).item() == pytest.approx(weight_gs, rel=1e-6)
        q_ref, _, _ = nv_fp4_with_static_gs(t, global_scale=_act_gs_nvfp4(tensor_max))
        torch.testing.assert_close(q_alpha1, q_ref)
