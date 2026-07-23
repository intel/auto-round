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
"""Tests for the small pure helpers in ``auto_round/inference/backend.py``."""

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
class TestBackendConstants:
    def test_backend_act_attrs_contains_act_bits(self):
        from auto_round.inference.backend import BACKEND_ACT_ATTRS

        assert "act_bits" in BACKEND_ACT_ATTRS

    def test_backend_act_attrs_contains_act_dynamic(self):
        from auto_round.inference.backend import BACKEND_ACT_ATTRS

        assert "act_dynamic" in BACKEND_ACT_ATTRS

    def test_mx_tensor_data_types(self):
        from auto_round.inference.backend import MX_TENSOR_DATA_TYPES

        assert "mx_fp" in MX_TENSOR_DATA_TYPES
        assert "mx_fp_rceil" in MX_TENSOR_DATA_TYPES
        assert "mx_int" in MX_TENSOR_DATA_TYPES


# ---------------------------------------------------------------------------
# BackendInfo dataclass
# ---------------------------------------------------------------------------
class TestBackendInfo:
    def test_minimal_construction(self):
        from auto_round.inference.backend import BackendInfo

        info = BackendInfo(
            device=["cpu"],
            sym=[True],
            packing_format=[""],
            bits=[4],
        )
        assert info.device == ["cpu"]
        assert info.sym == [True]
        assert info.bits == [4]
        assert info.priority == 0  # default
        assert info.checkers == []  # default

    def test_all_fields(self):
        from auto_round.inference.backend import BackendInfo

        info = BackendInfo(
            device=["cpu", "xpu"],
            sym=[True, False],
            packing_format=["ark", "triton"],
            bits=[2, 4, 8],
            compute_dtype=["bfloat16"],
            data_type=["int"],
            group_size=[32, 64, 128],
            act_bits=[8, 16],
            act_group_size=[32, 64],
            act_sym=[True, False],
            act_data_type=["mx_fp_rceil"],
            act_dynamic=[True],
            priority=10,
            checkers=["checker1"],
            alias=["cpu_xt"],
            requirements=["triton>=2.0"],
            systems=["linux"],
        )
        assert info.priority == 10
        assert info.alias == ["cpu_xt"]
        assert info.systems == ["linux"]
        assert info.requirements == ["triton>=2.0"]


# ---------------------------------------------------------------------------
# feature_multiply_checker
# ---------------------------------------------------------------------------
class TestFeatureMultiplyChecker:
    def test_both_divisible(self):
        from auto_round.inference.backend import feature_multiply_checker

        assert feature_multiply_checker(64, 64, {}, 32) is True

    def test_in_not_divisible(self):
        from auto_round.inference.backend import feature_multiply_checker

        assert feature_multiply_checker(33, 64, {}, 32) is False

    def test_out_not_divisible(self):
        from auto_round.inference.backend import feature_multiply_checker

        assert feature_multiply_checker(64, 33, {}, 32) is False

    def test_distinct_in_out_multipliers(self):
        from auto_round.inference.backend import feature_multiply_checker

        assert feature_multiply_checker(8, 16, {}, 8, 16) is True
        assert feature_multiply_checker(8, 17, {}, 8, 16) is False


# ---------------------------------------------------------------------------
# feature_multiply_checker_group_size
# ---------------------------------------------------------------------------
class TestFeatureMultiplyCheckerGroupSize:
    def test_all_divisible(self):
        from auto_round.inference.backend import feature_multiply_checker_group_size

        assert feature_multiply_checker_group_size(64, 64, {"group_size": 32}, 32) is True

    def test_group_size_fails(self):
        from auto_round.inference.backend import feature_multiply_checker_group_size

        assert feature_multiply_checker_group_size(64, 64, {"group_size": 7}, 32) is False

    def test_in_multiplier_fails(self):
        from auto_round.inference.backend import feature_multiply_checker_group_size

        assert feature_multiply_checker_group_size(33, 64, {"group_size": 32}, 32) is False

    def test_out_multiplier_fails(self):
        from auto_round.inference.backend import feature_multiply_checker_group_size

        assert feature_multiply_checker_group_size(64, 33, {"group_size": 32}, 32) is False

    def test_distinct_out_multiplier(self):
        from auto_round.inference.backend import feature_multiply_checker_group_size

        # Pass explicit out_feature_multiplier
        assert feature_multiply_checker_group_size(8, 16, {"group_size": 8}, 8, 16) is True


# ---------------------------------------------------------------------------
# feature_compatible_multiply_checker
# ---------------------------------------------------------------------------
class TestFeatureCompatibleMultiplyChecker:
    def test_in_div_by_group_size(self):
        from auto_round.inference.backend import feature_compatible_multiply_checker

        # in_feature=64 divisible by group_size=32 -> ok
        assert feature_compatible_multiply_checker(64, 64, {"group_size": 32}, 32) is True

    def test_in_less_than_group_size_with_compatible(self):
        """When in_feature < group_size but in*out is divisible, the check passes."""
        from auto_round.inference.backend import feature_compatible_multiply_checker

        # Need: in%32 == 0 AND out%32 == 0 AND (in%64==0 OR (in<64 AND in*out%64==0))
        # in=32, out=32, group=64: 32%32=0, 32%32=0, 32<64 AND 32*32%64==0 -> ok
        assert feature_compatible_multiply_checker(32, 32, {"group_size": 64}, 32) is True

    def test_in_less_than_group_size_incompatible(self):
        from auto_round.inference.backend import feature_compatible_multiply_checker

        # in=8, out=15, group=32: 8 < 32 and 8*15 = 120 not div by 32 -> fail
        assert feature_compatible_multiply_checker(8, 15, {"group_size": 32}, 32) is False

    def test_in_divisible_by_group_size(self):
        from auto_round.inference.backend import feature_compatible_multiply_checker

        # 64 divisible by 32 -> ok (first branch)
        assert feature_compatible_multiply_checker(64, 64, {"group_size": 32}, 32) is True

    def test_in_multiplier_fails(self):
        from auto_round.inference.backend import feature_compatible_multiply_checker

        assert feature_compatible_multiply_checker(33, 64, {"group_size": 32}, 32) is False


# ---------------------------------------------------------------------------
# get_cpu_manufacturer
# ---------------------------------------------------------------------------
class TestGetCpuManufacturer:
    def test_intel_cpu_returns_intel(self):
        from auto_round.inference.backend import get_cpu_manufacturer

        with patch(
            "auto_round.inference.backend.cpuinfo.get_cpu_info",
            return_value={"brand_raw": "Intel(R) Core(TM) i7-12700K"},
        ):
            assert get_cpu_manufacturer() == "intel"

    def test_amd_cpu_returns_others(self):
        from auto_round.inference.backend import get_cpu_manufacturer

        with patch(
            "auto_round.inference.backend.cpuinfo.get_cpu_info",
            return_value={"brand_raw": "AMD Ryzen 9 7950X"},
        ):
            assert get_cpu_manufacturer() == "others"

    def test_missing_brand_raw_returns_others(self):
        from auto_round.inference.backend import get_cpu_manufacturer

        with patch(
            "auto_round.inference.backend.cpuinfo.get_cpu_info",
            return_value={},
        ):
            assert get_cpu_manufacturer() == "others"

    def test_intel_in_middle_of_brand_returns_intel(self):
        from auto_round.inference.backend import get_cpu_manufacturer

        with patch(
            "auto_round.inference.backend.cpuinfo.get_cpu_info",
            return_value={"brand_raw": "GenuineIntel(R) CPU @ 2.40GHz"},
        ):
            # "intel" is in the brand_raw, so should return intel
            assert get_cpu_manufacturer() == "intel"
