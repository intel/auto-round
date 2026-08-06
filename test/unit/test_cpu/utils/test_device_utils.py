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

"""Tests for auto_round/utils/device.py"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest


class TestIsPackageAvailable:
    """Test is_package_available function."""

    def test_torch_available(self):
        from auto_round.utils.device import is_package_available

        assert is_package_available("torch") is True

    def test_nonexistent_package(self):
        from auto_round.utils.device import is_package_available

        assert is_package_available("nonexistent_package_xyz123") is False

    def test_numpy_available(self):
        from auto_round.utils.device import is_package_available

        assert is_package_available("numpy") is True


class TestIsHpuLazyMode:
    """Test is_hpu_lazy_mode function."""

    def test_lazy_mode_enabled(self):
        """Test PT_HPU_LAZY_MODE=1 returns True."""
        from auto_round.utils.device import is_hpu_lazy_mode

        with patch.dict(os.environ, {"PT_HPU_LAZY_MODE": "1"}, clear=False):
            assert is_hpu_lazy_mode() is True

    def test_lazy_mode_disabled(self):
        """Test PT_HPU_LAZY_MODE=0 returns False."""
        from auto_round.utils.device import is_hpu_lazy_mode

        with patch.dict(os.environ, {"PT_HPU_LAZY_MODE": "0"}, clear=False):
            assert is_hpu_lazy_mode() is False

    def test_lazy_mode_unset(self):
        """Test unset PT_HPU_LAZY_MODE returns True (default)."""
        from auto_round.utils.device import is_hpu_lazy_mode

        # Save original value if exists
        old_val = os.environ.pop("PT_HPU_LAZY_MODE", None)
        try:
            result = is_hpu_lazy_mode()
            assert result is True
        finally:
            if old_val is not None:
                os.environ["PT_HPU_LAZY_MODE"] = old_val


class TestUseHpuCompileMode:
    """Test _use_hpu_compile_mode function."""

    def test_compile_mode_true(self):
        """Test compile mode when torch >= 2.4 and lazy mode disabled."""
        from auto_round.utils.device import _use_hpu_compile_mode

        # Mock both is_hpu_lazy_mode and TORCH_VERSION_AT_LEAST_2_4 (imported inside function)
        with patch("auto_round.utils.device.is_hpu_lazy_mode", return_value=False), patch.dict(
            "sys.modules", {"auto_round.utils.common": MagicMock(TORCH_VERSION_AT_LEAST_2_4=True)}
        ):
            result = _use_hpu_compile_mode()
            assert result is True

    def test_compile_mode_false_lazy_on(self):
        """Test compile mode False when lazy mode is on."""
        from auto_round.utils.device import _use_hpu_compile_mode

        with patch("auto_round.utils.device.is_hpu_lazy_mode", return_value=True):
            assert _use_hpu_compile_mode() is False

    def test_compile_mode_false_torch_old(self):
        """Test compile mode False when torch < 2.4."""
        from auto_round.utils.device import _use_hpu_compile_mode

        with patch("auto_round.utils.device.is_hpu_lazy_mode", return_value=False), patch.dict(
            "sys.modules", {"auto_round.utils.common": MagicMock(TORCH_VERSION_AT_LEAST_2_4=False)}
        ):
            result = _use_hpu_compile_mode()
            assert result is False


class TestBumpDynamoCacheLimit:
    """Test _bump_dynamo_cache_limit function."""

    def test_bump_with_explicit_min_size(self):
        """Test _bump_dynamo_cache_limit with explicit min_size."""
        from auto_round.utils.device import _bump_dynamo_cache_limit

        # Mock torch._dynamo.config to verify it sets values
        mock_config = MagicMock()
        mock_config.cache_size_limit = 8
        mock_config.accumulated_cache_size_limit = 8
        mock_config.recompile_limit = 8

        with patch.dict("sys.modules", {"torch._dynamo.config": mock_config}):
            with patch("torch._dynamo.config", mock_config):
                _bump_dynamo_cache_limit(min_size=32)
                # Function should attempt to set values >= 32
                # Best effort - it may or may not raise depending on imports

    def test_bump_without_value_uses_default(self):
        """Test _bump_dynamo_cache_limit without min_size uses env default."""
        from auto_round.utils.device import _bump_dynamo_cache_limit

        # Should not raise even if torch._dynamo is not available
        # The function catches all exceptions (best effort)
        _bump_dynamo_cache_limit()

    def test_bump_handles_missing_dynamo(self):
        """Test _bump_dynamo_cache_limit handles missing torch._dynamo gracefully."""
        from auto_round.utils.device import _bump_dynamo_cache_limit

        # Remove torch._dynamo from sys.modules temporarily
        original_dynamo = sys.modules.pop("torch._dynamo", None)
        try:
            _bump_dynamo_cache_limit(min_size=16)
        finally:
            if original_dynamo is not None:
                sys.modules["torch._dynamo"] = original_dynamo


class TestNumbaAndTbb:
    """Test Numba and TBB availability functions."""

    def test_is_numba_available_returns_bool(self):
        """Test is_numba_available returns True or False without raising."""
        from auto_round.utils.device import is_numba_available

        result = is_numba_available()
        assert isinstance(result, bool)

    def test_can_pack_with_numba_returns_bool(self):
        """Test can_pack_with_numba returns True or False without raising."""
        from auto_round.utils.device import can_pack_with_numba

        result = can_pack_with_numba()
        assert isinstance(result, bool)


class TestIsTbbAvailable:
    """Test is_tbb_available function."""

    def test_is_tbb_available_returns_bool(self):
        """Test is_tbb_available returns True or False without raising."""
        from auto_round.utils.device import is_tbb_available

        result = is_tbb_available()
        assert isinstance(result, bool)


class TestOverrideCudaDeviceCapability:
    """Test override_cuda_device_capability context manager."""

    def test_enter_exit(self):
        """Test entering and exiting the context manager."""
        import torch

        from auto_round.utils.device import override_cuda_device_capability

        ctx = override_cuda_device_capability(9, 0)
        ctx.__enter__()
        ctx.__exit__(None, None, None)

    def test_overrides_capability(self):
        """Test it actually overrides CUDA device capability."""
        import torch

        from auto_round.utils.device import override_cuda_device_capability

        # Only test if CUDA is available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Save original
        original_capability = torch.cuda.get_device_capability

        with override_cuda_device_capability(100, 1):
            cap = torch.cuda.get_device_capability()
            assert cap == (100, 1)

        # Verify original is restored
        assert torch.cuda.get_device_capability == original_capability

    def test_with_decorator(self):
        """Test using as a decorator."""
        import torch

        from auto_round.utils.device import override_cuda_device_capability

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        @override_cuda_device_capability(8, 6)
        def check_cap():
            return torch.cuda.get_device_capability()

        cap = check_cap()
        assert cap == (8, 6)


class TestFakeCudaForHpu:
    """Test fake_cuda_for_hpu context manager."""

    def test_enter_exit(self):
        """Test entering and exiting the context manager."""
        from auto_round.utils.device import fake_cuda_for_hpu

        ctx = fake_cuda_for_hpu()
        ctx.__enter__()
        ctx.__exit__(None, None, None)

    def test_context_manager_usage(self):
        """Test using as a context manager."""
        from auto_round.utils.device import fake_cuda_for_hpu

        with fake_cuda_for_hpu():
            pass  # Should not raise


class TestFakeTritonForHpu:
    """Test fake_triton_for_hpu context manager."""

    def test_enter_exit(self):
        """Test entering and exiting the context manager."""
        from auto_round.utils.device import fake_triton_for_hpu

        ctx = fake_triton_for_hpu()
        ctx.__enter__()
        ctx.__exit__(None, None, None)

    def test_context_manager_usage(self):
        """Test using as a context manager."""
        from auto_round.utils.device import fake_triton_for_hpu

        with fake_triton_for_hpu():
            pass  # Should not raise


class TestCpuInfo:
    """Test CpuInfo class."""

    def test_creation(self):
        """Test CpuInfo can be created."""
        from auto_round.utils.device import CpuInfo

        info = CpuInfo()
        assert hasattr(info, "_bf16")
        assert hasattr(info, "bf16")

    def test_bf16_property(self):
        """Test bf16 property returns boolean."""
        from auto_round.utils.device import CpuInfo

        info = CpuInfo()
        assert isinstance(info.bf16, bool)

    def test_multiple_instances(self):
        """Test creating multiple CpuInfo instances."""
        from auto_round.utils.device import CpuInfo

        info1 = CpuInfo()
        info2 = CpuInfo()
        # Both should have bf16 property
        assert hasattr(info1, "bf16")
        assert hasattr(info2, "bf16")


class TestBytesToGigabytes:
    """Test bytes_to_gigabytes function."""

    def test_one_gigabyte(self):
        """Test conversion of 1 GB."""
        from auto_round.utils.device import bytes_to_gigabytes

        result = bytes_to_gigabytes(1024 * 1024 * 1024)
        assert result == pytest.approx(1.0, rel=0.01)

    def test_zero(self):
        """Test conversion of zero bytes."""
        from auto_round.utils.device import bytes_to_gigabytes

        assert bytes_to_gigabytes(0) == 0

    def test_multiple_gigabytes(self):
        """Test conversion of multiple GBs."""
        from auto_round.utils.device import bytes_to_gigabytes

        assert bytes_to_gigabytes(8 * 1024 * 1024 * 1024) == pytest.approx(8.0, rel=0.01)

    def test_partial_gigabyte(self):
        """Test conversion of partial GB."""
        from auto_round.utils.device import bytes_to_gigabytes

        # 512 MB = 0.5 GB
        assert bytes_to_gigabytes(512 * 1024 * 1024) == pytest.approx(0.5, rel=0.01)


class TestMemoryTrimming:
    """Test memory trimming functions."""

    def test_force_trim_malloc(self):
        """Test _force_trim_malloc doesn't raise."""
        from auto_round.utils.device import _force_trim_malloc

        _force_trim_malloc()  # Best effort - may not do anything

    def test_force_trim_malloc_disabled(self):
        """Test _force_trim_malloc with disabled env var."""
        from auto_round.utils.device import _force_trim_malloc

        with patch.dict(os.environ, {"AR_ENABLE_MALLOC_TRIM": "0"}, clear=False):
            _force_trim_malloc()  # Should return early

    def test_maybe_trim_malloc(self):
        """Test _maybe_trim_malloc doesn't raise."""
        from auto_round.utils.device import _maybe_trim_malloc

        _maybe_trim_malloc()  # Best effort - may not do anything

    def test_maybe_trim_malloc_disabled(self):
        """Test _maybe_trim_malloc with disabled env var."""
        from auto_round.utils.device import _maybe_trim_malloc

        with patch.dict(os.environ, {"AR_ENABLE_MALLOC_TRIM": "0"}, clear=False):
            _maybe_trim_malloc()  # Should return early

    def test_maybe_trim_malloc_custom_every(self):
        """Test _maybe_trim_malloc with custom AR_MALLOC_TRIM_EVERY."""
        from auto_round.utils.device import _maybe_trim_malloc

        with patch.dict(os.environ, {"AR_MALLOC_TRIM_EVERY": "1"}, clear=False):
            _maybe_trim_malloc()  # Should call libc.malloc_trim


class TestDeviceEnvironVariableMapping:
    """Test DEVICE_ENVIRON_VARIABLE_MAPPING constant."""

    def test_cuda_mapping_exists(self):
        """Test CUDA environment variable mapping exists."""
        from auto_round.utils.device import DEVICE_ENVIRON_VARIABLE_MAPPING

        assert "cuda" in DEVICE_ENVIRON_VARIABLE_MAPPING
        assert DEVICE_ENVIRON_VARIABLE_MAPPING["cuda"] == "CUDA_VISIBLE_DEVICES"

    def test_xpu_mapping_exists(self):
        """Test XPU environment variable mapping exists."""
        from auto_round.utils.device import DEVICE_ENVIRON_VARIABLE_MAPPING

        assert "xpu" in DEVICE_ENVIRON_VARIABLE_MAPPING
        assert DEVICE_ENVIRON_VARIABLE_MAPPING["xpu"] == "ZE_AFFINITY_MASK"

    def test_hpu_mapping_exists(self):
        """Test HPU environment variable mapping exists."""
        from auto_round.utils.device import DEVICE_ENVIRON_VARIABLE_MAPPING

        assert "hpu" in DEVICE_ENVIRON_VARIABLE_MAPPING
        assert DEVICE_ENVIRON_VARIABLE_MAPPING["hpu"] == "HABANA_VISIBLE_MODULES"

    def test_mappings_are_strings(self):
        """Test all mappings are string values."""
        from auto_round.utils.device import DEVICE_ENVIRON_VARIABLE_MAPPING

        for key, value in DEVICE_ENVIRON_VARIABLE_MAPPING.items():
            assert isinstance(key, str)
            assert isinstance(value, str)


class TestSetCudaVisibleDevices:
    """Test set_cuda_visible_devices function."""

    def test_single_device_cuda_string(self):
        """Test setting single device with 'cuda' string."""
        from auto_round.utils.device import set_cuda_visible_devices

        original = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            set_cuda_visible_devices("cuda")
            # Should set to "0" or similar
        finally:
            if original is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def test_single_device_index(self):
        """Test setting single device with numeric index."""
        from auto_round.utils.device import set_cuda_visible_devices

        original = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            set_cuda_visible_devices("0")
        finally:
            if original is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def test_multiple_devices(self):
        """Test setting multiple devices."""
        from auto_round.utils.device import set_cuda_visible_devices

        original = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            # Save original CUDA_VISIBLE_DEVICES
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
            set_cuda_visible_devices("0,2")
            # Should pick indices 0 and 2 = "0,2"
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0,2"
        finally:
            if original is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def test_auto_device(self):
        """Test 'auto' device does nothing."""
        from auto_round.utils.device import set_cuda_visible_devices

        original = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            if original is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)

            set_cuda_visible_devices("auto")
            # Should not modify the environment
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == original
        finally:
            if original is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def test_invalid_device_index_raises(self):
        """Test invalid device index raises ValueError."""
        from auto_round.utils.device import set_cuda_visible_devices

        original = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            # Set CUDA_VISIBLE_DEVICES with only 2 devices
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
            # Try to access device index 5 (out of range)
            with pytest.raises(ValueError):
                set_cuda_visible_devices("5")
        finally:
            if original is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def test_without_existing_cuda_visible_devices(self):
        """Test setting devices without pre-existing CUDA_VISIBLE_DEVICES."""
        from auto_round.utils.device import set_cuda_visible_devices

        original = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            set_cuda_visible_devices("0")
            # Should set CUDA_VISIBLE_DEVICES to "0"
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
        finally:
            if original is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)


class TestHpexAvailable:
    """Test is_hpex_available function."""

    def test_is_hpex_available_returns_bool(self):
        """Test is_hpex_available returns boolean."""
        from auto_round.utils.device import is_hpex_available

        result = is_hpex_available()
        assert isinstance(result, bool)


class TestCheckIsCpu:
    """Test check_is_cpu function."""

    def test_cpu_string(self):
        """Test checking 'cpu' string."""
        from auto_round.utils.device import check_is_cpu

        assert check_is_cpu("cpu") is True

    def test_cpu_torch_device(self):
        """Test checking torch.device('cpu')."""
        import torch

        from auto_round.utils.device import check_is_cpu

        assert check_is_cpu(torch.device("cpu")) is True

    def test_cuda_device(self):
        """Test checking CUDA device returns False."""
        from auto_round.utils.device import check_is_cpu

        assert check_is_cpu("cuda") is False
        assert check_is_cpu("cuda:0") is False


class TestIsPipelineParallelSupported:
    """Test is_pipeline_parallel_supported function."""

    def test_cuda_supported(self):
        """Test CUDA supports pipeline parallel."""
        from auto_round.utils.device import is_pipeline_parallel_supported

        assert is_pipeline_parallel_supported("cuda") is True

    def test_cpu_not_supported(self):
        """Test CPU does not support pipeline parallel."""
        from auto_round.utils.device import is_pipeline_parallel_supported

        assert is_pipeline_parallel_supported("cpu") is False

    def test_xpu_not_supported(self):
        """Test XPU does not support pipeline parallel."""
        from auto_round.utils.device import is_pipeline_parallel_supported

        assert is_pipeline_parallel_supported("xpu") is False

    def test_hpu_not_supported(self):
        """Test HPU does not support pipeline parallel."""
        from auto_round.utils.device import is_pipeline_parallel_supported

        assert is_pipeline_parallel_supported("hpu") is False


class TestCompileFunc:
    """Test compile_func function."""

    def test_compile_func_exists(self):
        """Test compile_func is callable."""
        from auto_round.utils.device import compile_func

        assert callable(compile_func)


class TestGetFirstAvailableAttr:
    """Test get_first_available_attr function."""

    def test_first_attr_exists(self):
        """Test returns first available attribute."""
        from auto_round.utils.device import get_first_available_attr

        class Obj:
            attr1 = "value1"
            attr2 = "value2"

        obj = Obj()
        assert get_first_available_attr(obj, ["attr1", "attr2"]) == "value1"

    def test_second_attr_exists(self):
        """Test returns second attribute when first is None."""
        from auto_round.utils.device import get_first_available_attr

        class Obj:
            attr1 = None
            attr2 = "value2"

        obj = Obj()
        assert get_first_available_attr(obj, ["attr1", "attr2"]) == "value2"

    def test_none_available(self):
        """Test returns default when no attr exists."""
        from auto_round.utils.device import get_first_available_attr

        class Obj:
            pass

        obj = Obj()
        assert get_first_available_attr(obj, ["attr1", "attr2"], "default") == "default"

    def test_no_default(self):
        """Test returns None when no attr exists and no default."""
        from auto_round.utils.device import get_first_available_attr

        class Obj:
            pass

        obj = Obj()
        assert get_first_available_attr(obj, ["attr1", "attr2"]) is None


class TestPatchXpuSdpa:
    """Test patch_xpu_sdpa_drop_causal_mask function."""

    def test_function_exists(self):
        """Test function is callable."""
        from auto_round.utils.device import patch_xpu_sdpa_drop_causal_mask

        assert callable(patch_xpu_sdpa_drop_causal_mask)

    def test_idempotent(self):
        """Test calling multiple times doesn't raise."""
        from auto_round.utils.device import patch_xpu_sdpa_drop_causal_mask

        patch_xpu_sdpa_drop_causal_mask()
        patch_xpu_sdpa_drop_causal_mask()  # Should not raise


class TestPartitionDictNumbers:
    """Test partition_dict_numbers function."""

    def test_partition_into_more_groups_than_items(self):
        """Test partitioning into more groups than items."""
        from auto_round.utils.device import partition_dict_numbers

        number_dict = {"a": 1, "b": 2}
        result = partition_dict_numbers(number_dict, 5)
        assert len(result) == 5

    def test_partition_into_equal_groups(self):
        """Test partitioning into same number of groups as items."""
        from auto_round.utils.device import partition_dict_numbers

        number_dict = {"a": 1, "b": 2, "c": 3}
        result = partition_dict_numbers(number_dict, 3)
        assert len(result) == 3

    def test_partition_into_fewer_groups(self):
        """Test partitioning into fewer groups than items."""
        from auto_round.utils.device import partition_dict_numbers

        number_dict = {"a": 10, "b": 20, "c": 30, "d": 40}
        result = partition_dict_numbers(number_dict, 2)
        assert len(result) == 2

    def test_sums_add_up(self):
        """Test all partitioned values sum to original."""
        from auto_round.utils.device import partition_dict_numbers

        number_dict = {"a": 10, "b": 20, "c": 30}
        result = partition_dict_numbers(number_dict, 2)

        total = sum(sum(g.values()) for g in result)
        assert total == 60


class TestParseAvailableDevices:
    """Test parse_available_devices function."""

    def test_auto_returns_list(self):
        """Test 'auto' returns a list."""
        from auto_round.utils.device import parse_available_devices

        result = parse_available_devices("auto")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_cpu_string(self):
        """Test 'cpu' string returns cpu list."""
        from auto_round.utils.device import parse_available_devices

        result = parse_available_devices("cpu")
        assert result == ["cpu"]

    def test_int_device(self):
        """Test integer device returns indexed device."""
        from auto_round.utils.device import parse_available_devices

        result = parse_available_devices(0)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_none_returns_default(self):
        """Test None returns default device."""
        from auto_round.utils.device import parse_available_devices

        result = parse_available_devices(None)
        assert isinstance(result, list)
        assert len(result) >= 1


class TestGetMoeMemoryRatio:
    """Test get_moe_memory_ratio function."""

    def test_non_moe_module(self):
        """Test non-MoE module returns 1.0."""
        import torch.nn as nn

        from auto_round.utils.device import get_moe_memory_ratio

        # Create a simple non-MoE module
        block = nn.Linear(10, 10)
        ratio, is_moe = get_moe_memory_ratio(block)
        assert ratio == 1.0
        assert is_moe is False


class TestEstimateTuningBlockMem:
    """Test estimate_tuning_block_mem function."""

    def test_returns_correct_tuple_length(self):
        """Test function returns 4-element tuple."""
        import torch
        import torch.nn as nn

        from auto_round.utils.device import estimate_tuning_block_mem

        # Create a simple block with a linear layer
        block = nn.Sequential(nn.Linear(10, 20))
        input_ids = [torch.randn(1, 5)]

        result = estimate_tuning_block_mem(block, input_ids, 1)
        assert (
            len(result) == 4
        )  # layer_memory_dict, layer_activation_memory, block_input_output_memory, additional_memory


class TestIsGaudi2:
    """Test is_gaudi2 function."""

    def test_returns_bool(self):
        """Test is_gaudi2 returns boolean."""
        from auto_round.utils.device import is_gaudi2

        result = is_gaudi2()
        assert isinstance(result, bool)


class TestGetArDevice:
    """Test get_ar_device function."""

    def test_get_cpu_device(self):
        """Test getting CPU device."""
        from auto_round.utils.device_manager import get_ar_device

        device = get_ar_device("cpu")
        assert device is not None

    def test_get_cuda_device(self):
        """Test getting CUDA device."""
        from auto_round.utils.device_manager import get_ar_device

        device = get_ar_device("cuda")
        assert device is not None


class TestDetectDeviceCount:
    """Test detect_device_count function."""

    def test_returns_int(self):
        """Test detect_device_count returns integer."""
        from auto_round.utils.device_manager import detect_device_count

        count = detect_device_count()
        assert isinstance(count, int)
        assert count >= 1


class TestGetAvailableDeviceTypes:
    """Test get_available_device_types function."""

    def test_returns_list(self):
        """Test get_available_device_types returns list."""
        from auto_round.utils.device_manager import get_available_device_types

        types = get_available_device_types()
        assert isinstance(types, list)


class TestGetCurrentDeviceManager:
    """Test get_current_device_manager function."""

    def test_returns_device_manager(self):
        """Test get_current_device_manager returns a manager."""
        from auto_round.utils.device_manager import get_current_device_manager

        manager = get_current_device_manager()
        assert manager is not None
        assert hasattr(manager, "is_available")
        assert hasattr(manager, "type")
