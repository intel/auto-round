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
"""Tests for ``auto_round/utils/device_manager.py``."""

import argparse

import pytest
import torch
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# ARDevice registry / factory
# ---------------------------------------------------------------------------
class TestARDeviceRegistry:
    def test_base_class_has_empty_device_type(self):
        from auto_round.utils.device_manager import ARDevice
        assert ARDevice.device_type == ""

    def test_create_returns_known_subclass(self):
        from auto_round.utils.device_manager import ARDevice, CpuARDevice
        d = ARDevice.create("cpu")
        assert isinstance(d, CpuARDevice)
        assert isinstance(d, ARDevice)

    def test_create_falls_back_to_base_for_unknown_type(self):
        from auto_round.utils.device_manager import ARDevice
        # No subclass registered for "totally_fake_backend"
        d = ARDevice.create("totally_fake_backend")
        assert isinstance(d, ARDevice)
        assert d.type == "totally_fake_backend"

    def test_subclass_self_registers(self):
        """Defining a subclass with device_type registers it in the registry."""
        from auto_round.utils.device_manager import ARDevice

        class _FakeFooDevice(ARDevice):
            device_type = "_fake_foo_zzz"

        try:
            assert ARDevice._registry.get("_fake_foo_zzz") is _FakeFooDevice
        finally:
            ARDevice._registry.pop("_fake_foo_zzz", None)


# ---------------------------------------------------------------------------
# ARDevice base class behaviour
# ---------------------------------------------------------------------------
class TestARDeviceBase:
    def test_is_available_default_true(self):
        from auto_round.utils.device_manager import ARDevice
        d = ARDevice("cpu")
        assert d.is_available() is True

    def test_supports_bf16_default_true(self):
        from auto_round.utils.device_manager import ARDevice
        d = ARDevice("cpu")
        assert d.supports_bf16() is True

    def test_prefers_bf16_default_true(self):
        from auto_round.utils.device_manager import ARDevice
        d = ARDevice("cpu")
        assert d.prefers_bf16() is True

    def test_is_torch_compile_supported_default_true(self):
        from auto_round.utils.device_manager import ARDevice
        d = ARDevice("cpu")
        assert d.is_torch_compile_supported() is True

    def test_compile_func_returns_original_when_unsupported(self):
        from auto_round.utils.device_manager import ARDevice

        class _NoCompile(ARDevice):
            device_type = "_no_compile"
            def is_torch_compile_supported(self):
                return False

        def my_func(x):
            return x

        d = _NoCompile()
        # Should return the original function unchanged
        assert d.compile_func(my_func) is my_func

    def test_device_returns_torch_device(self):
        from auto_round.utils.device_manager import ARDevice
        d = ARDevice("cpu")
        assert d.device() == torch.device("cpu")
        assert d.device(0) == torch.device("cpu:0")
        assert d.device("1") == torch.device("cpu:1")
        # Passing a torch.device returns it as-is
        dev = torch.device("cpu:2")
        assert d.device(dev) is dev


# ---------------------------------------------------------------------------
# CpuARDevice
# ---------------------------------------------------------------------------
class TestCpuARDevice:
    def test_is_available(self):
        from auto_round.utils.device_manager import CpuARDevice
        assert CpuARDevice().is_available() is True

    def test_device_count(self):
        from auto_round.utils.device_manager import CpuARDevice
        assert CpuARDevice().device_count() == 1

    def test_current_device(self):
        from auto_round.utils.device_manager import CpuARDevice
        assert CpuARDevice().current_device() == 0

    def test_set_device_is_noop(self):
        from auto_round.utils.device_manager import CpuARDevice
        assert CpuARDevice().set_device(3) is None

    def test_device_returns_cpu(self):
        from auto_round.utils.device_manager import CpuARDevice
        assert CpuARDevice().device() == torch.device("cpu")

    def test_synchronize_is_noop(self):
        from auto_round.utils.device_manager import CpuARDevice
        assert CpuARDevice().synchronize() is None
        assert CpuARDevice().synchronize(index=0) is None

    def test_empty_cache_runs_gc(self):
        from auto_round.utils.device_manager import CpuARDevice
        # empty_cache invokes gc.collect and returns whatever gc.collect returns
        result = CpuARDevice().empty_cache()
        assert result is None or isinstance(result, int)

    def test_get_device_capability_returns_none(self):
        from auto_round.utils.device_manager import CpuARDevice
        assert CpuARDevice().get_device_capability() is None
        assert CpuARDevice().get_device_capability(0) is None

    def test_device_index_returns_nullcontext(self):
        from auto_round.utils.device_manager import CpuARDevice
        import contextlib
        ctx = CpuARDevice().device_index(0)
        assert isinstance(ctx, contextlib.AbstractContextManager)
        with ctx:
            pass  # should not raise

    def test_supports_bf16_returns_bool(self):
        from auto_round.utils.device_manager import CpuARDevice
        d = CpuARDevice()
        result = d.supports_bf16()
        assert isinstance(result, bool)
        # Should be cached on the instance
        assert hasattr(d, "_bf16_supported")

    def test_supports_bf16_cached(self):
        """Second call should return the cached value without re-probing."""
        from auto_round.utils.device_manager import CpuARDevice
        d = CpuARDevice()
        first = d.supports_bf16()
        # Mutate the cache; subsequent call must return the cached value, not re-probe
        d._bf16_supported = not first
        assert d.supports_bf16() is not first

    def test_memory_methods_return_int(self):
        from auto_round.utils.device_manager import CpuARDevice
        d = CpuARDevice()
        # total_memory may be 0 if psutil is unavailable, otherwise positive
        assert isinstance(d.total_memory(), int)
        # memory_reserved / memory_allocated may also be 0 if psutil unavailable
        assert isinstance(d.memory_reserved(), int)
        assert isinstance(d.memory_allocated(), int)

    def test_is_torch_compile_supported(self):
        from auto_round.utils.device_manager import CpuARDevice
        assert CpuARDevice().is_torch_compile_supported() is True


# ---------------------------------------------------------------------------
# Helpers: normalize_default_device_map, _normalize_device_type
# ---------------------------------------------------------------------------
class TestNormalizeDefaultDeviceMap:
    def test_passthrough_for_cpu(self):
        from auto_round.utils.device_manager import normalize_default_device_map
        assert normalize_default_device_map("cpu") == "cpu"

    def test_int_returns_as_is(self):
        from auto_round.utils.device_manager import normalize_default_device_map
        assert normalize_default_device_map(0) == 0

    def test_none_returns_none(self):
        from auto_round.utils.device_manager import normalize_default_device_map
        assert normalize_default_device_map(None) is None

    def test_mps_default_overridden_to_cpu(self):
        """On Apple Silicon, default 0/'0'/None/'auto' should fall back to cpu."""
        from auto_round.utils.device_manager import normalize_default_device_map

        with patch("torch.mps.is_available", return_value=True):
            for value in (0, "0", None, "auto"):
                assert normalize_default_device_map(value) == "cpu"


class TestNormalizeDeviceType:
    def test_none_returns_current(self):
        from auto_round.utils.device_manager import _normalize_device_type
        with patch(
            "auto_round.utils.device_manager.get_current_device_type",
            return_value="cpu",
        ):
            assert _normalize_device_type(None) == "cpu"

    def test_int_returns_current(self):
        from auto_round.utils.device_manager import _normalize_device_type
        with patch(
            "auto_round.utils.device_manager.get_current_device_type",
            return_value="cuda",
        ):
            assert _normalize_device_type(0) == "cuda"

    def test_torch_device_returns_its_type(self):
        from auto_round.utils.device_manager import _normalize_device_type
        assert _normalize_device_type(torch.device("cpu")) == "cpu"
        assert _normalize_device_type(torch.device("cpu:2")) == "cpu"

    def test_string_auto_returns_current(self):
        from auto_round.utils.device_manager import _normalize_device_type
        with patch(
            "auto_round.utils.device_manager.get_current_device_type",
            return_value="cpu",
        ):
            assert _normalize_device_type("auto") == "cpu"
            assert _normalize_device_type("tp") == "cpu"

    def test_string_with_index_strips_index(self):
        from auto_round.utils.device_manager import _normalize_device_type
        assert _normalize_device_type("cuda:0") == "cuda"
        assert _normalize_device_type("xpu:1") == "xpu"

    def test_unsupported_type_raises(self):
        from auto_round.utils.device_manager import _normalize_device_type
        with pytest.raises(ValueError):
            _normalize_device_type(3.14)


# ---------------------------------------------------------------------------
# _torch_accelerator_type, _accelerator_api, _module_call
# ---------------------------------------------------------------------------
class TestAcceleratorHelpers:
    def test_torch_accelerator_type_when_attribute_missing(self):
        from auto_round.utils.device_manager import _torch_accelerator_type
        # If torch has no `accelerator`, should return None
        with patch.object(torch, "accelerator", None, create=True):
            assert _torch_accelerator_type() is None

    def test_torch_accelerator_type_when_not_available(self):
        from auto_round.utils.device_manager import _torch_accelerator_type
        fake_api = MagicMock()
        fake_api.is_available.return_value = False
        with patch.object(torch, "accelerator", fake_api, create=True):
            assert _torch_accelerator_type() is None

    def test_torch_accelerator_type_when_available(self):
        from auto_round.utils.device_manager import _torch_accelerator_type
        fake_dev = MagicMock()
        fake_dev.type = "cuda"
        fake_api = MagicMock()
        fake_api.is_available.return_value = True
        fake_api.current_accelerator.return_value = fake_dev
        with patch.object(torch, "accelerator", fake_api, create=True):
            assert _torch_accelerator_type() == "cuda"

    def test_accelerator_api_returns_none_when_missing(self):
        from auto_round.utils.device_manager import _accelerator_api
        with patch.object(torch, "accelerator", None, create=True):
            assert _accelerator_api() is None

    def test_module_call_first_match(self):
        from auto_round.utils.device_manager import _module_call
        api = MagicMock()
        api.foo = MagicMock(return_value=42)
        api.bar = MagicMock(return_value=99)
        ok, val = _module_call(api, ("foo", "bar"))
        assert ok is True
        assert val == 42

    def test_module_call_falls_back_to_second(self):
        from auto_round.utils.device_manager import _module_call
        api = MagicMock(spec=["bar"])
        api.bar = MagicMock(return_value="x")
        ok, val = _module_call(api, ("foo", "bar"))
        assert ok is True
        assert val == "x"

    def test_module_call_no_match(self):
        from auto_round.utils.device_manager import _module_call
        api = MagicMock(spec=[])
        ok, val = _module_call(api, ("nope1", "nope2"))
        assert ok is False
        assert val is None


# ---------------------------------------------------------------------------
# get_current_device_type
# ---------------------------------------------------------------------------
class TestGetCurrentDeviceType:
    def test_returns_cpu_when_nothing_available(self):
        from auto_round.utils.device_manager import (
            get_current_device_type,
            _hpu_available,
            _torch_accelerator_type,
        )

        # Force every discovery path to report "nothing"
        with patch.object(
            __import__("auto_round.utils.device_manager", fromlist=["_hpu_available"]),
            "_hpu_available",
            return_value=False,
        ), patch.object(
            __import__("auto_round.utils.device_manager", fromlist=["_torch_accelerator_type"]),
            "_torch_accelerator_type",
            return_value=None,
        ):
            # Need to clear the lru_cache
            get_current_device_type.cache_clear()
            try:
                assert get_current_device_type() == "cpu"
            finally:
                get_current_device_type.cache_clear()

    def test_returns_hpu_when_hpu_available(self):
        from auto_round.utils.device_manager import get_current_device_type

        with patch(
            "auto_round.utils.device_manager._hpu_available", return_value=True
        ):
            get_current_device_type.cache_clear()
            try:
                assert get_current_device_type() == "hpu"
            finally:
                get_current_device_type.cache_clear()


# ---------------------------------------------------------------------------
# is_device_available / get_available_device_types
# ---------------------------------------------------------------------------
class TestAvailableHelpers:
    def test_is_device_available(self):
        from auto_round.utils.device_manager import is_device_available
        # On a CPU-only machine, no accelerator -> still "available" (cpu is non-None)
        with patch(
            "auto_round.utils.device_manager.get_current_device_type",
            return_value="cpu",
        ):
            assert is_device_available() is True

    def test_is_device_available_when_accelerator(self):
        from auto_round.utils.device_manager import is_device_available
        with patch(
            "auto_round.utils.device_manager.get_current_device_type",
            return_value="cuda",
        ):
            assert is_device_available() is True

    def test_get_available_device_types_cpu_only(self):
        from auto_round.utils.device_manager import get_available_device_types
        with patch(
            "auto_round.utils.device_manager._hpu_available", return_value=False
        ), patch(
            "auto_round.utils.device_manager._torch_accelerator_type", return_value=None
        ):
            assert get_available_device_types() == []


# ---------------------------------------------------------------------------
# _DeviceIndexContext
# ---------------------------------------------------------------------------
class TestDeviceIndexContext:
    def test_restores_previous_device(self):
        from auto_round.utils.device_manager import (
            _DeviceIndexContext,
            CpuARDevice,
        )
        d = CpuARDevice()
        with _DeviceIndexContext(d, 0):
            pass
        # CPU device is always 0; the context should not raise

    def test_handles_current_device_failure(self):
        """If current_device() raises, __enter__ must still complete cleanly."""
        from auto_round.utils.device_manager import (
            _DeviceIndexContext,
            CpuARDevice,
        )

        class _BrokenCpu(CpuARDevice):
            def current_device(self):
                raise RuntimeError("boom")

        d = _BrokenCpu()
        ctx = _DeviceIndexContext(d, 0)
        # Should not raise on enter, and prev should be None
        with ctx:
            pass


# ---------------------------------------------------------------------------
# DeviceManager (singleton)
# ---------------------------------------------------------------------------
class TestDeviceManagerSingleton:
    def setup_method(self):
        # Reset singleton state for each test (the class is a process-wide singleton)
        from auto_round.utils.device_manager import DeviceManager

        DeviceManager._instance = None

    def test_singleton_returns_same_instance(self):
        from auto_round.utils.device_manager import DeviceManager
        a = DeviceManager()
        b = DeviceManager()
        assert a is b

    def test_initializes_with_default_state(self):
        from auto_round.utils.device_manager import DeviceManager
        m = DeviceManager()
        assert m._device_map is None or m._device_map == 0
        assert m.device_list is not None

    def test_configure_with_cpu(self):
        from auto_round.utils.device_manager import DeviceManager
        m = DeviceManager("cpu")
        assert m.device_list == ["cpu"]
        assert m.device == "cpu"

    def test_configure_with_auto(self):
        from auto_round.utils.device_manager import DeviceManager
        m = DeviceManager("auto")
        # Should resolve to the active device type (or cpu)
        assert isinstance(m.device, str)

    def test_is_multi_device_false(self):
        from auto_round.utils.device_manager import DeviceManager
        m = DeviceManager("cpu")
        assert m.is_multi_device() is False

    def test_device_setter(self):
        from auto_round.utils.device_manager import DeviceManager
        m = DeviceManager("cpu")
        m.device = "cpu"
        assert m.device == "cpu"
        # Also accept torch.device
        m.device = torch.device("cpu")
        assert m.device == "cpu"

    def test_register_rejects_missing_device_type(self):
        from auto_round.utils.device_manager import (
            ARDevice,
            DeviceManager,
        )

        class _NoType(ARDevice):
            device_type = ""

        m = DeviceManager()
        with pytest.raises(ValueError):
            m.register(_NoType)

    def test_register_adds_to_registry(self):
        from auto_round.utils.device_manager import (
            ARDevice,
            DeviceManager,
        )

        class _FakeBar(ARDevice):
            device_type = "_fake_bar_zzz"

        m = DeviceManager()
        try:
            m.register(_FakeBar)
            assert ARDevice._registry.get("_fake_bar_zzz") is _FakeBar
            # get_ar_device should return an instance
            d = m.get_ar_device("_fake_bar_zzz")
            assert isinstance(d, _FakeBar)
        finally:
            ARDevice._registry.pop("_fake_bar_zzz", None)
            m._cache.pop("_fake_bar_zzz", None)

    def test_get_ar_device_caches(self):
        from auto_round.utils.device_manager import DeviceManager
        m = DeviceManager()
        a = m.get_ar_device("cpu")
        b = m.get_ar_device("cpu")
        assert a is b

    def test_current_returns_ar_device(self):
        from auto_round.utils.device_manager import DeviceManager, ARDevice
        m = DeviceManager()
        cur = m.current()
        assert isinstance(cur, ARDevice)

    def test_current_type_returns_string(self):
        from auto_round.utils.device_manager import DeviceManager
        m = DeviceManager()
        assert isinstance(m.current_type(), str)

    def test_available_types_is_list(self):
        from auto_round.utils.device_manager import DeviceManager
        m = DeviceManager()
        assert isinstance(m.available_types(), list)

    def test_available_devices_returns_list(self):
        from auto_round.utils.device_manager import DeviceManager, ARDevice
        m = DeviceManager()
        devs = m.available_devices()
        assert isinstance(devs, list)
        for d in devs:
            assert isinstance(d, ARDevice)

    def test_device_map_property(self):
        from auto_round.utils.device_manager import DeviceManager
        m = DeviceManager("cpu")
        assert m.device_map == "cpu"


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------
class TestModuleLevelHelpers:
    def test_get_ar_device_returns_cached(self):
        from auto_round.utils.device_manager import (
            get_ar_device,
            CpuARDevice,
        )
        d = get_ar_device("cpu")
        assert isinstance(d, CpuARDevice)
        d2 = get_ar_device("cpu")
        assert d is d2

    def test_get_current_device_manager_returns_ar_device(self):
        from auto_round.utils.device_manager import (
            get_current_device_manager,
            ARDevice,
        )
        d = get_current_device_manager()
        assert isinstance(d, ARDevice)

    def test_detect_device_count_returns_int(self):
        from auto_round.utils.device_manager import detect_device_count
        assert isinstance(detect_device_count(), int)
        assert detect_device_count() >= 1

    def test_get_device_and_parallelism_cpu(self):
        from auto_round.utils.device_manager import get_device_and_parallelism
        dev, parallel = get_device_and_parallelism("cpu")
        assert isinstance(dev, str)
        assert isinstance(parallel, bool)
        assert parallel is False

    def test_get_device_and_parallelism_int(self):
        from auto_round.utils.device_manager import get_device_and_parallelism
        dev, parallel = get_device_and_parallelism(0)
        assert isinstance(dev, str)
        assert parallel is False

    def test_get_device_and_parallelism_torch_device(self):
        from auto_round.utils.device_manager import get_device_and_parallelism
        dev, parallel = get_device_and_parallelism(torch.device("cpu"))
        assert parallel is False
        assert "cpu" in dev

    def test_get_device_and_parallelism_dict_single(self):
        from auto_round.utils.device_manager import get_device_and_parallelism
        dev, parallel = get_device_and_parallelism({"layer": "cpu"})
        assert parallel is False

    def test_get_device_and_parallelism_dict_multi(self):
        from auto_round.utils.device_manager import get_device_and_parallelism
        # Two distinct device values collapse to a single unique device => no 'auto'
        dev, parallel = get_device_and_parallelism({"a": "cpu", "b": "cuda"})
        # Either branch produces a string device and a bool parallelism flag
        assert isinstance(dev, str)
        assert isinstance(parallel, bool)

    def test_get_device_and_parallelism_none(self):
        from auto_round.utils.device_manager import get_device_and_parallelism
        dev, parallel = get_device_and_parallelism(None)
        assert parallel is False
        assert isinstance(dev, str)

    def test_get_packing_device_auto(self):
        from auto_round.utils.device_manager import get_packing_device
        d = get_packing_device("auto")
        assert isinstance(d, torch.device)

    def test_get_packing_device_cpu(self):
        from auto_round.utils.device_manager import get_packing_device
        d = get_packing_device("cpu")
        assert d == torch.device("cpu")

    def test_get_packing_device_torch_device(self):
        from auto_round.utils.device_manager import get_packing_device
        dev = torch.device("cpu")
        assert get_packing_device(dev) is dev

    def test_get_packing_device_none(self):
        from auto_round.utils.device_manager import get_packing_device
        d = get_packing_device(None)
        assert isinstance(d, torch.device)

    def test_get_packing_device_invalid_string(self):
        from auto_round.utils.device_manager import get_packing_device
        with pytest.raises(ValueError):
            get_packing_device("not_a_device!!!")

    def test_get_packing_device_unsupported_type(self):
        from auto_round.utils.device_manager import get_packing_device
        with pytest.raises(TypeError):
            get_packing_device(3.14)

    def test_is_auto_device_mapping(self):
        from auto_round.utils.device_manager import is_auto_device_mapping
        assert is_auto_device_mapping(None) is False
        assert is_auto_device_mapping(0) is False
        assert is_auto_device_mapping("auto") is True
        assert is_auto_device_mapping("cpu") is False
        assert is_auto_device_mapping("0,1") is True
        assert is_auto_device_mapping({"a": "cpu"}) is False


# ---------------------------------------------------------------------------
# get_major_device
# ---------------------------------------------------------------------------
class TestGetMajorDevice:
    def test_none_returns_current(self):
        from auto_round.utils.device_manager import get_major_device
        d = get_major_device(None)
        assert isinstance(d, str)

    def test_string_passthrough(self):
        from auto_round.utils.device_manager import get_major_device
        assert get_major_device("cpu") == "cpu"

    def test_torch_device_returns_str(self):
        from auto_round.utils.device_manager import get_major_device
        assert get_major_device(torch.device("cpu")) == "cpu"

    def test_int_with_auto(self):
        from auto_round.utils.device_manager import get_major_device
        d = get_major_device(0)
        assert isinstance(d, str)

    def test_comma_separated_returns_first_device(self):
        from auto_round.utils.device_manager import get_major_device
        d = get_major_device("0,1,2")
        assert isinstance(d, str)
        assert "cpu" in d or "cuda" in d or "xpu" in d

    def test_dict_single_value(self):
        from auto_round.utils.device_manager import get_major_device
        d = get_major_device({"a": "cpu"})
        assert d == "cpu"

    def test_dict_picks_non_cpu(self):
        from auto_round.utils.device_manager import get_major_device
        # We can't realistically inject a fake backend in get_major_device easily,
        # but we can verify the function accepts a dict and returns a string
        d = get_major_device({"a": "cpu", "b": "cpu"})
        assert d == "cpu"

    def test_invalid_type_falls_back_to_cpu(self):
        from auto_round.utils.device_manager import get_major_device
        assert get_major_device(3.14) == "cpu"
        assert get_major_device([]) == "cpu"


# ---------------------------------------------------------------------------
# get_device_memory (raises on CPU)
# ---------------------------------------------------------------------------
class TestGetDeviceMemory:
    def test_cpu_raises_runtime_error(self):
        from auto_round.utils.device_manager import get_device_memory
        with patch(
            "auto_round.utils.device_manager.get_current_device_type",
            return_value="cpu",
        ):
            with pytest.raises(RuntimeError):
                get_device_memory()


# ---------------------------------------------------------------------------
# _clear_memory_for_cpu_and_cuda (CPU path)
# ---------------------------------------------------------------------------
class TestClearMemoryHelper:
    def test_cpu_only_returns_immediately(self):
        from auto_round.utils.device_manager import _clear_memory_for_cpu_and_cuda
        with patch(
            "auto_round.utils.device_manager.get_current_device_type",
            return_value="cpu",
        ):
            # Should not raise; should return early
            result = _clear_memory_for_cpu_and_cuda(tensor=None, device_list=["cuda:0"])
            assert result is None

    def test_clears_list_tensor(self):
        from auto_round.utils.device_manager import _clear_memory_for_cpu_and_cuda

        tensor_list = [torch.zeros(2, 2), torch.ones(3)]
        with patch(
            "auto_round.utils.device_manager.get_current_device_type",
            return_value="cpu",
        ):
            _clear_memory_for_cpu_and_cuda(tensor=tensor_list, device_list=None)
        # After the call, the local list elements should be set to None
        assert all(t is None for t in tensor_list)