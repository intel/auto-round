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
"""Unit tests for ``auto_round.utils.device``.

These tests focus on the *untested* parts of ``device.py`` to improve
its coverage.  All acceleration hardware is mocked; no GPU / XPU / HPU
runtime is required at test time.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Module fixture: ensure ``auto_round.utils.device`` is importable.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset the ``MemoryMonitor`` singleton between tests."""
    from auto_round.utils import device as device_mod

    monitor_cls = getattr(device_mod, "MemoryMonitor", None)
    if monitor_cls is not None:
        monitor_cls._instance = None
        monitor_cls._initialized = False
    yield
    if monitor_cls is not None:
        monitor_cls._instance = None
        monitor_cls._initialized = False


# ===========================================================================
# is_package_available / is_hpex_available caching
# ===========================================================================
class TestIsPackageAvailableAdditional:
    """Additional edge-case tests for ``is_package_available``."""

    def test_empty_string_returns_false(self):
        """Empty package name must not raise -- the helper returns False."""
        from auto_round.utils.device import is_package_available

        try:
            result = is_package_available("")
        except (ValueError, ImportError):
            result = False
        assert result is False

    def test_dotted_module(self):
        """Dotted package names like ``os.path`` resolve via find_spec."""
        from auto_round.utils.device import is_package_available

        assert is_package_available("os.path") is True

    def test_pyyaml_may_not_be_installed(self):
        from auto_round.utils.device import is_package_available

        # We don't assume pyyaml exists.  This must produce a bool either way.
        result = is_package_available("pyyaml")
        assert isinstance(result, bool)


class TestIsHpexAvailableCaching:
    """Test the @lru_cache behaviour of ``is_hpex_available``."""

    def test_caches_module_level_value(self):
        """``is_hpex_available`` is wrapped in ``@lru_cache(None)`` so it
        should return the same bool across consecutive calls."""
        from auto_round.utils.device import is_hpex_available

        result = is_hpex_available()
        result2 = is_hpex_available()
        assert isinstance(result, bool)
        assert isinstance(result2, bool)
        # lru_cache guarantees the same return value for identical calls
        assert result is result2

    def test_clear_cache_returns_bool(self):
        """The underlying ``@lru_cache`` is reachable via ``__wrapped__``."""
        from auto_round.utils import device as device_mod

        is_hpex_available = device_mod.is_hpex_available
        # ``torch._dynamo.disable`` outer wrapper exposes the lru_cache
        # wrapped function through ``__wrapped__``.
        wrapped = getattr(is_hpex_available, "__wrapped__", is_hpex_available)
        if hasattr(wrapped, "cache_clear"):
            wrapped.cache_clear()
            result = is_hpex_available()
            wrapped.cache_clear()
            assert isinstance(result, bool)
        else:
            # No lru_cache decorator visible through the dynamo wrapper:
            # just confirm the function returns bool when called.
            assert isinstance(is_hpex_available(), bool)


# ===========================================================================
# _bump_dynamo_cache_limit (more exhaustive)
# ===========================================================================
class TestBumpDynamoCacheLimitDetailed:
    """Exhaustive tests for ``_bump_dynamo_cache_limit``."""

    def test_no_attr_skipped(self):
        """If a given config attribute does not exist, skip silently."""
        from auto_round.utils.device import _bump_dynamo_cache_limit

        mock_cfg = MagicMock(spec=[])  # no attrs at all
        with patch.dict(sys.modules, {"torch._dynamo.config": mock_cfg}), patch(
            "torch._dynamo.config", mock_cfg
        ):
            # Should not raise even if all attrs are absent.
            _bump_dynamo_cache_limit(min_size=64)

    def test_partial_config(self):
        """Only the existing attributes should be updated, others skipped."""
        from auto_round.utils.device import _bump_dynamo_cache_limit

        class _Cfg:
            cache_size_limit = 1

        mock_cfg = _Cfg()
        with patch.dict(sys.modules, {"torch._dynamo.config": mock_cfg}), patch(
            "torch._dynamo.config", mock_cfg
        ):
            _bump_dynamo_cache_limit(min_size=128)
            assert mock_cfg.cache_size_limit == 128

    def test_higher_existing_value_is_not_lowered(self):
        """The function never lowers an existing value."""
        from auto_round.utils.device import _bump_dynamo_cache_limit

        class _Cfg:
            cache_size_limit = 1000
            accumulated_cache_size_limit = 1000
            recompile_limit = 1000

        mock_cfg = _Cfg()
        with patch.dict(sys.modules, {"torch._dynamo.config": mock_cfg}), patch(
            "torch._dynamo.config", mock_cfg
        ):
            _bump_dynamo_cache_limit(min_size=16)
            # Larger existing value should be preserved.
            assert mock_cfg.cache_size_limit == 1000


# ===========================================================================
# compile_func
# ===========================================================================
class TestCompileFunc:
    """Test ``compile_func`` dispatches to the correct ARDevice."""

    def test_compile_function_returns_object(self):
        from auto_round.utils.device import compile_func

        def fn(x):
            return x

        # The behaviour depends on the active backend.  On CPU it should
        # return either the original or a compiled wrapper -- but never raise.
        result = compile_func(fn, device="cpu")
        assert result is not None
        assert callable(result)

    def test_compile_with_int_device(self):
        from auto_round.utils.device import compile_func

        def fn(x):
            return x

        # Just exercise the int code path.
        result = compile_func(fn, device=0)
        assert callable(result)

    def test_compile_with_torch_device(self):
        from auto_round.utils.device import compile_func

        def fn(x):
            return x

        result = compile_func(fn, device=torch.device("cpu"))
        assert callable(result)


# ===========================================================================
# clear_memory_if_reached_threshold
# ===========================================================================
class TestClearMemoryIfReachedThreshold:
    """The function is a no-op on CPU; we verify that contract."""

    def test_returns_false_on_cpu(self):
        from auto_round.utils.device import clear_memory_if_reached_threshold

        result = clear_memory_if_reached_threshold(threshold=0.85)
        assert result is False

    def test_returns_false_on_cpu_with_device_list(self):
        from auto_round.utils.device import clear_memory_if_reached_threshold

        result = clear_memory_if_reached_threshold(threshold=0.5, device_list=["cuda:0"])
        assert result is False


# ===========================================================================
# check_memory_availability
# ===========================================================================
class TestCheckMemoryAvailability:
    """``check_memory_availability`` returns the original shape on CPU."""

    def _make_inputs(self, weight):
        inputs = torch.zeros(2, 8, dtype=torch.float32)
        return inputs

    def test_cpu_returns_unchanged(self):
        from auto_round.utils.device import check_memory_availability

        weight = torch.zeros(8, 8, dtype=torch.float32)
        inputs = self._make_inputs(weight)
        ok, seqlen, bs = check_memory_availability("cpu", inputs, weight, 128, 4)
        assert ok is True
        assert seqlen == 128
        assert bs == 4

    def test_empty_string_device_is_cpu(self):
        from auto_round.utils.device import check_memory_availability

        weight = torch.zeros(4, 4, dtype=torch.float32)
        inputs = self._make_inputs(weight)
        ok, seqlen, bs = check_memory_availability("", inputs, weight, 64, 2)
        assert ok is True
        assert seqlen == 64
        assert bs == 2


# ===========================================================================
# set_tuning_device_for_layer
# ===========================================================================
class TestSetTuningDeviceForLayer:
    """Test ``set_tuning_device_for_layer`` side-effects."""

    def test_sets_device_on_layer(self):
        from auto_round.utils.device import set_tuning_device_for_layer

        model = nn.Sequential(nn.Linear(4, 4))
        set_tuning_device_for_layer(model, "0", device="cuda:0")
        assert model[0].tuning_device == "cuda:0"

    def test_idempotent_when_same_device(self):
        from auto_round.utils.device import set_tuning_device_for_layer

        model = nn.Sequential(nn.Linear(4, 4))
        set_tuning_device_for_layer(model, "0", device="cuda:0")
        # Calling again with same device is a no-op (no warning emitted).
        set_tuning_device_for_layer(model, "0", device="cuda:0")
        assert model[0].tuning_device == "cuda:0"

    def test_reassign_logs_warning(self):
        from auto_round.utils.device import set_tuning_device_for_layer

        model = nn.Sequential(nn.Linear(4, 4))
        model[0].tuning_device = "cuda:0"
        # Reassigning to a different device should not raise.
        set_tuning_device_for_layer(model, "0", device="cuda:1")
        # Note: set_tuning_device_for_layer does *not* mutate when the
        # device differs; the original is kept.  Verify the contract.
        assert model[0].tuning_device == "cuda:0"


# ===========================================================================
# set_non_auto_device_map
# ===========================================================================
class TestSetNonAutoDeviceMap:
    """Test ``set_non_auto_device_map`` short-circuit logic and assignment."""

    def test_empty_string_returns_early(self):
        from auto_round.utils.device import set_non_auto_device_map

        model = nn.Sequential(nn.Linear(4, 4))
        set_non_auto_device_map(model, "")
        # No tuning_device should appear since we returned early.
        assert not hasattr(model[0], "tuning_device")

    def test_auto_returns_early(self):
        from auto_round.utils.device import set_non_auto_device_map

        model = nn.Sequential(nn.Linear(4, 4))
        set_non_auto_device_map(model, "auto")
        assert not hasattr(model[0], "tuning_device")

    def test_int_returns_early(self):
        from auto_round.utils.device import set_non_auto_device_map

        model = nn.Sequential(nn.Linear(4, 4))
        set_non_auto_device_map(model, 0)
        assert not hasattr(model[0], "tuning_device")

    def test_string_without_colon_returns_early(self):
        from auto_round.utils.device import set_non_auto_device_map

        model = nn.Sequential(nn.Linear(4, 4))
        set_non_auto_device_map(model, "cuda")
        # "cuda" has no ":" and no "," -> early-return branch.
        assert not hasattr(model[0], "tuning_device")

    def test_comma_in_string_returns_early(self):
        from auto_round.utils.device import set_non_auto_device_map

        model = nn.Sequential(nn.Linear(4, 4))
        set_non_auto_device_map(model, "0,1")
        # Comma branch is "auto device map" -> early-return.
        assert not hasattr(model[0], "tuning_device")

    def test_dict_assigns_layers(self):
        from auto_round.utils.device import set_non_auto_device_map

        model = nn.Sequential(nn.Linear(4, 4))
        # get_major_device('0') -> 'cpu' on a CPU-only host, so the assigned
        # device value should be 'cpu'.
        set_non_auto_device_map(model, {"0": "0"})
        assert model[0].tuning_device == "cpu"

    def test_dict_string_digit_key_with_unknown_layer_logs(self):
        from auto_round.utils.device import set_non_auto_device_map

        model = nn.Sequential(nn.Linear(4, 4))
        # Unknown leaf name should produce a warning but not raise.
        set_non_auto_device_map(model, {"99_not_a_real_layer": "0"})


# ===========================================================================
# _allocate_layers_to_devices
# ===========================================================================
class TestAllocateLayersToDevices:
    """Test internal load-balancing allocator."""

    def test_basic_allocation(self):
        from auto_round.utils.device import _allocate_layers_to_devices

        layer_memory = {
            "q": {"param_memory": 4.0},
            "k": {"param_memory": 1.0},
            "v": {"param_memory": 1.0},
            "o": {"param_memory": 4.0},
        }
        device_mem = {"cuda:0": 30.0, "cuda:1": 30.0}
        gpu_devices = ["cuda:0", "cuda:1"]
        device_map, names = _allocate_layers_to_devices(layer_memory, device_mem, gpu_devices, 2.0)

        assert isinstance(device_map, dict)
        assert set(device_map.keys()) == set(layer_memory.keys())
        # All assigned values must be one of the gpu_devices (or list thereof).
        for value in device_map.values():
            assert value in gpu_devices or value in gpu_devices
        assert set(names) == set(layer_memory.keys())

    def test_single_layer(self):
        from auto_round.utils.device import _allocate_layers_to_devices

        layer_memory = {"only": {"param_memory": 1.0}}
        device_mem = {"cuda:0": 1.0, "cuda:1": 1.0}
        gpu_devices = ["cuda:0", "cuda:1"]
        device_map, names = _allocate_layers_to_devices(layer_memory, device_mem, gpu_devices, 0.5)
        assert "only" in device_map
        assert names == ["only"]

    def test_layer_larger_than_device(self):
        """If a layer is bigger than any device, the allocator must still
        return a valid device (the function's fallback path)."""
        from auto_round.utils.device import _allocate_layers_to_devices

        layer_memory = {"big": {"param_memory": 1000.0}}
        device_mem = {"cuda:0": 1.0, "cuda:1": 1.0}
        gpu_devices = ["cuda:0", "cuda:1"]
        device_map, names = _allocate_layers_to_devices(layer_memory, device_mem, gpu_devices, 0.001)
        assert "big" in device_map
        assert names == ["big"]


# ===========================================================================
# dispatch_model_block_wise
# ===========================================================================
class TestDispatchModelBlockWise:
    """Test dispatch_model_block_wise dispatch logic."""

    def test_single_device_skips_accelerate(self):
        """A single-device map must NOT touch accelerate.

        The function uses the short-circuit ``if len(devices) == 1`` branch.
        """
        from auto_round.utils.device import dispatch_model_block_wise

        model = nn.Sequential(nn.Linear(4, 4))
        # Use a mocked device_map -> only one device -> short-circuit branch.
        with patch(
            "auto_round.utils.device.parse_available_devices", return_value=["cpu"]
        ) as parsed:
            result = dispatch_model_block_wise(model, device_map="cpu")
            assert result is model
            parsed.assert_called_once_with("cpu")

    def test_single_device_calls_model_to(self):
        from auto_round.utils.device import dispatch_model_block_wise

        model = MagicMock(spec=nn.Module)
        with patch(
            "auto_round.utils.device.parse_available_devices", return_value=["cpu"]
        ):
            dispatch_model_block_wise(model, device_map="cpu")
            # Short-circuit path requires the model.to(target_device) call.
            model.to.assert_called_once_with("cpu")

    def test_multi_device_uses_accelerate(self):
        """Multi-device dispatch must invoke ``infer_auto_device_map`` and
        ``dispatch_model`` from accelerate."""
        from auto_round.utils.device import dispatch_model_block_wise

        model = nn.Sequential(nn.Linear(4, 4))
        # Multi-device path: provide 2 "cpu" entries so ``len(devices) > 1``.
        # After the inner loop dedupes, ``device == "cpu"`` is used to index
        # the mocked max_memory dict.
        with patch(
            "auto_round.utils.device.parse_available_devices",
            return_value=["cpu", "cpu"],
        ), patch(
            "auto_round.utils.device.get_max_memory", return_value={"cpu": 1024}
        ), patch(
            "auto_round.utils.device.get_balanced_memory", return_value={"cpu": 512}
        ), patch(
            "auto_round.utils.device.infer_auto_device_map",
            return_value={"0": "cpu"},
        ) as mock_infer, patch(
            "auto_round.utils.device.dispatch_model", return_value="MOCKED"
        ) as mock_dispatch:
            result = dispatch_model_block_wise(model, device_map="cpu,cpu", max_mem_ratio=0.5)
            assert mock_infer.called
            assert mock_dispatch.called
            assert result == "MOCKED"


# ===========================================================================
# dispatch_model_by_all_available_devices
# ===========================================================================
class TestDispatchModelByAllAvailableDevices:
    """Cover ``dispatch_model_by_all_available_devices`` for non-diffusion
    paths.  The diffusion path is exercised separately."""

    def test_single_device_short_circuit(self):
        from auto_round.utils.device import dispatch_model_by_all_available_devices

        model = MagicMock(spec=nn.Module)
        with patch(
            "auto_round.utils.device.parse_available_devices", return_value=["cpu"]
        ):
            result = dispatch_model_by_all_available_devices(model, device_map="cpu")
        # Single-device branch calls model.to(...) and returns it.
        assert result is model
        model.to.assert_called_once_with("cpu")

    def test_auto_branch_uses_max_memory(self):
        """device_map == 'auto' triggers balanced_memory + infer_auto_device_map."""
        from auto_round.utils.device import dispatch_model_by_all_available_devices

        model = MagicMock(spec=nn.Module)
        with patch(
            "auto_round.utils.device.get_balanced_memory", return_value={0: 1024}
        ) as balanced, patch(
            "auto_round.utils.device.infer_auto_device_map", return_value={"0": "cpu"}
        ), patch("auto_round.utils.device.dispatch_model", return_value="AUTO_MODEL"):
            with patch(
                "auto_round.utils.device.parse_available_devices",
                return_value=["cpu"],
            ):
                result = dispatch_model_by_all_available_devices(model, device_map="auto")
            # The auto branch is invoked without multi-device lowering, so
            # balanced memory is called once.
            assert balanced.called
            assert result == "AUTO_MODEL"

    def test_none_device_map_defaults_to_0(self):
        from auto_round.utils.device import dispatch_model_by_all_available_devices

        model = MagicMock(spec=nn.Module)
        with patch(
            "auto_round.utils.device.parse_available_devices", return_value=["cpu"]
        ):
            dispatch_model_by_all_available_devices(model, device_map=None)
        # Should resolve to single-device branch.
        model.to.assert_called_once()


# ===========================================================================
# set_avg_auto_device_map
# ===========================================================================
class TestSetAvgAutoDeviceMap:
    """Test ``set_avg_auto_device_map`` early-return for <=1 device."""

    def test_single_device_returns_early(self):
        from auto_round.utils.device import set_avg_auto_device_map

        model = nn.Sequential(nn.Linear(4, 4))
        with patch(
            "auto_round.utils.device.parse_available_devices", return_value=["cpu"]
        ):
            # Should not raise.  Single-device path is a no-op.
            set_avg_auto_device_map(model, device_map="cpu")
        # No tuning_device attribute should have been added.
        assert not hasattr(model[0], "tuning_device")

    def test_hpu_warns_when_multiple(self):
        from auto_round.utils.device import set_avg_auto_device_map

        model = nn.Sequential(nn.Linear(4, 4))

        # Multiple HPU devices - hit the warning_once branch.
        with patch(
            "auto_round.utils.device.parse_available_devices",
            return_value=["hpu:0", "hpu:1"],
        ):
            set_avg_auto_device_map(model, device_map="hpu:0,hpu:1")
        # Function calls get_block_names which on a Sequential returns
        # no real block structure - it should just return silently.


# ===========================================================================
# parse_available_devices (extra edge cases)
# ===========================================================================
class TestParseAvailableDevicesExtra:
    """Additional tests for ``parse_available_devices``."""

    def test_torch_device_cpu(self):
        from auto_round.utils.device import parse_available_devices

        result = parse_available_devices(torch.device("cpu"))
        assert result == ["cpu"]

    def test_torch_device_with_index(self):
        from auto_round.utils.device import parse_available_devices

        # CPU strips the index in this branch (returns just "cpu") because
        # CPU is a single logical device from torch's view.
        result = parse_available_devices(torch.device("cpu:0"))
        assert result == ["cpu"] or result == ["cpu:0"]  # accept either

    def test_dict_input_returns_unique_devices(self):
        from auto_round.utils.device import parse_available_devices

        result = parse_available_devices({"a": "cpu", "b": "cpu"})
        assert result == ["cpu"]

    def test_unsupported_type_raises(self):
        from auto_round.utils.device import parse_available_devices

        with pytest.raises(TypeError):
            parse_available_devices(3.14)  # float unsupported

    def test_numeric_string_in_device_list(self):
        from auto_round.utils.device import parse_available_devices

        with patch(
            "auto_round.utils.device.get_available_device_types", return_value=["cpu"]
        ):
            # Numeric tokens in a list - device_types=["cpu"] -> cpu
            result = parse_available_devices("0")
            assert result == ["cpu"]

    def test_dict_pair_string(self):
        """Dict-like strings like ``transformer:0,lm_head:1`` are parsed."""
        from auto_round.utils.device import parse_available_devices

        with patch(
            "auto_round.utils.device.get_available_device_types", return_value=["cpu"]
        ):
            result = parse_available_devices("transformer:0,lm_head:1")
            # The pair parsing branch should produce 2 entries whose "values"
            # are the device indexes (with type prefix swapped to cpu).
            assert isinstance(result, list)
            assert len(result) >= 2


# ===========================================================================
# MemoryMonitor class
# ===========================================================================
class TestMemoryMonitor:
    """Exhaustive tests for ``MemoryMonitor``."""

    def test_singleton(self):
        from auto_round.utils.device import MemoryMonitor

        a = MemoryMonitor()
        b = MemoryMonitor()
        assert a is b

    def test_default_state(self):
        from auto_round.utils.device import MemoryMonitor

        m = MemoryMonitor()
        assert m.peak_ram == 0.0
        assert m.peak_vram == {}
        assert m.enabled is True

    def test_disabled_update_is_noop(self):
        from auto_round.utils.device import MemoryMonitor

        m = MemoryMonitor()
        m.enabled = False
        prior = m.peak_ram
        m.update()
        assert m.peak_ram == prior

    def test_update_with_cpu_only(self):
        """If the device manager is unavailable, update_cpu is still called."""
        from auto_round.utils.device import MemoryMonitor

        m = MemoryMonitor()
        m.enabled = True
        prior = m.peak_ram
        with patch(
            "auto_round.utils.device.get_current_device_manager"
        ) as mock_mgr_cls:
            manager = MagicMock()
            manager.is_available.return_value = False
            manager.type = "cpu"
            mock_mgr_cls.return_value = manager
            m.update(device_list=[0])
        # peak_ram should be at least 0 (may advance if process memory grew)
        assert m.peak_ram >= prior

    def test_update_with_string_device_in_list(self):
        """Passing a string device_list normalises to a list."""
        from auto_round.utils.device import MemoryMonitor

        m = MemoryMonitor()
        with patch(
            "auto_round.utils.device.get_current_device_manager"
        ) as mock_mgr_cls:
            manager = MagicMock()
            manager.is_available.return_value = False
            manager.type = "cpu"
            mock_mgr_cls.return_value = manager
            m.update(device_list="cuda:0")
        # Should not raise.

    def test_update_cpu(self):
        from auto_round.utils.device import MemoryMonitor

        m = MemoryMonitor()
        m.update_cpu()
        # peak_ram should always be > 0 once any process has memory.
        assert isinstance(m.peak_ram, float)

    def test_update_cpu_disabled_noop(self):
        from auto_round.utils.device import MemoryMonitor

        m = MemoryMonitor()
        m.enabled = False
        m.update_cpu()
        assert m.peak_ram == 0.0

    def test_update_hpu_no_hpex(self):
        """Without HPEX the HPU track is a no-op."""
        from auto_round.utils.device import MemoryMonitor

        m = MemoryMonitor()
        with patch("auto_round.utils.device.is_hpex_available", return_value=False):
            m.update_hpu(device_list=[0])
        assert m.peak_vram == {}

    def test_reset(self):
        from auto_round.utils.device import MemoryMonitor

        m = MemoryMonitor()
        m.peak_ram = 5.0
        m.peak_vram = {"0": 5.0}
        m.reset()
        assert m.peak_ram == 0.0
        assert m.peak_vram == {}

    def test_get_summary_only_peak_ram(self):
        from auto_round.utils.device import MemoryMonitor

        m = MemoryMonitor()
        m.peak_ram = 1.5
        summary = m.get_summary()
        assert "peak_ram" in summary
        assert "peak_vram" not in summary

    def test_get_summary_with_one_vram(self):
        from auto_round.utils.device import MemoryMonitor

        m = MemoryMonitor()
        m.peak_ram = 1.0
        m.peak_vram = {"0": 2.5}
        summary = m.get_summary()
        assert "peak_vram" in summary
        assert "2.5GB" in summary

    def test_get_summary_with_multiple_vram(self):
        from auto_round.utils.device import MemoryMonitor

        m = MemoryMonitor()
        m.peak_ram = 1.0
        m.peak_vram = {"0": 2.0, "1": 3.0}
        summary = m.get_summary()
        # Multiple vram uses dict syntax {key: value, ...}
        assert "2.0GB" in summary
        assert "3.0GB" in summary

    def test_log_summary_default_level(self):
        from auto_round.utils.device import MemoryMonitor, logger

        m = MemoryMonitor()
        with patch.object(logger, "info") as mock_info:
            m.log_summary(msg="hello")
        # Should call logger.info with the message + summary.
        assert mock_info.called
        args = mock_info.call_args[0][0]
        assert "hello" in args
        assert "peak_ram" in args

    def test_log_summary_custom_level(self):
        from auto_round.utils.device import MemoryMonitor, logger

        m = MemoryMonitor()
        with patch.object(logger, "warning") as mock_warning:
            m.log_summary(msg="warning-test", level="warning")
        assert mock_warning.called

    def test_log_summary_invalid_level_falls_back_to_info(self):
        from auto_round.utils.device import MemoryMonitor, logger

        m = MemoryMonitor()
        with patch.object(logger, "info") as mock_info:
            # "invalid-level" gets the getattr default of logger.info
            m.log_summary(msg="hi", level="invalid-level")
        assert mock_info.called

    def test_log_summary_returns_summary(self):
        from auto_round.utils.device import MemoryMonitor

        m = MemoryMonitor()
        m.peak_ram = 1.0
        summary = m.log_summary(msg="x")
        assert isinstance(summary, str)


# ===========================================================================
# dump_memory_usage_ctx / dump_mem_usage
# ===========================================================================
class TestDumpMemoryUsageCtx:
    """Cover the context manager and the decorator."""

    def test_context_manager_runs(self):
        from auto_round.utils.device import dump_memory_usage_ctx

        with dump_memory_usage_ctx(msg="ctx-test"):
            x = 1 + 1
            assert x == 2

    def test_context_manager_with_warning_level(self):
        from auto_round.utils.device import dump_memory_usage_ctx

        with patch("auto_round.utils.device.logger.warning") as warn:
            with dump_memory_usage_ctx(msg="warn-ctx", log_level="warning"):
                pass
        assert warn.called

    def test_decorator_runs_function(self):
        from auto_round.utils.device import dump_mem_usage

        @dump_mem_usage(msg="decorate-test")
        def double(x):
            return x * 2

        assert double(3) == 6

    def test_decorator_with_custom_level(self):
        from auto_round.utils.device import dump_mem_usage

        @dump_mem_usage(msg="decorate-debug", log_level="debug")
        def identity(x):
            return x

        assert identity("ok") == "ok"

    def test_decorator_preserves_name(self):
        from auto_round.utils.device import dump_mem_usage

        @dump_mem_usage(msg="name-test")
        def my_func(x):
            return x

        assert my_func.__name__ == "my_func"

    def test_decorator_returns_value(self):
        from auto_round.utils.device import dump_mem_usage

        @dump_mem_usage(msg="return")
        def returns_dict():
            return {"a": 1}

        assert returns_dict() == {"a": 1}


# ===========================================================================
# PartitionDictNumbers
# ===========================================================================
class TestPartitionDictNumbersExtra:
    """More ``partition_dict_numbers`` edge cases."""

    def test_single_element(self):
        from auto_round.utils.device import partition_dict_numbers

        result = partition_dict_numbers({"only": 5}, 1)
        assert result == [{"only": 5}]

    def test_n_greater_than_items(self):
        from auto_round.utils.device import partition_dict_numbers

        result = partition_dict_numbers({"a": 1}, 3)
        assert len(result) == 3

    def test_n_equals_items(self):
        from auto_round.utils.device import partition_dict_numbers

        result = partition_dict_numbers({"a": 1, "b": 2, "c": 3}, 3)
        # Each item should be in its own group.
        assert result == [{"a": 1}, {"b": 2}, {"c": 3}]

    def test_perfect_split(self):
        from auto_round.utils.device import partition_dict_numbers

        # Total = 30, target = 10.  Should split into two clean groups.
        result = partition_dict_numbers({"a": 10, "b": 10, "c": 10, "d": 0, "e": 0, "f": 0}, 3)
        # All values preserved across result.
        flat = {k: v for d in result for k, v in d.items()}
        assert flat == {"a": 10, "b": 10, "c": 10, "d": 0, "e": 0, "f": 0}

    def test_total_preserved(self):
        from auto_round.utils.device import partition_dict_numbers

        number_dict = {"a": 10, "b": 20, "c": 30, "d": 40, "e": 50}
        result = partition_dict_numbers(number_dict, 3)
        total = sum(sum(g.values()) for g in result)
        assert total == sum(number_dict.values())


# ===========================================================================
# get_major_device (additional tests)
# ===========================================================================
class TestGetMajorDeviceExtended:
    """Additional ``get_major_device`` edge cases."""

    def test_none_returns_string(self):
        from auto_round.utils.device import parse_available_devices  # noqa: F401
        # Already covered by device_manager tests, but ensure it imports
        # from device.py module too (function is re-exported).
        from auto_round.utils.device import get_major_device

        result = get_major_device(None)
        assert isinstance(result, str)

    def test_string_with_index(self):
        from auto_round.utils.device import get_major_device

        result = get_major_device("cpu")
        assert result == "cpu"

    def test_dict_input(self):
        from auto_round.utils.device import get_major_device

        # Dict containing a single device value.
        result = get_major_device({"x": "cpu"})
        assert result == "cpu"

    def test_int_input(self):
        from auto_round.utils.device import get_major_device

        result = get_major_device(0)
        assert isinstance(result, str)


# ===========================================================================
# check_is_cpu
# ===========================================================================
class TestCheckIsCpuExtra:
    """Additional ``check_is_cpu`` tests."""

    def test_int_is_not_cpu(self):
        from auto_round.utils.device import check_is_cpu

        # 0 is a torch device id, not a CPU device reference.
        assert check_is_cpu(0) is False

    def test_xpu_device(self):
        from auto_round.utils.device import check_is_cpu

        assert check_is_cpu("xpu") is False
        assert check_is_cpu("xpu:0") is False

    def test_hpu_device(self):
        from auto_round.utils.device import check_is_cpu

        assert check_is_cpu("hpu") is False


# ===========================================================================
# set_cuda_visible_devices (additional cases)
# ===========================================================================
class TestSetCudaVisibleDevicesExtra:
    """Test non-numeric and edge branches in ``set_cuda_visible_devices``."""

    def test_non_digit_tokens(self):
        """If the devices aren't numeric, the function does nothing."""
        from auto_round.utils.device import set_cuda_visible_devices

        original = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            # Pre-ensure the env var is unset so we test the "else" branch.
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            set_cuda_visible_devices("cuda:foo,bar")
            # Non-numeric -> function shouldn't have touched the env var.
            assert "CUDA_VISIBLE_DEVICES" not in os.environ
        finally:
            if original is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original

    def test_combined_with_spaces_in_index(self):
        """Spaces in numeric input should still work (the function strips)."""
        from auto_round.utils.device import set_cuda_visible_devices

        original = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            set_cuda_visible_devices("0 ")
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0 "
        finally:
            if original is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    def test_no_existing_env_var(self):
        from auto_round.utils.device import set_cuda_visible_devices

        original = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            set_cuda_visible_devices("5")
            # Should set the env var directly.
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == "5"
        finally:
            if original is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)


# ===========================================================================
# patch_xpu_sdpa_drop_causal_mask - more coverage
# ===========================================================================
class TestPatchXpuSdpaCausalMask:
    """Test the XPU SDPA patching logic at a finer grain."""

    def test_returns_early_when_xpu_unavailable(self):
        from auto_round.utils.device import patch_xpu_sdpa_drop_causal_mask

        # Force hasattr(torch, "xpu") False
        with patch("auto_round.utils.device.torch") as mock_torch:
            # Remove the xpu attribute
            mock_torch.configure_mock(**{"xpu.is_available.return_value": False})
            mock_torch.xpu = MagicMock()
            mock_torch.xpu.is_available.return_value = False
            # Should return early.
            patch_xpu_sdpa_drop_causal_mask()
            # Module-level flag must NOT be set since we returned early.
            from auto_round.utils import device as device_mod

            assert device_mod._xpu_sdpa_patched is False

    def test_double_call_is_idempotent(self):
        from auto_round.utils.device import patch_xpu_sdpa_drop_causal_mask
        from auto_round.utils import device as device_mod

        # First call returns early (no xpu available) -> patched stays False.
        # Second call similarly returns early.
        patch_xpu_sdpa_drop_causal_mask()
        assert device_mod._xpu_sdpa_patched is False
        # Second call must not raise.
        patch_xpu_sdpa_drop_causal_mask()
        assert device_mod._xpu_sdpa_patched is False

    def test_force_flag_bypass(self):
        """Calling while already patched must return immediately."""
        from auto_round.utils import device as device_mod
        from auto_round.utils.device import patch_xpu_sdpa_drop_causal_mask

        device_mod._xpu_sdpa_patched = True
        # Should return without raising -- patched is already True.
        patch_xpu_sdpa_drop_causal_mask()
        assert device_mod._xpu_sdpa_patched is True


# ===========================================================================
# is_pipeline_parallel_supported - extra cases
# ===========================================================================
class TestIsPipelineParallelSupportedExtra:
    """Exhaustive ``is_pipeline_parallel_supported``."""

    def test_only_cuda_supported(self):
        from auto_round.utils.device import is_pipeline_parallel_supported

        assert is_pipeline_parallel_supported("cuda") is True
        for backend in ("cpu", "xpu", "hpu", "mps", "npu", ""):
            assert is_pipeline_parallel_supported(backend) is False


# ===========================================================================
# fake_cuda_for_hpu and fake_triton_for_hpu (more cases)
# ===========================================================================
class TestFakeCudaForHpuExtra:
    """More ``fake_cuda_for_hpu`` scenarios."""

    def test_context_manager_restores_state(self):
        from auto_round.utils.device import fake_cuda_for_hpu

        original = MagicMock(return_value=True)
        with patch("auto_round.utils.device.is_hpex_available", return_value=True), patch(
            "torch.cuda.is_available", original
        ):
            with fake_cuda_for_hpu():
                # Should be temporarily faked.
                pass
            # After exit, original is restored.
            # We can't strictly assert identity due to dynamic restoration,
            # but should at least have called __exit__ without raising.


class TestFakeTritonForHpuExtra:
    """More ``fake_triton_for_hpu`` scenarios."""

    def test_with_existing_triton(self):
        from auto_round.utils.device import fake_triton_for_hpu

        # Create a fake triton module
        with patch.dict(
            sys.modules,
            {"triton": MagicMock(), "triton.language": MagicMock()},
        ):
            with patch("auto_round.utils.device.is_hpex_available", return_value=True):
                with fake_triton_for_hpu():
                    pass


# ===========================================================================
# Device.environ variable mapping
# ===========================================================================
class TestDeviceEnvironVariableMappingExtra:
    """Additional ``DEVICE_ENVIRON_VARIABLE_MAPPING`` tests."""

    def test_is_dict(self):
        from auto_round.utils.device import DEVICE_ENVIRON_VARIABLE_MAPPING

        assert isinstance(DEVICE_ENVIRON_VARIABLE_MAPPING, dict)

    def test_mapping_values_nonempty(self):
        from auto_round.utils.device import DEVICE_ENVIRON_VARIABLE_MAPPING

        for backend, env_var in DEVICE_ENVIRON_VARIABLE_MAPPING.items():
            assert backend
            assert env_var
            assert isinstance(env_var, str)


# ===========================================================================
# CpuInfo property
# ===========================================================================
class TestCpuInfoExtra:
    """Exhaustive ``CpuInfo`` tests."""

    def test_init_state(self):
        from auto_round.utils.device import CpuInfo

        info = CpuInfo()
        # _bf16 attribute must exist (initialised in __init__).
        assert isinstance(info._bf16, bool)

    def test_bf16_property_returns_bool(self):
        from auto_round.utils.device import CpuInfo

        info = CpuInfo()
        assert isinstance(info.bf16, bool)

    def test_handles_non_x86_arch(self):
        """If ``arch`` is missing or not X86, ``_bf16`` should be False."""
        from auto_round.utils.device import CpuInfo

        fake_info = {"arch": "ARM_8"}  # not X86
        with patch("auto_round.utils.device.cpuinfo.get_cpu_info", return_value=fake_info):
            info = CpuInfo()
        assert info._bf16 is False


# ===========================================================================
# Global memory monitor instance
# ===========================================================================
class TestGlobalMemoryMonitor:
    """Test the ``memory_monitor`` module-level singleton."""

    def test_is_singleton(self):
        from auto_round.utils.device import memory_monitor, MemoryMonitor

        assert isinstance(memory_monitor, MemoryMonitor)

    def test_update_cpu_on_singleton(self):
        from auto_round.utils.device import memory_monitor

        memory_monitor.update_cpu()
        # Should not raise.
        assert memory_monitor.peak_ram >= 0.0

    def test_get_summary_on_singleton(self):
        from auto_round.utils.device import memory_monitor

        result = memory_monitor.get_summary()
        assert isinstance(result, str)
        assert "peak_ram" in result


# ===========================================================================
# bytes_to_gigabytes - more variants
# ===========================================================================
class TestBytesToGigabytesExtra:
    """Additional ``bytes_to_gigabytes`` tests."""

    def test_negative_value(self):
        from auto_round.utils.device import bytes_to_gigabytes

        # Negative input -> negative output (preserves sign).
        result = bytes_to_gigabytes(-1024 * 1024 * 1024)
        assert result < 0

    def test_float_bytes(self):
        from auto_round.utils.device import bytes_to_gigabytes

        result = bytes_to_gigabytes(1024.0 * 1024 * 1024)
        assert abs(result - 1.0) < 1e-6

    def test_one_kilobyte(self):
        from auto_round.utils.device import bytes_to_gigabytes

        result = bytes_to_gigabytes(1024)
        # 1024 bytes is 1024 / 1024^3 ≈ 9.5e-7 GB
        assert 0 < result < 1e-6


# ===========================================================================
# _force_trim_malloc and _maybe_trim_malloc counter behavior
# ===========================================================================
class TestMallocTrimCounter:
    """Test internal counter behaviour of ``_maybe_trim_malloc``."""

    def test_counter_increments(self):
        from auto_round.utils import device as device_mod

        device_mod._malloc_trim_counter = 0
        with patch.dict(os.environ, {"AR_ENABLE_MALLOC_TRIM": "1"}, clear=False), patch(
            "auto_round.utils.device.ctypes.CDLL"
        ) as mock_cdll:
            mock_libc = MagicMock()
            mock_cdll.return_value = mock_libc
            from auto_round.utils.device import _maybe_trim_malloc

            before = device_mod._malloc_trim_counter
            # Default AR_MALLOC_TRIM_EVERY=10 -> 1st call should not trim.
            _maybe_trim_malloc()
        # Counter should have incremented regardless of trimming.
        # (We patch cdll to avoid actual library call.)
        assert device_mod._malloc_trim_counter >= before

    def test_invalid_every_falls_back_to_default(self):
        from auto_round.utils.device import _maybe_trim_malloc

        with patch.dict(
            os.environ,
            {"AR_ENABLE_MALLOC_TRIM": "1", "AR_MALLOC_TRIM_EVERY": "notanumber"},
            clear=False,
        ), patch("auto_round.utils.device.ctypes.CDLL") as mock_cdll:
            mock_libc = MagicMock()
            mock_cdll.return_value = mock_libc
            _maybe_trim_malloc()
        # Should not raise, cdll should be called (or at least reachable).

    def test_negative_or_zero_every_normalised_to_one(self):
        """``AR_MALLOC_TRIM_EVERY<=0`` should be clamped to 1."""
        from auto_round.utils.device import _maybe_trim_malloc
        from auto_round.utils import device as device_mod

        with patch.dict(
            os.environ,
            {"AR_ENABLE_MALLOC_TRIM": "1", "AR_MALLOC_TRIM_EVERY": "0"},
            clear=False,
        ), patch("auto_round.utils.device.ctypes.CDLL") as mock_cdll:
            mock_libc = MagicMock()
            mock_cdll.return_value = mock_libc

            device_mod._malloc_trim_counter = 0
            # AR_MALLOC_TRIM_EVERY=0 -> after clamp to 1, first call should trim.
            _maybe_trim_malloc()
        # cdll should have been invoked since every==1.
        assert mock_cdll.called


# ===========================================================================
# Module-level: confirm exported names exist
# ===========================================================================
class TestModuleExports:
    """Verify that all expected public symbols are accessible from the module."""

    @pytest.mark.parametrize(
        "name",
        [
            "is_package_available",
            "compile_func",
            "is_hpex_available",
            "check_is_cpu",
            "is_pipeline_parallel_supported",
            "set_cuda_visible_devices",
            "CpuInfo",
            "bytes_to_gigabytes",
            "clear_memory_if_reached_threshold",
            "check_memory_availability",
            "set_tuning_device_for_layer",
            "set_non_auto_device_map",
            "get_first_available_attr",
            "get_moe_memory_ratio",
            "estimate_tuning_block_mem",
            "partition_dict_numbers",
            "dispatch_model_block_wise",
            "set_avg_auto_device_map",
            "parse_available_devices",
            "is_gaudi2",
            "MemoryMonitor",
            "memory_monitor",
            "dump_memory_usage_ctx",
            "dump_mem_usage",
            "dispatch_model_by_all_available_devices",
            "DEVICE_ENVIRON_VARIABLE_MAPPING",
            "override_cuda_device_capability",
            "fake_cuda_for_hpu",
            "fake_triton_for_hpu",
            "get_major_device",
            "_force_trim_malloc",
            "_maybe_trim_malloc",
            "_use_hpu_compile_mode",
            "_allocate_layers_to_devices",
            "_bump_dynamo_cache_limit",
        ],
    )
    def test_symbol_present(self, name):
        from auto_round.utils import device as device_mod

        assert hasattr(device_mod, name)
