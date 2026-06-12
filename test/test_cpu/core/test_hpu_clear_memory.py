import importlib
from unittest.mock import patch

from auto_round.utils.device_manager import ClearMemory

dm_module = importlib.import_module("auto_round.utils.device_manager")


def test_clear_memory_hpu_path_trims_and_updates_monitor(monkeypatch):
    events = []
    tensors = [object(), object()]

    monkeypatch.setattr(dm_module.gc, "collect", lambda: events.append("gc"))

    def fake_force_trim():
        events.append("trim")

    def fake_update_hpu(device_list):
        events.append(("update_hpu", device_list))

    with patch("auto_round.utils.device.is_hpex_available", return_value=True), patch(
        "auto_round.utils.device._force_trim_malloc", fake_force_trim
    ), patch("auto_round.utils.device.memory_monitor.update_hpu", fake_update_hpu):
        ClearMemory(device_list=["hpu:0"])(tensors, device_list=["hpu:0"])

    assert tensors == [None, None]
    assert events == ["gc", "trim", ("update_hpu", ["hpu:0"])]
