"""Smoke test for the device-backend refactor (round 2)."""

from auto_round.utils.device import (
    detect_device,
    detect_device_count,
    out_of_vram,
    compile_func,
    is_hpu_lazy_mode,
    is_hpex_available,
    is_gaudi2,
    parse_available_devices,
    clear_memory,
    _clear_memory_for_cpu_and_cuda,
    DEVICE_ENVIRON_VARIABLE_MAPPING,
    get_device_and_parallelism,
    get_packing_device,
    check_memory_availability,
    memory_monitor,
)
from auto_round.utils.device_backend import (
    DeviceBackend,
    register_device_backend,
    get_device_backend,
    iter_registered_backends,
    is_accelerator_device,
    is_accelerator_type,
    get_known_device_types,
    strip_device_prefix,
    split_device_spec,
    auto_select_device,
)
import torch


def hr(s):
    print()
    print("==", s)


hr("Backends")
for b in iter_registered_backends():
    print(f"  {b.name:6s} avail={b.is_available()} prio={b.priority} parallel={b.supports_parallel}")

hr("get_known_device_types")
print("all     :", get_known_device_types())
print("non-cpu :", get_known_device_types(False))

hr("strip_device_prefix")
print(strip_device_prefix("cuda:0,cuda:1,xpu:2"))
print(strip_device_prefix("0,1"))
print(strip_device_prefix("npu:0,cuda:1"))  # npu unknown → kept

hr("split_device_spec")
print(split_device_spec(None))
print(split_device_spec(3))
print(split_device_spec(torch.device("cuda:1")))
print(split_device_spec("0, 1, 2"))

hr("get_device_and_parallelism")
for spec in [None, "cuda", "hpu", "0,1", "cuda:0,cuda:1", 0, "auto", {0: "cuda", 1: "cuda"}, {0: "cuda", 1: "cpu"}]:
    print(f"  {spec!r:35s} -> {get_device_and_parallelism(spec)}")

hr("get_packing_device")
print(get_packing_device("auto"))
print(get_packing_device(None))
print(get_packing_device("cpu"))

hr("check_memory_availability")
w = torch.zeros(8, 8)
i = torch.zeros(1, 8)
print(check_memory_availability("cpu", i, w, 1024, 4))

hr("MemoryMonitor.update via various inputs (CPU only host)")
memory_monitor.reset()
memory_monitor.update(["cpu", "cuda:0", "hpu:0"])
print(memory_monitor.get_summary())

hr("Demo: register an NPU-like backend in <30 lines")


@register_device_backend
class FakeNPUBackend(DeviceBackend):
    name = "fake_npu"
    aliases = ("fnpu",)
    priority = 70
    supports_parallel = True
    visible_devices_env = "FAKE_NPU_VISIBLE_DEVICES"
    oom_signatures = ("FAKE_NPU OOM",)

    def is_available(self) -> bool:
        return False  # demo only

    def device_count(self) -> int:
        return 0


print("by name :", get_device_backend("fake_npu"))
print("by alias:", get_device_backend("fnpu"))
print("env map :", DEVICE_ENVIRON_VARIABLE_MAPPING)

# Re-export sanity
from auto_round.utils.device_backend import get_visible_devices_env_mapping

print("env map (live)  :", get_visible_devices_env_mapping())

hr("OOM detection")
for msg in [
    "CUDA out of memory",
    "MODULE:PT_DEVMEM",
    "UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY",
    "HIP out of memory. Tried to allocate",
    "FAKE_NPU OOM during compute",
    "completely unrelated message",
]:
    print(f"  {msg!r:55s} -> {out_of_vram(msg)}")

hr("clear_memory paths")
clear_memory(device_list=["cpu"])
clear_memory()
_clear_memory_for_cpu_and_cuda(device_list=None)
_clear_memory_for_cpu_and_cuda(device_list=["cuda:0", "hpu:0", "cpu"])
print("OK")

