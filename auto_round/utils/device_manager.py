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
"""Unified device backend abstraction for AutoRound.

The goal of this module is to make supporting a *new* hardware device as cheap
as possible.  Instead of sprinkling ``torch.cuda.* / torch.xpu.* / torch.hpu.*``
branches across the code base, all device-specific operations are funnelled
through a single :class:`DeviceManager` wrapper that delegates to PyTorch's
unified device APIs:

* ``torch.accelerator`` (PyTorch >= 2.6, expanded in 2.12) -- discovery and
  synchronization, see https://docs.pytorch.org/docs/2.12/accelerator.html
* ``torch.get_device_module(device)`` -- returns the backend runtime module
  (``torch.cuda``, ``torch.xpu``, ``torch.mps`` ...) exposing the *same* method
  names (``empty_cache``, ``synchronize``, ``memory_reserved`` ...).

Because every PyTorch device backend exposes the same method surface, a new
device that PyTorch already supports works here with **zero** extra code.  Only
out-of-tree backends that are not yet integrated into ``torch.accelerator``
(currently Intel Gaudi / ``hpu``) need a tiny shim, handled below.

Typical usage::

    from auto_round.utils.device_manager import get_current_device_manager, get_ar_device

    dev = get_current_device_manager()   # active device (cuda/xpu/hpu/...)
    if dev.is_available():
        dev.empty_cache()
        free, total = dev.mem_get_info(0)

    cuda = get_ar_device("cuda")    # a specific backend
"""

from __future__ import annotations

import contextlib
import functools
import gc
import re
from typing import Optional, Union

import torch

from auto_round.logger import logger

__all__ = [
    "ARDevice",
    "DeviceManager",
    "device_manager",
    "normalize_default_device_map",
    "get_ar_device",
    "get_current_device_manager",
    "get_current_device_type",
    "is_device_available",
    "get_available_device_types",
    "get_major_device",
    "detect_device_count",
    "get_device_and_parallelism",
    "get_packing_device",
    "is_auto_device_mapping",
    "get_device_memory",
    "ClearMemory",
    "clear_memory",
]


# ---------------------------------------------------------------------------
# Backend discovery helpers
# ---------------------------------------------------------------------------
# Priority order used as a *fallback* hint when ``torch.accelerator`` is not
# available (PyTorch < 2.6).  ``hpu`` is kept explicit because Intel Gaudi is an
# out-of-tree backend that historically was not registered with
# ``torch.accelerator``.  Any backend that IS registered with
# ``torch.accelerator`` (cuda/xpu/mps/npu/...) is discovered automatically and
# does NOT need to appear in this list.
_PREFERRED_ORDER = ("cuda", "xpu", "hpu")  # add mps later


def normalize_default_device_map(device_map: Union[None, str, int, torch.device, dict]):
    """Normalize default device selection across entry points.

    On Apple Silicon, the default ``0`` / ``"0"`` / ``None`` / ``"auto"``
    selection would otherwise resolve to MPS.  That tends to OOM on larger
    models, so keep the historical behavior of defaulting to CPU unless the
    caller explicitly requests MPS.
    """
    if torch.mps.is_available() and device_map in (0, "0", None, "auto"):
        logger.warning(
            "MPS detected. Using CPU by default to avoid potential memory issues. "
            "Set --device_map=mps to force MPS usage."
        )
        return "cpu"
    return device_map


def _torch_accelerator_type() -> Optional[str]:
    """Return the canonical accelerator type reported by ``torch.accelerator``.

    A PyTorch build exposes at most one accelerator backend; this returns its
    type string (e.g. ``"cuda"``, ``"xpu"``, ``"mps"``, ``"npu"`` ...) or
    ``None`` when the API is unavailable / no accelerator is present.
    """
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is None:
        return None
    try:
        if not accelerator.is_available():
            return None
        current = accelerator.current_accelerator()
        return current.type if current is not None else None
    except Exception:
        return None


def _accelerator_api():
    """Return the ``torch.accelerator`` module when it is usable, else ``None``.

    "Usable" means the API exists *and* reports an available accelerator for the
    current build (so on CPU-only builds we fall back to the device module).
    """
    api = getattr(torch, "accelerator", None)
    if api is None:
        return None
    try:
        return api if api.is_available() else None
    except Exception:
        return None


def _module_call(api, names, *args):
    """Call the first existing attribute in ``names`` on ``api`` with ``args``.

    Tolerates PyTorch renames across versions (e.g. ``current_device_index`` in
    2.12 vs the deprecated ``current_device_idx`` in 2.6).  Returns
    ``(ok, result)``.
    """
    for name in names:
        fn = getattr(api, name, None)
        if callable(fn):
            return True, fn(*args)
    return False, None


@functools.lru_cache(None)
def _hpu_available() -> bool:
    """Whether the Intel Gaudi (hpu) backend is usable."""
    if hasattr(torch, "hpu") and torch.hpu.is_available():
        return True
    try:  # pragma: no cover - depends on Gaudi runtime
        import habana_frameworks.torch.hpu as _hthpu  # noqa: F401  pylint: disable=E0401

        return True
    except Exception:  # pragma: no cover
        return False


def _normalize_device_type(device: Union[None, str, int, torch.device]) -> Optional[str]:
    """Reduce any device spec to a bare backend type string (``"cuda"`` ...)."""
    if device is None:
        return get_current_device_type()
    if isinstance(device, int):
        return get_current_device_type()
    if isinstance(device, torch.device):
        return device.type
    if isinstance(device, str):
        if device in ("auto", "tp"):
            return get_current_device_type()
        return device.split(":")[0]
    raise ValueError("Device type not recognized")


@functools.lru_cache(None)
def get_current_device_type() -> str:
    """Return the active device backend type, or "cpu" if CPU-only.

    Discovery order:
      1. Intel Gaudi ("hpu") -- out-of-tree, may not register with torch.accelerator.
      2. "torch.accelerator" -- the canonical API, covers cuda/xpu/mps/npu/...
      3. Manual probing of :data:`_PREFERRED_ORDER` for older PyTorch releases.
    """
    # "hpu" first: it may not be registered with torch.accelerator.
    if _hpu_available():
        return "hpu"

    accel_type = _torch_accelerator_type()
    if accel_type is not None:
        return accel_type

    # PyTorch < 2.6: torch.accelerator may not exist; probe common backends.
    for dtype in _PREFERRED_ORDER:
        if dtype == "hpu":
            continue
        mod = getattr(torch, dtype, None)
        is_avail = getattr(mod, "is_available", None)
        if callable(is_avail) and is_avail():
            return dtype

    return "cpu"


def is_device_available() -> bool:
    """Whether any (non-CPU) device is available."""
    return get_current_device_type() is not None


def get_available_device_types() -> list[str]:
    """Return all available (non-CPU) backend types, in preferred order.

    Uses ``torch.accelerator`` so backends registered with PyTorch -- including
    out-of-tree ones such as ``npu`` -- are discovered automatically, without
    callers ever probing ``torch.cuda`` / ``torch.xpu`` / ... by hand.
    """
    available: list[str] = []
    # Out-of-tree hpu first (may not be registered with torch.accelerator).
    if _hpu_available():
        available.append("hpu")
    # The canonical accelerator reported by torch.accelerator (cuda/xpu/mps/npu/...).
    accel_type = _torch_accelerator_type()
    if accel_type is not None and accel_type not in available:
        available.append(accel_type)
    return available


# ---------------------------------------------------------------------------
# Device handles -- a small inheritance hierarchy
# ---------------------------------------------------------------------------
class _DeviceIndexContext:
    """Fallback for ``torch.accelerator.device_index`` on older PyTorch/backends."""

    def __init__(self, device: "ARDevice", index: int):
        self._device = device
        self._index = index
        self._prev = None

    def __enter__(self):
        try:
            self._prev = self._device.current_device()
        except Exception:
            self._prev = None
        self._device.set_device(self._index)
        return self

    def __exit__(self, *exc):
        if self._prev is not None:
            self._device.set_device(self._prev)
        return False


class ARDevice:
    """Base, backend-agnostic handle to a single PyTorch device *backend*.

    A :class:`Device` represents a backend *type* (``cuda``/``xpu``/...), not a
    single card -- every per-card operation takes an ``index`` so the same
    handle drives all cards of that backend (multi-card aware).

    The base implementation delegates to the backend runtime module obtained
    from :func:`get_device_module` and, when this is the build's active
    accelerator, to the unified ``torch.accelerator`` API.  Specialised
    backends subclass this and override only the methods that differ (e.g.
    :meth:`set_device` / :meth:`device_count`); subclasses self-register via the
    ``device_type`` class attribute so :class:`DeviceManager` can instantiate
    them by name.
    """

    #: Canonical backend type a subclass handles (e.g. ``"cuda"``).  Empty on
    #: the base class, which stays usable as a *generic* fallback for any
    #: PyTorch backend that lacks a dedicated subclass (e.g. a fresh ``npu``).
    device_type: str = ""

    _registry: dict[str, type["ARDevice"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        dtype = cls.__dict__.get("device_type", "")
        if dtype:
            ARDevice._registry[dtype] = cls

    @classmethod
    def create(cls, device_type: str) -> "ARDevice":
        """Instantiate the most specific :class:`Device` for ``device_type``."""
        subclass = cls._registry.get(device_type)
        if subclass is not None:
            return subclass()
        return ARDevice(device_type)

    @staticmethod
    def get_device_module(device: Union[None, str, int, torch.device] = None):
        """Return the backend runtime module for ``device`` (e.g. ``torch.cuda``).

        This is a thin, version-tolerant wrapper around ``torch.get_device_module``
        that also understands ``hpu`` and plain device strings/indices.

        Args:
            device: ``"cuda"``, ``"xpu:0"``, ``torch.device(...)``, an int index
                (interpreted against the current device) or ``None`` (current
                device).

        Returns:
            The module exposing the device runtime API, or ``None`` for CPU / when
            no device is available.
        """
        device_type = _normalize_device_type(device)
        if hasattr(torch, "get_device_module"):
            try:
                return torch.get_device_module(device_type)
            except Exception:
                pass
        return getattr(torch, device_type, None)

    def __init__(self, device_type: Optional[str] = None):
        self.type = device_type or self.device_type

        # Prefer the unified ``torch.accelerator`` API for runtime ops when this
        # handle represents the build's current accelerator (cuda/xpu/mps/npu/
        # ...).  Out-of-tree backends such as ``hpu`` are not exposed by
        # ``torch.accelerator`` and transparently fall back to ``self._module``.
        self._module = _accelerator_api() if self.type == _torch_accelerator_type() else None

        if self._module is None:
            self._module = self.get_device_module(self.type)

    # -- discovery ----------------------------------------------------------
    @property
    def module(self):
        """The backend runtime module (``torch.cuda`` ...) or ``None``."""
        return self._module

    def is_available(self) -> bool:
        """Whether this backend type is usable in the current build."""
        return True

    def device_count(self) -> int:
        fn = getattr(self._module, "device_count", None)
        return int(fn()) if callable(fn) else 0  # pylint: disable=E1102

    def current_device(self) -> int:
        ok, idx = _module_call(self._module, ("current_device_index", "current_device_idx", "current_device"))
        if ok:
            try:
                return int(idx)
            except Exception:
                pass
        return 0

    def set_device(self, index: Union[int, str, torch.device]) -> None:
        if self._module is None:
            return
        ok, _ = _module_call(self._module, ("set_device_index", "set_device_idx", "set_device"), index)
        if ok:
            return

    def device(self, index: Union[int, str, torch.device, None] = None) -> torch.device:
        """Build a ``torch.device`` for this backend / card ``index``."""
        if index is None:
            return torch.device(self.type)
        if isinstance(index, torch.device):
            return index
        if isinstance(index, str):
            return torch.device(index if ":" in index else f"{self.type}:{index}")
        return torch.device(f"{self.type}:{int(index)}")

    # def devices(self) -> list[torch.device]:
    #     """Enumerate ``torch.device`` for every card of this backend."""
    #     return [self.device(i) for i in range(self.device_count())]

    # -- runtime ------------------------------------------------------------
    def synchronize(self, index: Union[int, None] = None) -> None:
        if self._module is None:
            return
        fn = getattr(self._module, "synchronize", None)
        if not callable(fn):
            return
        try:
            fn(index) if index is not None else fn()  # pylint: disable=E1102
        except Exception:
            fn()  # pylint: disable=E1102

    def empty_cache(self) -> None:
        # ``torch.accelerator.empty_cache`` is broken on some backends (e.g. MPS
        # triggers a caching-allocator assertion).  Always use the per-device
        # runtime module (``torch.cuda`` / ``torch.mps`` / ...) instead.
        fn = getattr(self.module, "empty_cache", None)
        if callable(fn):
            try:
                fn()  # pylint: disable=E1102 # mps has issues
            except:
                pass

    def device_index(self, index: int):
        """Context manager that sets the current device index for this backend.

        Uses ``torch.accelerator.device_index`` when available; otherwise falls
        back to a tiny save/restore around :meth:`set_device`.
        """
        if self._module is not None:
            ctx = getattr(self._module, "device_index", None)
            if callable(ctx):
                return ctx(index)
        return _DeviceIndexContext(self, index)

    def total_memory(self, index: int = 0) -> int:
        fn = getattr(self._module, "get_memory_info", None)

        return fn(index)[1] if callable(fn) else None  # pylint: disable=E1102

    def memory_reserved(self, index: int = 0) -> int:
        if self._module is None:
            return 0
        fn = getattr(self._module, "memory_reserved", None) or getattr(self._module, "memory_cached", None)
        try:
            return int(fn(index)) if callable(fn) else 0  # pylint: disable=E1102
        except Exception:
            return 0

    def memory_allocated(self, index: int = 0) -> int:
        if self._module is None:
            return 0
        fn = getattr(self._module, "memory_allocated", None)
        try:
            return int(fn(index)) if callable(fn) else 0  # pylint: disable=E1102
        except Exception:
            return 0

    # def mem_get_info(self, index: int = 0) -> tuple[int, int]:
    #     """Return ``(free_bytes, total_bytes)`` for ``index``.
    #
    #     Falls back to ``total - reserved`` when the backend lacks a native
    #     ``mem_get_info`` implementation.
    #     """
    #     module = self.get_device_module(self.type) if self._module is _accelerator_api() else self._module
    #     fn = getattr(module, "get_memory_info", None)
    #
    #     return fn(index) if callable(fn) else (0, 0)  # pylint: disable=E1102

    # -- numeric format / mixed-precision policy ---------------------------
    def supports_bf16(self) -> bool:
        """Whether this backend can execute the ``bfloat16`` data type."""
        return True

    def prefers_bf16(self) -> bool:
        """Whether this backend prefers bf16 as the mixed-precision compute dtype.

        Defaults to ``True`` (bf16 is the preferred tuning dtype); backends that
        would rather honour the model's own non-fp32 dtype can override this.
        """
        return True

    def is_torch_compile_supported(self) -> bool:
        return True

    def compile_func(self, func):
        """Compile ``func`` using this backend's ``torch.compile`` customization.

        Generic compile machinery (the shared dynamo cache-limit bump) lives in
        :func:`auto_round.utils.device._bump_dynamo_cache_limit`; only the
        per-device knobs (whether to compile at all and which backend to use) are
        expressed here, so :func:`auto_round.utils.device.compile_func` stays
        device-agnostic.
        """
        # Lazy import: the helper lives in utils/device.py which imports this module.
        if not self.is_torch_compile_supported():
            return func
        from auto_round.utils.device import _bump_dynamo_cache_limit

        _bump_dynamo_cache_limit()

        return torch.compile(func)


def __repr__(self) -> str:  # pragma: no cover - debug aid
    return f"{type(self).__name__}(type={self.type!r})"


class HpuARDevice(ARDevice):
    """Intel Gaudi (HPU) -- an out-of-tree backend.

    ``hpu`` is not exposed through ``torch.accelerator``, so it always drives
    ``torch.hpu`` directly.  ``set_device`` is overridden to guard against
    builds where the runtime omits it.
    """

    device_type = "hpu"

    @staticmethod
    def get_device_module(device: Union[None, str, int, torch.device] = None):
        """Return the backend runtime module for ``device`` (e.g. ``torch.cuda``).

        This is a thin, version-tolerant wrapper around ``torch.get_device_module``
        that also understands ``hpu`` and plain device strings/indices.

        Args:
            device: ``"cuda"``, ``"xpu:0"``, ``torch.device(...)``, an int index
                (interpreted against the current device) or ``None`` (current
                device).

        Returns:
            The module exposing the device runtime API, or ``None`` for CPU / when
            no device is available.
        """

        if hasattr(torch, "hpu"):
            return torch.hpu
        try:  # pragma: no cover - depends on Gaudi runtime
            import habana_frameworks.torch.hpu as hthpu  # pylint: disable=E0401

            return hthpu
        except Exception:  # pragma: no cover
            return None

    def set_device(self, index: Union[int, str, torch.device]) -> None:
        if self._module is None:
            return
        fn = getattr(self._module, "set_device", None)
        if callable(fn):
            try:
                fn(index)  # pylint: disable=E1102
            except Exception:
                pass

    def is_available(self) -> bool:
        return _hpu_available()

    def is_torch_compile_supported(self) -> bool:
        # HPU only compiles in compile mode (lazy mode keeps the eager function).
        from auto_round.utils.device import _use_hpu_compile_mode

        return _use_hpu_compile_mode()

    def compile_func(self, func):
        if self.is_torch_compile_supported():
            return torch.compile(func, backend="hpu_backend")
        return func

    def memory_allocated(self, index: int = 0) -> int:
        return torch.hpu.memory_allocated(index)

    def memory_reserved(self, index: int = 0) -> int:  # TODO have a check
        return torch.hpu.memory_allocated(index)

    def device_count(self) -> int:
        import habana_frameworks.torch.hpu as hthpu  # pylint: disable=E0401

        return hthpu.device_count()


class MpsARDevice(ARDevice):
    """Apple Silicon (MPS) backend.

    MPS's caching allocator is not yet compatible with ``torch.accelerator``'s
    generic ``empty_cache`` path (PyTorch asserts internally), so we bypass it
    and call ``torch.mps`` methods directly.
    """

    device_type = "mps"

    def __init__(self, device_type: Optional[str] = None):
        # Always use torch.mps directly, never torch.accelerator.
        self.type = "mps"
        self._module = getattr(torch, "mps", None)

    @staticmethod
    def get_device_module(device: Union[None, str, int, torch.device] = None):
        """Return the backend runtime module for ``device`` (e.g. ``torch.cuda``).

        This is a thin, version-tolerant wrapper around ``torch.get_device_module``
        that also understands ``hpu`` and plain device strings/indices.

        Args:
            device: ``"cuda"``, ``"xpu:0"``, ``torch.device(...)``, an int index
                (interpreted against the current device) or ``None`` (current
                device).

        Returns:
            The module exposing the device runtime API, or ``None`` for CPU / when
            no device is available.
        """
        return torch.mps

    def is_available(self) -> bool:
        """Whether this backend type is usable in the current build."""
        return self._module.is_available()

    def current_device(self) -> int:
        return 0

    def set_device(self, index: Union[int, str, torch.device]) -> None:
        return None

    def device(self, index: Union[int, str, torch.device, None] = None) -> torch.device:
        """Build a ``torch.device`` for this backend / card ``index``."""
        return torch.device("mps")

    def device_index(self, index: int):
        """Context manager that sets the current device index for this backend.

        Uses ``torch.accelerator.device_index`` when available; otherwise falls
        back to a tiny save/restore around :meth:`set_device`.
        """
        if self._module is not None:
            ctx = getattr(self._module, "device_index", None)
            if callable(ctx):
                return ctx(index)
        return _DeviceIndexContext(self, index)

    def total_memory(self, index: int = 0) -> int:
        return torch.mps.recommended_max_memory()

    def memory_reserved(self, index: int = 0) -> int:
        return torch.mps.driver_allocated_memory()

    def memory_allocated(self, index: int = 0) -> int:
        return torch.mps.current_allocated_memory()

    # def mem_get_info(self, index: int = 0) -> tuple[int, int]:
    #     """Return ``(free_bytes, total_bytes)`` for ``index``.
    #
    #     Falls back to ``total - reserved`` when the backend lacks a native
    #     ``mem_get_info`` implementation.
    #     """
    #     module = self.get_device_module(self.type) if self._module is _accelerator_api() else self._module
    #     fn = getattr(module, "get_memory_info", None)
    #
    #     return fn(index) if callable(fn) else (0, 0)  # pylint: disable=E1102

    # -- numeric format / mixed-precision policy ---------------------------
    def supports_bf16(self) -> bool:
        """Whether this backend can execute the ``bfloat16`` data type."""
        return True

    def prefers_bf16(self) -> bool:
        """Whether this backend prefers bf16 as the mixed-precision compute dtype.

        Defaults to ``True`` (bf16 is the preferred tuning dtype); backends that
        would rather honour the model's own non-fp32 dtype can override this.
        """
        return True


class CpuARDevice(ARDevice):
    """First-class handle for the host CPU.

    CPU has no backend runtime module, so instead of letting every method fall
    through ``None`` checks we give it explicit, correct semantics:

    * :meth:`synchronize` / :meth:`empty_cache` are genuine no-ops (there is no
      async stream or caching allocator to flush on CPU).
    * memory introspection reports host RAM via ``psutil`` when available.
    """

    device_type = "cpu"

    @staticmethod
    def get_device_module(device: Union[None, str, int, torch.device] = None):
        return None

    # -- discovery ----------------------------------------------------------
    def is_available(self) -> bool:  # CPU is always present.
        return True

    def device_count(self) -> int:  # A single logical device from torch's view.
        return 1

    def current_device(self) -> int:
        return 0

    def set_device(self, index: Union[int, str, torch.device]) -> None:  # no-op
        return None

    def device(self, index: Union[int, str, torch.device, None] = None) -> torch.device:
        return torch.device("cpu")

    # -- runtime ------------------------------------------------------------
    def synchronize(self, index: Union[int, None] = None) -> None:  # no-op
        return None

    def empty_cache(self) -> None:  # no-op: CPU has no caching allocator.
        return gc.collect()

    def get_device_capability(self, index: Union[int, None] = None):
        return None

    def device_index(self, index: int):  # nothing to switch on CPU.
        return contextlib.nullcontext()

    # -- numeric format / mixed-precision policy ---------------------------
    def supports_bf16(self) -> bool:
        cached = getattr(self, "_bf16_supported", None)
        if cached is None:
            # Local import avoids a circular dependency (device.py imports this module).
            from auto_round.utils.device import CpuInfo

            cached = bool(CpuInfo().bf16)
            self._bf16_supported = cached
        return cached

    # -- memory introspection (host RAM) -----------------------------------
    def _virtual_memory(self):
        try:
            import psutil  # pylint: disable=C0415

            return psutil.virtual_memory()
        except Exception:
            return None

    def total_memory(self, index: int = 0) -> int:
        vm = self._virtual_memory()
        return int(vm.total) if vm is not None else 0

    def memory_reserved(self, index: int = 0) -> int:
        import psutil

        process = psutil.Process()
        current_ram = process.memory_info().rss
        return current_ram

    def memory_allocated(self, index: int = 0) -> int:
        return self.memory_reserved(index)

    # def mem_get_info(self, index: int = 0) -> tuple[int, int]:
    #     vm = self._virtual_memory()
    #     if vm is None:
    #         return 0, 0
    #     return int(vm.available), int(vm.total)

    def is_torch_compile_supported(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Device manager -- creates, caches and orchestrates Device handles
# ---------------------------------------------------------------------------
class DeviceManager:
    """Registry and orchestrator for :class:`Device` handles.

    Owns the mapping from backend type to a (cached) :class:`Device` instance,
    exposes the *current* device for the active backend, and enumerates every
    card across all available backends for multi-card scenarios.  Custom
    backends can be plugged in at runtime via :meth:`register` without touching
    this module.

    A manager can additionally be *configured* with a ``device_map`` so callers
    (e.g. the compressors) no longer keep their own ``device`` / ``device_list``
    state -- they ask the manager instead.

    The manager is a process-wide **singleton**: every ``DeviceManager(...)`` call
    returns the same instance.  Passing a ``device_map`` simply (re)configures that
    shared instance, so the active device / device_list is always single-sourced.
    """

    _instance: Optional["DeviceManager"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, device_map: Union[None, str, torch.device, int, dict] = None):
        # Initialise backing state once; later constructions reuse the singleton.
        if not getattr(self, "_initialized", False):
            self._cache: dict[str, ARDevice] = {}
            self._device_map = None
            self._device_list: Optional[list] = None
            self._major_device: Optional[str] = None
            self._initialized = True
        if device_map is not None:
            self.configure(device_map)

    # -- device_map configuration ------------------------------------------
    def configure(self, device_map: Union[None, str, torch.device, int, dict] = 0) -> "DeviceManager":
        """Resolve a ``device_map`` into a concrete device list and major device.

        Centralises the device-map parsing the compressors used to perform by
        hand, so they can rely on :attr:`device` / :attr:`device_list` instead of
        maintaining duplicate state.
        """
        if device_map is None:
            device_map = 0
        if isinstance(device_map, str):
            device_map = device_map.replace(" ", "")
        self._device_map = device_map
        # Lazy import: device.py imports this module, so a top-level import would
        # create a circular dependency.
        from auto_round.utils.device import parse_available_devices

        self._device_list = parse_available_devices(device_map)  # cuda:6
        self._major_device = get_major_device(device_map)  # cuda:4
        return self

    @property
    def device_map(self):
        """The raw ``device_map`` this manager was configured with."""
        return self._device_map

    @property
    def device_list(self) -> list:
        """All concrete devices selected by the configured ``device_map``."""
        if self._device_list is None:
            self.configure(self._device_map)
        return self._device_list

    @property
    def device(self) -> str:
        """The major (primary, non-CPU when possible) device string."""
        if self._major_device is None:
            self.configure(self._device_map)
        return self._major_device

    @device.setter
    def device(self, value: Union[str, torch.device]) -> None:
        """Override the major device (e.g. an OOM fallback to ``"cpu"``)."""
        self._major_device = str(value) if isinstance(value, torch.device) else value

    def is_multi_device(self) -> bool:
        """Whether more than one concrete device is selected."""
        return len(self.device_list) > 1

    # -- registration -------------------------------------------------------
    def register(self, device_cls: type[ARDevice]) -> None:
        """Register a custom :class:`Device` subclass and drop any stale cache."""
        dtype = device_cls.device_type
        if not dtype:
            raise ValueError("Device subclass must define a non-empty 'device_type'")
        ARDevice._registry[dtype] = device_cls
        self._cache.pop(dtype, None)

    # -- lookup -------------------------------------------------------------
    def get_ar_device(self, device_type: Union[None, str, int, torch.device] = None) -> ARDevice:
        """Return the cached :class:`Device` for ``device_type`` (default: current)."""
        normalized = _normalize_device_type(device_type) or "cpu"
        device = self._cache.get(normalized)
        if device is None:
            device = ARDevice.create(normalized)
            self._cache[normalized] = device
        return device

    def current(self) -> ARDevice:
        """Return the :class:`Device` for the active backend (or CPU)."""
        return self.get_ar_device(get_current_device_type())

    def current_type(self) -> str:
        return get_current_device_type()

    # -- multi-card / multi-backend ----------------------------------------
    def available_types(self) -> list[str]:
        """All available (non-CPU) backend types, in preferred order."""
        return get_available_device_types()

    def available_devices(self) -> list[ARDevice]:
        """One :class:`Device` per available (non-CPU) backend type."""
        return [self.get_ar_device(dtype) for dtype in self.available_types()]

    def all_devices(self) -> list[torch.device]:
        """Enumerate every card across all available backends (multi-card)."""
        devices: list[torch.device] = []
        for device in self.available_devices():
            devices.extend(device.devices())
        return devices


# Process-wide singleton manager.
device_manager = DeviceManager()


def get_ar_device(device_type: Union[None, str, int, torch.device] = None) -> ARDevice:
    """Return the cached :class:`Device` handle for a specific backend type."""
    return device_manager.get_ar_device(device_type)


def get_current_device_manager() -> ARDevice:
    """Return the :class:`Device` handle for the active backend (or CPU)."""
    return device_manager.current()


# ---------------------------------------------------------------------------
# Device resolution / parsing helpers (moved from utils/device.py)
# ---------------------------------------------------------------------------
def detect_device_count() -> int:
    """Detects the number of available computation devices."""
    return get_current_device_manager().device_count()


def get_device_and_parallelism(device: Union[str, torch.device, int, dict]) -> tuple[str, bool]:
    """Resolve a device spec into ``(device, parallelism)``.

    The multi-card *parallelism* policy itself is kept as a standalone function
    (:func:`auto_round.utils.device.is_pipeline_parallel_supported`) rather than
    living on the device manager.
    """
    if device is None:
        device = get_major_device(device)
        return device, False
    if isinstance(device, dict):
        unique_devices = set(device.values())
        if len(unique_devices) == 1:
            device = next(iter(unique_devices))
        else:
            device = "auto"
    if isinstance(device, torch.device):
        device = str(device)
    if isinstance(device, str):
        # A bare backend type (e.g. "cuda", "xpu", "hpu", "cpu", "mps") with no index
        if device not in ("auto", "tp") and ":" not in device and "," not in device and not device.isdigit():
            return get_major_device(device), False
        # Strip any "<type>:" prefixes (e.g. "cuda:0,1" -> "0,1") to obtain bare indices.
        device = re.sub(r"[a-zA-Z_]+:", "", device)
        devices = device.replace(" ", "").split(",")
    elif isinstance(device, int):
        devices = [str(device)]
    else:
        devices = [device]

    is_multi_card = all(s.isdigit() for s in devices) and len(devices) > 1
    if is_multi_card:
        # Pick the active backend generically rather than probing each one by hand.
        device_type = get_current_device_type() or "cpu"
        # Parallelism policy is intentionally not part of the device manager.
        from auto_round.utils.device import is_pipeline_parallel_supported

        return device_type, is_pipeline_parallel_supported(device_type)
    elif device == "auto":
        device = get_major_device(device)
        parallelism = True
    else:
        device = get_major_device(device)
        parallelism = False
    return device, parallelism


def get_packing_device(device: Union[str, torch.device, None] = "auto") -> torch.device:
    """Selects the packing device.

    - ``"auto"``: choose best available (active accelerator > CPU).
    - ``str``: parsed by ``torch.device`` (e.g., ``"cuda:2"``, ``"cpu"``).
    - ``torch.device``: returned as-is.
    - ``None``: treated as ``"auto"``.
    """
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        device_type = get_current_device_type()
        if device_type is not None and device_type != "cpu":
            return torch.device(f"{device_type}:0")
        return torch.device("cpu")

    if isinstance(device, torch.device):
        return device

    if isinstance(device, str):
        try:
            return torch.device(device)
        except Exception as e:
            raise ValueError(f"Invalid device string: {device}") from e

    raise TypeError(f"Unsupported device type: {type(device)} ({device})")


def is_auto_device_mapping(device_map: Union[str, int, dict, None]) -> bool:
    if device_map is None or isinstance(device_map, int):
        return False
    elif device_map == "auto":
        return True
    elif isinstance(device_map, str) and "," in device_map:
        return True
    elif isinstance(device_map, dict):
        return False
    else:
        return False


def get_major_device(device_map: Union[None, str, torch.device, int, dict] = None) -> str:
    if device_map is None or isinstance(device_map, (str, torch.device, int)):
        """Detects the appropriate computation device.

        Takes a specific device index/string or ``"auto"``/``None`` (auto-detect the
        active backend), and returns the resolved device as a string.
        "4,6"->cuda:4
        """

        def is_valid_digit(s):
            try:
                num = int(s)
                return 0 <= num
            except Exception:
                return False

        dev_idx = None
        device = device_map
        if is_valid_digit(device):
            dev_idx = int(device)
            device = "auto"
        if isinstance(device, str) and "," in device:  # device is "0,1,2"
            device_list = []
            for dev in device.split(","):
                if dev.isdigit():
                    device_list.append(int(dev))
                elif dev.split(":")[-1].isdigit():
                    device_list.append(int(dev.split(":")[-1]))
                elif 0 not in device_list:
                    device_list.append(0)
            dev_idx = device_list[0] if device_list else None
            device = "auto"
        if device is None or device == "auto":
            device_type = get_current_device_type()
            device = torch.device(device_type) if device_type is not None else torch.device("cpu")
            if dev_idx is not None and str(device) != "cpu":
                device = str(device) + f":{dev_idx}"
            return str(device)
        elif isinstance(device, torch.device):
            device = str(device)
        elif isinstance(device, str):  ## for cuda:0
            if device == "tp":  # pragma: no cover
                # should not specify card, e.g., cuda:0
                device = get_current_device_type() or "cpu"
            else:
                device = device
        return device

    if isinstance(device_map, dict) and device_map:
        tmp_devices = []
        for val in device_map.values():
            if isinstance(val, (str, torch.device, int)):  # could optimize
                tmp_device = get_major_device(val)
                tmp_device = tmp_device.split(":")[0]
                tmp_devices.append(tmp_device)
        tmp_devices = list(set(tmp_devices))
        device = None
        for tmp_device in tmp_devices:
            if tmp_device != "cpu":
                device = tmp_device
                break
        if device is None:
            device = tmp_devices[0]
        if len(tmp_devices) > 1:
            logger.warning_once(
                f"there are multiple device types in the device_map, "
                f"please make sure they are correct,use the first none-cpu device {device} as the core device "
            )

        return device
    logger.warning_once(f"device_map should be [str, torch.device, int, dict], but got {type(device_map)}")
    return "cpu"


def get_device_memory(i: int = 0) -> int:
    """Gets the total memory on the specified device, in gigabytes."""
    dev_mgr = get_current_device_manager()
    if not dev_mgr.is_available() or dev_mgr.type == "cpu":
        raise RuntimeError("No supported device found (CUDA/XPU/HPU/...).")
    return dev_mgr.total_memory(i) / 1024 / 1024 / 1024


def _clear_memory_for_cpu_and_cuda(
    tensor: Union[torch.Tensor, list, None] = None,
    device_list: Union[tuple, list, str, torch.device, None] = None,
):
    # ------------------------
    # Clear CPU-side references
    # ------------------------
    if isinstance(tensor, list):
        for i in range(len(tensor)):
            tensor[i] = None
    tensor = None
    gc.collect()

    # Lazy import: malloc-trim helpers live in utils/device.py.
    from auto_round.utils.device import _maybe_trim_malloc

    _maybe_trim_malloc()

    # ------------------------
    # Normalize device_list
    # ------------------------
    if isinstance(device_list, (str, torch.device)):
        device_list = [device_list]

    # -----------------------------------
    # Device-specific clearing
    # -----------------------------------
    # Group requested devices by backend type so we synchronize the exact
    # indices the caller asked for, then fall back to clearing the active
    # accelerator entirely when no list is provided.
    current_dev_type = get_current_device_type()
    if current_dev_type is None or current_dev_type == "cpu":
        return

    if not device_list:
        dev_mgr = get_current_device_manager()
        dev_mgr.synchronize()
        dev_mgr.empty_cache()
        return

    # Parse "<type>:<idx>" entries, grouping indices per backend.
    per_backend: dict[str, list[int]] = {}
    for dev in device_list:
        dev = str(dev)
        dev_type = dev.split(":")[0]
        if not dev_type or dev_type == "cpu" or dev_type.isdigit():
            # Bare indices (e.g. "0") are interpreted against the active device.
            dev_type = current_dev_type if dev_type.isdigit() else dev_type
            if not dev_type or dev_type == "cpu":
                continue
        devid = int(dev.split(":")[-1]) if ":" in dev else (int(dev) if dev.isdigit() else 0)
        per_backend.setdefault(dev_type, []).append(devid)

    for dev_type, ids in per_backend.items():
        dev_mgr = get_ar_device(dev_type)
        for devid in ids:
            dev_mgr.synchronize(devid)
        dev_mgr.empty_cache()


class ClearMemory:

    def __init__(self, device_list: Union[list, tuple, None] = None):
        self.device_list = device_list

    def __call__(
        self,
        tensor: Union[torch.Tensor, None, list] = None,
        device_list: Union[list, tuple, None] = None,
    ):
        # Lazy imports: these symbols live in utils/device.py.
        from auto_round.utils.device import _force_trim_malloc, is_hpex_available, memory_monitor

        if is_hpex_available():
            # Clear CPU-side references so Python can reclaim them.
            if isinstance(tensor, list):
                for i in range(len(tensor)):
                    tensor[i] = None
            tensor = None
            gc.collect()
            _force_trim_malloc()
            memory_monitor.update_hpu(device_list)
            return
        else:
            if device_list is not None:
                self.device_list = device_list
            final_device_list = self.device_list
            memory_monitor.update(final_device_list)
            _clear_memory_for_cpu_and_cuda(tensor, final_device_list)


clear_memory = torch._dynamo.disable()(ClearMemory(device_list=[0]))
