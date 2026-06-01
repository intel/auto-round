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

    from auto_round.utils.device_manager import get_current_device_manager, get_device_manager

    dev = get_current_device_manager()   # active device (cuda/xpu/hpu/...)
    if dev.is_available():
        dev.empty_cache()
        free, total = dev.mem_get_info(0)

    cuda = get_device_manager("cuda")    # a specific backend
"""

from __future__ import annotations

import functools
from typing import Optional, Union

import torch

__all__ = [
    "DeviceManager",
    "get_device_module",
    "get_device_manager",
    "get_current_device_manager",
    "get_current_device_type",
    "is_device_available",
    "get_available_device_types",
    "is_supported_device",
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
_PREFERRED_ORDER = ("cuda", "hpu", "xpu", "mps")


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


def _accel_call(api, names, *args):
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


def _backend_is_available(name: str) -> bool:
    """Whether a given in-tree backend type (``"cuda"``/``"xpu"``/``"mps"`` ...) is usable."""
    if name == "hpu":
        return _hpu_available()
    # MPS exposes availability under ``torch.backends.mps`` rather than ``torch.mps``.
    if name == "mps":
        backends_mps = getattr(getattr(torch, "backends", None), "mps", None)
        if backends_mps is not None and getattr(backends_mps, "is_available", lambda: False)():
            return True
    backend = getattr(torch, name, None)
    return backend is not None and bool(getattr(backend, "is_available", lambda: False)())


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
    if device_type is None or device_type == "cpu":
        return None
    if device_type == "hpu":
        if hasattr(torch, "hpu"):
            return torch.hpu
        try:  # pragma: no cover - depends on Gaudi runtime
            import habana_frameworks.torch.hpu as hthpu  # pylint: disable=E0401

            return hthpu
        except Exception:  # pragma: no cover
            return None
    # Prefer the official unified accessor when present.
    if hasattr(torch, "get_device_module"):
        try:
            return torch.get_device_module(device_type)
        except Exception:
            pass
    return getattr(torch, device_type, None)


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
    return None


@functools.lru_cache(None)
def get_current_device_type() -> Optional[str]:
    """Return the active device backend type, or ``None`` if CPU-only.

    Discovery order:
      1. Intel Gaudi (``hpu``) -- out-of-tree, may not register with torch.accelerator.
      2. ``torch.accelerator`` -- the canonical API, covers cuda/xpu/mps/npu/...
      3. Manual probing of :data:`_PREFERRED_ORDER` for older PyTorch releases.
    """
    # ``hpu`` first: it may not be registered with torch.accelerator.
    if _hpu_available():
        return "hpu"
    accel_type = _torch_accelerator_type()
    if accel_type is not None:
        return accel_type
    for name in _PREFERRED_ORDER:
        if _backend_is_available(name):
            return name
    return None


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
    # Fallback probing for older PyTorch without torch.accelerator.
    for name in _PREFERRED_ORDER:
        if name not in available and _backend_is_available(name):
            available.append(name)
    return available


def is_supported_device(device: Union[None, str, int, torch.device]) -> bool:
    """Whether ``device`` refers to CPU or a usable accelerator backend.

    Accepts any device spec the user might pass (``"cuda"``, ``"npu:0"``,
    ``torch.device(...)``, an int index, ``"auto"`` ...).  A non-CPU backend is
    considered supported when PyTorch exposes a runtime module for it (e.g.
    ``torch.npu`` provided by ``torch_npu``) or it is otherwise available -- so
    new accelerators work without editing a hardcoded allow-list.
    """
    if device is None or isinstance(device, int):
        return True
    if isinstance(device, torch.device):
        device_type = device.type
    else:
        device_type = str(device).split(":")[0]
    if device_type in ("cpu", "meta", "disk", "auto", "tp", ""):
        return True
    if _backend_is_available(device_type):
        return True
    # A backend torch can build a runtime module for (e.g. torch_npu's "npu").
    return get_device_module(device_type) is not None


# ---------------------------------------------------------------------------
# Device wrapper
# ---------------------------------------------------------------------------
class _DeviceIndexContext:
    """Fallback for ``torch.accelerator.device_index`` on older PyTorch/backends."""

    def __init__(self, manager: "DeviceManager", index: int):
        self._manager = manager
        self._index = index
        self._prev = None

    def __enter__(self):
        try:
            self._prev = self._manager.current_device()
        except Exception:
            self._prev = None
        self._manager.set_device(self._index)
        return self

    def __exit__(self, *exc):
        if self._prev is not None:
            self._manager.set_device(self._prev)
        return False


class DeviceManager:
    """Unified, backend-agnostic handle to a PyTorch device.

    All methods delegate to the underlying device runtime module obtained from
    :func:`get_device_module`, so the same code path works for CUDA, XPU, HPU,
    MPS and any future in-tree backend.  Methods degrade gracefully (no-op /
    sensible default) when the operation is unsupported by a given backend.
    """

    def __init__(self, device_type: str):
        self.type = device_type
        self._module = get_device_module(device_type)
        # Prefer the unified ``torch.accelerator`` API for runtime ops when this
        # manager represents the build's current accelerator (cuda/xpu/mps/npu/
        # ...).  Out-of-tree backends such as ``hpu`` are not exposed by
        # ``torch.accelerator`` and transparently fall back to ``self._module``.
        self._accel = _accelerator_api() if device_type == _torch_accelerator_type() else None

    # -- discovery ----------------------------------------------------------
    @property
    def module(self):
        """The backend runtime module (``torch.cuda`` ...) or ``None``."""
        return self._module

    def is_available(self) -> bool:
        if self._accel is not None:
            try:
                return bool(self._accel.is_available())
            except Exception:
                pass
        if self._module is None:
            return self.type == "cpu"
        fn = getattr(self._module, "is_available", None)
        return bool(fn()) if callable(fn) else True

    def device_count(self) -> int:
        if self._accel is not None:
            try:
                return int(self._accel.device_count())
            except Exception:
                pass
        if self._module is None:
            return 0
        fn = getattr(self._module, "device_count", None)
        try:
            return int(fn()) if callable(fn) else 0
        except Exception:
            return 0

    def current_device(self) -> int:
        if self._accel is not None:
            ok, idx = _accel_call(self._accel, ("current_device_index", "current_device_idx"))
            if ok:
                try:
                    return int(idx)
                except Exception:
                    pass
        if self._module is None:
            return 0
        fn = getattr(self._module, "current_device", None)
        try:
            return int(fn()) if callable(fn) else 0
        except Exception:
            return 0

    def set_device(self, index: Union[int, str, torch.device]) -> None:
        if self._accel is not None:
            ok, _ = _accel_call(self._accel, ("set_device_index", "set_device_idx"), index)
            if ok:
                return
        if self._module is None:
            return
        fn = getattr(self._module, "set_device", None)
        if callable(fn):
            fn(index)

    def device(self, index: Union[int, str, torch.device, None] = None) -> torch.device:
        """Build a ``torch.device`` for this backend."""
        if index is None:
            return torch.device(self.type)
        if isinstance(index, torch.device):
            return index
        if isinstance(index, str):
            return torch.device(index if ":" in index else f"{self.type}:{index}")
        return torch.device(f"{self.type}:{int(index)}")

    # -- runtime ------------------------------------------------------------
    def synchronize(self, index: Union[int, None] = None) -> None:
        if self._accel is not None:
            try:
                self._accel.synchronize(index) if index is not None else self._accel.synchronize()
                return
            except Exception:
                pass
        if self._module is None:
            return
        fn = getattr(self._module, "synchronize", None)
        if not callable(fn):
            return
        try:
            fn(index) if index is not None else fn()
        except Exception:
            fn()

    def empty_cache(self) -> None:
        # ``torch.accelerator`` has no cache API; this is always module-level.
        if self._module is None:
            return
        fn = getattr(self._module, "empty_cache", None)
        if callable(fn):
            fn()

    def get_device_capability(self, index: Union[int, None] = None):
        """Return the compute capability of the selected device, if exposed."""
        if self._accel is not None:
            ok, cap = _accel_call(self._accel, ("get_device_capability",), index)
            if ok:
                return cap
        if self._module is None:
            return None
        fn = getattr(self._module, "get_device_capability", None)
        if not callable(fn):
            return None
        try:
            return fn(index) if index is not None else fn()
        except Exception:
            return None

    def device_index(self, index: int):
        """Context manager that sets the current device index for this backend.

        Uses ``torch.accelerator.device_index`` when available; otherwise falls
        back to a tiny save/restore around :meth:`set_device`.
        """
        if self._accel is not None:
            ctx = getattr(self._accel, "device_index", None)
            if callable(ctx):
                return ctx(index)
        return _DeviceIndexContext(self, index)

    # -- memory introspection ----------------------------------------------
    def device_properties(self, index: int = 0):
        if self._module is None:
            return None
        fn = getattr(self._module, "get_device_properties", None)
        return fn(index) if callable(fn) else None

    def total_memory(self, index: int = 0) -> int:
        props = self.device_properties(index)
        return int(getattr(props, "total_memory", 0)) if props is not None else 0

    def memory_reserved(self, index: int = 0) -> int:
        if self._module is None:
            return 0
        fn = getattr(self._module, "memory_reserved", None) or getattr(self._module, "memory_cached", None)
        try:
            return int(fn(index)) if callable(fn) else 0
        except Exception:
            return 0

    def memory_allocated(self, index: int = 0) -> int:
        if self._module is None:
            return 0
        fn = getattr(self._module, "memory_allocated", None)
        try:
            return int(fn(index)) if callable(fn) else 0
        except Exception:
            return 0

    def mem_get_info(self, index: int = 0) -> tuple[int, int]:
        """Return ``(free_bytes, total_bytes)`` for ``index``.

        Falls back to ``total - reserved`` when the backend lacks a native
        ``mem_get_info`` implementation.
        """
        if self._module is None:
            return 0, 0
        fn = getattr(self._module, "mem_get_info", None)
        if callable(fn):
            try:
                free, total = fn(index)
                return int(free), int(total)
            except Exception:
                pass
        total = self.total_memory(index)
        return max(total - self.memory_reserved(index), 0), total

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"DeviceManager(type={self.type!r}, available={self.is_available()})"


_CPU_DEVICE_MANAGER = DeviceManager("cpu")


@functools.lru_cache(None)
def get_device_manager(device_type: str) -> DeviceManager:
    """Return a cached :class:`DeviceManager` for a specific backend type."""
    return DeviceManager(_normalize_device_type(device_type) or "cpu")


def get_current_device_manager() -> DeviceManager:
    """Return the :class:`DeviceManager` for the active backend (or CPU)."""
    device_type = get_current_device_type()
    if device_type is None:
        return _CPU_DEVICE_MANAGER
    return get_device_manager(device_type)
