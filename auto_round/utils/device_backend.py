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
"""Pluggable device backend abstraction.

This module centralises *all* device-specific logic behind a single base
class :class:`DeviceBackend`.  Adding support for a new accelerator
(e.g. ``npu``) only requires:

1. Subclassing :class:`DeviceBackend`.
2. Implementing the few abstract / overridable hooks
   (``is_available``, ``device_count``, ``synchronize``, ``empty_cache``,
   ``memory_allocated``, ``memory_reserved``, ``total_memory`` and -- if
   needed -- the optional hooks ``compile_func`` /
   ``extra_clear_memory`` / ``oom_signatures``).
3. Registering it via the :func:`register_device_backend` decorator.

Every other call-site in the project (``detect_device_count``,
``get_device_memory``, ``out_of_vram``, ``compile_func``,
``clear_memory``, ``parse_available_devices`` ...) dispatches through
this registry, so no other file needs to be edited when a new device
type is added.

See ``docs/adding_new_device.md`` for a concrete worked example.
"""
from __future__ import annotations

import functools
import os
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Union

import torch


__all__ = [
    "DeviceBackend",
    "CPUBackend",
    "CUDABackend",
    "XPUBackend",
    "HPUBackend",
    "register_device_backend",
    "get_device_backend",
    "iter_registered_backends",
    "iter_active_backends",
    "resolve_device_type",
    "auto_select_device",
    "current_accelerator_type",
    "get_visible_devices_env_mapping",
    "is_hpu_lazy_mode",
    "is_accelerator_device",
    "is_accelerator_type",
    "get_known_device_types",
    "strip_device_prefix",
    "split_device_spec",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _device_type_of(device: Union[None, str, int, torch.device]) -> Optional[str]:
    """Extract the textual device type (``cuda``/``hpu``/...) from any input.

    Returns ``None`` if it cannot be determined (e.g. plain ``int``).
    """
    if device is None:
        return None
    if isinstance(device, torch.device):
        return device.type
    if isinstance(device, int):
        return None  # caller decides via ``auto_select_device``
    if isinstance(device, str):
        s = device.strip().lower()
        if not s:
            return None
        # "cuda:0" / "xpu" / "hpu:1" / "cpu"
        return s.split(":", 1)[0]
    return None


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class DeviceBackend(ABC):
    """Abstract base for a torch device backend.

    Subclasses MUST set :attr:`name` and implement
    :meth:`is_available` / :meth:`device_count`.

    All other methods have sensible defaults written in terms of
    :attr:`torch_module` (``torch.cuda`` / ``torch.xpu`` / ...) so most
    new backends only need to override two or three things.
    """

    # ----- Class-level metadata --------------------------------------------------
    name: str = ""
    """Lower-case device type, e.g. ``"cuda"``, ``"hpu"``, ``"npu"``."""

    aliases: tuple[str, ...] = ()
    """Optional alternative spellings for :func:`get_device_backend` lookup."""

    visible_devices_env: Optional[str] = None
    """Environment variable controlling device visibility, if any."""

    oom_signatures: tuple[str, ...] = ()
    """Substrings found in OOM exception messages for this device."""

    priority: int = 0
    """Higher = preferred when auto-selecting a device.  CPU stays at 0."""

    supports_parallel: bool = False
    """Whether the backend supports data-/tensor-parallel multi-device runs.

    Used by :func:`auto_round.utils.device.get_device_and_parallelism` to
    decide whether seeing several device indices implies parallelism.
    Today only CUDA has this enabled; new backends opt-in by overriding.
    """

    # ----- torch handle ----------------------------------------------------------
    @property
    def torch_module(self):
        """Return the torch sub-module for this device.

        Prefers :func:`torch.get_device_module` (PyTorch ≥ 2.5), which is the
        unified accessor recommended by upstream and is the only way some
        out-of-tree devices expose themselves.  Falls back to
        ``getattr(torch, name)`` for older torch versions.
        """
        get_mod = getattr(torch, "get_device_module", None)
        if callable(get_mod):
            try:
                return get_mod(self.name)
            except Exception:
                pass
        return getattr(torch, self.name, None)

    # ----- Required hooks --------------------------------------------------------
    @abstractmethod
    def is_available(self) -> bool:  # pragma: no cover - abstract
        """Return ``True`` if at least one device of this type is usable."""

    def device_count(self) -> int:
        if not self.is_available():
            return 0
        mod = self.torch_module
        if mod is not None and hasattr(mod, "device_count"):
            try:
                return int(mod.device_count())
            except Exception:  # pragma: no cover - defensive
                return 0
        return 0

    # ----- Optional hooks (defaults work for cuda/xpu) --------------------------
    def synchronize(self, index: Union[int, None] = None) -> None:
        mod = self.torch_module
        if mod is None or not hasattr(mod, "synchronize"):
            return
        try:
            if index is None:
                mod.synchronize()
            else:
                mod.synchronize(index)
        except Exception:  # pragma: no cover - defensive
            pass

    def empty_cache(self) -> None:
        mod = self.torch_module
        if mod is not None and hasattr(mod, "empty_cache"):
            try:
                mod.empty_cache()
            except Exception:  # pragma: no cover
                pass

    def current_device(self) -> int:
        mod = self.torch_module
        if mod is not None and hasattr(mod, "current_device"):
            try:
                return int(mod.current_device())
            except Exception:  # pragma: no cover
                return 0
        return 0

    def memory_allocated(self, index: int = 0) -> int:
        mod = self.torch_module
        if mod is not None and hasattr(mod, "memory_allocated"):
            try:
                return int(mod.memory_allocated(index))
            except Exception:  # pragma: no cover
                return 0
        return 0

    def memory_reserved(self, index: int = 0) -> int:
        mod = self.torch_module
        if mod is None:
            return 0
        # HPU exposes ``memory_cached`` instead of ``memory_reserved``
        for fn_name in ("memory_reserved", "memory_cached"):
            fn = getattr(mod, fn_name, None)
            if fn is not None:
                try:
                    return int(fn(index))
                except Exception:  # pragma: no cover
                    return 0
        return 0

    def total_memory(self, index: int = 0) -> int:
        """Total physical memory of the device in bytes (best-effort)."""
        mod = self.torch_module
        if mod is None or not hasattr(mod, "get_device_properties"):
            return 0
        try:
            return int(mod.get_device_properties(index).total_memory)
        except Exception:  # pragma: no cover
            return 0

    # ---- Compile / kernel hooks ------------------------------------------------
    def compile_func(self, func):
        """Return a (possibly compiled) version of ``func``."""
        try:
            return torch.compile(func)
        except Exception:  # pragma: no cover - defensive
            return func

    def extra_clear_memory(self, device_list=None) -> bool:
        """Custom cleanup hook.

        Return ``True`` to indicate the backend has fully handled the
        clear-memory request and the generic CUDA/XPU path should be
        skipped.  Default: no-op, return ``False``.
        """
        return False

    # ----- OOM helper -----------------------------------------------------------
    def matches_oom(self, message: str) -> bool:
        return any(sig in message for sig in self.oom_signatures)

    # ----- Repr -----------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        return f"<DeviceBackend name={self.name!r} available={self.is_available()}>"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_BACKENDS: dict[str, DeviceBackend] = {}


def register_device_backend(cls=None, *, name: Optional[str] = None):
    """Class decorator that instantiates and registers a backend.

    Usage::

        @register_device_backend
        class NPUBackend(DeviceBackend):
            name = "npu"
            ...
    """

    def _do_register(klass):
        if not issubclass(klass, DeviceBackend):
            raise TypeError(f"{klass!r} is not a DeviceBackend subclass")
        instance = klass()
        key = (name or instance.name).lower()
        if not key:
            raise ValueError(f"DeviceBackend {klass!r} must define a non-empty 'name'")
        _BACKENDS[key] = instance
        for alias in instance.aliases:
            _BACKENDS[alias.lower()] = instance
        return klass

    if cls is None:
        return _do_register
    return _do_register(cls)


def iter_registered_backends() -> Iterable[DeviceBackend]:
    """Yield each backend exactly once (deduplicated across aliases)."""
    seen: set[int] = set()
    for backend in _BACKENDS.values():
        if id(backend) in seen:
            continue
        seen.add(id(backend))
        yield backend


def iter_active_backends() -> Iterable[DeviceBackend]:
    """Yield non-CPU backends that report ``is_available()``."""
    for backend in iter_registered_backends():
        if backend.name == "cpu":
            continue
        try:
            if backend.is_available():
                yield backend
        except Exception:  # pragma: no cover
            continue


def get_device_backend(device: Union[None, str, int, torch.device, DeviceBackend] = None) -> DeviceBackend:
    """Resolve any device spec to a :class:`DeviceBackend` instance.

    - ``None`` → best available backend (CUDA > XPU > HPU > NPU > ... > CPU)
    - ``DeviceBackend`` → returned as-is
    - ``str`` / ``torch.device`` → matched by device-type prefix
    - ``int`` → best available non-CPU backend
    """
    if isinstance(device, DeviceBackend):
        return device
    if device is None:
        return auto_select_device()
    if isinstance(device, int):
        # Pure index → fall back to the highest-priority active backend
        for backend in sorted(iter_active_backends(), key=lambda b: b.priority, reverse=True):
            return backend
        return _BACKENDS["cpu"]
    dtype = _device_type_of(device)
    if dtype and dtype in _BACKENDS:
        return _BACKENDS[dtype]
    return _BACKENDS["cpu"]


def auto_select_device() -> DeviceBackend:
    """Return the highest-priority *available* backend (falls back to CPU).

    Prefers :func:`torch.accelerator.current_accelerator` (PyTorch ≥ 2.6) when
    available so we agree with whatever upstream torch decides; otherwise
    walks the registry by descending priority.
    """
    # Fast path: ask torch which accelerator is currently active.
    accel_name = current_accelerator_type()
    if accel_name and accel_name in _BACKENDS:
        backend = _BACKENDS[accel_name]
        try:
            if backend.is_available():
                return backend
        except Exception:  # pragma: no cover
            pass

    candidates = [b for b in iter_registered_backends() if b.name != "cpu"]
    candidates.sort(key=lambda b: b.priority, reverse=True)
    for b in candidates:
        try:
            if b.is_available():
                return b
        except Exception:  # pragma: no cover
            continue
    return _BACKENDS["cpu"]


def get_known_device_types(include_cpu: bool = True) -> tuple[str, ...]:
    """Return every registered device-type name (deduplicated; aliases excluded).

    Useful for replacing hard-coded literals like ``["cuda", "xpu", "hpu"]``::

        if device in get_known_device_types(include_cpu=False):
            ...
    """
    types = []
    for backend in iter_registered_backends():
        if not include_cpu and backend.name == "cpu":
            continue
        types.append(backend.name)
    return tuple(types)


def is_accelerator_type(name: str) -> bool:
    """Return True if ``name`` is the device-type string of a *non-CPU* backend."""
    if not isinstance(name, str):
        return False
    return name.lower() in {b.name for b in iter_registered_backends() if b.name != "cpu"}


def is_accelerator_device(device: Union[None, str, int, torch.device]) -> bool:
    """Return True if ``device`` resolves to a non-CPU registered backend.

    Use this instead of patterns like ``"cuda" in device or "xpu" in device``.
    """
    if device is None:
        return False
    dtype = _device_type_of(device)
    if dtype is None:
        return False
    return is_accelerator_type(dtype)


def strip_device_prefix(spec: str) -> str:
    """Strip known device-type prefixes from a comma-separated spec.

    ``"cuda:0,cuda:1"`` → ``"0,1"``.  Tokens whose prefix is *not* a
    registered backend type are left untouched.  This generalises the
    one-off ``re.sub("xpu:|hpu:|cuda:", "", ...)`` previously scattered
    around the codebase.
    """
    known = {b.name for b in iter_registered_backends()} | {
        a for b in iter_registered_backends() for a in b.aliases
    }
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            head, tail = tok.split(":", 1)
            if head.lower() in known:
                out.append(tail)
                continue
        out.append(tok)
    return ",".join(out)


def split_device_spec(spec: Union[str, int, torch.device, None]) -> list[str]:
    """Normalise *anything* into a ``list[str]`` of device tokens.

    - ``None`` → ``[]``
    - ``int`` → ``["0"]``
    - ``torch.device("cuda:1")`` → ``["cuda:1"]``
    - ``"cuda:0,cuda:1"`` → ``["cuda:0", "cuda:1"]``
    - ``"0, 1"`` → ``["0", "1"]``
    """
    if spec is None:
        return []
    if isinstance(spec, int):
        return [str(spec)]
    if isinstance(spec, torch.device):
        return [str(spec)]
    if isinstance(spec, str):
        return [s.strip() for s in spec.split(",") if s.strip()]
    return [str(spec)]


def current_accelerator_type() -> Optional[str]:
    """Return the current accelerator's device-type string, if torch reports one.

    Wraps :func:`torch.accelerator.current_accelerator` (added in PyTorch 2.6),
    returning ``None`` on older versions or when no accelerator is active.
    """
    accel_mod = getattr(torch, "accelerator", None)
    if accel_mod is None:
        return None
    fn = getattr(accel_mod, "current_accelerator", None)
    if not callable(fn):
        return None
    try:
        dev = fn()
    except Exception:
        return None
    if dev is None:
        return None
    # Some torch builds return a torch.device; others return a string.
    if isinstance(dev, torch.device):
        return dev.type
    return str(dev).split(":", 1)[0].lower() or None


def resolve_device_type(device: Union[None, str, int, torch.device]) -> str:
    """Return the canonical device-type string (``"cuda"`` / ``"cpu"`` / ...)."""
    return get_device_backend(device).name


def get_visible_devices_env_mapping() -> dict[str, str]:
    """Return ``{device_type: env_var}`` for every backend that defines one."""
    return {b.name: b.visible_devices_env for b in iter_registered_backends() if b.visible_devices_env}


# ---------------------------------------------------------------------------
# Built-in backends
# ---------------------------------------------------------------------------


@register_device_backend
class CPUBackend(DeviceBackend):
    """CPU backend.

    All hooks are implemented via the standard library so calling code can
    use the same API regardless of accelerator availability:

    * ``empty_cache`` runs :func:`gc.collect` and (on Linux) hints glibc to
      release free heap pages back to the OS via ``malloc_trim``.
    * ``memory_allocated`` / ``memory_reserved`` return the process RSS via
      :mod:`psutil` -- the closest CPU analogue.
    * ``total_memory`` returns total system RAM.
    """

    name = "cpu"
    priority = 0
    oom_signatures = ("DefaultCPUAllocator: not enough memory",)

    def is_available(self) -> bool:  # pragma: no cover - trivial
        return True

    def device_count(self) -> int:
        return 1

    def synchronize(self, index=None) -> None:
        return

    def empty_cache(self) -> None:
        """Run a Python GC pass and (on Linux) trim the glibc heap."""
        import gc

        gc.collect()
        # Lazy import to avoid a top-level circular import on device.py.
        try:
            from auto_round.utils.device import _maybe_trim_malloc

            _maybe_trim_malloc()
        except Exception:  # pragma: no cover
            pass

    def memory_allocated(self, index: int = 0) -> int:
        try:
            import psutil

            return int(psutil.Process().memory_info().rss)
        except Exception:  # pragma: no cover
            return 0

    def memory_reserved(self, index: int = 0) -> int:
        # On CPU there is no separate "reserved" pool; reuse RSS.
        return self.memory_allocated(index)

    def total_memory(self, index: int = 0) -> int:
        try:
            import psutil

            return int(psutil.virtual_memory().total)
        except Exception:  # pragma: no cover
            return 0

    def current_device(self) -> int:
        return 0

    def compile_func(self, func):  # pragma: no cover - rarely useful on CPU
        try:
            return torch.compile(func)
        except Exception:
            return func


@register_device_backend
class CUDABackend(DeviceBackend):
    name = "cuda"
    aliases = ("gpu",)
    priority = 100
    supports_parallel = True
    visible_devices_env = "CUDA_VISIBLE_DEVICES"
    oom_signatures = (
        "CUDA out of memory",
        "HIP out of memory. Tried to allocate",  # ROCm reuses torch.cuda
    )

    def is_available(self) -> bool:
        try:
            return bool(torch.cuda.is_available())
        except Exception:  # pragma: no cover
            return False


@register_device_backend
class XPUBackend(DeviceBackend):
    name = "xpu"
    priority = 80
    visible_devices_env = "ZE_AFFINITY_MASK"
    oom_signatures = ("UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY",)

    def is_available(self) -> bool:
        try:
            return bool(hasattr(torch, "xpu") and torch.xpu.is_available())
        except Exception:  # pragma: no cover
            return False


def is_hpu_lazy_mode() -> bool:
    """Return ``True`` when HPU is operating in lazy mode (the default)."""
    return os.getenv("PT_HPU_LAZY_MODE") != "0"


@register_device_backend
class HPUBackend(DeviceBackend):
    """Habana Gaudi backend (Intel HPU).

    HPU has a few quirks that the base class handles via overrides:

    * ``torch.compile`` must be invoked with ``backend="hpu_backend"``
      and only when *not* in lazy mode and torch ≥ 2.4.
    * ``memory_reserved`` is exposed as ``memory_cached``.
    * Memory-clear semantics are different: there is no
      ``empty_cache`` call that actually frees device pages, so we run
      a forced glibc ``malloc_trim`` and let the caller skip the
      generic CUDA/XPU path.
    """

    name = "hpu"
    priority = 60
    visible_devices_env = "HABANA_VISIBLE_MODULES"
    oom_signatures = ("MODULE:PT_DEVMEM",)

    @functools.lru_cache(maxsize=None)
    def _hpex_available(self) -> bool:  # noqa: D401
        if not _try_import("habana_frameworks"):
            return False
        try:  # the import has documented side-effects; mirror old utils/device.py
            import habana_frameworks.torch.hpex  # pylint: disable=E0401  # noqa: F401
        except Exception:
            return False
        return True

    def is_available(self) -> bool:
        return self._hpex_available()

    @property
    def torch_module(self):
        return getattr(torch, "hpu", None)

    def device_count(self) -> int:
        if not self.is_available():
            return 0
        try:
            import habana_frameworks.torch.hpu as hthpu  # pylint: disable=E0401

            return int(hthpu.device_count())
        except Exception:  # pragma: no cover
            return 0

    def memory_reserved(self, index: int = 0) -> int:
        mod = self.torch_module
        if mod is None:
            return 0
        for fn_name in ("memory_cached", "memory_reserved"):
            fn = getattr(mod, fn_name, None)
            if fn is not None:
                try:
                    return int(fn(index))
                except Exception:  # pragma: no cover
                    return 0
        return 0

    def total_memory(self, index: int = 0) -> int:
        # HPU does not expose ``get_device_properties().total_memory`` reliably.
        mod = self.torch_module
        if mod is None:
            return 0
        try:
            return int(mod.memory_cached(index))
        except Exception:  # pragma: no cover
            return 0

    def empty_cache(self) -> None:
        # No-op on HPU: no user-callable cache eviction.
        return

    def compile_func(self, func):
        from auto_round.utils.common import TORCH_VERSION_AT_LEAST_2_4

        if TORCH_VERSION_AT_LEAST_2_4 and not is_hpu_lazy_mode():
            try:
                return torch.compile(func, backend="hpu_backend")
            except Exception:  # pragma: no cover
                return func
        return func

    def extra_clear_memory(self, device_list=None) -> bool:
        """HPU-specific: trim glibc heap and let MemoryMonitor track HPU VRAM.

        Returning True tells the generic clear path to short-circuit, since
        cuda/xpu cache eviction is irrelevant for HPU-only runs.
        """
        # Avoid circular import: device.py ultimately imports this module.
        try:
            from auto_round.utils.device import _force_trim_malloc, memory_monitor

            _force_trim_malloc()
            try:
                memory_monitor.update_hpu(device_list)
            except Exception:  # pragma: no cover
                pass
        except Exception:  # pragma: no cover
            pass
        return True

    @functools.lru_cache(maxsize=None)
    def is_gaudi2(self) -> bool:
        try:
            import habana_frameworks.torch.utils.experimental as htexp  # pylint: disable=E0401

            return htexp._get_device_type() == htexp.synDeviceType.synDeviceGaudi2
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _try_import(module_name: str) -> bool:
    """Return True if ``module_name`` can be imported without errors."""
    from importlib.util import find_spec

    try:
        return find_spec(module_name) is not None
    except Exception:  # pragma: no cover
        return False

