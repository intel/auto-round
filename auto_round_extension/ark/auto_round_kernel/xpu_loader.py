# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import sys
import sysconfig
from pathlib import Path
from typing import Iterable


_DEFAULT_MODULE_NAME = "auto_round_kernel._local.auto_round_kernel_xpu"


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def _package_module():
    return sys.modules[__package__]


def _extension_patterns() -> tuple[str, ...]:
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if ext_suffix:
        return (f"auto_round_kernel_xpu*{ext_suffix}", "auto_round_kernel_xpu*.so")
    return ("auto_round_kernel_xpu*.so",)


def _normalize_search_roots(search_roots: Iterable[Path | str] | None) -> list[Path]:
    if search_roots is None:
        package_dir = _package_dir()
        search_roots = (package_dir, package_dir / "xbuild", package_dir / "xbuild_diffuser")

    normalized: list[Path] = []
    seen: set[Path] = set()
    for root in search_roots:
        resolved = Path(root).expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        normalized.append(resolved)
    return normalized


def _find_xpu_extension(search_roots: Iterable[Path | str] | None = None) -> Path | None:
    patterns = _extension_patterns()
    candidates: list[Path] = []
    for root in _normalize_search_roots(search_roots):
        if not root.exists():
            continue
        for pattern in patterns:
            for candidate in sorted(root.glob(pattern)):
                if candidate.is_file():
                    candidates.append(candidate.resolve())
    return candidates[-1] if candidates else None


def load_xpu_lib(
    ext_path: Path | str,
    *,
    required_symbols: tuple[str, ...] = (),
    module_name: str = _DEFAULT_MODULE_NAME,
):
    ext_path = Path(ext_path).expanduser().resolve()
    if not ext_path.is_file():
        raise RuntimeError(f"Unable to locate XPU extension: {ext_path}")

    spec = importlib.util.spec_from_file_location(module_name, ext_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load extension spec from {ext_path}")

    sys.modules.pop(module_name, None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    missing = [symbol for symbol in required_symbols if not hasattr(module, symbol)]
    if missing:
        sys.modules.pop(module_name, None)
        raise RuntimeError(f"Loaded extension is missing required XPU bindings {missing}: {ext_path}")

    package_module = _package_module()
    package_module.xpu_lib = module
    return module


def ensure_xpu_lib(
    *,
    required_symbols: tuple[str, ...] = (),
    ext_path: Path | str | None = None,
    search_roots: Iterable[Path | str] | None = None,
    module_name: str = _DEFAULT_MODULE_NAME,
):
    package_module = _package_module()
    current_lib = getattr(package_module, "xpu_lib", None)
    if current_lib is not None and all(hasattr(current_lib, symbol) for symbol in required_symbols):
        return current_lib

    if ext_path is None:
        ext_path = _find_xpu_extension(search_roots)
    if ext_path is None:
        search_list = ", ".join(str(root) for root in _normalize_search_roots(search_roots))
        raise RuntimeError(f"Unable to locate built XPU extension under: {search_list}")

    return load_xpu_lib(ext_path, required_symbols=required_symbols, module_name=module_name)
