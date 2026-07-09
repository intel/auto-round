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

from __future__ import annotations

import ast
import importlib
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import requests

from auto_round import envs
from auto_round.export.export_to_gguf.config import ModelType as AutoRoundModelType
from auto_round.logger import logger

LLAMA_CPP_RAW_URL = "https://raw.githubusercontent.com/ggml-org/llama.cpp"
LLAMA_CPP_API_URL = "https://api.github.com/repos/ggml-org/llama.cpp"
REQUEST_TIMEOUT = 30

_ACTIVE_CONVERSION_ROOT: Path | None = None
_CONVERSION_MODULE: ModuleType | None = None
_CONVERSION_SOURCE = ""


@dataclass
class ConversionContext:
    module: ModuleType
    source: str

    @property
    def ModelBase(self):
        return getattr(self.module, "ModelBase")

    @property
    def ModelType(self):
        return getattr(self.module, "ModelType")

    @property
    def get_model_architecture(self):
        return getattr(self.module, "get_model_architecture")

    def model_type(self, model_type: AutoRoundModelType | Any):
        if int(model_type) == int(AutoRoundModelType.MMPROJ):
            return self.ModelType.MMPROJ
        return self.ModelType.TEXT

    def get_model_class(self, model_architecture: str, model_type: AutoRoundModelType | Any = AutoRoundModelType.TEXT):
        llama_model_type = self.model_type(model_type)
        if hasattr(self.module, "get_model_class"):
            return self.module.get_model_class(model_architecture, mmproj=(llama_model_type == self.ModelType.MMPROJ))
        return self.ModelBase.from_model_architecture(model_architecture, model_type=llama_model_type)

    def is_supported(
        self, model_architecture: str, model_type: AutoRoundModelType | Any = AutoRoundModelType.TEXT
    ) -> bool:
        try:
            self.get_model_class(model_architecture, model_type=model_type)
            return True
        except NotImplementedError:
            return False


class GGUFConversionError(ImportError):
    """Raised when llama.cpp conversion code is unavailable or incompatible."""


def _cache_root() -> Path:
    return Path(envs.AUTO_ROUND_CACHE or Path.home() / ".cache" / "auto-round") / "gguf_conversion"


def _bundled_root() -> Path:
    return Path(__file__).resolve().parent


def _local_llama_cpp_root() -> Path | None:
    value = envs.LLAMA_CPP_ROOT
    if not value:
        return None
    root = Path(value).expanduser().resolve()
    if not (root / "conversion" / "__init__.py").is_file():
        raise GGUFConversionError(
            f"LLAMA_CPP_ROOT={root} does not contain conversion/__init__.py. "
            "Please point it to a llama.cpp checkout."
        )
    return root


def _auto_update_enabled() -> bool:
    return envs.AUTO_ROUND_GGUF_AUTO_UPDATE


def _clear_loaded_conversion_modules() -> None:
    for name in list(sys.modules):
        if name == "conversion" or name.startswith("conversion."):
            del sys.modules[name]


def _configure_conversion_logger() -> None:
    conversion_logger = logging.getLogger("hf-to-gguf")
    conversion_logger.setLevel(logger.level)
    conversion_logger.propagate = False
    for handler in logger.handlers:
        if handler not in conversion_logger.handlers:
            conversion_logger.addHandler(handler)


def _import_conversion_from(root: Path, source: str) -> ConversionContext:
    global _ACTIVE_CONVERSION_ROOT, _CONVERSION_MODULE, _CONVERSION_SOURCE

    root = root.resolve()
    if _CONVERSION_MODULE is not None and _ACTIVE_CONVERSION_ROOT == root:
        _configure_conversion_logger()
        return ConversionContext(_CONVERSION_MODULE, _CONVERSION_SOURCE)

    _clear_loaded_conversion_modules()
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("conversion")
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass

    _ACTIVE_CONVERSION_ROOT = root
    _CONVERSION_MODULE = module
    _CONVERSION_SOURCE = source
    _configure_conversion_logger()
    return ConversionContext(module, source)


def _available_conversion_roots() -> list[tuple[Path, str]]:
    roots = []
    local_root = _local_llama_cpp_root()
    if local_root is not None:
        roots.append((local_root, f"LLAMA_CPP_ROOT={local_root}"))

    bundled_root = _bundled_root()
    if (bundled_root / "conversion" / "__init__.py").is_file():
        roots.append((bundled_root, "AutoRound bundled llama.cpp conversion"))

    return roots


def _load_config(model_path: str | Path) -> dict[str, Any]:
    config_path = Path(model_path) / "config.json"
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def _architecture_from_hparams(
    hparams: dict[str, Any], model_type: AutoRoundModelType | Any = AutoRoundModelType.TEXT
) -> str | None:
    if int(model_type) == int(AutoRoundModelType.MMPROJ):
        for key in ("vision_config", "vision_encoder"):
            nested = hparams.get(key)
            if isinstance(nested, dict) and nested.get("architectures"):
                return nested["architectures"][0]
    if hparams.get("architectures"):
        return hparams["architectures"][0]
    text_config = hparams.get("text_config") or hparams.get("llm_config") or hparams.get("language_config")
    if isinstance(text_config, dict) and text_config.get("architectures"):
        return text_config["architectures"][0]
    return None


def _fetch_text(url: str) -> str:
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def _latest_commit() -> str:
    response = requests.get(f"{LLAMA_CPP_API_URL}/commits/master", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()["sha"]


def _download_conversion_file(commit: str, relative_path: str, target_root: Path) -> Path:
    text = _fetch_text(f"{LLAMA_CPP_RAW_URL}/{commit}/{relative_path}")
    output_path = target_root / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def _literal_map_from_init(init_text: str, map_name: str) -> dict[str, str]:
    tree = ast.parse(init_text)
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == map_name:
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == map_name:
                    return ast.literal_eval(node.value)
    return {}


def _conversion_dependencies(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    deps = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 1 and node.module:
                deps.add(f"conversion/{node.module.split('.')[0]}.py")
            elif node.level == 0 and node.module and node.module.startswith("conversion."):
                deps.add(f"{node.module.replace('.', '/')}.py")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("conversion."):
                    deps.add(f"{alias.name.replace('.', '/')}.py")
    deps.discard("conversion/__init__.py")
    return deps


def _download_dependency_closure(commit: str, target_root: Path, initial_files: set[str]) -> None:
    pending = set(initial_files)
    downloaded = set()
    while pending:
        relative_path = pending.pop()
        if relative_path in downloaded:
            continue
        try:
            path = _download_conversion_file(commit, relative_path, target_root)
        except requests.HTTPError as error:
            if error.response is not None and error.response.status_code == 404:
                continue
            raise
        downloaded.add(relative_path)
        if relative_path.endswith(".py"):
            pending.update(_conversion_dependencies(path) - downloaded)


def _download_dynamic_conversion(model_architecture: str, model_type: AutoRoundModelType | Any) -> Path:
    commit = _latest_commit()
    target_root = _cache_root() / commit
    conversion_dir = target_root / "conversion"
    if conversion_dir.is_dir():
        return target_root

    tmp_root = target_root.with_name(f"{target_root.name}.tmp")
    shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    init_text = _fetch_text(f"{LLAMA_CPP_RAW_URL}/{commit}/conversion/__init__.py")
    init_path = tmp_root / "conversion" / "__init__.py"
    init_path.parent.mkdir(parents=True, exist_ok=True)
    init_path.write_text(init_text, encoding="utf-8")

    map_name = "MMPROJ_MODEL_MAP" if int(model_type) == int(AutoRoundModelType.MMPROJ) else "TEXT_MODEL_MAP"
    model_map = _literal_map_from_init(init_text, map_name)
    module_name = model_map.get(model_architecture)
    if module_name is None:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise NotImplementedError(f"Model {model_architecture} is not supported by latest llama.cpp conversion either.")

    initial_files = {"conversion/base.py", f"conversion/{module_name}.py"}
    _download_dependency_closure(commit, tmp_root, initial_files)

    shutil.rmtree(target_root, ignore_errors=True)
    tmp_root.rename(target_root)
    logger.info("Downloaded llama.cpp conversion %s for %s to %s", commit, model_architecture, target_root)
    return target_root


def _unsupported_message(model_architecture: str) -> str:
    return (
        f"Model {model_architecture} is not supported by AutoRound's bundled llama.cpp conversion. "
        "This may mean the model is newer than the bundled converter, or llama.cpp does not support it yet. "
        "To try the latest llama.cpp converter automatically, set AUTO_ROUND_GGUF_AUTO_UPDATE=1. "
        "Or use a local llama.cpp checkout with LLAMA_CPP_ROOT=/path/to/llama.cpp."
    )


def get_conversion(
    model_path: str | Path | None = None,
    model_type: AutoRoundModelType | Any = AutoRoundModelType.TEXT,
    hparams: dict[str, Any] | None = None,
    auto_update: bool | None = None,
) -> ConversionContext:
    if hparams is None and model_path is not None:
        hparams = _load_config(model_path)
    model_architecture = _architecture_from_hparams(hparams, model_type) if hparams is not None else None

    last_error: Exception | None = None
    for root, source in _available_conversion_roots():
        try:
            context = _import_conversion_from(root, source)
            if model_architecture is None or context.is_supported(model_architecture, model_type):
                return context
            last_error = NotImplementedError(_unsupported_message(model_architecture))
        except Exception as error:
            last_error = error

    if model_architecture is not None and (auto_update if auto_update is not None else _auto_update_enabled()):
        root = _download_dynamic_conversion(model_architecture, model_type)
        context = _import_conversion_from(root, f"cached llama.cpp conversion at {root}")
        if context.is_supported(model_architecture, model_type):
            return context
        raise NotImplementedError(f"Model {model_architecture} is not supported by latest llama.cpp conversion either.")

    if model_architecture is not None:
        raise NotImplementedError(_unsupported_message(model_architecture)) from last_error

    raise GGUFConversionError(
        "llama.cpp conversion code is not available. AutoRound expects a bundled conversion directory, "
        "or set LLAMA_CPP_ROOT=/path/to/llama.cpp."
    ) from last_error


def get_model_architecture(
    hparams: dict[str, Any],
    model_type: AutoRoundModelType | Any = AutoRoundModelType.TEXT,
    model_path: str | Path | None = None,
) -> str:
    context = get_conversion(model_path=model_path, model_type=model_type, hparams=hparams, auto_update=False)
    return context.get_model_architecture(hparams, context.model_type(model_type))
