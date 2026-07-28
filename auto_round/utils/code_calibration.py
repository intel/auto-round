# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
"""Code-model detection and automatic calibration dataset selection."""

import re
from dataclasses import dataclass
from typing import Any

from packaging.version import Version

_CODE_TOKENS = {"code", "coder", "coding", "programming", "swe", "devstral"}
_CODE_FAMILIES = {
    "codellama",
    "codegemma",
    "codestral",
    "deepseekcoder",
    "granitecode",
    "magicoder",
    "opencoder",
    "qwencoder",
    "santacoder",
    "stablecode",
    "starcoder",
    "wizardcoder",
}
_CODE_TASKS = {"code-generation", "software-engineering", "text-to-code"}
_GITHUB_CODE_CLEAN_MAX_VERSION = Version("3.6.0")


@dataclass(frozen=True)
class CodeModelDetection:
    is_code: bool
    source: str | None = None
    match: str | None = None


@dataclass(frozen=True)
class CodeCalibrationSelection:
    dataset: str
    datasets_version: str
    counts: dict[str, int]
    github_code_clean_enabled: bool


def _tokens(value: str) -> set[str]:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return set(re.findall(r"[a-z]+", value.lower()))


def _match_code_name(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    tokens = _tokens(value)
    token_matches = _CODE_TOKENS.intersection(tokens)
    if token_matches:
        return sorted(token_matches)[0]
    components = re.findall(r"[a-z0-9]+", value.lower())
    for family in sorted(_CODE_FAMILIES):
        if any(component == family or re.fullmatch(rf"{family}\d+", component) for component in components):
            return family
    return None


def _model_reference_name(value: str) -> str:
    """Return the model/repository name without unrelated parent directories."""
    return re.split(r"[/\\]", value.rstrip("/\\"))[-1]


def _config_value(config: Any, name: str) -> Any:
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(name)
    return getattr(config, name, None)


def detect_code_model(model: Any, config: Any = None) -> CodeModelDetection:
    """Detect models explicitly specialized for code using local metadata only."""
    candidates = []
    if isinstance(model, str):
        candidates.append(("model identifier", _model_reference_name(model)))
    else:
        model_name_or_path = getattr(model, "name_or_path", None)
        if isinstance(model_name_or_path, str):
            model_name_or_path = _model_reference_name(model_name_or_path)
        candidates.append(("model name_or_path", model_name_or_path))
        if config is None:
            config = getattr(model, "config", None)

    config_name_or_path = _config_value(config, "_name_or_path")
    if isinstance(config_name_or_path, str):
        config_name_or_path = _model_reference_name(config_name_or_path)
    candidates.extend(
        [
            ("config._name_or_path", config_name_or_path),
            ("config.model_type", _config_value(config, "model_type")),
        ]
    )
    architectures = _config_value(config, "architectures") or []
    if isinstance(architectures, str):
        architectures = [architectures]
    candidates.extend(("config.architectures", architecture) for architecture in architectures)

    for source, value in candidates:
        match = _match_code_name(value)
        if match:
            return CodeModelDetection(True, source, match)

    for field in ("finetuning_task", "task", "pipeline_tag"):
        value = _config_value(config, field)
        if isinstance(value, str) and value.lower().replace("_", "-") in _CODE_TASKS:
            return CodeModelDetection(True, f"config.{field}", value)

    task_specific_params = _config_value(config, "task_specific_params") or {}
    if isinstance(task_specific_params, dict):
        for task in task_specific_params:
            normalized_task = str(task).lower().replace("_", "-")
            if normalized_task in _CODE_TASKS:
                return CodeModelDetection(True, "config.task_specific_params", str(task))

    return CodeModelDetection(False)


def _allocate_samples(nsamples: int, weights: list[float]) -> list[int]:
    if nsamples < 1:
        raise ValueError("nsamples must be a positive integer")
    raw_counts = [nsamples * weight / sum(weights) for weight in weights]
    counts = [int(count) for count in raw_counts]
    remainder = nsamples - sum(counts)
    order = sorted(range(len(weights)), key=lambda index: (-(raw_counts[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def build_code_calibration_dataset(
    nsamples: int, datasets_version: str | Version | None = None
) -> CodeCalibrationSelection:
    """Build an exact-size code calibration mix compatible with datasets."""
    if datasets_version is None:
        import datasets

        datasets_version = datasets.__version__
    parsed_version = Version(str(datasets_version))

    sources = [("opencode-instruct", 50), ("github-code-clean", 40), ("mbpp:split=train", 10)]
    github_enabled = parsed_version <= _GITHUB_CODE_CLEAN_MAX_VERSION
    if not github_enabled:
        sources = [source for source in sources if source[0] != "github-code-clean"]

    allocated = _allocate_samples(nsamples, [weight for _, weight in sources])
    counts = {name: count for (name, _), count in zip(sources, allocated) if count}
    dataset = ",".join(f"{name}:num={count}" for name, count in counts.items())
    return CodeCalibrationSelection(dataset, str(parsed_version), counts, github_enabled)
