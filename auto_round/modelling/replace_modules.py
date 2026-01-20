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

from abc import ABC, abstractmethod
from typing import Dict, Type

import torch
from tqdm import tqdm
from transformers import PreTrainedModel

from auto_round.utils import LazyImport, is_hpex_available, logger

BUILTIN_MODULES = {
    "Llama4TextMoe": LazyImport("auto_round.modelling.llama4"),
    "GptOssMLP": LazyImport("auto_round.modelling.gpt_oss"),
    "Qwen3VLMoeTextSparseMoeBlock": LazyImport("auto_round.modelling.qwen3_vl_moe"),
    "DeepseekV2Attention": LazyImport("auto_round.modelling.deepseek_v2")
}


def _import_required_replacements(model: torch.nn.Module) -> None:
    """Scan model and trigger lazy imports for registered replacement modules."""
    imported = set()

    for _, module in model.named_modules():
        class_name = module.__class__.__name__

        if class_name in BUILTIN_MODULES and class_name not in imported:
            # Trigger import by accessing the LazyImport object
            _ = BUILTIN_MODULES[class_name].__name__  # or any attribute
            imported.add(class_name)
            logger.debug(f"Loaded replacement module for {class_name}")


class ReplacementModuleBase(ABC, torch.nn.Module):
    """
    Abstract base class for module replacement during calibration phase.

    Replacement modules replace original modules to ensure all components
    receive data for proper quantization statistics.

    Subclasses must:
    1. Implement `original_module_class()` to return the target module class name
    2. Implement `__init__()` with signature:
       (self, original, config)
    """

    # Registry: module class name -> replacement module class
    _replacement_registry: Dict[str, Type["ReplacementModuleBase"]] = {}

    def __init_subclass__(cls, **kwargs):
        """Automatically register subclasses in the replacement registry."""
        super().__init_subclass__(**kwargs)

        # Only register if it's a concrete implementation (not ABC)
        if not getattr(cls, "__abstractmethods__", None):
            if cls.original_module_class() is None:
                raise TypeError(
                    f"{cls.__name__} must implement 'original_module_class()' class method "
                    "to return the name of the module class it replaces"
                )

            if cls.original_module_class() in cls._replacement_registry:
                existing = cls._replacement_registry[cls.original_module_class()]
                raise ValueError(
                    f"Module '{cls.original_module_class()}' already registered to "
                    f"{existing.__name__}. Cannot register {cls.__name__}."
                )

            cls._replacement_registry[cls.original_module_class()] = cls
            logger.trace(f"Registered {cls.__name__} for replacing {cls.original_module_class()}")

    @classmethod
    def get_replacement_class(cls, module_class_name: str) -> Type["ReplacementModuleBase"]:
        """Get replacement class for a given module class name."""
        return cls._replacement_registry.get(module_class_name)

    @classmethod
    def is_registered(cls, module_class_name: str) -> bool:
        """Check if a module class has a replacement implementation."""
        return module_class_name in cls._replacement_registry

    @classmethod
    def is_to_be_replaced(
        cls,
        original: torch.nn.Module,
        module_class_name: str,
    ) -> bool:
        """Determine if the given module should be replaced.

        Users can extend this method to add custom logic for replacement.
        """
        return cls.is_registered(module_class_name)

    @classmethod
    def get_registered_modules(cls) -> list:
        """Get list of all registered module class names."""
        return list(cls._replacement_registry.keys())

    @classmethod
    @abstractmethod
    def original_module_class(cls) -> str:
        """Return the class name of the module this replaces."""
        pass

    @classmethod
    @abstractmethod
    def from_original(
        cls,
        original: torch.nn.Module,
        config,
    ) -> "ReplacementModuleBase":
        """Create replacement module from original module."""
        pass


# Note: adapted from llm-compressor
# https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/modeling/moe_context.py


def apply_replacements(
    model: torch.nn.Module,
) -> torch.nn.Module:
    """
    Function to apply module replacements to a model.

    This scans all modules in the model and replaces any registered modules with their
    replacement equivalents. Non-permanent modules are tracked for later restoration.

    The model is modified in-place, so the same model object should be used.

    Args:
        model: The model to apply module replacement to (modified in-place).

    Returns:
        The model with modules replaced.
    """
    _import_required_replacements(model)
    replaced = {}

    # Step 1: Collect all modules that need replacement
    logger.debug("Scanning for modules to replace")
    modules_to_replace = []
    for name, module in model.named_modules():
        # skip replaced modules
        if isinstance(module, ReplacementModuleBase):
            continue
        class_name = module.__class__.__name__
        if ReplacementModuleBase.is_to_be_replaced(module, class_name):
            modules_to_replace.append((name, module, class_name))

    # Step 2: Replace modules
    if modules_to_replace:
        logger.info(f"Found {len(modules_to_replace)} modules to replace")
        for name, module, class_name in tqdm(modules_to_replace, desc="Replacing modules"):
            replacement_cls = ReplacementModuleBase.get_replacement_class(class_name)
            replacement = replacement_cls.from_original(
                module,
                model.config,
            )
            model.set_submodule(name, replacement)
            replaced[name] = (module, replacement)
    else:
        logger.debug("No modules found for replacement")

    # Log what was replaced
    if replaced:
        logger.info(f"Replaced {len(replaced)} modules")

    return model
