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
from dataclasses import dataclass
from typing import Dict, Type

import torch
from tqdm import tqdm
from transformers import PreTrainedModel

from auto_round.utils import LazyImport, dump_mem_usage, dump_memory_usage_ctx, logger, memory_monitor

BUILTIN_MODULES = {
    "Llama4TextMoe": LazyImport("auto_round.modelling.llama4"),
    "GptOssMLP": LazyImport("auto_round.modelling.gpt_oss"),
    "Qwen3VLMoeTextSparseMoeBlock": LazyImport("auto_round.modelling.qwen3_vl_moe"),
    "DeepseekV2Attention": LazyImport("auto_round.modelling.deepseek_v2"),
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


@dump_mem_usage("Materializing model", log_level="info")
def materialize_model_(model: torch.nn.Module) -> None:
    def _materialize_module(module: torch.nn.Module) -> None:
        if isinstance(module, ReplacementModuleBase):
            module.materialize_weights()

    model.apply(_materialize_module)
    # check if any module on meta device remains
    found_meta = False
    for name, param in model.named_parameters():
        if param.device.type == "meta":
            logger.warning(f"Parameter {name} is still on meta device after materialization.")
            found_meta = True
    for name, buffer in model.named_buffers():
        if buffer.device.type == "meta":
            logger.warning(f"Buffer {name} is still on meta device after materialization.")
            found_meta = True
    if not found_meta:
        logger.debug("All parameters and buffers have been materialized from meta device.")
    release_original_module_(model)


def release_original_module_(model: torch.nn.Module) -> None:
    def _clear_source_module(module: torch.nn.Module) -> None:
        if isinstance(module, ReplacementModuleBase):
            module.release_original_module()

    model.apply(_clear_source_module)


def _has_meta_params_or_buffers(model: PreTrainedModel) -> bool:
    for _, param in model.named_parameters():
        if param.device.type == "meta":
            return True
    for _, buffer in model.named_buffers():
        if buffer.device.type == "meta":
            return True
    return False


def safe_to_cpu_(model: torch.nn.Module) -> None:
    if _has_meta_params_or_buffers(model):
        materialize_model_(model)
    model.to("cpu")


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

    def __init__(self, original: torch.nn.Module):
        super().__init__()
        _global_tracker.register_replacement(
            name=str(id(self)),
            original=original,
            replacement=self,
        )
        self._materialized = False

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
    ) -> bool:
        """Determine if the given module should be replaced.

        Users can extend this method to add custom logic for replacement.
        """
        return cls.is_registered(original.__class__.__name__)

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

    def materialize_weights(self):
        """Materialize weights if needed."""
        if not self._materialized:
            self._materialize_weights()
            self.post_process_materialization()

    def _materialize_weights(self) -> None:
        """Materialize weights from the original module.

        Subclasses should override this method to implement
        weight materialization logic.
        """
        pass

    def release_original_module(self) -> None:
        """Release reference to the original module to free memory."""
        # Release from global tracker
        _global_tracker.release_original(self)

    def _get_original_module(self) -> torch.nn.Module:
        """Get the original module associated with this replacement."""
        return _global_tracker.get_original(self)

    def post_process_materialization(self) -> None:
        """Mark the replacement module as materialized."""
        self._materialized = True
        self.release_original_module()


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
    replaced = []

    # Step 1: Collect all modules that need replacement
    logger.debug("Scanning for modules to replace")
    modules_to_replace = []
    for name, module in model.named_modules():
        # skip replaced modules
        if isinstance(module, ReplacementModuleBase):
            continue
        class_name = module.__class__.__name__
        if ReplacementModuleBase.is_registered(class_name) and ReplacementModuleBase.get_replacement_class(
            class_name
        ).is_to_be_replaced(module):
            modules_to_replace.append((name, module, class_name))

    # Step 2: Replace modules
    if modules_to_replace:
        logger.info(f"Found {len(modules_to_replace)} modules to replace")
        for name, module, class_name in tqdm(modules_to_replace, desc="Replacing modules"):
            module = model.get_submodule(name)
            with dump_memory_usage_ctx(f"Replacing module {name}"):
                replacement_cls = ReplacementModuleBase.get_replacement_class(class_name)
                replacement = replacement_cls.from_original(
                    module,
                    model.config,
                )
                model.set_submodule(name, replacement)
                replaced.append((name, replacement_cls))
    else:
        logger.debug("No modules found for replacement")

    # Log what was replaced
    if replaced:
        logger.info(f"Replaced {len(replaced)} modules")

    return model


@dataclass
class ReplacedModuleInfo:
    original_module: torch.nn.Module
    replacement_module: ReplacementModuleBase


class ModuleReplacementTracker:
    """Tracker to maintain mapping between replacement modules and their original modules.

    This is a singleton class - only one instance can exist.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModuleReplacementTracker, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if ModuleReplacementTracker._initialized:
            return

        # Map from replacement module id to original module
        self._replacement_to_original: Dict[int, torch.nn.Module] = {}
        # Map from module name to ReplacedModuleInfo
        self._name_to_info: Dict[str, ReplacedModuleInfo] = {}

        ModuleReplacementTracker._initialized = True

    @classmethod
    def get_instance(cls) -> "ModuleReplacementTracker":
        """Get the singleton instance of the tracker."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_replacement(self, name: str, original: torch.nn.Module, replacement: ReplacementModuleBase) -> None:
        """Register a module replacement."""
        self._replacement_to_original[id(replacement)] = original
        self._name_to_info[name] = ReplacedModuleInfo(original_module=original, replacement_module=replacement)
        logger.trace(f"Registered replacement for module: {name}")

    def get_original(self, replacement: ReplacementModuleBase) -> torch.nn.Module:
        """Get the original module for a given replacement module."""
        return self._replacement_to_original.get(id(replacement))

    def get_info_by_name(self, name: str) -> ReplacedModuleInfo:
        """Get replacement info by module name."""
        return self._name_to_info.get(name)

    def release_original(self, replacement: ReplacementModuleBase) -> None:
        """Release the original module associated with a replacement module."""
        replacement_id = id(replacement)
        if replacement_id in self._replacement_to_original:
            original = self._replacement_to_original[replacement_id]
            # Delete the original module to free memory
            del original
            del self._replacement_to_original[replacement_id]
            logger.trace(f"Released original module for replacement {replacement_id}")

    def release_all_originals(self) -> None:
        """Release all tracked original modules."""
        count = len(self._replacement_to_original)
        if count > 0:
            self._replacement_to_original.clear()
            logger.debug(f"Released {count} original modules from tracker")

    def clear(self) -> None:
        """Clear all tracked information."""
        self._replacement_to_original.clear()
        self._name_to_info.clear()
        logger.debug("Cleared module replacement tracker")


_global_tracker = ModuleReplacementTracker()
