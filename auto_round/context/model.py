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

import gc
import importlib
from typing import Any, Callable, Optional, Union

import torch
from packaging import version
from transformers import AutoConfig

from auto_round import envs
from auto_round.compressors.utils import get_shared_keys
from auto_round.context.base import BaseContext
from auto_round.logger import logger
from auto_round.modeling.unfused_moe import apply_model_monkey_patches
from auto_round.special_model_handler import _handle_special_model, update_module
from auto_round.utils import (
    check_and_mark_quantized_module,
    diffusion_load_model,
    is_diffusion_model,
    is_mllm_model,
    is_moe_model,
    is_moe_model_via_config,
    llm_load_model,
    mllm_load_model,
    unsupported_meta_device,
)
from auto_round.utils.device import _force_trim_malloc
from auto_round.utils.device_manager import device_manager, get_ar_device

__all__ = ["ModelContext"]

_CUSTOM_MOE_REPLACEMENT_MODULES = {
    "gpt_oss": "auto_round.modeling.fused_moe.gpt_oss",
}


class ModelContext(BaseContext):
    _is_initialized = False

    # model_related
    _model_loaded = False
    _init_model = False
    hook_handles = []

    def __init__(
        self,
        model: Union[torch.nn.Module, str, None] = None,
        tokenizer: Any = None,
        platform: str = "hf",
        model_dtype: Optional[Union[str, torch.dtype]] = None,
        trust_remote_code: bool = True,
        config: Optional[AutoConfig] = None,
        amp: bool = True,
        need_calib: bool = True,
        is_act_quantize: bool = False,
        quant_nontext_module: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.quantized = False
        self.is_mllm = False
        self.is_diffusion = False
        self.is_model_patched = False
        self.is_moe_model = False
        # Set by CalibCompressor._replace_forward; used by recover_forward to detect
        # new-arch diffusion mode where positional wrapper must be stripped after caching.
        self._has_true_orig_forward_set = False

        assert model is not None, "model must be provided for ModelContext"
        self.model = model
        self.tokenizer = tokenizer

        # MLLM / diffusion artifacts – always present so callers need no getattr guards.
        # _load_model() will populate the ones that are relevant to the model type.
        self.processor = None
        self.image_processor = None
        self.pipe = None

        # AWQ weight-clip thresholds kept for downstream block quantizers.
        # Populated by AWQTransform when ``apply_clip`` is enabled; keyed by
        # layer ``global_name`` -> per-group clip magnitude tensor. SignRound /
        # SignRoundV2 use these to initialize their tunable weight range.
        self.awq_clip_values: dict = {}

        if envs.AR_USE_MODELSCOPE:
            platform = "model_scope"
        self.platform = platform
        self.model_dtype = model_dtype
        self.trust_remote_code = trust_remote_code
        self.config = config
        self.amp = amp
        self.need_calib = need_calib
        self.quant_nontext_module = quant_nontext_module

        # Load model and run basic initialization eagerly so the model is ready
        # by the time BaseCompressor.post_init() runs.
        self._load_model()

        if unsupported_meta_device(self.model):
            raise RuntimeError(
                "AutoRound does not support parameters on meta device. "
                "Please use more GPUs by setting `--device 0,1,2,3` or just place the model on CPU."
            )
        check_and_mark_quantized_module(self.model)
        self.model = self.model.eval()
        self.shared_cache_keys = get_shared_keys(self.model)

        self.is_moe_model = is_moe_model(self.model)
        self._import_custom_moe_replacements(getattr(self.model, "config", None))

        self._set_amp_dtype()
        if is_act_quantize and self.amp_dtype == torch.float16:
            logger.warning("force to use bf16 for quantization tuning when enabling activation quantization")
            self.amp_dtype = torch.bfloat16
            if self.model.dtype != torch.bfloat16:
                self.model = self.model.to(torch.bfloat16)
        else:
            logger.debug(f"using {self.model.dtype} for quantization tuning")

        # Reclaim C heap fragmentation left by model/tokenizer loading so
        # that the quantize loop starts from a tighter RSS baseline.
        gc.collect()
        _force_trim_malloc()

    @property
    def device(self) -> str:
        """The active (major) device, single-sourced from the DeviceManager."""
        return device_manager.device

    @device.setter
    def device(self, value) -> None:
        device_manager.device = value

    def _load_model(self):
        if is_mllm_model(self.model, platform=self.platform):
            self.is_mllm = True
            if isinstance(self.model, str):
                self.model, self.processor, self.tokenizer, self.image_processor = mllm_load_model(
                    self.model, platform=self.platform, device="cpu", model_dtype=self.model_dtype
                )
        elif is_diffusion_model(self.model):
            self.is_diffusion = True
            self.pipe, self.model = diffusion_load_model(
                self.model, platform=self.platform, device="cpu", model_dtype=self.model_dtype
            )
        elif isinstance(self.model, str):
            config = self.config
            try:
                if config is None:
                    config = AutoConfig.from_pretrained(self.model, trust_remote_code=self.trust_remote_code)
                self._import_custom_moe_replacements(config)
            except (OSError, EnvironmentError, ValueError) as e:
                logger.debug(
                    "Failed to load config via AutoConfig.from_pretrained for %s: %s. "
                    "Proceeding without config-based checks.",
                    self.model,
                    e,
                )

            self.is_model_patched = apply_model_monkey_patches(
                model_name=self.model, trust_remote_code=self.trust_remote_code
            )
            import transformers

            if (
                not self.is_model_patched
                and config is not None
                and is_moe_model_via_config(config)
                and version.parse(transformers.__version__) >= version.parse("5.0.0")
            ):
                from auto_round.modeling.fused_moe.replace_modules import BUILTIN_MODULES

                model_type = getattr(config, "model_type", None)
                if model_type is not None and model_type not in BUILTIN_MODULES:
                    logger.warning(
                        "This MoE model has not been optimized by AutoRound yet, which may result in high RAM usage, "
                        "Please consider submitting an issue to https://github.com/intel/auto-round/issues"
                    )

            # Reclaim temporary HTTP/config objects from model type detection
            # and AutoConfig loading before the large model allocation.  This
            # reduces heap fragmentation especially on HPU where habana internal
            # allocations amplify fragmentation into persistent RSS growth.
            gc.collect()
            _force_trim_malloc()

            self.model, self.tokenizer = llm_load_model(
                self.model,
                platform=self.platform,
                device="cpu",  # always load cpu first
                model_dtype=self.model_dtype,
                trust_remote_code=self.trust_remote_code,
            )
        elif self.tokenizer is None and not self.is_diffusion and self.need_calib:
            raise ValueError("A tokenizer must be set for non-str model input")

        self._model_loaded = True

    def _import_custom_moe_replacements(self, model_or_config) -> None:
        model_type = getattr(model_or_config, "model_type", None)
        module_name = _CUSTOM_MOE_REPLACEMENT_MODULES.get(model_type)
        if module_name is None:
            return

        module = importlib.import_module(module_name)
        from auto_round.modeling.fused_moe.replace_modules import BUILTIN_MODULES

        BUILTIN_MODULES.setdefault(model_type, module)
        logger.debug(f"Loaded custom MoE replacement module for {model_type}")

    def _patch_custom_moe_modules(self) -> None:
        model_type = getattr(getattr(self.model, "config", None), "model_type", None)
        if model_type != "qwen3_vl_moe":
            return

        for module in self.model.modules():
            if module.__class__.__name__ != "Qwen3VLMoeTextSparseMoeBlock":
                continue
            if hasattr(module, "top_k"):
                continue

            gate = getattr(module, "gate", None)
            top_k = getattr(gate, "top_k", None)
            if top_k is not None:
                setattr(module, "top_k", top_k)

    def _set_amp_dtype(self) -> None:
        """Sets the automatic mixed precision (AMP) data type for the model based on the device and configuration.

        The device only exposes capability/preference primitives
        (``supports_bf16`` / ``prefers_bf16``); this method composes them into
        the final ``amp`` / ``amp_dtype`` decision.
        """
        device = get_ar_device(self.device)
        if not self.amp:
            self.amp_dtype = torch.float32
        else:
            amp_dtype = torch.bfloat16
            if self.model.dtype != torch.float32:
                amp_dtype = self.model.dtype
            # bf16-preferring backends (CPU/HPU/...) override the model dtype.
            if device.prefers_bf16():
                amp_dtype = torch.bfloat16
            # Fall back to fp32 (and disable amp) when bf16 is unsupported.
            if amp_dtype == torch.bfloat16 and not device.supports_bf16():
                self.amp = False
                amp_dtype = torch.float32
                logger.warning(
                    f"amp is set to FALSE as the current {self.device} device does not support the 'bf16' data type."
                )
            self.amp_dtype = amp_dtype
        if self.model.dtype != self.amp_dtype:
            self.model = self.model.to(self.amp_dtype)

    def apply_patches(self, formats):
        """Apply format-specific model structure patches.

        Must be called after formats are resolved (list[OutputFormat]) and before
        BaseQuantizer.post_init() so that configure_layer_config() operates on the
        final model structure (post update_module).  Eliminates the need for a
        subsequent refresh_quantizer_for_initialized_model() call.
        """
        # It is best to modify the model structure in the quantize function and check the format,
        # because it may cause the gguf format to not be exported normally.
        self._patch_custom_moe_modules()
        self.model = update_module(
            self.model, formats=formats, trust_remote_code=self.trust_remote_code, cleanup_original=False
        )
        self.model = _handle_special_model(self.model)

        # Temporary names must be assigned after handle_moe_model;
        # placing them earlier would cause them to be removed when the module is replaced.
        for n, m in self.model.named_modules():
            m.global_name = n

        if self.amp and self.model.dtype != self.amp_dtype:
            self.model = self.model.to(self.amp_dtype)

        self._init_model = True
        self._is_initialized = True

    def replace_forward(self, register_hook):
        """Replaces the forward function.
        register_hook(layer_name, module, hook_handles)
        """
        assert self._init_model, "should load and initialize model first"
        hook_handles = []

        for n, m in self.model.named_modules():
            register_hook(n, m, hook_handles)

        self.hook_handles = hook_handles

    def recover_forward(self, restore_positional_wrapper=None):
        """Recovers the forward function.

        Args:
            restore_positional_wrapper: If True, restores forward to the wrapped version
                (needed for LLM calibration where positional wrapper is used during quantization).
                If False, restores to the true original forward (needed for diffusion).
                If None (default), auto-detects: uses False for diffusion models.
        """
        assert self._init_model, "should load and initialize model first"

        # Auto-detect for diffusion: when _true_orig_forward is present (set by
        # CalibCompressor._replace_forward), we are in new-arch diffusion mode where
        # the positional wrapper must be fully removed after caching.
        if restore_positional_wrapper is None:
            restore_positional_wrapper = not getattr(self, "_has_true_orig_forward_set", False)
            if not restore_positional_wrapper:
                logger.debug("recover_forward: auto-detected diffusion mode, stripping positional wrapper")

        for n, m in self.model.named_modules():
            if hasattr(m, "orig_forward"):
                true_orig = getattr(m, "_true_orig_forward", m.orig_forward)
                if restore_positional_wrapper:
                    # Restore orig_forward so that any wrapper (e.g. from
                    # wrap_block_forward_positional_to_kwargs) can still access it.
                    # The wrapper holds a closure reference to orig_forward.
                    m.forward = getattr(m, "_wrapped_forward_before_replace", m.orig_forward)
                    m.orig_forward = true_orig
                else:
                    # Full recovery: restore the true original forward.  Used for diffusion
                    # where the positional wrapper must be fully removed after caching.
                    m.forward = true_orig
                    # Keep _true_orig_forward so the wrapped forward's base_hook can
                    # still call it during quantization tuning.
                    m._true_orig_forward = true_orig
                    delattr(m, "orig_forward")
                    if hasattr(m, "_wrapped_forward_before_replace"):
                        delattr(m, "_wrapped_forward_before_replace")
        for hook_handle in self.hook_handles:
            hook_handle.remove()
        self.hook_handles = []
