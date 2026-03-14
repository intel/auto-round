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

from typing import Any, Callable, Optional, Union

import torch
from packaging import version
from transformers import AutoConfig

from auto_round import envs
from auto_round.compressors.utils import get_shared_keys
from auto_round.context.base import BaseContext
from auto_round.logger import logger
from auto_round.modeling.unfused_moe import apply_model_monkey_patches
from auto_round.special_model_handler import update_module
from auto_round.utils import (
    CpuInfo,
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

__all__ = ["ModelContext"]


class ModelContext(BaseContext):
    _is_initialized = False
    quantized = False

    act_quantize = False

    # model_related
    _model_loaded = False
    _init_model = False
    is_mllm = False
    is_diffusion = False
    is_model_patched = False
    is_moe_model = False

    hook_handles = []

    def __init__(
        self,
        model,
        tokenizer=None,
        platform="hf",
        model_dtype=None,
        trust_remote_code=True,
        amp=True,
        need_calib=True,
        device="cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        if envs.AR_USE_MODELSCOPE:
            platform = "model_scope"
        self._platform = platform
        self._model_dtype = model_dtype
        self._trust_remote_code = trust_remote_code
        self._amp = amp

        self.need_calib = need_calib

    def _load_model(self):
        if is_mllm_model(self.model, platform=self._platform):
            self.is_mllm = True
            if isinstance(self.model, str):
                self.model, self.processor, self.tokenizer, self.image_processor = mllm_load_model(
                    self.model, platform=self._platform, device="cpu", model_dtype=self.model_dtype
                )
        elif is_diffusion_model(self.model):
            self.is_diffusion = True
            self.pipe, self.model = diffusion_load_model(
                self.model, platform=self._platform, device="cpu", model_dtype=self._model_dtype
            )
        elif isinstance(self.model, str):
            config: Optional[AutoConfig] = None
            try:
                config = AutoConfig.from_pretrained(self.model, trust_remote_code=self._trust_remote_code)
            except (OSError, EnvironmentError) as e:
                logger.debug(
                    "Failed to load config via AutoConfig.from_pretrained for %s: %s. "
                    "Proceeding without config-based checks.",
                    self.model,
                    e,
                )

            self.is_model_patched = apply_model_monkey_patches(
                model_name=self.model, trust_remote_code=self._trust_remote_code
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

            self.model, self.tokenizer = llm_load_model(
                self.model,
                platform=self._platform,
                device="cpu",  # always load cpu first
                model_dtype=self._model_dtype,
                trust_remote_code=self._trust_remote_code,
            )
        elif self.tokenizer is None and not self.diffusion and self.need_calib:
            raise ValueError("A tokenizer must be set for non-str model input")

        self._model_loaded = True

    def _set_amp_dtype(self) -> None:
        """Sets the automatic mixed precision (AMP) data type for the model based on the device and configuration."""
        self._amp_dtype = torch.bfloat16
        if self.model.dtype != torch.float32:
            self._amp_dtype = self.model.dtype
        if self.device == "cpu" or "hpu" in self.device:
            self._amp_dtype = torch.bfloat16
        if self._amp:
            if self.device == "cpu" and not CpuInfo().bf16:
                self._amp = False
                self._amp_dtype = torch.float32
                self.model = self.model.to(torch.float32)
                logger.warning(
                    f"amp is set to FALSE as the current {self.device} device does not support the 'bf16' data type."
                )
            else:
                if self.model.dtype != self._amp_dtype:
                    self.model = self.model.to(self._amp_dtype)
        else:
            self._amp_dtype = torch.float32
            self.model = self.model.to(torch.float32)

    def initialize(self, formats):
        # load and handle model
        if not self._model_loaded:
            self._load_model()

        if unsupported_meta_device(self.model):
            raise RuntimeError(
                "AutoRound does not support parameters on meta device. "
                "Please use more GPUs by setting `--device 0,1,2,3` or just place the model on CPU."
            )
        check_and_mark_quantized_module(self.model)
        self.model = self.model.eval()
        self.shared_cache_keys = get_shared_keys(self.model)

        # Important Note! This is not very robust, do NOT rely on it to do high risky thing
        self.is_moe_model = is_moe_model(self.model)

        self._set_amp_dtype()
        if self.act_quantize and self._amp_dtype == torch.float16:
            logger.warning("force to use bf16 to for quantization tuning when enabling activation quantization")
            self._amp_dtype = torch.bfloat16
            if self.model.dtype != torch.bfloat16:  # keep the model's buffer dtype unchanged
                self.model = self.model.to(torch.bfloat16)
        else:
            logger.info(f"using {self.model.dtype} for quantization tuning")

        # It is best to modify the model structure in the quantize function and check the format,
        # because it may cause the gguf format to not be exported normally.
        self.model = update_module(
            self.model, formats=formats, trust_remote_code=self._trust_remote_code, cleanup_original=False
        )

        # Temporary names must be assigned after handle_moe_model;
        # placing them earlier would cause them to be removed when the module is replaced.
        for n, m in self.model.named_modules():
            m.global_name = n

        if self._amp and self.model.dtype != self._amp_dtype:
            self.model = self.model.to(self._amp_dtype)

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

    def recover_forward(self):
        """Recovers the forward function."""
        assert self._init_model, "should load and initialize model first"

        for n, m in self.model.named_modules():
            if hasattr(m, "orig_forward"):
                m.forward = m.orig_forward
                delattr(m, "orig_forward")
        for hook_handle in self.hook_handles:
            hook_handle.remove()
        self.hook_handles = []
