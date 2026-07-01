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

import copy
from typing import Any


class VLLMMixin:
    """vLLM-specific initialization mixin.

    This mixin threads vLLM loading options into ``BaseCompressor`` so that
    ``ModelContext`` can initialize model/tokenizer through ``vllm_load_model``.
    It also overrides ``configure_layer_config`` to recognise vLLM
    ``LinearBase`` subclasses (e.g. ``QKVParallelLinear``) which are missed by
    the exact-type matching used in the mainline ``set_layer_config``.
    """

    def __init__(
        self,
        *args,
        enable_vllm_loading: bool = False,
        vllm_model_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        self.enable_vllm_loading = enable_vllm_loading
        self.vllm_model_kwargs = dict(vllm_model_kwargs or {})

        super().__init__(
            *args,
            enable_vllm_loading=self.enable_vllm_loading,
            vllm_model_kwargs=self.vllm_model_kwargs,
            **kwargs,
        )

    def _get_calibrator_kind(self) -> str:
        """Select the vLLM calibration strategy.

        Importing ``VLLMCalibrator`` here is intentional: the import
        triggers ``@register_calibrator("vllm")`` so the registry entry
        exists before ``get_calibrator("vllm")`` is called.
        """
        from auto_round.compressors.vllm.vllm_calibrator import VLLMCalibrator  # noqa: F401

        return "vllm"

    def configure_layer_config(self, enable_gguf_official_mixed: bool | None = True) -> None:
        """Extend base layer config with vLLM LinearBase layers.

        The mainline ``set_layer_config`` uses exact ``type(m) in
        supported_types`` matching, which misses vLLM layers such as
        ``QKVParallelLinear`` (subclass of ``LinearBase``, not registered in
        ``SUPPORTED_LAYER_TYPES``).  After the base implementation runs, this
        override iterates the model and adds any ``LinearBase`` instances that
        were skipped, using the global scheme derived from the resolved scheme
        attributes on ``self``.
        """
        super().configure_layer_config(enable_gguf_official_mixed)

        try:
            from vllm.model_executor.layers.linear import LinearBase
        except ImportError:
            return

        model = self.model_context.model
        bits = getattr(self, "bits", None)
        if bits is None or bits >= 16:
            return  # nothing to quantize

        # Build a global scheme dict from the resolved scheme attributes.
        scheme_keys = (
            "bits",
            "group_size",
            "data_type",
            "sym",
            "scale_dtype",
            "act_bits",
            "act_group_size",
            "act_data_type",
            "act_sym",
            "act_dynamic",
        )
        global_cfg: dict[str, Any] = {}
        for key in scheme_keys:
            val = getattr(self, key, None)
            if val is not None:
                global_cfg[key] = val

        # Identify block names so we can flag outside-block layers.
        from auto_round.utils import get_block_names

        quant_block_list = getattr(self, "quant_block_list", None)
        block_names_flat: set[str] = set(
            name for group in (quant_block_list or get_block_names(model)) for name in group
        )

        added: list[str] = []
        for n, m in model.named_modules():
            if not isinstance(m, LinearBase):
                continue
            if n in self.layer_config:
                continue  # already handled by super()

            cfg = copy.deepcopy(global_cfg)
            cfg["in_blocks"] = any(n == bn or n.startswith(f"{bn}.") for bn in block_names_flat)
            if not cfg["in_blocks"]:
                self.has_qlayer_outside_block = True

            self.layer_config[n] = cfg
            # Dispatch: set scheme attributes directly on the module so that
            # check_to_quantized(m) and quantizer.quantize_layer() work.
            for attr, value in cfg.items():
                setattr(m, attr, value)
            added.append(n)

        if added:
            from auto_round.utils import logger

            logger.info(
                f"VLLMMixin: added {len(added)} vLLM LinearBase layer(s) to " f"layer_config (first: {added[0]})"
            )
