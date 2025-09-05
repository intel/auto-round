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

from typing import Any, Callable, Dict, List, Union

import torch

from auto_round.export.export_to_gguf.config import GGUF_CONFIG, GGUF_INNER_CONFIG, ModelType
from auto_round.quantizers.base import BaseQuantizer, QuantizerType
from auto_round.utils import (
    _gguf_args_check,
    check_to_quantized,
    get_layer_config_by_gguf_format,
    get_lm_head_name,
    get_module,
    logger,
)


class ModeQuantizer(BaseQuantizer):
    quantizer_type = QuantizerType.DATA_TYPE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@BaseQuantizer.register("gguf")
class GGUFQuantizer(ModeQuantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _apply_config_to_layer(
        self,
        layer_name: str,
        config: dict[str, Any],
        check_fixed_by_user: bool = False,
    ) -> None:
        """Applies GGUF quantization configuration to a given layer.

        Args:
            layer_name (str): Name of the layer to configure.
            config (dict[str, Any]): GGUF layer configuration.
            check_fixed_by_user (bool): If True, preserve user-defined settings.
        """
        act_bits: int = 16
        scale_dtype: Any = self.scale_dtype
        keys: list[str] = ["bits", "group_size", "super_bits", "super_group_size", "data_type", "sym"]

        self.layer_config[layer_name] = self.layer_config.get(layer_name, {})

        for key in keys:
            if (
                key in self.layer_config[layer_name]
                and check_fixed_by_user
                # and self.layer_config[layer_name].get("fixed_by_user", False)
            ):
                continue
            self.layer_config[layer_name][key] = config.get(key)
            setattr(get_module(self.model, layer_name), key, config.get(key))

        self.layer_config[layer_name]["act_bits"] = act_bits
        self.layer_config[layer_name]["scale_dtype"] = scale_dtype
        setattr(get_module(self.model, layer_name), "act_bits", act_bits)
        setattr(get_module(self.model, layer_name), "scale_dtype", scale_dtype)

    def _check_compatibility(self) -> None:
        """Checks compatibility of the configurations and model."""
        super()._check_compatibility()
        has_gguf = False
        if hasattr(self, "formats"):
            has_besides_gguf = False
            for format_ in self.formats:
                if "gguf" in format_:
                    has_gguf = True
                elif format_ != "fake":
                    has_besides_gguf = True
            if has_gguf and has_besides_gguf:
                raise ValueError("GGUF format is not compatible with other formats, please choose only one of them")
            if has_gguf and self.iters != 0 and self.bits != 3:
                logger.warning(
                    "`iters=0` is recommended when exporting to GGUF format except for bits 3,"
                    " as we have optimized the RTN method for this case."
                    " We are likely to release new algorithm for certain configurations in the future."
                )

            only_gguf = True
            for format_ in self.formats:
                if not ("gguf" in format_ or "fake" in format_):
                    only_gguf = False
                    break
            if len(self.formats) == 1 and self.formats[0] == "fake":
                only_gguf = False
            if only_gguf:
                self.layer_config, gguf_format_config = get_layer_config_by_gguf_format(
                    self.layer_config, self.formats, self.model, model_type=ModelType.TEXT
                )
                if self.vlm:
                    self.layer_config, gguf_format_config = get_layer_config_by_gguf_format(
                        self.layer_config, self.formats, self.model, model_type=ModelType.MMPROJ
                    )

    def _check_need_to_quantize_lm_head_embedding(self) -> bool:
        """Checks if LM head and embedding layers need quantization for GGUF format.

        This function inspects the current model's formats and determines whether
        it needs to apply quantization settings to the embedding and LM head layers.
        The function modifies `self.layer_config` in-place and updates the model modules.

        Returns:
            bool: True if the LM head needs quantization, otherwise False.

        Raises:
            NotImplementedError: If multiple non-fake GGUF formats are specified.
        """
        gguf_scheme = False
        if isinstance(self.scheme, str) and "gguf" in self.scheme.lower():
            gguf_scheme = True

        if not hasattr(self, "formats") and not gguf_scheme:
            return False

        has_gguf: bool = gguf_scheme or any("gguf" in fmt for fmt in self.formats)
        if not has_gguf:
            return False
        if hasattr(self, "formats"):
            formats: list[str] = [fmt for fmt in self.formats if "fake" not in fmt]
            if not (len(formats) == 1 and "gguf" in formats[0]):
                raise NotImplementedError("Only one GGUF format can be set at a time.")
            target_format: str = formats[0]

        else:
            target_format = self.scheme.lower()

        tie_word_embeddings: bool = getattr(getattr(self.model, "config", None), "tie_word_embeddings", True)
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                key: str = "lm_head" if tie_word_embeddings else "embedding"
                config: dict[str, Any] = GGUF_INNER_CONFIG[GGUF_CONFIG[target_format][key]]
                self._apply_config_to_layer(name, config, True)

        if not tie_word_embeddings:
            lm_head_name: str = get_lm_head_name(self.model)
            config: dict[str, Any] = GGUF_CONFIG[GGUF_CONFIG[target_format]["lm_head"]]
            check_fixed_by_user = (
                self.layer_config[lm_head_name].get("fixed_by_user", False)
                if lm_head_name in self.layer_config
                else None
            )
            self._apply_config_to_layer(lm_head_name, config, check_fixed_by_user=check_fixed_by_user)
            return True

        return False

    def _set_layerwise_config(self, layer_config):
        has_qlayer_outside_block = super()._set_layerwise_config(layer_config)
        need_to_quantize_lm_head = self._check_need_to_quantize_lm_head_embedding()
        if need_to_quantize_lm_head:
            has_qlayer_outside_block = True
        return has_qlayer_outside_block

    def _pack_layer(self, module):
        from auto_round.export.export_to_gguf.export import pack_gguf_layer

        for _, tmp_m in module.named_modules():
            if not (hasattr(tmp_m, "bits") and check_to_quantized(tmp_m)):
                continue

            output_dir = self._get_save_folder_name(self.formats[0])
            model_type = ModelType.MMPROJ if self.vlm else ModelType.TEXT
            pack_gguf_layer(
                tmp_m.tmp_name,
                self.model,
                self.formats[0],
                output_dir,
                self.layer_config,
                self.tokenizer,
                processor=self.processor if hasattr(self, "processor") else None,
                image_processor=self.image_processor if hasattr(self, "image_processor") else None,
                model_type=model_type,
            )

    def _parse_format_to_list(self, format: str) -> list:
        """Parses the format string into a list of formats.

        This method checks the requested format(s) against the model's
        quantization settings and adjusts them if necessary. It ensures that
        the formats are compatible with the model's data type, bit width,
        and activation quantization settings.

        Args:
            format (str): The requested format(s) for quantization, separated by commas.

        Returns:
            list: A list of validated and updated formats.
        """
        _gguf_args_check(self, format, model_type=ModelType.TEXT)
        if self.vlm:
            _gguf_args_check(self, format, model_type=ModelType.MMPROJ)

        formats = format.replace("q*_", f"q{self.bits}_").replace(" ", "").split(",")
        if self.scale_dtype != torch.float32:
            only_gguf = True
            for format_ in formats:
                if not ("gguf" in format_ or "fake" in format_):
                    only_gguf = False
                    break
            if len(formats) == 1 and "fake" == formats[0]:
                only_gguf = False
            if only_gguf:
                self.scale_dtype = torch.float32
                logger.info("change `scale_dtype` to `torch.float32`")
        formats = super()._parse_format_to_list(format)
        return formats


@BaseQuantizer.register("mxfp")
class MXFPQuantizer(ModeQuantizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
