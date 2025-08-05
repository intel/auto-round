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

from auto_round.utils import logger

try:
    from compressed_tensors.quantization import (
        QuantizationConfig,
        QuantizationScheme,
        QuantizationStatus,
        is_preset_scheme,
        preset_name_to_scheme,
    )
except Exception as e:
    logger.warning(
        "Please install compressed-tensors via 'pip install compressed-tensors'"
        " to save llm-compressor format quantization config"
    )


def initialize_quantization(
    scheme, targets=["Linear"], config_groups=None, kv_cache_scheme=None, ignore=["lm_head"]
) -> QuantizationConfig:
    """
    Attach quantization schemes and observers to modules in the model according to
    the quantization config specified on this modifier

    :param model: model to attach schemes and observers to
    """

    # apply scheme and status to model
    scheme = scheme
    targets = targets
    config_groups = config_groups
    kv_cache_scheme = kv_cache_scheme
    ignore = ignore

    if scheme is not None and config_groups is not None:
        raise ValueError("Please specify either `scheme` or `config_groups`")

    if scheme is not None:
        # takes precedence over config_groups

        if isinstance(scheme, str) and is_preset_scheme(scheme):
            # attach targets to scheme
            scheme = {scheme: targets}

        config_groups = {}
        for idx, key in enumerate(scheme.keys()):
            if is_preset_scheme(key):
                scheme = preset_name_to_scheme(key, scheme[key])
            else:
                scheme = QuantizationScheme.model_validate({"targets": scheme[key], **scheme})

            group_name = f"group_{idx}"
            config_groups[group_name] = scheme

    if config_groups is None or len(config_groups) == 0:
        default_quant_scheme = QuantizationScheme(targets=targets)
        config_groups = {"group_0": default_quant_scheme}

    return QuantizationConfig(
        config_groups=config_groups,
        kv_cache_scheme=kv_cache_scheme,
        quantization_status=QuantizationStatus.COMPRESSED,
        ignore=ignore,
    )
