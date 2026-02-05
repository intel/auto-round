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
# Copyright (c) 2023 Intel Corporation
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


def check_compressed_tensors_supported():  # pragma: no cover
    try:
        import compressed_tensors  # noqa: F401

        return True
    except ImportError:
        logger.error(
            "Please install compressed-tensors via 'pip install compressed-tensors'" " to save as llm-compressor format"
        )
        exit(-1)


if check_compressed_tensors_supported():
    from compressed_tensors.quantization import (  # pylint: disable=E0401
        QuantizationConfig,
        QuantizationScheme,
        QuantizationStatus,
        is_preset_scheme,
        preset_name_to_scheme,
    )


# please refer to https://github.com/vllm-project/llm-compressor/blob/
# 29f4d5644b48e9c8ebb7e36d5be9f7c92747ceb7/src/llmcompressor/modifiers/quantization/quantization/mixin.py#L168
def initialize_quantization(scheme, targets=["Linear"], config_groups=None, kv_cache_scheme=None, ignore=["lm_head"]):
    """
    Attach quantization schemes to modules in the model and initialize the quantization config
    """

    # apply scheme and status to model
    scheme = scheme
    targets = targets
    config_groups = config_groups
    kv_cache_scheme = kv_cache_scheme
    ignore = ignore
    check_compressed_tensors_supported()
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
