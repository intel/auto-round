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

from auto_round.planning.contracts import CompressionPlan, FormatResolution, LayerConfig, thaw_mapping


def build_compression_plan(
    format_resolution: FormatResolution,
    layer_config: LayerConfig,
    *,
    regex_config: LayerConfig | None = None,
    has_qlayer_outside_block: bool = False,
) -> CompressionPlan:
    """Combine immutable resolution results into the compressor's authoritative plan."""
    merged = thaw_mapping(format_resolution.layer_config_patch)
    for name, config in layer_config.items():
        merged.setdefault(name, {}).update(dict(config))
    return CompressionPlan(
        scheme=format_resolution.scheme,
        formats=format_resolution.formats,
        layer_config=merged,
        regex_config=regex_config or {},
        has_qlayer_outside_block=has_qlayer_outside_block,
        scale_dtype=format_resolution.scale_dtype,
        quant_block_list=format_resolution.quant_block_list,
    )
