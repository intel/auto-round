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

import inspect

from auto_round.compressors.base import BaseCompressor
from auto_round.compressors.entry import PipelineCompressor


def test_post_init_builds_authoritative_compression_plan(tiny_opt_model_path):
    compressor = PipelineCompressor(
        tiny_opt_model_path,
        scheme="W4A16",
        format="auto_round",
        iters=0,
        nsamples=1,
        seqlen=8,
        dataset=["local calibration sample"],
        low_cpu_mem_usage=False,
    )

    compressor.post_init()

    assert compressor._format_resolution.scheme.preset_name == compressor.compression_plan.scheme.preset_name
    assert compressor._format_resolution.formats == compressor.compression_plan.formats
    assert compressor.compression_plan.scheme.value == compressor.scheme_context
    assert compressor.compression_plan.formats == tuple(compressor.formats)
    assert dict(compressor.compression_plan.layer_config) == compressor.layer_config
    assert dict(compressor.compression_plan.regex_config) == compressor.regex_config
    assert compressor.compression_plan.has_qlayer_outside_block == compressor.has_qlayer_outside_block


def test_legacy_state_views_cannot_mutate_authoritative_plan(tiny_opt_model_path):
    compressor = PipelineCompressor(
        tiny_opt_model_path,
        scheme="W4A16",
        format="auto_round",
        iters=0,
        nsamples=1,
        seqlen=8,
        dataset=["local calibration sample"],
        low_cpu_mem_usage=False,
    )
    compressor.post_init()

    compressor.scheme_context.bits = 2
    compressor.formats.clear()
    compressor.layer_config.clear()
    compressor.regex_config.clear()

    assert compressor.scheme_context.bits == compressor.compression_plan.scheme.value.bits == 4
    assert compressor.formats == list(compressor.compression_plan.formats)
    assert compressor.layer_config == {
        name: dict(config) for name, config in compressor.compression_plan.layer_config.items()
    }
    assert compressor.regex_config == {
        name: dict(config) for name, config in compressor.compression_plan.regex_config.items()
    }


def test_quantizer_sync_reads_authoritative_plan_instead_of_parallel_state():
    source = inspect.getsource(BaseCompressor._build_layer_config)

    assert "self.quantizer.layer_config = self.layer_config" not in source
    assert "self.quantizer.regex_config = self.regex_config" not in source
    assert "self.quantizer.scale_dtype = self.scale_dtype" not in source


def test_format_resolution_does_not_copy_scheme_fields_one_by_one():
    source = inspect.getsource(BaseCompressor._resolve_format_string)

    assert "fields(QuantizationScheme)" not in source


def test_final_layer_config_path_does_not_call_legacy_mutating_adapter():
    source = inspect.getsource(BaseCompressor.configure_layer_config)

    assert "set_layer_config(" not in source


def test_auto_scheme_layer_discovery_uses_pure_resolver():
    source = inspect.getsource(BaseCompressor._gen_auto_scheme)

    assert "set_layer_config(" not in source
