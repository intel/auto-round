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

import subprocess
import sys

import pytest
import torch

from auto_round.compressors.config_resolution import resolve_scheme_value
from auto_round.export.formats import OutputFormat, resolve_formats
from auto_round.export.formats.gguf.config import GGUF_CONFIG
from auto_round.utils import SUPPORTED_FORMATS


@pytest.mark.parametrize(
    "name,scheme,class_name,backend,immediate",
    [
        ("auto_round", "W4A16", "AutoRoundFormat", "auto_round:auto_gptq", True),
        ("auto_round:auto_gptq", "W4A16", "AutoRoundFormat", "auto_round:auto_gptq", True),
        ("auto_round:gptqmodel", "W4A16", "AutoRoundFormat", "auto_round:gptqmodel", True),
        ("auto_round:auto_awq", "W4A16", "AutoRoundFormat", "auto_round:auto_awq", True),
        ("auto_round:llm_compressor", "W4A16", "AutoRoundFormat", "auto_round:llm_compressor:wint_a16", True),
        ("fp8", "FP8_BLOCK", "FP8Format", "fp8", True),
        ("auto_round:fp8", "FP8_BLOCK", "AutoRoundFormat", "auto_round:fp8", True),
        ("mlx", "W4A16", "MLXFormat", "mlx", True),
        ("auto_round:mlx", "W4A16", "AutoRoundFormat", "mlx", True),
        ("auto_gptq", "W4A16", "AutoGPTQFormat", "auto_gptq", True),
        ("auto_awq", "W4A16", "AutoAWQFormat", "auto_awq", True),
        ("llm_compressor", "W4A16", "LLMCompressorFormat", "llm_compressor:wint_a16", True),
        ("gguf:q4_0", "W4A16", "GGUFFormat", "gguf:q4_0", True),
        ("fake", "W4A16", "FakeFormat", "fake", True),
        ("svdquant_nunchaku", "MXFP4", "SVDQuantNunchakuFormat", "svdquant_nunchaku", False),
    ],
)
def test_builtin_format_contract(name, scheme, class_name, backend, immediate):
    resolution = resolve_formats(resolve_scheme_value(scheme, {}), format=name, model=torch.nn.Module())

    (output_format,) = resolution.formats
    assert type(output_format).__name__ == class_name
    assert output_format.get_backend_name() == backend
    assert output_format.is_supported_immediate_packing() is immediate
    assert output_format.is_supported_immediate_saving() is immediate


def test_public_format_names_match_existing_contract():
    ordinary_names = {
        "auto_round",
        "auto_gptq",
        "auto_awq",
        "auto_round:auto_gptq",
        "auto_round:gptqmodel",
        "auto_round:auto_awq",
        "auto_round:llm_compressor",
        "fake",
        "llm_compressor",
        "fp8",
        "auto_round:fp8",
        "mlx",
        "auto_round:mlx",
        "svdquant_nunchaku",
    }
    assert set(SUPPORTED_FORMATS) == ordinary_names | GGUF_CONFIG.keys()
    assert "gguf" not in SUPPORTED_FORMATS
    assert "gptqmodel" not in SUPPORTED_FORMATS


class _TestFormat(OutputFormat):
    def pack_layer(self, *args, **kwargs):
        pass

    def save_quantized_as_format(self, output_dir, *, model, **kwargs):
        return model


def test_registered_format_is_visible_and_resolvable():
    name = "test_custom_export"
    OutputFormat.register(name)(_TestFormat)
    try:
        assert name in SUPPORTED_FORMATS
        result = resolve_formats(resolve_scheme_value("W4A16", {}), format=name)
        assert isinstance(result.formats[0], _TestFormat)
        assert result.formats[0].get_backend_name() == name
        assert name in list(SUPPORTED_FORMATS)
        assert name in str(SUPPORTED_FORMATS)
    finally:
        OutputFormat._format_list.pop(name, None)
    assert name not in SUPPORTED_FORMATS


def test_registration_can_change_public_visibility():
    name = "test_internal_export"
    try:
        OutputFormat.register(name, public=False)(_TestFormat)
        assert OutputFormat._format_list[name] is _TestFormat
        assert name not in SUPPORTED_FORMATS
        with pytest.raises(ValueError, match="is not supported"):
            resolve_formats(resolve_scheme_value("W4A16", {}), format=name)

        OutputFormat.register(name)(_TestFormat)
        assert name in SUPPORTED_FORMATS

        OutputFormat.register(name, public=False)(_TestFormat)
        assert name not in SUPPORTED_FORMATS
    finally:
        OutputFormat._format_list.pop(name, None)
        OutputFormat._non_public_names.discard(name)


def test_supported_formats_in_fresh_process():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from auto_round.utils import SUPPORTED_FORMATS; "
                "assert 'auto_round' in SUPPORTED_FORMATS; "
                "assert 'gguf:q4_0' in SUPPORTED_FORMATS; "
                "assert 'gptqmodel' not in SUPPORTED_FORMATS"
            ),
        ],
        check=True,
        timeout=60,
    )


def test_supported_formats_sequence_and_display():
    names = sorted(SUPPORTED_FORMATS)
    assert SUPPORTED_FORMATS[:] == tuple(names)
    assert SUPPORTED_FORMATS[0] == names[0]
    assert SUPPORTED_FORMATS[-1] == names[-1]
    assert str(SUPPORTED_FORMATS) == f"({', '.join(names)})"


def test_gguf_with_fake_resolves_fake_first():
    result = resolve_formats(resolve_scheme_value("W4A16", {}), format="gguf:q4_0,fake")

    assert [output_format.get_backend_name() for output_format in result.formats] == ["fake", "gguf:q4_0"]


@pytest.mark.parametrize("name,scheme", [("fake", "W4A16"), ("svdquant_nunchaku", "MXFP4")])
def test_noop_packing_preserves_layer(name, scheme):
    model = torch.nn.Sequential(torch.nn.Linear(32, 32))
    layer = model[0]
    weight = layer.weight.detach().clone()
    (output_format,) = resolve_formats(resolve_scheme_value(scheme, {}), format=name).formats

    output_format.pack_layer("0", model, device="cpu")

    assert model[0] is layer
    assert torch.equal(model[0].weight, weight)
