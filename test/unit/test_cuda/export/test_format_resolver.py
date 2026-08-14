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
"""GPU-side fast unit tests for the format-resolution layer.

Covers ``auto_round.export.formats`` (the backward-compat module),
``auto_round.export.formats.base.OutputFormat`` predicates/registration,
``auto_round.export.formats.resolver.resolve_formats``, and the
``auto_round.formats.AutoRoundExportFormat`` enum. None of these need a
real model or GPU.
"""

import pytest

from auto_round.compressors.config_resolution import (
    FormatCompatibilityError,
    resolve_scheme_value,
)
from auto_round.export.formats import (
    OutputFormat,
    resolve_formats,
)


# ==============================================================================
# OutputFormat predicates
# ==============================================================================


class _FakeFormat:
    """Lightweight stand-in that mirrors the bits the predicates actually read."""

    output_format = "auto_round"
    backend = None

    def __init__(self, output_format: str, backend=None):
        self.output_format = output_format
        self.backend = backend

    # Pull predicate methods directly off the real OutputFormat so we don't
    # duplicate their (already exercised) logic in this fake.
    is_gguf = OutputFormat.is_gguf
    is_fake = OutputFormat.is_fake
    is_gptq = OutputFormat.is_gptq
    is_awq = OutputFormat.is_awq
    is_llm_compressor = OutputFormat.is_llm_compressor
    get_backend_name = OutputFormat.get_backend_name


class TestOutputFormatPredicates:
    def test_is_gguf(self):
        assert _FakeFormat("gguf:q4_k_m").is_gguf() is True
        assert _FakeFormat("auto_round").is_gguf() is False

    def test_is_fake(self):
        assert _FakeFormat("fake").is_fake() is True
        assert _FakeFormat("auto_round").is_fake() is False

    def test_is_gptq(self):
        assert _FakeFormat("auto_gptq").is_gptq() is True
        assert _FakeFormat("auto_round").is_gptq() is False

    def test_is_gptq_propagates_via_backend(self):
        inner = _FakeFormat("auto_gptq")
        outer = _FakeFormat("auto_round:llm_compressor:auto_gptq", backend=inner)
        assert outer.is_gptq() is True

    def test_is_awq(self):
        assert _FakeFormat("auto_awq").is_awq() is True
        assert _FakeFormat("auto_round").is_awq() is False

    def test_is_llm_compressor(self):
        assert _FakeFormat("llm_compressor").is_llm_compressor() is True
        assert _FakeFormat("auto_round").is_llm_compressor() is False

    def test_get_backend_name_no_backend(self):
        assert _FakeFormat("auto_round").get_backend_name() == "auto_round"

    def test_get_backend_name_with_backend(self):
        inner = _FakeFormat("fp8_static")
        inner.backend = None
        outer = _FakeFormat("auto_round:fp8_static", backend=inner)
        assert outer.get_backend_name() == "fp8_static"


# ==============================================================================
# OutputFormat.register decorator
# ==============================================================================


class TestOutputFormatRegister:
    def test_register_adds_to_format_list(self):
        @OutputFormat.register("_test_register_xyz_")
        class _StubFormat(OutputFormat):
            format_name = "_test_register_xyz_"

        try:
            assert "_test_register_xyz_" in OutputFormat._format_list
            assert OutputFormat._format_list["_test_register_xyz_"] is _StubFormat
        finally:
            OutputFormat._format_list.pop("_test_register_xyz_", None)

    def test_register_multiple_names(self):
        @OutputFormat.register("_a_", "_b_")
        class _DualFormat(OutputFormat):
            format_name = "_dual_format_"

        try:
            assert "_a_" in OutputFormat._format_list
            assert "_b_" in OutputFormat._format_list
        finally:
            OutputFormat._format_list.pop("_a_", None)
            OutputFormat._format_list.pop("_b_", None)

    def test_register_without_names_raises(self):
        with pytest.raises(AssertionError):
            OutputFormat.register()


# ==============================================================================
# SUPPORTED_FORMATS presence
# ==============================================================================


class TestSupportedFormatsRegistry:
    def test_supported_formats_is_nonempty_set(self):
        assert isinstance(OutputFormat._format_list, dict)
        assert len(OutputFormat._format_list) > 0

    def test_fake_format_registered(self):
        assert "fake" in OutputFormat._format_list

    def test_auto_round_format_registered(self):
        assert "auto_round" in OutputFormat._format_list


class TestGetSupportMatrix:
    def test_returns_string(self):
        s = OutputFormat.get_support_matrix()
        assert isinstance(s, str)
        assert "support scheme" in s
        assert "fake" in s


# ==============================================================================
# Legacy AutoRoundExportFormat enum (auto_round.formats)
# ==============================================================================


class TestAutoRoundExportFormat:
    def test_enum_values(self):
        from auto_round.formats import AutoRoundExportFormat

        assert AutoRoundExportFormat.FP8_STATIC.value == "fp8_static"
        assert AutoRoundExportFormat.MXFP4.value == "mxfp4"
        assert AutoRoundExportFormat.NVFP4.value == "nvfp4"

    def test_inherits_from_str(self):
        from auto_round.formats import AutoRoundExportFormat

        # str-Enum mixin means we can compare directly to strings
        assert AutoRoundExportFormat.FP8 == "fp8"
        assert AutoRoundExportFormat.INT8 == "int8_w8a8"


# ==============================================================================
# resolve_formats()
# ==============================================================================


class TestResolveFormats:
    def test_resolve_formats_does_not_mutate_inputs(self):
        layer_config = {"layer": {"bits": 4}}
        scheme = resolve_scheme_value("W4A16", {})

        result = resolve_formats(scheme, format="auto_round", layer_config=layer_config, model=None)

        layer_config["layer"]["bits"] = 8
        assert scheme.value.bits == 4
        assert result.scheme.value.bits == 4
        assert result.layer_config_patch["layer"]["bits"] == 4

    def test_resolve_formats_rejects_gguf_with_real_companion(self):
        scheme = resolve_scheme_value("W4A16", {})

        with pytest.raises(FormatCompatibilityError):
            resolve_formats(scheme, format="gguf:q4_k_m,auto_round", model=None)

    def test_resolve_formats_allows_gguf_with_fake_companion(self):
        scheme = resolve_scheme_value("W4A16", {})

        result = resolve_formats(scheme, format="gguf:q4_k_m,fake", model=None)

        assert sorted(f.output_format for f in result.formats) == ["fake", "gguf"]
        assert any(f.is_gguf() for f in result.formats)
        assert any(f.is_fake() for f in result.formats)

    def test_resolve_formats_raises_on_unknown_format(self):
        scheme = resolve_scheme_value("W4A16", {})

        with pytest.raises(ValueError):
            resolve_formats(scheme, format="totally_made_up_format", model=None)
