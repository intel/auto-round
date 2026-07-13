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
"""Tests for the format-resolution helpers in ``auto_round/formats.py``."""

import pytest
import torch


# ---------------------------------------------------------------------------
# AutoRoundExportFormat enum
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# OutputFormat predicates (no real model required)
# ---------------------------------------------------------------------------
class TestOutputFormatPredicates:
    """Direct testing of the ``is_*`` predicates is tricky because
    ``OutputFormat`` is an ABC with abstract ``pack_layer`` / ``save_quantized``.

    We use a lightweight fake that inherits only from ``object`` and copies the
    relevant attributes/methods.  This keeps the tests focused on the
    predicates themselves, not on the concrete subclasses' implementation.
    """

    class _FakeFormat:
        # Mirror the bits the predicates actually read
        output_format = "auto_round"
        backend = None

        def __init__(self, output_format: str, backend=None):
            self.output_format = output_format
            self.backend = backend

        # Pull the predicate methods directly off OutputFormat to avoid
        # duplicating their (already exercised) logic in the fake.
        from auto_round.formats import OutputFormat as _of

        is_gguf = _of.is_gguf
        is_fake = _of.is_fake
        is_gptq = _of.is_gptq
        is_awq = _of.is_awq
        is_llm_compressor = _of.is_llm_compressor
        get_backend_name = _of.get_backend_name

    def test_is_gguf(self):
        fmt = self._FakeFormat("gguf:q4_k_m")
        assert fmt.is_gguf() is True
        fmt2 = self._FakeFormat("auto_round")
        assert fmt2.is_gguf() is False

    def test_is_fake(self):
        fmt = self._FakeFormat("fake")
        assert fmt.is_fake() is True
        fmt2 = self._FakeFormat("auto_round")
        assert fmt2.is_fake() is False

    def test_is_gptq(self):
        fmt = self._FakeFormat("auto_gptq")
        assert fmt.is_gptq() is True
        fmt2 = self._FakeFormat("auto_round")
        assert fmt2.is_gptq() is False

    def test_is_gptq_propagates_via_backend(self):
        inner = self._FakeFormat("auto_gptq")
        outer = self._FakeFormat("auto_round:llm_compressor:auto_gptq", backend=inner)
        assert outer.is_gptq() is True

    def test_is_awq(self):
        fmt = self._FakeFormat("auto_awq")
        assert fmt.is_awq() is True
        fmt2 = self._FakeFormat("auto_round")
        assert fmt2.is_awq() is False

    def test_is_llm_compressor(self):
        fmt = self._FakeFormat("llm_compressor")
        assert fmt.is_llm_compressor() is True
        fmt2 = self._FakeFormat("auto_round")
        assert fmt2.is_llm_compressor() is False

    def test_get_backend_name_no_backend(self):
        fmt = self._FakeFormat("auto_round")
        assert fmt.get_backend_name() == "auto_round"

    def test_get_backend_name_with_backend(self):
        inner = self._FakeFormat("fp8_static")
        inner.backend = None
        outer = self._FakeFormat("auto_round:fp8_static", backend=inner)
        assert outer.get_backend_name() == "fp8_static"


# ---------------------------------------------------------------------------
# OutputFormat.register decorator
# ---------------------------------------------------------------------------
class TestOutputFormatRegister:
    def test_register_adds_to_format_list(self):
        from auto_round.formats import OutputFormat

        @OutputFormat.register("_test_register_xyz_")
        class _StubFormat(OutputFormat):
            format_name = "_test_register_xyz_"

        try:
            assert "_test_register_xyz_" in OutputFormat._format_list
            assert OutputFormat._format_list["_test_register_xyz_"] is _StubFormat
        finally:
            OutputFormat._format_list.pop("_test_register_xyz_", None)

    def test_register_multiple_names(self):
        from auto_round.formats import OutputFormat

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
        from auto_round.formats import OutputFormat
        with pytest.raises(AssertionError):
            OutputFormat.register()


# ---------------------------------------------------------------------------
# is_support_scheme / check_scheme_args
# ---------------------------------------------------------------------------
class TestSchemeCompatibility:
    def _make_format(self, support_schemes=None):
        from auto_round.formats import OutputFormat

        class _StubFormat(OutputFormat):
            def pack_layer(self, *a, **kw): pass
            def save_quantized(self, *a, **kw): pass

        if support_schemes is not None:
            # is_support_scheme is a classmethod that reads cls.support_schemes
            _StubFormat.support_schemes = support_schemes

        obj = _StubFormat.__new__(_StubFormat)
        obj.output_format = "stub"
        obj.backend = None
        return obj

    def test_is_support_scheme_string_match(self):
        fmt = self._make_format(support_schemes=["W4A16", "MXFP4"])
        assert fmt.is_support_scheme("W4A16") is True
        assert fmt.is_support_scheme("mxfp4") is True  # upper-cased
        assert fmt.is_support_scheme("UNKNOWN") is False

    def test_is_support_scheme_unknown_scheme_returns_false(self):
        fmt = self._make_format(support_schemes=["W4A16"])
        assert fmt.is_support_scheme("not_a_scheme") is False

    def test_is_support_scheme_quantization_scheme_instance(self):
        from auto_round.schemes import QuantizationScheme

        fmt = self._make_format(support_schemes=[])
        # The default check_scheme_args returns True for any QuantizationScheme
        scheme = QuantizationScheme(bits=4, group_size=128, sym=True, data_type="int")
        assert fmt.is_support_scheme(scheme) is True

    def test_check_scheme_args_default_true(self):
        fmt = self._make_format()
        # The base class default is True
        assert fmt.check_scheme_args(None) is True


# ---------------------------------------------------------------------------
# SUPPORTED_FORMATS presence
# ---------------------------------------------------------------------------
class TestSupportedFormatsRegistry:
    def test_supported_formats_is_nonempty_set(self):
        from auto_round.formats import OutputFormat
        assert isinstance(OutputFormat._format_list, dict)
        assert len(OutputFormat._format_list) > 0

    def test_fake_format_registered(self):
        from auto_round.formats import OutputFormat
        assert "fake" in OutputFormat._format_list

    def test_auto_round_format_registered(self):
        from auto_round.formats import OutputFormat
        # auto_round should always be a registered format
        assert "auto_round" in OutputFormat._format_list


# ---------------------------------------------------------------------------
# get_support_matrix (just verify it returns a string without crashing)
# ---------------------------------------------------------------------------
class TestGetSupportMatrix:
    def test_returns_string(self):
        from auto_round.formats import OutputFormat
        s = OutputFormat.get_support_matrix()
        assert isinstance(s, str)
        assert "support scheme" in s
        # "fake" should appear in the matrix
        assert "fake" in s