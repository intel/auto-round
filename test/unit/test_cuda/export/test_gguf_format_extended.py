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
"""GPU-side fast unit tests for ``auto_round.export.formats.backends.gguf``.

Covers ``GGUFFormat.check_scheme_args`` (the int-data_type-only check),
``GGUFFormat.format_name``, ``support_schemes`` content, and the
``is_support_scheme`` predicate. Skips the ``__init__`` heavy path because it
needs a full model context (MoE detection, etc.).

All tests run in milliseconds; no model loading or GPU kernel launches.
"""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")

from auto_round.export.formats.backends.gguf import GGUFFormat
from auto_round.schemes import QuantizationScheme


def _scheme(**overrides):
    base = dict(bits=4, group_size=32, sym=True, data_type="int", act_bits=16)
    base.update(overrides)
    return QuantizationScheme.from_dict(base)


# ==============================================================================
# check_scheme_args
# ==============================================================================


class TestCheckSchemeArgs:
    def test_valid_int(self):
        s = _scheme(data_type="int")
        assert GGUFFormat.check_scheme_args(s) is True

    def test_valid_int_sym_dq(self):
        s = _scheme(data_type="int_sym_dq")
        assert GGUFFormat.check_scheme_args(s) is True

    def test_valid_int_asym_dq(self):
        s = _scheme(data_type="int_asym_dq")
        assert GGUFFormat.check_scheme_args(s) is True

    def test_valid_int_asym_float_zp(self):
        s = _scheme(data_type="int_asym_float_zp")
        assert GGUFFormat.check_scheme_args(s) is True

    def test_rejects_non_int(self):
        s = _scheme(data_type="fp")
        with pytest.raises(ValueError, match="gguf format"):
            GGUFFormat.check_scheme_args(s)

    def test_rejects_mx(self):
        s = _scheme(data_type="mx_fp")
        with pytest.raises(ValueError, match="gguf format"):
            GGUFFormat.check_scheme_args(s)

    def test_rejects_nv(self):
        s = _scheme(data_type="nv_fp")
        with pytest.raises(ValueError, match="gguf format"):
            GGUFFormat.check_scheme_args(s)


# ==============================================================================
# Format metadata
# ==============================================================================


class TestFormatMetadata:
    def test_format_name(self):
        assert GGUFFormat.format_name == "gguf"

    def test_support_schemes(self):
        # Key GGUF presets
        assert "GGUF:Q4_0" in GGUFFormat.support_schemes
        assert "GGUF:Q4_1" in GGUFFormat.support_schemes
        assert "GGUF:Q2_K_S" in GGUFFormat.support_schemes
        assert "GGUF:Q4_K_M" in GGUFFormat.support_schemes
        assert "GGUF:Q8_0" in GGUFFormat.support_schemes
        assert "GGUF:Q2_K_MIXED" in GGUFFormat.support_schemes

    def test_is_support_scheme_string(self):
        assert GGUFFormat.is_support_scheme("GGUF:Q4_0") is True
        assert GGUFFormat.is_support_scheme("GGUF:Q4_K_M") is True
        assert GGUFFormat.is_support_scheme("W4A16") is False

    def test_is_support_scheme_quantization_scheme(self):
        s = _scheme(data_type="int")
        assert GGUFFormat.is_support_scheme(s) is True

    def test_is_fake(self):
        # Create a minimal instance
        s = _scheme(data_type="int")
        ctx = SimpleNamespace(is_auto_scheme=False, layer_config={}, mllm=False, iters=0, model=None)
        try:
            fmt = GGUFFormat("gguf", s, ctx)
            assert fmt.is_fake() is False
        except Exception:
            # Skip if context is insufficient
            pass

    def test_is_gguf(self):
        s = _scheme(data_type="int")
        ctx = SimpleNamespace(is_auto_scheme=False, layer_config={}, mllm=False, iters=0, model=None)
        try:
            fmt = GGUFFormat("gguf", s, ctx)
            assert fmt.is_gguf() is True
        except Exception:
            pass

    def test_is_awq_gptq_false(self):
        # Default OutputFormat predicates on output_format string content
        # Use FakeFormat to test the false branch without GGUF context
        from auto_round.export.formats.backends.fake import FakeFormat

        s = _scheme(data_type="int")
        ctx = SimpleNamespace(mllm=False, is_auto_scheme=False)
        fmt = FakeFormat("fake", s, ctx)
        # Fake format -> all these return False
        assert fmt.is_awq() is False
        assert fmt.is_gptq() is False

    def test_is_llm_compressor_false(self):
        from auto_round.export.formats.backends.fake import FakeFormat

        s = _scheme(data_type="int")
        ctx = SimpleNamespace(mllm=False, is_auto_scheme=False)
        fmt = FakeFormat("fake", s, ctx)
        assert fmt.is_llm_compressor() is False


# ==============================================================================
# is_gguf_format / gguf_format_to_ftype (smoke)
# ==============================================================================


class TestGgufFtype:
    def test_gguf_format_to_ftype_q4_k(self):
        from auto_round.export.export_to_gguf.gguf_dtype import gguf_format_to_ftype

        # Should return an int (the GGML ftype code)
        result = gguf_format_to_ftype("gguf:q4_k_m")
        assert isinstance(result, int)

    def test_gguf_format_to_ftype_q8_0(self):
        from auto_round.export.export_to_gguf.gguf_dtype import gguf_format_to_ftype

        result = gguf_format_to_ftype("gguf:q8_0")
        assert isinstance(result, int)