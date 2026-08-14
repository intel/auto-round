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
"""GPU-side fast unit tests for ``auto_round.export.formats.backends.llm_compressor``.

Covers the ``LLMCompressorFormat.check_scheme_args`` predicate across all
branches (bits, data_type, super_bits/super_group_size, block_wfp8 tuple
requirements, etc.), the constructor's __init__ validation, and the
``check_and_reset_format`` decision tree (FP8 static path with warnings,
INT8 path, etc.).

All tests run in milliseconds; no model loading or GPU kernel launches.
"""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")

from auto_round.export.formats.backends.llm_compressor import LLMCompressorFormat
from auto_round.schemes import QuantizationScheme


def _scheme(**overrides):
    base = dict(
        bits=8,
        group_size=128,
        sym=True,
        data_type="fp",
        act_bits=16,
        act_group_size=None,
        act_sym=None,
        act_data_type=None,
        act_dynamic=None,
        super_bits=None,
        super_group_size=None,
    )
    base.update(overrides)
    return QuantizationScheme.from_dict(base)


# ==============================================================================
# check_scheme_args
# ==============================================================================


class TestCheckSchemeArgs:
    def test_valid_w4a16(self):
        s = _scheme(bits=4, data_type="int", group_size=128, sym=True, act_bits=16)
        assert LLMCompressorFormat.check_scheme_args(s) is True

    def test_valid_mxfp4(self):
        s = _scheme(bits=4, data_type="mx_fp", group_size=32, sym=True, act_bits=16)
        assert LLMCompressorFormat.check_scheme_args(s) is True

    def test_valid_w8a16(self):
        s = _scheme(bits=8, data_type="int", group_size=128, sym=True, act_bits=16)
        assert LLMCompressorFormat.check_scheme_args(s) is True

    def test_rejects_bits_3(self):
        s = _scheme(bits=3, data_type="int", group_size=128, sym=True, act_bits=16)
        with pytest.raises(ValueError, match="LLMCompressor format"):
            LLMCompressorFormat.check_scheme_args(s)

    def test_rejects_bits_5(self):
        s = _scheme(bits=5, data_type="int", group_size=128, sym=True, act_bits=16)
        with pytest.raises(ValueError, match="LLMCompressor format"):
            LLMCompressorFormat.check_scheme_args(s)

    def test_rejects_unknown_data_type(self):
        s = _scheme(bits=4, data_type="totally_made_up", group_size=128, sym=True, act_bits=16)
        with pytest.raises(ValueError, match="LLMCompressor format"):
            LLMCompressorFormat.check_scheme_args(s)

    def test_rejects_fp_with_bits_4(self):
        # "fp" with bits != 8 -> error
        s = _scheme(bits=4, data_type="fp", group_size=128, sym=True, act_bits=16)
        with pytest.raises(ValueError, match="LLMCompressor format"):
            LLMCompressorFormat.check_scheme_args(s)

    def test_rejects_int_with_bits_2(self):
        # "int" with bits not in [4, 8] -> error
        s = _scheme(bits=2, data_type="int", group_size=128, sym=True, act_bits=16)
        with pytest.raises(ValueError, match="LLMCompressor format"):
            LLMCompressorFormat.check_scheme_args(s)

    def test_rejects_super_bits(self):
        s = _scheme(bits=4, data_type="int", group_size=128, sym=True, act_bits=16, super_bits=6)
        with pytest.raises(ValueError, match="LLMCompressor format"):
            LLMCompressorFormat.check_scheme_args(s)

    def test_rejects_super_group_size(self):
        s = _scheme(bits=4, data_type="int", group_size=128, sym=True, act_bits=16, super_group_size=8)
        with pytest.raises(ValueError, match="LLMCompressor format"):
            LLMCompressorFormat.check_scheme_args(s)

    def test_block_wfp8_tuple_group_size_bits_mismatch(self):
        # group_size tuple requires bits=8
        s = _scheme(bits=4, data_type="fp", group_size=(128, 128), sym=True, act_bits=8,
                    act_data_type="fp", act_dynamic=True, act_group_size=128)
        with pytest.raises(ValueError, match="LLMCompressor format"):
            LLMCompressorFormat.check_scheme_args(s)

    def test_block_wfp8_tuple_non_fp_data_type(self):
        # group_size tuple requires data_type="fp"
        s = _scheme(bits=8, data_type="int", group_size=(128, 128), sym=True, act_bits=8,
                    act_data_type="fp", act_dynamic=True, act_group_size=128)
        with pytest.raises(ValueError, match="LLMCompressor format"):
            LLMCompressorFormat.check_scheme_args(s)

    def test_block_wfp8_tuple_wrong_length(self):
        # group_size tuple with len != 2
        s = _scheme(bits=8, data_type="fp", group_size=(128, 128, 64), sym=True, act_bits=8,
                    act_data_type="fp", act_dynamic=True, act_group_size=128)
        with pytest.raises(ValueError, match="LLMCompressor format"):
            LLMCompressorFormat.check_scheme_args(s)

    def test_block_wfp8_tuple_not_dynamic(self):
        # requires act_dynamic=True
        s = _scheme(bits=8, data_type="fp", group_size=(128, 128), sym=True, act_bits=8,
                    act_data_type="fp", act_dynamic=False, act_group_size=128)
        with pytest.raises(ValueError, match="LLMCompressor format"):
            LLMCompressorFormat.check_scheme_args(s)

    def test_block_wfp8_tuple_wrong_act_bits(self):
        # requires act_bits=8
        s = _scheme(bits=8, data_type="fp", group_size=(128, 128), sym=True, act_bits=16,
                    act_data_type="fp", act_dynamic=True, act_group_size=128)
        with pytest.raises(ValueError, match="LLMCompressor format"):
            LLMCompressorFormat.check_scheme_args(s)

    def test_block_wfp8_tuple_valid(self):
        s = _scheme(bits=8, data_type="fp", group_size=(128, 128), sym=True, act_bits=8,
                    act_data_type="fp", act_dynamic=True, act_group_size=128)
        assert LLMCompressorFormat.check_scheme_args(s) is True


# ==============================================================================
# __init__ and format_name
# ==============================================================================


class TestLLMCompressorFormatInit:
    def test_format_name(self):
        assert LLMCompressorFormat.format_name == "llm_compressor"

    def test_support_schemes(self):
        assert "W4A16" in LLMCompressorFormat.support_schemes
        assert "MXFP4" in LLMCompressorFormat.support_schemes
        assert "NVFP4" in LLMCompressorFormat.support_schemes

    def test_is_support_scheme_w4a16(self):
        assert LLMCompressorFormat.is_support_scheme("W4A16") is True

    def test_is_support_scheme_w5a16(self):
        assert LLMCompressorFormat.is_support_scheme("W5A16") is False

    def test_is_support_scheme_dict_scheme(self):
        s = _scheme(bits=4, data_type="int", group_size=128, sym=True, act_bits=16)
        assert LLMCompressorFormat.is_support_scheme(s) is True

    def test_init_unsupported_scheme_raises(self):
        s = _scheme(bits=3, data_type="int", group_size=128, sym=True, act_bits=16)
        ctx = SimpleNamespace(mllm=False)
        with pytest.raises(Exception):  # FormatCompatibilityError
            LLMCompressorFormat("llm_compressor", s, ctx)

    def test_init_unsupported_backend_format(self):
        # When format does NOT match the llm_compressor regex, the else branch
        # raises a KeyError for an unsupported backend format
        s = _scheme(bits=8, data_type="fp", group_size=128, sym=True, act_bits=16)
        ctx = SimpleNamespace(mllm=False)
        # Use a format string that doesn't match ^(auto_round:)?llm_compressor
        with pytest.raises(KeyError, match="Unsupported backend"):
            LLMCompressorFormat("totally_not_llm_compressor", s, ctx)


# ==============================================================================
# check_and_reset_format
# ==============================================================================


class TestCheckAndResetFormat:
    def test_w4a16_returns_tuple(self):
        # Plain W4A16 -> no special reset
        s = _scheme(bits=4, data_type="int", group_size=128, sym=True, act_bits=16)
        ctx = SimpleNamespace(layer_config={}, quant_block_list=None, mllm=False)
        fmt = LLMCompressorFormat("llm_compressor", s, ctx)
        result = fmt.check_and_reset_format(s, ctx)
        # Returns (None, scheme, layer_config, quant_block_list)
        assert result[0] is None
        assert result[1].bits == 4

    def test_fp8_static_warns_and_resets_act_group_size(self):
        # FP8 static export -> act_group_size reset to 0
        s = _scheme(bits=8, data_type="fp", group_size=128, sym=True, act_bits=8,
                    act_data_type="fp8_static", act_dynamic=False, act_group_size=128)
        ctx = SimpleNamespace(layer_config={}, quant_block_list=None, mllm=False)
        fmt = LLMCompressorFormat("llm_compressor", s, ctx)
        # The constructor may reset act_group_size to 0
        # After the call, scheme has updated values
        result = fmt.check_and_reset_format(s, ctx)
        # scheme.act_group_size should be 0 (reset by constructor)
        # result is (None, scheme, ...)
        assert result[1].act_group_size == 0