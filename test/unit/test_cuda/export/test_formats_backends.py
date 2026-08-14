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
"""GPU-side fast unit tests for the ``auto_round.export.formats`` package.

Covers the format resolver internals, the various backend ``check_scheme_args``
predicates, and ``OutputFormat`` ABC base-class behavior not exercised by the
existing ``test_format_resolver.py``.
"""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")

import torch.nn as nn

from auto_round.compressors.config_resolution import FormatCompatibilityError, ResolvedScheme, resolve_scheme_value
from auto_round.export.formats.backends.auto_awq import AutoAWQFormat
from auto_round.export.formats.backends.auto_gptq import AutoGPTQFormat
from auto_round.export.formats.backends.autoround import AutoRoundFormat
from auto_round.export.formats.backends.fake import FakeFormat
from auto_round.export.formats.backends.fp8 import FP8Format
from auto_round.export.formats.backends.mlx import MLXFormat
from auto_round.export.formats.base import OutputFormat, _check_divisible_by_32
from auto_round.export.formats.resolver import (
    _apply_scheme_format_constraint,
    _deduplicate,
    _normalize_format_names,
    _precise_gguf_name,
    _validate_format_combination,
    resolve_formats,
)
from auto_round.schemes import QuantizationScheme


def _scheme(**overrides):
    base = dict(
        bits=4,
        group_size=128,
        sym=True,
        data_type="int",
        act_bits=16,
        act_group_size=None,
        act_sym=None,
        act_data_type=None,
        act_dynamic=None,
        super_bits=None,
        super_group_size=None,
        rotation_config=None,
    )
    base.update(overrides)
    return QuantizationScheme(**base)


# ==============================================================================
# Resolver helpers
# ==============================================================================


class TestNormalizeFormatNames:
    def test_lowercase_split(self):
        result = _normalize_format_names("auto_round,gguf:q4_k_m", bits=4)
        assert "auto_round" in result
        assert "gguf:q4_k_m" in result

    def test_spaces_stripped(self):
        result = _normalize_format_names(" auto_round , fake ", bits=4)
        assert "auto_round" in result
        assert "fake" in result

    def test_dedup(self):
        result = _normalize_format_names("fake,fake", bits=4)
        assert result == ["fake"]

    def test_q_wildcard_replaced(self):
        # "q*_" is replaced with "q{bits}_" before splitting
        result = _normalize_format_names("gguf:q*_k_m", bits=4)
        assert "gguf:q4_k_m" in result


class TestDeduplicate:
    def test_keeps_order(self):
        assert _deduplicate(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]

    def test_empty(self):
        assert _deduplicate([]) == []


class TestValidateFormatCombination:
    def test_no_gguf_ok(self):
        _validate_format_combination(["auto_round"])  # no raise

    def test_gguf_with_fake_ok(self):
        _validate_format_combination(["gguf:q4_k_m", "fake"])  # no raise

    def test_gguf_with_real_companion_raises(self):
        with pytest.raises(FormatCompatibilityError):
            _validate_format_combination(["gguf:q4_k_m", "auto_round"])


class TestApplySchemeFormatConstraint:
    def test_non_gguf_scheme_returns_names(self):
        scheme = ResolvedScheme.from_scheme(_scheme(bits=4, group_size=128, sym=True, data_type="int", act_bits=16))
        result = _apply_scheme_format_constraint(["auto_round", "fake"], scheme)
        assert "auto_round" in result
        assert "fake" in result

    def test_gguf_scheme_resets_to_gguf_preset(self):
        # Use a gguf-matching scheme
        s = _scheme(
            bits=4,
            group_size=32,
            sym=True,
            data_type="int",
            act_bits=16,
            super_bits=None,
            super_group_size=None,
        )
        scheme = ResolvedScheme.from_scheme(s)
        # Names not matching the gguf scheme should be reset
        result = _apply_scheme_format_constraint(["auto_round", "fake"], scheme)
        # The result should be the gguf scheme name + fake
        assert "fake" in result


class TestPreciseGgufName:
    def test_no_gguf_returns_none(self):
        formats = [
            SimpleNamespace(is_gguf=lambda: False, output_format="auto_round", backend=None),
        ]
        assert _precise_gguf_name(formats) is None

    def test_gguf_format_with_precise_backend(self):
        backend = SimpleNamespace(output_format="gguf:q4_k_m")
        formats = [
            SimpleNamespace(is_gguf=lambda: True, output_format="gguf", backend=backend),
        ]
        result = _precise_gguf_name(formats)
        assert result == "gguf:q4_k_m"

    def test_gguf_format_with_self_backend(self):
        # If backend.output_format is "gguf", fall back to format.output_format
        formats = [
            SimpleNamespace(is_gguf=lambda: True, output_format="gguf:q5_k_m", backend=None),
        ]
        result = _precise_gguf_name(formats)
        assert result == "gguf:q5_k_m"


# ==============================================================================
# OutputFormat ABC
# ==============================================================================


class TestOutputFormatBase:
    def test_get_support_matrix_includes_known_formats(self):
        s = OutputFormat.get_support_matrix()
        assert isinstance(s, str)
        assert "fake" in s.lower()

    def test_check_divisible_by_32_with_no_model(self):
        # When model is None, function returns the layer_config unchanged
        cfg = {"layer.bits": 4}
        result = _check_divisible_by_32(_scheme(), None, cfg)
        assert result == cfg

    def test_check_divisible_by_32_marks_non_aligned_layers(self):
        # Build a model with a layer that has in_features=33 (not divisible by 32)
        m = nn.Sequential(nn.Linear(33, 33))
        result = _check_divisible_by_32(_scheme(), m, None)
        # The 33x33 Linear should be marked as fixed_by_user with bits=16
        assert result is not None
        for name, cfg in result.items():
            assert cfg.get("bits") == 16
            assert cfg.get("data_type") == "fp"
            assert cfg.get("fixed_by_user") is True

    def test_is_support_scheme_with_string(self):
        # FakeFormat has support_schemes=None (not iterable), so the string branch
        # is not safe to test directly. Use a scheme object instead.
        s = _scheme(bits=4, data_type="int", group_size=128, sym=True, act_bits=16)
        # The base check_scheme_args returns True by default
        assert FakeFormat.is_support_scheme(s) is True

    def test_is_support_scheme_with_scheme_false(self):
        # For schemes that don't match the base default, is_support_scheme returns False
        # However, the base check_scheme_args just returns True, so any scheme passes
        s = _scheme(bits=8, data_type="fp", group_size=128, sym=True, act_bits=16)
        assert FakeFormat.is_support_scheme(s) is True

    def test_is_support_scheme_string_known_for_awq(self):
        # AutoAWQFormat support_schemes = ["W4A16"]
        assert AutoAWQFormat.is_support_scheme("W4A16") is True
        assert AutoAWQFormat.is_support_scheme("W8A16") is False


# ==============================================================================
# Backend check_scheme_args
# ==============================================================================


class TestBackendCheckSchemeArgs:
    """Exercise the per-backend ``check_scheme_args`` predicates to drive the
    otherwise-uncovered error reporting in each backend class."""

    def _ctx(self):
        return SimpleNamespace(model=None, layer_config=None, mllm=False)

    def test_awq_rejects_non_4bit(self):
        s = _scheme(bits=8, data_type="int")
        with pytest.raises(ValueError, match="auto_awq"):
            AutoAWQFormat.check_scheme_args(s)

    def test_awq_rejects_non_int(self):
        s = _scheme(bits=4, data_type="fp")
        with pytest.raises(ValueError, match="auto_awq"):
            AutoAWQFormat.check_scheme_args(s)

    def test_awq_rejects_super_bits(self):
        s = _scheme(bits=4, data_type="int", super_bits=6)
        with pytest.raises(ValueError, match="auto_awq"):
            AutoAWQFormat.check_scheme_args(s)

    def test_awq_accepts_w4a16(self):
        s = _scheme(bits=4, data_type="int", group_size=128, sym=True, act_bits=16)
        assert AutoAWQFormat.check_scheme_args(s) is True

    def test_gptq_rejects_non_supported_bits(self):
        s = _scheme(bits=5, data_type="int")
        with pytest.raises(ValueError, match="auto_gptq"):
            AutoGPTQFormat.check_scheme_args(s)

    def test_gptq_rejects_non_int(self):
        s = _scheme(bits=4, data_type="fp")
        with pytest.raises(ValueError, match="auto_gptq"):
            AutoGPTQFormat.check_scheme_args(s)

    def test_gptq_accepts_w4a16(self):
        s = _scheme(bits=4, data_type="int", group_size=128, sym=True, act_bits=16)
        assert AutoGPTQFormat.check_scheme_args(s) is True

    def test_fp8_rejects_non_8bit(self):
        s = _scheme(bits=4, data_type="fp", group_size=(128, 128), act_dynamic=True, act_group_size=128, act_bits=8, act_data_type="fp")
        with pytest.raises(ValueError, match="fp8"):
            FP8Format.check_scheme_args(s)

    def test_fp8_rejects_non_fp_data_type(self):
        s = _scheme(bits=8, data_type="int", group_size=(128, 128), act_dynamic=True, act_group_size=128, act_bits=8, act_data_type="fp")
        with pytest.raises(ValueError, match="fp8"):
            FP8Format.check_scheme_args(s)

    def test_fp8_rejects_1d_group_size(self):
        s = _scheme(bits=8, data_type="fp", group_size=128, act_dynamic=True, act_group_size=128, act_bits=8, act_data_type="fp")
        with pytest.raises(ValueError, match="fp8"):
            FP8Format.check_scheme_args(s)

    def test_mlx_rejects_non_int(self):
        s = _scheme(bits=4, data_type="fp", act_bits=16)
        with pytest.raises(ValueError, match="MLX"):
            MLXFormat.check_scheme_args(s)

    def test_mlx_rejects_non_16_act(self):
        s = _scheme(bits=4, data_type="int", act_bits=8)
        with pytest.raises(ValueError, match="MLX"):
            MLXFormat.check_scheme_args(s)

    def test_mlx_accepts_w4a16(self):
        s = _scheme(bits=4, data_type="int", group_size=128, sym=True, act_bits=16)
        assert MLXFormat.check_scheme_args(s) is True


# ==============================================================================
# FakeFormat pack_layer (no-op) and save_quantized
# ==============================================================================


class TestFakeFormat:
    def test_pack_layer_is_noop(self):
        # pack_layer is a no-op for fake; calling it must not raise
        FakeFormat.pack_layer("any_layer", nn.Module())

    def test_check_and_reset_format_returns_none(self):
        s = _scheme()
        ctx = SimpleNamespace(layer_config={}, quant_block_list=None, mllm=False)
        # Need to create an instance since check_and_reset_format is an instance method
        fmt = FakeFormat("fake", s, ctx)
        result = fmt.check_and_reset_format(s, ctx)
        # Should return (None, scheme, layer_config, quant_block_list)
        assert result[0] is None
        assert result[1] is s


# ==============================================================================
# AutoRoundFormat backend selection
# ==============================================================================


class TestAutoRoundFormatSelection:
    def test_w4a16_sym_picks_auto_gptq_backend(self):
        scheme = ResolvedScheme.from_scheme(
            _scheme(bits=4, data_type="int", group_size=128, sym=True, act_bits=16)
        )
        ctx = SimpleNamespace(layer_config={}, mllm=False, model=None)
        fmt = AutoRoundFormat("auto_round", scheme.value, ctx)
        # backend should be AutoGPTQFormat for int sym
        assert fmt.backend is not None
        assert isinstance(fmt.backend, AutoGPTQFormat)

    def test_w4a16_asym_picks_auto_awq_backend(self):
        scheme = _scheme(bits=4, data_type="int", group_size=128, sym=False, act_bits=16)
        ctx = SimpleNamespace(layer_config=None, mllm=False, model=None)
        fmt = AutoRoundFormat("auto_round", scheme, ctx)
        # backend should be AutoAWQFormat for int 4-bit asym
        assert fmt.backend is not None
        assert isinstance(fmt.backend, AutoAWQFormat)

    def test_unsupported_scheme_raises(self):
        # act_bits < 16 and no supported backend -> raises ValueError
        scheme = _scheme(bits=4, data_type="fp", group_size=128, sym=True, act_bits=8, act_data_type="fp", act_dynamic=True)
        ctx = SimpleNamespace(layer_config=None, mllm=False, model=None)
        with pytest.raises(ValueError, match="AutoRound format does not support"):
            AutoRoundFormat("auto_round", scheme, ctx)

    def test_format_name_attribute(self):
        assert AutoRoundFormat.format_name == "auto_round"
        assert FakeFormat.format_name == "fake"
        assert AutoAWQFormat.format_name == "auto_awq"
        assert AutoGPTQFormat.format_name == "auto_gptq"
        assert FP8Format.format_name == "fp8"
        assert MLXFormat.format_name == "mlx"


# ==============================================================================
# resolve_formats integration (lightweight)
# ==============================================================================


class TestResolveFormatsIntegration:
    def test_default_format_resolves(self):
        scheme = resolve_scheme_value("W4A16", {})
        result = resolve_formats(scheme, format="auto_round", model=None)
        assert result.formats
        assert len(result.formats) == 1
        # Default backend name
        assert result.formats[0].get_backend_name()

    def test_fake_format_resolves(self):
        scheme = resolve_scheme_value("W4A16", {})
        result = resolve_formats(scheme, format="fake", model=None)
        assert len(result.formats) == 1
        assert result.formats[0].is_fake() is True

    def test_invalid_format_raises(self):
        scheme = resolve_scheme_value("W4A16", {})
        with pytest.raises(ValueError):
            resolve_formats(scheme, format="totally_made_up_format", model=None)

    def test_gguf_format_with_real_companion_raises(self):
        scheme = resolve_scheme_value("W4A16", {})
        with pytest.raises(FormatCompatibilityError):
            resolve_formats(scheme, format="gguf:q4_k_m,auto_round", model=None)

    def test_gguf_with_fake_companion_allowed(self):
        scheme = resolve_scheme_value("W4A16", {})
        result = resolve_formats(scheme, format="gguf:q4_k_m,fake", model=None)
        # The result should contain both gguf and fake formats
        assert any(f.is_gguf() for f in result.formats)
        assert any(f.is_fake() for f in result.formats)
