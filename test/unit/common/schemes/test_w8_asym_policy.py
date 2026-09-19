#
# Copyright © 2026 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

import pytest
import torch

from auto_round.schemes import QuantizationScheme, format_allows_w8_asym, parse_scheme

W8_ASYM_OVERRIDES = {"bits": 8, "sym": False, "data_type": "int", "group_size": 128}


class TestFormatAllowsW8Asym:
    def test_llm_compressor_spellings(self):
        assert format_allows_w8_asym("llm_compressor")
        assert format_allows_w8_asym("auto_round:llm_compressor")

    def test_mixed_multi_format_requires_flag(self):
        # The native artifact in a mixed request is unservable: require the flag.
        assert not format_allows_w8_asym("auto_round,llm_compressor")
        assert format_allows_w8_asym("auto_round:llm_compressor,llm_compressor")

    def test_fake_is_allowed(self):
        assert format_allows_w8_asym("fake")

    def test_native_and_others_are_refused(self):
        assert not format_allows_w8_asym("auto_round")
        assert not format_allows_w8_asym("auto_round:auto_gptq")
        assert not format_allows_w8_asym("auto_round:auto_awq")
        assert not format_allows_w8_asym("gguf")
        assert not format_allows_w8_asym(None)
        assert not format_allows_w8_asym("")


class TestParseSchemeW8AsymPolicy:
    def test_default_format_refuses(self):
        with pytest.raises(ValueError, match="8-bit asymmetric"):
            parse_scheme("W8A16", {"sym": False})

    def test_native_format_refuses(self):
        with pytest.raises(ValueError, match="8-bit asymmetric"):
            parse_scheme("W8A16", {"sym": False}, format="auto_round")

    def test_llm_compressor_format_allows(self):
        for fmt in ("llm_compressor", "auto_round:llm_compressor"):
            _, _, attrs = parse_scheme("W8A16", dict(W8_ASYM_OVERRIDES), format=fmt)
            assert attrs["sym"] is False
            assert attrs["bits"] == 8

    def test_fake_format_allows(self):
        _, _, attrs = parse_scheme("W8A16", dict(W8_ASYM_OVERRIDES), format="fake")
        assert attrs["sym"] is False

    def test_env_overrides_native_refusal(self, monkeypatch):
        monkeypatch.setenv("AR_ALLOW_W8_ASYM", "1")
        _, _, attrs = parse_scheme("W8A16", dict(W8_ASYM_OVERRIDES), format="auto_round")
        assert attrs["sym"] is False

    def test_scheme_object_env_is_honored(self, monkeypatch):
        # AutoRound(scheme=QuantizationScheme(bits=8, sym=False)): without the
        # opt-in env the uniform-object spelling still refuses for native
        # formats; with the env or an llm_compressor format it passes.
        obj = QuantizationScheme.from_dict(dict(W8_ASYM_OVERRIDES))
        with pytest.raises(ValueError, match="8-bit asymmetric"):
            parse_scheme(obj, {}, format="auto_round")
        monkeypatch.setenv("AR_ALLOW_W8_ASYM", "1")
        _, _, attrs = parse_scheme(obj, {}, format="auto_round")
        assert attrs["sym"] is False
        _, _, attrs = parse_scheme(obj, {}, format="llm_compressor")
        assert attrs["sym"] is False

    def test_w8_symmetric_never_touched(self):
        _, _, attrs = parse_scheme("W8A16", {}, format="auto_round")
        assert attrs["sym"] is True


class TestAutoSchemeW8AsymPolicy:
    def _scheme(self):
        from auto_round.auto_scheme.gen_auto_scheme import AutoScheme

        return AutoScheme(options=["W8A16", "W4A16"], avg_bits=6.0)

    def _w8_option(self, scheme):
        return [o for o in scheme.options if getattr(o, "bits", None) == 8][0]

    @pytest.mark.parametrize(
        "fmt,opt_in,expect_sym",
        [
            (None, False, True),  # default: pinned
            ("auto_round", False, True),  # native: pinned
            ("llm_compressor", False, False),  # llmc: kept asym
            (None, True, False),  # env opt-in: kept asym
        ],
    )
    def test_option_pinning_matrix(self, monkeypatch, fmt, opt_in, expect_sym):
        if opt_in:
            monkeypatch.setenv("AR_ALLOW_W8_ASYM", "1")
        scheme = self._scheme()
        parse_scheme(scheme, {"sym": False}, format=fmt)
        assert self._w8_option(scheme).sym is expect_sym

    def test_fixed_entries_flag_skips_pinning(self):
        from auto_round.auto_scheme.delta_loss import _enforce_w8_symmetric_entries

        cfg = {"lm_head": {"bits": 8, "data_type": "int"}}
        assert _enforce_w8_symmetric_entries(cfg) == 1
        assert cfg["lm_head"]["sym"] is True

        cfg = {"lm_head": {"bits": 8, "data_type": "int"}}
        assert _enforce_w8_symmetric_entries(cfg, allow_w8_asym=True) == 0
        assert "sym" not in cfg["lm_head"]

    def test_fixed_entries_flag_skips_explicit_refusal(self):
        from auto_round.auto_scheme.delta_loss import _enforce_w8_symmetric_entries

        cfg = {"lm_head": {"bits": 8, "sym": False, "data_type": "int"}}
        with pytest.raises(ValueError, match="8-bit asymmetric"):
            _enforce_w8_symmetric_entries(dict(cfg))
        # flag: no raise, no pin
        _enforce_w8_symmetric_entries(dict(cfg), allow_w8_asym=True)


class TestResolverW8AsymPolicy:
    def _resolve(self, layer_config, **kw):
        from auto_round.compressors.config_resolution import ResolvedScheme
        from auto_round.compressors.layer_config_resolver import resolve_layer_config

        model = torch.nn.Sequential(torch.nn.Linear(32, 32))
        scheme = ResolvedScheme.from_scheme(QuantizationScheme(bits=4, sym=False, act_bits=16, act_data_type="float"))
        return resolve_layer_config(
            model=model,
            scheme=scheme,
            layer_config=layer_config,
            supported_types=(torch.nn.Linear,),
            inner_supported_types=(),
            **kw,
        )

    def test_explicit_entry_llm_compressor_format_allowed(self):
        resolved = self._resolve(
            {"0": {"bits": 8, "sym": False, "data_type": "int", "group_size": 128}},
            format="auto_round:llm_compressor",
        )
        assert resolved["0"]["sym"] is False

    def test_explicit_entry_env_allowed(self, monkeypatch):
        monkeypatch.setenv("AR_ALLOW_W8_ASYM", "1")
        resolved = self._resolve({"0": {"bits": 8, "sym": False, "data_type": "int", "group_size": 128}})
        assert resolved["0"]["sym"] is False

    def test_inherited_entry_pinned_by_default(self):
        resolved = self._resolve({"0": {"bits": 8, "data_type": "int", "group_size": 128}})
        assert resolved["0"]["sym"] is True

    def test_inherited_entry_kept_asym_under_env(self, monkeypatch):
        monkeypatch.setenv("AR_ALLOW_W8_ASYM", "1")
        resolved = self._resolve({"0": {"bits": 8, "data_type": "int", "group_size": 128}})
        assert resolved["0"].get("sym") is not True
