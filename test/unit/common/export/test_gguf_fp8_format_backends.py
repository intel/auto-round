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
"""Fast tests for the GGUF / FP8 ``OutputFormat`` helpers in ``auto_round/export/formats/backends``.

Model downloads and llama.cpp conversion lookups are mocked, so no checkpoint is needed.
"""

import types
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from auto_round.compressors.config_resolution.errors import FormatCompatibilityError
from auto_round.export.formats.backends import gguf as gguf_fmt
from auto_round.export.formats.backends.fp8 import FP8Format
from auto_round.export.formats.backends.gguf import GGUFFormat
from auto_round.schemes import QuantizationScheme

gguf = pytest.importorskip("gguf")


def _scheme(**kwargs):
    base = dict(bits=4, group_size=128, sym=True, data_type="int", act_bits=16)
    base.update(kwargs)
    return QuantizationScheme.from_dict(base)


class TestGgufArgsCheck:
    def test_non_gguf_formats_leave_scheme_untouched(self):
        scheme = _scheme()
        assert GGUFFormat.gguf_args_check(scheme, model=None, platform="hf", formats="auto_round") is scheme
        assert scheme.bits == 4 and scheme.group_size == 128

    @pytest.fixture
    def conversion(self, tmp_path):
        conv = MagicMock()
        conv.is_supported.return_value = True
        with (
            patch("auto_round.export.export_to_gguf.llama_cpp_conversion.get_conversion", return_value=conv),
            patch("auto_round.utils.model.get_gguf_architecture", return_value="LlamaForCausalLM"),
        ):
            yield conv, str(tmp_path)

    def test_resets_scheme_to_gguf_config(self, conversion):
        _, model_dir = conversion
        scheme = _scheme(bits=2)
        with patch("auto_round.logger.logger") as log:
            out = GGUFFormat.gguf_args_check(scheme, model=model_dir, platform="hf", formats=["gguf:q4_k_m"])
        assert out.bits == 4 and out.group_size == 32 and out.sym is False
        assert out.data_type == "int_asym_dq" and out.super_bits == 6 and out.super_group_size == 8
        assert "reset to" in log.info.call_args.args[0]

    def test_q_1_with_multiple_formats_keeps_int(self, conversion):
        _, model_dir = conversion
        scheme = _scheme()
        out = GGUFFormat.gguf_args_check(scheme, model=model_dir, platform="hf", formats=["gguf:q4_1", "fake"])
        assert out.data_type == "int" and out.sym is False

    def test_model_object_and_unsupported_format(self, conversion):
        _, model_dir = conversion
        model = types.SimpleNamespace(name_or_path=model_dir)
        with patch("auto_round.logger.logger") as log:
            GGUFFormat.gguf_args_check(_scheme(), model=model, platform="hf", formats=["gguf:q9_z"])
        assert "not supported" in log.error.call_args.args[0]

    def test_remote_model_path_is_resolved(self, conversion):
        with patch("auto_round.utils.model.download_or_get_path", return_value=conversion[1]) as dl:
            GGUFFormat.gguf_args_check(_scheme(), model="org/repo", platform="hf", formats=["gguf:q8_0"])
        dl.assert_called_once_with("org/repo", "hf")

    def test_conflicting_k_quant_formats(self, conversion):
        with pytest.raises(FormatCompatibilityError, match="at the same time"):
            GGUFFormat.gguf_args_check(
                _scheme(), model=conversion[1], platform="hf", formats=["gguf:q4_k_m", "gguf:q2_k_s"]
            )

    def test_unsupported_architecture(self, conversion):
        conv, model_dir = conversion
        conv.is_supported.return_value = False
        with pytest.raises(FormatCompatibilityError, match="not supported"):
            GGUFFormat.gguf_args_check(_scheme(), model=model_dir, platform="hf", formats="gguf:q4_0")

    def test_outdated_gguf_py(self, tmp_path):
        with patch(
            "auto_round.export.export_to_gguf.llama_cpp_conversion.get_conversion", side_effect=AttributeError("x")
        ), pytest.raises(ImportError, match="latest gguf-py"):
            GGUFFormat.gguf_args_check(_scheme(), model=str(tmp_path), platform="hf", formats="gguf:q4_0")

    def test_check_scheme_args(self):
        assert GGUFFormat.check_scheme_args(_scheme(data_type="int_asym_dq"))
        with pytest.raises(ValueError, match="data_type=mx_fp"):
            GGUFFormat.check_scheme_args(_scheme(data_type="mx_fp"))


class TestGgufTypeHelpers:
    @pytest.mark.parametrize(
        "query, expected",
        [("gguf:q4_k", "gguf:q4_k"), ("gguf:q8_k", "gguf:q8_0"), ("gguf:q2_0", "gguf:q2_k"), ("gguf:q7_k", None)],
    )
    def test_search_gguf_type(self, query, expected):
        assert gguf_fmt._search_gguf_type(query) == expected

    def test_search_gguf_type_invalid(self):
        with pytest.raises(KeyError):
            gguf_fmt._search_gguf_type("gguf:bad")

    @pytest.mark.parametrize(
        "query, expected",
        [("GGUF:Q4_K", "gguf:q5_0"), ("gguf:q6_k", "gguf:q8_0"), ("gguf:q4_0", "gguf:q4_0")],
    )
    def test_type_fallback(self, query, expected):
        assert gguf_fmt.gguf_type_fallback(query) == expected
        assert gguf_fmt._gguf_type_fallback(query) == expected

    def test_digital_in_layer_name(self):
        assert gguf_fmt._get_digital_in_layer_name("model.layers.12.mlp") == 12
        assert gguf_fmt._get_digital_in_layer_name("lm_head") is None


class TestSelectGgufLayerType:
    @pytest.fixture
    def model(self):
        model = nn.Module()
        model.kv_b_proj = nn.Linear(256, 256, bias=False)
        model.config = types.SimpleNamespace(architectures=["DeepseekV2ForCausalLM"], qk_nope_head_dim=128)
        return model

    def _select(self, layer_name, config, layer, model, **overrides):
        kwargs = dict(
            target_gguf_format="gguf:q4_k_m",
            lm_head_name="lm_head",
            tie_word_embeddings=False,
            block_size=256,
            base_target_bits=4,
            dtype_selector=MagicMock(),
            gguf_name=None,
            i_layer=0,
        )
        kwargs.update(overrides)
        model_class = kwargs.pop("model_class", types.SimpleNamespace(model_arch=gguf.MODEL_ARCH.LLAMA))
        return gguf_fmt._select_gguf_layer_type(layer_name, config, layer, model, model_class, **kwargs)

    def test_default_mostly_type(self, model):
        assert self._select("layers.0.q", {"bits": 4}, nn.Linear(256, 8), model) == "gguf:q4_k"

    def test_fixed_by_user_without_bits(self, model):
        with patch.object(gguf_fmt, "logger") as log:
            assert self._select("layers.0.q", {"fixed_by_user": True}, nn.Linear(256, 8), model) == "gguf:q4_k"
        assert "requires providing bits" in log.warning.call_args.args[0]

    @pytest.mark.parametrize(
        "config, expected",
        [
            ({"bits": 8, "group_size": 32, "sym": True, "data_type": "int"}, "gguf:q8_0"),
            ({"bits": 3, "group_size": 64, "super_bits": 6, "sym": True}, "gguf:q3_k"),
            ({"bits": 5, "group_size": 64, "sym": False, "data_type": "int"}, "gguf:q5_k"),
            ({"bits": 8, "group_size": 64, "super_bits": 4, "sym": False}, "gguf:q8_0"),
        ],
    )
    def test_fixed_by_user_bits(self, model, config, expected):
        config = {"fixed_by_user": True, "extra_key": 1, **config}
        assert self._select("layers.0.q", config, nn.Linear(256, 8), model) == expected

    def test_fixed_by_user_unmatched_super_bits(self, model):
        config = {"fixed_by_user": True, "bits": 7, "group_size": 64, "super_bits": 4}
        with pytest.raises(ValueError, match="could not match"):
            self._select("layers.0.q", config, nn.Linear(256, 8), model)

    def test_fixed_by_user_invalid_bits(self, model):
        with pytest.raises(ValueError, match="invalid bit setting"):
            self._select("layers.0.q", {"fixed_by_user": True, "bits": 7, "group_size": 64}, nn.Linear(256, 8), model)

    def test_lm_head_untied(self, model):
        assert self._select("lm_head", {"bits": 4}, nn.Linear(256, 8), model) == "gguf:q6_k"
        falcon = types.SimpleNamespace(model_arch=gguf.MODEL_ARCH.FALCON)
        assert self._select("lm_head", {"bits": 4}, nn.Linear(256, 8), model, model_class=falcon) == "gguf:q8_0"
        assert self._select("lm_head", {"bits": 4}, nn.Linear(100, 8), model) == "gguf:bf16"

    def test_lm_head_tied_is_skipped(self, model):
        assert self._select("lm_head", {"bits": 4}, nn.Linear(256, 8), model, tie_word_embeddings=True) is None

    def test_embedding(self, model):
        emb = nn.Embedding(8, 256)
        assert self._select("embed", {"bits": 4}, emb, model) == "gguf:q4_k"
        assert self._select("embed", {"bits": 4}, emb, model, target_gguf_format="gguf:q4_0") == "gguf:q4_0"

    def test_bits_override_and_invalid(self, model):
        assert self._select("layers.0.q", {"bits": 8}, nn.Linear(256, 8), model) == "gguf:q8_0"
        with pytest.raises(ValueError, match="invalid bit setting"):
            self._select("layers.0.q", {"bits": 7}, nn.Linear(256, 8), model)

    def test_dtype_selector_used_for_named_tensor(self, model):
        selector = MagicMock()
        selector.select_gguf_type.return_value = "gguf:q6_k"
        out = self._select(
            "layers.3.v", {"bits": 4}, nn.Linear(256, 8), model, gguf_name="blk.3.attn_v", dtype_selector=selector
        )
        assert out == "gguf:q6_k"
        selector.select_gguf_type.assert_called_once_with("blk.3.attn_v.weight", 2, 0)

    def test_block_size_fallback(self, model):
        with patch.object(gguf_fmt, "logger") as log:
            assert self._select("layers.0.q", {"bits": 4}, nn.Linear(64, 8), model) == "gguf:q5_0"
        log.warning.assert_called_once()

    def test_deepseek_kv_b_proj_fallback(self, model):
        assert self._select("kv_b_proj", {"bits": 4}, model.kv_b_proj, model) == "gguf:q5_0"


class TestGetLayerConfigByGgufFormat:
    def _model(self, config=None):
        model = nn.Module()
        model.config = config if config is not None else MagicMock(to_dict=MagicMock(return_value={}))
        return model

    def test_outdated_gguf_py(self):
        model = self._model(types.SimpleNamespace())
        with pytest.raises(ImportError, match="latest gguf-py"):
            gguf_fmt.get_layer_config_by_gguf_format({}, "gguf:q4_k_m", model)

    def test_unknown_architecture_returns_unchanged(self):
        conv = MagicMock()
        conv.get_model_class.side_effect = NotImplementedError
        layer_config = {"a": {"bits": 4}}
        with patch("auto_round.export.export_to_gguf.llama_cpp_conversion.get_conversion", return_value=conv):
            assert gguf_fmt.get_layer_config_by_gguf_format(layer_config, "gguf:q4_k_m", self._model()) == (
                layer_config,
                {},
            )

    def test_unknown_layer_count_returns_unchanged(self):
        conv = MagicMock()
        layer_config = {"a": {"bits": 4}}
        cfg = types.SimpleNamespace(to_dict=dict)
        with patch("auto_round.export.export_to_gguf.llama_cpp_conversion.get_conversion", return_value=conv):
            out = gguf_fmt.get_layer_config_by_gguf_format(layer_config, "gguf:q4_k_m", self._model(cfg))
        assert out == (layer_config, {})


def _fp8_block_scheme(**kwargs):
    base = dict(
        bits=8,
        group_size=(128, 128),
        data_type="fp",
        act_bits=8,
        act_group_size=128,
        act_data_type="fp",
        act_dynamic=True,
    )
    base.update(kwargs)
    return QuantizationScheme.from_dict(base)


class TestFP8Format:
    def test_check_scheme_args_ok(self):
        assert FP8Format.check_scheme_args(_fp8_block_scheme())

    def test_check_scheme_args_reports_every_mismatch(self):
        bad = _scheme(act_dynamic=False, act_group_size=None, act_data_type="int")
        with pytest.raises(ValueError) as err:
            FP8Format.check_scheme_args(bad)
        for key in ("bits=4", "data_type=int", "group_size=128", "act_dynamic=False", "act_bits=16"):
            assert key in str(err.value)

    @pytest.mark.parametrize("output_format, kwarg", [("fp8", "quant_method"), ("auto_round:fp8", "backend")])
    def test_save_quantized(self, output_format, kwarg):
        fmt = FP8Format.__new__(FP8Format)
        fmt.output_format, fmt.backend = output_format, None
        layer_config = {"a": {"bits": 8, "act_bits": 8}, "b": {"bits": 16, "act_bits": 16}}
        set = {"group_size": (128, 128)}
        target = "auto_round.export.export_to_autoround.export_to_fp8.save_quantized_as_autoround"
        with patch(target, return_value="saved") as save:
            assert fmt.save_quantized("out", model=None, layer_config=layer_config, serialization_dict=set) == "saved"
        kwargs = save.call_args.kwargs
        assert kwargs[kwarg] == output_format
        assert set["weight_block_size"] == (128, 128) and set["ignored_layers"] == ["b"]

    def test_save_quantized_non_block_group_size(self):
        fmt = FP8Format.__new__(FP8Format)
        fmt.output_format, fmt.backend = "fp8", None
        set = {"group_size": -1}
        with patch("auto_round.export.export_to_autoround.export_to_fp8.save_quantized_as_autoround"):
            fmt.save_quantized("out", layer_config={}, serialization_dict=set)
        assert "weight_block_size" not in set

    def test_pack_layer_delegates(self):
        fmt = FP8Format.__new__(FP8Format)
        fmt.output_format = "fp8"
        with patch("auto_round.export.export_to_autoround.export_to_fp8.pack_layer") as pack:
            fmt.pack_layer("x", "model", device="cpu")
        pack.assert_called_once_with("x", "model", "fp8", device="cpu")
