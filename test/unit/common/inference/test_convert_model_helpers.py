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
"""CPU-only tests for helpers in ``auto_round/inference/convert_model.py``.

Optional kernels (gptqmodel, auto_gptq, mlx) are faked through ``sys.modules`` so
post-init / repacking branches are exercised without the real packages.
"""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from auto_round.inference import convert_model as cm
from auto_round.inference.backend import BackendInfo


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(8, 8)
        self.k = nn.Linear(8, 8)


class _TinyModel(nn.Module):
    def __init__(self, n_blocks=2):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Block() for _ in range(n_blocks)])
        self.lm_head = nn.Linear(8, 8)

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class _TinyLM(nn.Module):
    def __init__(self, tie=False, output_emb=True):
        super().__init__()
        self.embed = nn.Embedding(4, 8)
        self.body = nn.Linear(8, 8)
        self.lm_head = nn.Linear(8, 4, bias=False)
        self._tie = tie
        self._output_emb = output_emb

    def tie_weights(self):
        if self._tie:
            self.lm_head.weight = self.embed.weight

    def get_output_embeddings(self):
        return self.lm_head if self._output_emb else None


def _cfg(**kwargs):
    base = dict(bits=4, group_size=32, sym=True, extra_config={})
    base.update(kwargs)
    return types.SimpleNamespace(**base)


class TestSmallHelpers:
    def test_flatten_list(self):
        assert cm.flatten_list([1, [2, (3, [4])], "a"]) == [1, 2, 3, 4, "a"]

    def test_infer_target_device(self):
        assert cm.infer_target_device(None) == "cpu"
        assert cm.infer_target_device({"a": "cpu", "b": "disk"}) == "cpu"
        assert cm.infer_target_device({"a": "cpu", "b": "xpu:1"}) == "xpu"
        with patch.object(cm, "get_available_devices", return_value=["cuda", "cpu"]):
            assert cm.infer_target_device({"a": 0}) == "cuda"
            assert cm.infer_target_device("auto") == "cuda"

    def test_get_device_and_layer_features(self):
        lin = nn.Linear(3, 5)
        assert cm.get_device(lin) == torch.device("cpu")
        assert cm.get_device(torch.zeros(1)) == torch.device("cpu")
        assert cm._get_layer_features(lin) == (3, 5)
        assert cm._get_layer_features(cm.Conv1D(5, 3)) == (3, 5)
        assert cm._get_layer_features(nn.ReLU()) == (None, None)

    def test_get_available_devices_always_has_cpu(self):
        assert cm.get_available_devices()[-1] == "cpu"

    def test_import_exllamav2_kernels_missing_warns(self):
        with patch.dict(sys.modules, {"exllamav2_kernels": None}), patch.object(cm, "logger") as log:
            cm._import_exllamav2_kernels()
        assert log.warning_once.call_count == 2


class TestGetKeysToNotConvert:
    def test_non_llm_model(self):
        assert cm.get_keys_to_not_convert(nn.Sequential(nn.Linear(2, 2))) == []

    def test_untied_keeps_output_embedding(self):
        assert cm.get_keys_to_not_convert(_TinyLM()) == ["lm_head"]

    def test_tied_weights(self):
        keys = cm.get_keys_to_not_convert(_TinyLM(tie=True))
        assert {"embed", "lm_head"} <= set(keys)
        assert all(not k.endswith((".weight", ".bias")) for k in keys)

    def test_no_output_embedding_keeps_last_module(self):
        assert cm.get_keys_to_not_convert(_TinyLM(output_emb=False)) == ["lm_head"]

    def test_get_modules_to_not_convert_fallback(self):
        mods = cm._get_modules_to_not_convert(_TinyLM(), skip_modules=["body"], keep_in_fp32_modules=["embed"])
        assert mods == ["body", "embed"]
        mods = cm._get_modules_to_not_convert(_TinyLM(), skip_modules=["body"], add_default_skips=True)
        assert mods == ["lm_head", "body"]


class TestRemapPathsForTextModel:
    def _model(self, model_type="composite", text_type=None):
        text_config = types.SimpleNamespace(model_type=text_type) if text_type else None
        return types.SimpleNamespace(config=types.SimpleNamespace(model_type=model_type, text_config=text_config))

    def test_no_model_type(self):
        blocks, extra = cm._remap_paths_for_text_model(types.SimpleNamespace(), ["a"], {"x": 1})
        assert blocks == ["a"] and extra == {"x": 1}

    def test_remaps_with_text_submodel_mapping(self):
        rule = types.SimpleNamespace(source_patterns=[r"^model\.language_model"], target_patterns=["model"])
        mappings = {"composite": None, "text": [object(), rule]}
        with patch("transformers.conversion_mapping.get_checkpoint_conversion_mapping", side_effect=mappings.get):
            blocks, extra = cm._remap_paths_for_text_model(
                self._model(text_type="text"),
                ["model.language_model.layers", "other.layers"],
                {"model.language_model.layers.0.q": {"bits": 8}},
            )
        assert blocks == ["model.layers", "other.layers"]
        assert extra == {"model.layers.0.q": {"bits": 8}}

    @pytest.mark.parametrize("mapping", [None, [object()]])
    def test_without_usable_mapping(self, mapping):
        with patch("transformers.conversion_mapping.get_checkpoint_conversion_mapping", return_value=mapping):
            blocks, extra = cm._remap_paths_for_text_model(self._model(), ["a"], None)
        assert blocks == ["a"] and extra is None


class TestGetLayerConfig:
    def test_string_block_names_with_gptq_filters(self):
        cfg = _cfg(
            to_quant_block_names="model.layers",
            modules_in_block_to_quantize=[["q"]],
            dynamic={r"-:.*layers\.1\.q$": {}},
        )
        configs = cm.get_layer_config(_TinyModel(), cfg)
        assert configs["model.layers.0.q"].bits == 4
        assert configs["model.layers.0.k"].bits == 16
        assert configs["model.layers.1.q"].bits == 16
        assert "lm_head" not in configs

    def test_nested_quant_block_list_and_modules_to_not_convert(self):
        cfg = _cfg(
            quant_block_list=[["model.layers.0", "model.layers.1"]],
            modules_to_not_convert=["layers.0.k"],
        )
        configs = cm.get_layer_config(_TinyModel(), cfg)
        assert len(configs) == 4
        assert configs["model.layers.0.k"].bits == 16
        assert configs["model.layers.1.k"].bits == 4

    def test_block_list_falls_back_to_text_model_remap(self):
        cfg = _cfg(block_name_to_quantize=["model.language_model.layers"])
        with patch.object(cm, "_remap_paths_for_text_model", return_value=(["model.layers"], {})) as remap:
            configs = cm.get_layer_config(_TinyModel(1), cfg)
        remap.assert_called_once()
        assert sorted(configs) == ["model.layers.0.k", "model.layers.0.q"]


class TestConvertGptqV1ToV2:
    @pytest.mark.parametrize("version, cls_name", [("7.0.0", "ExllamaV2Linear"), ("6.0.0", "ExllamaV2QuantLinear")])
    def test_adds_zero_offset_for_4bit_int32(self, version, cls_name):
        cls = type(cls_name, (nn.Module,), {})
        exl = types.ModuleType("exllamav2")
        setattr(exl, cls_name, cls)
        fake = types.ModuleType("gptqmodel")
        fake.__version__ = version
        mods = {
            "gptqmodel": fake,
            "gptqmodel.nn_modules": types.ModuleType("n"),
            "gptqmodel.nn_modules.qlinear": types.ModuleType("q"),
            "gptqmodel.nn_modules.qlinear.exllamav2": exl,
        }
        layer4, layer8 = cls(), cls()
        layer4.bits, layer8.bits = 4, 8
        layer4.qzeros = torch.zeros(2, dtype=torch.int32)
        layer8.qzeros = torch.zeros(2, dtype=torch.int32)
        model = nn.ModuleDict({"a": layer4, "b": layer8})
        with patch.dict(sys.modules, mods):
            cm.convert_gptq_v1_to_v2_format(model)
        assert torch.all(layer4.qzeros == 0b00010001000100010001000100010001)
        assert torch.all(layer8.qzeros == 0)


class _FakeQLayer(nn.Module):
    def __init__(self, quant_type, with_qweight=True):
        super().__init__()
        self.QUANT_TYPE = quant_type
        if with_qweight:
            self.register_buffer("qweight", torch.zeros(1, dtype=torch.int32))
        self.post_init = MagicMock()


class TestMaybeConvertGptqToMlx:
    def _patch_env(self, from_gptq):
        qlinear_mlx = types.ModuleType("auto_round_extension.mlx.qlinear_mlx")
        qlinear_mlx.QuantLinearMLX = types.SimpleNamespace(from_gptq=from_gptq)
        mlx = types.ModuleType("mlx")
        return patch.dict(
            sys.modules,
            {"mlx": mlx, "mlx.core": types.ModuleType("mlx.core"), "auto_round_extension.mlx.qlinear_mlx": qlinear_mlx},
        )

    def _model(self):
        model = nn.Module()
        model.good = _FakeQLayer("gptq")
        model.bad = _FakeQLayer("gptq")
        model.native = _FakeQLayer("mlx")
        model.no_qweight = _FakeQLayer("gptq", with_qweight=False)
        model.config = types.SimpleNamespace(quantization_config=MagicMock(to_dict=lambda: {"sym": True}))
        return model

    def test_skips_off_darwin_or_without_mlx_backend(self):
        from_gptq = MagicMock()
        with self._patch_env(from_gptq), patch("platform.system", return_value="Linux"):
            cm._maybe_convert_gptq_to_mlx(self._model(), ["auto_round:mlx"])
        with self._patch_env(from_gptq), patch("platform.system", return_value="Darwin"):
            cm._maybe_convert_gptq_to_mlx(self._model(), ["auto_round:torch"])
        from_gptq.assert_not_called()

    def test_skips_when_mlx_missing(self):
        with patch.dict(sys.modules, {"mlx": None, "mlx.core": None}), patch("platform.system", return_value="Darwin"):
            cm._maybe_convert_gptq_to_mlx(self._model(), ["auto_round:mlx"])

    def test_converts_and_reports_failures(self):
        def from_gptq(module, sym):
            assert sym is True
            if module is model.bad:
                raise RuntimeError("boom")
            return nn.Identity()

        model = self._model()
        with (
            self._patch_env(from_gptq),
            patch("platform.system", return_value="Darwin"),
            patch.object(cm, "logger") as log,
        ):
            cm._maybe_convert_gptq_to_mlx(model, ["auto_round:mlx"])
        assert isinstance(model.good, nn.Identity)
        assert isinstance(model.bad, _FakeQLayer)
        assert isinstance(model.native, _FakeQLayer)
        assert log.info.called and log.warning.call_count == 2

    def test_no_candidate_layers(self):
        from_gptq = MagicMock()
        model = nn.Module()
        model.native = _FakeQLayer("mlx")
        with self._patch_env(from_gptq), patch("platform.system", return_value="Darwin"):
            cm._maybe_convert_gptq_to_mlx(model, ["auto_round:mlx"])
        from_gptq.assert_not_called()


class TestPostInit:
    @pytest.fixture
    def backend_infos(self):
        infos = {
            "auto_gptq:exllamav2": BackendInfo(["cuda"], [True], ["p"], [4], compute_dtype=["float16"]),
            "gptqmodel:exllamav2": BackendInfo(["cuda"], [True], ["p"], [4], compute_dtype=["float16", "bfloat16"]),
            "auto_round_kernel": BackendInfo(["cpu"], [True], ["p"], [4], compute_dtype=["float32", "bfloat16"]),
            "auto_round:humming": BackendInfo(["cuda"], [True], ["p"], [4], compute_dtype=["bfloat16"]),
            "auto_round:torch": BackendInfo(["cpu"], [True], ["p"], [4], compute_dtype=["float32"]),
        }
        with patch.object(cm, "BackendInfos", infos):
            yield

    def test_ark_and_humming_layers_are_repacked(self, backend_infos):
        model = _TinyModel(1)
        model.ark = _FakeQLayer("ark_gptq")
        model.hum = _FakeQLayer("humming_awq")
        model.other = _FakeQLayer("torch")
        cm.post_init(model, ["auto_round_kernel", "auto_round:humming"])
        model.ark.post_init.assert_called_once()
        model.hum.post_init.assert_called_once()
        model.other.post_init.assert_not_called()

    def test_forces_common_dtype(self, backend_infos):
        model = _TinyModel(1)
        with patch.object(cm, "logger") as log:
            cm.post_init(model, ["auto_round:humming"])
        log.warning.assert_called_once()
        # post_init reassigns a local; nn.Module.to is in-place, so the original object is converted.
        assert model.dtype == torch.bfloat16

    def test_no_dtype_change_when_already_supported(self, backend_infos):
        model = _TinyModel(1)
        cm.post_init(model, ["auto_round:torch"])
        assert model.dtype == torch.float32

    def test_autogptq_post_init(self, backend_infos):
        gptq_post_init = MagicMock(side_effect=lambda m, use_act_order: m)
        utils_mod = types.SimpleNamespace(autogptq_post_init=gptq_post_init)
        mods = {
            "auto_gptq": types.ModuleType("auto_gptq"),
            "auto_gptq.modeling": types.ModuleType("m"),
            "auto_gptq.modeling._utils": utils_mod,
        }
        with patch.dict(sys.modules, mods), patch.object(cm, "_import_exllamav2_kernels") as exl:
            cm.post_init(_TinyModel(1), ["auto_gptq:exllamav2"])
        gptq_post_init.assert_called_once()
        exl.assert_called_once()

    @pytest.mark.parametrize("version", ["5.6.0", "6.0.3"])
    def test_gptqmodel_post_init(self, backend_infos, version):
        gptq_post_init = MagicMock(side_effect=lambda m, use_act_order: m)
        fake = types.ModuleType("gptqmodel")
        fake.__version__ = version
        mods = {
            "gptqmodel": fake,
            "gptqmodel.utils": types.ModuleType("u"),
            "gptqmodel.utils.model": types.SimpleNamespace(hf_gptqmodel_post_init=gptq_post_init),
        }
        model = _TinyModel(1)
        model.config = types.SimpleNamespace(
            quantization_config=types.SimpleNamespace(packing_format="auto_round:gptq")
        )
        model.validated = nn.Module()
        model.validated.validate_once = MagicMock()
        with patch.dict(sys.modules, mods), patch.object(cm, "convert_gptq_v1_to_v2_format") as v1v2:
            cm.post_init(model, ["gptqmodel:exllamav2"])
        v1v2.assert_called_once_with(model)
        gptq_post_init.assert_called_once()
        assert model.validated.validate_once.called == (version != "5.6.0")
