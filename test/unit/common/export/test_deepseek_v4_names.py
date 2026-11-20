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
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from auto_round.export.common.deepseek_v4_names import (
    build_checkpoint_name_maps,
    prepare_native_export,
    restore_quantization_names,
    restore_tensor_name,
)


def test_native_names_preserve_packed_objects_and_nonweight_parameters(tmp_path):
    pairs = {
        "model.layers.0.attn_hc.base": "layers.0.hc_attn_base",
        "model.layers.0.self_attn.compressor.indexer.scorer.weights_proj.weight": "layers.0.attn.indexer.weights_proj.weight",
        "model.layers.0.self_attn.compressor.indexer.kv_proj.weight": "layers.0.attn.indexer.compressor.wkv.weight",
        "model.layers.0.self_attn.o_a_proj.weight": "layers.0.attn.wo_a.weight",
        "model.layers.0.mlp.experts.255.gate_proj.weight": "layers.0.ffn.experts.255.w1.weight",
        "model.hc_head.hc_base": "hc_head_base",
        "lm_head.weight": "head.weight",
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys(pairs.values())}))
    aliases, modules = build_checkpoint_name_maps(pairs.values())
    for loaded, native in pairs.items():
        assert restore_tensor_name(loaded, aliases, modules, pairs.values()) == native
    packed = torch.tensor([[0, 127, 255]], dtype=torch.uint8)
    state = {
        "model.layers.0.mlp.experts.255.gate_proj.weight_packed": packed,
        "model.layers.0.mlp.experts.255.gate_proj.weight_scale": torch.ones(1, 1, dtype=torch.float8_e4m3fn),
        "model.layers.0.mlp.experts.255.gate_proj.weight_global_scale": torch.tensor([123.0]),
        "model.layers.0.attn_hc.base": torch.tensor([1.0]),
        "model.layers.0.self_attn.o_a_proj.weight": torch.ones(2, 2, dtype=torch.bfloat16),
    }
    config = {"config_groups": {"nvfp4": {"targets": ["model.layers.0.mlp.experts.255.gate_proj", "RoutedExperts"]}}}
    model = SimpleNamespace(config=SimpleNamespace(quantization_config=config))
    restored, restored_config = prepare_native_export(model, state, str(tmp_path))
    assert restored["layers.0.ffn.experts.255.w1.weight_packed"] is packed
    assert restored["layers.0.hc_attn_base"] is state["model.layers.0.attn_hc.base"]
    assert model.config.quantization_config is config
    assert config["config_groups"]["nvfp4"]["targets"][0] == "model.layers.0.mlp.experts.255.gate_proj"
    assert restored_config["config_groups"]["nvfp4"]["targets"] == [
        "layers.0.ffn.experts.255.w1",
        "RoutedExperts",
    ]
    assert "layers.0.attn.wo_a" in restored_config["ignore"]


@pytest.mark.parametrize("config", [None, [], {}, {"config_groups": []}])
def test_native_config_requires_groups_dictionary(config):
    with pytest.raises(ValueError, match="DeepSeek V4.*config_groups dictionary"):
        restore_quantization_names(config, {}, [])


@pytest.mark.parametrize("group", [None, [], {}, {"targets": "Linear"}])
def test_native_config_requires_target_lists(group):
    with pytest.raises(ValueError, match="DeepSeek V4.*targets list"):
        restore_quantization_names({"config_groups": {"nvfp4": group}}, {}, [])


@pytest.mark.parametrize("container", [str, list, tuple])
def test_v4_aliases_accept_scalar_and_sequence_patterns(monkeypatch, container):
    mapping = pytest.importorskip("transformers.conversion_mapping")
    from transformers.core_model_loading import WeightRenaming

    from auto_round.modeling.deepseek_v4 import expand_layer_config

    source, target = r"\.attn\.", ".self_attn."
    rule = object.__new__(WeightRenaming)
    rule.source_patterns = source if container is str else container([source])
    rule.target_patterns = target if container is str else container([target])
    monkeypatch.setattr(mapping, "get_checkpoint_conversion_mapping", lambda _: [rule])
    native = "layers.0.attn.wq.weight"
    loaded = "layers.0.self_attn.wq.weight"
    expanded = expand_layer_config({native: {"bits": 4}})
    assert expanded[loaded] == {"bits": 4}
    aliases, _ = build_checkpoint_name_maps([native])
    assert aliases[loaded] == native


def test_unmapped_names_fail_closed():
    with pytest.raises(ValueError, match="Unmapped"):
        restore_tensor_name("model.unknown.weight_packed", {}, {}, {})


@pytest.mark.parametrize("remote", [False, True])
@pytest.mark.parametrize("failure", [None, RuntimeError, TypeError])
def test_save_model_disables_reverse_numeric_conversions(tmp_path, remote, failure):
    from auto_round.export.common.serialization import save_model

    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layers.0.ffn.experts.0.w1.weight": "model.safetensors"}})
    )
    model = MagicMock()
    model.dtype = torch.bfloat16
    model.name_or_path = "org/DeepSeek-V4-Flash" if remote else str(source)
    model.config = SimpleNamespace(
        model_type="deepseek_v4",
        _commit_hash="pinned-source-commit",
        quantization_config={
            "quant_method": "compressed-tensors",
            "config_groups": {"nvfp4": {"targets": ["model.layers.0.mlp.experts.0.gate_proj"]}},
        },
    )
    packed = torch.tensor([[255]], dtype=torch.uint8)
    model.state_dict.return_value = {"model.layers.0.mlp.experts.0.gate_proj.weight_packed": packed}
    original_config = model.config.quantization_config

    def save_pretrained(*args, **kwargs):
        assert model.config.quantization_config["config_groups"]["nvfp4"]["targets"] == ["layers.0.ffn.experts.0.w1"]
        if failure is not None:
            raise failure("simulated save failure")

    model.save_pretrained.side_effect = save_pretrained
    with (
        patch("auto_round.export.common.serialization.apply_post_save_source_fixes"),
        patch("auto_round.export.common.serialization._save_source_config"),
        patch("auto_round.export.common.serialization._save_model_configs") as save_configs,
        patch("safetensors.torch.save_file") as save_file,
        patch(
            "transformers.utils.hub.cached_file", return_value=str(source / "model.safetensors.index.json")
        ) as cached,
    ):
        if failure is RuntimeError:
            with pytest.raises(RuntimeError, match="simulated save failure"):
                save_model(model, str(tmp_path / "output"))
        else:
            # A second export must receive the original runtime names again.
            for _ in range(2):
                save_model(model, str(tmp_path / "output"))
                saved = json.loads((tmp_path / "output" / "quantization_config.json").read_text())
                assert saved["config_groups"]["nvfp4"]["targets"] == ["layers.0.ffn.experts.0.w1"]
                assert model.config.quantization_config is original_config
        if failure is TypeError:
            assert save_configs.call_count == save_file.call_count == 2
    assert model.config.quantization_config is original_config
    assert original_config["config_groups"]["nvfp4"]["targets"] == ["model.layers.0.mlp.experts.0.gate_proj"]
    if remote:
        assert cached.call_count == (1 if failure is RuntimeError else 2)
        cached.assert_called_with(model.name_or_path, "model.safetensors.index.json", revision="pinned-source-commit")
    else:
        cached.assert_not_called()
    kwargs = model.save_pretrained.call_args.kwargs
    assert kwargs["save_original_format"] is False
    assert kwargs["state_dict"]["layers.0.ffn.experts.0.w1.weight_packed"] is packed
