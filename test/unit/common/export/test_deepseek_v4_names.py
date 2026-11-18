import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from auto_round.export.common.deepseek_v4_names import (
    build_checkpoint_name_maps,
    prepare_native_export,
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
    restored = prepare_native_export(model, state, str(tmp_path))
    assert restored["layers.0.ffn.experts.255.w1.weight_packed"] is packed
    assert restored["layers.0.hc_attn_base"] is state["model.layers.0.attn_hc.base"]
    assert model.config.quantization_config["config_groups"]["nvfp4"]["targets"] == [
        "layers.0.ffn.experts.255.w1",
        "RoutedExperts",
    ]
    assert "layers.0.attn.wo_a" in model.config.quantization_config["ignore"]


def test_unmapped_names_fail_closed():
    with pytest.raises(ValueError, match="Unmapped"):
        restore_tensor_name("model.unknown.weight_packed", {}, {}, {})


def test_save_model_disables_reverse_numeric_conversions(tmp_path):
    from auto_round.export.common.serialization import save_model

    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layers.0.ffn.experts.0.w1.weight": "model.safetensors"}})
    )
    model = MagicMock()
    model.dtype = torch.bfloat16
    model.name_or_path = str(source)
    model.config = SimpleNamespace(
        model_type="deepseek_v4",
        quantization_config={
            "quant_method": "compressed-tensors",
            "config_groups": {"nvfp4": {"targets": ["model.layers.0.mlp.experts.0.gate_proj"]}},
        },
    )
    packed = torch.tensor([[255]], dtype=torch.uint8)
    model.state_dict.return_value = {"model.layers.0.mlp.experts.0.gate_proj.weight_packed": packed}
    with (
        patch("auto_round.export.common.serialization.apply_post_save_source_fixes"),
        patch("auto_round.export.common.serialization._save_source_config"),
    ):
        save_model(model, str(tmp_path / "output"))
    kwargs = model.save_pretrained.call_args.kwargs
    assert kwargs["save_original_format"] is False
    assert kwargs["state_dict"]["layers.0.ffn.experts.0.w1.weight_packed"] is packed
