import pytest
import torch
from torch import nn
from transformers.core_model_loading import Chunk, PrefixChange, WeightConverter

from auto_round.export.export_to_gguf.hf_checkpoint_restorer import HFCheckpointRestorer


class _DummyConfig:
    model_type = "dummy"


class _DummyModel:
    config = _DummyConfig()

    def __init__(self, state_dict, weight_conversions):
        self._state_dict = state_dict
        self._weight_conversions = weight_conversions

    def state_dict(self):
        return self._state_dict


def test_restorer_reverses_transformers_prefix_change():
    model = _DummyModel(
        {
            "model.layers.0.mlp.down_proj.weight": torch.ones(2, 2),
        },
        [PrefixChange(prefix_to_remove="language_model", model_prefix="model")],
    )

    restored = list(HFCheckpointRestorer(model).iter_tensors())

    assert [tensor.checkpoint_name for tensor in restored] == [
        "model.language_model.layers.0.mlp.down_proj.weight",
    ]
    assert restored[0].hf_names == ("model.layers.0.mlp.down_proj.weight",)
    assert torch.equal(restored[0].tensor_fn(), torch.ones(2, 2))


def test_restorer_concats_split_hf_tensors_back_to_checkpoint_tensor():
    model = _DummyModel(
        {
            "model.layers.0.mlp.gate_proj.weight": torch.ones(1, 2),
            "model.layers.0.mlp.up_proj.weight": torch.full((1, 2), 2.0),
        },
        [
            WeightConverter(
                source_patterns="mlp.gate_up_proj.weight",
                target_patterns=["mlp.gate_proj.weight", "mlp.up_proj.weight"],
                operations=[Chunk(dim=0)],
            ),
        ],
    )

    restored = list(HFCheckpointRestorer(model).iter_tensors())

    assert [tensor.checkpoint_name for tensor in restored] == [
        "model.layers.0.mlp.gate_up_proj.weight",
    ]
    assert restored[0].hf_names == (
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
    )
    assert torch.equal(restored[0].tensor_fn(), torch.tensor([[1.0, 1.0], [2.0, 2.0]]))


def test_restorer_preserves_model_order_across_converted_and_passthrough_tensors():
    model = _DummyModel(
        {
            "model.layers.0.self_attn.gate_proj.weight": torch.ones(1, 2),
            "model.layers.0.self_attn.up_proj.weight": torch.full((1, 2), 2.0),
            "model.layers.0.mlp.experts.weight": torch.ones(2, 2),
        },
        [
            WeightConverter(
                source_patterns="self_attn.gate_up_proj.weight",
                target_patterns=["self_attn.gate_proj.weight", "self_attn.up_proj.weight"],
                operations=[Chunk(dim=0)],
            ),
        ],
    )

    restored = list(HFCheckpointRestorer(model).iter_tensors())

    assert [tensor.checkpoint_name for tensor in restored] == [
        "model.layers.0.self_attn.gate_up_proj.weight",
        "model.layers.0.mlp.experts.weight",
    ]


def test_restorer_does_not_rerun_completed_conversion_group(monkeypatch):
    state_dict = {
        "model.layers.0.mlp.gate_proj.weight": torch.ones(1, 2),
        "model.layers.0.mlp.up_proj.weight": torch.full((1, 2), 2.0),
    }
    model = _DummyModel(
        state_dict,
        [
            WeightConverter(
                source_patterns="mlp.gate_up_proj.weight",
                target_patterns=["mlp.gate_proj.weight", "mlp.up_proj.weight"],
                operations=[Chunk(dim=0)],
            ),
        ],
    )

    def fail_if_converted(*args, **kwargs):
        raise AssertionError("completed conversion group was executed again")

    monkeypatch.setattr(WeightConverter, "convert", fail_if_converted)

    restored = list(HFCheckpointRestorer(model, completed_hf_names=set(state_dict)).iter_tensors())

    assert restored == []


def test_resolve_restored_qtype_keeps_higher_recipe_fallback_for_multi_source_layer_config():
    import gguf

    from auto_round.export.export_to_gguf.convert import resolve_restored_qtype

    diagnostics = []
    qtype = resolve_restored_qtype(
        layer_config={
            "model.layers.0.mlp.gate_proj": {"bits": 4, "super_bits": 6, "sym": False},
            "model.layers.0.mlp.up_proj": {"bits": 4, "super_bits": 6, "sym": False},
        },
        hf_names=(
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        ),
        checkpoint_name="model.layers.0.mlp.gate_up_proj.weight",
        gguf_name="blk.0.ffn_gate_up.weight",
        fallback_qtype=gguf.GGMLQuantizationType.Q5_K,
        diagnostics=diagnostics,
        allow_recipe_fallback=True,
    )

    assert qtype is None
    assert diagnostics == []


def test_resolve_restored_qtype_respects_mixed_layer_config_even_when_recipe_is_higher():
    import gguf

    from auto_round.export.export_to_gguf.convert import resolve_restored_qtype

    diagnostics = []
    qtype = resolve_restored_qtype(
        layer_config={
            "model.layers.0.mlp.gate_proj": {"bits": 4, "super_bits": 6, "sym": False},
            "model.layers.0.mlp.up_proj": {"bits": 4, "super_bits": 6, "sym": False},
        },
        hf_names=(
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        ),
        checkpoint_name="model.layers.0.mlp.gate_up_proj.weight",
        gguf_name="blk.0.ffn_gate_up.weight",
        fallback_qtype=gguf.GGMLQuantizationType.Q5_K,
        diagnostics=diagnostics,
        allow_recipe_fallback=False,
    )

    assert qtype == gguf.GGMLQuantizationType.Q4_K
    assert diagnostics == []


def test_resolve_restored_qtype_keeps_higher_layer_config_than_recipe_fallback():
    import gguf

    from auto_round.export.export_to_gguf.convert import resolve_restored_qtype

    diagnostics = []
    qtype = resolve_restored_qtype(
        layer_config={
            "model.layers.0.mlp.down_proj": {"bits": 6, "super_bits": 8, "sym": True},
        },
        hf_names=("model.layers.0.mlp.down_proj.weight",),
        checkpoint_name="model.layers.0.mlp.down_proj.weight",
        gguf_name="blk.0.ffn_down.weight",
        fallback_qtype=gguf.GGMLQuantizationType.Q5_K,
        diagnostics=diagnostics,
    )

    assert qtype == gguf.GGMLQuantizationType.Q6_K
    assert diagnostics == []


def test_resolve_restored_qtype_falls_back_and_records_conflicting_multi_source_layer_config():
    import gguf

    from auto_round.export.export_to_gguf.convert import resolve_restored_qtype

    diagnostics = []
    qtype = resolve_restored_qtype(
        layer_config={
            "model.layers.0.mlp.gate_proj": {"bits": 4, "super_bits": 6, "sym": False},
            "model.layers.0.mlp.up_proj": {"bits": 6, "super_bits": 8, "sym": True},
        },
        hf_names=(
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        ),
        checkpoint_name="model.layers.0.mlp.gate_up_proj.weight",
        gguf_name="blk.0.ffn_gate_up.weight",
        fallback_qtype=gguf.GGMLQuantizationType.Q5_K,
        diagnostics=diagnostics,
    )

    assert qtype is None
    assert diagnostics == [
        {
            "checkpoint_name": "model.layers.0.mlp.gate_up_proj.weight",
            "gguf_name": "blk.0.ffn_gate_up.weight",
            "reason": "multi_source_dtype_conflict",
            "fallback_qtype": "Q5_K",
            "hf_sources": [
                {
                    "name": "model.layers.0.mlp.gate_proj.weight",
                    "layer_config_qtype": "Q4_K",
                },
                {
                    "name": "model.layers.0.mlp.up_proj.weight",
                    "layer_config_qtype": "Q6_K",
                },
            ],
        }
    ]


def test_restorer_falls_back_to_original_tensor_when_weight_conversion_fails():
    model = _DummyModel(
        {
            "model.layers.0.mlp.experts.gate_up_proj": torch.tensor([], dtype=torch.bfloat16),
        },
        [
            WeightConverter(
                source_patterns="mlp.experts.gate_up_proj",
                target_patterns=["mlp.experts.*.gate_proj.weight", "mlp.experts.*.up_proj.weight"],
                operations=[Chunk(dim=1)],
            ),
        ],
    )

    restored = list(HFCheckpointRestorer(model).iter_tensors())

    assert [tensor.checkpoint_name for tensor in restored] == ["model.layers.0.mlp.experts.gate_up_proj"]
    assert restored[0].transform_kind == "passthrough"


def test_get_restored_tensors_replaces_empty_live_tensor_from_checkpoint(monkeypatch):
    from auto_round.export.export_to_gguf import convert

    tensor_name = "model.layers.0.self_attn.q_proj.weight"
    model = _DummyModel({tensor_name: torch.tensor([], dtype=torch.bfloat16)}, [])
    model.tensor_name_list = [tensor_name]

    def iter_checkpoint_tensors(cls):
        if tensor_name not in cls.model.tensor_name_list:
            yield tensor_name, torch.ones(2, 2)

    monkeypatch.setattr(convert, "_iter_extra_tensors", iter_checkpoint_tensors)

    class Converter:
        pass

    converter = Converter()
    converter.model = model
    restored = list(convert.get_restored_tensors(converter))

    assert [tensor.checkpoint_name for tensor in restored] == [tensor_name]
    assert restored[0].hf_names == (tensor_name,)
    assert torch.equal(restored[0].tensor_fn(), torch.ones(2, 2))


def test_name_resolution_diagnostics_are_never_written_to_export_directory(tmp_path, monkeypatch):
    from auto_round.export.export_to_gguf.convert import _flush_name_resolution_diagnostics

    monkeypatch.setenv("AR_GGUF_DEBUG_NAME_RESOLUTION", "1")

    class Converter:
        fname_out = tmp_path / "model.gguf"
        _gguf_name_resolution_diagnostics = [{"checkpoint_name": "x"}]

    _flush_name_resolution_diagnostics(Converter)

    assert not (tmp_path / "gguf_name_resolution.json").exists()


def test_gguf_writer_has_tensor_supports_current_gguf_dict_entries():
    import numpy as np
    from gguf.gguf_writer import GGUFWriter

    from auto_round.export.export_to_gguf.convert import _gguf_writer_has_tensor

    writer = GGUFWriter(path=None, arch="llama")
    writer.add_tensor("blk.24.attn_k_norm.weight", np.zeros((1,), dtype=np.float32))

    assert _gguf_writer_has_tensor(writer, "blk.24.attn_k_norm.weight")
    assert not _gguf_writer_has_tensor(writer, "blk.24.attn_v_norm.weight")


def test_gguf_shape_fallback_updates_layer_config_before_quantization():
    from auto_round.compressors.utils import _apply_gguf_shape_fallback
    from auto_round.export.export_to_gguf.config import GGUF_INNER_CONFIG

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(1152, 4096, bias=False)

    layer_config = {"proj": {**GGUF_INNER_CONFIG["gguf:q6_k"], "fixed_by_user": True, "scale_dtype": torch.float32}}

    _apply_gguf_shape_fallback(layer_config, Model())

    assert layer_config["proj"]["bits"] == 8
    assert layer_config["proj"]["group_size"] == 32
    assert layer_config["proj"]["super_bits"] is None
    assert layer_config["proj"]["super_group_size"] is None
    assert layer_config["proj"]["fixed_by_user"] is True
    assert layer_config["proj"]["scale_dtype"] is torch.float32


def test_quant_data_recomputes_scale_when_final_qtype_differs_from_layer_config(monkeypatch):
    import gguf
    import numpy as np

    from auto_round.export.export_to_gguf import convert

    captured = {}

    class Module:
        def __init__(self):
            self.weight = torch.ones(2, 32)
            self.scale = torch.ones(2, 16)

    module = Module()

    class Converter:
        model = object()
        layer_config = {"model.layers.0.self_attn.q_proj": {"bits": 6, "super_bits": 8, "sym": True}}

    def fake_get_module(_model, _layer_name):
        return module

    def fake_ggml_quant(data_torch, ggml_type, device=None, **kwargs):
        captured["data_shape"] = tuple(data_torch.shape)
        captured["ggml_type"] = ggml_type
        captured["scale"] = kwargs["scale"]
        return np.zeros((data_torch.shape[0], data_torch.shape[1]), dtype=np.float32)

    monkeypatch.setattr(convert, "get_module", fake_get_module)
    monkeypatch.setattr(convert, "ggml_quant", fake_ggml_quant)

    data, qtype = convert._quant_data(
        Converter(),
        torch.ones(2, 32),
        gguf.GGMLQuantizationType.Q8_0,
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "blk.0.attn_q.weight",
        0,
        device="cpu",
    )

    assert qtype == gguf.GGMLQuantizationType.Q8_0
    assert data.shape == (2, 32)
    assert captured["ggml_type"] == "q8_0"
    assert captured["data_shape"] == (2, 32)
    assert captured["scale"] is None


def test_quant_data_rejects_layer_scale_that_does_not_match_final_qtype(monkeypatch):
    import gguf

    from auto_round.export.export_to_gguf import convert

    class Module:
        def __init__(self):
            self.weight = torch.ones(2, 32)
            self.scale = torch.ones(2, 2)

    module = Module()

    class Converter:
        model = object()
        layer_config = {"model.layers.0.self_attn.q_proj": {"bits": 8, "group_size": 32, "sym": True}}

    monkeypatch.setattr(convert, "get_module", lambda _model, _layer_name: module)

    with pytest.raises(ValueError, match="q8_0 scale shape"):
        convert._quant_data(
            Converter(),
            torch.ones(2, 32),
            gguf.GGMLQuantizationType.Q8_0,
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "blk.0.attn_q.weight",
            0,
            device="cpu",
        )
