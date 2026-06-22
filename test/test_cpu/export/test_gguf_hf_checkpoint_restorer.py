import torch
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


def test_resolve_restored_qtype_uses_consistent_multi_source_layer_config():
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
    )

    assert qtype == gguf.GGMLQuantizationType.Q4_K
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
