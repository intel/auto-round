import inspect
import json
from dataclasses import replace

import pytest
import torch
from safetensors import safe_open

from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear
from auto_round.export.svdquant_mxfp4 import unpack_lowrank_weight
from auto_round.export.svdquant_nunchaku import (
    IdentitySVDQuantModelAdapter,
    MXFP4ResidualTensorProvider,
    SourceLinearRecord,
    SVDQuantExportConfig,
    SVDQuantExportRecord,
    collect_svdquant_tensors,
    pack_nunchaku_16bit_vector,
    save_svdquant_nunchaku_safetensors,
    unpack_nunchaku_16bit_vector,
)


def _toy_model(*, rank=3, bias=True, in_features=65, out_features=7):
    residual = torch.nn.Linear(in_features, out_features, bias=bias)
    residual.data_type = "mx_fp4e2m1"
    residual.bits = 4
    residual.group_size = 32
    residual.sym = True
    residual.act_data_type = "mx_fp4e2m1"
    residual.act_bits = 4
    residual.act_group_size = 32
    residual.act_sym = True
    residual.act_dynamic = True
    lora_down = torch.nn.Linear(in_features, rank, bias=False)
    lora_up = torch.nn.Linear(rank, out_features, bias=False)
    smooth = torch.arange(1, in_features + 1, dtype=torch.float32)
    layer = SVDQuantLinear(residual, lora_down, lora_up, smooth)
    return torch.nn.Sequential(layer)


def test_default_collection_emits_runtime_layout_tensors_without_debug_residual():
    model = _toy_model()

    tensors = collect_svdquant_tensors(model)

    assert set(tensors) == {
        "0.qweight",
        "0.wscales",
        "0.smooth",
        "0.smooth_orig",
        "0.lora_down",
        "0.lora_up",
        "0.bias",
    }
    assert tensors["0.qweight"].shape == (128, 64)
    assert tensors["0.qweight"].dtype == torch.int8
    assert tensors["0.wscales"].shape == (4, 128)
    assert tensors["0.wscales"].dtype == torch.uint8
    for key in ("0.smooth", "0.smooth_orig", "0.lora_down", "0.lora_up", "0.bias"):
        assert tensors[key].dtype == torch.bfloat16
    assert tensors["0.lora_down"].shape == (128, 16)
    assert tensors["0.lora_up"].shape == (128, 16)
    logical_down = model[0].lora_down.weight.detach().to(torch.bfloat16)
    logical_up = model[0].lora_up.weight.detach().to(torch.bfloat16)
    unpacked_down = unpack_lowrank_weight(tensors["0.lora_down"], down=True)
    unpacked_up = unpack_lowrank_weight(tensors["0.lora_up"], down=False)
    torch.testing.assert_close(unpacked_down[:3, :65], logical_down)
    torch.testing.assert_close(unpacked_up[:7, :3], logical_up)
    torch.testing.assert_close(
        unpack_nunchaku_16bit_vector(tensors["0.smooth"])[:65], model[0].smooth.to(torch.bfloat16)
    )
    torch.testing.assert_close(
        unpack_nunchaku_16bit_vector(tensors["0.bias"])[:7],
        model[0].residual_linear.bias.detach().to(torch.bfloat16),
    )


def test_adapter_maps_all_logical_source_records_with_model_level_visibility():
    model = torch.nn.Sequential(_toy_model()[0], _toy_model()[0])

    class CapturingAdapter(IdentitySVDQuantModelAdapter):
        def map_modules(self, received_model, records):
            assert received_model is model
            records = tuple(records)
            assert all(isinstance(record, SourceLinearRecord) for record in records)
            assert [record.name for record in records] == ["0", "1"]
            assert [record.lora_down.shape for record in records] == [(3, 65), (3, 65)]
            assert [record.scheme.data_type for record in records] == ["mx_fp4e2m1", "mx_fp4e2m1"]
            assert all(record.scheme.group_size == 32 and record.scheme.sym for record in records)
            assert all(record.scheme.act_data_type == "mx_fp4e2m1" for record in records)
            self.records = records
            return super().map_modules(received_model, records)

        def validate_records(self, sources, records):
            self.validated_records = (sources, records)

    adapter = CapturingAdapter()

    tensors = collect_svdquant_tensors(model, adapter=adapter)

    assert set(key.split(".", 1)[0] for key in tensors) == {"0", "1"}
    assert len(adapter.records) == 2
    assert adapter.validated_records[0] is adapter.records
    assert [record.prefix for record in adapter.validated_records[1]] == ["0", "1"]


def test_bias_and_smooth_vectors_match_layout_fixture_and_identity_padding():
    model = _toy_model(bias=False)
    model[0].register_buffer("smooth_orig", torch.arange(65, dtype=torch.float32))

    tensors = collect_svdquant_tensors(model)

    bias = unpack_nunchaku_16bit_vector(tensors["0.bias"])
    assert torch.count_nonzero(bias[:7]) == 0
    assert torch.equal(bias[7:], torch.ones_like(bias[7:]))
    smooth_orig = unpack_nunchaku_16bit_vector(tensors["0.smooth_orig"])
    torch.testing.assert_close(smooth_orig[:65], torch.arange(65, dtype=torch.bfloat16))
    assert torch.equal(smooth_orig[65:], torch.ones_like(smooth_orig[65:]))
    fixture = pack_nunchaku_16bit_vector(torch.arange(128, dtype=torch.float16))
    assert fixture[:32].tolist() == [
        0.0,
        1.0,
        8.0,
        9.0,
        2.0,
        3.0,
        10.0,
        11.0,
        4.0,
        5.0,
        12.0,
        13.0,
        6.0,
        7.0,
        14.0,
        15.0,
        16.0,
        17.0,
        24.0,
        25.0,
        18.0,
        19.0,
        26.0,
        27.0,
        20.0,
        21.0,
        28.0,
        29.0,
        22.0,
        23.0,
        30.0,
        31.0,
    ]


def test_debug_unpacked_is_explicit_and_not_runtime_loadable(tmp_path):
    config = SVDQuantExportConfig(debug_unpacked=True)

    tensors = collect_svdquant_tensors(_toy_model(), config=config)

    assert "0.residual.weight" in tensors
    output_path = tmp_path / "invalid.safetensors"
    with pytest.raises(ValueError, match="not runtime-loadable"):
        save_svdquant_nunchaku_safetensors(
            _toy_model(), output_path, config=SVDQuantExportConfig(debug_unpacked=True, runtime_loadable=True)
        )
    assert not output_path.exists()


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("weight_dtype", "mx_fp4e2m1", "weight_dtype"),
        ("activation_dtype", "fp16", "activation_dtype"),
        ("scale_dtype", "fp16", "scale_dtype"),
        ("group_size", 16, "group_size"),
        ("group_size", 32.0, "group_size"),
        ("low_rank_dtype", torch.float32, "low_rank_dtype"),
        ("debug_unpacked", 1, "debug_unpacked"),
    ],
)
def test_config_rejects_non_nunchaku_formats(field, value, message):
    with pytest.raises(ValueError, match=message):
        SVDQuantExportConfig(**{field: value})


@pytest.mark.parametrize("group_size", [32.0, True])
def test_mxfp4_residual_provider_requires_non_bool_integer_group_size(group_size):
    with pytest.raises(ValueError, match="group_size must be 32"):
        MXFP4ResidualTensorProvider(group_size=group_size)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("data_type", "int", "data_type"),
        ("bits", 8, "bits=4"),
        ("bits", 4.0, "bits=4"),
        ("group_size", (32, 32), "scalar group_size=32"),
        ("group_size", 64, "scalar group_size=32"),
        ("group_size", 32.0, "scalar group_size=32"),
        ("sym", False, "sym=True"),
        ("act_data_type", "int", "activation data_type"),
        ("act_bits", 16, "activation bits=4"),
        ("act_bits", 4.0, "activation bits=4"),
        ("act_group_size", 64, "activation scalar group_size=32"),
        ("act_group_size", 32.0, "activation scalar group_size=32"),
        ("act_sym", False, "activation sym=True"),
        ("act_dynamic", False, "act_dynamic=True"),
    ],
)
def test_collection_rejects_incompatible_selected_scheme(field, value, message):
    model = _toy_model()
    setattr(model[0].residual_linear, field, value)

    with pytest.raises(ValueError, match=message):
        collect_svdquant_tensors(model)


def test_collection_rejects_missing_selected_weight_or_activation_scheme():
    missing_weight = _toy_model()
    del missing_weight[0].residual_linear.data_type
    with pytest.raises(ValueError, match="missing.*data_type"):
        collect_svdquant_tensors(missing_weight)

    missing_activation = _toy_model()
    for field in ("act_data_type", "act_bits", "act_group_size", "act_sym", "act_dynamic"):
        delattr(missing_activation[0].residual_linear, field)
    with pytest.raises(ValueError, match="missing.*activation"):
        collect_svdquant_tensors(missing_activation)


def test_collection_accepts_normal_autoround_mxfp4_preset_values():
    model = _toy_model()
    model[0].residual_linear.data_type = "mx_fp"
    model[0].residual_linear.act_data_type = "mx_fp"

    tensors = collect_svdquant_tensors(model)

    assert "0.qweight" in tensors


def test_collection_rejects_nonfinite_values_and_mixed_ranks():
    nonfinite = _toy_model()
    nonfinite[0].smooth[0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        collect_svdquant_tensors(nonfinite)

    mixed = torch.nn.Sequential(_toy_model(rank=2)[0], _toy_model(rank=3)[0])
    with pytest.raises(ValueError, match="mixed SVDQuant ranks"):
        collect_svdquant_tensors(mixed)


def test_save_rejects_empty_model_and_missing_runtime_metadata_before_writing(tmp_path):
    empty_path = tmp_path / "empty.safetensors"
    with pytest.raises(ValueError, match="No SVDQuantLinear"):
        save_svdquant_nunchaku_safetensors(torch.nn.Linear(2, 2), empty_path)
    assert not empty_path.exists()

    runtime_path = tmp_path / "runtime.safetensors"
    with pytest.raises(ValueError, match="model_class.*serialized 'config'"):
        save_svdquant_nunchaku_safetensors(
            _toy_model(), runtime_path, config=SVDQuantExportConfig(runtime_loadable=True)
        )
    assert not runtime_path.exists()


def test_runtime_metadata_validation_precedes_source_cpu_copy_and_packing(monkeypatch, tmp_path):
    class Provider:
        def tensors_for(self, record):
            pytest.fail("residual packing must not run before runtime metadata validation")

    def reject_cpu_copy(self, *args, **kwargs):
        pytest.fail("source tensors must remain on their original device before serialization")

    monkeypatch.setattr(torch.Tensor, "cpu", reject_cpu_copy)

    with pytest.raises(ValueError, match="model_class.*serialized 'config'"):
        save_svdquant_nunchaku_safetensors(
            _toy_model(),
            tmp_path / "runtime.safetensors",
            config=SVDQuantExportConfig(runtime_loadable=True),
            residual_provider=Provider(),
        )


def test_collection_retains_source_device_storage_until_after_packing(monkeypatch):
    model = _toy_model()
    events = []
    original_cpu = torch.Tensor.cpu

    class Adapter(IdentitySVDQuantModelAdapter):
        def map_modules(self, received_model, records):
            records = tuple(records)
            events.append("map")
            assert records[0].residual_weight.device == model[0].residual_linear.weight.device
            assert records[0].residual_weight.data_ptr() == model[0].residual_linear.weight.data_ptr()
            return super().map_modules(received_model, records)

    class Provider:
        def __init__(self):
            self.delegate = MXFP4ResidualTensorProvider()

        def tensors_for(self, record):
            events.append("pack")
            return self.delegate.tensors_for(record)

    def tracked_cpu(self, *args, **kwargs):
        events.append("cpu")
        return original_cpu(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", tracked_cpu)

    collect_svdquant_tensors(model, adapter=Adapter(), residual_provider=Provider())

    assert events[0] == "map"
    assert events.index("pack") < events.index("cpu")


def test_runtime_adapter_metadata_and_validation_receive_resolved_rank(tmp_path):
    class RuntimeAdapter(IdentitySVDQuantModelAdapter):
        def metadata(self, model, rank):
            self.metadata_rank = rank
            return {"model_class": "ToyModel", "config": json.dumps({"hidden_size": 65})}

        def validate(self, tensors, metadata):
            self.validated = (set(tensors), metadata["model_class"])

    adapter = RuntimeAdapter()
    output_path = tmp_path / "runtime.safetensors"

    save_svdquant_nunchaku_safetensors(
        _toy_model(), output_path, config=SVDQuantExportConfig(runtime_loadable=True), adapter=adapter
    )

    assert adapter.metadata_rank == 3
    assert adapter.validated[1] == "ToyModel"
    assert "0.qweight" in adapter.validated[0]


def test_adapter_can_expand_records_and_remap_export_suffixes():
    class ExpandingAdapter(IdentitySVDQuantModelAdapter):
        def map_modules(self, model, records):
            source = tuple(records)[0]
            values = {
                "residual_weight": source.residual_weight,
                "lora_down": source.lora_down,
                "lora_up": source.lora_up,
                "smooth": source.smooth,
                "smooth_orig": source.smooth_orig,
                "bias": source.bias,
                "scheme": source.scheme,
                "sources": (source,),
            }
            return (
                SVDQuantExportRecord(prefix="left", key_mapping={"qweight": "packed_weight"}, **values),
                SVDQuantExportRecord(prefix="right", **values),
            )

    tensors = collect_svdquant_tensors(_toy_model(), adapter=ExpandingAdapter())

    assert "left.packed_weight" in tensors
    assert "left.qweight" not in tensors
    assert "right.qweight" in tensors


def test_adapter_can_recompose_independent_sources_at_configured_rank():
    model = torch.nn.Sequential(_toy_model()[0], _toy_model()[0])
    model[0].residual_linear.data_type = "mx_fp4"
    model[0].residual_linear.act_data_type = "mx_fp4"

    assert not torch.equal(model[0].lora_down.weight, model[1].lora_down.weight)
    assert not torch.equal(model[0].lora_up.weight, model[1].lora_up.weight)

    class FusionAdapter(IdentitySVDQuantModelAdapter):
        fused_record = None

        def map_modules(self, model, records):
            left, right = tuple(records)
            rank = left.lora_down.shape[0]
            effective_weights = tuple(
                source.residual_weight + source.lora_up @ source.lora_down for source in (left, right)
            )
            fused_weight = torch.cat(effective_weights, dim=0)
            u, singular_values, vh = torch.linalg.svd(fused_weight.float(), full_matrices=False)
            lora_up = (u[:, :rank] * singular_values[:rank]).to(fused_weight.dtype)
            lora_down = vh[:rank].to(fused_weight.dtype)
            self.fused_record = SVDQuantExportRecord(
                prefix="fused",
                residual_weight=fused_weight - lora_up @ lora_down,
                lora_down=lora_down,
                lora_up=lora_up,
                smooth=left.smooth,
                smooth_orig=left.smooth_orig,
                bias=torch.cat((left.bias, right.bias), dim=0),
                scheme=left.scheme,
                sources=(left, right),
            )
            return (self.fused_record,)

    adapter = FusionAdapter()
    tensors = collect_svdquant_tensors(model, adapter=adapter)

    assert "fused.qweight" in tensors
    assert tensors["fused.qweight"].shape == (128, 64)
    assert adapter.fused_record.lora_down.shape[0] == 3
    expected = torch.cat(
        tuple(
            module.residual_linear.weight.detach() + module.lora_up.weight.detach() @ module.lora_down.weight.detach()
            for module in model
        ),
        dim=0,
    )
    actual = adapter.fused_record.residual_weight + adapter.fused_record.lora_up @ adapter.fused_record.lora_down
    torch.testing.assert_close(actual, expected)


def test_adapter_provenance_rejects_dropped_and_foreign_sources():
    model = torch.nn.Sequential(_toy_model()[0], _toy_model()[0])

    class DroppingAdapter(IdentitySVDQuantModelAdapter):
        def map_modules(self, model, records):
            return super().map_modules(model, tuple(records)[:1])

    with pytest.raises(ValueError, match="dropped logical sources.*1"):
        collect_svdquant_tensors(model, adapter=DroppingAdapter())

    class ForeignAdapter(IdentitySVDQuantModelAdapter):
        def map_modules(self, model, records):
            source = tuple(records)[0]
            record = tuple(super().map_modules(model, (source,)))[0]
            return (replace(record, sources=(replace(source, name="foreign"),)),)

    with pytest.raises(ValueError, match="foreign logical source"):
        collect_svdquant_tensors(_toy_model(), adapter=ForeignAdapter())


def test_adapter_provenance_requires_source_rank_to_equal_output_rank():
    class RankChangingAdapter(IdentitySVDQuantModelAdapter):
        def map_modules(self, model, records):
            source = tuple(records)[0]
            record = tuple(super().map_modules(model, (source,)))[0]
            return (
                replace(
                    record,
                    lora_down=record.lora_down[:2],
                    lora_up=record.lora_up[:, :2],
                ),
            )

    with pytest.raises(ValueError, match="Nunchaku.*configured rank.*exact rank-sum fusion is unsupported"):
        collect_svdquant_tensors(_toy_model(), adapter=RankChangingAdapter())


def test_identity_adapter_uses_stable_model_prefix_for_root_svdquant_linear():
    root = _toy_model()[0]

    tensors = collect_svdquant_tensors(root)

    assert "model.qweight" in tensors
    assert not any(key.startswith(".") for key in tensors)


@pytest.mark.parametrize(
    "payload,message",
    [
        ({"qweight": torch.zeros(128, 64), "wscales": torch.zeros(4, 128, dtype=torch.uint8)}, "qweight"),
        ({"qweight": torch.zeros(128, 64, dtype=torch.int8)}, "qweight.*wscales"),
        (
            {
                "qweight": torch.zeros(128, 64, dtype=torch.int8),
                "wscales": torch.zeros(3, 128, dtype=torch.uint8),
            },
            "wscales shape",
        ),
    ],
)
def test_collection_rejects_malformed_packed_residual_payloads(payload, message):
    class Provider:
        def tensors_for(self, record):
            return payload

    with pytest.raises(ValueError, match=message):
        collect_svdquant_tensors(_toy_model(), residual_provider=Provider())


def test_collection_rejects_aligned_payload_that_is_too_small_for_logical_record():
    class WrongProvider:
        def tensors_for(self, record):
            return {
                "qweight": torch.zeros(128, 64, dtype=torch.int8),
                "wscales": torch.zeros(4, 128, dtype=torch.uint8),
            }

    with pytest.raises(ValueError, match=r"qweight shape.*\(256, 128\)"):
        collect_svdquant_tensors(_toy_model(in_features=129, out_features=129), residual_provider=WrongProvider())


def test_save_svdquant_nunchaku_safetensors_writes_metadata(tmp_path):
    output_path = tmp_path / "svdquant.safetensors"

    save_svdquant_nunchaku_safetensors(_toy_model(), str(output_path))

    with safe_open(output_path, framework="pt") as handle:
        keys = set(handle.keys())
        metadata = handle.metadata()
        quantization_config = json.loads(metadata["quantization_config"])

    assert "0.qweight" in keys
    assert "0.wscales" in keys
    assert "0.residual.weight" not in keys
    assert metadata == {
        "artifact_type": "generic_intermediate",
        "quantization_config": json.dumps(
            {
                "method": "svdquant",
                "weight": {"dtype": "fp4_e2m1_all", "scale_dtype": "ue8m0", "group_size": 32},
                "activation": {"dtype": "fp4_e2m1_all", "scale_dtype": "ue8m0", "group_size": 32},
                "rank": 3,
            },
            sort_keys=True,
        ),
    }
    assert quantization_config == {
        "method": "svdquant",
        "weight": {"dtype": "fp4_e2m1_all", "scale_dtype": "ue8m0", "group_size": 32},
        "activation": {"dtype": "fp4_e2m1_all", "scale_dtype": "ue8m0", "group_size": 32},
        "rank": 3,
    }


def test_svdquant_nunchaku_exporter_has_no_runtime_project_imports():
    import auto_round.export.svdquant_nunchaku as exporter

    source = inspect.getsource(exporter)

    assert "import deepcompressor" not in source
    assert "from deepcompressor" not in source
    assert "import nunchaku" not in source
    assert "from nunchaku" not in source
