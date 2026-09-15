import json

import pytest
import torch
from safetensors import safe_open

from auto_round.algorithms.transforms.svdquant.wrapper import SVDQuantLinear
from auto_round.export.svdquant_mxfp4 import unpack_lowrank_weight
from auto_round.export.svdquant_nunchaku import (
    MXFP4ResidualTensorProvider,
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
    expected_runtime_down = logical_down * model[0].smooth.to(torch.bfloat16).unsqueeze(0)
    torch.testing.assert_close(unpacked_down[:3, :65], expected_runtime_down)
    torch.testing.assert_close(unpacked_up[:7, :3], logical_up)
    torch.testing.assert_close(
        unpack_nunchaku_16bit_vector(tensors["0.smooth"])[:65], model[0].smooth.reciprocal().to(torch.bfloat16)
    )
    torch.testing.assert_close(
        unpack_nunchaku_16bit_vector(tensors["0.bias"])[:7],
        model[0].residual_linear.bias.detach().to(torch.bfloat16),
    )


def test_bias_and_smooth_vectors_match_layout_fixture_and_identity_padding():
    model = _toy_model(bias=False)
    model[0].register_buffer("smooth_orig", torch.arange(1, 66, dtype=torch.float32))

    tensors = collect_svdquant_tensors(model)

    bias = unpack_nunchaku_16bit_vector(tensors["0.bias"])
    assert torch.count_nonzero(bias[:7]) == 0
    assert torch.equal(bias[7:], torch.ones_like(bias[7:]))
    smooth_orig = unpack_nunchaku_16bit_vector(tensors["0.smooth_orig"])
    torch.testing.assert_close(
        smooth_orig[:65], torch.arange(1, 66, dtype=torch.float32).reciprocal().to(torch.bfloat16)
    )
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


def test_config_rejects_non_nunchaku_formats():
    cases = (
        ("weight_dtype", "mx_fp4e2m1", "weight_dtype"),
        ("activation_dtype", "fp16", "activation_dtype"),
        ("scale_dtype", "fp16", "scale_dtype"),
        ("group_size", 16, "group_size"),
        ("low_rank_dtype", torch.float32, "low_rank_dtype"),
        ("debug_unpacked", 1, "debug_unpacked"),
    )
    for field, value, message in cases:
        with pytest.raises(ValueError, match=message):
            SVDQuantExportConfig(**{field: value})


def test_collection_rejects_incompatible_selected_scheme():
    cases = (
        ("data_type", "int", "data_type"),
        ("bits", 8, "bits=4"),
        ("group_size", 64, "scalar group_size=32"),
        ("sym", False, "sym=True"),
        ("act_data_type", "int", "activation data_type"),
        ("act_bits", 16, "activation bits=4"),
        ("act_group_size", 64, "activation scalar group_size=32"),
        ("act_sym", False, "activation sym=True"),
        ("act_dynamic", False, "act_dynamic=True"),
    )
    for field, value, message in cases:
        model = _toy_model()
        setattr(model[0].residual_linear, field, value)
        with pytest.raises(ValueError, match=message):
            collect_svdquant_tensors(model)

    with pytest.raises(ValueError, match="group_size must be 32"):
        MXFP4ResidualTensorProvider(group_size=True)


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


def test_collection_rejects_malformed_packed_residual_payloads():
    cases = (
        ({"qweight": torch.zeros(128, 64), "wscales": torch.zeros(4, 128, dtype=torch.uint8)}, "qweight"),
        ({"qweight": torch.zeros(128, 64, dtype=torch.int8)}, "qweight.*wscales"),
        (
            {
                "qweight": torch.zeros(64, 64, dtype=torch.int8),
                "wscales": torch.zeros(4, 128, dtype=torch.uint8),
            },
            "qweight shape",
        ),
        (
            {
                "qweight": torch.zeros(128, 64, dtype=torch.int8),
                "wscales": torch.zeros(3, 128, dtype=torch.uint8),
            },
            "wscales shape",
        ),
    )
    for payload, message in cases:

        class Provider:
            def tensors_for(self, record):
                return payload

        with pytest.raises(ValueError, match=message):
            collect_svdquant_tensors(_toy_model(), residual_provider=Provider())


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
