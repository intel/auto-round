import json

import pytest
import torch

from auto_round.export.svdquant_adapters import resolve_svdquant_model_adapter
from auto_round.export.svdquant_adapters.wan import WAN_SVDQUANT_TARGET_MODULES, WanSVDQuantNunchakuAdapter
from auto_round.export.svdquant_nunchaku import SourceLinearRecord, SVDQuantLinearScheme

SCHEME = SVDQuantLinearScheme("mx_fp", 4, 32, True, "mx_fp", 4, 32, True, True)


def _model(num_layers=1):
    model = torch.nn.Module()
    model.config = {"_class_name": "WanTransformer3DModel", "num_layers": num_layers}
    return model


def _source(name):
    return SourceLinearRecord(
        name=name,
        residual_weight=torch.empty(64, 64),
        lora_down=torch.empty(16, 64),
        lora_up=torch.empty(64, 16),
        smooth=torch.ones(64),
        smooth_orig=torch.ones(64),
        bias=torch.empty(64),
        scheme=SCHEME,
    )


def test_auto_resolves_wan_adapter():
    adapter = resolve_svdquant_model_adapter("auto", _model())

    assert isinstance(adapter, WanSVDQuantNunchakuAdapter)


def test_wan_adapter_maps_every_projection_without_renaming():
    sources = tuple(_source(f"blocks.0.{path}") for path in WAN_SVDQUANT_TARGET_MODULES)
    adapter = WanSVDQuantNunchakuAdapter(config=_model().config, require_complete_model=True)

    records = tuple(adapter.map_modules(_model(), sources))

    assert tuple(record.prefix for record in records) == tuple(source.name for source in sources)
    assert all(record.sources == (source,) for record, source in zip(records, sources, strict=True))
    metadata = adapter.metadata(_model(), rank=16)
    assert metadata["model_class"] == "NunchakuWanTransformer3DModel"
    assert json.loads(metadata["config"])["num_layers"] == 1


def test_wan_adapter_rejects_incomplete_projection_set():
    adapter = WanSVDQuantNunchakuAdapter(config=_model().config, require_complete_model=True)

    with pytest.raises(ValueError, match="complete Wan projection mismatch"):
        tuple(adapter.map_modules(_model(), (_source("blocks.0.attn1.to_q"),)))


def test_wan_adapter_rejects_non_runtime_projection():
    adapter = WanSVDQuantNunchakuAdapter(config=_model().config, require_complete_model=False)

    with pytest.raises(ValueError, match="unrecognized Wan SVDQuant source"):
        tuple(adapter.map_modules(_model(), (_source("blocks.0.time_embedder.linear_1"),)))


def test_wan_extra_tensors_preserve_declared_fp32_values():
    model = _model()
    model._keep_in_fp32_modules = ["time_embedder", "scale_shift_table"]
    model.time_embedder = torch.nn.Linear(2, 2)
    model.proj_out = torch.nn.Linear(2, 2)
    model.register_buffer("scale_shift_table", torch.full((2,), 1.001, dtype=torch.float32))
    with torch.no_grad():
        model.time_embedder.weight.fill_(1.001)

    tensors = WanSVDQuantNunchakuAdapter().extra_tensors(model)

    for key in ("time_embedder.weight", "scale_shift_table"):
        assert tensors[key].dtype == torch.float32
        torch.testing.assert_close(tensors[key], model.state_dict()[key], rtol=0, atol=0)
    assert tensors["proj_out.weight"].dtype == torch.bfloat16
