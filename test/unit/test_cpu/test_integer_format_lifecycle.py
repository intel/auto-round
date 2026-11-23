import copy
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("kind", ["gptq", "awq"])
def test_real_integer_packing_preserves_source(kind):
    from auto_round.export.common.packing import pack_layers, prepare_model
    from auto_round.export.formats.awq.packing import pack_layer as awq_pack
    from auto_round.export.formats.gptq.packing import pack_layer as gptq_pack

    model = torch.nn.Sequential(torch.nn.Linear(128, 128, bias=False))
    layer = model[0]
    layer.bits, layer.group_size, layer.sym = 4, 128, True
    layer.scale = torch.ones(128, 1)
    layer.zp = torch.full((128, 1), 8.0)
    original = copy.deepcopy(model.state_dict())
    packed = prepare_model(model, False, device="cpu")
    pack = awq_pack if kind == "awq" else gptq_pack
    pack_layers(
        packed,
        ["0"],
        lambda name, model, device: pack(name, model, backend="auto_round:auto_gptq", device=device),
        device="cpu",
    )
    assert isinstance(model[0], torch.nn.Linear)
    assert torch.equal(model[0].weight, original["0.weight"])
    assert hasattr(packed[0], "qweight")
    assert packed[0].qweight.dtype == torch.int32


def test_prepare_model_copies_before_move():
    from auto_round.export.common.packing import prepare_model

    class Tracked(torch.nn.Linear):
        def to(self, device):
            self.moved = True
            return self

    source = Tracked(2, 2)
    packed = prepare_model(source, False, device="cpu")
    assert packed is not source
    assert packed.moved
    assert not hasattr(source, "moved")


def test_artifacts_no_output():
    from auto_round.export.common.artifacts import save_artifacts

    artifact = SimpleNamespace(save_pretrained=lambda path: pytest.fail("unexpected write"))
    save_artifacts(None, artifact, artifact, artifact)


@pytest.mark.parametrize("kind", ["gptq", "awq"])
@pytest.mark.parametrize("with_output", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_integer_save_real_checkpoint(kind, with_output, native, tmp_path, monkeypatch):
    import importlib
    import json

    module = importlib.import_module(f"auto_round.export.formats.{kind}.format")
    cls = module.AutoGPTQFormat if kind == "gptq" else module.AutoAWQFormat
    fmt = object.__new__(cls)
    fmt.output_format, fmt.backend = ("auto_gptq" if kind == "gptq" else "auto_awq"), None

    if native:
        fmt.output_format = "auto_round:" + fmt.output_format

    class TinyModel(torch.nn.Sequential):
        dtype = torch.float32

        def save_pretrained(self, output_dir, state_dict=None, **kwargs):
            torch.save(self.state_dict() if state_dict is None else state_dict, tmp_path / "weights.pt")

    model = TinyModel(torch.nn.Linear(128, 128, bias=False))
    model.register_parameter("anchor", torch.nn.Parameter(torch.ones(1)))
    model.config = SimpleNamespace()
    layer = model[0]
    layer.bits, layer.group_size, layer.sym = 4, 128, True
    layer.scale, layer.zp = torch.ones(128, 1), torch.full((128, 1), 8.0)
    layer.act_bits = 16
    config = {"bits": 4, "group_size": 128, "sym": True, "data_type": "int", "quant_block_list": [["0"]]}
    config["regex_config"] = {}
    layers = {"0": {"bits": 4, "group_size": 128, "sym": True, "in_blocks": True}}
    if kind == "awq":
        monkeypatch.setattr(module, "_collect_modules_to_not_convert", lambda *args, **kwargs: [])
    output = str(tmp_path) if with_output else None
    packed = fmt.save_quantized(output, model=model, layer_config=layers, serialization_dict=config, inplace=False)
    assert packed is not model
    assert isinstance(model[0], torch.nn.Linear)
    assert hasattr(packed[0], "qweight")
    assert (tmp_path / "weights.pt").exists() == with_output
    if with_output:
        metadata = json.loads((tmp_path / "quantization_config.json").read_text())
        assert metadata["quant_method"] == ("auto-round" if native else kind)
        assert metadata.get("provider", "auto-round") == "auto-round"
        assert "0.qweight" in torch.load(tmp_path / "weights.pt", weights_only=True)


def test_pack_layers_meta_and_snapshot():
    from auto_round.export.common.packing import pack_layers

    meta = torch.nn.Linear(2, 2, device="meta")
    pack_layers(meta, ["weight"], lambda *args, **kwargs: pytest.fail("meta packing"))
    names = ["0", "1"]
    seen = []

    def pack(name, model, device=None):
        seen.append(name)
        names.clear()

    pack_layers(torch.nn.Linear(2, 2), names, pack)
    assert seen == ["0", "1"]


@pytest.mark.parametrize("forced", [False, True])
def test_awq_unservable_layer_honors_explicit_user_config(forced):
    from auto_round.export.formats.awq.packing import pack_layer

    model = torch.nn.Sequential(torch.nn.Linear(128, 64, bias=False))
    layer = model[0]
    layer.bits, layer.group_size, layer.sym = 4, 128, True
    layer.scale, layer.zp = torch.ones(64, 1), torch.full((64, 1), 8.0)
    layer.fixed_by_user = forced
    pack_layer("0", model, backend="auto_awq", device="cpu")
    assert hasattr(model[0], "qweight") == forced
    assert isinstance(model[0], torch.nn.Linear) != forced
