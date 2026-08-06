import os
import shutil
import sys
import types
from collections import UserDict

import pytest
import torch
from packaging import version

from auto_round import AutoRound
from auto_round.utils import model as model_utils

from ...helpers import get_model_path, transformers_version

flux_name_or_path = get_model_path("black-forest-labs/FLUX.1-dev")


@pytest.fixture
def setup_flux():
    """Fixture to set up the Flux model and tokenizer."""
    from diffusers import AutoPipelineForText2Image

    model_name = flux_name_or_path
    # use bf16 to reduce the saved model size
    pipe = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    output_dir = "./tmp/test_quantized_flux"
    return pipe, output_dir


def test_flux_saving(setup_flux):
    pipe, output_dir = setup_flux
    autoround = AutoRound(
        pipe,
        tokenizer=None,
        scheme="W4A16",
        iters=0,
        num_inference_steps=2,
        disable_opt_rtn=True,
    )
    autoround.quantize_and_save(output_dir)
    assert os.path.exists(os.path.join(output_dir, "model_index.json"))
    assert os.path.exists(os.path.join(output_dir, "transformer", "quantization_config.json"))
    shutil.rmtree(output_dir, ignore_errors=True)


def test_flux(setup_flux):
    pipe, output_dir = setup_flux
    autoround = AutoRound(
        pipe,
        tokenizer=None,
        scheme="MXFP4",
        iters=0,
        num_inference_steps=2,
        disable_opt_rtn=True,  # We change the logic, for opt-rtn, we always do calibration which is slow on cpu
    )
    # skip model saving since it takes much time
    autoround.quantize()
    shutil.rmtree(output_dir, ignore_errors=True)


class _AttrConfig(UserDict):
    def __getattr__(self, name):
        try:
            return self.data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        if name in {"data", "save_pretrained"}:
            super().__setattr__(name, value)
        else:
            self.data[name] = value


class _FakeSaveableModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = _AttrConfig(config)
        hidden_size = int(config.get("hidden_size", 8))
        self.proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "diffusion_pytorch_model.bin"))


class FakeTransformer2DModel(_FakeSaveableModel):
    @classmethod
    def from_config(cls, config):
        return cls(config)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        import json

        json.dump(payload, f)


def _make_random_init_repo(tmp_path):
    model_dir = tmp_path / "mini_max_dummy"
    model_index = {
        "_class_name": "MiniMaxDummyPipeline",
        "_diffusers_version": "0.0.0",
        "transformer": ["diffusers", "FakeTransformer2DModel"],
        "processor": ["custom", "DummyProcessor"],
        "tokenizer": ["custom", "DummyTokenizer"],
    }
    _write_json(model_dir / "model_index.json", model_index)
    _write_json(model_dir / "transformer" / "config.json", {"hidden_size": 8, "torch_dtype": "float32"})
    _write_json(model_dir / "processor" / "processor_config.json", {"type": "dummy"})
    _write_json(model_dir / "tokenizer" / "tokenizer_config.json", {"type": "dummy"})
    with open(model_dir / "README.md", "w", encoding="utf-8") as f:
        f.write("dummy pipeline\n")
    return model_dir


def test_diffusion_random_init_loads_transformer_and_preserves_metadata(tmp_path, monkeypatch):
    model_dir = _make_random_init_repo(tmp_path)
    fake_diffusers = types.SimpleNamespace(FakeTransformer2DModel=FakeTransformer2DModel)

    monkeypatch.setattr(model_utils, "check_diffusers_installed", lambda: None)
    monkeypatch.setattr(model_utils, "_check_accelerate_version", lambda: None)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    pipe, model = model_utils.diffusion_load_model(str(model_dir), init_mode="random", device="cpu")

    assert model is pipe.transformer
    assert isinstance(model, FakeTransformer2DModel)

    output_dir = tmp_path / "saved"
    model.save_pretrained(output_dir / "transformer")
    pipe.processor.save_pretrained(output_dir / "processor")
    pipe.tokenizer.save_pretrained(output_dir / "tokenizer")
    pipe.save_config(output_dir)

    assert (output_dir / "transformer" / "config.json").exists()
    assert (output_dir / "processor" / "processor_config.json").exists()
    assert (output_dir / "tokenizer" / "tokenizer_config.json").exists()
    assert (output_dir / "README.md").exists()
    assert (output_dir / "model_index.json").exists()


def test_diffusion_random_init_requires_transformer_dir(tmp_path, monkeypatch):
    model_dir = tmp_path / "missing_transformer"
    _write_json(
        model_dir / "model_index.json",
        {
            "_class_name": "MiniMaxDummyPipeline",
            "transformer": ["diffusers", "FakeTransformer2DModel"],
        },
    )

    monkeypatch.setattr(model_utils, "check_diffusers_installed", lambda: None)
    monkeypatch.setattr(model_utils, "_check_accelerate_version", lambda: None)
    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(FakeTransformer2DModel=FakeTransformer2DModel))

    with pytest.raises(FileNotFoundError, match="transformer/ directory"):
        model_utils.diffusion_load_model(str(model_dir), init_mode="random", device="cpu")


# def test_flux_calib(setup_flux):
#     pipe, output_dir = setup_flux
#     autoround = AutoRound(
#         pipe,
#         tokenizer=None,
#         scheme="NVFP4",
#         iters=1,
#         num_inference_steps=2,
#         nsamples=2,
#         dataset="coco2014",
#     )
#     # skip model saving since it takes much time
#     all_inputs = autoround.cache_inter_data(["transformer_blocks.0"], 2)
#     assert len(all_inputs["transformer_blocks.0"]["hidden_states"]) == 4
#     shutil.rmtree(output_dir, ignore_errors=True)
