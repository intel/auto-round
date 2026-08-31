import json
import os
import shutil
from test.helpers import get_model_path, transformers_version
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from packaging import version

from auto_round import AutoRound
from auto_round.calibration.diffusion import _prepare_pipeline_for_calibration
from auto_round.utils.model import diffusion_load_model

flux_name_or_path = get_model_path("black-forest-labs/FLUX.1-dev")


def test_diffusion_load_preserves_declared_fp32_modules(tmp_path):
    from diffusers import WanTransformer3DModel

    transformer = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=8,
        ffn_dim=32,
        num_layers=1,
        rope_max_seq_len=16,
    )
    transformer.save_pretrained(tmp_path / "transformer")
    model_index = {
        "_class_name": "TestPipeline",
        "transformer": ["diffusers", "WanTransformer3DModel"],
    }
    (tmp_path / "model_index.json").write_text(json.dumps(model_index))

    class Config(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    class TestPipeline:
        def __init__(self, model):
            self.transformer = model
            self.config = Config()
            self.components = {"transformer": model}

        @classmethod
        def from_pretrained(cls, path, torch_dtype):
            transformer_dtype = torch_dtype["transformer"] if isinstance(torch_dtype, dict) else torch_dtype
            model = WanTransformer3DModel.from_pretrained(
                os.path.join(path, "transformer"), torch_dtype=transformer_dtype
            )
            return cls(model)

        @staticmethod
        def load_config(path):
            return json.loads((tmp_path / "model_index.json").read_text())

    pipelines = SimpleNamespace(pipeline_utils=SimpleNamespace(DiffusionPipeline=TestPipeline))
    with patch("auto_round.utils.common.LazyImport", return_value=pipelines):
        for load_kwargs in ({"default_torch_dtype": torch.bfloat16}, {"model_dtype": "bf16"}):
            _, loaded = diffusion_load_model(str(tmp_path), **load_kwargs)

            assert loaded.dtype == torch.bfloat16
            assert loaded.patch_embedding.weight.dtype == torch.bfloat16
            assert loaded.condition_embedder.time_embedder.linear_1.weight.dtype == torch.float32
            assert loaded.blocks[0].norm2.weight.dtype == torch.float32


def test_low_gpu_memory_diffusion_calibration_uses_model_cpu_offload():
    class Pipeline:
        device = torch.device("cpu")

        def __init__(self):
            self.offload_device = None

        def enable_model_cpu_offload(self, *, device):
            self.offload_device = device

        def to(self, _device):
            raise AssertionError("low GPU memory calibration must not move the full pipeline")

    pipe = Pipeline()

    mode = _prepare_pipeline_for_calibration(pipe, "cuda:0", low_gpu_mem_usage=True)

    assert mode == "model"
    assert pipe.offload_device == torch.device("cuda:0")


def test_regular_diffusion_calibration_moves_pipeline_to_device():
    class Pipeline:
        device = None

        def __init__(self):
            self.target_device = None

        def to(self, device):
            self.target_device = device

    pipe = Pipeline()
    target_device = torch.device("cpu")

    mode = _prepare_pipeline_for_calibration(pipe, target_device, low_gpu_mem_usage=False)

    assert mode is None
    assert pipe.target_device == target_device


def test_regular_diffusion_calibration_moves_all_components_when_pipeline_reports_target_device():
    class Pipeline:
        device = torch.device("cuda:0")

        def __init__(self):
            self.target_device = None

        def to(self, device):
            self.target_device = device

    pipe = Pipeline()
    target_device = torch.device("cuda:0")

    _prepare_pipeline_for_calibration(pipe, target_device, low_gpu_mem_usage=False)

    assert pipe.target_device == target_device


@pytest.fixture
def setup_flux():
    """Fixture to set up the Flux model and tokenizer."""
    from diffusers import AutoPipelineForText2Image

    model_name = flux_name_or_path
    # use bf16 to reduce the saved model size
    pipe = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    output_dir = "./tmp/test_quantized_flux"
    return pipe, output_dir


@pytest.mark.timeout(120)
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


@pytest.mark.timeout(150)
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
