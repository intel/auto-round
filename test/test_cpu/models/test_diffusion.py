import os
import shutil

import pytest
import torch
from packaging import version

from auto_round import AutoRound
from auto_round.compressors.diffusion_mixin import (
    DiffusionMixin,
    _move_pipeline_to_model_device_for_calibration,
    _pipeline_needs_dtype_alignment,
    _prepare_single_device_pipeline_for_calibration,
)
from auto_round.utils.model import _resolve_diffusion_load_dtype

from ...helpers import get_model_path, transformers_version

flux_name_or_path = get_model_path("black-forest-labs/FLUX.1-dev")


def test_diffusion_model_dtype_is_resolved_before_loading():
    assert _resolve_diffusion_load_dtype("auto", "bf16") is torch.bfloat16
    assert _resolve_diffusion_load_dtype(torch.float16, None) is torch.float16


def test_already_aligned_diffusion_pipeline_does_not_need_post_load_cast():
    pipe = type("Pipeline", (), {})()
    pipe.transformer = torch.nn.Linear(2, 2).to(torch.bfloat16)
    pipe.vae = torch.nn.Linear(2, 2).to(torch.bfloat16)
    pipe.components = {"transformer": pipe.transformer, "vae": pipe.vae}

    assert _pipeline_needs_dtype_alignment(pipe, torch.bfloat16) is False

    pipe.vae = pipe.vae.float()
    assert _pipeline_needs_dtype_alignment(pipe, torch.bfloat16) is True


def test_dispatched_diffusion_pipeline_is_not_moved_during_calibration():
    class Pipeline:
        device = torch.device("cpu")

        def to(self, _device):
            raise AssertionError("a dispatched pipeline must preserve its accelerate hooks")

    model = type(
        "DispatchedModel",
        (),
        {
            "device": torch.device("cuda:0"),
            "hf_device_map": {"transformer_blocks.0": 0, "transformer_blocks.1": 1},
        },
    )()

    assert _move_pipeline_to_model_device_for_calibration(Pipeline(), model) is False


def test_diffusion_calibration_requests_latent_output_when_supported():
    class Pipeline:
        device = torch.device("cpu")

        def __call__(self, prompt, output_type="pil", **kwargs):
            del prompt, output_type, kwargs

    mixin = DiffusionMixin.__new__(DiffusionMixin)
    mixin.guidance_scale = 3.5
    mixin.num_inference_steps = 1
    mixin.generator_seed = None
    mixin.pipeline_call_kwargs = {}
    mixin._requires_calibration_image = lambda: False

    kwargs = mixin._build_pipeline_call_kwargs(Pipeline(), ["test prompt"])

    assert kwargs["output_type"] == "latent"

    mixin.pipeline_call_kwargs = {"output_type": "pil"}
    kwargs = mixin._build_pipeline_call_kwargs(Pipeline(), ["test prompt"])
    assert kwargs["output_type"] == "pil"


def test_single_device_low_gpu_memory_uses_model_cpu_offload():
    class Pipeline:
        def __init__(self):
            self.offload_device = None

        def enable_model_cpu_offload(self, *, device):
            self.offload_device = device

        def enable_group_offload(self, **_kwargs):
            raise AssertionError("single-device calibration must prefer model CPU offload")

        def to(self, _device):
            raise AssertionError("low GPU memory calibration must not move the full pipeline")

    pipe = Pipeline()

    assert _prepare_single_device_pipeline_for_calibration(pipe, "cuda:0", low_gpu_mem_usage=True) == "model"
    assert pipe.offload_device == "cuda:0"


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
    )
    # skip model saving since it takes much time
    autoround.quantize()
    shutil.rmtree(output_dir, ignore_errors=True)


def test_flux_calib(setup_flux):
    pipe, output_dir = setup_flux
    autoround = AutoRound(
        pipe,
        tokenizer=None,
        scheme="NVFP4",
        iters=1,
        num_inference_steps=2,
        nsamples=2,
        dataset="coco2014",
    )
    # skip model saving since it takes much time
    all_inputs = autoround.cache_inter_data(["transformer_blocks.0"], 2)
    assert len(all_inputs["transformer_blocks.0"]["hidden_states"]) == 4
    shutil.rmtree(output_dir, ignore_errors=True)
