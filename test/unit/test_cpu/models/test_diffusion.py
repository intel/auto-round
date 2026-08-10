import os
import shutil
from test.helpers import get_model_path, transformers_version

import pytest
import torch
from packaging import version

from auto_round import AutoRound
from auto_round.calibration.diffusion import _prepare_pipeline_for_calibration

flux_name_or_path = get_model_path("black-forest-labs/FLUX.1-dev")


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
        device = torch.device("cpu")

        def __init__(self):
            self.target_device = None

        def to(self, device):
            self.target_device = device

    pipe = Pipeline()

    mode = _prepare_pipeline_for_calibration(pipe, "cuda:0", low_gpu_mem_usage=False)

    assert mode is None
    assert pipe.target_device == "cuda:0"


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
