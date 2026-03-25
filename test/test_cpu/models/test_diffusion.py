import os
import shutil

import pytest
import torch
from packaging import version

from auto_round import AutoRound

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
    assert len(all_inputs['transformer_blocks.0']['hidden_states']) == 4
    shutil.rmtree(output_dir, ignore_errors=True)
