import shutil

import pytest

from packaging import version

from auto_round import AutoRound

from ...helpers import get_model_path, transformers_version

flux_name_or_path = get_model_path("black-forest-labs/FLUX.1-dev")


@pytest.fixture
def setup_flux():
    """Fixture to set up the Flux model and tokenizer."""
    from diffusers import AutoPipelineForText2Image
    model_name = flux_name_or_path
    pipe = AutoPipelineForText2Image.from_pretrained(model_name)
    output_dir = "./tmp/test_quantized_flux"
    return pipe, output_dir


@pytest.mark.skipif(
    transformers_version >= version.parse("5.0.0"),
    reason="cannot import name 'MT5Tokenizer' from 'transformers', https://github.com/huggingface/diffusers/issues/13035",
)
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
