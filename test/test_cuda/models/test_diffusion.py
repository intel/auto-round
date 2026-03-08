import copy
import os
import re
import shutil

import pytest
import requests
from packaging import version
from PIL import Image

from auto_round import AutoRoundDiffusion

from ...envs import require_gptqmodel, require_optimum, require_vlm_env
from ...helpers import transformers_version


class TestAutoRound:
    model_name = "/dataset/FLUX.1-dev"

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        # ===== SETUP (setup_class) =====
        print("[Setup] Running before any test in class")

        # Yield to hand control to the test methods
        yield

        # ===== TEARDOWN (teardown_class) =====
        print("[Teardown] Running after all tests in class")
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    @require_optimum
    @pytest.mark.skipif(
        transformers_version >= version.parse("5.0.0"),
        reason="cannot import name 'MT5Tokenizer' from 'transformers', https://github.com/huggingface/diffusers/issues/13035",
    )
    def test_diffusion_tune(self):
        from diffusers import AutoPipelineForText2Image

        ## load the model
        pipe = AutoPipelineForText2Image.from_pretrained(self.model_name).to("cuda")
        model = pipe.transformer

        layer_config = {}
        # skip some layers since it takes much time
        for n, m in model.named_modules():
            if m.__class__.__name__ != "Linear":
                continue
            match = re.search(r"blocks\.(\d+)", n)
            if match and int(match.group(1)) > 0:
                layer_config[n] = {"bits": 16, "act_bits": 16}

        ## quantize the model
        autoround = AutoRoundDiffusion(
            pipe,
            tokenizer=None,
            scheme="MXFP4",
            iters=1,
            nsamples=1,
            num_inference_steps=2,
            layer_config=layer_config,
            dataset="/dataset/captions_source.tsv",
        )
        # skip model saving since it takes much time
        autoround.quantize()

    @pytest.mark.skipif(
        transformers_version >= version.parse("5.0.0"),
        reason="cannot import name 'MT5Tokenizer' from 'transformers', https://github.com/huggingface/diffusers/issues/13035",
    )
    def test_diffusion_rtn(self):
        from diffusers import AutoPipelineForText2Image

        ## load the model
        pipe = AutoPipelineForText2Image.from_pretrained(self.model_name)

        ## quantize the model
        autoround = AutoRoundDiffusion(
            pipe,
            tokenizer=None,
            scheme="MXFP4",
            iters=0,
            num_inference_steps=2,
            dataset="/dataset/captions_source.tsv",
        )
        # skip model saving since it takes much time
        autoround.quantize()

    def test_diffusion_model_checker(self):
        from auto_round.utils import is_diffusion_model

        assert is_diffusion_model("/dataset/FLUX.1-dev")
        assert is_diffusion_model("/models/stable-diffusion-2-1")
        assert is_diffusion_model("/models/stable-diffusion-xl-base-1.0")
        assert is_diffusion_model("/models/Qwen3-8B") is False
