import copy
import os
import re
import shutil

import pytest
import requests
from diffusers import AutoPipelineForText2Image
from PIL import Image

from auto_round import AutoRoundDiffusion
from auto_round.testing_utils import require_gptqmodel, require_optimum, require_vlm_env


class TestAutoRound:
    @classmethod
    def setup_class(self):
        self.model_name = "/dataset/FLUX.1-dev"

    @classmethod
    def teardown_class(self):
        shutil.rmtree("runs", ignore_errors=True)

    @require_optimum
    def test_diffusion_tune(self):
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

    def test_diffusion_rtn(self):
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

        self.assertTrue(is_diffusion_model("/dataset/FLUX.1-dev"))
        self.assertTrue(is_diffusion_model("/models/stable-diffusion-2-1"))
        self.assertTrue(is_diffusion_model("/models/stable-diffusion-xl-base-1.0"))
        self.assertFalse(is_diffusion_model("/models/Qwen3-8B"))
