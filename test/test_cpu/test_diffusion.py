import copy
import os
import re
import shutil
import sys
import unittest

import requests

sys.path.insert(0, "../..")

from diffusers import AutoPipelineForText2Image
from PIL import Image

from auto_round import AutoRoundDiffusion
from auto_round.testing_utils import require_gptqmodel, require_optimum, require_vlm_env


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model_name = "/tf_dataset/auto_round/models/black-forest-labs/FLUX.1-dev"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("runs", ignore_errors=True)

    @require_optimum
    def test_diffusion_tune(self):
        # Load the model
        pipe = AutoPipelineForText2Image.from_pretrained(self.model_name)
        model = pipe.transformer

        layer_config = {}
        # Skip some layers since it takes much time
        for n, m in model.named_modules():
            if m.__class__.__name__ != "Linear":
                continue
            match = re.search(r"blocks\.(\d+)", n)
            if match and int(match.group(1)) > 0:
                layer_config[n] = {"bits": 16, "act_bits": 16}

        # Quantize the model
        autoround = AutoRoundDiffusion(
            pipe,
            tokenizer=None,
            scheme="MXFP4",
            iters=1,
            nsamples=1,
            num_inference_steps=1,
            layer_config=layer_config,
            dataset="/tf_dataset/auto_round/datasets/coco2014/captions/captions_source.tsv",
            batch_size=1,
        )
        # skip model saving since it takes much time
        autoround.quantize()

    def test_diffusion_rtn(self):
        # Load the model
        pipe = AutoPipelineForText2Image.from_pretrained(self.model_name)

        # Quantize the model
        autoround = AutoRoundDiffusion(
            pipe,
            tokenizer=None,
            scheme="MXFP4",
            iters=0,
            num_inference_steps=2,
            dataset="/tf_dataset/auto_round/datasets/coco2014/captions/captions_source.tsv",
        )
        # Skip model saving since it takes much time
        autoround.quantize()

    def test_diffusion_model_checker(self):
        from auto_round.utils import is_diffusion_model

        self.assertTrue(is_diffusion_model("/tf_dataset/auto_round/models/black-forest-labs/FLUX.1-dev"))
        self.assertFalse(is_diffusion_model("/tf_dataset/auto_round/models/Qwen/Qwen3-8B"))


if __name__ == "__main__":
    unittest.main()
