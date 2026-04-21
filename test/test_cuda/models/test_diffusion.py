import copy
import os
import re
import shutil

import pytest
import requests
from packaging import version
from PIL import Image

from auto_round import AutoRoundDiffusion

from ...envs import multi_card, require_gptqmodel, require_optimum, require_vlm_env
from ...helpers import get_captions_dataset_path, get_model_path, transformers_version


class TestAutoRound:

    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.fixture(autouse=True, scope="class")
    def setup_and_teardown_class(self):
        yield
        shutil.rmtree("runs", ignore_errors=True)

    def test_diffusion_rtn(self, tiny_flux_model_path):
        from diffusers import AutoPipelineForText2Image

        ## load the model
        pipe = AutoPipelineForText2Image.from_pretrained(tiny_flux_model_path)
        # build tiny model for testing since the full model is too large to quantize and evaluate in CI
        pipe.transformer.transformer_blocks = pipe.transformer.transformer_blocks[:2]
        pipe.transformer.single_transformer_blocks = pipe.transformer.single_transformer_blocks[:2]

        ## quantize the model
        autoround = AutoRoundDiffusion(
            pipe,
            tokenizer=None,
            scheme="MXFP4",
            iters=0,
            disable_opt_rtn=True,
            num_inference_steps=2,
            dataset=get_captions_dataset_path(),
        )
        # skip model saving since it takes much time
        autoround.quantize()

    @require_optimum
    def test_diffusion_tune(self, tiny_flux_model_path, tmp_path):
        from diffusers import AutoPipelineForText2Image

        ## load the model
        pipe = AutoPipelineForText2Image.from_pretrained(tiny_flux_model_path)
        model = pipe.transformer
        # build tiny model for testing since the full model is too large to quantize and evaluate in CI
        pipe.transformer.transformer_blocks = pipe.transformer.transformer_blocks[:2]
        pipe.transformer.single_transformer_blocks = pipe.transformer.single_transformer_blocks[:2]

        layer_config = {}
        # skip some layers since it takes much time
        for n, m in model.named_modules():
            if m.__class__.__name__ != "Linear":
                continue
            match = re.search(r"blocks\.(\d+)", n)
            if match and int(match.group(1)) > 0:
                layer_config[n] = {"bits": 16, "act_bits": 16}

        ## quantize the model
        # https://raw.githubusercontent.com/mlcommons/inference/refs/heads/master/text_to_image/coco2014/captions/captions_source.tsv
        autoround = AutoRoundDiffusion(
            pipe,
            tokenizer=None,
            scheme="MXFP4",
            iters=1,
            nsamples=1,
            num_inference_steps=2,
            layer_config=layer_config,
            dataset=get_captions_dataset_path(),
        )
        # skip model saving since it takes much time
        autoround.quantize_and_save(tmp_path)

    @pytest.mark.skip_ci(reason="Download large model; Time-consuming")
    def test_diffusion_model_checker(self):
        from auto_round.utils import is_diffusion_model

        assert is_diffusion_model(get_model_path("black-forest-labs/FLUX.1-dev"))
        assert is_diffusion_model(get_model_path("sd2-community/stable-diffusion-2-1"))
        assert is_diffusion_model(get_model_path("stabilityai/stable-diffusion-xl-base-1.0"))
        assert is_diffusion_model(get_model_path("Qwen/Qwen3-8B")) is False

    @multi_card
    @pytest.mark.skip_ci(reason="multiple card test")
    def test_diffusion_tune_on_multi_cards(self, tiny_flux_model_path, tmp_path):
        from diffusers import AutoPipelineForText2Image

        ## load the model
        pipe = AutoPipelineForText2Image.from_pretrained(tiny_flux_model_path)
        model = pipe.transformer
        # build tiny model for testing since the full model is too large to quantize and evaluate in CI
        pipe.transformer.transformer_blocks = pipe.transformer.transformer_blocks[:2]
        pipe.transformer.single_transformer_blocks = pipe.transformer.single_transformer_blocks[:2]

        layer_config = {}
        # skip some layers since it takes much time
        for n, m in model.named_modules():
            if m.__class__.__name__ != "Linear":
                continue
            match = re.search(r"blocks\.(\d+)", n)
            if match and int(match.group(1)) > 0:
                layer_config[n] = {"bits": 16, "act_bits": 16}

        ## quantize the model
        # https://raw.githubusercontent.com/mlcommons/inference/refs/heads/master/text_to_image/coco2014/captions/captions_source.tsv
        autoround = AutoRoundDiffusion(
            pipe,
            tokenizer=None,
            scheme="MXFP4",
            iters=1,
            nsamples=1,
            num_inference_steps=2,
            layer_config=layer_config,
            dataset=get_captions_dataset_path(),
            device_map="0,1",
        )
        # skip model saving since it takes much time
        autoround.quantize_and_save(tmp_path)
