# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

import torch

from auto_round.algorithms.alg_config import AlgConfig
from auto_round.context.model import ModelContext
from auto_round.logger import logger


class DiffusionMixin:
    """Diffusion-specific functionality mixin.

    This mixin adds diffusion model-specific functionality to any compressor
    (CalibCompressor, ZeroShotCompressor, ImatrixCompressor, etc). It handles
    diffusion models (like Stable Diffusion, FLUX) that require special pipeline
    handling and data generation logic.

    Can be combined with:
    - CalibCompressor (for AutoRound with calibration)
    - ImatrixCompressor (for RTN with importance matrix)
    - ZeroShotCompressor (for basic RTN)

    Diffusion-specific parameters:
        guidance_scale: Control how much image generation follows text prompt
        num_inference_steps: Reference number of denoising steps
        generator_seed: Seed for initial noise generation
    """

    def __init__(self, *args, guidance_scale=7.5, num_inference_steps=50, generator_seed=None, **kwargs):
        # Store diffusion-specific attributes
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.generator_seed = generator_seed
        self.pipe = None  # Will be set during model loading
        self.pipe_config = None

        # Call parent class __init__ (will be CalibCompressor, ImatrixCompressor, etc)
        super().__init__(*args, **kwargs)

    def post_init(self):
        """Override post_init to handle diffusion-specific model loading."""
        # Load diffusion model as pipeline before standard initialization
        if isinstance(self.model_context.model, str):
            self._load_diffusion_model()

        # Continue with standard post_init
        super().post_init()

    def _load_diffusion_model(self):
        """Load diffusion model using pipeline.

        This method loads the full diffusion pipeline and extracts the
        transformer/unet component for quantization.
        """
        from auto_round.utils import diffusion_load_model

        if isinstance(self.model_context.model, str):
            # Load diffusion pipeline
            logger.info(f"Loading diffusion model from {self.model_context.model}")
            pipe, pipe_config = diffusion_load_model(
                pretrained_model_name_or_path=self.model_context.model,
                platform=self.platform,
                device=self.compress_context.device,
                trust_remote_code=self.model_context.trust_remote_code,
            )
            self.pipe = pipe
            self.pipe_config = pipe_config

            # Extract the transformer/unet component as the model
            if hasattr(pipe, "transformer"):
                extracted_model = pipe.transformer
                logger.info("Extracted transformer from diffusion pipeline")
            elif hasattr(pipe, "unet"):
                extracted_model = pipe.unet
                logger.info("Extracted unet from diffusion pipeline")
            else:
                raise ValueError("Cannot find transformer or unet in diffusion pipeline")

            # Replace the model path with the actual model
            self.model_context.model = extracted_model

    @torch.no_grad()
    def calib(self, nsamples, bs):
        """Perform diffusion-specific calibration for quantization.

        Override parent's calib method to use diffusion dataset loading logic.
        """
        from auto_round.compressors.diffusion.dataset import get_diffusion_dataloader

        if self.pipe is None:
            raise ValueError("Diffusion pipeline must be loaded before calibration")

        logger.info(f"Preparing diffusion dataloader with {nsamples} samples")

        # Get diffusion dataloader
        self.dataloader = get_diffusion_dataloader(
            pipe=self.pipe,
            dataset=self.dataset,
            nsamples=nsamples,
            batch_size=bs,
            seed=self.seed,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            generator_seed=self.generator_seed,
        )

        # Process data through the model for calibration
        total_cnt = 0
        for data in self.dataloader:
            if data is None:
                continue

            # Diffusion data is usually already properly formatted
            if isinstance(data, dict):
                # Move all tensors to device
                data_new = {}
                for key, value in data.items():
                    if isinstance(value, torch.Tensor):
                        data_new[key] = value.to(self.model_context.model.device)
                    else:
                        data_new[key] = value
            else:
                data_new = data

            try:
                if isinstance(data_new, dict):
                    self.model_context.model(**data_new)
                else:
                    self.model_context.model(data_new)
            except NotImplementedError:
                pass
            except Exception as e:
                logger.warning(f"Calibration forward pass failed: {e}")

            total_cnt += bs
            if total_cnt >= nsamples:
                break

        if total_cnt == 0:
            logger.error("no data has been cached, please provide more data")
            exit(-1)
        elif total_cnt < nsamples:
            logger.warning(f"Insufficient number of samples: required {nsamples}, but only {total_cnt} were processed.")
