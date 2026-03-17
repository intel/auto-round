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
from auto_round.logger import logger


class MLLMMixin:
    """MLLM-specific functionality mixin.

    This mixin adds MLLM-specific functionality to any compressor (CalibCompressor,
    ZeroShotCompressor, ImatrixCompressor, etc). It handles multi-modal models
    (vision-language models) that require special data loading and processing logic.

    Can be combined with:
    - CalibCompressor (for AutoRound with calibration)
    - ImatrixCompressor (for RTN with importance matrix)
    - ZeroShotCompressor (for basic RTN)

    MLLM-specific parameters:
        processor: Multi-modal processor for encoding/decoding data
        image_processor: Image processor for models like LLaVA
        template: Template for processing different MLLMs
        extra_data_dir: Path to extra data (images, audio, videos)
        quant_nontext_module: Whether to quantize non-text modules
    """

    def __init__(
        self,
        *args,
        processor=None,
        image_processor=None,
        template=None,
        extra_data_dir=None,
        quant_nontext_module=False,
        **kwargs,
    ):
        # Store MLLM-specific attributes before calling super().__init__
        self.processor = processor
        self.image_processor = image_processor
        self.template = template
        self.extra_data_dir = extra_data_dir
        self.quant_nontext_module = quant_nontext_module
        self.template_obj = None

        # Call parent class __init__ (will be CalibCompressor, ImatrixCompressor, etc)
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def calib(self, nsamples, bs):
        """Perform MLLM-specific calibration for quantization.

        Override parent's calib method to use MLLM dataset loading logic.
        """
        from transformers import PreTrainedModel

        from auto_round.compressors.mllm.dataset import get_mllm_dataloader
        from auto_round.compressors.mllm.template import get_template
        from auto_round.special_model_handler import MISTRAL_3_2_MODELS

        # Handle template selection
        if isinstance(self.model_context.model, PreTrainedModel):
            model_type = getattr(self.model_context.model.config, "model_type", None)
            if model_type == "llava" and self.template is None:
                self.template = "default"

        if hasattr(self.model_context.model, "name_or_path"):
            name = self.model_context.model.name_or_path
            if any([m in name for m in MISTRAL_3_2_MODELS]):
                self.template = "mistral3_2"

        # Get template
        if self.template is not None:
            self.template_obj = get_template(self.template)
        elif hasattr(self.model_context.model.config, "model_type"):
            self.template_obj = get_template(self.model_context.model.config.model_type)
        else:
            self.template_obj = get_template("default")

        logger.info(f"Using MLLM template: {self.template or 'default'}")

        # Get MLLM dataloader
        self.dataloader = get_mllm_dataloader(
            self.model_context.model,
            self.tokenizer,
            self.dataset,
            self.processor,
            self.image_processor,
            nsamples,
            self.quantize_config.seqlen,
            self.seed,
            bs,
            self.template_obj,
            self.extra_data_dir,
        )

        # Process data through the model for calibration
        total_cnt = 0
        for data in self.dataloader:
            if data is None:
                continue

            # MLLM data is usually already properly formatted
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
