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

import torch

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
        processor: Multi-modal processor override (normally loaded by ModelContext)
        image_processor: Image processor override (e.g. for LLaVA)
        template: Template name for processing different MLLMs
        extra_data_dir: Path to extra data (images, audio, videos)
        quant_nontext_module: Whether to quantize non-text modules

    Design note:
        ``ModelContext._load_model()`` is responsible for loading the model and its
        associated artifacts (processor, tokenizer, image_processor).  This mixin
        reads those artifacts from ``self.model_context`` during calibration.
        If the caller passes explicit ``processor`` / ``image_processor`` overrides,
        they are written into ``model_context`` after ``super().__init__()`` so that
        ``model_context`` remains the single source of truth.
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
        self.template = template
        self.extra_data_dir = extra_data_dir
        self.quant_nontext_module = quant_nontext_module
        self.template_obj = None

        # Pass quant_nontext_module to ModelContext so get_block_names can include vision blocks
        kwargs.setdefault("quant_nontext_module", quant_nontext_module)

        # super().__init__() creates model_context, which eagerly loads the model and
        # populates model_context.processor / image_processor / tokenizer.
        super().__init__(*args, **kwargs)

        # Apply user-provided overrides into model_context (single source of truth).
        if processor is not None:
            self.model_context.processor = processor
        if image_processor is not None:
            self.model_context.image_processor = image_processor

    @torch.no_grad()
    def calib(self, nsamples, bs):
        """Perform MLLM-specific calibration for quantization.

        Override parent's calib method to use MLLM dataset loading logic.
        All multimodal artifacts are read from ``self.model_context``.
        """
        from transformers import PreTrainedModel

        from auto_round.compressors.mllm.dataset import get_mllm_dataloader
        from auto_round.compressors.mllm.template import get_template
        from auto_round.special_model_handler import MISTRAL_3_2_MODELS

        mc = self.model_context
        processor = mc.processor
        image_processor = mc.image_processor
        tokenizer = mc.tokenizer

        # Handle template selection
        if isinstance(mc.model, PreTrainedModel):
            model_type = getattr(mc.model.config, "model_type", None)
            if model_type == "llava" and self.template is None:
                self.template = "default"

        if hasattr(mc.model, "name_or_path"):
            name = mc.model.name_or_path
            if any([m in name for m in MISTRAL_3_2_MODELS]):
                self.template = "mistral3_2"

        template_name = self.template
        if template_name is None and hasattr(mc.model.config, "model_type"):
            template_name = mc.model.config.model_type
        if template_name is None:
            template_name = "default"

        self.template_obj = get_template(
            template_name,
            model=mc.model,
            tokenizer=tokenizer,
            processor=processor,
            image_processor=image_processor,
            use_rtn=getattr(self.quantize_config, "iters", None) == 0,
            quiet=not self.quant_nontext_module,
        )

        logger.info(f"Using MLLM template: {template_name}")

        dataset = self.dataset.replace(" ", "") if isinstance(self.dataset, str) else self.dataset
        if dataset is None:
            dataset = self.template_obj.default_dataset

        if isinstance(self.dataset, str):
            dataset = self.dataset.replace(" ", "")
            (
                self.dataloader,
                self.batch_size,
                self.seqlen,
                self.gradient_accumulate_steps,
            ) = get_mllm_dataloader(
                template=self.template_obj,
                model=mc.model,
                tokenizer=tokenizer,
                processor=processor,
                image_processor=image_processor,
                dataset=dataset,
                extra_data_dir=self.extra_data_dir,
                seqlen=self.seqlen,
                bs=bs,
                seed=self.seed,
                nsamples=nsamples,
                quant_nontext_module=self.quant_nontext_module,
            )

        # Process data through the model for calibration
        total_cnt = 0
        for data in self.dataloader:
            if data is None:
                continue

            if isinstance(data, dict):
                data_new = {
                    key: value.to(mc.model.device) if isinstance(value, torch.Tensor) else value
                    for key, value in data.items()
                }
            else:
                data_new = data

            try:
                if isinstance(data_new, dict):
                    mc.model(**data_new)
                else:
                    mc.model(data_new)
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

    def save_quantized(self, output_dir=None, format="auto_round", inplace=True, **kwargs):
        """Save the quantized model to the specified output directory in the specified format.

        Args:
            output_dir (str, optional): The directory to save the quantized model. Defaults to None.
            format (str, optional): The format in which to save the model. Defaults to "auto_round".
            inplace (bool, optional): Whether to modify the model in place. Defaults to True.
            **kwargs: Additional keyword arguments specific to the export format.

        Returns:
            object: The compressed model object.
        """
        mc = self.model_context
        processor = mc.processor
        image_processor = mc.image_processor
        tokenizer = mc.tokenizer

        if processor is not None and not hasattr(processor, "chat_template"):
            processor.chat_template = None
        compressed_model = super().save_quantized(
            output_dir=output_dir,
            format=format,
            inplace=inplace,
            processor=processor,
            image_processor=image_processor,
            quant_nontext_module=self.quant_nontext_module if hasattr(self, "quant_nontext_module") else False,
            **kwargs,
        )
        return compressed_model
