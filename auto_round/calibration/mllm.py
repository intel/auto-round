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
"""MLLM (vision-language) calibration strategy.

Inherits :class:`LLMCalibrator` to reuse ``collect`` / ``cache_inter_data``
and overrides :meth:`calib` to drive the model with multimodal data.

MLLM-specific runtime state (``template`` / ``extra_data_dir`` /
``quant_nontext_module`` / ``template_obj``) lives on the *Compressor*
(populated by ``MLLMMixin.__init__``) and is read here through
``self.compressor``.
"""

import torch

from auto_round.calibration.llm import LLMCalibrator
from auto_round.calibration.register import register_calibrator
from auto_round.logger import logger
from auto_round.utils import to_device


@register_calibrator("mllm")
class MLLMCalibrator(LLMCalibrator):
    """Calibrator for multimodal (vision-language) models."""

    def __init__(self, compressor):
        super().__init__(compressor)
        self.processor = compressor.processor
        self.image_processor = compressor.image_processor
        self.template = compressor.template
        self.quant_nontext_module = compressor.quant_nontext_module
        self.extra_data_dir = compressor.extra_data_dir

    @torch.no_grad()
    def calib(self, nsamples: int, bs: int) -> None:
        """Drive the multimodal model so block-forward hooks fire.

        Verbatim port of the legacy ``MLLMMixin.calib``.
        """
        from transformers import PreTrainedModel

        from auto_round.compressors.mllm.dataset import get_mllm_dataloader
        from auto_round.compressors.mllm.template import get_template
        from auto_round.special_model_handler import MISTRAL_3_2_MODELS, NOT_SUPPORT_ONLY_TEXT_MODELS
        from auto_round.utils.model import resolve_model_type

        image_processor = self.image_processor
        tokenizer = self.tokenizer
        # Handle template selection
        if isinstance(self.model, PreTrainedModel):
            model_type = getattr(self.model.config, "model_type", None)
            if model_type == "llava" and self.template is None:
                self.template = "default"

        if hasattr(self.model, "name_or_path"):
            name = self.model.name_or_path
            if any([m in name for m in MISTRAL_3_2_MODELS]):
                self.template = "mistral3_2"

        template_name = self.template
        if template_name is None:
            template_name = resolve_model_type(self.model) or getattr(self.model.config, "model_type", None)
        if template_name is None:
            template_name = "default"

        template_obj = get_template(
            template_name,
            model=self.model,
            tokenizer=tokenizer,
            processor=self.processor,
            image_processor=image_processor,
            quiet=not self.quant_nontext_module,
        )

        logger.info(f"Using MLLM template: {template_name}")

        dataset = self.dataset.replace(" ", "") if isinstance(self.dataset, str) else self.dataset
        if dataset is None:
            dataset = template_obj.default_dataset

        if isinstance(dataset, str):
            dataset = dataset.replace(" ", "")
            # Switch text-only dataset to MLLM dataset when quant_nontext_module=True,
            # as text datasets cannot calibrate vision modules.
            from auto_round.calib_dataset import CALIB_DATASETS

            if self.quant_nontext_module and dataset in CALIB_DATASETS:
                logger.warning(
                    "Text only dataset cannot be used for calibrating non-text modules,"
                    " switching to liuhaotian/llava_conv_58k"
                )
                dataset = "liuhaotian/llava_conv_58k"
            elif dataset in CALIB_DATASETS and template_obj.model_type in NOT_SUPPORT_ONLY_TEXT_MODELS:
                logger.warning(
                    f"{getattr(self.model.config, 'model_type', template_obj.model_type)}"
                    f" does not support for {dataset},"
                    " will use liuhaotian/llava_conv_58k with default config as an alternative."
                )
                dataset = "liuhaotian/llava_conv_58k"
            orig_bs = self.batch_size
            (
                self.dataloader,
                self.batch_size,
                self.seqlen,
            ) = get_mllm_dataloader(
                template=template_obj,
                model=self.model,
                tokenizer=tokenizer,
                processor=self.processor,
                image_processor=image_processor,
                dataset=dataset,
                extra_data_dir=self.extra_data_dir,
                seqlen=self.seqlen,
                bs=bs,
                seed=self.seed,
                nsamples=nsamples,
                quant_nontext_module=self.quant_nontext_module,
            )
            if orig_bs != 1 and self.batch_size == 1:
                self.is_only_supported_bs1 = True
        else:
            self.dataloader = self.dataset

        # Process data through the model for calibration
        total_cnt = 0
        for data in self.dataloader:
            if data is None:
                continue

            try:
                if isinstance(data, str):
                    # List-of-strings dataset: process through template → model inputs
                    processed = template_obj.processor.get_input(
                        text=data, images=None, max_length=self.seqlen, squeeze=False
                    )
                    data_new = {k: to_device(v, self.model.device) for k, v in processed.items()}
                elif isinstance(data, dict) and "text" in data:
                    # FakeDataLoader-style {"text": ..., "image": ...}: process through template
                    text = data["text"]
                    if isinstance(text, dict):
                        text = [text]
                    input_text = template_obj._encode(text)
                    processed = template_obj.processor.get_input(
                        text=input_text,
                        images=data.get("image", None),
                        max_length=self.seqlen,
                        squeeze=False,
                    )
                    data_new = {}
                    for key, value in processed.items():
                        tensor_val = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
                        data_new[key] = to_device(tensor_val, self.model.device)
                elif isinstance(data, dict):
                    data_new = {
                        key: value.to(self.model.device) if isinstance(value, torch.Tensor) else value
                        for key, value in data.items()
                    }
                else:
                    data_new = data

                if isinstance(data_new, dict):
                    self.model(**data_new)
                else:
                    self.model(data_new)
            except NotImplementedError:
                pass
            except Exception as e:
                logger.warning(f"Calibration forward pass failed: {e}")
                continue

            total_cnt += bs
            if total_cnt >= nsamples:
                break

        if total_cnt == 0:
            logger.error("no data has been cached, please provide more data")
            exit(-1)
        elif total_cnt < nsamples:
            logger.warning(f"Insufficient number of samples: required {nsamples}, but only {total_cnt} were processed.")
