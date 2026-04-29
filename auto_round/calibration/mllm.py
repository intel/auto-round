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

    @torch.no_grad()
    def calib(self, nsamples: int, bs: int) -> None:
        """Drive the multimodal model so block-forward hooks fire.

        Verbatim port of the legacy ``MLLMMixin.calib``.
        """
        from transformers import PreTrainedModel

        from auto_round.compressors.mllm.dataset import get_mllm_dataloader
        from auto_round.compressors.mllm.template import get_template
        from auto_round.special_model_handler import MISTRAL_3_2_MODELS

        c = self.compressor
        mc = c.model_context
        processor = mc.processor
        image_processor = mc.image_processor
        tokenizer = mc.tokenizer

        # Handle template selection
        if isinstance(mc.model, PreTrainedModel):
            model_type = getattr(mc.model.config, "model_type", None)
            if model_type == "llava" and c.template is None:
                c.template = "default"

        if hasattr(mc.model, "name_or_path"):
            name = mc.model.name_or_path
            if any([m in name for m in MISTRAL_3_2_MODELS]):
                c.template = "mistral3_2"

        template_name = c.template
        if template_name is None and hasattr(mc.model.config, "model_type"):
            template_name = mc.model.config.model_type
        if template_name is None:
            template_name = "default"

        c.template_obj = get_template(
            template_name,
            model=mc.model,
            tokenizer=tokenizer,
            processor=processor,
            image_processor=image_processor,
            use_rtn=getattr(c.quantize_config, "iters", None) == 0,
            quiet=not c.quant_nontext_module,
        )

        logger.info(f"Using MLLM template: {template_name}")

        dataset = c.dataset.replace(" ", "") if isinstance(c.dataset, str) else c.dataset
        if dataset is None:
            dataset = c.template_obj.default_dataset

        if isinstance(c.dataset, str):
            dataset = c.dataset.replace(" ", "")
            # Switch text-only dataset to MLLM dataset when quant_nontext_module=True,
            # as text datasets cannot calibrate vision modules.
            from auto_round.calib_dataset import CALIB_DATASETS

            if c.quant_nontext_module and dataset in CALIB_DATASETS:
                logger.warning(
                    "Text only dataset cannot be used for calibrating non-text modules,"
                    " switching to liuhaotian/llava_conv_58k"
                )
                dataset = "liuhaotian/llava_conv_58k"
            (
                c.dataloader,
                c.batch_size,
                c.seqlen,
                c.gradient_accumulate_steps,
            ) = get_mllm_dataloader(
                template=c.template_obj,
                model=mc.model,
                tokenizer=tokenizer,
                processor=processor,
                image_processor=image_processor,
                dataset=dataset,
                extra_data_dir=c.extra_data_dir,
                seqlen=c.seqlen,
                bs=bs,
                seed=c.seed,
                nsamples=nsamples,
                quant_nontext_module=c.quant_nontext_module,
            )
        else:
            c.dataloader = c.dataset

        # Process data through the model for calibration
        total_cnt = 0
        for data in c.dataloader:
            if data is None:
                continue

            try:
                if isinstance(data, str):
                    # List-of-strings dataset: process through template → model inputs
                    processed = c.template_obj.processor.get_input(
                        text=data, images=None, max_length=c.seqlen, squeeze=False
                    )
                    data_new = {k: to_device(v, mc.model.device) for k, v in processed.items()}
                elif isinstance(data, dict) and "text" in data:
                    # FakeDataLoader-style {"text": ..., "image": ...}: process through template
                    text = data["text"]
                    if isinstance(text, dict):
                        text = [text]
                    input_text = c.template_obj._encode(text)
                    processed = c.template_obj.processor.get_input(
                        text=input_text,
                        images=data.get("image", None),
                        max_length=c.seqlen,
                        squeeze=False,
                    )
                    data_new = {}
                    for key, value in processed.items():
                        tensor_val = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
                        data_new[key] = to_device(tensor_val, mc.model.device)
                elif isinstance(data, dict):
                    data_new = {
                        key: value.to(mc.model.device) if isinstance(value, torch.Tensor) else value
                        for key, value in data.items()
                    }
                else:
                    data_new = data

                if isinstance(data_new, dict):
                    mc.model(**data_new)
                else:
                    mc.model(data_new)
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
