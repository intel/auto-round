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

from auto_round.logger import logger


class MLLMMixin:
    """MLLM-specific functionality mixin.

    This mixin adds MLLM-specific functionality to any compressor (DataDrivenCompressor,
    ZeroShotCompressor, ImatrixCompressor, etc). It handles multi-modal models
    (vision-language models) that require special data loading and processing logic.

    Can be combined with:
    - DataDrivenCompressor (for AutoRound with calibration)
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

        # Mirror old arch: reset batch_size to 1 when quantizing non-text modules,
        # because vision encoder blocks have non-standard hidden_states shapes that
        # break batch_dim detection, and image collation fails with batch_size > 1.
        if quant_nontext_module:
            # ``batch_size`` is owned by the compressor / shared
            # CalibrationState; it only comes from kwargs now (entry.py forwards
            # it explicitly).  AlgConfig no longer carries it.
            batch_size = kwargs.get("batch_size", None)
            if batch_size is not None and batch_size != 1:
                grad_acc = kwargs.get("gradient_accumulate_steps", 1)
                new_grad_acc = batch_size * grad_acc
                kwargs["gradient_accumulate_steps"] = new_grad_acc
                kwargs["batch_size"] = 1
                # Also patch ``gradient_accumulate_steps`` on AlgConfig (still
                # owned there) so behaviour matches the old arch.
                _alg_cfg = args[0] if args else None
                if _alg_cfg is not None:
                    cfgs = _alg_cfg if isinstance(_alg_cfg, list) else [_alg_cfg]
                    for cfg in cfgs:
                        if hasattr(cfg, "gradient_accumulate_steps"):
                            cfg.gradient_accumulate_steps = new_grad_acc
                logger.warning(
                    f"reset batch_size({batch_size}) to 1 and "
                    f"gradient_accumulate_steps to {new_grad_acc} "
                    f"because batch_size={batch_size} cannot be used for calibrating non-text modules."
                )

        # super().__init__() creates model_context, which eagerly loads the model and
        # populates model_context.processor / image_processor / tokenizer.
        super().__init__(*args, **kwargs)

        # Apply user-provided overrides into model_context (single source of truth).
        if processor is not None:
            self.model_context.processor = processor
        if image_processor is not None:
            self.model_context.image_processor = image_processor

    def _get_calibrator_kind(self) -> str:
        """Select the MLLM calibration strategy.

        ``MLLMCalibrator`` lives at :mod:`auto_round.calibration.mllm`
        and owns what used to be ``MLLMMixin.calib``.
        """
        return "mllm"

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
