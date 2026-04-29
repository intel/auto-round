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
import os
from typing import Union

import torch

from auto_round.logger import logger


class DiffusionMixin:
    """Diffusion-specific functionality mixin.

    This mixin adds diffusion model-specific functionality to any compressor
    (DataDrivenCompressor, ZeroShotCompressor, ImatrixCompressor, etc). It handles
    diffusion models (like Stable Diffusion, FLUX) that require special pipeline
    handling and data generation logic.

    Can be combined with:
    - DataDrivenCompressor (for AutoRound with calibration)
    - ImatrixCompressor (for RTN with importance matrix)
    - ZeroShotCompressor (for basic RTN)

    Diffusion-specific parameters:
        guidance_scale: Control how much image generation follows text prompt
        num_inference_steps: Reference number of denoising steps
        generator_seed: Seed for initial noise generation

    Design note:
        ``ModelContext._load_model()`` loads the diffusion pipeline and sets
        ``model_context.pipe`` and ``model_context.model`` (the unet/transformer).
        This mixin reads ``self.model_context.pipe`` directly during calibration and
        saving so that ``model_context`` remains the single source of truth.
    """

    def __init__(self, *args, guidance_scale=7.5, num_inference_steps=50, generator_seed=None, **kwargs):
        # Store diffusion-specific attributes
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.generator_seed = generator_seed

        # Mirror old-arch DiffusionCompressor.__init__: when iters > 0, diffusion calibration
        # cannot use batch_size > 1 for non-text modules; fold the extra batch into
        # gradient_accumulate_steps so the effective sample count is unchanged.
        # The authoritative batch_size lives on the AlgConfig (args[0]); kwargs may also
        # carry it from AutoRoundCompatible. Patch BOTH (same pattern as MLLMMixin).
        iters = kwargs.get("iters", None)
        _alg_cfg = args[0] if args else None
        if iters is None and _alg_cfg is not None:
            cfgs = _alg_cfg if isinstance(_alg_cfg, list) else [_alg_cfg]
            for cfg in cfgs:
                if hasattr(cfg, "iters") and cfg.iters is not None:
                    iters = cfg.iters
                    break
        if iters is None:
            iters = 200

        if iters > 0:
            # ``batch_size`` is owned by the compressor / shared
            # CalibrationState; it only comes from kwargs now (entry.py forwards
            # it explicitly).  AlgConfig no longer carries it.  Treat a missing
            # value as ``BaseCompressor``'s default (8) so the reset path always
            # triggers when the user didn't explicitly opt out.
            batch_size = kwargs.get("batch_size", 8)
            if batch_size != 1:
                grad_acc = kwargs.get("gradient_accumulate_steps", 1)
                if _alg_cfg is not None:
                    cfgs = _alg_cfg if isinstance(_alg_cfg, list) else [_alg_cfg]
                    for cfg in cfgs:
                        if hasattr(cfg, "gradient_accumulate_steps") and cfg.gradient_accumulate_steps is not None:
                            grad_acc = cfg.gradient_accumulate_steps
                            break
                new_grad_acc = batch_size * grad_acc
                kwargs["gradient_accumulate_steps"] = new_grad_acc
                kwargs["batch_size"] = 1
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

        # Call parent class __init__ (will be DataDrivenCompressor, ImatrixCompressor, etc)
        super().__init__(*args, **kwargs)

        # Mirror old-arch DiffusionCompressor._align_device_and_dtype: unconditionally
        # cast the full diffusion pipeline (VAE, text encoder, etc.) to the transformer's
        # dtype so that calibration's pipe(...) call doesn't crash with dtype mismatches
        # when the transformer is force-cast to bf16 for activation quantization.
        # Note: pipe.dtype only reflects the primary component, so an equality check would
        # miss mixed-dtype pipelines where e.g. the VAE is still float32.
        pipe = getattr(self.model_context, "pipe", None)
        model = getattr(self.model_context, "model", None)
        if pipe is not None and model is not None:
            is_nextstep = hasattr(model, "config") and getattr(model.config, "model_type", None) == "nextstep"
            if not is_nextstep:
                pipe.to(model.dtype)

    def _get_calibrator_kind(self) -> str:
        """Select the diffusion calibration strategy.

        ``DiffusionCalibrator`` lives at
        :mod:`auto_round.calibration.diffusion` and owns what used to be
        ``DiffusionMixin.calib`` / ``_get_block_forward_func`` /
        ``_should_stop_cache_forward``.
        """
        return "diffusion"

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
        if output_dir is None:
            return super().save_quantized(output_dir, format=format, inplace=inplace, **kwargs)

        pipe = self.model_context.pipe
        compressed_model = None
        for name in pipe.components.keys():
            val = getattr(pipe, name)
            sub_module_path = (
                os.path.join(output_dir, name) if os.path.basename(os.path.normpath(output_dir)) != name else output_dir
            )
            if (
                hasattr(val, "config")
                and hasattr(val.config, "_name_or_path")
                and val.config._name_or_path == self.model.config._name_or_path
            ):
                compressed_model = super().save_quantized(
                    output_dir=sub_module_path if not self.compress_context.is_immediate_saving else output_dir,
                    format=format,
                    inplace=inplace,
                    **kwargs,
                )
            elif val is not None and hasattr(val, "save_pretrained"):
                val.save_pretrained(sub_module_path)
        pipe.config.save_pretrained(output_dir)
        return compressed_model
