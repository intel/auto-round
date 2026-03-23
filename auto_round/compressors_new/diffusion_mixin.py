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
from tqdm import tqdm

from auto_round.logger import logger
from auto_round.utils.model import wrap_block_forward_positional_to_kwargs


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
        self.diffusion = True

        # Call parent class __init__ (will be CalibCompressor, ImatrixCompressor, etc)
        super().__init__(*args, **kwargs)

    def _get_block_forward_func(self, name: str):
        """Diffusion models pass positional args; wrap the base forward func accordingly.

        The MRO guarantees that super() resolves to CalibCompressor._get_block_forward_func,
        mirroring the old-arch pattern in compressors/diffusion/compressor.py.
        """
        return wrap_block_forward_positional_to_kwargs(super()._get_block_forward_func(name))

    @torch.no_grad()
    def calib(self, nsamples, bs):
        """Perform diffusion-specific calibration for quantization.

        Override parent's calib method to use diffusion dataset loading logic.
        The diffusion pipeline is read from ``self.model_context.pipe``.
        """
        from auto_round.compressors.diffusion.dataset import get_diffusion_dataloader

        pipe = self.model_context.pipe
        if pipe is None:
            raise ValueError(
                "Diffusion pipeline not found in model_context. " "Ensure the model was loaded as a diffusion model."
            )

        logger.warning(
            "Diffusion model will catch nsamples * num_inference_steps inputs, "
            "you can reduce nsamples or num_inference_steps if OOM or take too much time."
        )
        if isinstance(self.dataset, str):
            dataset = self.dataset.replace(" ", "")
            self.dataloader, self.batch_size, self.gradient_accumulate_steps = get_diffusion_dataloader(
                dataset=dataset,
                bs=self.batch_size,
                seed=self.seed,
                nsamples=self.nsamples,
                gradient_accumulate_steps=self.gradient_accumulate_steps,
            )
        else:
            self.dataloader = self.dataset
        total_cnt = 0

        total = nsamples if not hasattr(self.dataloader, "len") else min(nsamples, len(self.dataloader))
        if pipe.dtype != self.model.dtype:
            pipe.to(self.model.dtype)

        if (
            hasattr(self.model, "hf_device_map")
            and len(self.model.hf_device_map) > 0
            and pipe.device != self.model.device
            and torch.device(self.model.device).type in ["cuda", "xpu"]
        ):
            logger.error(
                "Diffusion model is activated sequential model offloading, it will crash during moving to GPU/XPU. "
                "Please use model path for quantization or "
                "move the pipeline object to GPU/XPU before passing them into API."
            )
            exit(-1)

        if pipe.device != self.model.device:
            pipe.to(self.model.device)
        with tqdm(range(1, total + 1), desc="cache block inputs") as pbar:
            for ids, prompts in self.dataloader:
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                try:
                    pipe(
                        prompt=prompts,
                        guidance_scale=self.guidance_scale,
                        num_inference_steps=self.num_inference_steps,
                        generator=(
                            None
                            if self.generator_seed is None
                            else torch.Generator(device=pipe.device).manual_seed(self.generator_seed)
                        ),
                    )
                except NotImplementedError:
                    pass
                except Exception as error:
                    raise error
                step = len(prompts)
                total_cnt += step
                pbar.update(step)
                if total_cnt >= nsamples:
                    break
        if total_cnt == 0:
            logger.error(
                f"no data has been cached, please provide more data with sequence length >={self.seqlen} in the "
                f"dataset or decease the sequence length"
            )
            exit(-1)
        elif total_cnt < nsamples:
            logger.warning(
                f"Insufficient number of samples collected may affect the quantization. "
                f"target samples count is {nsamples}, while valid samples count is {total_cnt}"
            )
            if total_cnt < self.batch_size:
                raise ValueError(
                    f"valid samples is less than batch_size({self.batch_size}),"
                    " please adjust self.batch_size or seqlen."
                )
            max_len = (total_cnt // self.batch_size) * self.batch_size
            for k, v in self.inputs.items():
                for key in v:
                    if isinstance(v[key], list) and len(v[key]) == total_cnt:
                        self.inputs[k][key] = v[key][:max_len]

        # torch.cuda.empty_cache()

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
                    output_dir=sub_module_path if not self.is_immediate_saving else output_dir,
                    format=format,
                    inplace=inplace,
                    **kwargs,
                )
            elif val is not None and hasattr(val, "save_pretrained"):
                val.save_pretrained(sub_module_path)
        pipe.config.save_pretrained(output_dir)
        return compressed_model
