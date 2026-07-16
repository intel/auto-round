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
"""Diffusion calibration strategy.

Inherits :class:`LLMCalibrator` for ``collect`` / ``cache_inter_data``,
and customises:

- :meth:`calib` — drive the diffusion ``pipe`` with prompts / inference steps.
- :meth:`should_stop` — always ``False`` so all denoising steps execute.
- :meth:`wrap_block_forward` — convert positional → kwargs for diffusion blocks.
"""

import inspect
from typing import Optional

import torch
from tqdm import tqdm

from auto_round.calibration.llm import LLMCalibrator
from auto_round.calibration.register import register_calibrator
from auto_round.compressors import BaseCompressor
from auto_round.logger import logger
from auto_round.utils.device_manager import device_manager
from auto_round.utils.model import wrap_block_forward_positional_to_kwargs


@register_calibrator("diffusion")
class DiffusionCalibrator(LLMCalibrator):
    """Calibrator for diffusion models (Stable Diffusion / FLUX / ...)."""

    # ── Overrides for diffusion-specific behaviour ─────────────────────────
    def __init__(self, compressor: BaseCompressor):
        super().__init__(compressor)
        self.pipe = compressor.pipe
        self.guidance_scale = compressor.guidance_scale
        self.num_inference_steps = compressor.num_inference_steps
        self.generator_seed = compressor.generator_seed  # make sure pass

    def should_stop(self, name: str) -> bool:
        """Diffusion models must run *all* denoising steps to collect enough inputs.

        Mirrors the legacy ``DiffusionMixin._should_stop_cache_forward`` which
        always returns ``False`` so the pipeline never exits early.
        """
        return False

    def wrap_block_forward(self, forward_fn):
        """Wrap positional-arg block forward into kwargs form for diffusion blocks."""
        return wrap_block_forward_positional_to_kwargs(forward_fn)

    def _get_calibration_image(self, batch_size: int):
        """Return a synthetic PIL Image for I2V pipeline calibration."""
        params = inspect.signature(self.pipe.__call__).parameters
        width_param = params.get("width")
        height_param = params.get("height")
        width = (
            832
            if width_param is None or width_param.default in (inspect.Parameter.empty, None)
            else width_param.default
        )
        height = (
            480
            if height_param is None or height_param.default in (inspect.Parameter.empty, None)
            else height_param.default
        )
        from PIL import Image  # pylint: disable=E0401

        image = Image.new("RGB", (int(width), int(height)), color=(127, 127, 127))
        if batch_size == 1:
            return image
        return [image.copy() for _ in range(batch_size)]

    @torch.no_grad()
    def calib(self, nsamples: int, bs: int) -> None:
        """Drive the diffusion pipeline so block-forward hooks fire.

        Verbatim port of the legacy ``DiffusionMixin.calib``.
        """
        from auto_round.compressors.diffusion.dataset import get_diffusion_dataloader

        pipe = self.pipe
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
            self.dataloader, self.batch_size = get_diffusion_dataloader(
                dataset=dataset, bs=bs, seed=self.seed, nsamples=nsamples
            )
        else:
            self.dataloader = self.dataset
        total_cnt = 0

        total = nsamples if not hasattr(self.dataloader, "len") else min(nsamples, len(self.dataloader))

        if (
            hasattr(self.model, "hf_device_map")
            and len(self.model.hf_device_map) > 1
            and pipe.device != self.model.device
            and torch.device(self.model.device).type in ["cuda", "xpu"]
        ):
            logger.error(
                "Diffusion model is activated sequential model offloading, it will crash during moving to GPU/XPU. "
                "Please use model path for quantization or "
                "move the pipeline object to GPU/XPU before passing them into API."
            )
            exit(-1)

        target_device = device_manager.device
        if pipe.device != torch.device(target_device):
            pipe.to(target_device)
        pipeline_fn = getattr(pipe, "_autoround_pipeline_fn", None)
        # Check if this is an I2V pipeline (needs calibration image)
        requires_image = False
        if hasattr(self, "_requires_calibration_image"):
            image_param = inspect.signature(self.pipe.__call__).parameters.get("image")
            if image_param is not None and image_param.default is inspect.Parameter.empty:
                requires_image = True

        with tqdm(range(1, total + 1), desc="cache block inputs") as pbar:
            for ids, prompts in self.dataloader:
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                try:
                    generator = (
                        None
                        if self.generator_seed is None
                        else torch.Generator(device=pipe.device).manual_seed(self.generator_seed)
                    )
                    if requires_image:
                        image = self._get_calibration_image(len(prompts) if isinstance(prompts, list) else 1)
                        pipe(
                            image,
                            prompt=prompts,
                            guidance_scale=self.guidance_scale,
                            num_inference_steps=self.num_inference_steps,
                            generator=generator,
                        )
                    elif pipeline_fn is not None:
                        pipeline_fn(
                            pipe,
                            prompts,
                            guidance_scale=self.guidance_scale,
                            num_inference_steps=self.num_inference_steps,
                            generator=generator,
                        )
                    else:
                        pipe(
                            prompts,
                            guidance_scale=self.guidance_scale,
                            num_inference_steps=self.num_inference_steps,
                            generator=generator,
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
                f"no data has been cached, please provide more data with sequence length >={c.seqlen} in the "
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
                    " please adjust c.batch_size or seqlen."
                )
            max_len = (total_cnt // self.batch_size) * self.batch_size
            for k, v in self.inputs.items():
                for key in v:
                    if isinstance(v[key], list) and len(v[key]) == total_cnt:
                        self.inputs[k][key] = v[key][:max_len]
