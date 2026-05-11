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

import torch
from tqdm import tqdm

from auto_round.calibration.llm import LLMCalibrator
from auto_round.calibration.register import register_calibrator
from auto_round.logger import logger
from auto_round.utils.model import wrap_block_forward_positional_to_kwargs


@register_calibrator("diffusion")
class DiffusionCalibrator(LLMCalibrator):
    """Calibrator for diffusion models (Stable Diffusion / FLUX / ...)."""

    # ── Overrides for diffusion-specific behaviour ─────────────────────────

    def should_stop(self, name: str) -> bool:
        """Diffusion models must run *all* denoising steps to collect enough inputs.

        Mirrors the legacy ``DiffusionMixin._should_stop_cache_forward`` which
        always returns ``False`` so the pipeline never exits early.
        """
        return False

    def wrap_block_forward(self, forward_fn):
        """Wrap positional-arg block forward into kwargs form for diffusion blocks."""
        return wrap_block_forward_positional_to_kwargs(forward_fn)

    @torch.no_grad()
    def calib(self, nsamples: int, bs: int) -> None:
        """Drive the diffusion pipeline so block-forward hooks fire.

        Verbatim port of the legacy ``DiffusionMixin.calib``.
        """
        from auto_round.compressors.diffusion.dataset import get_diffusion_dataloader

        c = self.compressor
        pipe = c.model_context.pipe
        if pipe is None:
            raise ValueError(
                "Diffusion pipeline not found in model_context. " "Ensure the model was loaded as a diffusion model."
            )

        logger.warning(
            "Diffusion model will catch nsamples * num_inference_steps inputs, "
            "you can reduce nsamples or num_inference_steps if OOM or take too much time."
        )
        if isinstance(c.dataset, str):
            dataset = c.dataset.replace(" ", "")
            c.dataloader, c.batch_size, c.gradient_accumulate_steps = get_diffusion_dataloader(
                dataset=dataset,
                bs=c.batch_size,
                seed=c.seed,
                nsamples=c.nsamples,
                gradient_accumulate_steps=c.gradient_accumulate_steps,
            )
        else:
            c.dataloader = c.dataset
        total_cnt = 0

        total = nsamples if not hasattr(c.dataloader, "len") else min(nsamples, len(c.dataloader))

        if (
            hasattr(c.model, "hf_device_map")
            and len(c.model.hf_device_map) > 1
            and pipe.device != c.model.device
            and torch.device(c.model.device).type in ["cuda", "xpu"]
        ):
            logger.error(
                "Diffusion model is activated sequential model offloading, it will crash during moving to GPU/XPU. "
                "Please use model path for quantization or "
                "move the pipeline object to GPU/XPU before passing them into API."
            )
            exit(-1)

        if pipe.device != c.model.device:
            pipe.to(c.model.device)
        with tqdm(range(1, total + 1), desc="cache block inputs") as pbar:
            for ids, prompts in c.dataloader:
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                try:
                    pipe(
                        prompts,
                        guidance_scale=c.guidance_scale,
                        num_inference_steps=c.num_inference_steps,
                        generator=(
                            None
                            if c.generator_seed is None
                            else torch.Generator(device=pipe.device).manual_seed(c.generator_seed)
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
                f"no data has been cached, please provide more data with sequence length >={c.seqlen} in the "
                f"dataset or decease the sequence length"
            )
            exit(-1)
        elif total_cnt < nsamples:
            logger.warning(
                f"Insufficient number of samples collected may affect the quantization. "
                f"target samples count is {nsamples}, while valid samples count is {total_cnt}"
            )
            if total_cnt < c.batch_size:
                raise ValueError(
                    f"valid samples is less than batch_size({c.batch_size})," " please adjust c.batch_size or seqlen."
                )
            max_len = (total_cnt // c.batch_size) * c.batch_size
            for k, v in c.inputs.items():
                for key in v:
                    if isinstance(v[key], list) and len(v[key]) == total_cnt:
                        c.inputs[k][key] = v[key][:max_len]
