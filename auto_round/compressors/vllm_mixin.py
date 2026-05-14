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

import gc

import torch
from tqdm import tqdm

from auto_round.logger import logger


class VllmMixin:
    """vLLM-specific behavior for new-architecture compressors."""

    def __init__(self, *args, use_vllm_loading=False, **kwargs):
        self.use_vllm_loading = use_vllm_loading

        # Keep vLLM path on a conservative setup that is known to be stable.
        if use_vllm_loading:
            cfg = args[0] if args else None
            cfgs = cfg if isinstance(cfg, list) else [cfg]
            for item in cfgs:
                if item is None:
                    continue
                if hasattr(item, "batch_size") and item.batch_size != 1:
                    logger.warning("vLLM loading in new architecture forces batch_size=1.")
                    item.batch_size = 1
                if hasattr(item, "iters") and item.iters and item.iters > 0:
                    logger.warning("vLLM loading currently supports RTN path only; force iters=0.")
                    item.iters = 0

        super().__init__(*args, use_vllm_loading=use_vllm_loading, **kwargs)

    @torch.no_grad()
    def calib(self, nsamples, bs):
        if not getattr(self.model_context, "use_vllm_loading", False):
            return super().calib(nsamples, bs)

        llm = getattr(self.model_context, "llm", None)
        if llm is None:
            logger.warning("vLLM loading requested but engine is unavailable; fallback to regular calibration.")
            return super().calib(nsamples, bs)

        from vllm import SamplingParams

        from auto_round.calib_dataset import get_dataloader

        if isinstance(self.dataset, str):
            dataset = self.dataset.replace(" ", "")
            self.dataloader = get_dataloader(
                self.model_context.tokenizer,
                self.seqlen,
                dataset,
                self.seed,
                1,
                nsamples,
            )
        else:
            self.dataloader = self.dataset

        sampling_params = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=1)
        total_cnt = 0
        total = nsamples if not hasattr(self.dataloader, "__len__") else min(nsamples, len(self.dataloader))

        with tqdm(total=total, desc="cache block inputs") as pbar:
            for prompts in self.dataloader:
                if prompts is None:
                    continue

                try:
                    if isinstance(prompts, dict) and "input_ids" in prompts:
                        input_ids = prompts["input_ids"]
                    elif isinstance(prompts, torch.Tensor):
                        input_ids = prompts
                    else:
                        input_ids = None

                    if isinstance(input_ids, torch.Tensor):
                        batch_prompts = [{"prompt_token_ids": row.tolist()} for row in input_ids]
                        llm.generate(batch_prompts, sampling_params)
                        step = len(batch_prompts)
                    else:
                        llm.generate(prompts, sampling_params)
                        step = len(prompts) if isinstance(prompts, list) else 1
                except NotImplementedError:
                    continue

                total_cnt += step
                pbar.update(step)
                if total_cnt >= nsamples:
                    break

        if total_cnt == 0:
            logger.error(
                "no data has been cached, please provide more data with sequence length "
                f">={self.seqlen} in the dataset or decrease the sequence length"
            )
            exit(-1)
        if total_cnt < nsamples:
            logger.warning(
                "Insufficient number of samples collected may affect the quantization. "
                f"target samples count is {nsamples}, while valid samples count is {total_cnt}"
            )

        self.model_context.llm = None
        del llm
        gc.collect()
