# Copyright (c) 2025 Intel Corporation
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
from __future__ import annotations

from typing import Any, Callable, Union

import torch

from auto_round.compressors import AdamCompressor, LLMCompressor, MLLMCompressor
from auto_round.schemes import QuantizationScheme
from auto_round.utils import is_mllm_model


class AutoRound:
    def __new__(
        cls,
        model: Union[torch.nn.Module, str],
        tokenizer=None,
        scheme: Union[str, dict, QuantizationScheme] = "W4A16",
        layer_config: dict[str, Union[str, dict, QuantizationScheme]] = None,
        dataset: Union[str, list, tuple, torch.utils.data.DataLoader] = "NeelNanda/pile-10k",
        iters: int = 200,
        seqlen: int = 2048,
        nsamples: int = 128,
        batch_size: int = 8,
        gradient_accumulate_steps: int = 1,
        low_gpu_mem_usage: bool = False,
        device_map: Union[str, torch.device, int, dict] = 0,
        enable_torch_compile: bool = False,
        seed: int = 42,
        fp_layers: str = None,

        # for adam
        adam: bool = False,
        # for MLLM
        mllm=False,
        processor=None,
        image_processor=None,
        quant_nontext_module: bool = False,
        **kwargs,
    ):
        model_cls = []
        if mllm or is_mllm_model(model):
            model_cls.append(MLLMCompressor)
            mllm_kwargs = {
                "mllm": mllm,
                "processor": processor,
                "image_processor": image_processor,
                "quant_nontext_module": quant_nontext_module,
            }
            kwargs.update(mllm_kwargs)
        else:
            model_cls.append(LLMCompressor)
        if adam:
            model_cls.append(AdamCompressor)
        dynamic_compressor = type("AutoRound", tuple(model_cls), {})
        return dynamic_compressor(
            model=model,
            tokenizer=tokenizer,
            scheme=scheme,
            layer_config=layer_config,
            dataset=dataset,
            iters=iters,
            seqlen=seqlen,
            nsamples=nsamples,
            batch_size=batch_size,
            gradient_accumulate_steps=gradient_accumulate_steps,
            low_gpu_mem_usage=low_gpu_mem_usage,
            device_map=device_map,
            enable_torch_compile=enable_torch_compile,
            seed=seed,
            fp_layers=fp_layers,
            **kwargs,
        )


from auto_round.logger import deprecated


@deprecated("AutoRound")
class AutoRoundAdam(AutoRound):
    def __init__(self, *args, **kwargs):
        super().__init__()


@deprecated("AutoRound")
class AutoRoundMLLM(AutoRound):
    def __init__(self, *args, **kwargs):
        super().__init__()
