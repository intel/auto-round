# Copyright (c) 2024 Intel Corporation
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

import logging
import random
import time
from typing import TYPE_CHECKING, List, Optional, Union

import lm_eval
from lm_eval import simple_evaluate as lm_simple_evaluate
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from lm_eval.models.huggingface import HFLM


def simple_evaluate_user_model(
        user_model,
        tokenizer,
        batch_size: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        **kwargs
):
    hflm = HFLM(pretrained=user_model, tokenizer=tokenizer, batch_size=batch_size, max_batch_size=max_batch_size)
    return lm_simple_evaluate(
        model=hflm,
        model_args=None,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        **kwargs)


def simple_evaluate(
        model,
        model_args: Optional[Union[str, dict]] = None,
        batch_size: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        device: Optional[str] = None,
        **kwargs):
    try:
        from auto_round import AutoRoundConfig
    except:
        from auto_round.auto_quantizer import AutoHfQuantizer

    return lm_simple_evaluate(
        model=model,
        model_args=model_args,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        device=device,
        **kwargs)
