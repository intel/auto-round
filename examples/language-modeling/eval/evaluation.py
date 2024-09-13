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


def simple_evaluate(
        model,
        model_args: Optional[Union[str, dict]] = None,
        user_model = None,
        batch_size: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        device: Optional[str] = None,
        **kwargs):

    try:
        from auto_round import AutoRoundConfig
    except:
        from auto_round.auto_quantizer import AutoHfQuantizer

    if model_args is None:
        model_args = ""
    
    if isinstance(model_args, dict):
        lm = lm_eval.api.registry.get_model(model).create_from_arg_obj(
            model_args,
            {
                "batch_size": batch_size,
                "max_batch_size": max_batch_size,
                "device": device,
            },
        )

    else:
        lm = lm_eval.api.registry.get_model(model).create_from_arg_string(
            model_args,
            {
                "batch_size": batch_size,
                "max_batch_size": max_batch_size,
                "device": device,
            },
        )
    if user_model is not None:
        lm._model = user_model
    return lm_simple_evaluate(
        model=lm,
        model_args=model_args,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        device=device,
        **kwargs)



if __name__ == "__main__":

    import sys

    sys.path.insert(0, '../../')
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", default="/models/opt-125m/"
    )
    parser.add_argument(
        "--eval_bs", default=1,
    )
    parser.add_argument(
        "--trust_remote_code", action='store_true',
        help="Whether to enable trust_remote_code"
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="PyTorch device (e.g. cpu/cuda:0/hpu) for evaluation."
    )
    parser.add_argument("--tasks",
                        default="lambada_openai,hellaswag,winogrande,piqa,mmlu,truthfulqa_mc1," \
                                "openbookqa,boolq,rte,arc_easy,arc_challenge",
                        help="lm-eval tasks for lm_eval version 0.4.2")

    args = parser.parse_args()
    s = time.time()
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=args.trust_remote_code)

    if hasattr(config, "quantization_config"):
        quantization_config = config.quantization_config
        if "quant_method" in quantization_config and "auto-round" in quantization_config["quant_method"]:
            from auto_round.auto_quantizer import AutoHfQuantizer
        elif "quant_method" in quantization_config and quantization_config["quant_method"] == "gptq":
            if args.device == "hpu":
                from auto_round.auto_quantizer import AutoHfQuantizer

    test_tasks = args.tasks
    if isinstance(test_tasks, str):
        test_tasks = test_tasks.split(',')
    model_name = args.model_name.rstrip('/')
    from lm_eval.utils import make_table

    model_args = f"pretrained={args.model_name}"
    model_args += ",dtype=float16"
    if args.trust_remote_code:
        model_args += f",trust_remote_code=True"
    result = simple_evaluate(model="hf",
                             model_args=model_args,
                             tasks=test_tasks,
                             device=args.device,
                             batch_size=args.eval_bs)
    print(make_table(result))

    print("cost time: ", time.time() - s)