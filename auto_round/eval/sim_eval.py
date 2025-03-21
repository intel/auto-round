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

import torch
import sys
sys.path.insert(0, "../..")

from auto_round.auto_quantizer import AutoRoundConfig

from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation import simple_evaluate_user_model
from lm_eval.utils import make_table

# torch.nn.Linear = QuantLinear

if __name__ == "__main__":
    # maybe need to modify transformers/modeling_utils, add "F8_E4M3": torch.float8_e4m3fn to str_to_torch_dtype
    # need to change quant_method to intel/auto-round
    model_name = "opt-125m-w4g128-auto-gptq/"
    batch_size = 16
    tasks = "piqa"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto" 
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    result = simple_evaluate_user_model(model, tokenizer, batch_size=batch_size, tasks=tasks) 
    print(make_table(result))


