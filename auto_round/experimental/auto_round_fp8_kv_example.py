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

from fp8_kv_cache import (
    initialize_quantized_kv_cache,
    prep_attention_module_for_calibration,
    freeze_module_quantization_,
)

from loguru import logger
import sys
import torch

logger.add(sys.stderr, level="TRACE")


# Example
from transformers import AutoModelForCausalLM, AutoTokenizer

# Select model and load it.
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_ID = "/data5/yliu7/HF_HOME/meta-llama/Llama-3.2-1B-Instruct/"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model.eval()

model.apply(initialize_quantized_kv_cache)
model.apply(prep_attention_module_for_calibration)


from auto_round import AutoRound
from transformers import AutoModelForCausalLM, AutoTokenizer


autoround = AutoRound(
    model,
    tokenizer,
    data_type="fp8_to_int_sym",
    act_data_type="fp8_sym",
    bits=8,
    group_size=-1,
    sym=True,
    # Use 0 after https://github.com/intel/auto-round/pull/662
    iters=1, 
    seqlen=2,
)
model, qconfig = autoround.quantize()
assert model is not None, f"Expected q_model to be not None"


model.apply(freeze_module_quantization_)

for name, param in model.named_parameters():
    if "k_scale" in name or "v_scale" in name:
        print(f"{name}: {param.shape}, {param.dtype}, {param.item()}")


###################

# # Example
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Select model and load it.
# MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


# os.environ["LLM_COMPRESSOR_LOG_LEVEL"] = "DEBUG"


# MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL_ID = "/data5/yliu7/HF_HOME/meta-llama/Llama-3.2-1B-Instruct/"
# model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# model.eval()

# model.apply(initialize_quantized_kv_cache)
# model.apply(prep_attention_module_for_calibration)

# sample = {
#     name: torch.ones((1, 32)).long()
#     for name in ["input_ids", "attention_mask", "labels"]
# }

# with torch.no_grad():
#     _ = model(**sample)

# breakpoint()
# model.apply(freeze_module_quantization_)

# for name, param in model.named_parameters():
#     if "k_scale" in name or "v_scale" in name:
#         print(f"{name}: {param.shape}, {param.dtype}, {param.item()}")
