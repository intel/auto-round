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

import contextlib
from collections import defaultdict
from functools import wraps

import torch
from transformers.models.llama.modeling_llama import eager_attention_forward


class RuntimeStats:
    INPUT_RANGES = {}
    cur_layer_idx = -1


# Save the original torch.matmul
original_matmul = torch.matmul


def matmul_record_inputs(mat1, mat2):
    if RuntimeStats.cur_layer_idx not in RuntimeStats.INPUT_RANGES:
        RuntimeStats.INPUT_RANGES[RuntimeStats.cur_layer_idx] = {}
    CUR_INPUT_RANGES = RuntimeStats.INPUT_RANGES[RuntimeStats.cur_layer_idx]
    # FIXME: record the max, and Q-DQ inputs
    if "mat1" not in CUR_INPUT_RANGES:
        cur_max = mat1.abs().max().item()
    else:
        cur_max = max(CUR_INPUT_RANGES["mat1"], mat1.abs().max().item())
    CUR_INPUT_RANGES["mat1"] = cur_max

    if "mat2" not in CUR_INPUT_RANGES:
        cur_max = mat2.abs().max().item()
    else:
        cur_max = max(CUR_INPUT_RANGES["mat2"], mat2.abs().max().item())
    CUR_INPUT_RANGES["mat2"] = cur_max
    return original_matmul(mat1, mat2)


# @contextlib.contextmanager
# def replace_matmul_with_record():
#     """
#     Context manager to temporarily replace torch.matmul with matmul_record_inputs.
#     """
#     # Save the original torch.matmul
#     original_matmul = torch.matmul

#     try:
#         # Replace torch.matmul with matmul_record_inputs
#         torch.matmul = matmul_record_inputs
#         yield  # Execute the code inside the context
#     finally:
#         # Restore the original torch.matmul
#         torch.matmul = original_matmul


def replace_matmul_decorator(func):
    """
    Decorator to temporarily replace torch.matmul with matmul_record_inputs
    during the execution of the decorated function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Save the original torch.matmul
        original_matmul = torch.matmul
        try:
            # Replace torch.matmul with matmul_record_inputs
            torch.matmul = matmul_record_inputs
            # Call the original function
            return func(*args, **kwargs)
        finally:
            # Restore the original torch.matmul
            torch.matmul = original_matmul

    return wrapper


@replace_matmul_decorator
def llama_eager_attention_forward(*args, **kwargs):

    return eager_attention_forward(*args, **kwargs)


# ATTN_FUNCTIONS = {llama.LlamaAttention: llama_eager_attention_forward}
