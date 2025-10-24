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

import pytest
from vllm.platforms import current_platform

MODELS = [
    # "/data5/yliu7/HF_HOME/unsloth-gpt-oss-20b-BF16-ar-MXFP4/"
    # "/data5/yliu7/HF_HOME/Qwen2.5-0.5B-Instruct-test-FP8_STATIC-fp8kv/"
    "/data6/yiliu4/Qwen3-15B-A2B-Base-MXFP4-fp8attention"
]


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="only supports CUDA backend.",
)
@pytest.mark.parametrize("model", MODELS)
def test_auto_round(vllm_runner, model):
    with vllm_runner(model, enforce_eager=True) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=8)
    assert output
    print(f"output is: {output[0][1]}")
