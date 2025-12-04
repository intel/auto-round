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
import torch
from vllm.platforms import current_platform


def cuda_capability_at_least(major, minor):
    device_capability = torch.cuda.get_device_capability()
    return device_capability[0] >= major or (device_capability[0] == major and device_capability[1] >= minor)


MODELS = ["/home/yiliu7/workspace/auto-round/examples/Qwen2.5-0.5B-Instruct-ar-MXFP4-fp8"]


@pytest.fixture(autouse=True)
def set_vllm_ar_env(monkeypatch):
    monkeypatch.setenv("VLLM_AR_MXFP4_MODULAR_MOE", "1")
    monkeypatch.setenv("VLLM_MXFP4_PRE_UNPACK_TO_FP8", "1")
    monkeypatch.setenv("VLLM_MXFP4_PRE_UNPACK_WEIGHTS", "0")
    monkeypatch.setenv("VLLM_ENABLE_STATIC_MOE", "0")
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "0")
    monkeypatch.setenv("VLLM_ENABLE_AR_EXT", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION", "1")
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "FLASHINFER")


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="only supports CUDA backend.",
)
@pytest.mark.skipif(
    not cuda_capability_at_least(10, 0), reason="FP8 KV cache only supported on CUDA with compute capability >= 10.0"
)
@pytest.mark.parametrize("model", MODELS)
def test_auto_fp8_kv(vllm_runner, model):
    with vllm_runner(model, 
    enforce_eager=True, 
    kv_cache_dtype="fp8", gpu_memory_utilization=0.1) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=8)
        assert (
            llm.llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.kv_cache_dtype
            == torch.uint8
        ), f"Expected kv_cache_dtype to be torch.uint8, but got {llm.llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner.kv_cache_dtype}"
    assert output
    print(f"output is: {output[0][1]}")
