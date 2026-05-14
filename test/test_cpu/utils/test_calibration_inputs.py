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

from types import SimpleNamespace

import torch

from auto_round.calibration.inputs import preprocess_block_inputs, split_inputs


def test_split_inputs_llm():
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }

    input_ids, input_others = split_inputs(inputs, "input_ids", is_diffusion=False)

    assert torch.equal(input_ids, torch.tensor([[1, 2, 3]]))
    assert "input_ids" not in inputs
    assert input_others is inputs
    assert set(input_others) == {"attention_mask"}


def test_split_inputs_diffusion():
    inputs = {
        "hidden_states": torch.randn(1, 4),
        "encoder_hidden_states": torch.randn(1, 4),
        "timestep": torch.tensor([1]),
    }

    input_ids, input_others = split_inputs(inputs, "ignored", is_diffusion=True)

    assert set(input_ids) == {"hidden_states", "encoder_hidden_states"}
    assert "hidden_states" not in input_others
    assert "encoder_hidden_states" not in input_others
    assert set(input_others) == {"timestep"}


def test_preprocess_block_inputs_casts_tensors(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "auto_round.calibration.inputs.clear_memory",
        lambda device_list: calls.append(tuple(device_list)),
    )

    model_context = SimpleNamespace(is_diffusion=False, amp=True, amp_dtype=torch.bfloat16)
    compress_context = SimpleNamespace(cache_device=torch.device("cpu"), device_list=["cpu"])
    inputs = {
        "input_ids": torch.ones((1, 3), dtype=torch.float16),
        "attention_mask": torch.ones((1, 3), dtype=torch.float16),
        "position_ids": torch.ones((1, 3), dtype=torch.int64),
    }

    input_ids, input_others = preprocess_block_inputs(
        inputs,
        model_context=model_context,
        compress_context=compress_context,
    )

    assert calls == [("cpu",)]
    assert input_ids.dtype == torch.bfloat16
    assert input_others["attention_mask"].dtype == torch.bfloat16
    assert input_others["position_ids"].dtype == torch.int64
    assert "input_ids" not in inputs


def test_preprocess_block_inputs_keeps_list_dtype():
    model_context = SimpleNamespace(is_diffusion=False, amp=True, amp_dtype=torch.bfloat16)
    compress_context = SimpleNamespace(cache_device=torch.device("cpu"), device_list=["cpu"])
    cached_states = [torch.ones((1, 3), dtype=torch.float16)]
    inputs = {
        "input_ids": torch.ones((1, 3), dtype=torch.float16),
        "past_key_values": cached_states,
    }

    _, input_others = preprocess_block_inputs(
        inputs,
        model_context=model_context,
        compress_context=compress_context,
    )

    # The extracted helper intentionally preserves the legacy implementation,
    # which iterates list entries through to_dtype() without writing the result
    # back.  Lock this down so the refactor does not change runtime behaviour.
    assert input_others["past_key_values"][0].dtype == torch.float16
