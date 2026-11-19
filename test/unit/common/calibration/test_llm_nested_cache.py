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

import pytest
import torch

from auto_round.algorithms.block_runner import BlockForwardRunner
from auto_round.calibration.llm import LLMCalibrator
from auto_round.modeling.deepseek_v4 import select_shared_rope
from auto_round.parallel.execution import ParallelExecution


class NestedRopeBlock(torch.nn.Module):
    def forward(self, hidden_states, **kwargs):
        rope = kwargs["position_embeddings"]
        return hidden_states + rope["main"][0] + rope["compress"][1]


def test_non_v4_runner_does_not_select_nested_dicts():
    rope = {"main": (torch.randn(2, 4, 8), torch.randn(2, 4, 8))}
    runner = BlockForwardRunner(amp=False, shared_cache_keys=("position_embeddings",), enable_torch_compile=False)
    _, others = runner.select_batch([torch.zeros(1, 4, 8)] * 2, {"position_embeddings": rope}, torch.tensor([1]))
    assert others["position_embeddings"] is rope


def test_non_v4_capture_does_not_enable_nested_rope():
    calibrator = object.__new__(LLMCalibrator)
    calibrator.model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
    calibrator.parallel = ParallelExecution()
    calibrator.inputs = {}
    calibrator.batch_dim = 0
    calibrator.batch_size = 1
    calibrator.shared_cache_keys = ("position_embeddings",)
    calibrator.has_variable_block_shape = False
    calibrator._should_stop_cache_forward = lambda name: True
    capture = calibrator._make_block_forward_func("block")
    with pytest.raises(NotImplementedError):
        capture(NestedRopeBlock(), torch.zeros(1, 4, 8), position_embeddings={"main": (torch.ones(1, 4, 8),)})
    assert "position_embeddings" not in calibrator.inputs["block"]


@pytest.mark.parametrize("model_type,enabled", [("deepseek_v4", True), ("llama", False)])
def test_runner_factory_selects_v4_adapter_only(model_type, enabled):
    model = SimpleNamespace(config=SimpleNamespace(model_type=model_type))
    orchestrator = SimpleNamespace(model_context=SimpleNamespace(model=model))
    runner = BlockForwardRunner.from_orchestrator(orchestrator, enable_torch_compile=False)
    assert (runner.shared_cache_selector is select_shared_rope) == enabled


@pytest.mark.parametrize("batch_size", [1, 2])
def test_capture_and_replay_nested_shared_rope(batch_size):
    calibrator = object.__new__(LLMCalibrator)
    calibrator.model = SimpleNamespace(config=SimpleNamespace(model_type="deepseek_v4"))
    calibrator.parallel = ParallelExecution()
    calibrator.inputs = {}
    calibrator.batch_dim = 0
    calibrator.batch_size = batch_size
    calibrator.shared_cache_keys = ("position_embeddings",)
    calibrator.has_variable_block_shape = False
    calibrator._should_stop_cache_forward = lambda name: True
    block = NestedRopeBlock()
    capture = calibrator._make_block_forward_func("block")
    expected = []
    for offset in (1.0, 3.0):
        hidden = torch.zeros(batch_size, 4, 8)
        rope = {
            "main": (torch.full((1, 4, 8), offset), torch.zeros(1, 4, 8)),
            "compress": (torch.zeros(1, 4, 8), torch.ones(1, 4, 8)),
        }
        expected.append(block(hidden, position_embeddings=rope))
        with pytest.raises(NotImplementedError):
            capture(block, hidden, position_embeddings=rope)
    cached = dict(calibrator.inputs["block"])
    assert "position_embeddings" in cached
    inputs = cached.pop("hidden_states")
    runner = BlockForwardRunner(
        batch_size=batch_size,
        amp=False,
        shared_cache_keys=calibrator.shared_cache_keys,
        enable_torch_compile=False,
        shared_cache_selector=select_shared_rope,
    )
    selections = [[index] for index in range(batch_size * 2)] + [[batch_size, 0], [batch_size * 2 - 1, 0, batch_size]]
    for indices in selections:
        selected, others = runner.select_batch(inputs, cached, torch.tensor(indices))
        actual = runner._forward_one_batch(block, selected, others, eager=True)
        reference = torch.cat(
            [expected[index // batch_size][index % batch_size : index % batch_size + 1] for index in indices]
        )
        torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("batch_dim", [0, 1])
@pytest.mark.parametrize("collection_batches", [1, 3])
def test_nested_rope_selects_rows_within_collection_batches(batch_dim, collection_batches):
    collection_size = 2
    shape = (collection_size, 3, 4) if batch_dim == 0 else (3, collection_size, 4)
    hidden = torch.zeros(shape)
    inputs = [sample for _ in range(collection_batches) for sample in hidden.split(1, dim=batch_dim)]
    entries = []
    for batch in range(collection_batches):
        value = torch.arange(hidden.numel()).reshape(shape) + 100 * batch
        entries.append({"nested": {"main": (value, value + 1)}, "metadata": "shared"})
    runner = BlockForwardRunner(
        batch_size=2,
        batch_dim=batch_dim,
        amp=False,
        shared_cache_keys=("position_embeddings",),
        shared_cache_selector=select_shared_rope,
        enable_torch_compile=False,
    )
    cached = entries[0] if collection_batches == 1 else entries
    for indices in ([1], [1, 0], [len(inputs) - 1, 0, 1]):
        _, others = runner.select_batch(inputs, {"position_embeddings": cached}, torch.tensor(indices))
        expected = torch.cat(
            [
                entries[i // collection_size]["nested"]["main"][0].narrow(batch_dim, i % collection_size, 1)
                for i in indices
            ],
            dim=batch_dim,
        )
        torch.testing.assert_close(others["position_embeddings"]["nested"]["main"][0], expected)
        assert others["position_embeddings"]["metadata"] == "shared"
