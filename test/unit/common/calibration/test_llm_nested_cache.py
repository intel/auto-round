import pytest
import torch

from auto_round.algorithms.block_runner import BlockForwardRunner
from auto_round.calibration.llm import LLMCalibrator
from auto_round.parallel.execution import ParallelExecution


class NestedRopeBlock(torch.nn.Module):
    def forward(self, hidden_states, **kwargs):
        rope = kwargs["position_embeddings"]
        return hidden_states + rope["main"][0] + rope["compress"][1]


@pytest.mark.parametrize("batch_size", [1, 2])
def test_capture_and_replay_nested_shared_rope(batch_size):
    calibrator = object.__new__(LLMCalibrator)
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
        batch_size=batch_size, amp=False, shared_cache_keys=calibrator.shared_cache_keys, enable_torch_compile=False
    )
    for index in range(batch_size * 2):
        selected, others = runner.select_batch(inputs, cached, torch.tensor([index]))
        actual = runner._forward_one_batch(block, selected, others, eager=True)
        torch.testing.assert_close(actual, expected[index // batch_size][index % batch_size : index % batch_size + 1])
