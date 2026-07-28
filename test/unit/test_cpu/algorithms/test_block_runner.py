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

import torch

from auto_round.algorithms.block_runner import BlockForwardRunner


class GlmMoeDsaDecoderLayer(torch.nn.Module):
    def __init__(self, shared=False):
        super().__init__()
        self.shared = shared

    def forward(self, hidden_states, prev_topk_indices=None):
        if self.shared:
            assert prev_topk_indices is not None
            expected_indices = (hidden_states[..., :1] - 1).to(torch.long)
            torch.testing.assert_close(prev_topk_indices, expected_indices)
            topk_indices = prev_topk_indices
        else:
            topk_indices = hidden_states[..., :1].to(torch.long)
        return hidden_states + 1, topk_indices


class TupleOutputBlock(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states + 1, hidden_states + 2


def test_glm_dsa_topk_indices_propagate_between_blocks():
    runner = BlockForwardRunner(batch_size=2, device="cpu", cache_device="cpu", amp=False)
    inputs = [torch.full((1, 3, 2), value, dtype=torch.float32) for value in range(3)]

    reference_outputs = runner(GlmMoeDsaDecoderLayer(), inputs, {})
    next_inputs = runner.last_output_dict

    assert next_inputs is not None
    assert set(next_inputs) == {"hidden_states", "prev_topk_indices"}
    assert next_inputs["hidden_states"] is reference_outputs

    shared_outputs = runner(GlmMoeDsaDecoderLayer(shared=True), next_inputs, {})

    assert len(shared_outputs) == len(inputs)
    assert runner.last_output_dict is not None


def test_unregistered_tuple_output_keeps_first_tensor_behavior():
    runner = BlockForwardRunner(batch_size=2, device="cpu", cache_device="cpu", amp=False)
    inputs = [torch.zeros((1, 2, 2)), torch.ones((1, 2, 2))]

    outputs = runner(TupleOutputBlock(), inputs, {})

    assert runner.last_output_dict is None
    for output, input_tensor in zip(outputs, inputs):
        torch.testing.assert_close(output, input_tensor + 1)


def test_indexed_single_sample_forward_preserves_one_batch_dimension():
    runner = BlockForwardRunner(
        batch_dim=0,
        batch_size=1,
        device="cpu",
        cache_device="cpu",
        amp=True,
        amp_dtype=torch.bfloat16,
    )
    sample = torch.randn(1, 3, 4)

    output = runner(torch.nn.Identity(), [sample], {}, indices=torch.tensor([0]))

    assert output.shape == sample.shape
    torch.testing.assert_close(output, sample)


def test_indexed_diffusion_outputs_preserve_batch_dimension():
    class FluxTransformerBlock(torch.nn.Module):
        def forward(self, hidden_states, **_kwargs):
            return hidden_states + 1, hidden_states + 2

    runner = BlockForwardRunner(
        batch_dim=0,
        batch_size=1,
        device="cpu",
        cache_device="cpu",
        amp=True,
        amp_dtype=torch.bfloat16,
        is_diffusion=True,
    )
    sample = torch.randn(1, 3, 4)

    output = runner(
        FluxTransformerBlock(),
        {"hidden_states": [sample]},
        {},
        indices=torch.tensor([0]),
    )

    assert output.shape == sample.shape
    assert runner.last_output_dict["encoder_hidden_states"].shape == sample.shape
    assert runner.last_output_dict["hidden_states"].shape == sample.shape
