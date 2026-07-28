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
