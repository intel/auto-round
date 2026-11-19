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

"""Tests for calibration/inputs.py."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from auto_round.calibration.inputs import split_inputs
from auto_round.calibration.utils import _update_inputs


class TestSplitInputs:
    """Tests for split_inputs."""

    def test_non_v4_keeps_original_token_normalization(self):
        hidden = [torch.randn(1, 4, 8)]
        model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
        with patch(
            "auto_round.context.model.ModelContext", return_value=SimpleNamespace(is_diffusion=False, model=model)
        ):
            normalized, _ = _update_inputs({"hidden_states": hidden, "input_ids": [torch.ones(1, 4)]}, None)
        assert normalized == {"input_ids": hidden}
        _, others = split_inputs(normalized, "input_ids", is_diffusion=False, model_type="llama")
        assert others == {}

    def test_real_token_ids_survive_hidden_state_normalization(self):
        hidden = [torch.randn(1, 4, 8), torch.randn(1, 4, 8)]
        tokens = [torch.tensor([[2, 3, 4, 5]]), torch.tensor([[6, 7, 8, 9]])]
        cache = {"hidden_states": hidden, "input_ids": tokens}
        model = SimpleNamespace(config=SimpleNamespace(model_type="deepseek_v4"))
        with patch(
            "auto_round.context.model.ModelContext", return_value=SimpleNamespace(is_diffusion=False, model=model)
        ):
            normalized, _ = _update_inputs(cache, None)
        actual_hidden, others = split_inputs(normalized, "input_ids", is_diffusion=False, model_type="deepseek_v4")
        assert actual_hidden is hidden
        assert others["input_ids"] is tokens
        assert "_ar_block_token_ids" not in others

    def test_diffusion_extracts_hidden_state(self):
        """Diffusion mode extracts the primary hidden_states; aux tensors stay for replay."""
        inputs = {
            "hidden_states": torch.randn(2, 4),
            "hidden_state_v2": torch.randn(2, 4),
            "attention_mask": torch.randn(2, 4),
        }
        input_ids, input_others = split_inputs(inputs, "input_ids", is_diffusion=True)

        assert "hidden_states" in input_ids
        assert "hidden_state_v2" in input_others
        assert "attention_mask" in input_others
        assert "hidden_states" not in input_others
        # Original dict was mutated
        assert "hidden_states" not in inputs

    def test_diffusion_shared_cache_keys_excluded(self):
        """Test shared_cache_keys are NOT extracted even if they contain hidden_state."""
        inputs = {
            "hidden_states": torch.randn(2, 4),
            "shared_key": torch.randn(2, 4),
        }
        input_ids, input_others = split_inputs(
            inputs, "input_ids", is_diffusion=True, shared_cache_keys=("shared_key",)
        )

        assert "hidden_states" in input_ids
        assert "shared_key" not in input_ids
        assert "shared_key" in input_others

    def test_non_diffusion_pops_first_input(self):
        """Test non-diffusion pops first_input_name from inputs."""
        inputs = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.randn(2, 4),
        }
        result_ids, input_others = split_inputs(inputs, "input_ids", is_diffusion=False)

        assert torch.equal(result_ids, torch.tensor([1, 2, 3]))
        assert "input_ids" not in input_others
        assert "attention_mask" in input_others

    def test_non_diffusion_missing_first_input(self):
        """Test non-diffusion returns None when first_input_name is absent."""
        inputs = {"attention_mask": torch.randn(2, 4)}
        result_ids, input_others = split_inputs(inputs, "input_ids", is_diffusion=False)

        assert result_ids is None
        assert "attention_mask" in input_others

    def test_diffusion_empty_hidden_state(self):
        """Test diffusion returns empty dict when no hidden_state keys."""
        inputs = {"attention_mask": torch.randn(2, 4)}
        input_ids, input_others = split_inputs(inputs, "input_ids", is_diffusion=True)

        assert input_ids == {}
        assert "attention_mask" in input_others
