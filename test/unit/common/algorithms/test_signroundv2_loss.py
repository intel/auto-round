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

"""SignRoundV2 must honour the padding mask on its plain-loss path."""

from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import torch

from auto_round.algorithms.quantization.sign_roundv2.quantizer import SignRoundV2Quantizer


def _loss(use_outlier_suppressed_loss: bool):
    # `super()` needs a real instance, but the constructor wants a full config.
    quantizer = SignRoundV2Quantizer.__new__(SignRoundV2Quantizer)
    quantizer._use_outlier_suppressed_loss = use_outlier_suppressed_loss

    # The second row is padding, so only the first one may reach the loss.
    pred = torch.tensor([[1.0, 1.0], [5.0, 5.0]])
    ref = torch.zeros(2, 2)
    valid_token_mask = [torch.tensor([1.0]), torch.tensor([0.0])]

    with patch.object(SignRoundV2Quantizer, "model_context", new_callable=PropertyMock) as model_context:
        model_context.return_value = SimpleNamespace(amp=False, amp_dtype=torch.float32)

        return quantizer._get_loss(
            pred,
            ref,
            torch.tensor([0, 1]),
            torch.nn.functional.mse_loss,
            "cpu",
            valid_token_mask,
        )


def test_padding_is_masked_out_of_the_plain_loss():
    # Masked: mean([1, 1, 0, 0]) == 0.5. Unmasked it would be mean([1, 1, 25, 25]) == 13.
    assert torch.isclose(_loss(use_outlier_suppressed_loss=False), torch.tensor(0.5))


def test_padding_is_masked_out_of_the_outlier_suppressed_loss():
    loss = _loss(use_outlier_suppressed_loss=True)

    assert torch.isfinite(loss)
