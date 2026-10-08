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

"""The loss is computed under autocast only when mixed precision is enabled."""

import warnings
from types import SimpleNamespace

import pytest
import torch

from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer


def _get_loss(amp: bool, amp_dtype: torch.dtype):
    quantizer = SimpleNamespace(model_context=SimpleNamespace(amp=amp, amp_dtype=amp_dtype))
    pred = torch.ones(2, 4)
    ref = torch.zeros(2, 4)

    return SignRoundQuantizer._get_loss(
        quantizer,
        pred,
        ref,
        indices=torch.tensor([0, 1]),
        loss_func=torch.nn.functional.mse_loss,
        device="cpu",
    )


def test_no_autocast_when_amp_is_disabled():
    # Without mixed precision `amp_dtype` is float32, which CPU autocast does
    # not support: entering it warns and disables itself.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        loss = _get_loss(amp=False, amp_dtype=torch.float32)

    assert torch.isclose(loss, torch.tensor(1.0))


@pytest.mark.parametrize("amp_dtype", [torch.bfloat16, torch.float32])
def test_loss_is_finite_with_amp_enabled(amp_dtype):
    loss = _get_loss(amp=True, amp_dtype=amp_dtype)

    assert torch.isfinite(loss)
