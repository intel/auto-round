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
"""Regression tests for estimate_tuning_block_mem wrapper-safety.

SignRound(V2) wrappers expose `bits` but keep `act_bits` on the wrapped
orig_layer; the estimator must survive such modules (previously an
AttributeError aborted the estimate) and must report the documented
4-tuple width the call site unpacks.
"""

import torch
import torch.nn as nn

from auto_round.utils.device import estimate_tuning_block_mem


def _wrapper_module():
    orig = nn.Linear(8, 8)
    orig.act_bits = 16  # what a real orig_layer carries

    class _Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = orig.weight
            self.bits = 4  # check_to_quantized reads this on the WRAPPER
            self.in_features = 8
            self.out_features = 8
            self.orig_layer = orig

    return _Wrapper()


def test_estimator_survives_v2_wrapper_modules():
    """Regression: wrapped modules raised AttributeError inside the
    estimator (act_bits lives on orig_layer, not the wrapper)."""
    block = nn.Sequential(_wrapper_module())
    inputs = [torch.randn(2, 4, 8)]
    result = estimate_tuning_block_mem(block, inputs, 2)
    assert isinstance(result, tuple) and len(result) == 4, result


def test_estimator_plain_linear_reports_four_tuple():
    block = nn.Sequential(nn.Linear(8, 8))
    inputs = [torch.randn(2, 4, 8)]
    result = estimate_tuning_block_mem(block, inputs, 2)
    assert isinstance(result, tuple) and len(result) == 4, result
