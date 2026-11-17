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

from auto_round.data_type.int import search_scales


def test_search_scales_top_three_is_reproducible_without_global_rng_mutation():
    data = torch.linspace(-1.0, 1.0, 256).reshape(8, 32)
    torch.manual_seed(1234)
    expected_rng_values = torch.rand(3)

    torch.manual_seed(1234)
    first = search_scales(data, bits=4, candidate_count=3)
    actual_rng_values = torch.rand(3)
    second = search_scales(data, bits=4, candidate_count=3)

    assert torch.equal(first, second)
    assert torch.equal(expected_rng_values, actual_rng_values)
