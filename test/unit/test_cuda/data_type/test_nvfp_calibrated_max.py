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

import pytest
import torch

from auto_round.data_type.nvfp import nvfp4_v2_with_global_scale


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cpu_tensor_max_with_cuda_input():
    tensor = torch.ones(1, 16, device="cuda")
    q_float, scale_float, _ = nvfp4_v2_with_global_scale(tensor, tensor_max=2.0)
    q_cpu, scale_cpu, _ = nvfp4_v2_with_global_scale(tensor, tensor_max=torch.tensor([2.0]))
    torch.testing.assert_close(q_cpu, q_float)
    torch.testing.assert_close(scale_cpu, scale_float)
