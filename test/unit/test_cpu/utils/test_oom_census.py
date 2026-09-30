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
"""Census grouping/attribution helpers (CPU-safe: fake 'accelerator' via
patched device check is unnecessary -- grouping only skips cpu/meta)."""

import torch

from auto_round.utils.oom import _group_tensors_by_shape, _is_oom


def test_group_tensors_skips_cpu_and_meta():
    objs = [torch.zeros(4), torch.zeros(4, device="meta"), torch.ones(4)]
    groups, skipped = _group_tensors_by_shape(objs)
    assert groups == [] and skipped == 0


def test_is_oom_matches_torch_and_message():
    assert _is_oom(RuntimeError("CUDA out of memory"))
    assert not _is_oom(RuntimeError("something else"))
