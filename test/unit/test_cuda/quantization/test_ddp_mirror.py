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
"""CUDA-tier tests for DDP mirror building.

The resolver requires single-device block placement (a block whose weights
span several CUDA devices is declined loudly), so
ReplicaGroup builds consistent single-device replicas from a whole source.
"""

import pytest
import torch

from auto_round.algorithms.parallel.data_parallel import (
    DDPPlan,
    ReplicaGroup,
    resolve_tune_ddp_plan_,
)

_requires_multi_cuda = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs >=2 CUDA devices for a sharded source block"
)


class _TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = torch.nn.Linear(8, 8, bias=False)
        self.l1 = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.l1(torch.nn.functional.relu(self.l0(x)))


def _make_sharded_block():
    block = _TinyBlock().cuda(0)
    block.l0.tuning_device = "cuda:0"
    block.l1.tuning_device = "cuda:1"
    block.l1.to("cuda:1")  # emulate the auto block device map shard
    # plain-attr tensors + a params-dict entry, the wrapper state _relocate covers
    block.l0.anchor = torch.zeros(4, device="cuda:0")
    block.l1.anchor = torch.zeros(4, device="cuda:1")
    block.l0.params = {"v": torch.nn.Parameter(torch.ones(8, device="cuda:0"), requires_grad=True)}
    block.l1.params = {"v": torch.nn.Parameter(torch.ones(8, device="cuda:1"), requires_grad=True)}
    return block


def _make_whole_block():
    block = _TinyBlock().cuda(0)
    block.l0.tuning_device = "cuda:0"
    block.l1.tuning_device = "cuda:0"
    block.l0.anchor = torch.zeros(4, device="cuda:0")
    block.l1.anchor = torch.zeros(4, device="cuda:0")
    block.l0.params = {"v": torch.nn.Parameter(torch.ones(8, device="cuda:0"), requires_grad=True)}
    block.l1.params = {"v": torch.nn.Parameter(torch.ones(8, device="cuda:0"), requires_grad=True)}
    return block


def _quantizer():
    from types import SimpleNamespace

    q = SimpleNamespace(iters=10, gradient_accumulate_steps=1, enable_lfq=False)
    q._get_scaler = lambda: None
    return q


def _quantizer_with_policy(world=2):
    from types import SimpleNamespace

    from auto_round.algorithms.parallel.data_parallel import ParallelPolicy

    q = _quantizer()
    q.compress_context = SimpleNamespace(parallel_policy=ParallelPolicy(world=world))
    return q


@_requires_multi_cuda
class TestShardedSourceDeclines:
    def test_spanning_block_raises_with_placement_reason(self, monkeypatch):
        block = _make_sharded_block()
        with pytest.raises(RuntimeError, match="spans 2 CUDA devices"):
            resolve_tune_ddp_plan_(_quantizer_with_policy(), block, [torch.zeros(1, 8)], None, "cuda:0")


@_requires_multi_cuda
class TestReplicaGroupFromSingleDeviceSource:
    def test_mirrors_are_single_device_and_consistent(self):
        block = _make_whole_block()
        plan = DDPPlan(world=2, devices=[torch.device("cuda:0"), torch.device("cuda:1")], shard_size=1)
        group = ReplicaGroup(block, plan)
        try:
            assert group.world == 2
            home_dev = torch.device("cuda:0")
            assert {p.device for p in block.parameters()} == {home_dev}
            assert str(block.l1.tuning_device) == str(home_dev)
            # mirror whole on its device, including wrapper-style state
            mirror = group.mirrors[0]
            mirror_dev = torch.device("cuda:1")
            assert {p.device for p in mirror.parameters()} == {mirror_dev}
            assert mirror.l0.anchor.device == mirror_dev
            assert mirror.l0.params["v"].device == mirror_dev
            assert mirror.l1.params["v"].device == mirror_dev
            assert str(mirror.l0.tuning_device) == str(mirror_dev)
            # per-replica round params resolve on the replica devices
            params = group.round_params()
            assert len(params) == 2
            assert all(p.device == mirror_dev for p in params[1])
            assert all(p.device == home_dev for p in params[0])
        finally:
            group.teardown()
