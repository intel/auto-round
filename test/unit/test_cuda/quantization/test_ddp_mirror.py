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
"""CUDA-tier tests for DDP mirror building from module-sharded source blocks.

The data-driven multi-GPU lane shards a block's leaves across the device list
(set_auto_device_map_for_block_with_tuning), so the DDP source block may span
several devices when the plan resolves. These tests pin the gather-then-mirror
contract: the source is gathered whole onto the plan home (pinned subtrees
preserved, per-leaf tuning_device strings repointed) and ReplicaGroup builds
consistent single-device replicas from it.
"""

import pytest
import torch

from auto_round.algorithms.quantization.sign_round.data_parallel import (
    DDPPlan,
    ReplicaGroup,
    gather_block_for_mirroring_,
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


@_requires_multi_cuda
class TestGatherBlockForMirroring:
    def test_gather_collects_sharded_state_and_repoints_tuning_device(self):
        block = _make_sharded_block()
        home = torch.device("cuda:0")
        moved = gather_block_for_mirroring_(block, home)
        assert moved is True
        assert {p.device for p in block.parameters()} == {home}
        assert block.l0.anchor.device == home
        assert block.l1.anchor.device == home
        assert block.l0.params["v"].device == home
        assert block.l1.params["v"].device == home
        assert str(block.l1.tuning_device) == str(home)

    def test_gather_is_a_noop_on_a_whole_block(self):
        block = _TinyBlock().cuda(0)
        moved = gather_block_for_mirroring_(block, torch.device("cuda:0"))
        assert moved is False

    def test_gather_preserves_cpu_pinned_subtrees(self):
        block = _make_sharded_block()
        from auto_round.utils.model import _pin_module_execution_on_cpu

        pinned = torch.nn.Linear(4, 4, bias=False)  # stands in for a pinned ngram table
        block.table = pinned
        _pin_module_execution_on_cpu(pinned)  # what move_to_device_preserving_cpu_pinned keys on
        home = torch.device("cuda:0")
        gather_block_for_mirroring_(block, home)
        assert pinned.weight.device.type == "cpu"


@_requires_multi_cuda
class TestReplicaGroupFromShardedSource:
    def test_mirrors_are_single_device_and_consistent(self):
        block = _make_sharded_block()
        plan = DDPPlan(world=2, devices=[torch.device("cuda:0"), torch.device("cuda:1")], shard_size=1)
        # mirror the quantizer's actual sequence: gather the sharded source
        # whole onto the plan home BEFORE building the ReplicaGroup
        gather_block_for_mirroring_(block, torch.device("cuda:0"))
        group = ReplicaGroup(block, plan, grad_transport="bf16")
        try:
            assert group.world == 2
            # home gathered whole onto the plan leader
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
