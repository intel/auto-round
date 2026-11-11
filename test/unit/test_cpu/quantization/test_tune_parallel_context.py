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
"""Unit tests for the collection-phase TuneParallelContext.

The engine primitives (sharded_nograd_forward, ...) are faked: these tests
pin the context's collection-path semantics and serial fallbacks. The tune
itself runs on thread ranks (test_spmd_ranks).
"""

import logging
from unittest import mock

import pytest
import torch

from auto_round.algorithms.parallel.tune_parallel import TuneParallelContext


class TestCollectionContext:
    def test_for_collection_non_list_inputs_keeps_serial(self):
        composer = mock.Mock()
        ctx = TuneParallelContext.for_collection(composer, block=torch.nn.Linear(2, 2), fp_inputs={"a": [1]})
        assert ctx.devices is None

    def test_collect_forward_serial_when_devices_none(self):
        ctx = TuneParallelContext()
        bf = mock.Mock(return_value="out")
        assert ctx.collect_forward(bf, "blk", "inp", "oth", out_dev="cpu") == "out"
        bf.assert_called_once_with("blk", "inp", "oth", cache_device="cpu")

    def test_collect_forward_sharded_when_devices_set(self):
        ctx = TuneParallelContext()
        ctx.devices = [torch.device("cpu"), torch.device("cpu")]
        with mock.patch(
            "auto_round.algorithms.parallel.tune_parallel.sharded_nograd_forward",
            return_value="sharded",
        ) as snf:
            assert ctx.collect_forward("bf", "blk", "inp", "oth") == "sharded"
        from auto_round.algorithms.parallel.data_parallel import ENGINE_CALIBRATION_STATS

        snf.assert_called_once_with(
            "bf",
            "blk",
            "inp",
            "oth",
            None,
            ctx.devices,
            merge_stats=ENGINE_CALIBRATION_STATS,
            pool=None,
            max_devices=0,
        )

    def test_collect_forward_undeclared_stats_stay_serial(self):
        """merge_stats=None (quantizer hooks without a calibration_stats
        declaration): the pass must not shard -- the mirrors' statistics
        could not be folded back."""
        ctx = TuneParallelContext()
        ctx.devices = [torch.device("cpu"), torch.device("cpu")]
        bf = mock.Mock(return_value="serial")
        with mock.patch("auto_round.algorithms.parallel.tune_parallel.sharded_nograd_forward") as snf:
            assert ctx.collect_forward(bf, "blk", "inp", "oth", hook_pass=True, merge_stats=None) == "serial"
        snf.assert_not_called()
        bf.assert_called_once_with("blk", "inp", "oth", cache_device=None)

    def test_collect_forward_hook_pass_cap_from_policy(self):
        ctx = TuneParallelContext()
        ctx.devices = [torch.device("cpu")] * 8
        with mock.patch(
            "auto_round.algorithms.parallel.tune_parallel.sharded_nograd_forward",
            return_value="sharded",
        ) as snf:
            ctx.collect_forward("bf", "blk", "inp", "oth", hook_pass=True)
            # default: no cap (full-width hook passes)
            assert snf.call_args.kwargs["max_devices"] == 0
            # the cap is set on the context (folded from the env into the
            # ParallelPolicy at the entry)
            ctx.collect_forward_cap = 2
            ctx.collect_forward("bf", "blk", "inp", "oth", hook_pass=True)
            assert snf.call_args.kwargs["max_devices"] == 2
            # non-hook passes stay uncapped
            ctx.collect_forward("bf", "blk", "inp", "oth", hook_pass=False)
            assert snf.call_args.kwargs["max_devices"] == 0

    def test_capped_hook_pass_keeps_the_pool(self):
        """The cap limits WHICH pool replicas run, never pool eligibility:
        the full-width pool is passed through with max_devices set."""
        ctx = TuneParallelContext()
        ctx.devices = [torch.device("cpu")] * 8
        ctx.collect_forward_cap = 2
        sentinel = object()
        ctx.pool = sentinel
        blk = torch.nn.Linear(2, 2)  # pool eligibility requires a real Module
        with mock.patch(
            "auto_round.algorithms.parallel.tune_parallel.sharded_nograd_forward",
            return_value="sharded",
        ) as snf:
            assert ctx.collect_forward("bf", blk, [1] * 8, "oth", hook_pass=True) == "sharded"
        assert snf.call_args.kwargs["pool"] is sentinel
        assert snf.call_args.kwargs["max_devices"] == 2

    def test_distribute_pools_noop_without_devices(self):
        ctx = TuneParallelContext()
        pool = [torch.zeros(1), torch.zeros(1)]
        with mock.patch("auto_round.algorithms.parallel.tune_parallel.distribute_pool") as dp:
            ctx.distribute_pools(pool, None)
        dp.assert_not_called()


@pytest.fixture()
def _autoround_log_propagate():
    """Temporarily enable propagation on the ``autoround`` logger so pytest's
    caplog fixture (handler at the root logger) can capture warnings; the
    logger is configured with propagate=False in production."""

    _logger = logging.getLogger("autoround")
    original = _logger.propagate
    _logger.propagate = True
    yield
    _logger.propagate = original
