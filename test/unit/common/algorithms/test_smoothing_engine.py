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

from auto_round.algorithms.transforms.smoothing import SmoothEngine, SmoothGroup


class IndependentStrategy:
    """A third-party policy with no dependency on AWQ or SVDQuant."""

    def __init__(self, events, fail_at=None):
        self.events = events
        self.fail_at = fail_at
        self.target = None

    def _stage(self, stage):
        self.events.append(stage)
        if stage == self.fail_at:
            raise RuntimeError(stage)

    def prepare(self, group, calibration):
        self.target = sum(calibration) / len(calibration)
        self._stage("prepare")

    def candidates(self):
        return [1.0, 2.0, 3.0]

    def score(self, candidate):
        self._stage("score")
        return (candidate - self.target) ** 2

    def apply(self, candidate):
        self._stage("apply")
        return candidate * 10

    def clear(self):
        self.target = None
        self.events.append("clear")


def make_group(key="custom"):
    # Generic groups also support non-linear targets; algorithms own constraints.
    module = torch.nn.Identity()
    return SmoothGroup(key, (key,), (module,), key, module, key, module)


def test_independent_strategy_can_extend_engine():
    events = []
    strategy = IndependentStrategy(events)
    assert SmoothEngine().run(make_group(), [1.0, 3.0], strategy) == 20
    assert events == ["prepare", "score", "score", "score", "apply", "clear"]
    assert strategy.target is None


@pytest.mark.parametrize("stage", ["prepare", "score", "apply"])
def test_engine_cleans_strategy_after_failure(stage):
    events = []
    strategy = IndependentStrategy(events, fail_at=stage)
    with pytest.raises(RuntimeError, match=stage):
        SmoothEngine().run(make_group(), [2.0], strategy)
    assert events[-1] == "clear"
    assert strategy.target is None


def test_search_and_apply_can_be_separated_for_multiple_groups():
    engine = SmoothEngine()
    groups = [make_group("first"), make_group("second")]
    events = []
    strategies = [IndependentStrategy(events), IndependentStrategy(events)]
    with engine.session(strategies[0]), engine.session(strategies[1]):
        results = [engine.search(group, [1.5], strategy) for group, strategy in zip(groups, strategies)]
        assert "apply" not in events
        # First candidate wins an exact tie for any independent policy.
        assert all(result.candidate == 1.0 for result in results)
        assert [
            engine.apply(group, strategy, result) for group, strategy, result in zip(groups, strategies, results)
        ] == [10, 10]
    assert events[-2:] == ["clear", "clear"]


def test_result_cannot_be_applied_to_another_group():
    engine = SmoothEngine()
    strategy = IndependentStrategy([])
    with engine.session(strategy):
        result = engine.search(make_group("source"), [2.0], strategy)
        with pytest.raises(ValueError, match="different group"):
            engine.apply(make_group("other"), strategy, result)
