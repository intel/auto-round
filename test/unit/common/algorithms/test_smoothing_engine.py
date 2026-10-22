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

    def __init__(self):
        self.target = None
        self.applied = False

    def prepare(self, group, calibration):
        self.target = sum(calibration) / len(calibration)

    def candidates(self):
        return [1.0, 2.0, 3.0]

    def score(self, candidate):
        return (candidate - self.target) ** 2

    def apply(self, candidate):
        self.applied = True
        return candidate

    def clear(self):
        self.target = None


def make_group(key="custom"):
    # Generic groups also support non-linear targets; algorithms own constraints.
    module = torch.nn.Identity()
    return SmoothGroup(key, (key,), (module,), key, module, key, module)


def test_independent_strategy_can_extend_engine():
    strategy = IndependentStrategy()
    assert SmoothEngine().run(make_group(), [1.0, 3.0], strategy) == 2.0
    assert strategy.target is None


def test_engine_cleans_strategy_after_failure(monkeypatch):
    strategy = IndependentStrategy()

    def fail(candidate):
        raise RuntimeError("score failed")

    monkeypatch.setattr(strategy, "score", fail)
    with pytest.raises(RuntimeError, match="score failed"):
        SmoothEngine().run(make_group(), [2.0], strategy)
    assert strategy.target is None


def test_search_does_not_apply_until_requested():
    engine = SmoothEngine()
    group = make_group()
    strategy = IndependentStrategy()
    with engine.session(strategy):
        result = engine.search(group, [2.0], strategy)
        assert not strategy.applied
        assert engine.apply(group, strategy, result) == 2.0
        assert strategy.applied


def test_result_cannot_be_applied_to_another_group():
    engine = SmoothEngine()
    strategy = IndependentStrategy()
    with engine.session(strategy):
        result = engine.search(make_group("source"), [2.0], strategy)
        with pytest.raises(ValueError, match="different group"):
            engine.apply(make_group("other"), strategy, result)
