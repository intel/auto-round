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

from auto_round.algorithms.transforms.smoothing.calibration import SmoothGroupCalibration, move_to_device
from auto_round.algorithms.transforms.smoothing.replay import (
    normalize_tensors,
    output_squared_error,
    temporary_modules,
)
from auto_round.algorithms.transforms.smoothing.search import (
    InvalidSmoothCandidateError,
    NoFiniteCandidateError,
    search_candidates,
)


def test_calibration_reservoir_keeps_paired_calls_and_detached_copies():
    capture = SmoothGroupCalibration(group="custom", limit=3)
    for index in range(30):
        inputs = torch.full((1, 2), float(index), requires_grad=True)
        capture.begin_call(inputs)
        capture.finish_call((inputs,), {"nested": [inputs]}, {"output": inputs * 2})
        with torch.no_grad():
            inputs.fill_(-1)
    assert capture.seen_calls == 30
    assert len(capture.projection_inputs) == len(capture.evaluation_calls) == 3
    assert capture.pending_input is None
    assert capture.pending_slot is None
    for inputs, call in zip(capture.projection_inputs, capture.evaluation_calls):
        assert inputs.grad_fn is None and not inputs.requires_grad
        assert inputs.min() >= 0
        torch.testing.assert_close(inputs, call.args[0])
        torch.testing.assert_close(inputs, call.kwargs["nested"][0])
        torch.testing.assert_close(inputs * 2, call.output["output"])


def test_replay_preserves_integer_masks_and_nested_outputs():
    values = {"hidden": [torch.ones(2)], "mask": torch.ones(2, dtype=torch.int64)}
    moved = move_to_device(values, torch.device("cpu"), torch.bfloat16)
    assert moved["hidden"][0].dtype == torch.bfloat16
    assert moved["mask"].dtype == torch.int64
    assert normalize_tensors(moved) == (moved["hidden"][0],)


def test_search_keeps_order_invalid_candidates_and_first_ties():
    visited = []
    invalid = []

    def score(candidate):
        visited.append(candidate)
        if candidate == "invalid":
            raise InvalidSmoothCandidateError("trial failed")
        return {"nan": float("nan"), "first": 1, "later": 1}[candidate]

    selected, error = search_candidates(
        ["first", "invalid", "nan", "later"],
        score,
        module_name="proj",
        on_invalid=lambda candidate, exc: invalid.append((candidate, str(exc))),
    )
    assert (selected, error) == ("first", 1)
    assert visited == ["first", "invalid", "nan", "later"]
    assert invalid[0] == ("invalid", "trial failed")
    assert invalid[1][0] == "nan"
    with pytest.raises(ValueError, match="no finite candidate"):
        search_candidates([1, 2], lambda _: float("inf"), module_name="proj")


@pytest.mark.parametrize("fail", [False, True])
def test_temporary_modules_restore_originals(fail):
    root = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    originals = list(root)
    trial = torch.nn.Identity()
    try:
        with temporary_modules(root, [("0", trial), ("1", trial)]):
            assert root[0] is trial and root[1] is trial
            if fail:
                raise RuntimeError("evaluation failed")
    except RuntimeError:
        assert fail
    assert list(root) == originals


def test_temporary_modules_restore_after_partial_install_failure():
    class FailOnce(torch.nn.Sequential):
        def set_submodule(self, name, module):
            if name == "1" and isinstance(module, torch.nn.Identity):
                raise RuntimeError("installation failed")
            return super().set_submodule(name, module)

    root = FailOnce(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
    originals = list(root)
    with (
        pytest.raises(RuntimeError, match="installation failed"),
        temporary_modules(root, [("0", torch.nn.Identity()), ("1", torch.nn.Identity())]),
    ):
        pytest.fail("partial installation must not reach evaluation")
    assert list(root) == originals


def test_error_reduction_matches_original_svdquant_accumulation():
    torch.manual_seed(7)
    actual = (torch.randn(2, 3), torch.randn(1, 4))
    reference = tuple(torch.randn_like(tensor) for tensor in actual)
    expected = torch.zeros((), dtype=torch.float64)
    for value, target in zip(actual, reference):
        expected += torch.sum((value.float() - target.float()).square()).double().cpu()
    torch.testing.assert_close(output_squared_error(actual, reference), expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="count changed"):
        output_squared_error(actual, reference[:1])
    with pytest.raises(ValueError, match="shape changed"):
        output_squared_error(actual[:1], (torch.zeros(1),))


def test_search_keeps_first_exact_tie():
    selected, error = search_candidates(["first", "last"], lambda _: 0.0, module_name="proj")
    assert (selected, error) == ("first", 0.0)


def test_search_can_propagate_awq_trial_failure():
    visited = []

    def score(candidate):
        visited.append(candidate)
        raise RuntimeError("QDQ failure")

    with pytest.raises(RuntimeError, match="QDQ failure"):
        search_candidates([0, 1], score, module_name="proj")
    assert visited == [0]


@pytest.mark.parametrize("errors", [[], [float("nan"), float("inf")]])
def test_no_finite_candidate_has_distinct_algorithm_fallback(errors):
    with pytest.raises(NoFiniteCandidateError):
        search_candidates(errors, lambda error: error, module_name="proj")


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError, TypeError, torch.OutOfMemoryError])
def test_execution_errors_are_never_skipped(error_type):
    def fail(_):
        raise error_type("execution failure")

    with pytest.raises(error_type, match="execution failure"):
        search_candidates([0, 1], fail, module_name="proj")


def test_calibration_clear_releases_pending_and_retained_calls():
    capture = SmoothGroupCalibration("group", 2)
    inputs = torch.ones(1, 2)
    capture.begin_call(inputs)
    capture.finish_call((inputs,), {}, inputs)
    capture.begin_call(inputs)
    capture.clear()
    assert not capture.projection_inputs and not capture.evaluation_calls
    assert capture.pending_input is None and capture.pending_slot is None
    assert capture.seen_calls == 0


@pytest.mark.parametrize("stage", ["fp", "q", "preprocess"])
def test_failed_preprocessor_removes_hooks_and_clears_cache(stage):
    from auto_round.algorithms.composer import AlgorithmComposer, BlockContext
    from auto_round.algorithms.quantization.rtn.config import RTNConfig
    from auto_round.algorithms.transforms.smoothing.calibration import clear_caches

    block = torch.nn.Linear(2, 2)

    class Probe:
        def __init__(self):
            self.cache = {"input": torch.ones(1, 2)}

        def register_fp_input_forward_hooks(self, module):
            return [module.register_forward_hook(lambda *args: None)]

        register_qinput_forward_hooks = register_fp_input_forward_hooks

        def pre_quantize_block(self, ctx):
            raise RuntimeError("preprocess failure")

        def post_quantize_block(self, ctx):
            clear_caches(self.cache, scope="probe")

    probe = Probe()
    composer = AlgorithmComposer([RTNConfig(disable_opt_rtn=True)])
    composer.preprocessors = [probe]
    calls = 0

    def forward(module, inputs, others):
        nonlocal calls
        calls += 1
        if (stage == "fp" and calls == 1) or (stage == "q" and calls == 2):
            raise RuntimeError("calibration failure")
        return module(inputs[0])

    composer.block_forward = forward
    ctx = BlockContext(model=block, block_names=["block"], block_name="block", block_index=0)
    with pytest.raises(RuntimeError, match="failure"):
        composer.compress_block(block, [torch.ones(1, 2)], {}, ctx)
    assert not block._forward_hooks
    assert not probe.cache
