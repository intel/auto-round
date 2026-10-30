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
"""Tests for ``auto_round/utils/oom.py`` (OOM tensor census diagnostics).

Accelerator tensors are emulated with ``MagicMock(spec=torch.Tensor)`` so the
census logic runs on CPU-only machines.
"""

import sys
import threading
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from auto_round.utils import oom


def _fake_tensor(device="cuda:0", dtype=torch.float16, shape=(4, 8)):
    t = MagicMock(spec=torch.Tensor)
    t.device = torch.device(device)
    t.dtype = dtype
    t.shape = torch.Size(shape)
    t.numel.return_value = int(torch.Size(shape).numel())
    t.element_size.return_value = torch.empty((), dtype=dtype).element_size()
    return t


def _logged(logger_mock):
    return "\n".join(c.args[0] % c.args[1:] if len(c.args) > 1 else c.args[0] for c in logger_mock.error.call_args_list)


class Holder:
    def __init__(self, t):
        self.weight = t


class TestShapeAndGrouping:
    def test_shape_key(self):
        assert oom._shape_key(torch.Size([2, 3])) == (2, 3)

    def test_shape_key_falls_back_to_string(self):
        class Weird:
            def __iter__(self):
                return iter(["a"])

        assert oom._shape_key(Weird()) == ("('a',)",)

    def test_group_tensors_by_shape(self):
        big = _fake_tensor(shape=(64, 64))
        small_a = _fake_tensor(shape=(2, 2))
        small_b = _fake_tensor(shape=(2, 2))
        other_dev = _fake_tensor(device="xpu:0", dtype=torch.float32, shape=(8,))
        broken = _fake_tensor()
        broken.numel.side_effect = RuntimeError("unreadable")
        objs = [torch.zeros(10), "not a tensor", big, small_a, small_b, other_dev, broken]

        groups, skipped = oom._group_tensors_by_shape(objs)
        assert skipped == 1
        assert groups[0] == (("cuda:0", "torch.float16", (64, 64)), [1, 64 * 64 * 2])
        as_dict = dict(groups)
        assert as_dict[("cuda:0", "torch.float16", (2, 2))] == [2, 16]
        assert as_dict[("xpu:0", "torch.float32", (8,))] == [1, 32]
        assert all(key[0] != "cpu" for key in as_dict)

    def test_representatives_one_per_group(self):
        a1, a2 = _fake_tensor(shape=(3,)), _fake_tensor(shape=(3,))
        b = _fake_tensor(shape=(5,))
        bad = MagicMock(spec=torch.Tensor)
        type(bad).device = property(lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        objs = [bad, torch.ones(1), a1, a2, b]
        groups, _ = oom._group_tensors_by_shape([a1, a2, b])
        reps = list(oom._representatives(groups, objs))
        assert [r[0] for r in reps] == [a1, b] or [r[0] for r in reps] == [b, a1]
        assert len(reps) == 2


class TestReferrerDescriptions:
    def test_is_census_noise(self):
        skip = [1]
        assert oom._is_census_noise(skip, {id(skip)})
        assert oom._is_census_noise(list(range(5000)), set())
        assert oom._is_census_noise({("a", "b", "c"): 1}, set())
        assert not oom._is_census_noise({"x": 1}, set())
        assert not oom._is_census_noise([1, 2], set())

    def test_describe(self):
        assert oom._describe({"a": 1, "b": 2}) == "dict[a,b]"
        assert oom._describe([1, 2, 3]) == "list(len=3)"
        assert oom._describe((1,)) == "tuple(len=1)"
        assert oom._describe(sys._getframe()) == "frame"
        assert oom._describe(i for i in range(2)) == "generator"
        assert oom._describe(Holder(None)).endswith(".Holder")
        assert oom._describe(3) == "int"

    def test_attr_name_of(self):
        t = torch.zeros(1)
        assert oom._attr_name_of(Holder(t), t) == "weight"
        assert oom._attr_name_of(Holder(None), t) is None
        assert oom._attr_name_of(5, t) is None

    def test_describe_frame_names_holding_locals(self):
        target = torch.zeros(1)
        held_in_list = [target]  # noqa: F841 - inspected via frame locals
        desc = oom._describe_frame(sys._getframe(), target)
        assert "test_describe_frame_names_holding_locals" in desc
        assert "target" in desc and "held_in_list" in desc

    def test_describe_frame_falls_back(self):
        assert oom._describe_frame(object(), None) == "frame"

    def test_describe_referrers_finds_attribute_and_list_owner(self):
        target = torch.zeros(3)
        holder = Holder(target)
        container = [target]
        owner = types.SimpleNamespace(bucket=container)
        out = oom._describe_referrers(target, skip_ids=set(), limit=20)
        joined = "\n".join(out)
        assert "Holder.weight" in joined or "dict[weight]" in joined
        assert "list(len=1)" in joined
        assert "bucket" in joined
        assert holder.weight is target and owner.bucket is container

    def test_describe_referrers_respects_limit(self):
        target = torch.zeros(1)
        holders = [Holder(target) for _ in range(10)]
        assert len(oom._describe_referrers(target, skip_ids=set(), limit=3)) == 3
        assert len(holders) == 10

    def test_describe_referrers_limit_inside_container_unwrap(self):
        target = torch.zeros(1)
        container = [target]
        owners = [types.SimpleNamespace(bucket=container) for _ in range(5)]
        out = oom._describe_referrers(target, skip_ids=set(), limit=2)
        assert len(out) == 2 and out[1].startswith("  <- ")
        assert len(owners) == 5

    def test_describe_referrers_frame_holder(self):
        target = torch.zeros(1)
        frame = sys._getframe()
        with patch("gc.get_referrers", return_value=[frame, frame]):
            out = oom._describe_referrers(target, skip_ids=set(), limit=1)
        assert len(out) == 1
        assert "test_describe_referrers_frame_holder" in out[0] and "target" in out[0]


class TestCensusDump:
    def test_dump_reports_inventory_and_holders(self):
        big = _fake_tensor(shape=(1024, 1024))
        holder = Holder(big)
        fake_gc = types.SimpleNamespace(get_objects=lambda: [big, _fake_tensor(device="xpu:0", shape=(2,)), holder])
        with patch.object(oom, "logger") as log, patch.object(torch.cuda, "device_count", return_value=0):
            oom._dump_census(fake_gc, context="unit-test")
        text = _logged(log)
        assert "context: unit-test" in text
        assert "cuda:0 live tensors" in text and "xpu:0 live tensors" in text
        assert "[1024, 1024] x1" in text
        assert "held by" in text

    def test_dump_reports_skipped_tensors(self):
        broken = _fake_tensor()
        broken.numel.side_effect = RuntimeError("unreadable")
        fake_gc = types.SimpleNamespace(get_objects=lambda: [broken])
        with patch.object(oom, "logger") as log, patch.object(torch.cuda, "device_count", return_value=0):
            oom._dump_census(fake_gc)
        assert "skipped 1 unreadable" in _logged(log)

    def test_public_dump_delegates(self):
        with patch.object(oom, "_dump_census") as dump:
            oom.dump_oom_tensor_census_("ctx")
        dump.assert_called_once()
        assert dump.call_args.args[1] == "ctx"


class TestOomDetectionAndHooks:
    @pytest.mark.parametrize(
        "exc, expected",
        [
            (torch.OutOfMemoryError("cuda oom"), True),
            (RuntimeError("HPU Out Of Memory while allocating"), True),
            (RuntimeError("shape mismatch"), False),
            (ValueError("out of memory"), False),
        ],
    )
    def test_is_oom(self, exc, expected):
        assert oom._is_oom(exc) is expected

    def test_context_manager_dumps_on_oom_and_reraises(self):
        with (
            patch.object(oom, "dump_oom_tensor_census_") as dump,
            pytest.raises(torch.OutOfMemoryError),
            oom.oom_census("ctx"),
        ):
            raise torch.OutOfMemoryError("oom")
        dump.assert_called_once_with("ctx")

    def test_context_manager_ignores_other_errors(self):
        with patch.object(oom, "dump_oom_tensor_census_") as dump:
            with pytest.raises(KeyError), oom.oom_census("ctx"):
                raise KeyError("x")
            with oom.oom_census("ctx"):
                pass
        dump.assert_not_called()

    def test_install_hooks_is_idempotent_and_chains(self, monkeypatch):
        prior_sys, prior_thread = MagicMock(), MagicMock()
        monkeypatch.setattr(sys, "excepthook", prior_sys)
        monkeypatch.setattr(threading, "excepthook", prior_thread)
        monkeypatch.setattr(oom, "_OOM_HOOK_INSTALLED", False)

        assert oom.install_oom_census_hook() is True
        assert oom.install_oom_census_hook() is False

        err = torch.OutOfMemoryError("oom")
        args = types.SimpleNamespace(exc_value=err, thread=types.SimpleNamespace(name="worker"))
        no_thread_args = types.SimpleNamespace(exc_value=err, thread=None)
        with patch.object(oom, "dump_oom_tensor_census_") as dump:
            sys.excepthook(torch.OutOfMemoryError, err, None)
            sys.excepthook(ValueError, ValueError("x"), None)
            threading.excepthook(args)
            threading.excepthook(no_thread_args)

        assert [c.args[0] for c in dump.call_args_list] == [
            "uncaught",
            "uncaught (thread worker)",
            "uncaught (thread ?)",
        ]
        assert prior_sys.call_count == 2
        assert prior_thread.call_count == 2
