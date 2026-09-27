# coding=utf-8
# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for the one-deep background writer used by the shard write/flush."""

import time
import unittest

from auto_round.compressors.orchestrator import _OneDeepWriter


class TestOneDeepWriter(unittest.TestCase):
    def test_dispatch_runs_and_join_blocks(self):
        w = _OneDeepWriter()
        done = []
        w.dispatch(lambda: done.append(1))
        w.join()
        self.assertEqual(done, [1])
        self.assertIsNone(w._t)

    def test_one_deep_previous_joined_before_next_starts(self):
        w = _OneDeepWriter()
        order = []

        def slow():
            order.append("slow-start")
            time.sleep(0.2)
            order.append("slow-end")

        def fast():
            order.append("fast-start")

        w.dispatch(slow)
        w.dispatch(fast)  # must join slow first
        w.join()
        self.assertEqual(order[0], "slow-start")
        self.assertEqual(order[1], "slow-end")
        self.assertEqual(order[2], "fast-start")

    def test_exception_reraised_on_join(self):
        w = _OneDeepWriter()

        def boom():
            raise RuntimeError("writer failed")

        w.dispatch(boom)
        with self.assertRaisesRegex(RuntimeError, "writer failed"):
            w.join()
        # subsequent dispatches still work (state cleared)
        ran = []
        w.dispatch(lambda: ran.append(1))
        w.join()
        self.assertEqual(ran, [1])

    def test_join_without_dispatch_is_noop(self):
        w = _OneDeepWriter()
        w.join()  # must not raise


class TestBackgroundResumeMark(unittest.TestCase):
    """The one-deep discipline keeps mark_block_done's order-assert valid."""

    def _fake_state(self):
        import types

        state = types.SimpleNamespace(completed_blocks=[], calls=[])

        def mark_block_done(block_name, q_input, input_ids):
            expected = ["b0", "b1", "b2"][len(state.completed_blocks)]
            assert block_name == expected, f"out of order: expected {expected}, got {block_name}"
            state.calls.append((block_name, q_input, input_ids))
            state.completed_blocks.append(block_name)

        state.mark_block_done = mark_block_done
        return state

    def test_marks_run_in_order_through_the_writer(self):
        from auto_round.compressors.orchestrator import _OneDeepWriter

        state = self._fake_state()
        w = _OneDeepWriter()
        w.dispatch(lambda: state.mark_block_done("b0", None, "i0"))
        w.dispatch(lambda: state.mark_block_done("b1", None, "i1"))  # joins b0 first
        w.dispatch(lambda: state.mark_block_done("b2", None, "i2"))
        w.join()
        self.assertEqual(state.completed_blocks, ["b0", "b1", "b2"])
        self.assertEqual([c[2] for c in state.calls], ["i0", "i1", "i2"])

    def test_worker_exception_surfaces_at_next_dispatch(self):
        from auto_round.compressors.orchestrator import _OneDeepWriter

        state = self._fake_state()
        w = _OneDeepWriter()
        w.dispatch(lambda: state.mark_block_done("b0", None, "i0"))

        def boom():
            raise RuntimeError("resume save failed")

        w.dispatch(boom)  # joins b0 (ok), then starts boom
        with self.assertRaisesRegex(RuntimeError, "resume save failed"):
            w.join()


class TestSnapshotPoolRefs(unittest.TestCase):
    def test_skeleton_frozen_tensors_by_reference(self):
        import torch

        from auto_round.utils.resume import snapshot_pool_refs

        t = torch.zeros(4)
        pool = {"hidden_states": [t, t], "aux": ({"inner": t},)}
        snap = snapshot_pool_refs(pool)
        self.assertIsNot(snap, pool)
        self.assertIsNot(snap["hidden_states"], pool["hidden_states"])
        self.assertIs(snap["hidden_states"][0], t)  # tensors passed by ref
        # mutating the original container afterwards must not affect the snap
        pool["hidden_states"].append(torch.ones(4))
        self.assertEqual(len(snap["hidden_states"]), 2)
