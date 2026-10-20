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
"""Tests for the batched-search dispatch helpers (grouping, per-device execution)."""

import threading
import unittest
from unittest import mock

import torch

import auto_round.algorithms.quantization.search_dispatch as search_dispatch
from auto_round.algorithms.quantization.search_dispatch import group_items_by_device, run_items_by_device


def _linear(out=8, inn=8):
    m = torch.nn.Linear(inn, out, bias=False)
    m.bits = 4  # pass check_to_quantized
    return m


class TestGroupItemsByDevice(unittest.TestCase):
    def test_groups_by_device_string(self):
        items = [("a", "cuda:0"), ("b", "cuda:1"), ("c", "cuda:0")]
        groups = group_items_by_device(items, device_of=lambda item: item[1])
        self.assertEqual([k for k in groups], ["cuda:0", "cuda:1"])
        self.assertEqual([(i, it) for i, it in groups["cuda:0"]], [(0, ("a", "cuda:0")), (2, ("c", "cuda:0"))])
        self.assertEqual([(i, it) for i, it in groups["cuda:1"]], [(1, ("b", "cuda:1"))])

    def test_empty_items(self):
        self.assertEqual(list(group_items_by_device([], device_of=lambda it: "cpu").items()), [])

    def test_missing_device_falls_back_to_none_bucket(self):
        items = [("a", None), ("b", "cuda:0")]
        groups = group_items_by_device(items, device_of=lambda item: item[1], none_key="shared")
        self.assertIn("shared", groups)
        self.assertIn("cuda:0", groups)


class TestRunItemsByDevice(unittest.TestCase):
    def test_runs_all_items_and_records_devices(self):
        items = [f"m{i}" for i in range(6)]
        # 3 fake devices, 2 items each
        device_of = lambda it: f"cuda:{int(it[1:]) % 3}"  # noqa: E731
        groups = group_items_by_device(items, device_of=device_of)
        seen = []
        lock = threading.Lock()

        def fn(_idx, item):
            with lock:
                seen.append((item, threading.current_thread().name))

        run_items_by_device(groups, fn)
        self.assertEqual(sorted(s[0] for s in seen), items)
        # items on different devices ran on different threads
        threads = {s[0]: s[1] for s in seen}
        self.assertNotEqual(threads["m0"], threads["m1"])

    def test_exception_propagates_fail_visible(self):
        items = ["a", "b"]
        groups = group_items_by_device(items, device_of=lambda it: "cpu")

        def fn(_idx, item):
            if item == "a":
                raise RuntimeError("boom")

        with self.assertRaisesRegex(RuntimeError, "boom"):
            run_items_by_device(groups, fn)

    def test_single_group_runs_inline(self):
        # one device -> no thread spawn; fn runs on the calling thread
        items = ["a", "b"]
        groups = group_items_by_device(items, device_of=lambda it: "cpu")
        caller = threading.current_thread().name
        seen = []
        run_items_by_device(groups, lambda _idx, it: seen.append(threading.current_thread().name))
        self.assertEqual(seen, [caller, caller])

    def test_cuda_device_ctx_applied(self):
        recorded = []
        real_ctx = torch.cuda.device

        class FakeCtx:
            def __init__(self, dev):
                recorded.append(str(dev))

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        items = ["a", "b"]
        groups = group_items_by_device(items, device_of=lambda it: "cuda:0" if it == "a" else "cuda:1")
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "device", FakeCtx),
        ):
            run_items_by_device(groups, lambda _idx, it: None, use_cuda_ctx=True)
        self.assertEqual(sorted(recorded), ["cuda:0", "cuda:1"])  # one ctx per device group


class TestRtnSearchShard(unittest.TestCase):
    def test_optimized_rtn_stages_into_batched_driver(self):
        from auto_round.algorithms.quantization.rtn.quantizer import OptimizedRTNQuantizer

        q = object.__new__(OptimizedRTNQuantizer)
        staged_seen = []
        # 'model' is a read-only property; the patched driver ignores it anyway

        def fake_via_rtn(self_, m, disable_opt_rtn=None, defer_search=False):
            assert defer_search is True
            staged_seen.append((getattr(m, "global_name", None), m))
            return "wrapper"

        block = torch.nn.Module()
        for i in range(3):
            m = _linear()
            m.name = f"m{i}"
            m.global_name = f"m{i}"
            setattr(block, f"m{i}", m)

        with (
            mock.patch.object(type(q), "_quantize_layer_via_rtn", fake_via_rtn),
            mock.patch.object(type(q), "model", torch.nn.Module()),
            mock.patch(
                "auto_round.algorithms.quantization.rtn.batched_search.run_batched_rtn_search",
                side_effect=lambda model, staged: staged_seen.extend([]) or [],
            ) as drv,
        ):
            q.quantize_block(block, None, None, None, None, None)
        self.assertEqual(drv.call_count, 1)
        self.assertEqual([n for n, _m in staged_seen], ["m0", "m1", "m2"])

    def test_single_device_runs_serial(self):
        from auto_round.algorithms.quantization.rtn.quantizer import OptimizedRTNQuantizer

        q = object.__new__(OptimizedRTNQuantizer)
        caller = threading.current_thread().name
        calls = []
        q.quantize_layer_outside_block = lambda m: calls.append(threading.current_thread().name)

        block = torch.nn.Module()
        m = _linear()
        m.name = "m0"
        m.global_name = "m0"
        m.tuning_device = "cpu"
        block.m0 = m
        # strict serial baseline = batching AND multigpu spreading both off;
        # batching-off alone now spreads per-module searches (see the
        # quadrant test below)
        with (
            mock.patch.object(search_dispatch, "batched_search_disabled", return_value=True),
            mock.patch.object(search_dispatch, "cross_device_search_disabled", return_value=True),
        ):
            q.quantize_block(block, None, None, None, None, None)
        self.assertEqual(calls, [caller])

    def test_batching_off_spreads_per_module_when_multigpu_on(self):
        """AR_DISABLE_BATCHED_SEARCH alone no longer pins everything to the
        home device: the searches stay PER-MODULE (no stacked row sharing)
        but route through the staged executor so idle devices can borrow
        them."""
        from auto_round.algorithms.quantization.rtn import batched_search
        from auto_round.algorithms.quantization.rtn.quantizer import OptimizedRTNQuantizer

        q = object.__new__(OptimizedRTNQuantizer)
        block = torch.nn.Module()
        m = _linear()
        m.name = "m0"
        m.global_name = "m0"
        m.tuning_device = "cpu"
        block.m0 = m
        seen = {}

        class _W:  # minimal deferred wrapper: the spy finishes it

            def _run_deferred_search_now(self):
                pass

        q._quantize_layer_via_rtn = lambda layer, defer_search=False: _W()
        seen_run = {}

        def _spy_run(model, staged, max_batch=None):
            seen_run["max_batch"] = max_batch
            seen_run["n"] = len(staged)
            for name, w in staged:  # finish the deferred wrappers
                w._run_deferred_search_now()
            return True

        with (
            mock.patch.object(batched_search, "run_batched_rtn_search", side_effect=_spy_run),
            mock.patch.object(search_dispatch, "batched_search_disabled", return_value=True),
            mock.patch.object(search_dispatch, "cross_device_search_disabled", return_value=False),
            mock.patch.object(type(q), "model", new_callable=mock.PropertyMock),
        ):
            q.quantize_block(block, None, None, None, None, None)
        self.assertEqual(seen_run.get("max_batch"), 1)  # singleton chunks: zero stacking
        self.assertGreaterEqual(seen_run.get("n", 0), 1)


class TestOomCensus(unittest.TestCase):
    def test_context_manager_plugs_anywhere_and_reraises(self):
        from auto_round.utils.oom import oom_census

        with (
            mock.patch("auto_round.utils.oom.dump_oom_tensor_census_") as census,
            self.assertRaisesRegex(torch.OutOfMemoryError, "boom"),
            oom_census("surgical frame"),
        ):
            raise torch.OutOfMemoryError("boom")
        census.assert_called_once_with("surgical frame")

    def test_context_manager_passes_non_oom_through_silently(self):
        from auto_round.utils.oom import oom_census

        with (
            mock.patch("auto_round.utils.oom.dump_oom_tensor_census_") as census,
            self.assertRaisesRegex(ValueError, "other"),
            oom_census("frame"),
        ):
            raise ValueError("other")
        census.assert_not_called()

    def test_global_hook_fires_on_uncaught_oom(self):
        import sys
        import threading

        import auto_round.utils.oom as oom_mod

        # The latch and the excepthooks are process-global: an earlier test in
        # the same process (e.g. the model-free pipeline installs the hook for
        # real) must not decide this test's outcome. Snapshot the state, start
        # from pristine hooks, and restore exactly what was there before.
        self.addCleanup(setattr, sys, "excepthook", sys.excepthook)
        self.addCleanup(setattr, threading, "excepthook", threading.excepthook)
        self.addCleanup(setattr, oom_mod, "_OOM_HOOK_INSTALLED", oom_mod._OOM_HOOK_INSTALLED)
        oom_mod._OOM_HOOK_INSTALLED = False
        sys.excepthook = sys.__excepthook__
        threading.excepthook = threading.__excepthook__

        installed = oom_mod.install_oom_census_hook()
        self.assertFalse(oom_mod.install_oom_census_hook())  # idempotent
        with mock.patch.object(oom_mod, "dump_oom_tensor_census_") as census:
            try:
                raise torch.OutOfMemoryError("boom")
            except torch.OutOfMemoryError:
                sys.excepthook(*sys.exc_info())
        self.assertTrue(installed)
        self.assertEqual(census.call_count, 1)

    def test_symbolic_dims_and_bad_tensors_do_not_kill_census(self):
        import auto_round.utils.oom as oom_mod

        class UnhashableDim:
            def __int__(self):
                raise TypeError("cannot convert SymInt to int")

            def __hash__(self):
                raise TypeError("unhashable")

        class SymishTensor(torch.Tensor):
            pass

        t = torch.randn(4, 4)
        with mock.patch.object(type(t), "shape", new_callable=lambda *a: property(lambda self: (UnhashableDim(),))):
            groups = oom_mod._group_tensors_by_shape([t])
        self.assertEqual(groups, ([], 0))  # cpu tensor excluded; symbolic shape did not raise

        class ExplodingTensor:
            pass

        groups = oom_mod._group_tensors_by_shape([ExplodingTensor(), torch.randn(4, 4)])
        self.assertEqual(groups, ([], 0))

    def test_cpu_tensors_excluded_from_census_groups(self):
        # cpu-only box: grouping must not count cpu (or meta) tensors
        import auto_round.utils.oom as oom_mod

        t = torch.randn(4, 4)
        groups = oom_mod._group_tensors_by_shape([t])
        self.assertEqual(groups, ([], 0))

    def test_message_based_oom_triggers_census(self):
        # HPU-class OOMs surface as RuntimeError("... out of memory ...")
        from auto_round.utils.oom import oom_census

        with (
            mock.patch("auto_round.utils.oom.dump_oom_tensor_census_") as census,
            self.assertRaisesRegex(RuntimeError, "Out of Memory in HPU"),
            oom_census("hpu frame"),
        ):
            raise RuntimeError("Out of Memory in HPU allocator")
        census.assert_called_once_with("hpu frame")

    def test_plain_runtime_error_does_not_trigger_census(self):
        from auto_round.utils.oom import oom_census

        with (
            mock.patch("auto_round.utils.oom.dump_oom_tensor_census_") as census,
            self.assertRaisesRegex(RuntimeError, "shape mismatch"),
            oom_census("frame"),
        ):
            raise RuntimeError("shape mismatch")
        census.assert_not_called()

    def test_reexport_from_search_dispatch(self):
        import auto_round.algorithms.quantization.search_dispatch as dispatch_mod
        import auto_round.utils.oom as oom_mod

        self.assertIs(dispatch_mod.dump_oom_tensor_census_, oom_mod.dump_oom_tensor_census_)

    def test_census_never_masks_and_never_raises(self):
        import auto_round.algorithms.quantization.search_dispatch as dispatch_mod

        # CPU-only box: the census must swallow its own failures and return cleanly
        dispatch_mod.dump_oom_tensor_census_("test")
        with mock.patch("auto_round.utils.oom._group_tensors_by_shape", side_effect=RuntimeError("boom")):
            dispatch_mod.dump_oom_tensor_census_("test")  # diagnostics failure swallowed


class TestResolveLossDevice(unittest.TestCase):
    """The tune loop stages pred/ref AND the loss mask on one concrete device.

    Regression (hy3 4-GPU iters>0): card-0-risk relocation moved the staging
    device off the primary while the masked loss still targeted the primary,
    crashing the mask multiply across devices; an explicit None from the
    placement pass silently skipped the staging move entirely.
    """

    def test_none_coalesces_to_primary(self):
        from auto_round.algorithms.quantization.sign_round.quantizer import resolve_loss_device

        self.assertEqual(resolve_loss_device(None, "cuda:0"), "cuda:0")
        self.assertEqual(resolve_loss_device("cuda:1", "cuda:0"), "cuda:1")

    def test_masked_loss_stages_mask_on_the_passed_device(self):
        # the masked loss contract: the mask is moved onto the ``device`` the
        # caller staged pred/ref on, so all three multiply on one device
        from types import SimpleNamespace

        import torch

        import auto_round.algorithms.quantization.sign_round.quantizer as sr_mod
        from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer

        q = object.__new__(SignRoundQuantizer)
        pred = torch.randn(2, 3)
        ref = torch.randn(2, 3)
        mask = [torch.ones(1, dtype=torch.long), torch.ones(1, dtype=torch.long)]
        try:
            sr_mod.SignRoundQuantizer.model_context = property(lambda self: SimpleNamespace(amp=False))
            loss = q._get_loss(pred, ref, [0, 1], torch.nn.MSELoss(), "cpu", valid_token_mask=mask)
        finally:
            if "model_context" in sr_mod.SignRoundQuantizer.__dict__:
                delattr(sr_mod.SignRoundQuantizer, "model_context")
        self.assertTrue(torch.isfinite(loss))


class TestOrchestratorPhaseSpans(unittest.TestCase):
    """Source contract: every span the orchestrator summary stops is opened.

    The tick chain opens spans at phase boundaries; the gated summary stops the
    last one before reading totals. A stopped-but-never-opened key only blows up
    when AR_PERF_COUNTERS is enabled on a real run, so pin the pairing here.
    """

    def test_every_stopped_span_is_opened(self):
        import ast
        import inspect

        from auto_round.compressors import orchestrator as orch

        tree = ast.parse(inspect.getsource(orch))
        opened, stopped = set(), set()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "_perf"):
                continue
            key = node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) else None
            if key is None:
                continue
            if node.func.attr in ("tick", "start"):
                opened.add(key)
            elif node.func.attr == "stop":
                stopped.add(key)
        self.assertTrue(stopped, "no _perf.stop() calls found in the orchestrator")
        self.assertEqual(
            stopped - opened,
            set(),
            f"spans stopped but never opened: {sorted(stopped - opened)}",
        )


class TestTunePhaseLine(unittest.TestCase):
    def _fmt(self):
        from auto_round.algorithms.quantization.sign_round.quantizer import _tune_phase_line

        return _tune_phase_line

    def test_ar_perf_recorder_follows_env(self):
        import os
        import time

        from auto_round.utils.perf_counters import _PerfCounters

        with mock.patch.dict(os.environ, {"AR_PERF_COUNTERS": "1"}):
            perf = _PerfCounters()
        self.assertTrue(perf.enabled)
        perf.start("s_fwd")
        time.sleep(0.001)
        perf.stop("s_fwd")
        perf.start("s_fwd")
        perf.stop("s_fwd")
        self.assertGreater(perf.get("s_fwd"), 0.0)
        self.assertEqual(set(perf.totals), {"s_fwd"})
        perf.set("rest", perf.get("s_fwd") - perf.get("sampler"))
        self.assertAlmostEqual(perf.get("rest"), perf.get("s_fwd"))
        perf.start("s_loss")
        self.assertEqual(perf.get("s_loss"), 0.0)  # open span reads as zero until stopped

        # tick chain: closing each span opens the next
        with mock.patch.dict(os.environ, {"AR_PERF_COUNTERS": "1"}):
            perf = _PerfCounters()
        for k in ("attach", "compress", "swap", "sweep", "write"):
            perf.tick(k)
        self.assertEqual(set(perf.totals), {"attach", "compress", "swap", "sweep"})
        # observe: count + total + max
        perf.observe("quant_mod", 0.5)
        perf.observe("quant_mod", 1.5)
        self.assertEqual(perf.counts["quant_mod"], 2)
        self.assertAlmostEqual(perf.totals["quant_mod"], 2.0)
        self.assertAlmostEqual(perf.maxima["quant_mod"], 1.5)

        with mock.patch.dict(os.environ, {"AR_PERF_COUNTERS": "0"}):
            perf = _PerfCounters()
        self.assertFalse(perf.enabled)
        perf.start("s_fwd")
        perf.stop("s_fwd")
        perf.tick("swap")
        perf.observe("s_fwd", 1.0)
        perf.set("s_fwd", 1.0)
        self.assertEqual(perf.get("s_fwd"), 0.0)

    def test_base_line_matches_ddp_formatter(self):
        line = self._fmt()({"wrap": 1.5, "prepare": 0.0, "loop": 9.5, "tail": 0.5}, 10)
        self.assertEqual(line, "[perf] tune phases (iters=10): wrap=1.50s prepare=0.00s loop=9.50s tail=0.50s")

    def test_loop_split_appended_with_serial(self):
        line = self._fmt()(
            {
                "wrap": 1.0,
                "prepare": 0.0,
                "loop": 9.5,
                "tail": 0.5,
                "lp_sampler": 0.0,
                "lp_snap": 0.06,
                "lp_step": 0.02,
                "lp_rest": 0.82,
                "lp_serial": 8.6,
            },
            10,
        )
        self.assertIn("(loop: sampler=0.00s snap=0.06s step=0.02s rest=0.82s serial: fwd+loss+bwd=8.60s)", line)

    def test_loop_split_omitted_without_keys(self):
        line = self._fmt()({"wrap": 1.0, "prepare": 0.0, "loop": 9.5, "tail": 0.5}, 10)
        self.assertNotIn("loop:", line.split("loop=9.50s")[-1])

    def test_serial_omitted_when_zero(self):
        line = self._fmt()(
            {
                "wrap": 1.0,
                "prepare": 0.0,
                "loop": 9.5,
                "tail": 0.5,
                "lp_sampler": 0.0,
                "lp_snap": 0.0,
                "lp_step": 0.0,
                "lp_rest": 9.5,
                "lp_serial": 0.0,
            },
            10,
        )
        self.assertNotIn("serial:", line)


class TestSnapshotRouting(unittest.TestCase):
    """snapshot_best_params: non-CPU cache device -> per-param device copies."""

    def _fake_block(self):
        class W(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.orig_layer = torch.nn.Linear(8, 8, bias=False)
                self.params = {"value": torch.nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4))}

        block = torch.nn.Module()
        block.w0 = W()
        block.w1 = W()
        return block

    def test_cpu_cache_device_uses_host_path(self):
        from auto_round.compressors.utils import snapshot_best_params

        block = self._fake_block()
        out = snapshot_best_params(block, torch.device("cpu"))
        for n in ("w0", "w1"):
            self.assertTrue(out[n]["value"].data_ptr() != block.get_submodule(n).params["value"].data_ptr())
            self.assertTrue(torch.equal(out[n]["value"], block.get_submodule(n).params["value"]))
            self.assertEqual(out[n]["value"].device.type, "cpu")

    def test_non_cpu_cache_device_keeps_params_on_own_device(self):
        from auto_round.compressors.utils import snapshot_best_params

        block = self._fake_block()
        # router keys on the cache-device label; params live on cpu in this test
        out = snapshot_best_params(block, "cuda:0")
        for n in ("w0", "w1"):
            src = block.get_submodule(n).params["value"]
            self.assertEqual(out[n]["value"].device, src.device)  # own device, not the cache device
            self.assertTrue(torch.equal(out[n]["value"], src))

    def test_local_snapshot_storage_is_distinct(self):
        from auto_round.compressors.utils import snapshot_best_params

        block = self._fake_block()
        out = snapshot_best_params(block, "cuda:0")
        src = block.w0.params["value"]
        with torch.no_grad():
            src.add_(1.0)
            self.assertFalse(torch.equal(out["w0"]["value"], src))  # snapshot isolated from mutation
            src.sub_(1.0)

    def test_local_copy_failure_falls_back_to_host(self):
        from auto_round.compressors.utils import collect_best_params_local

        block = self._fake_block()

        class FlakyData:
            # local path passes a torch.device object; the host fallback passes the "cpu" string
            device = torch.device("cpu")

            def to(self, device=None, copy=False):
                if isinstance(device, torch.device):
                    raise RuntimeError("simulated local-copy failure")
                return torch.zeros(2, 2)

        class Pseudo:
            data = FlakyData()

        block.w0.params["flaky"] = Pseudo()
        out = collect_best_params_local(block)
        self.assertIn("w0", out)  # host fallback still produced a snapshot
        self.assertIn("flaky", out["w0"])

    def test_invalid_cache_device_label_keeps_historical_behavior(self):
        from auto_round.compressors.utils import snapshot_best_params

        block = self._fake_block()
        with self.assertRaisesRegex(RuntimeError, "Invalid device"):
            snapshot_best_params(block, "not-a-device")  # same failure as the historical path


class TestNonCudaFamilies(unittest.TestCase):
    """Non-cuda device coverage: graceful degradation, never a crash."""

    def test_worker_ctx_unknown_and_cpu_keys_run_bare(self):
        from auto_round.algorithms.quantization.search_dispatch import _device_worker_ctx, _null_ctx

        for key in ["cpu", "bogus-device", "cuda:99"]:  # cuda:99 -> unavailable cuda -> bare
            ctx = _device_worker_ctx(key)
            self.assertIsNotNone(ctx)

    def test_run_items_by_device_non_cuda_groups(self):
        from auto_round.algorithms.quantization.search_dispatch import group_items_by_device, run_items_by_device

        ran = []
        groups = group_items_by_device(["cpu", "cpu", "meta"], lambda item: item)
        run_items_by_device(groups, lambda idx, item: ran.append((idx, item)))
        self.assertEqual(sorted(ran), [(0, "cpu"), (1, "cpu"), (2, "meta")])

    def test_grouped_backstop_degrades_off_cuda(self):
        import torch

        from auto_round.modeling.fused_moe.grouped_experts import _native_grouped_mm_usable

        x = torch.randn(4, 8)
        w = torch.randn(2, 8, 8)
        offs = torch.tensor([2, 4], dtype=torch.int32)
        # cpu tensors must never gamble on the native path (would raise
        # mid-forward on families F.grouped_mm does not support); the backstop
        # is only reached when the transformers helper is unavailable
        import auto_round.modeling.fused_moe.grouped_experts as ge

        with mock.patch.object(ge, "_transformers_can_use_grouped_mm", None):
            self.assertFalse(_native_grouped_mm_usable(x, w, offs))


class TestWrapSearchWorkerBuckets(unittest.TestCase):
    """Planner contracts: exclusive per-worker buckets with locality-preferring balance."""

    @staticmethod
    def _fake(seed, shape=(6, 128)):
        import types as _types

        from auto_round.data_type.utils import resolve_optimized_init_scale_fn

        w = torch.randn(*shape, generator=torch.Generator().manual_seed(seed))
        fn = resolve_optimized_init_scale_fn("int", 1e-5)

        class _W:
            supports_batched_search = True
            _deferred_search_inputs = (w, "int", 4, None, 1e-5, fn)
            init_scale = None
            orig_layer = _types.SimpleNamespace(group_size=128)
            name = f"m{seed}"

            def _run_deferred_search_now(self):
                *_, f = self._deferred_search_inputs
                self.init_scale = f(self._deferred_search_inputs[0], 4, None)

            def finalize_batched_search(self, res):
                self.init_scale = res
                self._deferred_search_inputs = None

        return _W()

    def test_concentrated_homes_spread_across_viable_workers(self):
        # chunks whose weights share one home must spread across the viable
        # workers (balanced), not pile onto the home device
        wrappers = [self._fake(i) for i in range(8)]
        captured = {}

        def fake_run(keyed, fn):
            captured["buckets"] = {k: [c for _idx, it in v for c in it[1]] for k, v in keyed.items()}

        with (
            mock.patch.object(search_dispatch, "pick_search_worker_devices", return_value=["cpu", "cuda:9"]),
            mock.patch.object(search_dispatch, "probe_usable_bytes", return_value=2**40),
            mock.patch.object(search_dispatch, "run_items_by_device", side_effect=fake_run),
        ):
            search_dispatch.run_batched_wrap_search(wrappers, max_batch=1)
        buckets = captured["buckets"]
        self.assertEqual(set(buckets), {"cpu", "cuda:9"})
        counts = [len(items) for items in buckets.values()]
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_homes_without_other_workers_run_inline(self):
        # viable = home only: a single bucket executes inline and the real
        # stacked search runs on the weights' own device
        wrappers = [self._fake(i) for i in range(3)]
        with mock.patch.object(search_dispatch, "pick_search_worker_devices", return_value=["cpu"]):
            self.assertTrue(search_dispatch.run_batched_wrap_search(wrappers, max_batch=2))
        for wr in wrappers:
            self.assertIsNotNone(wr.init_scale)

    def test_rebalance_uses_least_loaded_first(self):
        # three chunks and one extra worker: the split is 2/1, not 3/0
        wrappers = [self._fake(i) for i in range(3)]
        captured = {}

        def fake_run(keyed, fn):
            captured["buckets"] = {k: [c for _idx, it in v for c in it[1]] for k, v in keyed.items()}

        with (
            mock.patch.object(search_dispatch, "pick_search_worker_devices", return_value=["cpu", "cuda:9"]),
            mock.patch.object(search_dispatch, "probe_usable_bytes", return_value=2**40),
            mock.patch.object(search_dispatch, "run_items_by_device", side_effect=fake_run),
        ):
            search_dispatch.run_batched_wrap_search(wrappers, max_batch=1)
        counts = sorted(len(v) for v in captured["buckets"].values())
        self.assertEqual(counts, [1, 2])


class TestSearchTransientBytes(unittest.TestCase):
    def test_plain_grid_search_counts_copy_chain_only(self):
        from auto_round.algorithms.quantization.search_dispatch import search_transient_bytes
        from auto_round.data_type.int import search_scales

        self.assertEqual(search_transient_bytes(search_scales, 4000), 4000 * 4 * 4)

    def test_neuqi_search_adds_internal_budget_once_per_call(self):
        # the sweep budget is per-call: doubling the stacked modules doubles
        # the copy chain, not the internal budget
        from auto_round.algorithms.quantization.search_dispatch import search_transient_bytes
        from auto_round.data_type.neuqi import _MAX_TMP_ELEMS, neuqi_anchor_margins

        one = search_transient_bytes(neuqi_anchor_margins, 1000)
        two = search_transient_bytes(neuqi_anchor_margins, 2000)
        self.assertEqual(one, 1000 * 4 * 4 + _MAX_TMP_ELEMS * 4 * 8)
        self.assertEqual(two - one, 1000 * 4 * 4)  # chain scales, budget does not

    def test_budget_is_override_only(self):
        import os

        from auto_round.algorithms.quantization import search_dispatch

        saved = os.environ.pop("AR_SEARCH_BATCH_GB", None)
        try:
            self.assertIsNone(search_dispatch._wrap_batch_max_elems())
        finally:
            if saved is not None:
                os.environ["AR_SEARCH_BATCH_GB"] = saved


class TestPlanCapacityChunks(unittest.TestCase):
    def _plan(self, entries, probes, picks=None):
        calls = {}

        def fake_probe(dev):
            calls.setdefault(dev, 0)
            calls[dev] += 1
            return probes.get(dev)

        with mock.patch.object(search_dispatch, "probe_usable_bytes", side_effect=fake_probe):
            if picks is None:
                picks = lambda ws, home_device=None: [home_device]
            with mock.patch.object(search_dispatch, "pick_search_worker_devices", side_effect=picks):
                return search_dispatch._plan_capacity_chunks(entries), calls

    def test_sizes_chunk_to_the_executing_device_capacity(self):
        per, mods = 1000, list(range(10))
        plans, _ = self._plan([("cuda:0", per, 0, 1024, mods)], {"cuda:0": 2 * 10 * per})
        ((worker, home, chunk),) = plans
        self.assertEqual((worker, home, len(chunk)), ("cuda:0", "cuda:0", 10))  # usable//2 // per

    def test_serial_chunks_do_not_drain_capacity(self):
        # chunks on a device run one at a time, so each is sized to the
        # device's own headroom instead of draining a cumulative budget
        per, mods = 1000, list(range(30))
        plans, _ = self._plan([("cuda:0", per, 0, 10, mods)], {"cuda:0": 2 * 10 * per})
        sizes = [len(c) for _w, _h, c in plans]
        self.assertEqual(sizes, [10, 10, 10])

    def test_work_spreads_round_robin_home_first(self):
        per = 1000
        mods = list(range(2))
        probes = {"cuda:0": 2 * 3 * per, "cuda:1": 2 * 2 * per}  # cuda:0 roomier than home
        picks = lambda ws, home_device=None: [home_device, "cuda:0"]
        plans, _ = self._plan([("cuda:1", per, 0, 1, mods)], probes, picks)
        workers = [w for w, _h, _c in plans]
        # equal counts: home takes its share first, the overflow goes on
        self.assertEqual(workers, ["cuda:1", "cuda:0"])

    def test_tight_home_chews_small_batches_and_shares_with_roomy_device(self):
        per = 1000
        mods = list(range(6))
        probes = {"cuda:1": 2 * 1 * per, "cuda:7": 2 * 4 * per}
        picks = lambda ws, home_device=None: [home_device, "cuda:7"]
        plans, _ = self._plan([("cuda:1", per, 0, 1024, mods)], probes, picks)
        got = [(w, len(c)) for w, _h, c in plans]
        # round-robin: home takes 3 (one per batch, its cap is 1), cuda:7
        # takes 3 in one batch (its cap is 4)
        self.assertEqual(got, [("cuda:1", 1), ("cuda:1", 1), ("cuda:1", 1), ("cuda:7", 3)])

    def test_sweep_budget_sized_per_chunk(self):
        per, sweep, mods = 1000, 10_000, list(range(30))
        plans, _ = self._plan([("cuda:0", per, sweep, 1024, mods)], {"cuda:0": 2 * (10 * per + sweep)})
        sizes = [len(c) for _w, _h, c in plans]
        # each chunk's own transient (chain + sweep) fits half the probe:
        # (usable//2 - sweep)//per = 10 per chunk, serially re-earned
        self.assertEqual(sizes, [10, 10, 10])

    def test_unprobeable_home_keeps_fixed_width(self):
        mods = list(range(5))
        plans, _ = self._plan([("cpu", 1000, 0, 2, mods)], {"cpu": None})
        self.assertEqual(
            [(w, h, len(c)) for w, h, c in plans], [("cpu", "cpu", 2), ("cpu", "cpu", 2), ("cpu", "cpu", 1)]
        )

    def test_balanced_homes_move_nothing(self):
        # a placement that already spans the candidates keeps every module
        # resident: only overflow from an over-quota home moves
        per = 1000
        group_home1 = list(range(4))
        group_home2 = ["a", "b", "c", "d"]
        probes = {"cuda:1": 2**41, "cuda:2": 2**41}
        picks = lambda ws, home_device=None: [home_device, "cuda:1", "cuda:2"]
        plans, _ = self._plan([("cuda:1", per, 0, 8, group_home1), ("cuda:2", per, 0, 8, group_home2)], probes, picks)
        for worker, home, chunk in plans:
            self.assertEqual(worker, home)
