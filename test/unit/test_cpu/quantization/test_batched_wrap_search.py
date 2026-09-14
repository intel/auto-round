# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
"""Batched same-shape wrap-search tests: grouping, bit-parity, deferral protocol."""

import os
import threading
import types
import unittest
from unittest import mock

import torch

import auto_round.algorithms.quantization.search_dispatch as search_dispatch
from auto_round.algorithms.quantization.search_dispatch import run_batched_wrap_search


def _mk_inputs(seed=0, shape=(6, 128), dtype="int", bits=4, thresh=1e-5, device="cpu"):
    from auto_round.data_type.utils import resolve_optimized_init_scale_fn

    g = torch.Generator().manual_seed(seed)
    w = torch.randn(*shape, generator=g)
    im = torch.rand(shape[1], generator=g) + 0.5  # RAW per-column imatrix, as production stages it
    fn = resolve_optimized_init_scale_fn(dtype, thresh)
    return w, im, dtype, bits, thresh, device, fn


class FakeDeferred:
    """Duck-typed deferred wrapper: stages inputs, finalizes with the result."""

    supports_batched_search = True

    def __init__(self, seed=0, shape=(6, 128), dtype="int", bits=4, device="cpu", thresh=1e-5, search_fn=None):
        w, im, dt, b, th, dev, fn = _mk_inputs(seed, shape, dtype, bits, thresh, device)
        if search_fn is not None:
            fn = search_fn
        self._deferred_search_inputs = (w, dt, b, im, th, fn)
        self.init_scale = None
        self.finalized_on_thread = None
        self.name = f"m{seed}"
        self.orig_layer = types.SimpleNamespace(group_size=128)

    def _run_deferred_search_now(self):
        from auto_round.data_type.utils import reshape_imatrix_for_weight

        w, _dt, b, im_raw, _th, fn = self._deferred_search_inputs
        self.init_scale = fn(w, b, reshape_imatrix_for_weight(im_raw, w, self.orig_layer.group_size))
        self._deferred_search_inputs = None

    def finalize_batched_search(self, init_scale):
        self.init_scale = init_scale
        self._deferred_search_inputs = None
        self.finalized_on_thread = threading.current_thread().name


class TestNonStackableAndDictResults(unittest.TestCase):
    def test_split_leading_dicts(self):
        # GGUF K-quant funcs return scale/zp as dicts whose tensor values
        # carry the flattened batch dim; sharing the whole dict per module
        # gives every layer N-times the metadata (CI: gguf q2_k export)
        from auto_round.algorithms.quantization.rtn.batched_search import _split_leading

        scale = {"scale": torch.arange(8.0).reshape(8, 1), "d_scale": torch.arange(4.0).reshape(4, 1), "meta": None}
        parts = _split_leading(scale, 2)
        self.assertEqual(len(parts), 2)
        for i, part in enumerate(parts):
            self.assertEqual(part["scale"].shape, (4, 1))
            self.assertTrue(torch.equal(part["scale"], scale["scale"].reshape(2, 4, 1)[i]))
            self.assertEqual(part["d_scale"].shape, (2, 1))
            self.assertIsNone(part["meta"])  # non-tensor values are shared

    def test_search_fn_stackable_predicate(self):
        import functools

        from auto_round.algorithms.quantization.search_dispatch import _search_fn_stackable
        from auto_round.data_type.int import search_scales
        from auto_round.data_type.nvfp import search_nvfp4_scale

        self.assertFalse(_search_fn_stackable(search_nvfp4_scale))
        self.assertFalse(_search_fn_stackable(functools.partial(search_nvfp4_scale)))

        def shim(w, bits, im):
            return None

        shim._torchdynamo_orig_callable = search_nvfp4_scale
        self.assertFalse(_search_fn_stackable(shim))  # compiled wrappers unwrap
        self.assertTrue(_search_fn_stackable(search_scales))
        self.assertTrue(_search_fn_stackable(functools.partial(search_scales)))

    def test_nv_search_group_runs_per_module(self):
        # the nv-fp4 search is not batch-aware; its groups must never receive
        # a stacked weight (CI: test_nvfp4_alg_ext IndexError)
        import functools

        from auto_round.data_type.nvfp import search_nvfp4_scale

        calls = []

        def per_module(w, bits, im):
            calls.append(tuple(w.shape))
            return torch.ones(w.shape[0], 1)

        fn = functools.wraps(search_nvfp4_scale)(per_module)
        wrappers = [FakeDeferred(seed=1, search_fn=fn), FakeDeferred(seed=2, search_fn=fn)]
        self.assertTrue(run_batched_wrap_search(wrappers))
        self.assertEqual(calls, [(6, 128), (6, 128)])  # two single-module calls, never [2, 6, 128]
        for wr in wrappers:
            self.assertIsNotNone(wr.init_scale)


class TestBatchedBitParity(unittest.TestCase):
    def test_stacked_search_matches_individual_exactly(self):
        fakes = [FakeDeferred(seed=i) for i in range(5)]
        individual = [_individual(f) for f in fakes]
        stacked_w = torch.stack([f._deferred_search_inputs[0] for f in fakes])
        stacked_im = _stacked_im(fakes, stacked_w)
        batched = fakes[0]._deferred_search_inputs[5](stacked_w, 4, stacked_im)
        for i, ref in enumerate(individual):
            self.assertTrue(torch.equal(batched[i], ref), f"module {i} diverged")

    def test_parity_across_shapes_and_bits(self):
        for shape, bits in [((8, 128), 4), ((13, 128), 3), ((4, 64), 2)]:
            fakes = [FakeDeferred(seed=i + bits * 10, shape=shape, bits=bits) for i in range(4)]
            refs = [
                f._deferred_search_inputs[5](f._deferred_search_inputs[0], bits, f._deferred_search_inputs[3])
                for f in fakes
            ]
            _sw = torch.stack([f._deferred_search_inputs[0] for f in fakes])
            batched = fakes[0]._deferred_search_inputs[5](_sw, bits, _stacked_im(fakes, _sw))
            for i, ref in enumerate(refs):
                self.assertTrue(torch.equal(batched[i], ref))


class TestRunBatchedWrapSearch(unittest.TestCase):
    def test_batches_same_shape_and_finalizes(self):
        fakes = [FakeDeferred(seed=i) for i in range(4)] + [FakeDeferred(seed=99, shape=(3, 128))]
        expected = [_individual(f) for f in fakes]
        handled = run_batched_wrap_search(fakes)
        self.assertTrue(handled)
        for f, ref in zip(fakes, expected):
            self.assertIsNone(f._deferred_search_inputs)
            self.assertIsNotNone(f.init_scale)
            self.assertTrue(torch.equal(f.init_scale, ref), f.name)

    def test_single_deferred_runs_individually(self):
        fakes = [FakeDeferred(seed=0)]
        handled = run_batched_wrap_search(fakes)
        self.assertTrue(handled)  # still handled (individual fallback), inputs consumed
        self.assertIsNotNone(fakes[0].init_scale)

    def test_shape_mismatch_groups_separately(self):
        a = [FakeDeferred(seed=i, shape=(6, 128)) for i in range(2)]
        b = [FakeDeferred(seed=i, shape=(5, 128)) for i in range(2)]
        run_batched_wrap_search(a + b)
        for f in a + b:
            self.assertIsNotNone(f.init_scale)

    def test_dtype_mismatch_groups_separately(self):
        a = FakeDeferred(seed=0)
        b = FakeDeferred(seed=1, dtype="mx_fp4")
        run_batched_wrap_search([a, b])  # different resolved fn -> separate groups, no raise
        self.assertIsNotNone(a.init_scale)
        self.assertIsNotNone(b.init_scale)

    def test_distinct_search_fns_never_share_a_batch(self):
        def fake_fn_a(w, bits, im):
            return w.abs().amax(dim=-1).squeeze(-1)

        def fake_fn_b(w, bits, im):
            return -w.abs().amax(dim=-1).squeeze(-1)

        a = FakeDeferred(seed=0, search_fn=fake_fn_a)
        b = FakeDeferred(seed=1, search_fn=fake_fn_b)
        exp_a = fake_fn_a(a._deferred_search_inputs[0], 4, a._deferred_search_inputs[3])
        exp_b = fake_fn_b(b._deferred_search_inputs[0], 4, b._deferred_search_inputs[3])
        run_batched_wrap_search([a, b])
        # each kept its own search semantics (no shared stack across fns)
        self.assertTrue(torch.equal(a.init_scale, exp_a))
        self.assertTrue(torch.equal(b.init_scale, exp_b))

    def test_empty_list(self):
        self.assertFalse(run_batched_wrap_search([]))

    def test_batch_chunking_respects_cap(self):
        fakes = [FakeDeferred(seed=i) for i in range(6)]
        calls = []
        real_stack = torch.stack

        def spy_stack(tensors, *a, **k):
            calls.append(len(tensors))
            return real_stack(tensors, *a, **k)

        with mock.patch.object(torch, "stack", side_effect=spy_stack):
            run_batched_wrap_search(fakes, max_batch=2)
        # 3 chunks x (weights + imatrices) = 6 stacks, each of size 2
        self.assertEqual(sorted(calls), [2] * 6)
        for f in fakes:
            self.assertIsNotNone(f.init_scale)

    def test_element_budget_caps_batch_size(self):
        # the ELEMENT budget (not the VRAM probe) binds: budget mocked to two
        # modules' worth of elements -> 6 modules split into batches of 2
        fakes = [FakeDeferred(seed=i, shape=(64, 32)) for i in range(6)]
        calls = []
        real_stack = torch.stack

        def spy_stack(tensors, *a, **k):
            calls.append(len(tensors))
            return real_stack(tensors, *a, **k)

        import auto_round.algorithms.quantization.search_dispatch as dispatch_mod

        per_module = 2 * 64 * 32  # weight + imatrix elements
        with mock.patch.object(torch, "stack", side_effect=spy_stack), mock.patch.object(
            dispatch_mod, "probe_usable_bytes", return_value=2**40
        ), mock.patch.object(dispatch_mod, "_wrap_batch_max_elems", return_value=2 * per_module):
            run_batched_wrap_search(fakes)
        self.assertEqual(calls, [2, 2, 2, 2, 2, 2])  # 3 batches x 2 stacks (weights + imatrix)

    def test_env_gb_override_caps_batches(self):
        import auto_round.algorithms.quantization.search_dispatch as dispatch_mod

        # tiny modules (0.26M elems each): 1 GiB budget would allow ~1000 -> force 2 per batch via GB override
        fakes = [FakeDeferred(seed=i, shape=(1024, 128)) for i in range(4)]
        calls = []
        real_stack = torch.stack

        def spy_stack(tensors, *a, **k):
            calls.append(len(tensors))
            return real_stack(tensors, *a, **k)

        with mock.patch.object(dispatch_mod.envs, "AR_SEARCH_BATCH_GB", 0.001), mock.patch.object(
            torch, "stack", side_effect=spy_stack
        ):
            run_batched_wrap_search(fakes)
        # 0.001 GiB = ~268K elements -> 268214 // (2*1024*128=262144) = 1 module/batch;
        # 4 batches x (weights + imatrices) = 8 stacks of size 1
        self.assertEqual(calls, [1] * 8)
        for f in fakes:
            self.assertIsNotNone(f.init_scale)

    def test_env_gb_override_invalid_raises(self):
        import os

        from auto_round import envs as envs_mod

        fakes = [FakeDeferred(seed=i) for i in range(4)]
        # through the real parser (module __getattr__ -> lambda raises)
        os.environ["AR_SEARCH_BATCH_GB"] = "not-a-number"
        try:
            with self.assertRaises(ValueError):
                run_batched_wrap_search(fakes)
        finally:
            os.environ.pop("AR_SEARCH_BATCH_GB", None)
            # sibling patch.object tests may leave a static module attr behind
            if "AR_SEARCH_BATCH_GB" in vars(envs_mod):
                delattr(envs_mod, "AR_SEARCH_BATCH_GB")

    def test_kill_switch_disables(self):
        fakes = [FakeDeferred(seed=i) for i in range(3)]
        with mock.patch.object(search_dispatch.envs, "AR_DISABLE_BATCHED_SEARCH", True):
            handled = run_batched_wrap_search(fakes)
        self.assertFalse(handled)
        self.assertIsNone(fakes[0].init_scale)  # inputs untouched: caller runs them per module
        for f in fakes:
            f._run_deferred_search_now()  # the caller's documented fallback
        self.assertIsNotNone(fakes[0].init_scale)


def _individual(fake):
    from auto_round.data_type.utils import reshape_imatrix_for_weight

    w, _dt, b, im_raw, _th, fn = fake._deferred_search_inputs
    return fn(w, b, reshape_imatrix_for_weight(im_raw, w, fake.orig_layer.group_size))


def _stacked_im(fakes, stacked_w):
    from auto_round.algorithms.quantization.search_dispatch import _materialize_wrap_imatrix

    return _materialize_wrap_imatrix(fakes, stacked_w)


class TestV2DeferralProtocol(unittest.TestCase):
    def _bare_v2(self):
        from auto_round.algorithms.quantization.sign_roundv2.quantizer import SignRoundOptimizedWrapperLinear

        w = object.__new__(SignRoundOptimizedWrapperLinear)
        w.init_scale = None
        w._deferred_search_inputs = None
        w.orig_layer = types.SimpleNamespace(group_size=128)
        return w

    def test_run_now_and_finalize_assign(self):
        from auto_round.data_type.utils import resolve_optimized_init_scale_fn

        w = self._bare_v2()
        weight = torch.randn(6, 128)
        imatrix_raw = torch.rand(128) + 0.5  # raw column, as production stages it
        fn = resolve_optimized_init_scale_fn("int", 1e-5)
        w._deferred_search_inputs = (weight, "int", 4, imatrix_raw, 1e-5, fn)
        from auto_round.data_type.utils import reshape_imatrix_for_weight

        ref = fn(weight, 4, reshape_imatrix_for_weight(imatrix_raw, weight, 128))
        w._run_deferred_search_now()
        self.assertTrue(torch.equal(w.init_scale, ref))
        self.assertIsNone(w._deferred_search_inputs)

        w2 = self._bare_v2()
        w2._deferred_search_inputs = (weight, "int", 4, imatrix_raw, 1e-5, fn)
        w2.finalize_batched_search(ref)
        self.assertTrue(torch.equal(w2.init_scale, ref))
        self.assertIsNone(w2._deferred_search_inputs)


class TestRealV2Construction(unittest.TestCase):
    """The kwarg must survive the REAL base __init__ chain (regression: kwargs swallowed it)."""

    def _layer(self):
        import torch.nn as nn

        layer = nn.Linear(128, 64, bias=False)
        layer.data_type = "int"
        layer.bits = 4
        layer.sym = True
        layer.group_size = 128
        layer.iters = 10
        layer.act_bits = 16
        return layer

    def _make(self, defer_search):
        from auto_round.algorithms.quantization.sign_roundv2.quantizer import SignRoundOptimizedWrapperLinear

        layer = self._layer()
        return SignRoundOptimizedWrapperLinear(
            layer,
            enable_minmax_tuning=False,
            enable_norm_bias_tuning=False,
            enable_torch_compile=False,
            device="cpu",
            defer_search=defer_search,
        )

    def test_deferred_stages_through_real_init(self):
        w = self._make(defer_search=True)
        self.assertIsNotNone(w._deferred_search_inputs, "defer_search was swallowed by the base __init__ kwargs")
        self.assertIsNone(w.init_scale)
        w._run_deferred_search_now()
        self.assertIsNotNone(w.init_scale)

    def test_non_deferred_searches_inline_through_real_init(self):
        w = self._make(defer_search=False)
        self.assertIsNone(w._deferred_search_inputs)
        self.assertIsNotNone(w.init_scale)

    def test_kwarg_absent_by_default(self):
        w = self._make(defer_search=False)
        self.assertIsNone(w._deferred_search_inputs)  # old callers unchanged


class TestWorkerBucketKeys(unittest.TestCase):
    def test_multi_worker_keys_are_device_parseable(self):
        """Regression: the threading key once carried the chunk entry dict itself."""
        import torch.nn as nn

        from auto_round.algorithms.quantization.rtn import batched_search

        model = nn.Module()
        staged = []
        for i in range(2):
            layer = TestBatchedRtnSearchParity()._layer(seed=i)
            setattr(model, f"l{i}", layer)
            staged.append((f"l{i}", TestBatchedRtnSearchParity()._make_wrapper(layer)))
        captured = {}

        def fake_run(keyed, fn):
            captured["keys"] = list(keyed.keys())

        with mock.patch.object(
            batched_search, "pick_search_worker_devices", side_effect=[["cuda:9"], ["cpu"]]
        ), mock.patch.object(batched_search, "run_items_by_device", side_effect=fake_run):
            # max_batch=1 -> two singleton chunks -> two distinct worker buckets;
            # run_items_by_device is mocked so nothing actually moves to cuda:9
            batched_search.run_batched_rtn_search(model, staged, max_batch=1)
        # distinct worker bucket vs home -> multi-bucket path taken; every key must parse
        self.assertEqual(sorted(captured["keys"]), ["cpu", "cuda:9"])
        for k in captured["keys"]:
            torch.device(k)  # the exact operation that crashed on the server


class TestEagerUnwrap(unittest.TestCase):
    def test_compiled_fn_unwraps_to_original(self):
        import torch

        def orig(x):
            return x * 2

        compiled = torch.compile(orig)
        unwrapped = getattr(compiled, "_torchdynamo_orig_callable", None) or compiled
        self.assertIs(unwrapped, orig)  # offloaded calls bypass the compiled wrapper
        self.assertEqual(unwrapped(torch.ones(2)).sum().item(), 4.0)

    def test_plain_fn_passes_through(self):
        fn = lambda x: x  # noqa: E731
        self.assertIs(getattr(fn, "_torchdynamo_orig_callable", None) or fn, fn)


class TestKwargRelocation(unittest.TestCase):
    def test_non_tensors_and_cpu_scalars_untouched(self):
        from auto_round.algorithms.quantization.rtn.batched_search import _relocate_tensor_kwargs

        scal = torch.tensor(1.0)
        kwargs = {"bits": 4, "v": scal, "group_size": 128, "data_type": "int"}
        out = _relocate_tensor_kwargs(dict(kwargs), "cpu")
        self.assertIs(out["v"], scal)  # cpu scalar: same object, no copy
        self.assertEqual(out["bits"], 4)

    def test_returns_same_dict_semantics(self):
        from auto_round.algorithms.quantization.rtn.batched_search import _relocate_tensor_kwargs

        kwargs = {"imatrix": None, "tensor_min": None}
        out = _relocate_tensor_kwargs(kwargs, "cpu")
        self.assertIsNone(out["imatrix"])


class TestSearchWorkerPicking(unittest.TestCase):
    def _pick(self, working_set, free_map, home="cuda:0"):
        import auto_round.algorithms.quantization.search_dispatch as dispatch_mod

        with mock.patch.object(torch.cuda, "device_count", return_value=len(free_map)), mock.patch.object(
            dispatch_mod, "probe_usable_bytes", side_effect=lambda k: free_map.get(k)
        ):
            return dispatch_mod.pick_search_worker_devices(working_set, home_device=home)

    def test_full_home_goes_to_idle_devices(self):
        free = {"cuda:0": 1 * 2**30, "cuda:1": 12 * 2**30, "cuda:2": 12 * 2**30}
        ws = 4 * 2**30
        self.assertEqual(self._pick(ws, free), ["cuda:1", "cuda:2"])  # home excluded, idles viable

    def test_all_full_falls_back_to_home(self):
        free = {"cuda:0": 0, "cuda:1": 1 * 2**30}
        self.assertEqual(self._pick(4 * 2**30, free), ["cuda:0"])

    def test_home_participates_when_it_fits(self):
        free = {"cuda:0": 10 * 2**30, "cuda:1": 10 * 2**30}
        self.assertEqual(self._pick(2 * 2**30, free), ["cuda:0", "cuda:1"])

    def test_kill_switch_pins_home(self):
        import auto_round.algorithms.quantization.search_dispatch as dispatch_mod

        with mock.patch.object(dispatch_mod.envs, "AR_DISABLE_MULTIGPU_SEARCH", True):
            self.assertEqual(dispatch_mod.pick_search_worker_devices(4 * 2**30, home_device="cuda:3"), ["cuda:3"])

    def test_no_cuda_returns_home(self):
        import auto_round.algorithms.quantization.search_dispatch as dispatch_mod

        with mock.patch.object(torch.cuda, "device_count", return_value=0):
            self.assertEqual(dispatch_mod.pick_search_worker_devices(4 * 2**30, home_device="cpu"), ["cpu"])

    def test_rtn_driver_offloads_to_viable_worker(self):
        # CPU-only: workers = [home] so behavior is the parity path already covered;
        # this pins that the bucket machinery runs end-to-end with the picker patched.
        import torch.nn as nn

        from auto_round.algorithms.quantization.rtn.batched_search import run_batched_rtn_search

        model = nn.Module()
        staged = []
        for i in range(2):
            layer = TestBatchedRtnSearchParity()._layer(seed=i)
            setattr(model, f"l{i}", layer)
            staged.append((f"l{i}", TestBatchedRtnSearchParity()._make_wrapper(layer)))
        with mock.patch(
            "auto_round.algorithms.quantization.search_dispatch.pick_search_worker_devices",
            return_value=["cpu"],
        ):
            run_batched_rtn_search(model, staged)
        for i in range(2):
            self.assertIsNotNone(getattr(model, f"l{i}").scale)


class TestWrapperBlockDrivesBatching(unittest.TestCase):
    def _fake_block(self):
        block = torch.nn.Module()
        for i in range(3):
            m = torch.nn.Linear(8, 8, bias=False)
            m.bits = 4
            setattr(block, f"l{i}", m)
        return block

    def test_protocol_class_deferred_and_batched(self):
        import auto_round.wrapper as wrapper_mod

        calls = {"created": 0, "finalized": 0}

        class FakeBatched(torch.nn.Module):
            supports_batched_search = True

            def __init__(self, layer, defer_search=False, **kwargs):
                super().__init__()
                self.orig_layer = layer
                self.init_scale = None
                self._deferred_search_inputs = None
                calls["created"] += 1
                assert defer_search is True
                w = layer.weight.data.reshape(-1, layer.weight.shape[1])
                from auto_round.data_type.utils import resolve_optimized_init_scale_fn

                self._deferred_search_inputs = (
                    w,
                    "int",
                    4,
                    None,  # uniform importance: production stages None, not ones
                    1e-5,
                    resolve_optimized_init_scale_fn("int", 1e-5),
                )

            def _run_deferred_search_now(self):
                w, _dt, b, im, _th, fn = self._deferred_search_inputs
                self.init_scale = fn(w, b, torch.ones_like(w) if im is None else im)
                self._deferred_search_inputs = None

            def finalize_batched_search(self, init_scale):
                self.init_scale = init_scale
                self._deferred_search_inputs = None
                calls["finalized"] += 1

        block = self._fake_block()
        q, u = wrapper_mod.wrapper_block(
            block, False, False, enable_torch_compile=False, device="cpu", wrapper_cls=FakeBatched
        )
        self.assertEqual(q, ["l0", "l1", "l2"])
        self.assertEqual(calls["created"], 3)
        self.assertEqual(calls["finalized"], 3)  # all three searched via the batch driver
        for m in (block.l0, block.l1, block.l2):
            self.assertIsNotNone(m.init_scale)

    def test_kill_switch_runs_per_module(self):
        import auto_round.algorithms.quantization.search_dispatch as dispatch_mod
        import auto_round.wrapper as wrapper_mod

        stats = {"inline": 0, "now": 0}

        class FakeBatched(torch.nn.Module):
            supports_batched_search = True

            def __init__(self, layer, defer_search=False, **kwargs):
                super().__init__()
                self.orig_layer = layer
                self.init_scale = None
                self._deferred_search_inputs = None
                if defer_search:
                    self._deferred_search_inputs = ("staged",)
                else:  # kill switch: the class must search inline, as the real wrapper does
                    stats["inline"] += 1
                    self.init_scale = torch.zeros(1)

            def _run_deferred_search_now(self):
                stats["now"] += 1
                self.init_scale = torch.zeros(1)
                self._deferred_search_inputs = None

            def finalize_batched_search(self, init_scale):
                raise AssertionError("must not batch under the kill switch")

        block = self._fake_block()
        with mock.patch.object(dispatch_mod.envs, "AR_DISABLE_BATCHED_SEARCH", True):
            wrapper_mod.wrapper_block(
                block, False, False, enable_torch_compile=False, device="cpu", wrapper_cls=FakeBatched
            )
        self.assertEqual(stats["inline"], 3)  # no deferral at all under the kill switch
        self.assertEqual(stats["now"], 0)
        self.assertIsNotNone(block.l0.init_scale)


class TestBatchedRtnSearchParity(unittest.TestCase):
    """iters=0 batching: stacked vs serial must be bit-identical (incl. the imatrix path)."""

    def setUp(self):
        # parity is asserted against a serial CPU arm; on GPU-rich boxes the
        # multigpu offload would move the batched arm to idle CUDA workers
        # (different device numerics) -- out of scope for these tests
        self._env = mock.patch.dict(os.environ, {"AR_DISABLE_MULTIGPU_SEARCH": "1"})
        self._env.start()
        self.addCleanup(self._env.stop)

    def _layer(self, seed, sym=True, with_imatrix=True):
        import torch.nn as nn

        g = torch.Generator().manual_seed(seed)
        layer = nn.Linear(128, 64, bias=False)
        with torch.no_grad():
            layer.weight.copy_(torch.randn(64, 128, generator=g))
        layer.data_type = "int"
        layer.bits = 4
        layer.sym = sym
        layer.group_size = 128
        layer.iters = 0
        layer.act_bits = 16
        layer.scale_dtype = torch.float16
        if with_imatrix:
            layer.imatrix = torch.rand(128, generator=g) + 0.5
        return layer

    def _make_wrapper(self, layer):
        from auto_round.wrapper import WrapperLinear

        return WrapperLinear(
            layer,
            device="cpu",
            enable_minmax_tuning=False,
            enable_norm_bias_tuning=False,
            enable_round_tuning=False,
            enable_torch_compile=False,
            disable_opt_rtn=False,
            iters=0,
        )

    def _run(self, sym, with_imatrix):
        from auto_round.algorithms.quantization.rtn.batched_search import run_batched_rtn_search

        # serial arm (production callers run under no_grad)
        serial = []
        with torch.no_grad():
            for i in range(3):
                layer = self._layer(seed=i, sym=sym, with_imatrix=with_imatrix)
                w = self._make_wrapper(layer)
                out = w.unwrapper({})
                serial.append((out.weight.data.clone(), out.scale, out.zp))
        # batched arm
        import torch.nn as nn

        model = nn.Module()
        staged = []
        for i in range(3):
            layer = self._layer(seed=i, sym=sym, with_imatrix=with_imatrix)
            setattr(model, f"l{i}", layer)
            w = self._make_wrapper(layer)
            staged.append((f"l{i}", w))
        run_batched_rtn_search(model, staged)
        for i in range(3):
            got = getattr(model, f"l{i}")
            ref_w, ref_scale, ref_zp = serial[i]
            self.assertTrue(torch.equal(got.weight.data, ref_w), f"weight mismatch module {i}")
            if isinstance(ref_scale, torch.Tensor):
                self.assertTrue(torch.equal(got.scale, ref_scale), f"scale mismatch module {i}")
            if ref_zp is not None and isinstance(ref_zp, torch.Tensor):
                self.assertTrue(torch.equal(got.zp, ref_zp), f"zp mismatch module {i}")

    def test_parity_sym_with_imatrix(self):
        self._run(sym=True, with_imatrix=True)

    def test_parity_sym_no_imatrix(self):
        self._run(sym=True, with_imatrix=False)

    def test_parity_asym_with_imatrix(self):
        self._run(sym=False, with_imatrix=True)

    def test_parity_act_quant_runs_full_unwrapper_tail(self):
        """Regression (review R4-3): the stacked write-back used _apply_qdq +
        attach(orig_layer), skipping the unwrapper tail -- act-quantized layers
        (act_bits <= 8, W4A8/W4A4) silently lost WrapperWALayer and the act
        metadata the serial and singleton paths attach."""
        import torch.nn as nn

        from auto_round.algorithms.quantization.rtn.batched_search import run_batched_rtn_search
        from auto_round.wrapper import WrapperWALayer

        def _act_layer(seed):
            layer = self._layer(seed)
            layer.act_bits = 8  # enables the act tail in unwrapper
            layer.act_data_type = "int"
            layer.act_sym = False
            layer.act_dynamic = False  # static act: the tail computes act_scale
            layer.act_group_size = -1
            return layer

        # serial arm: the full tail attaches a WrapperWALayer
        with torch.no_grad():
            w = self._make_wrapper(_act_layer(0))
            serial_out = w.unwrapper({})
        self.assertIsInstance(serial_out, WrapperWALayer)

        # batched arm: same module type + act metadata + bit-identical weights
        model = nn.Module()
        staged = []
        for i in range(3):
            layer = _act_layer(seed=i)
            setattr(model, f"l{i}", layer)
            staged.append((f"l{i}", self._make_wrapper(layer)))
        run_batched_rtn_search(model, staged)
        for i in range(3):
            got = getattr(model, f"l{i}")
            self.assertIsInstance(got, WrapperWALayer, f"module {i} lost the act wrapper")
            self.assertIsNotNone(getattr(got, "act_quant_func", None), f"module {i} missing act_quant_func")
            # the tail sets static-act attrs on the wrapped orig layer
            self.assertTrue(hasattr(got.orig_layer, "act_scale"), f"module {i} missing act_scale")
            self.assertIsNotNone(getattr(got.orig_layer, "scale", None), f"module {i} missing weight scale")
        ref_w = None
        with torch.no_grad():
            w_ref = self._make_wrapper(_act_layer(0))
            ref_w = w_ref.unwrapper({}).weight.data.clone()
        self.assertTrue(torch.equal(getattr(model, "l0").weight.data, ref_w))

    def _run_shape_dtype(self, out_f, in_f, gs, dtype):
        """Batched-vs-serial parity for arbitrary shapes/dtypes with an imatrix
        (regressions R6-2/R6-3: bf16 weights + fp32 imatrix; in % gs != 0)."""
        import torch.nn as nn

        from auto_round.algorithms.quantization.rtn.batched_search import run_batched_rtn_search

        def _layer(seed):
            import torch.nn as nn

            g = torch.Generator().manual_seed(seed)
            layer = nn.Linear(in_f, out_f, bias=False)
            with torch.no_grad():
                layer.weight.copy_(torch.randn(out_f, in_f, generator=g).to(dtype))
            layer.data_type = "int"
            layer.bits = 4
            layer.sym = False
            layer.group_size = gs
            layer.iters = 0
            layer.act_bits = 16
            layer.scale_dtype = torch.float32
            layer.imatrix = (torch.rand(in_f, generator=g) + 0.5).float()  # fp32, like the hook
            return layer

        with torch.no_grad():
            serial = []
            for i in range(3):
                w = self._make_wrapper(_layer(i))
                out = w.unwrapper({})
                serial.append((out.weight.data.clone(), out.scale))

        model = nn.Module()
        staged = []
        for i in range(3):
            layer = _layer(i)
            setattr(model, f"l{i}", layer)
            staged.append((f"l{i}", self._make_wrapper(layer)))
        run_batched_rtn_search(model, staged)
        for i in range(3):
            got = getattr(model, f"l{i}")
            ref_w, ref_scale = serial[i]
            self.assertTrue(torch.equal(got.weight.data, ref_w), f"weight mismatch module {i}")
            self.assertTrue(torch.equal(got.scale, ref_scale), f"scale mismatch module {i}")

    def test_parity_bf16_weights_fp32_imatrix(self):
        self._run_shape_dtype(64, 128, 128, torch.bfloat16)

    def test_parity_fp16_weights_fp32_imatrix(self):
        self._run_shape_dtype(64, 128, 128, torch.float16)

    def test_parity_group_size_not_dividing_in_features(self):
        self._run_shape_dtype(3, 10, 4, torch.float32)

    def test_staged_key_normalizes_compiled_quant_fn(self):
        """Regression (R6-4): with enable_torch_compile, every wrapper carries a
        UNIQUE torch.compile wrapper around the SHARED eager fn; keying the raw
        callable collapsed all groups to singletons (batching silently dead)."""
        from auto_round.algorithms.quantization.rtn.batched_search import _staged_key

        class _Compiled:
            def __init__(self, orig):
                self._torchdynamo_orig_callable = orig

            def __call__(self, *a, **k):  # pragma: no cover
                raise AssertionError("compiled fn must not run here")

        def _shared_eager(w, bits, imatrix):  # pragma: no cover - identity stub
            return None

        layers, wrappers = [], []
        for i in range(2):
            layer = self._layer(seed=i)
            layers.append(layer)
            w = self._make_wrapper(layer)
            w.weight_quant_func = _Compiled(_shared_eager)  # distinct wrappers, shared orig
            wrappers.append(w)
        k0 = _staged_key(wrappers[0], wrappers[0].orig_layer.weight, None)
        k1 = _staged_key(wrappers[1], wrappers[1].orig_layer.weight, None)
        self.assertEqual(k0, k1)

    def test_quant_call_kwargs_never_relocates_imatrix_to_meta(self):
        """Regression (R6-1): a meta-resident stored weight must not become the
        imatrix relocation target."""
        layer = self._layer(seed=0)
        layer.imatrix = torch.rand(layer.weight.shape[1]) + 0.5
        w = self._make_wrapper(layer)
        # simulate the low-memory lane: stored weight on meta, real data via get_weight
        real = layer.weight.data.clone()
        layer.weight = torch.nn.Parameter(torch.empty_like(real, device="meta"), requires_grad=False)
        layer.get_weight = lambda: real
        kwargs = w._quant_call_kwargs(torch.tensor(0.0), torch.tensor(1.0), torch.tensor(1.0))
        im = kwargs.get("imatrix")
        self.assertIsNotNone(im)
        self.assertNotEqual(im.device.type, "meta")
        self.assertEqual(im.device.type, "cpu")  # wrapper device in these tests

    def test_threaded_write_back_swaps_compiled_act_fn_to_eager(self):
        """Regression (review R5-1): on worker threads the unwrapper act tail
        would make its FIRST call to the torch.compile-wrapped act_quant_func
        -- the same dynamo trace-lock race the weight fn had. The threaded
        write-back must swap BOTH compiled callables (weight AND act) to their
        eager originals and restore them afterwards."""
        from auto_round.algorithms.quantization.rtn.batched_search import swap_wrapper_callables_to_eager

        ran_eager = []

        def _fake_eager_act(x, **kwargs):
            ran_eager.append(True)
            return x, torch.ones(1), None

        class _Compiled:
            def __init__(self, orig):
                self._torchdynamo_orig_callable = orig

            def __call__(self, *a, **k):  # pragma: no cover - must never run on workers
                raise AssertionError("compiled callable ran while swapped out")

        w = self._make_wrapper(self._layer(0))
        compiled_w = _Compiled(lambda *a, **k: (None, None, None))
        compiled_a = _Compiled(_fake_eager_act)
        w.weight_quant_func = compiled_w
        w.act_quant_func = compiled_a

        restore = swap_wrapper_callables_to_eager(w)
        self.assertIsNotNone(restore)
        # swapped: both attrs now point at the eager originals
        self.assertIs(w.weight_quant_func, compiled_w._torchdynamo_orig_callable)
        self.assertIs(w.act_quant_func, _fake_eager_act)
        w.act_quant_func(None)  # the eager act callable is what runs on the worker
        restore()
        # restored: both compiled callables back in place
        self.assertIs(w.weight_quant_func, compiled_w)
        self.assertIs(w.act_quant_func, compiled_a)

        # wrappers without compiled callables need no swap
        w2 = self._make_wrapper(self._layer(1))
        self.assertIsNone(swap_wrapper_callables_to_eager(w2))

    def _conv1d_layer(self, seed, nf, nx):
        from transformers.pytorch_utils import Conv1D

        g = torch.Generator().manual_seed(seed)
        layer = Conv1D(nf, nx)  # weight is [nx, nf]; quant math runs on the transpose
        with torch.no_grad():
            layer.weight.copy_(torch.randn(nx, nf, generator=g))
        layer.data_type = "int"
        layer.bits = 4
        layer.sym = True
        layer.group_size = min(nx, 128)
        layer.iters = 0
        layer.act_bits = 16
        layer.scale_dtype = torch.float16
        layer.imatrix = torch.rand(nx, generator=g) + 0.5
        return layer

    def _run_conv1d(self, nf, nx):
        from auto_round.algorithms.quantization.rtn.batched_search import run_batched_rtn_search

        with torch.no_grad():
            serial = []
            for i in range(3):
                w = self._make_wrapper(self._conv1d_layer(seed=i, nf=nf, nx=nx))
                out = w.unwrapper({})
                serial.append((out.weight.data.clone(), out.scale, out.zp))
        import torch.nn as nn

        model = nn.Module()
        staged = []
        for i in range(3):
            layer = self._conv1d_layer(seed=i, nf=nf, nx=nx)
            setattr(model, f"c{i}", layer)
            staged.append((f"c{i}", self._make_wrapper(layer)))
        run_batched_rtn_search(model, staged)
        for i in range(3):
            got = getattr(model, f"c{i}")
            ref_w, ref_scale, ref_zp = serial[i]
            self.assertTrue(torch.equal(got.weight.data, ref_w), f"weight mismatch module {i}")
            self.assertEqual(tuple(got.weight.data.shape), (nx, nf))  # Conv1D layout restored
            if isinstance(ref_scale, torch.Tensor):
                self.assertTrue(torch.equal(got.scale, ref_scale), f"scale mismatch module {i}")

    def _run_kwargs_mix(self, mutate):
        # two same-shape modules whose per-layer quant kwargs differ must NOT
        # share a stacked chunk (chunk[0]'s kwargs would silently apply to
        # both), and each must still match its serial result bit-for-bit
        from auto_round.algorithms.quantization.rtn.batched_search import run_batched_rtn_search

        with torch.no_grad():
            serial = []
            for i in range(2):
                layer = self._layer(seed=i, sym=True, with_imatrix=False)
                mutate(layer, i)
                w = self._make_wrapper(layer)
                out = w.unwrapper({})
                serial.append(out.weight.data.clone())
        import torch.nn as nn

        model = nn.Module()
        staged = []
        for i in range(2):
            layer = self._layer(seed=i, sym=True, with_imatrix=False)
            mutate(layer, i)
            setattr(model, f"k{i}", layer)
            staged.append((f"k{i}", self._make_wrapper(layer)))
        stacks = []
        real_stack = torch.stack

        def spy_stack(tensors, *a, **k):
            stacks.append(len(tensors))
            return real_stack(tensors, *a, **k)

        with mock.patch.object(torch, "stack", side_effect=spy_stack):
            run_batched_rtn_search(model, staged)
        self.assertEqual(stacks, [])  # no stacked call: differing kwargs -> per-module
        for i in range(2):
            got = getattr(model, f"k{i}")
            self.assertTrue(torch.equal(got.weight.data, serial[i]), f"weight mismatch module {i}")

    def test_mixed_scale_dtype_never_shares_batch(self):
        def mutate(layer, i):
            layer.scale_dtype = torch.float16 if i == 0 else torch.bfloat16

        self._run_kwargs_mix(mutate)

    def test_super_bits_on_one_layer_never_shares_batch(self):
        def mutate(layer, i):
            if i == 1:
                layer.super_bits = 2
                layer.super_group_size = 16

        self._run_kwargs_mix(mutate)

    def test_global_scale_difference_never_shares_batch(self):
        import torch.nn as nn

        from auto_round.algorithms.quantization.rtn.batched_search import run_batched_rtn_search

        with torch.no_grad():
            serial = []
            for i in range(2):
                layer = self._layer(seed=i, sym=True, with_imatrix=False)
                w = self._make_wrapper(layer)
                w.weight_global_scale = torch.tensor(1.0 + i)
                out = w.unwrapper({})
                serial.append(out.weight.data.clone())

        model = nn.Module()
        staged = []
        for i in range(2):
            layer = self._layer(seed=i, sym=True, with_imatrix=False)
            w = self._make_wrapper(layer)
            w.weight_global_scale = torch.tensor(1.0 + i)
            setattr(model, f"g{i}", layer)
            staged.append((f"g{i}", w))
        stacks = []
        real_stack = torch.stack

        def spy_stack(tensors, *a, **k):
            stacks.append(len(tensors))
            return real_stack(tensors, *a, **k)

        with mock.patch.object(torch, "stack", side_effect=spy_stack):
            run_batched_rtn_search(model, staged)
        self.assertEqual(stacks, [])
        for i in range(2):
            got = getattr(model, f"g{i}")
            self.assertTrue(torch.equal(got.weight.data, serial[i]), f"weight mismatch module {i}")

    def test_parity_conv1d_square(self):
        self._run_conv1d(nf=64, nx=64)

    def test_parity_conv1d_nonsquare(self):
        # non-square is where the missing transpose is a hard crash, not silent corruption
        self._run_conv1d(nf=64, nx=128)

    def test_oom_chunk_falls_back_per_module(self):
        import torch.nn as nn

        from auto_round.algorithms.quantization.rtn import batched_search

        model = nn.Module()
        staged = []
        for i in range(2):
            layer = self._layer(seed=i)
            setattr(model, f"l{i}", layer)
            staged.append((f"l{i}", self._make_wrapper(layer)))
        real_fn = staged[0][1].weight_quant_func
        calls = {"n": 0, "raised": 0}

        def boom(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:  # only the stacked call raises; per-module fallback delegates
                calls["raised"] += 1
                raise torch.OutOfMemoryError("simulated")
            return real_fn(*a, **k)

        # patch BOTH wrappers with the SAME callable so the group key still matches
        with mock.patch.object(staged[0][1], "weight_quant_func", boom), mock.patch.object(
            staged[1][1], "weight_quant_func", boom
        ):
            batched_search.run_batched_rtn_search(model, staged)
        self.assertEqual(calls["raised"], 1)  # the stacked call raised exactly once
        self.assertEqual(calls["n"], 3)  # then the two per-module fallbacks delegated to the real fn
        for i in range(2):
            got = getattr(model, f"l{i}")
            self.assertIsNotNone(getattr(got, "scale", None))  # per-module fallback quantized it
