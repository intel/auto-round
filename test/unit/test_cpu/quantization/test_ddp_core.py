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
"""CPU-tier tests for the ported single-process DDP tuning surface.

Covers the device-agnostic contract: shard-constrained samplers, plan
resolution with injected free-VRAM data, transport encoding math, pool
distribution layout, gather-on-CPU paths, env defaults, and the
block-runner device-safe concatenation. Mirror/replica behavior on real
CUDA devices lives in the CUDA tier (test_ddp_mirror.py + e2e).
"""

import pytest
import torch

from auto_round.algorithms.block_runner import _cat_device_safe
from auto_round.algorithms.quantization.sign_round.data_parallel import (
    _encode_transport,
    _xchg,
    distribute_pool,
    gather_block_for_mirroring_,
    resolve_ddp_plan,
)
from auto_round.compressors.utils import IndexSampler, shard_samplers


class TestShardSamplers:
    def test_layout_and_epoch_coverage(self):
        samplers = shard_samplers(nsamples=8, world=4, batch_per_replica=1)
        assert samplers is not None and len(samplers) == 4
        drawn = [s.next_batch() for s in samplers]
        # each replica draws only from its own contiguous shard
        for r, batch in enumerate(drawn):
            assert all(r * 2 <= j < (r + 1) * 2 for j in batch)
        # one shard-epoch covers every sample exactly once (batch=1 x shard=2 draws)
        rest = [s.next_batch() for s in samplers]
        assert sorted(j for b in drawn + rest for j in b) == list(range(8))

    def test_ineligible_layouts_return_none(self):
        assert shard_samplers(nsamples=7, world=4, batch_per_replica=1) is None  # indivisible
        assert shard_samplers(nsamples=8, world=1, batch_per_replica=8) is None  # single device
        assert shard_samplers(nsamples=8, world=4, batch_per_replica=3) is None  # draw > shard

    def test_index_sampler_explicit_pool(self):
        s = IndexSampler(4, 2, indices=[10, 11, 12, 13])
        b = s.next_batch()
        assert sorted(b) in ([10, 11], [12, 13]) or set(b) <= {10, 11, 12, 13}
        with pytest.raises(ValueError):
            IndexSampler(3, 2, indices=[1, 2])


class TestResolveDDPPlan:
    def _plan(self, world, free, footprint=100):
        return resolve_ddp_plan(
            world,
            torch.device("cuda", 0),  # device OBJECTS only -- no CUDA runtime touched
            8,
            visible_cuda_devices=[0, 1, 2, 3],
            explicit_devices=["0", "1", "2", "3"],  # bare indices: normalized to cuda:N
            vram_free_bytes={torch.device("cuda", i): free[i] for i in range(4)},
            mirror_footprint_bytes=footprint,
            margin_bytes=0,  # toy byte values in these tests
        )

    def test_full_world_when_vram_fits(self):
        plan = self._plan(4, [1000, 1000, 1000, 1000])
        assert plan.enabled and plan.world == 4
        assert plan.shard_size == 2

    def test_device_without_mirror_fit_is_dropped(self):
        plan = self._plan(4, [1000, 10, 1000, 1000])  # cuda:1 too small
        assert plan.enabled
        assert torch.device("cuda", 1) not in plan.devices
        assert plan.world == 3  # reduced by the VRAM guard
        assert any("world reduced" in n for n in plan.notes)
        # the quantizer gates non-power-of-two worlds (fail-visible downgrade to
        # serial with an INFO) -- resolve itself just reports the fitting subset

    def test_world_collapses_to_one_when_nothing_fits(self):
        plan = self._plan(4, [5, 5, 5, 5], footprint=100)
        assert not plan.enabled

    def test_non_power_of_two_request_resolves_but_caller_gates(self):
        # resolve reports the fitting subset; the shared resolver's
        # power-of-two gate is what disables engagement for world=3
        plan = resolve_ddp_plan(
            3,
            torch.device("cuda", 0),
            12,
            visible_cuda_devices=[0, 1, 2, 3],
            explicit_devices=["0", "1", "2"],
            vram_free_bytes={torch.device("cuda", i): 1 << 30 for i in range(3)},
            mirror_footprint_bytes=1,
            margin_bytes=0,
        )
        assert plan.world == 3 and plan.enabled


class TestTransportMath:
    def test_fp32_passthrough(self):
        t = torch.randn(16)
        out = _xchg(t, t.device, torch.float32, "fp32")
        assert out is t

    def test_bf16_transport_roundtrip_preserves_signs(self):
        t = torch.randn(1024)
        out = _xchg(t, t.device, torch.float32, "bf16")
        assert torch.equal(torch.sign(out), torch.sign(t))

    def test_int8_transport_preserves_signs(self):
        t = torch.randn(1024) * 0.01
        out = _xchg(t, t.device, torch.float32, "int8")
        # signs are what sign-SGD consumes; |t| below the int8 quantum rounds
        # to zero (sign 0), magnitudes carry bounded int8 error
        assert ((torch.sign(out) == torch.sign(t)) | (out == 0)).all()
        assert (out - t).abs().max() < 1e-3

    def test_encode_transport_meta(self):
        t = torch.randn(8)
        payload, meta = _encode_transport(t, "int8")
        assert payload.dtype == torch.int8 and meta is not None
        payload, meta = _encode_transport(t, "bf16")
        assert payload.dtype == torch.bfloat16 and meta is None
        payload, meta = _encode_transport(t, "fp32")
        assert payload is t and meta is None


class TestDistributePool:
    def test_plan_layout_covers_pool_exactly(self):
        # distribute_pool gives device r the contiguous range [r*shard,(r+1)*shard);
        # on uniform CPU devices that is a no-op, so pin the layout arithmetic
        # through the plan the pool follows
        from auto_round.algorithms.quantization.sign_round.data_parallel import resolve_ddp_plan

        plan = resolve_ddp_plan(
            4,
            torch.device("cuda", 0),
            8,
            visible_cuda_devices=[0, 1, 2, 3],
            explicit_devices=["0", "1", "2", "3"],  # avoids torch.cuda.device_count() on CUDA-less hosts
            vram_free_bytes={torch.device("cuda", i): 1 << 40 for i in range(4)},
            mirror_footprint_bytes=1,
            margin_bytes=0,
        )
        assert plan.world == 4
        assert plan.shard_size * plan.world == 8
        # device objects in the plan are cuda:N (unusable on CUDA-less hosts);
        # exercise the scatter loop with the degenerate all-cpu device list
        distribute_pool([torch.zeros(2) for _ in range(8)], [torch.device("cpu")] * 4)
        # indivisible / too-small pools are left alone by contract (serial cats handle them)
        distribute_pool([torch.zeros(2)], [torch.device("cpu")] * 4)


class TestGatherOnCPU:
    def test_noop_on_whole_block(self):
        block = torch.nn.Linear(4, 4)
        assert gather_block_for_mirroring_(block, torch.device("cpu")) is False

    def test_repoints_stale_tuning_device_strings(self):
        block = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        block[1].tuning_device = "cuda:5"
        assert gather_block_for_mirroring_(block, torch.device("cpu")) is True
        assert str(block[1].tuning_device) == "cpu"

    def test_moves_wrapper_dict_state(self):
        block = torch.nn.Linear(4, 4)
        block.params = {"v": torch.nn.Parameter(torch.ones(4), requires_grad=True)}
        gather_block_for_mirroring_(block, torch.device("cpu"))
        assert block.params["v"].device.type == "cpu"


class TestCatDeviceSafe:
    def test_same_device_is_plain_cat(self):
        parts = [torch.zeros(2, 3), torch.ones(2, 3)]
        out = _cat_device_safe(parts, dim=0)
        assert torch.equal(out, torch.cat(parts, dim=0))

    def test_empty_selection_raises(self):
        with pytest.raises(ValueError):
            _cat_device_safe([], dim=0)


class TestEnvDefaults:
    def test_ddp_defaults_resolve(self, monkeypatch):
        for name in ("AR_TUNE_DDP_WORLD", "AR_TUNE_DDP_DEVICES"):
            monkeypatch.delenv(name, raising=False)
        from auto_round import envs

        assert envs.AR_TUNE_DDP_WORLD == 1
        assert envs.AR_TUNE_DDP_DEVICES == ""

    def test_world_env_round_trip(self, monkeypatch):
        monkeypatch.setenv("AR_TUNE_DDP_WORLD", "4")
        from auto_round import envs

        assert envs.AR_TUNE_DDP_WORLD == 4


class TestTupleKwargSlicing:
    """transformers-v5 rope arrives as position_embeddings=(cos, sin) with a
    per-sample batch dim; shard/batch forwards smaller than the cached batch
    must slice tuple-of-tensors kwargs elementwise or crash in rope."""

    def _runner(self, n=8, batch_size=4):
        from auto_round.algorithms.block_runner import BlockForwardRunner

        inputs = [torch.randn(1, 6, 4) for _ in range(n)]
        return (
            BlockForwardRunner(
                batch_dim=0, batch_size=batch_size, device=torch.device("cpu"), cache_device="cpu", amp=False
            ),
            inputs,
        )

    def test_tuple_kwargs_sliced_to_batch_indices(self):
        runner, inputs = self._runner()
        cos = torch.randn(8, 6, 2)
        sin = torch.randn(8, 6, 2)
        _, others = runner.select_batch(inputs, {"position_embeddings": (cos, sin)}, [3, 5])
        assert isinstance(others["position_embeddings"], tuple)
        assert others["position_embeddings"][0].shape == (2, 6, 2)
        assert others["position_embeddings"][1].shape == (2, 6, 2)
        assert torch.equal(others["position_embeddings"][0][0], cos[3])
        assert torch.equal(others["position_embeddings"][0][1], cos[5])

    def test_broadcast_tuple_elements_pass_through(self):
        runner, inputs = self._runner()
        table = torch.randn(1, 6, 2)  # broadcast-shaped: not per-sample
        _, others = runner.select_batch(inputs, {"position_embeddings": (table, table)}, [3, 5])
        assert others["position_embeddings"][0] is table  # unsliced, broadcastable

    def test_mixed_and_non_tensor_tuples_untouched(self):
        runner, inputs = self._runner()
        val = (torch.randn(8, 6, 2), "flag")
        _, others = runner.select_batch(inputs, {"weird": val}, [3, 5])
        assert others["weird"] is val  # mixed tuple treated as opaque


class TestSharedCacheShardPick:
    """shared_cache_keys kwargs arrive as ONE ENTRY PER BATCH (list[n_pool/batch]).
    A sub-batch draw (DDP shard) must pick the owning batch's entry and slice its
    within-batch rows; full-batch draws keep the legacy pick exactly."""

    def _runner(self):
        from auto_round.algorithms.block_runner import BlockForwardRunner

        inputs = [torch.randn(1, 6, 4) for _ in range(128)]
        return (
            BlockForwardRunner(
                batch_dim=0,
                batch_size=8,
                device=torch.device("cpu"),
                cache_device="cpu",
                amp=False,
                shared_cache_keys=("position_embeddings",),
            ),
            inputs,
        )

    def _pe(self):
        # 16 batch entries, each an (cos, sin) tuple of [8, S, D] per-sample tensors
        return [(torch.full((8, 6, 2), float(b)), torch.full((8, 6, 2), -float(b))) for b in range(16)]

    def test_shard_picks_owning_batch_and_rows(self):
        runner, inputs = self._runner()
        pe = self._pe()
        _, others = runner.select_batch(inputs, {"position_embeddings": pe}, [32, 33])
        cos, sin = others["position_embeddings"]
        assert cos.shape == (2, 6, 2) and sin.shape == (2, 6, 2)
        assert torch.all(cos == 4.0) and torch.all(sin == -4.0)  # entry 4, rows 0-1

    def test_shard_wrapping_rows_across_batch_boundary_fails_safe(self):
        runner, inputs = self._runner()
        pe = self._pe()
        _, others = runner.select_batch(inputs, {"position_embeddings": pe}, [7, 8])
        # batch 0 entry (indices[0]=7), rows 7 and 0 -- within-entry slice
        cos, _ = others["position_embeddings"]
        assert cos.shape == (2, 6, 2) and torch.all(cos == 0.0)

    def test_full_batch_draw_keeps_legacy_pick(self):
        runner, inputs = self._runner()
        pe = self._pe()
        _, others = runner.select_batch(inputs, {"position_embeddings": pe}, list(range(8, 16)))
        # legacy: multi-index full-batch -> val[0], untouched (batch 0's entry, 8 rows)
        cos, _ = others["position_embeddings"]
        assert isinstance(cos, torch.Tensor) and cos.shape[0] == 8 and torch.all(cos == 0.0)

    def test_single_index_sub_batch_draw_per_batch_list(self):
        runner, inputs = self._runner()
        pe = self._pe()
        _, others = runner.select_batch(inputs, {"position_embeddings": pe}, [5])
        # per-batch list: sample 5 -> batch 0 entry, row 5
        cos, _ = others["position_embeddings"]
        assert cos.shape == (1, 6, 2) and torch.all(cos == 0.0)

    def test_per_sample_list_sub_batch_draw(self):
        runner, inputs = self._runner()
        # per-sample layout: 128 entries, each a [1, 6, 2] tensor
        ps = [torch.full((1, 6, 2), float(i)) for i in range(128)]
        _, others = runner.select_batch(inputs, {"position_embeddings": ps}, [32, 33])
        assert others["position_embeddings"].shape == (2, 6, 2)
        assert torch.all(others["position_embeddings"][0] == 32.0)


class TestLoggingGlobals:
    def test_engaged_log_globals_initialized(self):
        """Regression: the env-strip once dropped the module-level
        ``_ENGAGED_LOGGED_SIG`` init, and the ``global`` read in
        resolve_tune_ddp_plan_ then NameError'd -- but only on GPU runs,
        because a CPU home declines before the sig block (CPU tests cannot
        reach the engaged path)."""
        from auto_round.algorithms.quantization.sign_round import data_parallel as dp

        assert dp._ENGAGED_LOGGED_SIG is None
        assert isinstance(dp._coll_mirror_setup_logged, set)


class TestRequestedWorldErrors:
    """A requested parallel world is a requirement: infeasible -> RuntimeError, never silent serial."""

    def _fake_quantizer(self):
        from types import SimpleNamespace

        q = SimpleNamespace(
            iters=10,
            gradient_accumulate_steps=1,
            enable_lfq=False,
            _resolved_ddp_plan=None,
        )
        q._get_scaler = lambda: None
        return q

    def test_infeasible_world_raises(self, monkeypatch):
        import torch

        from auto_round.algorithms.quantization.sign_round.data_parallel import resolve_tune_ddp_plan_

        monkeypatch.setenv("AR_TUNE_DDP_WORLD", "2")
        block = torch.nn.Sequential(torch.nn.Linear(4, 4))
        with pytest.raises(RuntimeError, match="ineligible"):
            resolve_tune_ddp_plan_(self._fake_quantizer(), block, [torch.zeros(1)], None, "cpu")

    def test_iters0_is_not_a_decline_reason(self, monkeypatch):
        """The DDP world shards the collection at iters=0 too (campaign
        semantics restored): with a fake quantizer at iters=0 the only
        ineligibility reason left must be the non-CUDA home, never iters."""
        import torch

        from auto_round.algorithms.quantization.sign_round.data_parallel import resolve_tune_ddp_plan_

        monkeypatch.setenv("AR_TUNE_DDP_WORLD", "2")
        q = self._fake_quantizer()
        q.iters = 0
        block = torch.nn.Sequential(torch.nn.Linear(4, 4))
        with pytest.raises(RuntimeError) as excinfo:
            resolve_tune_ddp_plan_(q, block, [torch.zeros(1)], None, "cpu")
        assert "iters" not in str(excinfo.value)
        assert "not CUDA" in str(excinfo.value)

    def test_no_world_set_stays_serial(self, monkeypatch):
        import torch

        from auto_round.algorithms.quantization.sign_round.data_parallel import resolve_tune_ddp_plan_

        monkeypatch.delenv("AR_TUNE_DDP_WORLD", raising=False)
        block = torch.nn.Sequential(torch.nn.Linear(4, 4))
        plan = resolve_tune_ddp_plan_(self._fake_quantizer(), block, [torch.zeros(1)], None, "cpu")
        assert plan.world == 1


class TestRtnPhaseLogging:
    """The OptRTN iters=0 path logs per-phase walls (norm vs per-layer quant)."""

    def test_quantize_block_logs_phases(self, caplog, _autoround_log_propagate):
        import torch

        from auto_round.algorithms.quantization.rtn.quantizer import OptimizedRTNQuantizer

        class _Layer(torch.nn.Linear):
            def __init__(self):
                super().__init__(4, 4)
                self.global_name = "model.test.proj"

        class _Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = _Layer()
                self.proj.imatrix = torch.ones(4)
                self.proj.imatrix_cnt = 2

        class _Q(OptimizedRTNQuantizer):
            def __init__(self):
                pass  # bypass config init; only exercise quantize_block plumbing

            def quantize_layer_outside_block(self, layer, **kwargs):
                pass

        from auto_round import envs as _envs

        monkey = None
        import os

        os.environ["AR_PERF_COUNTERS"] = "1"
        try:
            with caplog.at_level("INFO"):
                _Q().quantize_block(_Block(), [], {}, [], None, None)
            assert any("[perf] rtn phases:" in r.message for r in caplog.records)
        finally:
            os.environ.pop("AR_PERF_COUNTERS", None)


class TestLoopTailContract:
    """The v1 tune loop must record ``init_loss = total_loss`` at its tail.

    The DDP-arc loop surgery once dropped this upstream pair (together with the
    per-iter debug line); the end-of-block summary then formatted ``None`` and
    took down every iters>0 CPU test (CI build 75613, 93 failures), with the
    aborted-block leftovers cascading into the 4 WrapperLinear.linear_forward
    failures. The e2e pins are the iters>0 tests themselves; this AST contract
    pins the exact regression class hermetically: the assignment must live
    inside the iters loop of ``quantize_block``.
    """

    def test_init_loss_assigned_inside_loop(self):
        import ast
        import inspect

        from auto_round.algorithms.quantization.sign_round import quantizer as v1

        src = inspect.getsource(v1)
        found = False
        for fn in ast.walk(ast.parse(src)):
            if isinstance(fn, ast.FunctionDef) and fn.name == "quantize_block":
                for node in ast.walk(fn):
                    if isinstance(node, ast.For):
                        for st in ast.walk(node):
                            if (
                                isinstance(st, ast.Assign)
                                and any(isinstance(t, ast.Name) and t.id == "init_loss" for t in st.targets)
                                and isinstance(st.value, ast.Name)
                                and st.value.id == "total_loss"
                            ):
                                found = True
        assert found, "v1 quantize_block lost the loop-tail `init_loss = total_loss` assignment"


class TestTunePhaseLine:
    """Formatter for the AR_PERF_COUNTERS per-block tune phase breakdown."""

    def test_formats_all_four_buckets(self):
        from auto_round.algorithms.quantization.sign_round.quantizer import _tune_phase_line

        line = _tune_phase_line({"wrap": 8.21, "prepare": 0.42, "loop": 3.5, "tail": 0.47}, 0)
        assert "iters=0" in line
        assert "wrap=8.21s" in line and "prepare=0.42s" in line
        assert "loop=3.50s" in line and "tail=0.47s" in line

    def test_missing_buckets_default_to_zero(self):
        from auto_round.algorithms.quantization.sign_round.quantizer import _tune_phase_line

        line = _tune_phase_line({}, 10)
        assert "wrap=0.00s" in line and "tail=0.00s" in line and "iters=10" in line


class TestBlockHasTuningEntries:
    """DDP decline check: all-float blocks must be detectable pre-engagement."""

    def test_plain_block_has_no_entries(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import block_has_tuning_entries

        assert block_has_tuning_entries(torch.nn.Sequential(torch.nn.Linear(8, 8))) is False

    def test_minmax_only_block_counts_as_tunable(self):
        import torch.nn as nn

        from auto_round.algorithms.quantization.sign_round.data_parallel import block_has_tuning_entries

        class MinmaxOnly(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.params = {"wmax": nn.Parameter(torch.ones(1))}

        # full-mirror can still tune minmax params
        assert block_has_tuning_entries(torch.nn.Sequential(MinmaxOnly())) is True

    def test_round_block_counts(self):
        import torch.nn as nn

        from auto_round.algorithms.quantization.sign_round.data_parallel import block_has_tuning_entries

        class RoundHolder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.params = {"v": nn.Parameter(torch.ones(4))}

        assert block_has_tuning_entries(torch.nn.Sequential(RoundHolder())) is True


@pytest.fixture()
def _autoround_log_propagate():
    """Temporarily enable propagation on the ``autoround`` logger so pytest's
    caplog fixture (handler at the root logger) can capture warnings; the
    logger is configured with propagate=False in production."""
    import logging

    logger = logging.getLogger("autoround")
    original = logger.propagate
    logger.propagate = True
    yield
    logger.propagate = original


class TestShardedRtnSearch:
    """iters=0 OptRTN: per-layer searches shard round-robin across the DDP devices."""

    def _fake_quantizer(self):
        from types import SimpleNamespace

        from auto_round.algorithms.quantization.rtn.quantizer import OptimizedRTNQuantizer

        q = OptimizedRTNQuantizer.__new__(OptimizedRTNQuantizer)
        q._resolved_ddp_plan = SimpleNamespace(world=2, devices=[torch.device("cpu", 0), torch.device("cpu", 1)])
        q._search_calls = []
        # q.model: property without setter on a bare instance -> getattr
        # default in _shard_rtn_searches treats it as "no global model"

        def _core(layer, disable_opt_rtn=None, tuning_device=None):
            q._search_calls.append((layer.global_name, tuning_device))
            return layer  # identity "quantized" layer

        q._quantize_layer_core = _core
        return q

    def _block(self, n=4):
        class _L(torch.nn.Linear):
            def __init__(self, i):
                super().__init__(4, 4)
                self.global_name = f"model.test.layer{i}"
                self.to_quantized = True  # check_to_quantized marker
                self.bits = 4

        class _B(torch.nn.Module):
            def __init__(self):
                super().__init__()
                for i in range(n):
                    setattr(self, f"layer{i}", _L(i))

        return _B()

    def test_sharded_round_robin(self, monkeypatch):
        from auto_round.algorithms.quantization.rtn import quantizer as rtn_q

        q = self._fake_quantizer()
        block = self._block(4)
        rtn_q._shard_rtn_searches(q, block)
        # all layers searched, round-robin over the two plan devices
        assert len(q._search_calls) == 4
        devs = [d for _, d in q._search_calls]
        assert devs[0] == devs[2] and devs[1] == devs[3] and devs[0] != devs[1]
        # results written back through set_module semantics (block modules replaced)
        assert all(hasattr(m, "global_name") for _n, m in block.named_modules() if hasattr(m, "global_name"))

    def test_world1_keeps_serial_loop(self, monkeypatch):
        from auto_round.algorithms.quantization.rtn import quantizer as rtn_q

        q = self._fake_quantizer()
        q._resolved_ddp_plan.world = 1
        block = self._block(2)
        rtn_q._shard_rtn_searches(q, block)
        # serial path: single shared device for every layer
        assert {d for _, d in q._search_calls} == {
            rtn_q.device_manager.device if hasattr(rtn_q, "device_manager") else q._search_calls[0][1]
        }

    def test_no_plan_is_serial(self):
        from auto_round.algorithms.quantization.rtn import quantizer as rtn_q

        q = self._fake_quantizer()
        q._resolved_ddp_plan = None
        block = self._block(2)
        rtn_q._shard_rtn_searches(q, block)
        assert len(q._search_calls) == 2


class TestResolverRtnSafe:
    """The resolver must not assume SignRound-family quantizer methods."""

    def test_quantizer_without_get_scaler_resolves(self, monkeypatch):
        from types import SimpleNamespace

        import torch

        from auto_round.algorithms.quantization.sign_round.data_parallel import resolve_tune_ddp_plan_

        monkeypatch.setenv("AR_TUNE_DDP_WORLD", "2")
        q = SimpleNamespace(iters=0, gradient_accumulate_steps=1, enable_lfq=False, _resolved_ddp_plan=None)
        # no _get_scaler, no calibration_context: must NOT AttributeError
        with pytest.raises(RuntimeError, match="not CUDA"):
            resolve_tune_ddp_plan_(q, torch.nn.Sequential(torch.nn.Linear(4, 4)), [torch.zeros(1)], None, "cpu")


class TestCollectionShardingFailVisible:
    """_ddp_collection_devices must not swallow the resolver's requirement error."""

    def _composer(self):
        from auto_round.algorithms.composer import AlgorithmComposer

        composer = AlgorithmComposer.__new__(AlgorithmComposer)
        composer.block_forward = type("R", (), {"output_config": ["hidden_states"]})()
        composer.block_quantizer = object()  # opaque; the resolver is monkeypatched
        return composer

    def test_requirement_error_propagates(self, monkeypatch):
        import torch

        import auto_round.algorithms.quantization.sign_round.data_parallel as dp

        def _raise(quantizer, block, fp_inputs, fp_outputs, home, world=None, log=True):
            raise RuntimeError("parallel tuning with world=4 is ineligible: no mirror device")

        monkeypatch.setattr(dp, "resolve_tune_ddp_plan_", _raise)
        block = torch.nn.Sequential(torch.nn.Linear(4, 4))
        with pytest.raises(RuntimeError, match="ineligible"):
            self._composer()._ddp_collection_devices(block, [torch.zeros(1)])

    def test_reachability_error_warns_and_declines(self, monkeypatch, caplog, _autoround_log_propagate):
        import torch

        import auto_round.algorithms.quantization.sign_round.data_parallel as dp

        def _raise(quantizer, block, fp_inputs, fp_outputs, home, world=None, log=True):
            raise OSError("boom")

        monkeypatch.setattr(dp, "resolve_tune_ddp_plan_", _raise)
        block = torch.nn.Sequential(torch.nn.Linear(4, 4))
        with caplog.at_level("WARNING"):
            out = self._composer()._ddp_collection_devices(block, [torch.zeros(1)])
        assert out is None
        assert any("resolver unreachable" in r.message for r in caplog.records)


import unittest  # noqa: E402
from types import SimpleNamespace  # noqa: E402


class TestShardedWrapSearch(unittest.TestCase):
    """Mirror-sharded wrap-time init-scale searches (mirrors-first pattern).

    Contract exercised: wrappers with ``_init_search_deferred`` set run their
    init-scale search once on exactly one replica (round-robin); the result is
    broadcast to the same-named wrappers on every replica; every replica then
    finalizes (compiles its own quant func) so tuning starts from identical
    state on all copies.
    """

    def _fake_wrapper(self, name, seed):
        import torch as _t

        class _W(_t.nn.Module):
            def __init__(self):
                super().__init__()
                self.name = name
                self.weight = _t.nn.Parameter(_t.ones(2, 2) * seed)
                self.init_scale = None
                self._init_search_deferred = True
                self.searched = 0
                self.compiled = 0

            def run_deferred_init_search(self):
                # deterministic given (weight, imatrix): init_scale = seed
                self.init_scale = float(self.weight[0, 0].item())
                self.searched += 1

            def _finalize_deferred_init(self, val=None):
                if self.init_scale is None:
                    self.init_scale = val
                self.compiled += 1
                self._init_search_deferred = False

        return _W()

    def _fake_group(self, block, world):
        import torch as _t

        # real ReplicaGroup.replicas = [home block] + mirrors -- mirror that
        replicas = [block]
        for _i in range(1, world):
            rep = _t.nn.Module()
            for n, m in block.named_modules():
                if hasattr(m, "run_deferred_init_search"):
                    w = self._fake_wrapper(n.split(".")[-1], float(m.weight[0, 0].item()))
                    parts = n.split(".")
                    parent = rep
                    for p_ in parts[:-1]:
                        if not hasattr(parent, p_):
                            parent.add_module(p_, _t.nn.Module())
                        parent = getattr(parent, p_)
                    parent.add_module(parts[-1], w)
            replicas.append(rep)
        plan = SimpleNamespace(world=world, devices=[_t.device("cpu", i) for i in range(world)])
        return SimpleNamespace(replicas=replicas, plan=plan)

    def _block(self, seeds):
        import torch as _t

        block = _t.nn.Module()
        for i, seed in enumerate(seeds):
            block.add_module(f"l{i}", self._fake_wrapper(f"l{i}", seed))
        return block

    def _wrappers(self, mod):
        return [m for _, m in mod.named_modules() if hasattr(m, "run_deferred_init_search")]

    def test_sharded_round_robin_and_broadcast(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import run_deferred_wrap_searches

        seeds = [1.0, 2.0, 3.0, 4.0]
        block = self._block(seeds)
        group = self._fake_group(block, 2)
        run_deferred_wrap_searches(block, group)
        # home block filled with deterministic values
        self.assertEqual([w.init_scale for w in self._wrappers(block)], seeds)
        # every replica broadcast-complete and finalized
        for rep in group.replicas:
            self.assertEqual([w.init_scale for w in self._wrappers(rep)], seeds)
            for w in self._wrappers(rep):
                self.assertEqual(w.compiled, 1)
                self.assertFalse(w._init_search_deferred)
        # round-robin: each name searched exactly once across the group
        total_searches = sum(w.searched for rep in group.replicas for w in self._wrappers(rep))
        self.assertEqual(total_searches, len(seeds))

    def test_serial_fallback_fills_all(self):
        from auto_round.algorithms.quantization.sign_round.data_parallel import run_deferred_wrap_searches

        block = self._block([10.0, 11.0, 12.0])
        run_deferred_wrap_searches(block, None)
        self.assertEqual([w.init_scale for w in self._wrappers(block)], [10.0, 11.0, 12.0])
        self.assertTrue(all(w.compiled == 1 for w in self._wrappers(block)))

    def test_no_deferred_wrappers_is_noop(self):
        import torch as _t

        from auto_round.algorithms.quantization.sign_round.data_parallel import run_deferred_wrap_searches

        block = _t.nn.Module()
        block.add_module("plain", _t.nn.Linear(2, 2))
        group = self._fake_group(block, 2)
        run_deferred_wrap_searches(block, group)  # must not raise

    def _v2_layer(self):
        import torch as _t

        layer = _t.nn.Linear(8, 8, bias=False)
        layer.bits = 4
        layer.group_size = -1
        layer.data_type = "int"
        layer.sym = True
        layer.act_bits = 16
        layer.act_data_type = "int"
        layer.act_dynamic = True
        layer.scale_dtype = _t.float32
        return layer

    def _patch_v2(self):
        import torch as _t

        import auto_round.algorithms.quantization.sign_roundv2.quantizer as v2q

        calls = []

        def _fake_search(weight_reshape, data_type, bits, imatrix, thresh):
            calls.append(1)
            return _t.tensor([0.5])

        orig = (v2q.search_optimized_init_scale, v2q.get_optimized_quant_func)
        v2q.search_optimized_init_scale = _fake_search
        v2q.get_optimized_quant_func = lambda dt: (lambda *a, **k: None)
        return v2q, calls, orig

    def test_v2_wrapper_honors_defer(self):
        v2q, calls, orig = self._patch_v2()
        try:
            w = v2q.SignRoundOptimizedWrapperLinear(
                self._v2_layer(), enable_torch_compile=False, device="cpu", defer_init_search=True
            )
            self.assertEqual(len(calls), 0)
            self.assertTrue(w._init_search_deferred)
            self.assertIsNotNone(w.weight_quant_func)
            w.run_deferred_init_search()
            self.assertEqual(len(calls), 1)
            self.assertIsNotNone(w.init_scale)
            w._finalize_deferred_init()
            self.assertFalse(w._init_search_deferred)
        finally:
            v2q.search_optimized_init_scale, v2q.get_optimized_quant_func = orig

    def test_v2_finalize_without_search_takes_broadcast(self):
        import torch as _t

        v2q, _calls, orig = self._patch_v2()
        try:
            w = v2q.SignRoundOptimizedWrapperLinear(
                self._v2_layer(), enable_torch_compile=False, device="cpu", defer_init_search=True
            )
            self.assertIsNone(w.init_scale)  # deferred: present but unset until search/finalize
            w._finalize_deferred_init(_t.tensor([0.25]))
            self.assertIsNotNone(w.init_scale)  # broadcast value applied
            self.assertFalse(w._init_search_deferred)
        finally:
            v2q.search_optimized_init_scale, v2q.get_optimized_quant_func = orig

    def test_undeferred_v2_wrapper_unchanged(self):
        v2q, calls, orig = self._patch_v2()
        try:
            v2q.SignRoundOptimizedWrapperLinear(self._v2_layer(), enable_torch_compile=False, device="cpu")
            self.assertEqual(len(calls), 1)  # serial path unchanged: search at wrap
        finally:
            v2q.search_optimized_init_scale, v2q.get_optimized_quant_func = orig

    def test_v2_deferred_requires_supported_dtype(self):
        v2q, _calls, orig = self._patch_v2()
        try:
            v2q.get_optimized_quant_func = lambda dt: None  # unsupported data_type
            with self.assertRaises(ValueError):
                v2q.SignRoundOptimizedWrapperLinear(
                    self._v2_layer(), enable_torch_compile=False, device="cpu", defer_init_search=True
                )
        finally:
            v2q.search_optimized_init_scale, v2q.get_optimized_quant_func = orig
