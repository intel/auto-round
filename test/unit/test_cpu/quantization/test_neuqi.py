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
from types import SimpleNamespace

import pytest
import torch

from auto_round.data_type.int import quant_tensor_asym
from auto_round.data_type.neuqi import _log_search_engaged, neuqi_search_scale_zero, quant_tensor_opt_rtn_asym
from auto_round.data_type.register import QUANT_FUNC_WITH_DTYPE
from auto_round.data_type.utils import get_quant_func


def _heavy_tailed_data(n_groups=64, group_size=128, seed=0):
    gen = torch.Generator().manual_seed(seed)
    data = torch.randn(n_groups, group_size, generator=gen)
    # inject a few large outliers per group to make min/max suboptimal
    data[:, :5] *= 120.0
    return data


def _mse(x, y):
    return ((x.float() - y.float()) ** 2).mean().item()


def _oracle_loss_fp64(data, qw, scale, zp, maxq):
    """fp64 loss of one group under fixed (scale, zp) -- the tie-break oracle."""
    r = torch.round(data.double() / scale.double()).clamp_(min=-maxq - 1, max=2 * maxq + 1)
    q = (r + zp.double()).clamp_(0, maxq)
    deq = scale.double() * (q - zp.double())
    loss = (deq - data.double()) ** 2
    if qw is not None:
        loss = loss * qw.double()
    return loss.sum(-1)


def _assert_launch_latches(monkeypatch, N, attempt, impl_attr, checked_attr, broken_attr, gates, tail_args):
    """A raising kernel/compile launch must return None, latch its broken flag,
    and never retry (all three latch sites share this contract)."""
    calls = {"n": 0}

    def _boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("launch failed")

    monkeypatch.setattr(N, impl_attr, _boom)
    monkeypatch.setattr(N, checked_attr, True)
    for gate_name, gate_ret in gates.items():
        monkeypatch.setattr(N, gate_name, lambda *a, _r=gate_ret, **k: _r)
    fake = SimpleNamespace(is_cuda=True, device=torch.device("cuda"))
    out = getattr(N, attempt)(fake, *tail_args)
    assert out is None
    assert getattr(N, broken_attr)
    out2 = getattr(N, attempt)(fake, *tail_args)
    assert out2 is None and calls["n"] == 1  # latched: no second launch


def _reset_neuqi_state(defaults):
    """Reset neuqi module latch state (call before AND after each test)."""
    import auto_round.data_type.neuqi as N

    for attr, default in defaults.items():
        if hasattr(N, attr):
            setattr(N, attr, default)


class TestNeuqiAsymSearch:
    def test_search_output_contract(self):
        """Validity/integrality/shape contract of the joint search outputs."""
        import auto_round.data_type.neuqi as N

        torch.manual_seed(5)
        data = torch.randn(512, 64) * 0.05
        qw = torch.rand(512, 64) + 0.1
        scale, zp = N.neuqi_search_scale_zero(data, bits=4, qw=qw, coarse_n=4, fine_n=3)
        assert scale.shape == (512, 1) and zp.shape == (512, 1)
        assert torch.all(zp == zp.round())
        assert (zp >= 0).all() and (zp <= 15).all()
        assert torch.all(scale > 0)

        data = _heavy_tailed_data(n_groups=32)
        scale, zp = neuqi_search_scale_zero(data, bits=4)
        assert torch.all(zp == zp.round())
        assert torch.all(zp >= 0) and torch.all(zp <= 15)
        assert torch.all(scale > 0)

        gen = torch.Generator().manual_seed(2)
        for shape in [(4, 33, 128), (17, 65), (3, 5, 7, 11)]:
            tensor = torch.randn(*shape, generator=gen)
            qdq, scale, zp = quant_tensor_opt_rtn_asym(tensor.clone(), bits=4, group_size=32)
            assert qdq.shape == tensor.shape

    def test_beats_minmax_on_heavy_tailed_data(self):
        data = _heavy_tailed_data()
        bits, group_size = 4, 128

        minmax_qdq, _, _ = quant_tensor_asym(data.clone(), bits=bits, group_size=group_size)
        neuqi_qdq, scale, zp = quant_tensor_opt_rtn_asym(data.clone(), bits=bits, group_size=group_size)

        assert _mse(neuqi_qdq, data) < _mse(minmax_qdq, data)

    def test_close_to_dense_exhaustive_reference(self):
        """On a small case the search must approach a dense exhaustive grid."""
        gen = torch.Generator().manual_seed(3)
        n, g, bits = 8, 16, 2
        maxq = 2**bits - 1
        data = torch.randn(n, g, generator=gen)
        data[:, 0] *= 50.0

        scale, zp = neuqi_search_scale_zero(data, bits=bits)
        r = torch.round(data / scale)
        q = (r + zp).clamp(0, maxq)
        our_loss = (((scale * (q - zp)) - data) ** 2).sum().item()

        # brute force: dense additive scale grid per group x all integer zero points
        best = torch.full((n,), float("inf"))
        wmin = torch.clamp(data.min(-1).values, max=0)
        wmax = torch.clamp(data.max(-1).values, min=0)
        s0 = ((wmax - wmin) / maxq).clamp(min=1e-5)
        for i in range(401):
            frac = 0.05 + (1.0 - 0.05) * i / 400
            s = (s0 * frac).unsqueeze(-1)
            r = torch.round(data / s)
            for z in range(maxq + 1):
                q = (r + z).clamp(0, maxq)
                loss = ((s * (q - z) - data) ** 2).sum(-1)
                best = torch.minimum(best, loss)
        exhaustive_loss = best.sum().item()
        # within 1% of the (much denser) exhaustive grid
        assert our_loss <= exhaustive_loss * 1.01 + 1e-9

    def test_weighting_changes_solution(self):
        """A per-element weight that de-emphasizes outliers must pull the scale down."""
        gen = torch.Generator().manual_seed(1)
        data = torch.randn(16, 64, generator=gen)
        data[:, :4] *= 100.0

        qw = torch.ones_like(data)
        qw[:, :4] = 1e-4  # ignore outliers

        # grid pinned to the measured 64/32: the scale comparison below is a
        # heuristic (only loss dominance is a hard invariant) and finer grids
        # shift near-tie selections between the two runs
        scale_plain, _ = neuqi_search_scale_zero(data, bits=4, coarse_n=64, fine_n=32)
        scale_weighted, _ = neuqi_search_scale_zero(data, bits=4, qw=qw, coarse_n=64, fine_n=32)
        # ignoring outliers must never increase the scale and must shrink it somewhere
        assert torch.all(scale_weighted <= scale_plain + 1e-6)
        assert torch.any(scale_weighted < scale_plain)

        def weighted_loss(scale):
            maxq = 15
            r = torch.round(data / scale).clamp(-maxq - 1, 2 * maxq + 1)
            losses = []
            for z in range(maxq + 1):
                q = (r + z).clamp(0, maxq)
                losses.append((((scale * (q - z) - data) ** 2) * qw).sum(-1))
            return torch.stack(losses, -1).min(-1).values.sum()

        assert weighted_loss(scale_weighted) < weighted_loss(scale_plain)

    def test_constant_group_is_exact(self):
        data = torch.full((4, 128), 0.25)
        qdq, scale, zp = quant_tensor_opt_rtn_asym(data.clone(), bits=4, group_size=128)
        # the fp16 scale cast limits exactness to one quantization step (0.25 / 15)
        assert (qdq - data).abs().max() <= 0.25 / 15
        assert torch.all(scale > 0)


class TestNeuqiAsymIntegration:
    """Asym joint search through the quant-func API: output validity across
    bit widths and imatrix weighting that steers the solution."""

    def test_output_validity_and_imatrix_weighting(self):
        data = _heavy_tailed_data(n_groups=32)
        imatrix = torch.rand(128) + 0.1

        qdq_plain, _, _ = quant_tensor_opt_rtn_asym(data.clone(), bits=4, group_size=128)
        qdq_im, _, _ = quant_tensor_opt_rtn_asym(data.clone(), bits=4, group_size=128, imatrix=imatrix)

        qw = imatrix.expand(32, 128)
        weighted = lambda q: (((q - data) ** 2) * qw).sum().item()  # noqa: E731
        assert weighted(qdq_im) < weighted(qdq_plain)

        for bits in (2, 3, 4, 8):
            d = _heavy_tailed_data(n_groups=8, group_size=64, seed=bits)
            qdq, scale, zp = quant_tensor_opt_rtn_asym(d.clone(), bits=bits, group_size=64)
            maxq = 2**bits - 1
            assert torch.all(zp >= 0) and torch.all(zp <= maxq)
            assert torch.all(scale > 0)
            assert qdq.shape == d.shape


class TestNeuqiAsymFusedSweep:
    """The torch.compile-fused zero-point sweep must reproduce the eager reference.

    ``AR_NEUQI_BACKEND=compile`` routes every chunk through the fused expression.
    Selections must match the eager sweep; when compilation is unavailable in the
    environment (e.g. no host C++ compiler for Inductor) the dispatch must fall
    back to the eager expression and still return correct results."""

    def _search_with_backend(self, monkeypatch, backend, data, bits, qw, **kwargs):
        import auto_round.data_type.neuqi as N
        from auto_round import envs

        monkeypatch.setenv("AR_NEUQI_BACKEND", backend)
        monkeypatch.setattr(N, "_fused_zp_fns", {})
        monkeypatch.setattr(N, "_fused_zp_broken", False)
        scale, zp = N.neuqi_search_scale_zero(data.clone(), bits, qw=qw.clone() if qw is not None else None, **kwargs)
        return scale, zp, N

    @pytest.mark.enable_torch_compile
    def test_fused_matches_eager(self, monkeypatch):
        """One setup, the structural combo set: bits=2 (smallest zp grid),
        bits=4 (shipping default), bits=8 (zp grid of 256, largest sweep
        tensors), and one weighted arm at the extreme -- the qw multiply is
        size-independent, so a single weighted combo covers it. Selections
        must match the eager sweep; a flip is acceptable only between
        fp64-tied candidates."""
        for bits, weighted in ((2, False), (4, False), (8, False), (8, True)):
            gen = torch.Generator().manual_seed(bits * 10 + int(weighted))
            n, g = 128, 64
            data = torch.randn(n, g, generator=gen)
            data[:, :5] *= 120.0  # heavy tails: near-tie zero points are common here
            qw = torch.rand(n, g, generator=gen) + 0.1 if weighted else None

            scale_e, zp_e, _ = self._search_with_backend(monkeypatch, "eager", data, bits, qw, coarse_n=8, fine_n=4)
            scale_f, zp_f, N = self._search_with_backend(monkeypatch, "compile", data, bits, qw, coarse_n=8, fine_n=4)
            if N._fused_zp_broken or not N._fused_zp_fns:
                pytest.skip("torch.compile/inductor unavailable in this environment")

            mismatch = (zp_e != zp_f).logical_or(scale_e != scale_f).nonzero().flatten()
            for i_ in mismatch.tolist():
                loss_e = _oracle_loss_fp64(
                    data[i_], qw[i_] if qw is not None else None, scale_e[i_], zp_e[i_], 2**bits - 1
                )
                loss_f = _oracle_loss_fp64(
                    data[i_], qw[i_] if qw is not None else None, scale_f[i_], zp_f[i_], 2**bits - 1
                )
                assert (loss_f - loss_e).abs().item() <= 1e-6 * loss_e.abs().item()

    def test_last_layout_matches_mid_reference(self, monkeypatch):
        """The CUDA layout (group axis last) must match the reference layout's
        selections up to fp64 ties -- it runs on CUDA, its numerics are validated
        here on CPU."""
        import auto_round.data_type.neuqi as N
        from auto_round import envs

        gen = torch.Generator().manual_seed(31)
        data = torch.randn(128, 64, generator=gen)
        data[:, :4] *= 90.0
        qw = torch.rand(128, 64, generator=gen) + 0.1

        monkeypatch.setattr(N, "_fused_zp_fns", {})
        monkeypatch.setattr(N, "_fused_zp_broken", False)
        monkeypatch.setenv("AR_NEUQI_BACKEND", "eager")
        monkeypatch.setattr(N, "_zp_expr_for", lambda dt: N._zp_expr_mid)
        scale_m, zp_m = N.neuqi_search_scale_zero(data.clone(), bits=4, qw=qw, coarse_n=16, fine_n=4)

        monkeypatch.setenv("AR_NEUQI_BACKEND", "compile")
        monkeypatch.setattr(N, "_zp_expr_for", lambda dt: N._zp_expr_last)
        scale_l, zp_l = N.neuqi_search_scale_zero(data.clone(), bits=4, qw=qw, coarse_n=16, fine_n=4)

        if N._fused_zp_broken or not N._fused_zp_fns:
            pytest.skip("torch.compile/inductor unavailable in this environment")

        mismatch = (zp_m != zp_l).logical_or(scale_m != scale_l).nonzero().flatten()
        maxq = 15
        for i in mismatch.tolist():
            loss_m = _oracle_loss_fp64(data[i], qw[i], scale_m[i], zp_m[i], maxq)
            loss_l = _oracle_loss_fp64(data[i], qw[i], scale_l[i], zp_l[i], maxq)
            assert (loss_l - loss_m).abs().item() <= 1e-6 * loss_m.abs().item()

    def test_compile_failure_falls_back_to_eager(self, monkeypatch):
        """A broken host toolchain must degrade to the eager sweep, not crash."""
        import auto_round.data_type.neuqi as N
        from auto_round import envs

        data = _heavy_tailed_data(n_groups=16, seed=11)

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated inductor failure")

        monkeypatch.setenv("AR_NEUQI_BACKEND", "compile")
        monkeypatch.setattr(N, "_get_fused_zp_fn", _boom)
        monkeypatch.setattr(N, "_fused_zp_fns", {})
        monkeypatch.setattr(N, "_fused_zp_broken", False)

        scale_f, zp_f = N.neuqi_search_scale_zero(data.clone(), bits=4, coarse_n=4, fine_n=2)
        assert N._fused_zp_broken is True  # permanently latched after the first failure
        monkeypatch.setattr(N, "_fused_zp_broken", False)
        monkeypatch.setenv("AR_NEUQI_BACKEND", "eager")
        scale_e, zp_e = N.neuqi_search_scale_zero(data.clone(), bits=4, coarse_n=4, fine_n=2)
        torch.testing.assert_close(scale_f, scale_e)
        torch.testing.assert_close(zp_f, zp_e)


class TestNeuqiAsymBatchedSweep:
    """The all-candidates batched sweep must reproduce the per-candidate search.

    The batched sweep evaluates every coarse/fine candidate in one fused
    call per group chunk ([C, K, Z, g], min over z fused in-kernel, then first-min
    over candidates). Selections must match the per-candidate sweep; ties may
    resolve differently only within fp64-oracle tolerance."""

    def _search(self, monkeypatch, backend, batch, data, bits, qw, **kwargs):
        import auto_round.data_type.neuqi as N
        from auto_round import envs

        monkeypatch.setenv("AR_NEUQI_BACKEND", backend)
        monkeypatch.setattr(N, "_zp_batch_wanted", lambda dt: batch == "on")
        monkeypatch.setattr(N, "_fused_zp_fns", {})
        monkeypatch.setattr(N, "_fused_zp_broken", False)
        monkeypatch.setattr(N, "_batched_broken", False)
        scale, zp = N.neuqi_search_scale_zero(data.clone(), bits, qw=qw.clone() if qw is not None else None, **kwargs)
        return scale, zp, N

    @pytest.mark.enable_torch_compile
    def test_batched_matches_per_candidate(self, monkeypatch):
        """One setup, the structural combo set (see test_fused_matches_eager):
        batched selections must match the per-candidate sweep up to
        fp64-oracle ties."""
        for bits, weighted in ((2, False), (4, False), (8, False), (8, True)):
            gen = torch.Generator().manual_seed(bits * 7 + int(weighted))
            n, g = 128, 64
            data = torch.randn(n, g, generator=gen)
            data[:, :4] *= 100.0
            qw = torch.rand(n, g, generator=gen) + 0.1 if weighted else None

            scale_ref, zp_ref, _ = self._search(monkeypatch, "eager", "off", data, bits, qw, coarse_n=8, fine_n=3)
            scale_b, zp_b, N = self._search(monkeypatch, "compile", "on", data, bits, qw, coarse_n=8, fine_n=3)
            if N._batched_broken or N._zp_expr_batched not in N._fused_zp_fns:
                pytest.skip("torch.compile/inductor unavailable in this environment")

            mismatch = (zp_ref != zp_b).logical_or(scale_ref != scale_b).nonzero().flatten()
            maxq = 2**bits - 1
            for i_ in mismatch.tolist():
                loss_ref = _oracle_loss_fp64(
                    data[i_], qw[i_] if qw is not None else None, scale_ref[i_], zp_ref[i_], maxq
                )
                loss_b = _oracle_loss_fp64(data[i_], qw[i_] if qw is not None else None, scale_b[i_], zp_b[i_], maxq)
                assert (loss_b - loss_ref).abs().item() <= 1e-6 * loss_ref.abs().item()

    def test_batch_failure_falls_back_to_per_candidate(self, monkeypatch):
        """A batched-call failure must degrade to the per-candidate sweep, not crash."""
        import auto_round.data_type.neuqi as N
        from auto_round import envs

        data = _heavy_tailed_data(n_groups=24, group_size=64, seed=17)

        monkeypatch.setenv("AR_NEUQI_BACKEND", "compile")
        monkeypatch.setattr(N, "_zp_batch_wanted", lambda dt: True)
        monkeypatch.setattr(N, "_fused_zp_fns", {})
        monkeypatch.setattr(N, "_fused_zp_broken", False)
        monkeypatch.setattr(N, "_batched_broken", False)
        monkeypatch.setattr(N, "_eval_batched_chunk", lambda *a, **k: None)  # simulate failure

        scale_f, zp_f = N.neuqi_search_scale_zero(data.clone(), bits=4, coarse_n=6, fine_n=2)

        # reference: per-candidate sweep (batching off never touches the patched fn)
        monkeypatch.setattr(N, "_zp_batch_wanted", lambda dt: False)
        scale_ref, zp_ref = N.neuqi_search_scale_zero(data.clone(), bits=4, coarse_n=6, fine_n=2)
        torch.testing.assert_close(scale_f, scale_ref)
        torch.testing.assert_close(zp_f, zp_ref)


class TestNeuqiAsymTritonSweep:
    """The extension Triton sweep integrates behind the batched dispatch.

    Triton kernels cannot run on CPU hosts, so the plumbing (scale construction,
    first-min bookkeeping, latch fallbacks, policy) is tested with a mock that
    honors the kernel's contract; the kernel itself is exercised on CUDA by
    ``test/unit/test_cuda/quantization/test_neuqi_triton.py`` and by the
    self-contained GPU benchmark."""

    def _mock_sweep_from(self, monkeypatch, impl):
        import auto_round.data_type.neuqi as N

        monkeypatch.setattr(N, "_triton_checked", True)
        monkeypatch.setattr(N, "_triton_sweep", impl)
        monkeypatch.setattr(N, "_triton_broken", False)

    def test_triton_mock_matches_reference(self, monkeypatch):
        """Driving the batched passes through a contract-honoring mock must
        reproduce the per-candidate reference selections exactly."""
        import auto_round.data_type.neuqi as N
        from auto_round import envs

        def mock(data, qw, scales, maxq):  # same contract as neuqi_sweep_triton
            zp_grid = torch.arange(0, maxq + 1, device=data.device, dtype=torch.float32)
            return N._zp_expr_batched(data, qw, scales, zp_grid, maxq)

        self._mock_sweep_from(monkeypatch, mock)
        data = _heavy_tailed_data(n_groups=64, group_size=64, seed=23)
        qw = torch.rand(64, 64) + 0.1

        monkeypatch.setenv("AR_NEUQI_BACKEND", "triton")
        monkeypatch.setattr(N, "_zp_batch_wanted", lambda dt: True)
        monkeypatch.setattr(N, "_fused_zp_fns", {})
        monkeypatch.setattr(N, "_fused_zp_broken", False)
        monkeypatch.setattr(N, "_batched_broken", False)
        scale_t, zp_t = N.neuqi_search_scale_zero(data.clone(), bits=4, qw=qw, coarse_n=8, fine_n=3)

        monkeypatch.setenv("AR_NEUQI_BACKEND", "eager")
        monkeypatch.setattr(N, "_zp_batch_wanted", lambda dt: False)
        scale_ref, zp_ref = N.neuqi_search_scale_zero(data.clone(), bits=4, qw=qw, coarse_n=8, fine_n=3)
        torch.testing.assert_close(scale_t, scale_ref)
        torch.testing.assert_close(zp_t, zp_ref)

    def test_triton_failure_latches_to_compile_path(self, monkeypatch):
        """A raising Triton sweep must latch off and finish via the torch paths."""
        import auto_round.data_type.neuqi as N
        from auto_round import envs

        def boom(data, qw, scales, maxq):
            raise RuntimeError("simulated triton failure")

        self._mock_sweep_from(monkeypatch, boom)
        data = _heavy_tailed_data(n_groups=32, group_size=64, seed=29)

        monkeypatch.setenv("AR_NEUQI_BACKEND", "triton")
        monkeypatch.setattr(N, "_zp_batch_wanted", lambda dt: True)
        monkeypatch.setattr(N, "_fused_zp_fns", {})
        monkeypatch.setattr(N, "_fused_zp_broken", False)
        monkeypatch.setattr(N, "_batched_broken", False)
        scale_f, zp_f = N.neuqi_search_scale_zero(data.clone(), bits=4, coarse_n=6, fine_n=2)
        assert N._triton_broken is True

        monkeypatch.setenv("AR_NEUQI_BACKEND", "eager")
        scale_ref, zp_ref = N.neuqi_search_scale_zero(data.clone(), bits=4, coarse_n=6, fine_n=2)
        torch.testing.assert_close(scale_f, scale_ref)
        torch.testing.assert_close(zp_f, zp_ref)

    def test_resolver_logs_engagement(self, monkeypatch):
        """A successful resolution announces itself: the Triton arm is silent
        otherwise, and live runs must be able to tell which backend served."""
        import sys
        import types

        import auto_round.data_type.neuqi as N

        stub = types.ModuleType("auto_round_extension.triton.neuqi_sweep")
        stub.neuqi_sweep_triton = object()
        monkeypatch.setitem(sys.modules, "auto_round_extension.triton.neuqi_sweep", stub)
        recorded = []
        monkeypatch.setattr(N, "logger", types.SimpleNamespace(info=recorded.append, debug=recorded.append))
        monkeypatch.setattr(N, "_triton_checked", False)
        monkeypatch.setattr(N, "_triton_sweep", None)
        assert N._triton_sweep_fn() is stub.neuqi_sweep_triton
        assert any("Triton zero-point sweep engaged" in str(r) for r in recorded)


class TestEnableNeuqiAPI:
    """enable_neuqi (RTNConfig/SignRoundConfig kwarg + --enable_neuqi CLI flag):
    opt-in gate for the NeUQI grid search on BOTH symmetry classes."""

    def test_dispatch_table(self):
        """Off: plain asym min/max + incumbent sym; opted in: joint search on asym."""
        from auto_round.data_type.int import quant_tensor_asym as plain_asym

        func, name = get_quant_func("int", 4, False, disable_opt_rtn=False, group_size=None, iters=0)
        assert "opt_rtn" not in name and name.endswith("_asym") and func is plain_asym
        func, name = get_quant_func("int", 4, False, disable_opt_rtn=False, group_size=None, iters=0, enable_neuqi=True)
        assert name == "opt_rtn_int_asym" and func is QUANT_FUNC_WITH_DTYPE["opt_rtn_int_asym"]
        _, sym_name = get_quant_func("int", 4, True, disable_opt_rtn=False, group_size=None, iters=0)
        assert sym_name == "opt_rtn_int_sym"

    def test_config_flag_semantics(self):
        """enable_neuqi defaults off and accepts True on both symmetry classes."""
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        assert RTNConfig(bits=4, group_size=32, sym=False).enable_neuqi is False
        on = RTNConfig(bits=4, group_size=32, sym=False, disable_opt_rtn=False, enable_neuqi=True)
        assert on.enable_neuqi is True
        sym_on = RTNConfig(bits=4, group_size=32, sym=True, disable_opt_rtn=False, enable_neuqi=True)
        assert sym_on.enable_neuqi is True  # engages the two-stage symmetric search

    def test_neuqi_with_opt_rtn_disabled_raises(self):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        with pytest.raises(ValueError, match="--enable_neuqi requires the optimized-RTN path"):
            RTNConfig(bits=4, group_size=32, sym=False, disable_opt_rtn=True, enable_neuqi=True)

    def test_default_is_off(self):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig

        cfg = RTNConfig(bits=4, group_size=32, sym=False)
        assert cfg.enable_neuqi is False


class TestNeuqiGridHelpers:
    """Shared coarse-grid helper invariants (both symmetry classes use _coarse_grid)."""

    def test_coarse_grid_values_and_freshness(self):
        """The eager coarse-grid helper must be numerically identical to the
        previous inline torch.logspace, and a cached grid is a cross-stream
        hazard in the expert fan-out (49aa6405: filled on the first caller's
        stream, read from other threads' streams) -- it must be recreated per
        call, never lru_cache'd."""
        import math

        import auto_round.data_type.neuqi as N

        for lo in (0.03125, 0.25):
            got = N._coarse_grid(lo, 64, torch.device("cpu"), torch.float32)
            want = torch.logspace(math.log10(lo), 0.0, 64, dtype=torch.float32)
            assert got.shape == want.shape
            assert torch.equal(got, want)

        args = (0.001, 8, torch.device("cpu"), torch.float32)
        a = N._coarse_grid(*args)
        b = N._coarse_grid(*args)
        assert a is not b, "grid must not be shared state across calls"
        assert not hasattr(N._coarse_grid, "cache_info"), "tensor-returning helper must not be lru_cache'd"
        torch.testing.assert_close(a, b)  # values stay deterministic across calls

    def test_coarse_grid_is_dynamo_opaque(self):
        """torch.logspace lowers to an fp64 libdevice.pow inside compiled
        regions; inductor fused it into the batched sweep's index math where
        Triton rejects fp64 ("unexpected type fp64") -- see the BPT worker
        crash. The grid must be built eagerly and enter traced regions as a
        plain tensor input (no logspace op in the traced graph)."""
        import torch._dynamo as dynamo

        from auto_round.data_type.neuqi import _coarse_grid

        assert getattr(_coarse_grid, "_torchdynamo_disable", False) is True

        def traced(x):
            return _coarse_grid(0.25, 8, x.device, torch.float32) * x

        ex = dynamo.explain(traced)(torch.ones(8))
        op_names = [getattr(op, "__name__", str(op)) for ops in ex.ops_per_graph for op in ops]
        assert not any("logspace" in n for n in op_names), f"logspace leaked into the traced graph: {op_names}"
        assert any("mul" in n for n in op_names), "the sweep math itself should stay traced"


class TestNeuqiBackendPolicy:
    """Env x device gate matrices for every backend selector (pure policy:
    no data, no kernels -- the selections themselves are covered by the
    sweep-parity classes)."""

    def test_env_device_matrix(self, monkeypatch):
        import auto_round.data_type.neuqi as N

        monkeypatch.setenv("AR_NEUQI_BACKEND", "auto")
        assert N._zp_wants_compile("cuda") is True
        assert N._zp_wants_compile("cpu") is False and N._zp_wants_compile("hpu") is False
        assert N._zp_batch_wanted("cuda") is True and N._zp_batch_wanted("cpu") is False
        assert N._zp_wants_triton("cuda") is True and N._zp_wants_triton("cpu") is False
        assert N._sym_shared_wants_triton("cuda") is True and N._sym_shared_wants_triton("cpu") is False
        monkeypatch.setenv("AR_NEUQI_BACKEND", "compile")
        assert N._zp_wants_compile("cpu") is True
        assert N._sym_shared_wants_triton("cuda") is False
        monkeypatch.setenv("AR_NEUQI_BACKEND", "eager")
        assert N._zp_wants_compile("cuda") is False
        assert N._zp_batch_wanted("cuda") is False  # batched needs the fused backend
        assert N._zp_wants_triton("cuda") is False
        assert N._sym_shared_wants_triton("cuda") is False
        monkeypatch.setenv("AR_NEUQI_BACKEND", "triton")
        assert N._zp_wants_triton("cpu") is True  # forced (availability still checked)
        assert N._sym_shared_wants_triton("cuda") is True
        # unrecognized values behave as the reference sweep
        monkeypatch.setenv("AR_NEUQI_BACKEND", "nonsense")
        assert N._zp_wants_compile("cuda") is False

    def test_layout_policy(self):
        import auto_round.data_type.neuqi as N

        # device-automatic: the CUDA Triton fusion shape on cuda, the
        # zero-point-last layout everywhere else
        assert N._zp_expr_for("cuda") is N._zp_expr_last
        assert N._zp_expr_for("cpu") is N._zp_expr_mid
        assert N._zp_expr_for("hpu") is N._zp_expr_mid

    def test_cpu_default_is_eager(self, monkeypatch):
        """AR_NEUQI_BACKEND defaults to auto: CPU tensors never trigger a compile."""
        import auto_round.data_type.neuqi as N
        from auto_round import envs

        monkeypatch.setenv("AR_NEUQI_BACKEND", "auto")
        monkeypatch.setattr(N, "_fused_zp_fns", {})
        monkeypatch.setattr(N, "_fused_zp_broken", False)
        N.neuqi_search_scale_zero(_heavy_tailed_data(n_groups=8, seed=13).clone(), bits=4, coarse_n=4, fine_n=2)
        assert not N._fused_zp_fns  # no compile attempt on CPU under auto


class TestNeuqiItersGuard:
    """--enable_neuqi with iters>0: the zero-shot search dispatch stays keyed
    to iters=0 (the tuning path resolves its plain init here); the search
    itself reaches SignRound as the wrapper's frozen initialization."""

    def test_iters_dispatch(self):
        from auto_round.data_type.utils import get_quant_func

        fn, name = get_quant_func("int", 4, False, disable_opt_rtn=False, group_size=128, iters=20, enable_neuqi=True)
        assert name == "int_asym"  # tuning path initializes plain; the anchor owns the search
        fn, name = get_quant_func("int", 4, False, disable_opt_rtn=False, group_size=128, iters=0, enable_neuqi=True)
        assert name.startswith("opt_rtn_int_asym")


class TestNeuqiSymSearch:
    """Two-stage symmetric scale search (zero point fixed at 0)."""

    def _skewed_data(self, n_groups=512, group_size=128, seed=0):
        gen = torch.Generator().manual_seed(seed)
        data = torch.randn(n_groups, group_size, generator=gen)
        data[:, :6] *= 60.0  # outliers make the clip search matter
        # half the groups skew negative so the mirrored clamp family is in play
        data[::2] *= -1.0
        return data

    def _sym_loss(self, data, qw, scale):
        """Literal loss of the shipped symmetric formula (signed scales allowed)."""
        iq = (data / scale).round().clamp(-8, 7)
        err = (scale * iq - data) ** 2
        if qw is not None:
            err = err * qw
        return err.sum(-1)

    def test_returns_signed_scales_with_magnitude_floor(self):
        from auto_round.data_type.neuqi import neuqi_search_scale_sym

        data = self._skewed_data()
        qw = torch.rand(*data.shape) + 0.05
        scale = neuqi_search_scale_sym(data, bits=4, qw=qw, coarse_n=32, fine_n=16)
        assert scale.shape == (data.shape[0], 1)
        assert (scale.abs() >= 1e-5).all()
        # both clamp conventions must actually win somewhere on skewed data
        assert (scale < 0).any() and (scale > 0).any()

    def test_mirrored_family_matches_literal_negative_scale(self):
        """The mirrored-clamp loss used inside the search must equal the literal
        negative-scale evaluation of the shipped formula (guards the sign of the
        mirrored error term)."""
        from auto_round.data_type.neuqi import _sym_loss_chunk

        torch.manual_seed(3)
        data = torch.randn(64, 96)
        qw = torch.rand(64, 96) + 0.1
        scales = torch.rand(64, 8) * 0.05 + 0.005  # [C, K] positive candidates
        loss, idx, mirror = _sym_loss_chunk(data, qw, scales, nmax=8)
        for c in range(data.shape[0]):
            k = idx[c].item()
            s = scales[c, k]
            signed = -s if mirror[c] else s
            ref = self._sym_loss(data[c : c + 1], qw[c : c + 1], torch.tensor([[signed]]))[0]
            assert torch.allclose(loss[c], ref, rtol=1e-4, atol=1e-3)

    def test_signed_search_beats_positive_only_bruteforce_on_skewed_groups(self):
        """Skewed groups: allowing the mirrored family must reach (near-)the signed
        brute-force optimum, strictly below the positive-only optimum."""
        from auto_round.data_type.neuqi import neuqi_search_scale_sym

        data = self._skewed_data(128, 96, seed=7)
        qw = torch.rand(*data.shape) + 0.05
        scale = neuqi_search_scale_sym(data, bits=4, qw=qw, coarse_n=64, fine_n=32)
        got = self._sym_loss(data, qw, scale)
        # dense positive-only reference over the same anchor
        s0 = data.abs().amax(dim=-1, keepdim=True) / 8
        fracs = torch.logspace(-0.9, 0.3, 2000)
        ref = torch.full_like(got, float("inf"))
        for f in fracs:
            l = self._sym_loss(data, qw, s0 * f)
            ref = torch.minimum(ref, l)
        # aggregate must win: the mirrored family is reachable, and the few
        # two-stage basin misses (coarse grid skips a narrow basin) stay small
        assert got.sum() <= ref.sum() * 1.001
        # and on genuinely skewed groups it must beat the positive-only optimum
        assert (got < ref - 1e-3).float().mean() > 0.25

    def test_covers_incumbent_uniform_grid(self):
        """Total weighted loss must not exceed search_scales (both searches see the
        same data, weights and signed-scale families)."""
        from auto_round.data_type.int import search_scales
        from auto_round.data_type.neuqi import neuqi_search_scale_sym

        data = self._skewed_data(512, 128, seed=11)
        qw = torch.rand(*data.shape) + 0.05
        s_neuqi = neuqi_search_scale_sym(data.clone(), bits=4, qw=qw.clone(), coarse_n=64, fine_n=32)
        s_old = search_scales(data.clone(), bits=4, qw=qw.clone())
        l_neuqi = self._sym_loss(data, qw, s_neuqi).sum().item()
        l_old = self._sym_loss(data, qw, s_old).sum().item()
        assert l_neuqi <= l_old * 1.001


class TestNeuqiSymSharedCoarse:
    """Shared-multiplier coarse pass: gating, latch-down, and driver math."""

    _DEFAULTS = {"_sym_shared_triton": None, "_sym_shared_checked": False, "_sym_shared_broken": False}

    @pytest.fixture(autouse=True)
    def _reset_shared_state(self):
        _reset_neuqi_state(self._DEFAULTS)
        yield
        _reset_neuqi_state(self._DEFAULTS)

    def test_attempt_latches_on_failure(self, monkeypatch):
        import auto_round.data_type.neuqi as N

        _assert_launch_latches(
            monkeypatch,
            N,
            attempt="_sym_shared_triton_attempt",
            impl_attr="_sym_shared_triton",
            checked_attr="_sym_shared_checked",
            broken_attr="_sym_shared_broken",
            gates={"_sym_shared_wants_triton": True},
            tail_args=(None, None, None, 7),
        )

    def test_cpu_tensor_never_latches(self, monkeypatch):
        import auto_round.data_type.neuqi as N

        monkeypatch.delenv("AR_NEUQI_BACKEND", raising=False)
        data = torch.randn(8, 16)
        out = N._sym_shared_triton_attempt(data, None, None, None, 7)
        assert out is None
        assert not N._sym_shared_broken  # gated before any launch

    def test_coarse_pass_parity_and_fallback(self, monkeypatch):
        """Driver math (dn normalization, s0^2 unnormalization, frac mapping) via an
        eager stand-in mirrors the per-candidate path; launch None falls back."""
        import auto_round.data_type.neuqi as N
        from auto_round.data_type.neuqi import _sym_loss_chunk

        torch.manual_seed(11)
        n, g, bits = 700, 128, 4
        nmax = int(2 ** (bits - 1))
        data = torch.randn(n, g)
        data[:, :3] *= 60.0
        qw = torch.rand(n, g) + 0.1
        s0 = (torch.abs(data).amax(dim=-1, keepdim=True) / nmax).clamp_(min=1e-5)
        import math

        coarse = torch.logspace(math.log10(0.25), math.log10(2.0), 96)

        def eager_shared(dn, q, fracs, invf, nm):
            # eager mirror of _sym_shared_kernel (normalized space)
            d = dn.unsqueeze(1)
            f = fracs.view(1, -1, 1)
            r = torch.round(d * invf.view(1, -1, 1))
            q_std = r.clamp(min=-nm, max=nm - 1)
            q_mir = r.clamp(min=-(nm - 1), max=nm)
            e_std = f * q_std - d
            e_mir = f * q_mir - d
            l_std = e_std * e_std
            l_mir = e_mir * e_mir
            if q is not None:
                l_std = l_std * q.unsqueeze(1)
                l_mir = l_mir * q.unsqueeze(1)
            l_std = l_std.sum(-1)
            l_mir = l_mir.sum(-1)
            mirror = l_mir < l_std
            loss = torch.where(mirror, l_mir, l_std)
            best_loss, best_idx = loss.min(dim=-1)
            return best_loss, best_idx, mirror.gather(1, best_idx.unsqueeze(1)).squeeze(1)

        # force small chunks so the loop runs multiple iterations
        monkeypatch.setattr(N, "_MAX_TMP_ELEMS", 2 * 96 * 128 * 4)
        out = N._sym_coarse_pass_shared(data, qw, s0, coarse, nmax, launch=eager_shared)
        assert out is not None
        loss_s, frac_s, mirror_s = out

        scales = s0 * coarse.view(1, -1)
        loss_r, idx_r, mirror_r = _sym_loss_chunk(data, qw, scales, nmax)
        torch.testing.assert_close(frac_s, coarse.index_select(0, idx_r), rtol=0, atol=0)
        torch.testing.assert_close(mirror_s, mirror_r, rtol=0, atol=0)
        torch.testing.assert_close(loss_s, loss_r, rtol=1e-4, atol=1e-5)

        # weighted arm agrees too
        out_w = N._sym_coarse_pass_shared(data, qw, s0, coarse, nmax, launch=eager_shared)
        assert out_w is not None

        # fallback: a launch that declines must produce None end to end
        assert N._sym_coarse_pass_shared(data, qw, s0, coarse, nmax, launch=lambda *a: None) is None

    def test_two_stage_core_uses_shared_launch(self, monkeypatch):
        """The two-stage core routes the coarse pass through the shared helper."""
        import auto_round.data_type.neuqi as N

        torch.manual_seed(12)
        data = torch.randn(64, 128)
        qw = torch.rand(64, 128) + 0.1

        def spy_launch(dn, q, fracs, invf, nm):
            spy_launch.called = True
            return None  # decline -> per-candidate fallback

        spy_launch.called = False
        monkeypatch.setattr(N, "_sym_coarse_pass_shared", lambda *a, **k: (None if not spy_launch.called else None))
        # direct wiring check: the core calls _sym_coarse_pass_shared at all
        real = N._sym_coarse_pass_shared

        def wrapper(data_, qw_, s0_, coarse_, nmax_):
            wrapper.called = True
            return real(data_, qw_, s0_, coarse_, nmax_)

        wrapper.called = False
        monkeypatch.setattr(N, "_sym_coarse_pass_shared", wrapper)
        N.neuqi_search_scale_sym(data, 4, qw=qw, coarse_n=32, fine_n=8)
        assert wrapper.called


class TestNeuqiGridEnv:
    """Grid resolution reaches both entries from the shared env pins. The
    unpinned default is backend-aware (wide only on the accelerated lanes --
    the eager sweep costs ~coarse x fine evaluations per group); env pins
    override it everywhere and invalid pins raise at the read sites."""

    @pytest.mark.parametrize("entry", ["asym", "sym"])
    def test_env_pins_and_defaults(self, monkeypatch, entry):
        import auto_round.data_type.neuqi as N

        for k in ("AR_NEUQI_COARSE", "AR_NEUQI_FINE"):
            monkeypatch.delenv(k, raising=False)
        seen = {}
        if entry == "asym":
            monkeypatch.setattr(N, "_log_search_engaged", lambda c, f: seen.update(coarse=c, fine=f))
            run = lambda: N.neuqi_search_scale_zero(torch.randn(4, 8), 4)  # noqa: E731
        else:

            def spy(data, qw, bits, coarse, fine_n, q_scale_thresh):
                seen.update(coarse=coarse.numel(), fine=fine_n)
                return torch.ones(data.shape[0], 1), torch.zeros(data.shape[0])

            monkeypatch.setattr(N, "_two_stage_sym_core", spy)
            run = lambda: N.neuqi_search_scale_sym(torch.randn(4, 8), 4)  # noqa: E731
        run()
        assert seen == {"coarse": 64, "fine": 32}  # eager-lane default (CPU tensors)
        monkeypatch.setenv("AR_NEUQI_COARSE", "96")
        monkeypatch.setenv("AR_NEUQI_FINE", "24")
        run()
        assert seen == {"coarse": 96, "fine": 24}  # pins override

    def test_default_grid_is_backend_aware(self, monkeypatch):
        import auto_round.data_type.neuqi as N

        monkeypatch.setenv("AR_NEUQI_BACKEND", "auto")
        assert N._default_grid("cuda") == (256, 64)  # accelerated lanes keep the wide grid
        assert N._default_grid("cpu") == (64, 32)  # eager lane narrows
        monkeypatch.setenv("AR_NEUQI_BACKEND", "eager")
        assert N._default_grid("cuda") == (64, 32)  # forced eager narrows even on CUDA
        monkeypatch.setenv("AR_NEUQI_BACKEND", "compile")
        assert N._default_grid("cpu") == (256, 64)  # forced compile keeps the wide grid

    @pytest.mark.parametrize("entry", ["asym", "sym"])
    def test_invalid_pins_raise(self, monkeypatch, entry):
        import auto_round.data_type.neuqi as N

        monkeypatch.setenv("AR_NEUQI_COARSE", "1")
        with pytest.raises(ValueError, match="too small"):
            if entry == "asym":
                N.neuqi_search_scale_zero(torch.randn(4, 8), 4)
            else:
                N.neuqi_search_scale_sym(torch.randn(4, 8), 4)
        monkeypatch.setenv("AR_NEUQI_COARSE", "8")
        monkeypatch.setenv("AR_NEUQI_FINE", "1")
        with pytest.raises(ValueError, match="too small"):
            if entry == "asym":
                N.neuqi_search_scale_zero(torch.randn(4, 8), 4)
            else:
                N.neuqi_search_scale_sym(torch.randn(4, 8), 4)


class TestNeuqiAsymSharedCoarse:
    """Shared-multiplier coarse pass (joint search): gating, latch, driver math."""

    _DEFAULTS = {"_neuqi_shared_sweep": None, "_neuqi_shared_checked": False, "_neuqi_shared_broken": False}

    @pytest.fixture(autouse=True)
    def _reset_shared_state(self):
        _reset_neuqi_state(self._DEFAULTS)
        yield
        _reset_neuqi_state(self._DEFAULTS)

    def test_attempt_respects_backend_and_batch_gates(self, monkeypatch):
        import auto_round.data_type.neuqi as N

        fake = SimpleNamespace(is_cuda=True, device=torch.device("cuda"))
        for wants, batched, should_call in ((False, True, False), (True, False, False), (True, True, True)):
            calls = {"n": 0}

            def _fn(*a, **k):
                calls["n"] += 1
                return "sentinel"

            monkeypatch.setattr(N, "_zp_wants_triton", lambda dt: wants)
            monkeypatch.setattr(N, "_zp_batch_wanted", lambda dt: batched)
            monkeypatch.setattr(N, "_neuqi_shared_sweep", _fn)
            monkeypatch.setattr(N, "_neuqi_shared_checked", True)
            out = N._neuqi_shared_attempt(fake, None, None, None, 15)
            if should_call:
                assert out == "sentinel" and calls["n"] == 1
            else:
                assert out is None and calls["n"] == 0

    def test_attempt_latches_on_failure(self, monkeypatch):
        import auto_round.data_type.neuqi as N

        _assert_launch_latches(
            monkeypatch,
            N,
            attempt="_neuqi_shared_attempt",
            impl_attr="_neuqi_shared_sweep",
            checked_attr="_neuqi_shared_checked",
            broken_attr="_neuqi_shared_broken",
            gates={"_zp_wants_triton": True, "_zp_batch_wanted": True},
            tail_args=(None, None, None, 15),
        )

    def test_cpu_tensor_never_latches(self):
        import auto_round.data_type.neuqi as N

        data = torch.randn(8, 16)
        out = N._zp_coarse_pass_shared(data, None, torch.full((8, 1), 0.1), torch.tensor([0.5, 1.0]), 15)
        assert out is None
        assert not N._neuqi_shared_broken  # gated before any launch

    def test_coarse_pass_parity_and_fallback(self, monkeypatch):
        """Driver math (dn normalization, s0^2 unnormalization, frac/zp mapping) via
        an eager stand-in mirrors the per-candidate reference; launch None falls back."""
        import auto_round.data_type.neuqi as N

        torch.manual_seed(13)
        n, g, bits = 600, 128, 4
        maxq = 2**bits - 1
        data = torch.randn(n, g)
        data[:, :3] *= 70.0
        qw = torch.rand(n, g) + 0.1
        wmin = torch.clamp(data.min(dim=-1).values, max=0).unsqueeze(-1)
        wmax = torch.clamp(data.max(dim=-1).values, min=0).unsqueeze(-1)
        s0 = ((wmax - wmin) / maxq).clamp_(min=1e-5)
        import math

        coarse = torch.logspace(math.log10(0.25), 0.0, 80)

        def eager_shared(dn, q, fracs, invf, mq):
            # eager mirror of _neuqi_shared_kernel (normalized space)
            d = dn.unsqueeze(-2).unsqueeze(-2)  # [C, 1, 1, g]
            f = fracs.view(1, -1, 1, 1)  # [1, K, 1, 1]
            zp = torch.arange(0, mq + 1, device=dn.device, dtype=dn.dtype).view(1, 1, -1, 1)
            r = torch.round(d * invf.view(1, -1, 1, 1)).clamp_(min=-mq - 1, max=2 * mq + 1)
            qd = (r + zp).clamp_(0, mq)
            err = f * (qd - zp) - d
            loss = err * err
            if q is not None:
                loss = loss * q.unsqueeze(-2).unsqueeze(-2)
            loss = loss.sum(dim=-1)  # [C, K, Z]
            best_z, zp_arg = loss.min(dim=-1)
            best, k_arg = best_z.min(dim=1)  # first-minimum over k
            zp_best = zp_arg.gather(1, k_arg.unsqueeze(1)).squeeze(1)
            return best, k_arg, zp_best

        # force small chunks so the loop runs multiple iterations
        monkeypatch.setattr(N, "_MAX_TMP_ELEMS", 262144 * 3)
        loss_s, frac_s, zp_s = N._zp_coarse_pass_shared(data, qw, s0, coarse, maxq, launch=eager_shared)

        # per-candidate reference over the identical grid
        scales = s0 * coarse.view(1, -1)  # [C, K]
        d = data.unsqueeze(-2).unsqueeze(-2)
        zp = torch.arange(0, maxq + 1).view(1, 1, -1, 1)
        r = torch.round(d / scales.unsqueeze(-1).unsqueeze(-1)).clamp_(min=-maxq - 1, max=2 * maxq + 1)
        qd = (r + zp).clamp_(0, maxq)
        err = scales.unsqueeze(-1).unsqueeze(-1) * (qd - zp) - d
        loss = err * err * qw.unsqueeze(-2).unsqueeze(-2)
        loss = loss.sum(dim=-1)  # [C, K, Z]
        best_z, zp_arg = loss.min(dim=-1)
        loss_ref, k_ref = best_z.min(dim=1)
        zp_ref = zp_arg.gather(1, k_ref.unsqueeze(1)).squeeze(1)

        torch.testing.assert_close(frac_s, coarse.index_select(0, k_ref), rtol=0, atol=0)
        torch.testing.assert_close(zp_s, zp_ref.to(torch.float32), rtol=0, atol=0)
        torch.testing.assert_close(loss_s, loss_ref, rtol=1e-4, atol=1e-4)

        # fallback: a launch that declines must produce None end to end
        assert N._zp_coarse_pass_shared(data, qw, s0, coarse, maxq, launch=lambda *a: None) is None

    def test_search_wires_shared_coarse(self, monkeypatch):
        """neuqi_search_scale_zero routes its coarse pass through the shared helper."""
        import auto_round.data_type.neuqi as N

        torch.manual_seed(14)
        data = torch.randn(64, 128)
        qw = torch.rand(64, 128) + 0.1

        def wrapper(data_, qw_, s0_, coarse_, maxq_):
            wrapper.called = True
            return None  # decline -> batched fallback still correct

        wrapper.called = False
        monkeypatch.setattr(N, "_zp_coarse_pass_shared", wrapper)
        N.neuqi_search_scale_zero(data, 4, qw=qw, coarse_n=32, fine_n=8)
        assert wrapper.called


class TestNeuqiSymTritonSweep:
    """Triton sym-search dispatch: gating, latch-down, and parity where CUDA exists."""

    @pytest.fixture(autouse=True)
    def _reset_backend_state(self):
        import auto_round.data_type.neuqi as N

        for attr in ("_sym_triton", "_sym_triton_checked", "_sym_triton_broken"):
            if hasattr(N, attr):
                setattr(N, attr, None if attr != "_sym_triton_broken" else False)
        N._sym_triton_checked = False
        N._sym_search_warmed.clear()
        N._sym_chunk_compiled = None
        N._sym_compile_failed = False
        yield
        import auto_round.data_type.neuqi as N2

        N2._sym_triton_broken = False
        N2._sym_triton_checked = False
        N2._sym_triton = None

    def test_eval_on_cpu_matches_reference(self):
        from auto_round.data_type.neuqi import _sym_loss_chunk, _sym_search_eval

        torch.manual_seed(3)
        data = torch.randn(96, 32)
        scales = torch.rand(96, 7) + 0.5
        qw = torch.rand(96, 32) + 0.1
        for q in (qw, None):
            ref = _sym_loss_chunk(data, q, scales, 7)
            out = _sym_search_eval(data, q, scales, 7)
            torch.testing.assert_close(out[0], ref[0], rtol=1e-5, atol=1e-6)
            mism = (out[1] != ref[1]).float().mean().item()
            assert mism < 5e-3, f"candidate index mismatch fraction {mism}"
            mism_m = (out[2] != ref[2]).float().mean().item()
            assert mism_m < 5e-2, f"convention mismatch fraction {mism_m}"

    def test_attempt_skips_and_latches_on_failure(self, monkeypatch):
        import auto_round.data_type.neuqi as N

        # a CUDA-shaped stand-in: the gate only consults is_cuda/device before
        # the launch attempt (real-CUDA parity is covered by the skipif test)
        _assert_launch_latches(
            monkeypatch,
            N,
            attempt="_sym_triton_attempt",
            impl_attr="_sym_triton",
            checked_attr="_sym_triton_checked",
            broken_attr="_sym_triton_broken",
            gates={"_sym_wants_triton": True},
            tail_args=(None, None, 7),
        )

    def test_cpu_never_reaches_triton(self, monkeypatch):
        import auto_round.data_type.neuqi as N

        def _boom(*a, **k):
            raise AssertionError("CPU tensors must never reach the Triton kernel")

        monkeypatch.setattr(N, "_sym_triton", _boom)
        monkeypatch.setattr(N, "_sym_triton_checked", True)
        out = N._sym_search_eval(torch.randn(8, 32), None, torch.rand(8, 2) + 0.5, 7)
        assert out[0].shape == (8,)

    def test_wrapper_rejects_cpu_tensors(self):
        pytest.importorskip("auto_round_extension.triton.neuqi_sweep")
        from auto_round_extension.triton.neuqi_sweep import sym_search_triton

        with pytest.raises(AssertionError):
            sym_search_triton(torch.randn(4, 32), None, torch.rand(4, 2) + 0.5, 7)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for the Triton kernel")
    def test_triton_matches_reference_on_cuda(self):
        from auto_round.data_type.neuqi import _sym_loss_chunk
        from auto_round_extension.triton.neuqi_sweep import sym_search_triton

        torch.manual_seed(4)
        dev = torch.device("cuda")
        data = torch.randn(4096, 128, device=dev)
        data[:, :5] *= 120.0  # outlier columns like real weights
        scales = torch.rand(4096, 33, device=dev) + 0.25
        qw = torch.rand(4096, 128, device=dev) + 0.1
        for q in (qw, None):
            ref = _sym_loss_chunk(data, q, scales, 7)
            out = sym_search_triton(data.contiguous(), q.contiguous() if q is not None else None, scales, 7)
            torch.testing.assert_close(out[0], ref[0], rtol=1e-4, atol=1e-5)
            mism = (out[1] != ref[1]).float().mean().item()
            assert mism < 1e-2, f"candidate index mismatch fraction {mism}"
            mism_m = (out[2] != ref[2]).float().mean().item()
            assert mism_m < 5e-2, f"convention mismatch fraction {mism_m}"


class TestNeuqiSymCompiledCore:
    """The compiled sym chunk core: parity with eager, backend gating, latch-down."""

    @pytest.fixture(autouse=True)
    def _reset_core_state(self):
        import auto_round.data_type.neuqi as N

        N._sym_chunk_compiled = None
        N._sym_compile_failed = False
        N._sym_compile_logged.clear()
        yield
        N._sym_chunk_compiled = None
        N._sym_compile_failed = False

    def test_compiled_matches_eager(self, monkeypatch):
        monkeypatch.setenv("AR_NEUQI_BACKEND", "compile")
        torch.manual_seed(0)
        from auto_round.data_type.neuqi import _sym_loss_chunk, _sym_loss_chunk_eval

        data = torch.randn(64, 32)
        scales = torch.rand(64, 5) + 0.5
        for qw in (torch.rand(64, 32) + 0.1, None):
            out_e = _sym_loss_chunk(data, qw, scales, 7)
            out_c = _sym_loss_chunk_eval(data, qw, scales, 7)
            torch.testing.assert_close(out_c[0], out_e[0], rtol=1e-5, atol=1e-6)
            assert torch.equal(out_c[1], out_e[1])
            assert torch.equal(out_c[2], out_e[2])
        # where the toolchain allows inductor codegen the compiled core must
        # have ENGAGED (not silently latched down to eager); environments
        # without a C++ compiler (e.g. Windows without MSVC) latch by design
        import auto_round.data_type.neuqi as N

        if not N._sym_compile_failed:
            assert N._sym_chunk_compiled is not None

    def test_eager_backend_never_compiles(self, monkeypatch):
        monkeypatch.setenv("AR_NEUQI_BACKEND", "eager")
        from auto_round.data_type import neuqi

        def _boom(*a, **k):
            raise AssertionError("torch.compile must not be called under backend=eager")

        monkeypatch.setattr(neuqi.torch, "compile", _boom)
        out = neuqi._sym_loss_chunk_eval(torch.randn(8, 32), None, torch.rand(8, 2) + 0.5, 7)
        assert out[0].shape == (8,)

    def test_failure_latches_down_to_eager(self, monkeypatch):
        monkeypatch.setenv("AR_NEUQI_BACKEND", "compile")
        from auto_round.data_type import neuqi

        calls = {"n": 0}

        def _flaky_compile(fn):
            def _raise(*a, **k):
                calls["n"] += 1
                raise RuntimeError("inductor unavailable")

            return _raise

        monkeypatch.setattr(neuqi.torch, "compile", _flaky_compile)
        data, scales = torch.randn(8, 32), torch.rand(8, 2) + 0.5
        for _ in range(2):  # second call must go straight to eager
            out = neuqi._sym_loss_chunk_eval(data, None, scales, 7)
            assert torch.isfinite(out[0]).all()
        assert calls["n"] == 1
        assert neuqi._sym_compile_failed

    def test_search_end_to_end_matches_eager_backend(self, monkeypatch):
        """The compiled core routes every chunk through; results must match an
        eager-backend run."""
        from auto_round.data_type.neuqi import neuqi_search_scale_sym

        torch.manual_seed(1)
        data = torch.randn(128, 32)

        def _run():
            return neuqi_search_scale_sym(data.clone(), 4, coarse_n=8, fine_n=4)

        monkeypatch.setenv("AR_NEUQI_BACKEND", "eager")
        eager_plain = _run()
        monkeypatch.setenv("AR_NEUQI_BACKEND", "compile")
        comp_plain = _run()
        torch.testing.assert_close(comp_plain, eager_plain, rtol=0, atol=0)


class TestNeuqiSymDynamoGuard:
    def test_sym_search_never_traced_by_dynamo(self):
        """The sym search is loop-heavy and eager-shaped; under
        --enable_torch_compile it must be skipped by dynamo (tracing it unrolls
        the chunk loops into giant graphs - minutes of CPU compile, GPU idle)."""
        from auto_round.data_type.neuqi import _two_stage_sym_core, neuqi_search_scale_sym

        for fn in (neuqi_search_scale_sym, _two_stage_sym_core):
            assert getattr(fn, "_torchdynamo_disable", False), fn.__name__


class TestNeuqiSymIntegration:
    def test_sym_neuqi_dispatch(self):
        fn, name = get_quant_func("int", 4, True, disable_opt_rtn=False, group_size=32, iters=0, enable_neuqi=True)
        assert name == "opt_rtn_int_sym_neuqi"
        assert "opt_rtn_int_sym_neuqi" in QUANT_FUNC_WITH_DTYPE

    def test_wrapper_qdq_matches_returned_signed_scale(self):
        from auto_round.data_type.neuqi import quant_tensor_opt_rtn_sym_neuqi

        torch.manual_seed(13)
        w = torch.randn(32, 256)
        w[:, :20] *= -40.0  # negative-heavy columns exercise the mirrored family
        qdq, scale, zp = quant_tensor_opt_rtn_sym_neuqi(w, bits=4, group_size=32)
        assert zp == 8
        assert scale.shape == (32 * 256 // 32, 1)
        s2d = torch.repeat_interleave(scale.reshape(32, 8), 32, dim=1)
        manual = (w / s2d).round().clamp(-8, 7) * s2d
        assert torch.allclose(qdq, manual, rtol=1e-3, atol=1e-4)
        assert (scale < 0).any()  # mirrored family won somewhere

    def test_sym_wrapper_supports_imatrix_off_1d_2d(self):
        """qw=None (imatrix off), per-column 1-D and per-row 2-D weights all flow;
        weighting must steer the search (lower weighted loss than the unweighted
        solution under the same weights)."""
        from auto_round.data_type.neuqi import neuqi_search_scale_sym, quant_tensor_opt_rtn_sym_neuqi

        torch.manual_seed(17)
        w = torch.randn(32, 256)
        w[:, :24] *= -40.0
        for im in (None, torch.rand(256) + 0.05, torch.rand(32, 256) + 0.05):
            qdq, scale, _ = quant_tensor_opt_rtn_sym_neuqi(w.clone(), bits=4, group_size=32, imatrix=im)
            assert torch.isfinite(qdq).all()

        d = w.reshape(-1, 32)
        qw = (torch.rand(32, 256) + 0.05).reshape(-1, 32)
        s_q = neuqi_search_scale_sym(d.clone(), bits=4, qw=qw.clone())
        s_p = neuqi_search_scale_sym(d.clone(), bits=4, qw=None)

        def wloss(s):
            iq = (d / s).round().clamp(-8, 7)
            return ((s * iq - d) ** 2 * qw).sum().item()

        assert wloss(s_q) < wloss(s_p)


class _FakeQuantLayer(torch.nn.Module):
    """Minimal stand-in for a quantisable layer (see also hadamard test_patch)."""

    def __init__(self, out_features=8, in_features=32, bits=4, sym=False, group_size=32):
        super().__init__()
        self.bits = bits
        self.sym = sym
        self.group_size = group_size
        self.data_type = "int"
        self.act_bits = 16
        self.act_data_type = "int"
        self.act_sym = True
        self.act_dynamic = True
        self.iters = 200
        self.tuning_device = "cpu"
        torch.manual_seed(7)
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = None
        self.imatrix = None


class TestNeuqiFrozenInit:
    """enable_neuqi on the tuning path: the search anchors the STE grid
    once at wrapper init and the range margins stay pinned at 1.0."""

    def _wrapper(self, enable_neuqi, sym=False, enable_round_tuning=True):
        from auto_round.wrapper import WrapperLinear

        layer = _FakeQuantLayer(sym=sym)
        wrapper = WrapperLinear(
            layer,
            device="cpu",
            enable_minmax_tuning=True,
            enable_norm_bias_tuning=False,
            enable_round_tuning=enable_round_tuning,
            enable_torch_compile=False,
            disable_opt_rtn=True,
            enable_neuqi=enable_neuqi,
            iters=5,
        )
        return wrapper, layer

    def test_neuqi_anchor_replaces_minmax_asym(self):
        from auto_round.data_type.neuqi import neuqi_search_scale_zero

        wrapper, layer = self._wrapper(True, sym=False)
        w = layer.weight.data.to(torch.float32).reshape(-1, layer.group_size)
        s, zp = neuqi_search_scale_zero(w, layer.bits)
        maxq = int(2**layer.bits) - 1
        assert torch.allclose(wrapper.weight_min, -(zp * s).squeeze(-1), atol=1e-6)
        assert torch.allclose(wrapper.weight_max, ((maxq - zp) * s).squeeze(-1), atol=1e-6)
        # frozen margins: the searched grid IS the init; SignRound tunes values only
        assert wrapper._neuqi_frozen_margins is True
        # non-tunable params become plain tensors, not entries in params
        assert "min_scale" not in wrapper.params and "max_scale" not in wrapper.params

    def test_sym_anchor_uses_two_stage_search(self):
        from auto_round.data_type.int import quant_tensor_sym
        from auto_round.data_type.neuqi import neuqi_search_scale_sym, quant_tensor_opt_rtn_sym_neuqi

        wrapper, layer = self._wrapper(True, sym=True)
        w = layer.weight.data.to(torch.float32).reshape(-1, layer.group_size)
        s_signed = neuqi_search_scale_sym(w, layer.bits)
        nmax = int(2 ** (layer.bits - 1))
        mag = (s_signed.abs() * nmax).squeeze(-1)
        mirrored = (s_signed < 0).squeeze(-1)
        # the STE tie-break maps wmax_abs < wmin_abs to the POSITIVE scale:
        # the losing side is stored at HALF magnitude so the comparison
        # resolves to the winner's convention in any storage dtype
        assert torch.allclose(wrapper.weight_min, torch.where(mirrored, -mag / 2, -mag))
        assert torch.allclose(wrapper.weight_max, torch.where(mirrored, mag, mag / 2))
        assert wrapper._neuqi_frozen_margins is True
        # R1-1/R2-1 regression guard: the anchored grid must reproduce the
        # search's winning qdq exactly, per group, under BOTH clamp conventions
        ref_qdq, _, _ = quant_tensor_opt_rtn_sym_neuqi(w, bits=layer.bits, group_size=layer.group_size)
        ste_qdq, ste_scale, _ = quant_tensor_sym(
            w,
            bits=layer.bits,
            group_size=layer.group_size,
            min_scale=torch.tensor(1.0),
            max_scale=torch.tensor(1.0),
            tensor_min=wrapper.weight_min,
            tensor_max=wrapper.weight_max,
        )
        assert torch.equal(ste_qdq.reshape(ref_qdq.shape), ref_qdq) or torch.allclose(
            ste_qdq.reshape(ref_qdq.shape), ref_qdq, atol=1e-7
        ), "anchored STE grid diverges from the search winner"

    def test_sym_anchor_survives_low_precision_storage(self, monkeypatch):
        # R2-1: the anchor is cast to the weight dtype; a sub-precision bias
        # (2^-20) would be erased by bf16/fp16 rounding and flip every
        # standard-convention group to the mirrored grid. The half-magnitude
        # bias must reproduce the search qdq for bf16 and fp16 weights too.
        import auto_round.wrapper as wrapper_mod
        from auto_round.data_type.int import quant_tensor_sym
        from auto_round.data_type.neuqi import neuqi_search_scale_sym, quant_tensor_opt_rtn_sym_neuqi

        for dtype in (torch.bfloat16, torch.float16):
            layer = _FakeQuantLayer(sym=True)
            layer.weight.data = layer.weight.data.to(torch.float32).to(dtype)
            wrapper = wrapper_mod.WrapperLinear(
                layer,
                device="cpu",
                enable_minmax_tuning=True,
                enable_round_tuning=True,
                enable_torch_compile=False,
                disable_opt_rtn=True,
                enable_neuqi=True,
                iters=5,
            )
            assert wrapper.weight_min.dtype == dtype
            w32 = layer.weight.data.to(torch.float32).reshape(-1, layer.group_size)
            s_signed = neuqi_search_scale_sym(w32, layer.bits)
            ref_qdq, _, _ = quant_tensor_opt_rtn_sym_neuqi(w32, bits=layer.bits, group_size=layer.group_size)
            ste_qdq, ste_scale, _ = quant_tensor_sym(
                layer.weight.data,
                bits=layer.bits,
                group_size=layer.group_size,
                min_scale=torch.tensor(1.0),
                max_scale=torch.tensor(1.0),
                tensor_min=wrapper.weight_min,
                tensor_max=wrapper.weight_max,
            )
            ste_qdq = ste_qdq.to(torch.float32).reshape(ref_qdq.shape)
            # (1) the R2-1 invariant: no group's clamp convention flips (the
            # pre-fix failure was a systematic all-group sign flip)
            ste_sign = torch.sign(ste_scale.reshape(-1))
            want_sign = torch.sign(s_signed.reshape(-1))
            assert torch.equal(
                ste_sign, want_sign
            ), f"{dtype}: convention flipped on {(ste_sign != want_sign).sum()} groups"
            # (2) residual diffs are bf16/fp16 boundary-rounding jitter of the
            # STE division on low-precision weights, bounded by one quantum
            diff = (ste_qdq - ref_qdq.to(torch.float32)).abs()
            quantum = s_signed.abs().reshape(-1, 1)
            assert (diff <= quantum + 1e-6).all(), f"{dtype}: diff {(diff.max()):.3e} exceeds one quantum"

    def test_anchor_stays_off_when_inappropriate(self):
        """Flag off: plain min/max with tunable margins. Zero-shot path: the
        search runs inside the quant-func dispatch, so the wrapper-side anchor
        must not fire there either."""
        wrapper, layer = self._wrapper(False, sym=False)
        w = layer.weight.data.reshape(-1, layer.group_size)
        assert torch.allclose(wrapper.weight_min, w.min(1)[0].clamp(max=0), atol=1e-7)
        assert torch.allclose(wrapper.weight_max, w.max(1)[0].clamp(min=0), atol=1e-7)
        assert wrapper._neuqi_frozen_margins is False
        assert wrapper.params["min_scale"].requires_grad is True
        assert wrapper.params["max_scale"].requires_grad is True

        wrapper, _ = self._wrapper(True, enable_round_tuning=False)
        assert wrapper.weight_min is None and wrapper.weight_max is None
        assert wrapper._neuqi_frozen_margins is False


class TestSignRoundEnableNeuqi:
    """SignRoundConfig carries enable_neuqi; True = the frozen-init anchor."""

    def test_config_flag(self):
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

        assert SignRoundConfig(bits=4, group_size=32).enable_neuqi is False
        assert SignRoundConfig(bits=4, group_size=32, sym=False, enable_neuqi=True).enable_neuqi is True


class TestExpertBatching:
    """Same-shape expert projections quantize in one stacked search call
    (row-independent: results identical to per-module calls)."""

    @pytest.fixture(autouse=True)
    def _tie_free_grid(self, monkeypatch):
        # exact stacked-vs-per-module equality holds while the grid produces no
        # fp-summation near-ties for the seeded data; the narrow eager default
        # (64/32) hits ties that first-min resolves differently between the
        # batched expression and the per-candidate expression (the sweep parity
        # tests cover the tie semantics) -- pin the wide grid this exact-equality
        # contract was verified against
        monkeypatch.setenv("AR_NEUQI_COARSE", "256")
        monkeypatch.setenv("AR_NEUQI_FINE", "64")

    def teardown_method(self):
        import auto_round.algorithms.quantization.rtn.quantizer as _qmod

        if "model_context" in _qmod.OptimizedRTNQuantizer.__dict__:
            del _qmod.OptimizedRTNQuantizer.model_context

    def _expert(self, name, out=16, inn=32, bits=4, group_size=32, sym=False, seed=0):
        m = torch.nn.Linear(inn, out, bias=False)
        torch.manual_seed(seed)
        with torch.no_grad():
            m.weight.copy_(torch.randn(out, inn))
        m.global_name = name
        m.bits = bits
        m.group_size = group_size
        m.sym = sym
        m.data_type = "int"
        return m

    def _quantizer(self, enable_neuqi=True, disable_opt_rtn=False, is_moe=True):
        from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig
        from auto_round.algorithms.quantization.rtn.quantizer import OptimizedRTNQuantizer

        cfg = OptimizedRTNConfig(bits=4, group_size=32, sym=False, enable_neuqi=enable_neuqi)
        cfg.disable_opt_rtn = disable_opt_rtn
        cfg.orig_disable_opt_rtn = False if not disable_opt_rtn else True
        q = OptimizedRTNQuantizer(cfg)
        if is_moe:
            # model_context is a read-only property backed by the run ctx;
            # patch the class property for the MoE-heuristic branch
            import auto_round.algorithms.quantization.rtn.quantizer as _qmod

            _qmod.OptimizedRTNQuantizer.model_context = property(lambda self: type("MC", (), {"is_moe_model": True})())
        return q

    def test_sym_and_asym_batched_and_equivalent(self):
        # THE sym gap fix: sym modules join the batches under neuqi, and the
        # stacked call matches per-module search outputs exactly
        q = self._quantizer()
        mods = [self._expert(f"blk.experts.{i}.gate_proj", sym=False, seed=i) for i in range(4)]
        sym_mods = [self._expert(f"blk.experts.{i}.up_proj", sym=True, seed=100 + i) for i in range(4)]
        batches, singles = q._split_expert_batches(mods + sym_mods)
        assert len(batches) == 2 and not singles
        assert {m.sym for m in batches[0]} == {False} and {m.sym for m in batches[1]} == {True}

        from auto_round.data_type.neuqi import quant_tensor_opt_rtn_asym, quant_tensor_opt_rtn_sym_neuqi

        assert q._quantize_expert_batch(batches[0], "cpu") == []
        for m in mods:
            ref_qdq, ref_scale, ref_zp = quant_tensor_opt_rtn_asym(
                m.weight.data, bits=m.bits, group_size=m.group_size, v=0.0, q_scale_thresh=1e-5, imatrix=None
            )
            assert torch.equal(m.weight.data, ref_qdq)
            assert torch.equal(m.scale.reshape(-1), ref_scale.reshape(-1))
            assert torch.equal(m.zp.reshape(-1), ref_zp.reshape(-1))
            assert m.q_scale_thresh == 1e-5

        assert q._quantize_expert_batch(batches[1], "cpu") == []
        for m in sym_mods:
            ref_qdq, ref_scale, ref_zp = quant_tensor_opt_rtn_sym_neuqi(
                m.weight.data, bits=m.bits, group_size=m.group_size, v=0.0, q_scale_thresh=1e-5, imatrix=None
            )
            assert torch.equal(m.weight.data, ref_qdq)
            assert torch.equal(m.scale.reshape(-1), ref_scale.reshape(-1))
            assert m.zp == ref_zp
        # R2-5: the batch writeback must stamp the resolved data_type both classes
        assert all(m.data_type == "opt_rtn_int_asym" for m in mods)
        assert all(m.data_type == "opt_rtn_int_sym_neuqi" for m in sym_mods)

    def test_imatrix_weighting_stacked_equivalence(self):
        q = self._quantizer()
        mods = [self._expert(f"blk.experts.{i}.down_proj", sym=False, seed=7 + i) for i in range(3)]
        for i, m in enumerate(mods):
            torch.manual_seed(50 + i)
            m.imatrix = torch.rand(m.weight.shape[1])
        batches, singles = q._split_expert_batches(mods)
        assert len(batches) == 1 and not singles
        assert q._quantize_expert_batch(batches[0], "cpu") == []
        from auto_round.data_type.neuqi import quant_tensor_opt_rtn_asym

        for m in mods:
            ref_qdq, ref_scale, ref_zp = quant_tensor_opt_rtn_asym(
                m.weight.data,
                bits=m.bits,
                group_size=m.group_size,
                v=0.0,
                q_scale_thresh=1e-5,
                imatrix=m.imatrix,
            )
            assert torch.equal(m.weight.data, ref_qdq)
            assert torch.equal(m.scale.reshape(-1), ref_scale.reshape(-1))

    def test_batch_splitting_rules(self):
        """One setup for every _split_expert_batches rule: gating, expert-name
        filtering, shape/act-quant/symmetry segregation."""
        q = self._quantizer()

        # gating: off -> inactive; disable_opt_rtn -> inactive
        assert self._quantizer(enable_neuqi=False)._expert_search_active() is False
        assert self._quantizer(enable_neuqi=True, disable_opt_rtn=True)._expert_search_active() is False

        # non-expert and odd-shaped modules stay single
        dense = self._expert("blk.mlp.gate_proj")
        odd = self._expert("blk.experts.0.gate_proj", inn=48, group_size=32)  # in % g != 0 is fine for grouping
        mixed_shape = self._expert("blk.experts.1.gate_proj", inn=64)
        batches, singles = q._split_expert_batches([dense, odd, mixed_shape])
        assert not batches and {dense, odd, mixed_shape} == set(singles)

        # act quantization needs the per-module wrapper path
        m = self._expert("blk.experts.0.gate_proj")
        m.act_bits = 8
        batches, singles = q._split_expert_batches([m])
        assert not batches and singles == [m]

        # same (parent, proj, shape, bits, gs) but different symmetry: not
        # mergeable into one stacked call -> both stay single (group of 1)
        a = self._expert("blk.experts.0.gate_proj", sym=False)
        sym_m = self._expert("blk.experts.1.gate_proj", sym=True)
        batches, singles = q._split_expert_batches([a, sym_m])
        assert not batches and len(singles) == 2

    def test_oom_returns_only_unwritten(self, monkeypatch):
        # R2-2: a mid-batch OOM must leave already-written modules untouched
        # and return ONLY the unwritten remainder (per-module fallback must
        # never re-quantize a qdq'd weight)
        import auto_round.algorithms.quantization.rtn.quantizer as Q
        import auto_round.data_type.neuqi as N

        q = self._quantizer()
        mods = [self._expert(f"blk.experts.{i}.gate_proj", seed=i) for i in range(3)]
        batches, singles = q._split_expert_batches(mods)
        assert len(batches) == 1

        calls = []
        orig = N.quant_tensor_opt_rtn_asym

        def oom_after_first(weights, *a, **kw):
            calls.append(1)
            if len(calls) > 1:
                raise torch.OutOfMemoryError("simulated")
            return orig(weights, *a, **kw)

        monkeypatch.setattr(N, "quant_tensor_opt_rtn_asym", oom_after_first)
        # one module per stacked call -> the failure hits mid-batch
        monkeypatch.setattr(Q, "_EXPERT_BATCH_MAX_ELEMS", mods[0].weight.numel())
        before = mods[0].weight.data.clone()
        remaining = q._quantize_expert_batch(batches[0], "cpu")
        assert len(remaining) == 2, "modules after the failed chunk must be returned unwritten"
        assert remaining == mods[1:]
        # the first chunk was written and must NOT be re-quantized later
        assert not torch.equal(mods[0].weight.data, before)
        assert hasattr(mods[0], "scale") and mods[0].data_type == "opt_rtn_int_asym"
        # unwritten modules are untouched
        assert not hasattr(mods[1], "scale") and not hasattr(mods[2], "scale")

    def test_act_quant_modules_stay_single(self):
        q = self._quantizer()
        m = self._expert("blk.experts.0.gate_proj")
        m.act_bits = 8  # act quantization needs the per-module wrapper path
        batches, singles = q._split_expert_batches([m])
        assert not batches and singles == [m]

    def test_sym_key_separates_batches(self):
        q = self._quantizer()
        a = self._expert("blk.experts.0.gate_proj", sym=False)
        s = self._expert("blk.experts.1.gate_proj", sym=True)
        batches, singles = q._split_expert_batches([a, s])
        # same (parent, proj, shape, bits, gs) but different symmetry: not
        # mergeable into one stacked call -> both stay single (group of 1)
        assert not batches and len(singles) == 2


class TestImatrixAutoRule:
    """asym + neuqi must auto-collect the imatrix (the search consumes qw);
    plain asym keeps it off (min/max init ignores it)."""

    def test_rule(self):
        from auto_round.algorithms.quantization.rtn.config import OptimizedRTNConfig, RTNConfig
        from auto_round.autoround import _select_rtn_compressor_base_cls

        cfg = RTNConfig(bits=4, group_size=128, sym=False, enable_neuqi=True)
        _select_rtn_compressor_base_cls(cfg, scheme=None, format="auto_round", base_kwargs={})
        assert cfg.enable_imatrix is True and isinstance(cfg, OptimizedRTNConfig)

        cfg = RTNConfig(bits=4, group_size=128, sym=False, enable_neuqi=False)
        _select_rtn_compressor_base_cls(cfg, scheme=None, format="auto_round", base_kwargs={})
        assert cfg.enable_imatrix is False


class TestEnableNeuqiCliFlag:
    """The CLI gate is --enable_neuqi (per the governing PR design); the
    boolean enable_neuqi kwarg is the API surface."""

    def test_cli_flag_registration(self):
        from auto_round.algorithms.quantization.rtn.config import RTNConfig
        from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig

        reg = RTNConfig.get_registered_args()
        p = next(p for p in reg.parameters if "--enable_neuqi" in p.option_strings)
        assert p.field == "enable_neuqi"
        assert p.argparse_kwargs["action"] == "store_true"
        assert any("--enable_neuqi" in q.option_strings for q in SignRoundConfig.get_registered_args().parameters)
