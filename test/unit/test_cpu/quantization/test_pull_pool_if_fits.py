# coding=utf-8
# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""Unit tests for the iters>0 hot-pool bulk pull gate (``_pull_pool_if_fits``).

The helper decides whether a tune-loop pool moves in bulk onto the device
that reads it every iteration (entry device for inputs, loss device for
references). The gate arithmetic is exercised with tensor/block fakes and a
monkeypatched free-memory probe: no real multi-GPU devices are needed.
"""

import unittest
from unittest import mock

import torch

from auto_round.algorithms.quantization.sign_round.quantizer import _pull_pool_if_fits

_GB = 2**30

_GIB = 2**30


class _FakeTensor:  # pylint: disable=too-few-public-methods
    def __init__(self, device, numel, esize=4):
        self.device = torch.device(device)
        self._numel = numel
        self._esize = esize
        self.moved_to = None

    def numel(self):
        return self._numel

    def element_size(self):
        return self._esize

    def to(self, device):
        self.moved_to = torch.device(device)
        self.device = torch.device(device)
        return self


class _FakeParam(_FakeTensor):  # pylint: disable=too-few-public-methods
    pass


class _FakeBlock:  # pylint: disable=too-few-public-methods
    def __init__(self, params):
        self._params = params

    def parameters(self):
        return list(self._params)

    def modules(self):
        return [self]


def _run(pool, block, free, target="cuda:1", iters=20, label="tune-reference", **kw):
    with mock.patch("auto_round.utils.device.probe_usable_bytes", return_value=free), mock.patch(
        "auto_round.utils.pool_placement._working_allowance_bytes", return_value=int(1.4 * _GIB)
    ):
        return _pull_pool_if_fits(pool, target, block, 8, iters, label, **kw)


class TestPullPoolIfFits(unittest.TestCase):
    def test_moves_when_free_covers_pool_and_state(self):
        # 8-GPU-shaped arithmetic: 4 GiB pool, ~6.4 GiB state on the target.
        pool = [_FakeTensor("cuda:0", numel=_GIB // 4) for _ in range(4)]
        block = _FakeBlock([_FakeParam("cuda:1", numel=int(0.46e9))])
        out = _run(pool, block, free=19.5 * _GIB)
        self.assertEqual(len(out), len(pool))
        self.assertTrue(all(t.moved_to == torch.device("cuda:1") for t in out))

    def test_declines_when_state_density_eats_headroom(self):
        # 4-GPU-shaped arithmetic: same pool, ~12.9 GiB state on the target,
        # ~16 GiB free -> reserve alone exceeds free minus pool.
        pool = [_FakeTensor("cuda:0", numel=_GIB // 4) for _ in range(4)]
        block = _FakeBlock([_FakeParam("cuda:1", numel=int(0.92e9))])
        out = _run(pool, block, free=16.1 * _GIB)
        self.assertIs(out, pool)
        self.assertTrue(all(t.moved_to is None for t in pool))

    def test_input_pull_charges_routed_budget(self):
        # the OOMed 4-GPU lane: entry free ~15.9 GiB, pool 4, routed budget
        # 12 GiB (tokens x top_k x hidden x 6, from the pr/streaming formula)
        # -> declines; with no MoE activation cost the same numbers pull
        with mock.patch(
            "auto_round.algorithms.quantization.sign_round.quantizer._block_activation_bytes",
            return_value=int(12 * _GIB),
        ):
            pool = [_FakeTensor("cuda:2", numel=_GIB // 4) for _ in range(4)]
            block = _FakeBlock([])
            out = _run(pool, block, free=15.9 * _GIB, target="cuda:0", label="tune-input", charge_activation=True)
            self.assertTrue(all(t.moved_to is None for t in out))
        with mock.patch(
            "auto_round.algorithms.quantization.sign_round.quantizer._block_activation_bytes", return_value=0
        ):
            pool2 = [_FakeTensor("cuda:2", numel=_GIB // 4) for _ in range(4)]
            out2 = _run(pool2, block, free=15.9 * _GIB, target="cuda:0", label="tune-input", charge_activation=True)
            self.assertTrue(all(t.moved_to == torch.device("cuda:0") for t in out2))

    def test_stacks_count_wrapper_and_orig_layer_once(self):
        # at tune time named_modules yields the wrapper AND its orig_layer
        # child, each with a same-shape weight -- counting both doubled
        # retention (10.3 GiB charged where the honest charge is ~5.2)
        from types import SimpleNamespace

        import torch.nn as nn

        import auto_round.algorithms.quantization.sign_round.quantizer as q

        class _MoeMLP(nn.Module):

            pass

        class _Experts(nn.Module):

            pass

        def _mk_w():
            return SimpleNamespace(numel=lambda: 16, device="cuda:1", nbytes=64, shape=(4, 4))

        blk = nn.Module()
        mlp = _MoeMLP()
        experts = _Experts()
        for i in range(2):
            leaf = nn.Module()
            orig = nn.Module()
            w = _mk_w()
            object.__setattr__(leaf, "weight", w)
            object.__setattr__(orig, "weight", w)  # same tensor: id-dedup path
            leaf.orig_layer = orig
            setattr(experts, str(i), leaf)
        # distinct-tensor orig: the .orig_layer name-skip path
        leaf3 = nn.Module()
        orig3 = nn.Module()
        object.__setattr__(leaf3, "weight", _mk_w())
        object.__setattr__(orig3, "weight", _mk_w())
        leaf3.orig_layer = orig3
        experts.extra = leaf3
        mlp.experts = experts
        blk.mlp = mlp

        got = q._grouped_stack_bytes(blk, "cuda:1")
        # 3 logical projections x 16 elems x 6 B = 288 B minimum (plus transient)
        self.assertGreaterEqual(got, 288)
        # double-counted form would charge >= 5 x 16 x 6 = 480 retention-only
        detail = q._grouped_stack_bytes_detail(blk, "cuda:1")
        self.assertLess(detail["retention"], 480)

    def test_stacks_mode_gate_returns_dict_under_linear_loop_env(self):
        # explicit linear_loop env: the detail walk returns the zero DICT
        # (a bare 0 here raised TypeError inside _grouped_stack_bytes and the
        # broad except silently killed the whole activation model)
        from types import SimpleNamespace

        import auto_round.algorithms.quantization.sign_round.quantizer as q
        from auto_round import envs

        blk = SimpleNamespace(modules=lambda: iter([]), named_modules=lambda: iter([]))
        with mock.patch.object(envs, "AR_MOE_EXPERTS_IMPL", "linear_loop"):
            detail = q._grouped_stack_bytes_detail(blk, "cuda:1")
            total = q._grouped_stack_bytes(blk, "cuda:1")
        self.assertEqual(detail, {"retention": 0, "transient": 0})
        self.assertEqual(total, 0)

    def test_stacks_follow_config_after_auto_switch(self):
        # after the auto pick switches the run to linear_loop, the gates must
        # charge the LOOP composition (no stacks, routed re-engaged), not the
        # env-stale grouped one
        from types import SimpleNamespace

        import auto_round.algorithms.quantization.sign_round.quantizer as q
        from auto_round import envs

        blk = SimpleNamespace(modules=lambda: iter([]), named_modules=lambda: iter([]))
        cfg = type("C", (), {})()
        cfg._experts_implementation = "linear_loop"
        with mock.patch.object(envs, "AR_MOE_EXPERTS_IMPL", "auto"):
            self.assertEqual(q._grouped_stack_bytes(blk, "cuda:1", config=cfg), 0)

    def test_shared_experts_not_charged_as_routed(self):
        # shared experts are plain modules outside the dispatch; stacking or
        # routing charges on them are phantom bytes
        from types import SimpleNamespace

        import torch.nn as nn

        import auto_round.algorithms.quantization.sign_round.quantizer as q

        class _MoeMLP(nn.Module):

            pass

        class _Experts(nn.Module):

            pass

        blk = nn.Module()
        mlp = _MoeMLP()
        experts = _Experts()
        shared = _Experts()
        for i in range(2):
            leaf = nn.Module()
            object.__setattr__(
                leaf,
                "weight",
                SimpleNamespace(numel=lambda: 16, device="cuda:1", nbytes=64, shape=(4, 4)),
            )
            setattr(experts, str(i), leaf)
        shared_leaf = nn.Module()
        object.__setattr__(
            shared_leaf, "weight", SimpleNamespace(numel=lambda: 16, device="cuda:1", nbytes=64, shape=(4, 4))
        )
        shared.gate_proj = shared_leaf
        mlp.experts = experts
        mlp.shared_expert = shared
        blk.mlp = mlp

        got = q._grouped_stack_bytes(blk, "cuda:1")
        # only the two routed leaves: 2 x 16 x 6 = 192 B minimum
        self.assertGreaterEqual(got, 192)
        detail = q._grouped_stack_bytes_detail(blk, "cuda:1")
        self.assertLess(detail["retention"], 192 + 192)  # shared would add 96 more

    def test_routed_recorder_self_removes_only_on_record(self):
        # a miss (non-dispatch arg shapes) must NOT disarm the recorder;
        # a successful natural dispatch records and removes the hook
        import torch
        import torch.nn as nn

        import auto_round.algorithms.quantization.sign_round.quantizer as q

        class _MoeMLP(nn.Module):

            pass

        class _HYV3Experts(nn.Module):  # no num_experts, not a ModuleList

            def forward(self, *args):
                return args

        blk = nn.Module()
        mlp = _MoeMLP()
        experts = _HYV3Experts()
        mlp.experts = experts
        blk.mlp = mlp

        q._ensure_routed_shape_recorders_(blk)
        self.assertTrue(experts._routed_rec_handles_, "recorder must attach on the HYV3-style container")

        # miss: two tensors but index ndim 0 -> no record, hook stays
        experts(torch.zeros(2, 4), torch.zeros(()))
        self.assertFalse(hasattr(experts, "_routed_shape_rec_"))
        self.assertTrue(experts._routed_rec_handles_)

        # natural dispatch: (hidden [B,T,H], index [N,K]) -> record + remove
        experts(torch.zeros(2, 4, 8), torch.zeros(16, 8))
        self.assertEqual(experts._routed_shape_rec_, (8, 128))
        self.assertFalse(experts._routed_rec_handles_)

    def test_logical_state_walk_dedups_wrapper_values(self):
        # real-structure form of the a2414b42 fix: a module with a .params
        # dict carrying an fp32 value PLUS the original weight must be
        # charged once (14 B logical), not twice
        import torch
        import torch.nn as nn

        import auto_round.algorithms.quantization.sign_round.quantizer as q

        blk = nn.Module()
        lin = nn.Linear(4, 4)  # weight 16 + bias 4 = 20 elems on cpu
        wrapper = nn.Module()
        wrapper.value = nn.Parameter(torch.zeros(4, 4))  # registered fp32 value
        wrapper.params = {"value": wrapper.value}  # same object, as the real wrappers do
        blk.lin = lin
        blk.wrap = wrapper
        got = q._logical_state_by_device(blk)
        # deduped: the registered value is identified by identity via .params
        # and excluded; only the original 20 elems carry the 14 B charge
        self.assertEqual(got.get("cpu", 0), 20 * 14)
        # without dedup the walk would see 36 elems (504 bytes)
        self.assertLess(got.get("cpu", 0), 36 * 14)

    def test_routed_budget_dropped_when_grouped_stacks_present(self):
        # grouped mode builds no per-expert route caches (linear_loop
        # calibration); charging routed x6 ALONGSIDE stacks double-charged
        # the working 5x3090 lane and falsely switched it to linear_loop
        from types import SimpleNamespace

        import auto_round.algorithms.quantization.sign_round.quantizer as q
        import auto_round.utils.device as dev_mod

        blk = SimpleNamespace(modules=lambda: iter([SimpleNamespace(num_experts=8)]))
        est = ({}, 0.0, 0.0, 0.0, {"cuda:1": 1.0, "cuda:0": 0.5}, {"cuda:1": 24})
        with mock.patch.object(dev_mod, "estimate_tuning_block_mem", return_value=est):
            with mock.patch.object(q, "_routed_budget_bytes", return_value=8 * _GB):
                with mock.patch.object(
                    q, "_grouped_stack_bytes", side_effect=lambda b, d, config=None: 2 * _GB if d == "cuda:1" else 0
                ):
                    got = q._activation_bytes_by_device(blk, [], 8, None)
        self.assertIn("cuda:1", got)
        # with routed charged it would be max(1, 8) + 2 = 10 GiB; honest = 1 + 2
        self.assertLess(got["cuda:1"], 4 * _GB)
        self.assertGreaterEqual(got["cuda:1"], 3 * _GB)

    def test_grouped_stacks_detect_hy3_container(self):
        # HYV3Experts carries NEITHER num_experts NOR ModuleList -- the MoE
        # marker sits on an ancestor. The old container walk returned 0 stacks
        # on the real lane, silently disabling the activation charge and the
        # auto linear_loop pick (4x3090 grouped OOM with no switch).
        from types import SimpleNamespace

        import torch.nn as nn

        import auto_round.algorithms.quantization.sign_round.quantizer as q

        class _MoeMLP(nn.Module):

            pass

        class _HYV3Experts(nn.Module):  # no num_experts, not a ModuleList

            pass

        blk = nn.Module()
        mlp = _MoeMLP()
        experts = _HYV3Experts()
        for i in range(2):
            leaf = nn.Linear(4, 4)
            object.__setattr__(
                leaf, "weight", SimpleNamespace(numel=lambda: 16, device="cuda:1", nbytes=32, shape=(4, 4))
            )
            setattr(experts, str(i), leaf)
        mlp.experts = experts
        blk.mlp = mlp

        got = q._grouped_stack_bytes(blk, "cuda:1")
        self.assertGreater(got, 0)
        # retention 2 leaves x 16 elems x 6 B = 192 B minimum (plus transient)
        self.assertGreaterEqual(got, 192)
        # no expert leaves homed elsewhere
        self.assertEqual(q._grouped_stack_bytes(blk, "cuda:0"), 0)

    def test_activation_charge_is_per_device(self):
        # fwd/bwd executes on EVERY device the block spans: each expert's
        # routed-row caches land on the expert's WEIGHT HOME. The routed
        # budget must split by homed-expert count, the shared/dense modules
        # charge their own home -- not everything onto one scalar.
        from types import SimpleNamespace

        import torch.nn as nn

        import auto_round.algorithms.quantization.sign_round.quantizer as q

        GB = 2**30

        def _dev_linear(in_f, out_f, dev):
            m = nn.Linear(in_f, out_f)
            # bypass nn.Module's Parameter-only assignment: __dict__ shadow
            object.__setattr__(m, "weight", SimpleNamespace(nbytes=in_f * out_f * 2, device=dev))
            m.orig_layer = m
            m.bits = 4
            return m

        class _MoeMLP(nn.Module):  # class name carries the MoE marker (hy3 HYV3MoeMLP)

            pass

        blk = nn.Module()
        # 6 experts over 3 peers (2 each), shared MLP on cuda:0 (hy3 shape)
        mlp = _MoeMLP()
        mlp.experts = nn.ModuleList([_dev_linear(4096, 1536, f"cuda:{1 + i % 3}") for i in range(6)])
        mlp.experts.num_experts = 6
        blk.mlp = mlp
        blk.shared = _dev_linear(4096, 13312, "cuda:0")
        # the ratio walk reads num_experts off the (model) config
        config = type("C", (), {"num_experts_per_tok": 8, "num_experts": 192})()
        ref = torch.zeros(1, 2048, 4096)  # fp32: hidden_bytes 16384

        by_dev = q._activation_bytes_by_device(blk, [ref], 8, config)
        self.assertIsNotNone(by_dev)
        # routed total = 8*2048 tokens * top_k 8 * 16384 B * 6 = 12.0 GiB,
        # split by 2/6 homed experts per peer
        for peer in ("cuda:1", "cuda:2", "cuda:3"):
            self.assertAlmostEqual(by_dev[peer] / GB, 12.0 / 3, delta=0.1)
        # cuda:0 hosts only the shared MLP: estimator term, no routed slice
        self.assertAlmostEqual(by_dev["cuda:0"] / GB, 16384 * 13312 * 4 * 2 / GB, delta=0.1)

    def test_routed_budget_formula_reproduces_measured_lane(self):
        # hy3: batch 8 x seq 2048 x top_k 8 x hidden 4096 x fp32 x 6 ~= 12.0 GiB
        # (measured loop retention: 12.7 GiB of batch cats + routed caches)
        import torch.nn as nn

        import auto_round.algorithms.quantization.sign_round.quantizer as q

        class _Experts(nn.Module):  # routed container (hy3 mlp.experts)
            num_experts = 192

        ref = torch.zeros(1, 2048, 4096)  # fp32 pool sample
        block = nn.Sequential(_Experts(), nn.Linear(4, 4))
        config = type("C", (), {"num_experts_per_tok": 8})()
        got = q._block_activation_bytes(block, [ref], 8, config)
        self.assertAlmostEqual(got / _GIB, 12.0, delta=0.05)

    def test_recorded_dispatch_shapes_beat_missing_config(self):
        # config absent + module attrs absent -> recorded (top_k, rows) decides
        import torch.nn as nn

        import auto_round.algorithms.quantization.sign_round.quantizer as q

        class _Experts(nn.Module):
            num_experts = 192

        exp = _Experts()
        exp._routed_shape_rec_ = (8, 8 * 2048 * 8)  # seen at batch 8, seq 2048
        block = nn.Sequential(exp)
        ref = torch.zeros(1, 2048, 4096)
        got = q._block_activation_bytes(block, [ref], 8, config=None)
        self.assertAlmostEqual(got / _GIB, 12.0, delta=0.05)

    def test_estimator_and_routed_composed_by_max(self):
        # a big shared expert (plain module) makes the estimator term win;
        # routed alone would undercharge it -- max covers both
        import torch.nn as nn

        import auto_round.algorithms.quantization.sign_round.quantizer as q

        class _Experts(nn.Module):
            num_experts = 192

        class _Shared(nn.Module):  # plain shared MLP, full-token width
            def __init__(self):
                super().__init__()
                self.gate_proj = nn.Linear(4096, 13312)

        exp = _Experts()
        exp.gate_proj = nn.Linear(4096, 1536)
        block = nn.Sequential(exp, _Shared())
        block.config = type("C", (), {"num_experts_per_tok": 8})()
        # mark them quantizable-ish for the estimator's check_to_quantized
        for m in (exp.gate_proj, block[1].gate_proj):
            m.orig_layer = m
            m.bits = 4
            m.act_bits = 16
            m.group_size = 128
        ref = torch.zeros(1, 2048, 4096)
        got = q._block_activation_bytes(block, [ref], 8, block.config)
        routed = 8 * 2048 * 8 * 4096 * 4 * 6  # 12 GiB
        # shared 13312-wide gate at full tokens x2 grads ~ 1.56 GiB + expert
        # module output... estimator path runs the real estimator; assert the
        # composition never returns less than the routed term
        self.assertGreaterEqual(got, routed)

    def test_state_charged_only_for_params_on_target(self):
        pool = [_FakeTensor("cuda:0", numel=_GIB // 4)]
        block = _FakeBlock([_FakeParam("cuda:2", numel=int(4 * _GIB))])
        out = _run(pool, block, free=7 * _GIB)
        # peers' state is not charged: 7 - (1.4 + 0.5) >= 4 GiB pool bytes
        self.assertTrue(all(t.moved_to == torch.device("cuda:1") for t in out))

    def test_already_local_pool_is_noop(self):
        pool = [_FakeTensor("cuda:1", numel=_GIB // 4)]
        block = _FakeBlock([])
        out = _run(pool, block, free=0)  # probe value irrelevant
        self.assertIs(out, pool)
        self.assertTrue(all(t.moved_to is None for t in pool))

    def test_non_list_pool_untouched(self):
        block = _FakeBlock([])
        out = _run({"a": 1}, block, free=100 * _GIB)
        self.assertEqual(out, {"a": 1})

    def test_probe_failure_keeps_sharded(self):
        pool = [_FakeTensor("cuda:0", numel=_GIB // 4)]
        block = _FakeBlock([])
        # free=None -> decline path (no move); helper never raises
        with mock.patch("auto_round.utils.device.probe_usable_bytes", side_effect=RuntimeError("no cuda")), mock.patch(
            "auto_round.utils.pool_placement._working_allowance_bytes", return_value=0
        ):
            out = _pull_pool_if_fits(pool, "cuda:1", block, 8, 20, "tune-reference")
        self.assertIs(out, pool)
        self.assertTrue(all(t.moved_to is None for t in pool))


class TestTuningStateBytes(unittest.TestCase):
    def test_wrapper_params_excluded_from_state(self):
        import torch.nn as nn

        block = nn.Linear(4, 4)  # 16 + 4 params, any device (cpu here)
        # wrapper-style tuning tensor: same numel as the weight, registered
        # nowhere but present in a .params dict; identity-excluded from the count
        value = torch.zeros_like(block.weight)
        block.params = {"value": value}
        from auto_round.algorithms.quantization.sign_round.quantizer import _tuning_state_bytes

        # cpu target: logical params = 20 (weight 16 + bias 4), value excluded
        self.assertEqual(_tuning_state_bytes(block, "cpu"), 20 * 14)


if __name__ == "__main__":
    unittest.main()
