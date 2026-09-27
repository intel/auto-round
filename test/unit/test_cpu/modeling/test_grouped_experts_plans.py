# coding=utf-8
# Copyright 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-device grouped experts plans and the align-hook exemption."""

import unittest

import torch
from torch import nn

from auto_round.modeling.fused_moe.grouped_experts import (
    _hooks_are_alignment_only,
    _projection_is_supported,
    _run_routes,
)


def _linear(out=4, inp=4):
    lin = nn.Linear(inp, out, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.randn_like(lin.weight))
    return lin


def _expert(gate=True, seed=0):
    torch.manual_seed(seed)
    mod = nn.Module()
    mod.up_proj = _linear()
    mod.down_proj = _linear()
    if gate:
        mod.gate_proj = _linear()
    return mod


class _Experts(nn.Module):
    def __init__(self, n=2):
        super().__init__()
        self.n = n
        for i in range(n):
            setattr(self, str(i), _expert(seed=i))
        self.act_fn = nn.functional.silu

    def forward(self, x, idx, w):
        return _run_routes(self, x, idx, w, self.n)


class TestHookExemption(unittest.TestCase):
    def test_no_hooks_is_alignment_only(self):
        self.assertTrue(_hooks_are_alignment_only(_linear()))

    def test_plain_hook_is_not_exempt(self):
        lin = _linear()
        lin.register_forward_hook(lambda m, i, o: o)
        self.assertFalse(_hooks_are_alignment_only(lin))
        self.assertFalse(_projection_is_supported(lin))  # calibration hooks must fire

    def test_align_hook_without_offload_is_exempt(self):
        from accelerate.hooks import AlignDevicesHook

        lin = _linear()
        lin.register_forward_pre_hook(AlignDevicesHook(execution_device=lin.weight.device))
        self.assertTrue(_hooks_are_alignment_only(lin))
        self.assertTrue(_projection_is_supported(lin))

    def test_align_hook_with_offload_is_not_exempt(self):
        from accelerate.hooks import AlignDevicesHook

        lin = _linear()
        hook = AlignDevicesHook(execution_device=lin.weight.device, offload=True)
        lin.register_forward_pre_hook(hook)
        self.assertFalse(_hooks_are_alignment_only(lin))
        self.assertFalse(_projection_is_supported(lin))


class TestRunRoutesSingleGroupCPU(unittest.TestCase):
    def test_matches_reference(self):
        torch.manual_seed(7)
        experts = _Experts(4)
        n_tokens, n_experts, top_k, hidden = 12, 4, 2, 4
        x = torch.randn(n_tokens, hidden)
        idx = torch.randint(0, n_experts, (n_tokens, top_k))
        w = torch.rand(n_tokens, top_k)

        out = experts(x, idx, w)

        ref = torch.zeros_like(x)
        for t in range(n_tokens):
            for k in range(top_k):
                e = getattr(experts, str(int(idx[t, k])))
                h = nn.functional.silu(e.gate_proj(x[t])) * e.up_proj(x[t])
                ref[t] += w[t, k] * e.down_proj(h)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def _wrapped_linear(self, bits, seed):
        from auto_round.wrapper import WrapperLinear

        layer = _linear()
        torch.manual_seed(seed)
        with torch.no_grad():
            layer.weight.copy_(torch.randn_like(layer.weight))
        layer.bits = bits
        layer.group_size = -1
        layer.sym = True
        layer.data_type = "int"
        layer.iters = 0
        layer.act_bits = 16
        layer.scale_dtype = torch.float16
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

    def test_mixed_bits_across_experts_falls_back(self):
        # different quant signatures inside one device group must reject the
        # grouped path (sharing one grouped GEMM across differently-quantized
        # experts is unsound), not silently merge them. Plain Linears carry no
        # quant signature at all (None == None), so the mix needs wrappers.
        experts = _Experts(2)
        getattr(experts, "0").up_proj = self._wrapped_linear(bits=4, seed=1)
        getattr(experts, "1").up_proj = self._wrapped_linear(bits=8, seed=2)
        x = torch.randn(6, 4)
        idx = torch.tensor([[0, 1]] * 6)  # both experts active
        w = torch.rand(6, 2)
        self.assertIsNone(_run_routes(experts, x, idx, w, experts.n))

    def test_weightless_packed_linear_falls_back(self):
        # packed QuantLinear types (e.g. MXFP4QuantLinear) store packed weights
        # and expose no .weight; the support probe must reject them BEFORE any
        # device read instead of raising AttributeError (CI: qwen3-vl-moe-mxfp)
        experts = _Experts(2)
        getattr(experts, "0").up_proj = nn.Module()  # no .weight, not a wrapper
        x = torch.randn(6, 4)
        idx = torch.tensor([[0, 1]] * 6)
        w = torch.rand(6, 2)
        self.assertIsNone(_run_routes(experts, x, idx, w, experts.n))

    def test_missing_slot_falls_back(self):
        torch.manual_seed(3)
        experts = _Experts(2)
        object.__setattr__(getattr(experts, "1"), "up_proj", None)  # slot absent on expert 1
        x = torch.randn(6, 4)
        idx = torch.tensor([[0, 1]] * 6)
        w = torch.rand(6, 2)
        self.assertIsNone(_run_routes(experts, x, idx, w, experts.n))


@unittest.skipIf(
    not (torch.cuda.is_available() and torch.cuda.device_count() >= 2),
    "needs >=2 CUDA devices",
)
class TestRunRoutesMultiDeviceCUDA(unittest.TestCase):
    def test_two_device_groups_match_reference(self):
        torch.manual_seed(11)
        experts = _Experts(4)  # tiny CPU build; nn.Module.to() cannot copy out of meta
        # experts 0,1 on cuda:0; 2,3 on cuda:1 (weights move explicitly)
        for i in range(4):
            e = getattr(experts, str(i))
            dev = f"cuda:{i // 2}"
            for slot in ("up_proj", "down_proj", "gate_proj"):
                getattr(e, slot).to(dev)
        n_tokens, n_experts, top_k, hidden = 64, 4, 2, 4
        x = torch.randn(n_tokens, hidden, device="cuda:2")  # input on a third device
        idx = torch.randint(0, n_experts, (n_tokens, top_k), device="cuda:2")
        w = torch.rand(n_tokens, top_k, device="cuda:2")

        out = experts(x, idx, w)
        self.assertEqual(out.device, x.device)

        # reference on cuda:2 by explicit per-expert moves
        ref = torch.zeros_like(x)
        for t in range(n_tokens):
            for k in range(top_k):
                e = getattr(experts, str(int(idx[t, k])))
                dev = next(p.device for p in e.up_proj.parameters())
                xt = x[t].to(dev)
                h = nn.functional.silu(e.gate_proj(xt)) * e.up_proj(xt)
                ref[t] += (w[t, k] * e.down_proj(h).to(x.device)).to(x.device)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


class TestAtomicPlacement(unittest.TestCase):
    """The allocator must never split one expert's projections across devices."""

    def _layer_dict(self, n_experts=6):
        d = {}
        for i in range(n_experts):
            for slot in ("gate_proj", "up_proj", "down_proj"):
                d[f"mlp.experts.{i}.{slot}"] = {"param_memory": 1.0, "is_moe_expert": True}
        d["self_attn.q_proj"] = {"param_memory": 1.0, "is_moe_expert": False}
        d["mlp.shared_experts.gate_proj"] = {"param_memory": 1.0, "is_moe_expert": False}
        return d

    def test_preassign_keeps_experts_atomic_and_spreads(self):
        from auto_round.utils.device import _preassign_moe_experts

        layer_dict = self._layer_dict(8)
        budgets = {"cuda:0": 12.0, "cuda:1": 12.0}
        assigned = _preassign_moe_experts(layer_dict, budgets, ["cuda:0", "cuda:1"], mem_per_param=1.0)
        self.assertTrue(assigned)
        # every expert's three slots share one device
        for i in range(8):
            devs = {assigned[f"mlp.experts.{i}.{s}"] for s in ("gate_proj", "up_proj", "down_proj")}
            self.assertEqual(len(devs), 1, f"expert {i} split: {devs}")
        # spread across both devices
        self.assertEqual({d for d in assigned.values()}, {"cuda:0", "cuda:1"})

    def test_preassign_shrinks_chunk_on_overflow(self):
        from auto_round.utils.device import _preassign_moe_experts

        # budgets so tight that a 16-expert chunk cannot fit, but single experts can
        layer_dict = self._layer_dict(6)
        budgets = {"cuda:0": 3.5, "cuda:1": 3.5}
        assigned = _preassign_moe_experts(layer_dict, budgets, ["cuda:0", "cuda:1"], mem_per_param=1.0)
        for i in range(6):
            devs = {assigned.get(f"mlp.experts.{i}.{s}") for s in ("gate_proj", "up_proj", "down_proj")}
            self.assertEqual(len(devs), 1, f"expert {i} split or unplaced: {devs}")

    def test_balancer_groups_experts_atomically(self):
        from auto_round.utils.device import _allocate_layers_to_devices

        layer_dict = self._layer_dict(4)
        budgets = {"cuda:0": 100.0, "cuda:1": 100.0}
        device_map, _names = _allocate_layers_to_devices(layer_dict, budgets, ["cuda:0", "cuda:1"], 1.0)
        for i in range(4):
            devs = {device_map[f"mlp.experts.{i}.{s}"] for s in ("gate_proj", "up_proj", "down_proj")}
            self.assertEqual(len(devs), 1, f"expert {i} split by balancer: {devs}")
