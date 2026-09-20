# Copyright (c) 2025 Intel Corporation
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

"""Tests for lm_head chain-tail inputs (block-loop chain output -> final norm -> lm_head).

Covers: row extraction from list/dict chain states, lm_head name resolution,
the tail derivation (norm applied, width sanity, closed-form fallbacks), the
early-stop override gate, and the outside-block lane consuming tail inputs
without issuing capture passes.
"""

import logging
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

import auto_round.compressors.orchestrator as orch
from auto_round.compressors.orchestrator import CompressionOrchestrator


class _TinyModel(nn.Module):
    def __init__(self, hidden=8, vocab=16):
        super().__init__()
        self.model = nn.Module()
        self.model.norm = nn.RMSNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.hf_device_map = {}


def _orchestrator_like(model):
    o = SimpleNamespace(
        model_context=SimpleNamespace(model=model),
        _tail_fed_layers_=[],
        _lm_head_chain_tail_=None,
        _lm_head_norm_name_=None,
    )
    o._chain_hidden_rows = CompressionOrchestrator._chain_hidden_rows  # staticmethod: bind directly
    for name in (
        "_resolve_lm_head_name_",
        "_lm_head_tail_inputs_",
        "_discover_final_norm_",
        "_attach_tail_imatrix_",
        "_quantizer_requests_q_inputs_",
    ):
        setattr(o, name, MethodType(getattr(CompressionOrchestrator, name), o))
    return o


class TestChainHiddenRows:
    def test_plain_list_passthrough(self):
        rows = [torch.zeros(1, 4) for _ in range(3)]
        assert CompressionOrchestrator._chain_hidden_rows(rows) is rows

    def test_dict_with_hidden_states(self):
        rows = [torch.zeros(1, 4) for _ in range(2)]
        assert CompressionOrchestrator._chain_hidden_rows({"hidden_states": rows}) is rows

    def test_structured_output_takes_hidden_states(self):
        rows = [torch.zeros(1, 4)]
        state = {"hidden_states": rows, "conv": [torch.zeros(1)]}
        assert CompressionOrchestrator._chain_hidden_rows(state) is rows

    def test_nested_dict_under_hidden_states(self):
        rows = [torch.zeros(1, 4)]
        assert CompressionOrchestrator._chain_hidden_rows({"hidden_states": {"inner": rows}}) is rows


class TestResolveLmHeadName:
    def setup_method(self):
        self.o = _orchestrator_like(_TinyModel())

    def test_exact_leaf_match(self):
        assert self.o._resolve_lm_head_name_(["lm_head"]) == "lm_head"

    def test_dotted_leaf_match(self):
        assert self.o._resolve_lm_head_name_(["model.lm_head", "other"]) == "model.lm_head"

    def test_substring_fallback(self):
        assert self.o._resolve_lm_head_name_(["proj.lm_head_w8"]) == "proj.lm_head_w8"

    def test_none_when_absent(self):
        assert self.o._resolve_lm_head_name_(["embed_tokens"]) is None
        assert self.o._resolve_lm_head_name_([]) is None


class TestLmHeadTailInputs:
    def setup_method(self):
        self.model = _TinyModel()
        self.o = _orchestrator_like(self.model)
        self.o._lm_head_norm_name_ = "model.norm"
        # SignRound-style default: the quantized-input chain is requested
        self.o.alg_composer = SimpleNamespace(block_quantizer=SimpleNamespace(enable_quanted_input=True))

    def _rows(self, scale=1.0):
        return [torch.randn(1, 5, 8) * scale for _ in range(3)]

    def test_norm_applied_to_fp_and_q_rows(self):
        fp, q = self._rows(), self._rows(2.0)
        self.o._lm_head_chain_tail_ = (q, fp)
        out = self.o._lm_head_tail_inputs_("lm_head")
        assert out is not None
        norm = self.model.model.norm
        with torch.no_grad():
            expected_fp = [norm(r) for r in fp]
            expected_q = [norm(r) for r in q]
        for got, exp in zip(out[0], expected_fp):
            assert torch.allclose(got, exp, atol=1e-6)
        for got, exp in zip(out[1], expected_q):
            assert torch.allclose(got, exp, atol=1e-6)

    def test_norm_runs_on_weight_device_rows_park_on_host(self):
        """Cross-device contract: the row handed to the norm sits on the norm's
        weight device, the returned rows park on the host (the tune loop
        streams them per micro-batch). Chain-tail rows can be cuda-resident
        while the norm's weight lives elsewhere - mixing them crashes RMSNorm."""
        fp, q = self._rows(), self._rows(2.0)
        self.o._lm_head_chain_tail_ = (q, fp)
        norm = self.model.model.norm
        seen_devices = []
        orig_forward = norm.forward

        def recording_forward(x):
            seen_devices.append(x.device)
            return orig_forward(x)

        norm.forward = recording_forward
        try:
            out = self.o._lm_head_tail_inputs_("lm_head")
        finally:
            norm.forward = orig_forward
        assert out is not None
        wdev = norm.weight.device
        assert all(d == wdev for d in seen_devices), f"norm saw rows on {seen_devices}, weight on {wdev}"
        assert all(r.device.type == "cpu" for r in out[0])
        assert all(r.device.type == "cpu" for r in out[1])

    def test_missing_tail_falls_back(self):
        assert self.o._lm_head_tail_inputs_("lm_head") is None

    def test_bad_row_format_falls_back(self):
        self.o._lm_head_chain_tail_ = (None, "not-rows")
        assert self.o._lm_head_tail_inputs_("lm_head") is None

    def test_width_mismatch_falls_back(self):
        self.model.model.norm = nn.RMSNorm(4)  # wrong width vs lm_head.in_features=8
        self.o._lm_head_chain_tail_ = (self._rows(), self._rows())
        assert self.o._lm_head_tail_inputs_("lm_head") is None

    def test_missing_norm_falls_back(self):
        self.o._lm_head_norm_name_ = None
        self.o._lm_head_chain_tail_ = (self._rows(), self._rows())
        assert self.o._lm_head_tail_inputs_("lm_head") is None

    def test_malformed_q_rows_degrade_to_fp_only(self):
        self.o._lm_head_chain_tail_ = (None, self._rows())
        self.o.alg_composer = SimpleNamespace(block_quantizer=SimpleNamespace(enable_quanted_input=True))
        out = self.o._lm_head_tail_inputs_("lm_head")
        assert out is not None and out[1] is None

    def _capturing_logger(self):
        import auto_round.compressors.orchestrator as orch_mod

        records = []

        class _Rec:
            def info(self, msg, *a):
                records.append((logging.INFO, msg % a if a else msg))

            def warning(self, msg, *a):
                records.append((logging.WARNING, msg % a if a else msg))

        orig = orch_mod.logger
        orch_mod.logger = _Rec()
        return records, lambda: setattr(orch_mod, "logger", orig)

    def test_q_absence_by_config_is_info_not_warning(self):
        """RTN (iters=0) defaults enable_quanted_input=False: the missing q chain
        is the configured path - no WARNING, just an info line."""
        self.o._lm_head_chain_tail_ = (None, self._rows())
        self.o.alg_composer = SimpleNamespace(block_quantizer=SimpleNamespace(enable_quanted_input=False))
        records, restore = self._capturing_logger()
        try:
            out = self.o._lm_head_tail_inputs_("lm_head")
        finally:
            restore()
        assert out is not None and out[1] is None
        assert all(lvl < logging.WARNING for lvl, _ in records)
        assert any("disabled by config" in msg for _, msg in records)

    def test_q_absence_when_requested_stays_warning(self):
        self.o._lm_head_chain_tail_ = (None, self._rows())
        self.o.alg_composer = SimpleNamespace(block_quantizer=SimpleNamespace(enable_quanted_input=True))
        records, restore = self._capturing_logger()
        try:
            self.o._lm_head_tail_inputs_("lm_head")
        finally:
            restore()
        assert any(lvl == logging.WARNING and "cannot be honored" in msg for lvl, msg in records)

    def test_structured_chain_state(self):
        fp = {"hidden_states": self._rows()}
        q = {"hidden_states": self._rows(2.0)}
        self.o._lm_head_chain_tail_ = (q, fp)
        out = self.o._lm_head_tail_inputs_("lm_head")
        assert out is not None and len(out[0]) == 3


class TestFinalNormDiscovery:
    """The discovery must cover LayerNorm-with-bias finals (GPT-J/OPT class),
    not just single-param RMSNorms (Qwen/LLaMA class)."""

    def _disc(self, model, blocks):
        return _orchestrator_like(model)._discover_final_norm_(blocks)

    def test_layernorm_with_bias_final_is_found(self):
        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = nn.Module()
                self.body.blocks = nn.Module()  # block prefix marker
                self.ln_f = nn.LayerNorm(8)
                self.lm_head = nn.Linear(8, 16)

        assert self._disc(_M(), [["body.blocks"]]) == "ln_f"

    def test_rmsnorm_single_param_final_still_found(self):
        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = nn.Module()
                self.body.blocks = nn.Module()
                self.norm = nn.RMSNorm(8)
                self.lm_head = nn.Linear(8, 16)

        assert self._disc(_M(), [["body.blocks"]]) == "norm"

    def test_block_internal_norms_excluded(self):
        class _M(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = nn.Module()
                self.body.blocks = nn.Module()
                self.body.blocks.ln_1 = nn.LayerNorm(8)
                self.ln_f = nn.LayerNorm(8)
                self.lm_head = nn.Linear(8, 16)

        assert self._disc(_M(), [["body.blocks"]]) == "ln_f"


class TestTailImatrix:
    """The fp-input imatrix must equal what the quantizer hook would accumulate."""

    def _rows(self):
        torch.manual_seed(0)
        return [torch.randn(1, 7, 8) for _ in range(3)]

    def test_imatrix_matches_hook_math(self):
        model = _TinyModel()
        o = _orchestrator_like(model)
        rows = self._rows()
        o._attach_tail_imatrix_("lm_head", rows)
        expected = None
        n = 0
        for row in rows:
            flat = row.reshape(-1, row.shape[-1]).to(torch.float32)
            sq = torch.sum(flat.pow(2), dim=0)
            expected = sq if expected is None else expected + sq
            n += flat.shape[0]
        assert hasattr(model.lm_head, "imatrix")
        assert torch.allclose(model.lm_head.imatrix, expected, atol=1e-5)
        assert model.lm_head.imatrix_cnt == n

    def test_imatrix_never_overwrites_existing(self):
        model = _TinyModel()
        o = _orchestrator_like(model)
        model.lm_head.imatrix = torch.ones(8)
        o._attach_tail_imatrix_("lm_head", self._rows())
        assert torch.equal(model.lm_head.imatrix, torch.ones(8))

    def test_imatrix_skipped_without_rows(self):
        model = _TinyModel()
        o = _orchestrator_like(model)
        o._attach_tail_imatrix_("lm_head", [])
        assert not hasattr(model.lm_head, "imatrix")


class TestLaneConsumesTailInputs:
    """The outside-block lane feeds lm_head from the tail and skips capture passes."""

    def _run_lane(self, monkeypatch):
        model = _TinyModel()
        o = _orchestrator_like(model)
        o._tail_fed_layers_ = ["lm_head"]
        o._lm_head_norm_name_ = "model.norm"
        fp, q = [torch.randn(1, 5, 8) for _ in range(2)], [torch.randn(1, 5, 8) for _ in range(2)]
        o._lm_head_chain_tail_ = (q, fp)

        captured_calls = []

        class _Composer:
            def need_quanted_input(self):
                return True

            def compress_layer_outside_block(self, layer, fp_inputs=None, q_inputs=None, **kw):
                # snapshot the lists: the lane frees them in place afterwards
                # (clear_memory nulls elements), which must not hide what ran
                captured_calls.append(
                    (
                        [r for r in fp_inputs] if fp_inputs is not None else None,
                        [r for r in q_inputs] if q_inputs is not None else None,
                        o._lm_head_chain_tail_ is None,
                    )
                )

        o.alg_composer = _Composer()
        o.compress_context = SimpleNamespace(
            is_immediate_packing=False, is_immediate_saving=False, cache_device="cuda:0"
        )
        o.calibration_context = SimpleNamespace(nsamples=2)
        o.formats = []
        o.act_bits = 16
        o.act_dynamic = True

        cache_calls = []

        def _no_cache_data(*args, **kwargs):
            cache_calls.append((args, kwargs))
            return {}

        o.cache_data = _no_cache_data
        o.model = model
        monkeypatch.setattr(orch, "memory_monitor", SimpleNamespace(update=lambda: None, log_summary=lambda: None))

        lane = MethodType(orch.CompressionOrchestrator._quantize_layers_outside_blocks, o)
        lane(["lm_head"], {}, token_ids=None)
        return captured_calls, cache_calls, o, (q, fp)

    def test_lane_passes_tail_rows_and_skips_capture(self, monkeypatch):
        captured_calls, cache_calls, o, (raw_q, raw_fp) = self._run_lane(monkeypatch)
        assert len(captured_calls) == 1
        fp_used, q_used, tail_released = captured_calls[0]
        assert fp_used is not None and q_used is not None
        # host parking by design: no cache-device pull for tail rows, and the
        # raw tail is released BEFORE the tune loop builds its buffers
        assert all(r.device.type == "cpu" for r in fp_used)
        assert all(r.device.type == "cpu" for r in q_used)
        assert tail_released
        norm = o.model_context.model.model.norm
        with torch.no_grad():
            exp_fp = [norm(r) for r in raw_fp]
            exp_q = [norm(r) for r in raw_q]
        for got, exp in zip(fp_used, exp_fp):
            assert torch.allclose(got, exp, atol=1e-6)
        for got, exp in zip(q_used, exp_q):
            assert torch.allclose(got, exp, atol=1e-6)
        # the tail rows were consumed and released; no capture pass was issued at all
        assert o._lm_head_chain_tail_ is None
        assert cache_calls == []

    def test_lane_attaches_tail_imatrix(self, monkeypatch):
        captured_calls, cache_calls, o, _ = self._run_lane(monkeypatch)
        lm = o.model_context.model.lm_head
        assert hasattr(lm, "imatrix") and lm.imatrix.numel() == lm.in_features
        # parity with the hook math over the exact rows the lane received
        rows = captured_calls[0][0]
        expected = None
        for row in rows:
            flat = row.reshape(-1, row.shape[-1]).to(torch.float32)
            sq = torch.sum(flat.pow(2), dim=0)
            expected = sq if expected is None else expected + sq
        assert torch.allclose(lm.imatrix, expected, atol=1e-5)
        assert lm.imatrix_cnt == sum(r.numel() // r.shape[-1] for r in rows)
