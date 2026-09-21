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
the tail derivation (mocked capture through the model's own post-block code,
RTN fallbacks), the
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
        _tail_stub_arity_=None,
        _tail_lane_blocks_=[],
        _offloader=None,
    )
    o._chain_hidden_rows = CompressionOrchestrator._chain_hidden_rows  # staticmethod: bind directly
    for name in (
        "_lm_head_tail_inputs_",
        "_mocked_tail_capture_",
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
        model = _ForwardModel()
        o = _orchestrator_like(model)
        o._tail_fed_layers_ = ["lm_head"]
        o._tail_stub_arity_ = 1
        o._tail_lane_blocks_ = ["model.layers.0", "model.layers.1"]
        o._offloader = None
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


class _ForwardModel(nn.Module):
    """A runnable transformers-shaped model: ModuleList body, post-block tail, head."""

    def __init__(self, hidden=8, vocab=16, n_layers=2, scale=None):
        super().__init__()
        self.hidden_size = hidden
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.model.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.head_scale = scale  # minicpm3-style pre-head scalar (Class A glue)
        self.hf_device_map = {}

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.model.embed(input_ids) if hasattr(self.model, "embed") else input_ids.float().unsqueeze(-1)
        hidden = hidden.expand(-1, -1, self.hidden_size).clone()
        for layer in self.model.layers:
            out = layer(hidden)
            hidden = out[0] if isinstance(out, tuple) else out
        hidden = self.model.norm(hidden)
        if self.head_scale is not None:
            hidden = hidden / self.head_scale
        return self.lm_head(hidden)


def _capturing_logger_():
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


def _capture_orchestrator(model, arity=1, blocks=None):
    o = SimpleNamespace(
        model_context=SimpleNamespace(model=model),
        _tail_fed_layers_=[],
        _lm_head_chain_tail_=None,
        _tail_stub_arity_=arity,
        _tail_lane_blocks_=blocks if blocks is not None else ["model.layers.0", "model.layers.1"],
        _offloader=None,
    )
    o._chain_hidden_rows = CompressionOrchestrator._chain_hidden_rows
    o._quantizer_requests_q_inputs_ = MethodType(CompressionOrchestrator._quantizer_requests_q_inputs_, o)
    for name in ("_lm_head_tail_inputs_", "_attach_tail_imatrix_", "_mocked_tail_capture_"):
        setattr(o, name, MethodType(getattr(CompressionOrchestrator, name), o))
    return o


def _rows(n=3, b=1, s=4, h=8, gen=None):
    gen = gen or torch.Generator().manual_seed(0)
    return [torch.randn(b, s, h, generator=gen) for _ in range(n)]


def _ids_like(rows):
    return [torch.zeros(r.shape[0], r.shape[1], dtype=torch.long) for r in rows]


class TestMockedTailCapture:
    """The capture pass: stubs + injector + CaptureHead run the model's own post-block code."""

    def test_captured_rows_match_the_models_own_norm(self):
        model = _ForwardModel()
        o = _capture_orchestrator(model)
        fp = _rows()
        out = o._mocked_tail_capture_("lm_head", fp, None, _ids_like(fp))
        assert out is not None
        with torch.no_grad():
            expected = [model.model.norm(r) for r in fp]
        for got, exp in zip(out[0], expected):
            assert torch.allclose(got, exp, atol=1e-6)
        assert out[1] is None

    def test_captured_rows_include_pre_head_scalar_glue(self):
        model = _ForwardModel(scale=16.0)
        o = _capture_orchestrator(model)
        fp = _rows()
        out = o._mocked_tail_capture_("lm_head", fp, None, _ids_like(fp))
        assert out is not None
        with torch.no_grad():
            expected = [model.model.norm(r) / 16.0 for r in fp]
        for got, exp in zip(out[0], expected):
            assert torch.allclose(got, exp, atol=1e-6)

    def test_q_variant_captured_when_provided(self):
        model = _ForwardModel()
        o = _capture_orchestrator(model)
        fp, q = _rows(), _rows()
        out = o._mocked_tail_capture_("lm_head", fp, q, _ids_like(fp))
        assert out is not None and out[1] is not None and len(out[1]) == len(fp)

    def test_missing_smoke_metadata_returns_none(self):
        o = _capture_orchestrator(_ForwardModel(), arity=None)
        out = o._mocked_tail_capture_("lm_head", _rows(), None, _ids_like(_rows()))
        assert out is None

    def test_raising_model_returns_none_with_warning(self):
        model = _ForwardModel()

        def broken(*a, **k):
            raise RuntimeError("boom")

        model.forward = broken
        o = _capture_orchestrator(model)
        records, restore = _capturing_logger_()
        try:
            out = o._mocked_tail_capture_("lm_head", _rows(), None, _ids_like(_rows()))
        finally:
            restore()
        assert out is None
        assert any(lvl >= logging.WARNING and "falls back" in msg for lvl, msg in records)

    def test_head_and_layers_restored_after_capture(self):
        model = _ForwardModel()
        layers_before = list(model.model.layers)
        head_before = model.lm_head
        o = _capture_orchestrator(model)
        o._mocked_tail_capture_("lm_head", _rows(), None, _ids_like(_rows()))
        assert model.lm_head is head_before
        for before, after in zip(layers_before, model.model.layers):
            assert before is after

    def test_lm_head_tail_inputs_end_to_end(self):
        model = _ForwardModel()
        o = _capture_orchestrator(model)
        o._lm_head_chain_tail_ = (None, _rows())
        o.alg_composer = SimpleNamespace(block_quantizer=SimpleNamespace(enable_quanted_input=False))
        out = o._lm_head_tail_inputs_("lm_head", token_ids=_ids_like(_rows()))
        assert out is not None
        with torch.no_grad():
            expected = [model.model.norm(r) for r in _rows()]
        for got, exp in zip(out[0], expected):
            assert torch.allclose(got, exp, atol=1e-6)

    def test_missing_tail_falls_back(self):
        o = _capture_orchestrator(_ForwardModel())
        o._lm_head_chain_tail_ = None
        records, restore = _capturing_logger_()
        try:
            out = o._lm_head_tail_inputs_("lm_head")
        finally:
            restore()
        assert out is None
        assert any("no chain tail" in msg for _, msg in records)

    def test_bad_row_format_falls_back(self):
        o = _capture_orchestrator(_ForwardModel())
        o._lm_head_chain_tail_ = (None, {"hidden_states": "not-a-list"})
        assert o._lm_head_tail_inputs_("lm_head") is None

    def test_malformed_q_rows_degrade_to_fp_only(self):
        model = _ForwardModel()
        o = _capture_orchestrator(model)
        o._lm_head_chain_tail_ = (None, _rows())
        o.alg_composer = SimpleNamespace(block_quantizer=SimpleNamespace(enable_quanted_input=False))
        # q variant malformed relative to fp rows: capture proceeds fp-only
        out = o._lm_head_tail_inputs_("lm_head", token_ids=_ids_like(_rows()))
        assert out is not None and out[1] is None

    def test_q_absence_by_config_is_info_not_warning(self):
        model = _ForwardModel()
        o = _capture_orchestrator(model)
        o._lm_head_chain_tail_ = (None, _rows())
        o.alg_composer = SimpleNamespace(block_quantizer=SimpleNamespace(enable_quanted_input=False))
        records, restore = _capturing_logger_()
        try:
            out = o._lm_head_tail_inputs_("lm_head", token_ids=_ids_like(_rows()))
        finally:
            restore()
        assert out is not None and out[1] is None
        assert any("disabled by config" in msg for lvl, msg in records if lvl == logging.INFO)
        assert not any("cannot be honored" in msg for _, msg in records)

    def test_q_absence_when_requested_stays_warning(self):
        model = _ForwardModel()
        o = _capture_orchestrator(model)
        o._lm_head_chain_tail_ = (None, _rows())
        o.alg_composer = SimpleNamespace(block_quantizer=SimpleNamespace(enable_quanted_input=True))
        records, restore = _capturing_logger_()
        try:
            out = o._lm_head_tail_inputs_("lm_head", token_ids=_ids_like(_rows()))
        finally:
            restore()
        assert out is not None
        assert any("cannot be honored" in msg for lvl, msg in records if lvl >= logging.WARNING)

    def test_meta_chain_module_reloaded_via_offloader(self):
        import torch.nn as nn

        model = _ForwardModel()

        class _FakeOffloader:
            def __init__(self, model):
                self.model = model
                self.reloaded = []

            def reload(self, model, name):
                self.reloaded.append(name)
                # fake the reload: swap meta parameters for real cpu ones
                mod = None
                for n, m in model.named_modules():
                    if n == name:
                        mod = m
                for pname, p in list(mod.named_parameters(recurse=False)):
                    if p.is_meta:
                        setattr(mod, pname, nn.Parameter(torch.ones_like(p, device="cpu")))

        offloader = _FakeOffloader(model)
        with torch.no_grad():
            model.model.norm.weight = nn.Parameter(torch.empty(8, device="meta"))
        o = _capture_orchestrator(model)
        o._offloader = offloader
        fp = _rows()
        out = o._mocked_tail_capture_("lm_head", fp, None, _ids_like(fp))
        assert out is not None
        assert any("norm" in n for n in offloader.reloaded)

    def test_meta_chain_module_without_offloader_returns_none(self):
        import torch.nn as nn

        model = _ForwardModel()
        with torch.no_grad():
            model.model.norm.weight = nn.Parameter(torch.empty(8, device="meta"))
        o = _capture_orchestrator(model)
        o._offloader = None
        out = o._mocked_tail_capture_("lm_head", _rows(), None, _ids_like(_rows()))
        assert out is None

    def test_negative_padding_ids_are_clamped(self):
        model = _ForwardModel()
        o = _capture_orchestrator(model)
        fp = _rows(n=2)
        ids = [torch.tensor([[-100, -100, 5, 7]]), torch.tensor([[3, -100, 9, 1]])]
        out = o._mocked_tail_capture_("lm_head", fp, None, token_ids=ids)
        assert out is not None and len(out[0]) == 2

    def test_records_mismatch_returns_none(self):
        model = _ForwardModel()

        def once_then_silent(input_ids, attention_mask=None, **kwargs):
            # forward that stops calling the head after the first sample
            if not hasattr(model, "_calls"):
                model._calls = 0
            model._calls += 1
            if model._calls > 1:
                raise RuntimeError("simulated early stop before the head")
            return _ForwardModel.forward(model, input_ids, attention_mask, **kwargs)

        model.forward = once_then_silent
        o = _capture_orchestrator(model)
        layers_before = list(model.model.layers)
        head_before = model.lm_head
        out = o._mocked_tail_capture_("lm_head", _rows(n=2), None, _ids_like(_rows(n=2)))
        assert out is None
        # restore still happened on the failure path
        assert model.lm_head is head_before
        for before, after in zip(layers_before, model.model.layers):
            assert before is after


class TestTailLaneDecision:
    """Init-time lane selection: smoke PASS -> tail-fed; FAIL -> capture walk + restrictions."""

    @staticmethod
    def _decider(model):
        o = _orchestrator_like(model)
        o.inplace = True
        o.formats = []
        o.compress_context = SimpleNamespace(is_immediate_packing=True, is_immediate_saving=False)
        o._decide_tail_fed_lane_ = MethodType(CompressionOrchestrator._decide_tail_fed_lane_, o)
        return o

    def _model_with_blocks(self):
        model = _TinyModel()
        model.model.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])
        return model

    def test_smoke_pass_selects_tail_fed_lane(self, monkeypatch):
        import auto_round.compressors.orchestrator as orch_mod

        calls = []

        def fake_smoke(model, lm_head_name, block_names, seq_len=2):
            calls.append((lm_head_name, list(block_names)))
            return (True, 1)

        monkeypatch.setattr(orch_mod, "tail_smoke_check", fake_smoke)
        o = self._decider(self._model_with_blocks())
        o._decide_tail_fed_lane_(["lm_head"], [["model.layers.0", "model.layers.1"]])
        assert o._tail_fed_layers_ == ["lm_head"]
        assert o._tail_stub_arity_ == 1
        assert o._tail_lane_blocks_ == ["model.layers.0", "model.layers.1"]
        # the tail-only relaxations stay granted on the pass path
        assert o.inplace is True and o.compress_context.is_immediate_packing is True
        assert calls == [("lm_head", ["model.layers.0", "model.layers.1"])]

    def test_smoke_fail_keeps_capture_walk_and_restores_restrictions(self, monkeypatch):
        import auto_round.compressors.orchestrator as orch_mod

        monkeypatch.setattr(orch_mod, "tail_smoke_check", lambda *a, **k: (False, None))
        o = self._decider(self._model_with_blocks())
        records, restore = _capturing_logger_()
        try:
            o._decide_tail_fed_lane_(["lm_head"], [["model.layers.0", "model.layers.1"]])
        finally:
            restore()
        assert o._tail_fed_layers_ == []
        assert o._tail_stub_arity_ is None
        # the relaxations granted on the tail-only premise are reverted
        assert o.inplace is False
        assert o.compress_context.is_immediate_packing is False
        assert any(lvl >= logging.WARNING and "keeps the ordinary capture walk" in msg for lvl, msg in records)

    def test_smoke_fail_reverts_immediate_saving_with_packing(self, monkeypatch):
        import auto_round.compressors.orchestrator as orch_mod

        monkeypatch.setattr(orch_mod, "tail_smoke_check", lambda *a, **k: (False, None))
        o = self._decider(self._model_with_blocks())
        o.compress_context.is_immediate_saving = True  # granted only while packing is on
        o._decide_tail_fed_lane_(["lm_head"], [["model.layers.0"]])
        # the granted pair reverts together: saving is never on while packing is off
        assert o.compress_context.is_immediate_packing is False
        assert o.compress_context.is_immediate_saving is False

    def test_head_outside_plan_skips_smoke(self, monkeypatch):
        import auto_round.compressors.orchestrator as orch_mod

        def fail_if_called(*a, **k):
            raise AssertionError("smoke must not run when the head is outside the plan")

        monkeypatch.setattr(orch_mod, "tail_smoke_check", fail_if_called)
        o = self._decider(self._model_with_blocks())
        o._decide_tail_fed_lane_(["other.layer"], [["model.layers.0", "model.layers.1"]])
        assert o._tail_fed_layers_ == []
        # relaxations granted elsewhere are left alone when the lane never engages
        assert o.inplace is True and o.compress_context.is_immediate_packing is True


class TestStreamShapeGuard:
    """Stream-split tails must decline loudly instead of returning mangled rows."""

    def test_wrong_width_rows_return_none(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])
                self.lm_head = nn.Linear(8, 16)

            def forward(self, input_ids, attention_mask=None, **kwargs):
                hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 8).clone()
                # ngram-style split: feed a narrower stream than the head input width
                return self.lm_head(hidden[..., :4].reshape(-1, 4))

        o = _capture_orchestrator(Model(), arity=1)
        assert o._tail_lane_blocks_  # lane engaged
        captured = o._mocked_tail_capture_("lm_head", [torch.randn(2, 8)], None, token_ids=None)
        assert captured is None  # shape validation declined the capture
