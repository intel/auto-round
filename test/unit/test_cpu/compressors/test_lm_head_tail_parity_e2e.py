# coding=utf-8
# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end parity: lm_head tuned from the calibration chain tail (chunked,
no whole-model capture) must reproduce the full-residency capture path.

Both arms run the real data-driven pipeline on the same tiny model with the
same synthetic calibration data and the same seed; the only difference is
where lm_head's fp/q rows come from:

- capture arm: lm_head stays in the collection walk's layer targets (the
  pre-branch behavior - whole-model forward per batch, capture passes);
- tail arm: the branch behavior - the block loop's chain output through the
  final norm, host-parked, no capture passes.

The tail lane's imatrix attachment is neutralized for the result-parity arm:
the capture path never populates an imatrix for outside-block layers (its
hooks fire only inside per-block compression), so equalizing on "no imatrix"
isolates the chunking/tail mechanism - the imatrix feature has its own
hook-math parity tests.
"""

import pytest
import torch

import auto_round.algorithms.composer as composer_mod
import auto_round.compressors.orchestrator as orchestrator_mod
from auto_round import AutoRound
from auto_round.utils import get_module


class _Loader:
    """Deterministic synthetic calibration data - same batches every run."""

    def __init__(self, vocab, seed=1234, batches=2, seqlen=16):
        g = torch.Generator().manual_seed(seed)
        self.batches = [torch.randint(0, vocab - 1, (1, seqlen), generator=g) for _ in range(batches)]

    def __iter__(self):
        yield from self.batches


class _RowSpy:
    """Records what the lane feeds to compress_layer_outside_block."""

    def __init__(self):
        self.calls = {}

    def install(self, monkeypatch):
        spy = self
        orig = composer_mod.AlgorithmComposer.compress_layer_outside_block

        def wrapper(self_, layer, fp_inputs=None, q_inputs=None, **kw):
            name = getattr(layer, "global_name", None)
            if name and "lm_head" in name:
                spy.calls[name] = (
                    [r.detach().clone() for r in fp_inputs] if fp_inputs is not None else None,
                    [r.detach().clone() for r in q_inputs] if q_inputs is not None else None,
                )
            return orig(self_, layer, fp_inputs=fp_inputs, q_inputs=q_inputs, **kw)

        monkeypatch.setattr(composer_mod.AlgorithmComposer, "compress_layer_outside_block", wrapper)


def _quantize_lmonly(tiny_gptj_model_path, loader):
    torch.manual_seed(1234)
    layer_config = {
        "transformer.h": {"bits": 16},
        "lm_head": {"bits": 4, "group_size": 32, "sym": True},
    }
    autoround = AutoRound(
        tiny_gptj_model_path,
        bits=4,
        group_size=32,
        sym=True,
        iters=2,
        seqlen=16,
        dataset=loader,
        layer_config=layer_config,
        amp=False,
    )
    autoround.quantize()
    return autoround


def _lm_head_weight(model):
    lm = get_module(model, "lm_head")
    return lm.weight.detach().clone()


class TestLmHeadTailParityE2E:
    def test_tail_rows_and_quantized_weight_match_capture_path(self, tiny_gptj_model_path, monkeypatch, tmp_path):
        from transformers import AutoConfig

        vocab = AutoConfig.from_pretrained(tiny_gptj_model_path).vocab_size

        # ── capture arm: full-residency walk collects lm_head's inputs ──────
        spy_capture = _RowSpy()
        spy_capture.install(monkeypatch)
        monkeypatch.setattr(
            orchestrator_mod.CompressionOrchestrator, "_resolve_lm_head_name_", lambda self, names: None
        )
        run_capture = _quantize_lmonly(tiny_gptj_model_path, _Loader(vocab))
        monkeypatch.undo()

        # ── tail arm: chain-tail rows, no capture passes ────────────────────
        spy_tail = _RowSpy()
        spy_tail.install(monkeypatch)
        monkeypatch.setattr(
            orchestrator_mod.CompressionOrchestrator, "_attach_tail_imatrix_", lambda self, name, rows: None
        )
        run_tail = _quantize_lmonly(tiny_gptj_model_path, _Loader(vocab))
        monkeypatch.undo()

        # both arms must have tuned lm_head through the lane
        assert spy_capture.calls and spy_tail.calls
        (cap_fp, cap_q), (tail_fp, tail_q) = (
            next(iter(spy_capture.calls.values())),
            next(iter(spy_tail.calls.values())),
        )
        assert cap_fp is not None and tail_fp is not None
        assert len(cap_fp) == len(tail_fp)

        # 1. row parity: the chain-tail derivation feeds the same calibration
        #    rows the whole-model capture walk collected
        for c, t in zip(cap_fp, tail_fp):
            assert c.shape == t.shape
            assert torch.allclose(c.float(), t.float(), atol=1e-5), (c - t).abs().max()
        if cap_q is not None and tail_q is not None:
            for c, t in zip(cap_q, tail_q):
                assert torch.allclose(c.float(), t.float(), atol=1e-5)

        # 2. result parity: same rows + same seed + same (absent) imatrix ->
        #    identical tuning -> identical quantized weights
        w_cap = _lm_head_weight(run_capture.model)
        w_tail = _lm_head_weight(run_tail.model)
        assert w_cap.shape == w_tail.shape
        assert torch.equal(w_cap, w_tail), (w_cap.float() - w_tail.float()).abs().max()
