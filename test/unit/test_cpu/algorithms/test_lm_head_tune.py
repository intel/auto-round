# coding=utf-8
# Copyright (c) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Chunked tuning for huge-output layers (lm_head-class vocabularies, iters>0).

The outside-block SignRound path chunks position forwards, row-blocks
fake-quant forwards and the final quantize, preallocates gradient buffers,
parks best-parameter snapshots on the host above a size threshold, and
streams unwrap parameters back window-by-window.
"""

import inspect
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import torch.nn as nn

from auto_round.compressors.orchestrator import CompressionOrchestrator


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8)


class _NoNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(4, 8)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(4, 8)


class _TrailingTree(nn.Module):
    """lm_head followed by an attached checkpoint-only placeholder subtree -
    the module-order "last leaf" lands inside the placeholder, not on lm_head."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8)
        self.mtp = nn.Module()
        self.mtp.pre_fc_norm_hidden = nn.LayerNorm(4)


def _outside_tune_harness(qmod, layer, cap):
    """Fresh pinned Linear + fake quantizer for quantize_layer_outside_block tests."""
    torch.manual_seed(11)
    fresh = torch.nn.Linear(16, 5)
    with torch.no_grad():
        fresh.weight.copy_(layer.weight)
        fresh.bias.copy_(layer.bias)
    fresh.global_name = "lm_head"
    fresh.bits = 8
    fresh.group_size = 32
    fresh.sym = True
    fresh.data_type = "int"
    fresh.scale_dtype = None
    fresh.iters = 3
    fresh.act_bits = 16
    fresh.act_sym = True
    fresh.act_data_type = None
    fresh.act_group_size = None
    cfg = type(
        "C",
        (),
        {
            "iters": 3,
            "lr": 5e-3,
            "compute_lr": lambda self, bits: None,
            "compute_minmax_lr": lambda self, bits: None,
        },
    )()
    quant = SimpleNamespace(
        config=cfg,
        _config=cfg,
        iters=3,
        lr=5e-3,
        minmax_lr=1e-3,
        lr_scheduler=None,
        enable_minmax_tuning=False,
        gradient_accumulate_steps=1,
        not_use_best_mse=False,
        dynamic_max_gap=0,
        optimizer=qmod.SignSGD,
        lr_is_auto=False,
        model=torch.nn.Module(),
        calibration_context=SimpleNamespace(batch_size=1),
        model_context=SimpleNamespace(amp=False, amp_dtype=torch.bfloat16),
        compress_context=SimpleNamespace(enable_torch_compile=False, cache_device="cpu"),
    )
    for name in (
        "_get_scaler",
        "_scale_loss_and_backward",
        "build_weight_qdq",
        "_step",
        "_maybe_log_low_bit_lr",
        "_compute_valid_token_mask",
    ):
        setattr(quant, name, MethodType(getattr(qmod.SignRoundQuantizer, name), quant))
    quant._preallocate_tuning_grads_ = qmod.SignRoundQuantizer._preallocate_tuning_grads_
    quant._logged_low_bit_lr = set()
    return quant, fresh


class TestOutsideTuneChunking:
    """Position-chunked forward/backward for huge-output outside-block layers.

    A 248k-vocab lm_head tuned at seqlen 2048 OOMs a 24GB GPU if the MSE is
    computed over the full logits; chunking rows keeps transients bounded and
    must not change the tune (gradients accumulate to the same values)."""

    def test_rows_per_chunk_clamps_to_output_budget(self, monkeypatch):
        import auto_round.algorithms.quantization.sign_round.quantizer as qmod

        # 64M-element budget: big vocab -> few rows per chunk, small dims -> everything
        assert qmod._outside_tune_rows_per_chunk(248320, 2048, 2**26) == 270
        assert qmod._outside_tune_rows_per_chunk(64, 32, 2**26) == 32
        assert qmod._outside_tune_rows_per_chunk(7, 100, 2**5) == 4  # at least one row

    def test_chunked_tune_matches_single_shot(self, monkeypatch):
        """Same layer, same seed: many tiny chunks must reproduce the single-shot
        tuned parameter bit-for-bit (gradient accumulation is exact)."""

        import auto_round.algorithms.quantization.sign_round.quantizer as qmod

        torch.manual_seed(3)
        layer = torch.nn.Linear(16, 5)
        fp = [torch.randn(1, 13, 16) for _ in range(2)]

        results = []
        for cap in (2**26, 2**4):  # single shot vs 3 rows per chunk
            quant, fresh = _outside_tune_harness(qmod, layer, cap)
            monkeypatch.setattr(qmod, "_OUTSIDE_TUNE_CHUNK_OUT_ELEMS", cap, raising=False)
            qmod.SignRoundQuantizer.quantize_layer_outside_block(
                quant, fresh, fp_inputs=[t.clone() for t in fp], input_ids=None
            )
            results.append(fresh.weight.detach().clone())
        assert torch.equal(results[0], results[1]), "chunked tune diverged from single-shot"

    def test_huge_value_parameter_skips_torch_compile(self, monkeypatch):
        """Inductor's backward adds a full-size buffer for billion-element
        rounding parameters; the wrapper must stay eager for those."""

        import auto_round.algorithms.quantization.sign_round.quantizer as qmod

        captured = {}
        real_wrapper = qmod.WrapperLinear

        def recording_wrapper(layer, **kwargs):
            captured.update(kwargs)
            return real_wrapper(layer, **kwargs)

        monkeypatch.setattr(qmod, "WrapperLinear", recording_wrapper)
        torch.manual_seed(5)
        layer = torch.nn.Linear(16, 5)
        layer.global_name = "lm_head"
        layer.bits = 8
        layer.group_size = 32
        layer.sym = True
        layer.data_type = "int"
        layer.scale_dtype = None
        layer.iters = 2
        layer.act_bits = 16
        layer.act_sym = True
        layer.act_data_type = None
        layer.act_group_size = None
        cfg = type(
            "C",
            (),
            {
                "iters": 2,
                "lr": 5e-3,
                "compute_lr": lambda self, bits: None,
                "compute_minmax_lr": lambda self, bits: None,
            },
        )()
        quant = SimpleNamespace(
            config=cfg,
            _config=cfg,
            iters=2,
            lr=5e-3,
            minmax_lr=1e-3,
            lr_scheduler=None,
            enable_minmax_tuning=False,
            gradient_accumulate_steps=1,
            not_use_best_mse=False,
            dynamic_max_gap=0,
            optimizer=qmod.SignSGD,
            lr_is_auto=False,
            model=torch.nn.Module(),
            calibration_context=SimpleNamespace(batch_size=1),
            model_context=SimpleNamespace(amp=False, amp_dtype=torch.bfloat16),
            compress_context=SimpleNamespace(enable_torch_compile=True, cache_device="cpu"),
        )
        for name in ("_get_scaler", "_scale_loss_and_backward", "_step", "_maybe_log_low_bit_lr", "build_weight_qdq"):
            setattr(quant, name, MethodType(getattr(qmod.SignRoundQuantizer, name), quant))
        quant._preallocate_tuning_grads_ = qmod.SignRoundQuantizer._preallocate_tuning_grads_
        quant._logged_low_bit_lr = set()
        fp = [torch.randn(1, 7, 16) for _ in range(2)]

        # 16x5=80 elements fit the default budget: compile honored
        qmod.SignRoundQuantizer.quantize_layer_outside_block(quant, layer, fp_inputs=[t.clone() for t in fp])
        assert captured["enable_torch_compile"] is True

        # shrink the budget below the layer size: compile must be skipped
        monkeypatch.setattr(qmod, "_OUTSIDE_TUNE_CHUNK_OUT_ELEMS", 16, raising=False)
        fresh = torch.nn.Linear(16, 5)
        for attr, val in (
            ("global_name", "lm_head"),
            ("bits", 8),
            ("group_size", 32),
            ("sym", True),
            ("data_type", "int"),
            ("scale_dtype", None),
            ("iters", 2),
            ("act_bits", 16),
            ("act_sym", True),
            ("act_data_type", None),
            ("act_group_size", None),
        ):
            setattr(fresh, attr, val)
        qmod.SignRoundQuantizer.quantize_layer_outside_block(quant, fresh, fp_inputs=[t.clone() for t in fp])
        assert captured["enable_torch_compile"] is False


def _mk_quant_linear(out_f, in_f, bits=4, group_size=4):
    layer = torch.nn.Linear(in_f, out_f)
    layer.global_name = "lm_head"
    layer.bits = bits
    layer.group_size = group_size
    layer.sym = True
    layer.data_type = "int"
    layer.scale_dtype = None
    layer.iters = 2
    layer.act_bits = 16
    layer.act_sym = True
    layer.act_data_type = None
    layer.act_group_size = None
    return layer


class TestRowBlockedWrapperForward:
    """Huge-weight wrappers quantize one row block at a time; outputs and
    gradients must match the full-tensor path exactly (groups never straddle
    output rows)."""

    def _wrapper(self, layer, minmax=False):
        from auto_round.wrapper import WrapperLinear

        return WrapperLinear(layer, enable_minmax_tuning=minmax, enable_torch_compile=False, device="cpu")

    def test_blocked_output_and_gradients_match(self, monkeypatch):
        import auto_round.wrapper as wmod

        torch.manual_seed(9)
        layer = _mk_quant_linear(24, 10, group_size=4)  # 240 elements
        x = torch.randn(3, 2, 10)
        target = torch.randn(3, 2, 24)

        results = []
        for cap in (2**26, 60):  # full path vs 4-row blocks (4 rows x 10 in)
            torch.manual_seed(21)
            fresh = _mk_quant_linear(24, 10, group_size=4)
            with torch.no_grad():
                fresh.weight.copy_(layer.weight)
                fresh.bias.copy_(layer.bias)
            wrapper = self._wrapper(fresh, minmax=True)
            monkeypatch.setattr(wmod, "_ROW_BLOCKED_WEIGHT_ELEMS", cap, raising=False)
            out = wrapper(x)
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()
            results.append(
                (out.detach().clone(), wrapper.value.grad.detach().clone(), wrapper.min_scale.grad.detach().clone())
            )
        assert torch.allclose(results[0][0], results[1][0], atol=1e-6), "blocked forward output diverged"
        assert torch.allclose(results[0][1], results[1][1], atol=1e-7), "blocked value grad diverged"
        assert torch.allclose(results[0][2], results[1][2], atol=1e-7), "blocked min_scale grad diverged"

    def test_blocked_with_per_group_init_scale_matches(self, monkeypatch):
        """Per-group init_scale (OptRTN/AWQ anchors) must follow the same row
        window as the other tuning parameters."""
        import auto_round.wrapper as wmod

        torch.manual_seed(4)
        layer = _mk_quant_linear(24, 10, group_size=4)  # 6 groups per row -> 144 groups
        x = torch.randn(2, 3, 10)
        target = torch.randn(2, 3, 24)
        n_groups = 24 * 3  # ceil handled by layout: in=10, gs=4 -> 3 groups/row

        results = []
        for cap, scale_shape in ((2**26, (n_groups, 1)), (60, (n_groups, 1))):
            torch.manual_seed(23)
            fresh = _mk_quant_linear(24, 10, group_size=4)
            with torch.no_grad():
                fresh.weight.copy_(layer.weight)
                fresh.bias.copy_(layer.bias)
            wrapper = self._wrapper(fresh, minmax=True)
            torch.manual_seed(31)
            wrapper.init_scale = torch.rand(scale_shape) * 0.01 + 0.005
            monkeypatch.setattr(wmod, "_ROW_BLOCKED_WEIGHT_ELEMS", cap, raising=False)
            out = wrapper(x)
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()
            results.append(out.detach().clone())
        assert torch.allclose(results[0], results[1], atol=1e-6), "blocked init_scale forward diverged"

    def test_per_tensor_group_size_never_blocks(self, monkeypatch):
        import auto_round.wrapper as wmod

        layer = _mk_quant_linear(24, 10, group_size=0)
        wrapper = self._wrapper(layer)
        monkeypatch.setattr(wmod, "_ROW_BLOCKED_WEIGHT_ELEMS", 60, raising=False)
        assert wrapper._use_row_blocked_output() is False

    def test_attached_none_scheme_fields_still_block(self, monkeypatch):
        """The plan machinery attaches every scheme field (None when unset) to
        quantized layers; a None super_bits must not disable blocking."""
        import auto_round.wrapper as wmod

        layer = _mk_quant_linear(24, 10, group_size=4)
        layer.super_bits = None
        layer.super_group_size = None
        layer.rotation_config = None
        layer.act_dynamic = True
        wrapper = self._wrapper(layer)
        monkeypatch.setattr(wmod, "_ROW_BLOCKED_WEIGHT_ELEMS", 60, raising=False)
        assert wrapper._use_row_blocked_output() is True

    def test_real_super_bits_never_blocks(self, monkeypatch):
        import auto_round.wrapper as wmod

        layer = _mk_quant_linear(24, 10, group_size=4)
        layer.super_bits = 6
        layer.super_group_size = 8
        wrapper = self._wrapper(layer)
        monkeypatch.setattr(wmod, "_ROW_BLOCKED_WEIGHT_ELEMS", 60, raising=False)
        assert wrapper._use_row_blocked_output() is False

    def test_unwrapper_matches_full_path_when_blocked(self, monkeypatch):
        """The final quantize/dequantize in unwrapper must equal the full-tensor
        computation when it runs row-blocked (multi-block budget)."""
        import auto_round.wrapper as wmod

        torch.manual_seed(11)
        layer = _mk_quant_linear(24, 10, group_size=4)

        outputs = []
        for cap in (2**26, 40):  # full path vs 4-row blocks
            torch.manual_seed(5)
            fresh = _mk_quant_linear(24, 10, group_size=4)
            with torch.no_grad():
                fresh.weight.copy_(layer.weight)
                fresh.bias.copy_(layer.bias)
            wrapper = self._wrapper(fresh, minmax=True)
            with torch.no_grad():
                torch.manual_seed(6)
                for p in wrapper.parameters():
                    p.add_(torch.randn_like(p) * 1e-3)
                # per-group OptRTN/AWQ anchors take part in the final quantize
                wrapper.init_scale = torch.rand(24 * 3, 1) * 0.01 + 0.005
            monkeypatch.setattr(wmod, "_ROW_BLOCKED_WEIGHT_ELEMS", cap, raising=False)
            best = {k: v.detach().clone() for k, v in wrapper.state_dict().items()}
            restored = wrapper.unwrapper(best)
            outputs.append(restored.weight.detach().clone())
        assert torch.equal(outputs[0], outputs[1]), "blocked unwrapper diverged from full quantize"

    def test_row_block_bounds_partition_rows(self, monkeypatch):
        import auto_round.wrapper as wmod

        layer = _mk_quant_linear(24, 10, group_size=4)
        wrapper = self._wrapper(layer)
        monkeypatch.setattr(wmod, "_ROW_BLOCKED_WEIGHT_ELEMS", 40, raising=False)
        bounds = wrapper.row_block_bounds()
        flat = [r for b in bounds for r in b]
        assert flat[0] == 0 and flat[-1] == 24
        assert all(e > s for s, e in bounds)
        assert all(bounds[i][1] == bounds[i + 1][0] for i in range(len(bounds) - 1)), "gaps/overlaps"
        assert len(bounds) > 1, "expected multiple blocks under the tiny budget"

    def test_blocked_with_per_row_imatrix_matches(self, monkeypatch):
        """A 2-D per-output-row imatrix must slice with the weight block.

        The full-tensor imatrix meeting a block-sized weight crashes the scale
        search inside the quant function; the blocked path slices per-row
        layouts and passes per-column layouts through unchanged.
        """
        import auto_round.wrapper as wmod

        torch.manual_seed(6)
        layer = _mk_quant_linear(24, 10, group_size=4)
        layer.imatrix = torch.rand(24, 10) + 0.1  # per-row layout
        x = torch.randn(2, 3, 10)

        results = []
        for cap in (2**26, 60):  # full path vs 4-row blocks
            torch.manual_seed(21)
            fresh = _mk_quant_linear(24, 10, group_size=4)
            fresh.imatrix = layer.imatrix.clone()
            with torch.no_grad():
                fresh.weight.copy_(layer.weight)
            wrapper = self._wrapper(fresh)
            monkeypatch.setattr(wmod, "_ROW_BLOCKED_WEIGHT_ELEMS", cap, raising=False)
            out = wrapper(x)
            results.append(out.detach().clone())
        assert torch.allclose(results[0], results[1], atol=1e-6), "per-row imatrix blocked forward diverged"


class TestTuningGradBuffers:
    """Outside-block tuning must keep gradient buffers stable in memory."""

    def _params(self):
        w = torch.nn.Parameter(torch.randn(6, 5))
        s = torch.nn.Parameter(torch.randn(6, 1))
        return [w, s]

    def test_preallocate_creates_zero_grads(self):
        from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer

        params = self._params()
        opt = torch.optim.SGD(params, lr=1e-3)
        SignRoundQuantizer._preallocate_tuning_grads_(opt, "probe")
        for p in params:
            assert p.grad is not None
            assert p.grad.shape == p.shape
            assert torch.count_nonzero(p.grad) == 0

    def test_preallocate_idempotent_and_keeps_values(self):
        from auto_round.algorithms.quantization.sign_round.quantizer import SignRoundQuantizer

        params = self._params()
        opt = torch.optim.SGD(params, lr=1e-3)
        SignRoundQuantizer._preallocate_tuning_grads_(opt, "probe")
        params[0].grad.add_(1.0)
        SignRoundQuantizer._preallocate_tuning_grads_(opt, "probe")
        assert torch.all(params[0].grad == 1.0), "pre-allocation must not clobber live gradients"

    def test_optimizer_step_zeroes_in_place(self):
        """The shared step helper must zero grads without dropping the buffers."""
        import inspect

        from auto_round.algorithms.quantization.sign_round import quantizer as qmod

        src = inspect.getsource(qmod.SignRoundQuantizer._step)
        assert "zero_grad(set_to_none=False)" in src

    def test_outside_block_loop_releases_cached_blocks(self):

        import auto_round.algorithms.quantization.sign_round.quantizer as _sr_q

        src = inspect.getsource(CompressionOrchestrator._quantize_zero_shot)
        assert "empty_cache()" in src  # device-manager based: works on cuda/xpu/hpu
        # census contract: header-only baseline at wrapper-ready; the tensor-list
        # walk only in the lane's OOM catch, with an explicit device (the default
        # torch.device("cuda") never equals an indexed "cuda:0" and hides tensors)
        q_src = Path(_sr_q.__file__).read_text(encoding="utf-8")
        assert "walk=False" in q_src  # wrapper-ready baseline is header-only
        lane_src = inspect.getsource(CompressionOrchestrator._quantize_layers_outside_blocks)
        assert "outside-block layer {layer_name} OOM (at failure)" in lane_src
        assert "device_manager.device" in lane_src


class TestGradScatterSlice:
    """The row-window view must produce the exact gradients of a plain slice."""

    def test_blocked_gradients_match_plain_slice(self):
        from auto_round.wrapper import _GradScatterSlice

        torch.manual_seed(7)
        param = torch.nn.Parameter(torch.randn(12, 8))
        x = torch.randn(3, 8)
        ref = torch.randn(3, 6)

        plain = torch.nn.Parameter(param.detach().clone())
        plain_out = torch.nn.functional.linear(x, plain[6:12])
        torch.nn.functional.mse_loss(plain_out, ref, reduction="sum").backward()

        param.grad = torch.zeros_like(param)
        scatter_out = torch.nn.functional.linear(x, _GradScatterSlice.apply(param, 6, 12))
        torch.nn.functional.mse_loss(scatter_out, ref, reduction="sum").backward()

        assert param.grad[0:6].abs().sum() == 0, "outside rows must stay zero"
        assert torch.allclose(param.grad[6:12], plain.grad[6:12], atol=1e-6), "scatter must equal plain-slice grad"

    def test_accumulates_across_multiple_views(self):
        from auto_round.wrapper import _GradScatterSlice

        param = torch.nn.Parameter(torch.randn(8, 4))
        param.grad = torch.zeros_like(param)
        y = torch.randn(2, 4)
        for _ in range(3):
            out = torch.nn.functional.linear(y, _GradScatterSlice.apply(param, 2, 5))
            out.sum().backward()
        assert torch.allclose(param.grad[2:5], torch.ones(3, 4) * y.sum(0).unsqueeze(0) * 3, atol=1e-6)
        assert param.grad[0:2].abs().sum() == 0 and param.grad[5:].abs().sum() == 0


class TestSignSGDMemory:
    """The sign update must not allocate a gradient-sized temporary."""

    def _step_ref(self, param, grad_seq, **kw):
        import copy

        from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD

        ref = torch.nn.Parameter(param.detach().clone())
        opt = SignSGD([ref], lr=kw["lr"])
        for g in grad_seq:
            ref.grad = g.clone()
            opt.step()
            opt.zero_grad(set_to_none=False)
        return ref.detach()

    def test_plain_update_matches_reference_and_leaves_grad_zeroed(self):
        from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD

        torch.manual_seed(3)
        base = torch.randn(8, 5)
        param = torch.nn.Parameter(base.clone())
        opt = SignSGD([param], lr=0.1)
        grad = torch.randn(8, 5)
        ref = base - 0.1 * torch.sign(grad)
        param.grad = grad
        opt.step()
        assert torch.allclose(param.detach(), ref, atol=1e-6)
        opt.zero_grad(set_to_none=False)
        assert (
            param.grad is not None and torch.count_nonzero(param.grad) == 0
        ), "in-place sign must leave the grad buffer zeroable in place"

    def test_momentum_buffer_survives_in_place_sign(self):
        from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD

        torch.manual_seed(4)
        base = torch.randn(6, 3)
        param = torch.nn.Parameter(base.clone())
        opt = SignSGD([param], lr=0.05, momentum=0.9)
        g1, g2 = torch.randn(6, 3), torch.randn(6, 3)
        param.grad = g1.clone()
        opt.step()
        opt.zero_grad(set_to_none=False)
        param.grad = g2.clone()
        opt.step()
        # reference: v2 = 0.9*sign-free momentum chain is irrelevant; SignSGD
        # applies sign(v) each step, so track v explicitly
        v1 = g1.clone()
        p1 = base - 0.05 * torch.sign(v1)
        v2 = 0.9 * v1 + g2
        p2 = p1 - 0.05 * torch.sign(v2)
        assert torch.allclose(param.detach(), p2, atol=1e-6), "momentum buffer must stay unsigned"


class TestRowBlockThreshold:
    """Blocking must engage only for genuinely huge weights."""

    def test_default_threshold_spares_ffn_sized_layers(self):
        """A 67M-element FFN projection (the old threshold) stays whole-layer:
        blocking it would split the compiled graph and add per-block overhead
        for no memory benefit (~2GiB of intermediates fits easily)."""
        from auto_round.wrapper import WrapperLinear

        layer = _mk_quant_linear(2048, 32768, group_size=128)  # 2**26 elements
        assert layer.weight.numel() == 2**26
        w = WrapperLinear(layer, enable_minmax_tuning=False, enable_torch_compile=False, device="cpu")
        assert w.row_block_active() is False

    def test_threshold_still_catches_vocabulary_heads(self, monkeypatch):
        import auto_round.wrapper as wmod
        from auto_round.wrapper import WrapperLinear

        layer = _mk_quant_linear(64, 512, group_size=128)
        w = WrapperLinear(layer, enable_minmax_tuning=False, enable_torch_compile=False, device="cpu")
        monkeypatch.setattr(wmod, "_ROW_BLOCKED_WEIGHT_ELEMS", 64, raising=False)
        assert w.row_block_active() is True


class TestBestParamSnapshotDevice:
    """Huge-layer snapshots must park on the host, not beside the live params."""

    def test_device_selection(self):
        import torch

        from auto_round.compressors.utils import BestParamsSlot, select_snapshot_device

        # a CPU test box cannot honor a cuda home, so the ladder's host fallback
        # is the observable contract here; device-level ladder cases live in
        # test_best_params_slot.py
        wrapper = SimpleNamespace(device=torch.device("cpu"), orig_layer=None, params={"value": torch.zeros(4, 4)})
        assert select_snapshot_device(wrapper) == torch.device("cpu")
        assert BestParamsSlot(wrapper).refresh(wrapper)["value"].device.type == "cpu"


class TestUnwrapStreaming:
    """Huge-layer unwrap must tolerate host-resident best parameters."""

    def test_blocked_unwrap_accepts_cpu_best_params(self, monkeypatch):
        import auto_round.wrapper as wmod

        torch.manual_seed(13)
        layer = _mk_quant_linear(24, 10, group_size=4)
        wrapper = self._make(layer)
        with torch.no_grad():
            torch.manual_seed(14)
            for p in wrapper.parameters():
                p.add_(torch.randn_like(p) * 1e-3)
        monkeypatch.setattr(wmod, "_ROW_BLOCKED_WEIGHT_ELEMS", 40, raising=False)
        cpu_best = {k: v.detach().clone() for k, v in wrapper.state_dict().items()}
        restored = wrapper.unwrapper(cpu_best)
        assert restored is layer or restored.weight.shape == (24, 10)

    @staticmethod
    def _make(layer):
        from auto_round.wrapper import WrapperLinear

        return WrapperLinear(layer, enable_minmax_tuning=True, enable_torch_compile=False, device="cpu")


class TestMaskedChunkedTune:
    def test_masked_chunked_tune_matches_single_shot(self, monkeypatch):
        """3-D rows + a -100-derived valid-token mask must chunk on the sequence axis.

        The mask is [rows, 1] after unsqueeze while 3-D inputs carry rows on
        dim 1; slicing the wrong axis broadcasts a [chunk, 1] mask against a
        [1, chunk, out] chunk and crashes - the exact huge-lm_head scenario.
        """

        import auto_round.algorithms.quantization.sign_round.quantizer as qmod

        torch.manual_seed(3)
        layer = torch.nn.Linear(16, 5)
        fp = [torch.randn(1, 13, 16) for _ in range(2)]
        # every sample's last position is -100 (calibration convention) so the
        # valid-token mask is real and non-trivial
        input_ids = [torch.randint(0, 100, (1, 13)) for _ in range(2)]
        for ids in input_ids:
            ids[:, -1] = -100

        results = []
        for cap in (2**26, 2**4):  # single shot vs 3 rows per chunk
            quant, fresh = _outside_tune_harness(qmod, layer, cap)
            monkeypatch.setattr(qmod, "_OUTSIDE_TUNE_CHUNK_OUT_ELEMS", cap, raising=False)
            qmod.SignRoundQuantizer.quantize_layer_outside_block(
                quant, fresh, fp_inputs=[t.clone() for t in fp], input_ids=[t.clone() for t in input_ids]
            )
            results.append(fresh.weight.detach().clone())
        assert torch.equal(results[0], results[1]), "masked chunked tune diverged from single-shot"


class TestBlockwiseInterleavedLoop:
    def test_blockwise_masked_chunked_matches_single_shot(self, monkeypatch):
        """Row-blocked forward + position chunking + mask + bias, all composed.

        This is the actual 248k-vocab lm_head configuration: the wrapper's
        row-block threshold forces forward_rows blocks, the chunk budget
        forces position slices, and -100-derived masks cover the loss. The
        result must stay bit-identical to the single-shot reference.
        """

        import auto_round.algorithms.quantization.sign_round.quantizer as qmod
        import auto_round.wrapper as wmod

        torch.manual_seed(3)
        layer = torch.nn.Linear(16, 5)  # bias present on purpose
        fp = [torch.randn(1, 13, 16) for _ in range(2)]
        input_ids = [torch.randint(0, 100, (1, 13)) for _ in range(2)]
        for ids in input_ids:
            ids[:, -1] = -100

        imx = torch.rand(5, 16) + 0.1  # per-output-row importance layout
        results = []
        for cap, row_block in ((2**26, 2**27), (2**4, 2**3)):  # single shot vs blockwise+chunked
            quant, fresh = _outside_tune_harness(qmod, layer, cap)
            fresh.imatrix = imx.clone()  # row-blocked final quantize must slice it too
            monkeypatch.setattr(qmod, "_OUTSIDE_TUNE_CHUNK_OUT_ELEMS", cap, raising=False)
            monkeypatch.setattr(wmod, "_ROW_BLOCKED_WEIGHT_ELEMS", row_block, raising=False)
            qmod.SignRoundQuantizer.quantize_layer_outside_block(
                quant, fresh, fp_inputs=[t.clone() for t in fp], input_ids=[t.clone() for t in input_ids]
            )
            results.append(fresh.weight.detach().clone())
        assert torch.equal(results[0], results[1]), "blockwise masked chunked tune diverged from single-shot"


class TestMasked2DChunkedTune:
    def test_masked_2d_rows_chunk_on_dim0(self, monkeypatch):
        """2-D inputs concatenate samples on dim 0; masks flatten to [rows, 1]."""

        import auto_round.algorithms.quantization.sign_round.quantizer as qmod

        torch.manual_seed(3)
        layer = torch.nn.Linear(16, 5)
        fp = [torch.randn(13, 16) for _ in range(2)]  # 2-D: rows on dim 0
        input_ids = [torch.randint(0, 100, (1, 13)) for _ in range(2)]
        for ids in input_ids:
            ids[:, -1] = -100

        results = []
        for cap in (2**26, 2**4):
            quant, fresh = _outside_tune_harness(qmod, layer, cap)
            monkeypatch.setattr(qmod, "_OUTSIDE_TUNE_CHUNK_OUT_ELEMS", cap, raising=False)
            qmod.SignRoundQuantizer.quantize_layer_outside_block(
                quant, fresh, fp_inputs=[t.clone() for t in fp], input_ids=[t.clone() for t in input_ids]
            )
            results.append(fresh.weight.detach().clone())
        assert torch.equal(results[0], results[1]), "masked 2-D chunked tune diverged from single-shot"
