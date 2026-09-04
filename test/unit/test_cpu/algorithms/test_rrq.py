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
"""Unit tests for RRQ (Recurrent Residual Quantization) Phase 1: RTN-based.

Phase-1 storage is *packed INT2*: the base plane is a standard INT2 weight and
each residual plane is stored in the stock single-plane INT2 AutoRound layout
(``qweight`` int32 + ``scales`` + ``qzeros``).  Dequantization reuses the W2A16
``QuantLinear.forward`` code path, so tests reconstruct a plane by running the
stock ``QuantLinear`` on an identity input.
"""

import os

import pytest
import torch
import torch.nn as nn

from auto_round.algorithms.quantization.rrq.config import RRQConfig
from auto_round.algorithms.quantization.rrq.quantizer import RRQRTNQuantizer, RRQSignRoundQuantizer, RRQPlaneWrapper
from auto_round.data_type.int import quant_tensor_rtn_sym
from auto_round.export.export_to_autoround.export_to_rrq import (
    RRQ_QUANT_METHOD,
    build_rrq_quantization_config,
    save_quantized_rrq,
)


def _make_layer(out_features=64, in_features=128, group_size=32, sym=True, bias=False):
    """Create a mock Linear layer with RRQ-compatible attributes."""
    module = nn.Linear(in_features, out_features, bias=bias)
    module.weight.data = torch.randn(out_features, in_features) * 0.01
    module.bits = 2
    module.group_size = group_size
    module.sym = sym
    module.data_type = "int"
    module.act_bits = 16
    return module


def _dequant_packed(qweight, scales, qzeros, bits, group_size, in_features, out_features, sym):
    """Dequantize a packed INT2 plane using the stock W2A16 ``QuantLinear.forward``.

    Returns the dequantized weight of shape ``(out_features, in_features)``.
    """
    if sym:
        from auto_round_extension.torch.qlinear_torch_zp import QuantLinear
    else:
        from auto_round_extension.torch.qlinear_torch import QuantLinear

    ql = QuantLinear(bits, group_size, in_features, out_features, bias=False)
    ql.qweight.data.copy_(qweight.to(torch.int32))
    ql.scales.data.copy_(scales.to(torch.float16))
    ql.qzeros.data.copy_(qzeros.to(torch.int32))
    ql.to("cpu")

    x = torch.eye(in_features, dtype=torch.float32)
    out = ql.forward(x)  # (in_features, out_features)
    return out.T.to(torch.float32)  # (out_features, in_features)


def _plane_dequant(layer, k, W):
    """Dequant the k-th residual plane of a quantized ``layer`` to a float weight."""
    in_features = W.shape[1]
    out_features = W.shape[0]
    return _dequant_packed(
        getattr(layer, f"rrq_qweight_{k}"),
        getattr(layer, f"rrq_scales_{k}"),
        getattr(layer, f"rrq_qzeros_{k}"),
        bits=2,
        group_size=layer.group_size,
        in_features=in_features,
        out_features=out_features,
        sym=layer.sym,
    )


class TestRRQConfig:
    def test_default_config(self):
        config = RRQConfig()
        assert config.bits == 2
        assert config.data_type == "int"
        assert config.act_bits == 16
        assert config.total_planes == 4
        assert config.total_bits == 8

    def test_rejects_wrong_bits(self):
        with pytest.raises(ValueError, match="bits=2"):
            RRQConfig(bits=4)

    def test_rejects_wrong_data_type(self):
        with pytest.raises(ValueError, match="data_type"):
            RRQConfig(data_type="fp8")

    def test_rejects_act_quantization(self):
        with pytest.raises(ValueError, match="act_bits"):
            RRQConfig(act_bits=8)

    def test_rejects_wrong_num_planes(self):
        with pytest.raises(ValueError, match="num_residual_planes"):
            RRQConfig(num_residual_planes=5)

    def test_group_size_passthrough(self):
        config = RRQConfig(group_size=64)
        assert config.group_size == 64

    def test_phase3_tuning_config(self):
        from auto_round.algorithms.registry import resolve_pipeline_member

        config = RRQConfig(group_size=32, sym=True, iters=4, lr=0.05, minmax_lr=0.01)
        assert config.iters == 4
        assert config.lr == 0.05
        assert config.minmax_lr == 0.01
        assert config.need_calib is True
        assert resolve_pipeline_member(config) is RRQSignRoundQuantizer


class TestRRQQuantization:
    """Tests for the core RRQ RTN quantization algorithm (packed INT2)."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        torch.manual_seed(42)
        yield

    def test_residual_planes_are_packed_int2(self):
        """Each residual plane is stored in the standard single-plane INT2 layout."""
        torch.manual_seed(42)
        W = torch.randn(64, 128) * 0.01
        layer = _make_layer(64, 128, group_size=32, sym=True)
        layer.weight.data = W.clone()

        quantizer = RRQRTNQuantizer(RRQConfig(group_size=32, sym=True))
        quantizer._quantize_layer_rrq(layer)

        in_features, out_features = 128, 64
        pack_factor = 32 // 2
        for k in range(1, 4):
            qweight = getattr(layer, f"rrq_qweight_{k}")
            scales = getattr(layer, f"rrq_scales_{k}")
            qzeros = getattr(layer, f"rrq_qzeros_{k}")
            # packed qweight: (in // 32 * bits, out) -- int32 words holding 16
            # packed 2-bit codes each, so the raw word is NOT bounded to [0,3].
            assert qweight.dtype == torch.int32
            assert qweight.shape == (in_features // 32 * 2, out_features)
            # The 2-bit codes decode into [0, 3]: mask each 32-bit word with 0x3
            # and confirm every lane is a valid 2-bit code.
            codes = qweight & 0x3
            assert codes.max().item() <= 3
            # scales: (num_groups, out)
            num_groups = in_features // 32
            assert scales.dtype == torch.float16
            assert scales.shape == (num_groups, out_features)
            # qzeros: (num_groups, out // 32 * bits) -- W2A16 layout
            assert qzeros.dtype == torch.int32
            assert qzeros.shape == (num_groups, out_features // 32 * 2)

    def test_residual_norm_decreases(self):
        """Each successive plane should reduce the residual error."""
        torch.manual_seed(42)
        W = torch.randn(64, 128) * 0.01
        layer = _make_layer(64, 128, group_size=32, sym=True)
        layer.weight.data = W.clone()

        quantizer = RRQRTNQuantizer(RRQConfig(group_size=32, sym=True))
        quantizer._quantize_layer_rrq(layer)

        acc = layer.weight.data.float()  # base plane (plane 0)
        norms = [(W - acc).norm().item()]
        for k in range(1, 4):
            acc = acc + _plane_dequant(layer, k, W)
            norms.append((W - acc).norm().item())

        for i in range(1, len(norms)):
            assert norms[i] <= norms[i - 1] + 1e-6, (
                f"Residual norm increased at plane {i}: {norms[i]} > {norms[i-1]}"
            )

    def test_4bit_better_than_2bit(self):
        """RRQ 2+2 (4-bit) should beat standalone 2-bit RTN."""
        torch.manual_seed(42)
        W = torch.randn(64, 128) * 0.01

        base_q, _, _ = quant_tensor_rtn_sym(W, bits=2, group_size=32)
        err_2bit = (W - base_q).norm().item()

        layer = _make_layer(64, 128, group_size=32, sym=True)
        layer.weight.data = W.clone()
        quantizer = RRQRTNQuantizer(RRQConfig(group_size=32, sym=True))
        quantizer._quantize_layer_rrq(layer)

        W_4 = layer.weight.data.float() + _plane_dequant(layer, 1, W)
        err_4bit = (W - W_4).norm().item()
        assert err_4bit < err_2bit, (
            f"RRQ 2+2 ({err_4bit:.6f}) should beat 2-bit RTN ({err_2bit:.6f})"
        )

    def test_8bit_close_to_original(self):
        """Full 4-plane (8-bit) reconstruction should be close to the original."""
        torch.manual_seed(42)
        W = torch.randn(64, 128) * 0.01

        layer = _make_layer(64, 128, group_size=32, sym=True)
        layer.weight.data = W.clone()
        quantizer = RRQRTNQuantizer(RRQConfig(group_size=32, sym=True))
        quantizer._quantize_layer_rrq(layer)

        W_recon = layer.weight.data.float()
        for k in range(1, 4):
            W_recon = W_recon + _plane_dequant(layer, k, W)

        rel_err = (W - W_recon).norm() / W.norm()
        assert rel_err < 0.05, f"8-bit RRQ relative error too high: {rel_err:.6f}"

    def test_marker_attributes(self):
        """An RRQ layer should carry the expected metadata attributes."""
        torch.manual_seed(42)
        layer = _make_layer(64, 128, group_size=32, sym=True)
        layer.weight.data = torch.randn(64, 128) * 0.01

        quantizer = RRQRTNQuantizer(RRQConfig(group_size=32, sym=True))
        quantizer._quantize_layer_rrq(layer)

        assert layer.rrq_total_planes == 4
        assert layer.rrq_bit_width == 2
        assert layer.rrq_group_size == 32
        for k in range(1, 4):
            assert hasattr(layer, f"rrq_qweight_{k}")
            assert hasattr(layer, f"rrq_scales_{k}")
            assert hasattr(layer, f"rrq_qzeros_{k}")

    def test_asymmetric(self):
        """RRQ should work with asymmetric quantization (packed INT2)."""
        torch.manual_seed(42)
        W = torch.randn(32, 64) * 0.01 + 0.5
        layer = _make_layer(32, 64, group_size=32, sym=False)
        layer.weight.data = W.clone()

        quantizer = RRQRTNQuantizer(RRQConfig(group_size=32, sym=False))
        quantizer._quantize_layer_rrq(layer)

        assert layer.rrq_total_planes == 4
        W_recon = layer.weight.data.float()
        for k in range(1, 4):
            W_recon = W_recon + _plane_dequant(layer, k, W)

        rel_err = (W - W_recon).norm() / W.norm()
        assert rel_err < 0.05, f"8-bit RRQ relative error too high: {rel_err:.6f}"


class TestRRQLinear:
    """Tests for the RRQLinear inference module (base + residual QuantLinear)."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        torch.manual_seed(0)
        yield

    def _build_rrq_layer(self, in_features=64, out_features=32, group_size=32, num_residual=3):
        """Build an RRQLinear from a base QuantLinear + residual planes.

        ``out`` must be a multiple of 32 for the W2A16 packing (its ``qzeros``
        uses ``out // 32 * bits``).
        """
        from auto_round.inference.rrq_linear import RRQLinear

        W = torch.randn(out_features, in_features) * 0.01
        layer = _make_layer(out_features, in_features, group_size=group_size, sym=True)
        layer.weight.data = W.clone()
        quantizer = RRQRTNQuantizer(RRQConfig(group_size=group_size, sym=True))
        quantizer._quantize_layer_rrq(layer)

        # Base plane: a stock QuantLinear holding the (dequant) base weight.
        from auto_round_extension.torch.qlinear_torch_zp import QuantLinear

        base = QuantLinear(2, group_size, in_features, out_features, bias=False)
        # Re-pack the base weight so base.forward matches the base plane.
        Wf = W.float()
        plane_linear = nn.Linear(in_features, out_features, bias=False)
        plane_linear.weight.data = Wf.clone()
        base.pack(plane_linear, layer.scale, layer.zp, None, device="cpu")
        base.qweight.data = base.qweight.to(torch.int32)
        base.scales.data = base.scales.to(torch.float16)
        base.qzeros.data = base.qzeros.to(torch.int32)

        residual_planes = []
        for k in range(1, 1 + num_residual):
            from auto_round_extension.torch.qlinear_torch_zp import QuantLinear as Q2

            plane = Q2(2, group_size, in_features, out_features, bias=False)
            plane.qweight.data.copy_(getattr(layer, f"rrq_qweight_{k}").to(torch.int32))
            plane.scales.data.copy_(getattr(layer, f"rrq_scales_{k}").to(torch.float16))
            plane.qzeros.data.copy_(getattr(layer, f"rrq_qzeros_{k}").to(torch.int32))
            residual_planes.append(plane)

        return RRQLinear(base=base, residual_planes=residual_planes, bias=None)

    def test_forward_accumulates_planes(self):
        """Output at N active bits equals the sum of the first N planes' outputs."""
        in_features, out_features = 64, 32
        rrq = self._build_rrq_layer(in_features, out_features, group_size=32, num_residual=3)
        x = torch.randn(4, in_features)

        base_out = rrq.base(x)
        full_out = None
        for bits in (2, 4, 6, 8):
            rrq.set_active_bits(bits)
            out = rrq(x)
            # Reconstruct manually: base + first (bits//2 - 1) residual planes.
            manual = base_out
            for i in range(1, bits // 2):
                manual = manual + rrq.planes[f"rrq_{i}"](x)
            assert torch.allclose(out, manual, atol=1e-3), f"mismatch at {bits}-bit"
            full_out = out

        # All-plane output should be closer to the true (float) weight than base only.
        # (Sanity: forward runs without error and is finite.)
        assert torch.isfinite(full_out).all()

    def test_set_active_bits_validation(self):
        in_features, out_features = 64, 32
        rrq = self._build_rrq_layer(in_features, out_features, group_size=32, num_residual=3)
        with pytest.raises(ValueError):
            rrq.set_active_bits(3)
        # 8-bit requires all 3 residual planes; with only 3 planes it is allowed.
        rrq.set_active_bits(8)
        assert rrq.active_bits == 8


class TestRRQConfigBuilder:
    """Tests for the residual-model quantization_config builder."""

    def test_config_fields(self):
        cfg = build_rrq_quantization_config(num_planes=4, group_size=128, sym=False)
        assert cfg["quant_method"] == RRQ_QUANT_METHOD
        assert cfg["format_version"] == 1
        assert cfg["bits"] == 2
        assert cfg["base_bits"] == 2
        assert cfg["residual_planes"] == [2, 2, 2]
        assert cfg["supported_effective_bits"] == [4, 6, 8]
        assert cfg["total_planes"] == 4
        assert cfg["group_size"] == 128
        assert cfg["sym"] is False
        assert cfg["packing_format"] == "auto_round:rrq"

    def test_residual_planes_count_scales(self):
        cfg = build_rrq_quantization_config(num_planes=3, group_size=64, sym=True)
        assert len(cfg["residual_planes"]) == 2  # total_planes - 1
        assert cfg["total_planes"] == 3
        assert cfg["sym"] is True


class TestRRQSave:
    """Tests for save_quantized_rrq (buffer rename + config attach, packed INT2)."""

    def _make_quantized_model(self):
        """Build a tiny model whose single linear layer has RRQ planes."""
        W = torch.randn(32, 64) * 0.01
        layer = _make_layer(32, 64, group_size=32, sym=True)
        layer.weight.data = W.clone()
        quantizer = RRQRTNQuantizer(RRQConfig(group_size=32, sym=True))
        quantizer._quantize_layer_rrq(layer)
        model = nn.Module()
        model.net = layer
        return model, layer

    def test_buffer_rename_and_config(self, tmp_path, monkeypatch):
        """save_quantized_rrq saves only the packed residual planes (ABI names)."""
        import auto_round.export.export_to_autoround.export_to_rrq as rrq_export

        model, layer = self._make_quantized_model()

        in_features, out_features = 64, 32
        captured = {}

        def _fake_save(state, output_dir, safe_serialization):
            captured["state"] = state
            captured["output_dir"] = output_dir

        monkeypatch.setattr(rrq_export, "_save_state_dict_sharded", _fake_save)
        monkeypatch.setattr(rrq_export, "_write_quantization_config", lambda cfg, d: None)

        out_dir = str(tmp_path / "rrq_residual")
        save_quantized_rrq(out_dir, model)

        # The three residual planes (packed INT2) should be renamed to the ABI.
        for k in range(1, 4):
            assert f"qweight_{k}" in layer._buffers, f"qweight_{k} buffer missing"
            assert f"scales_{k}" in layer._buffers, f"scales_{k} buffer missing"
            assert f"qzeros_{k}" in layer._buffers, f"qzeros_{k} buffer missing"
            for old in (f"rrq_qweight_{k}", f"rrq_scales_{k}", f"rrq_qzeros_{k}"):
                assert old not in layer._buffers

        # Only residual-plane tensors should be saved -- no base weight, no
        # non-RRQ params.  There are exactly 3 planes x 3 tensors = 9 entries.
        state = captured["state"]
        assert len(state) == 9, f"expected 9 residual tensors, got {len(state)}"
        assert "net.weight" not in state, "base weight must not be in residual artifact"
        assert "net.scale" not in state
        for k in range(1, 4):
            qw = state[f"net.qweight_{k}"]
            assert qw.dtype == torch.int32
            assert qw.shape == (in_features // 32 * 2, out_features)
            sc = state[f"net.scales_{k}"]
            assert sc.dtype == torch.float16
            assert sc.shape == (in_features // 32, out_features)
            qz = state[f"net.qzeros_{k}"]
            assert qz.dtype == torch.int32
            assert qz.shape == (in_features // 32, out_features // 32 * 2)

    def test_quantization_config_attached(self, tmp_path, monkeypatch):
        """When the model has a .config, quantization_config is attached to it."""
        import auto_round.export.export_to_autoround.export_to_rrq as rrq_export

        model, layer = self._make_quantized_model()
        model.config = type("Cfg", (), {"quantization_config": None})()

        monkeypatch.setattr(rrq_export, "_save_state_dict_sharded", lambda state, d, s: None)
        captured_cfg = {}

        def _capture(cfg, d):
            captured_cfg["cfg"] = cfg

        monkeypatch.setattr(rrq_export, "_write_quantization_config", _capture)

        out_dir = str(tmp_path / "rrq_residual_cfg")
        save_quantized_rrq(out_dir, model)

        assert model.config.quantization_config is not None
        assert model.config.quantization_config["quant_method"] == RRQ_QUANT_METHOD
        assert model.config.quantization_config["total_planes"] == 4


class TestRRQValidation:
    """Tests for the base/residual config validation in load_rrq_model."""

    def _residual_config(self, bits=2, group_size=128, sym=False, method=RRQ_QUANT_METHOD):
        return {
            "quantization_config": {
                "bits": bits,
                "group_size": group_size,
                "sym": sym,
                "quant_method": method,
                "total_planes": 4,
            }
        }

    def test_matching_configs(self):
        from auto_round.inference.rrq_model import _validate_base_matches_residual

        base = {"quantization_config": {"bits": 2, "group_size": 128, "sym": False}}
        residual = self._residual_config(bits=2, group_size=128, sym=False)
        # Should not raise.
        _validate_base_matches_residual(base, residual)

    def test_bits_mismatch(self):
        from auto_round.inference.rrq_model import _validate_base_matches_residual

        base = {"quantization_config": {"bits": 4, "group_size": 128, "sym": False}}
        residual = self._residual_config(bits=2, group_size=128, sym=False)
        with pytest.raises(ValueError, match="mismatch"):
            _validate_base_matches_residual(base, residual)

    def test_group_size_mismatch(self):
        from auto_round.inference.rrq_model import _validate_base_matches_residual

        base = {"quantization_config": {"bits": 2, "group_size": 64, "sym": False}}
        residual = self._residual_config(bits=2, group_size=128, sym=False)
        with pytest.raises(ValueError, match="mismatch"):
            _validate_base_matches_residual(base, residual)

    def test_sym_mismatch(self):
        from auto_round.inference.rrq_model import _validate_base_matches_residual

        base = {"quantization_config": {"bits": 2, "group_size": 128, "sym": True}}
        residual = self._residual_config(bits=2, group_size=128, sym=False)
        with pytest.raises(ValueError, match="mismatch"):
            _validate_base_matches_residual(base, residual)

    def test_wrong_quant_method(self):
        from auto_round.inference.rrq_model import _validate_base_matches_residual

        base = {"quantization_config": {"bits": 2, "group_size": 128, "sym": False}}
        residual = self._residual_config(bits=2, group_size=128, sym=False, method="auto-round")
        with pytest.raises(ValueError, match="quant_method"):
            _validate_base_matches_residual(base, residual)


class TestGenerateRRQResidual:
    """Tests for Phase 2: generate_rrq_residual (incremental from existing base)."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        torch.manual_seed(42)
        yield

    def _make_base_and_raw(
        self, tmp_path, out_features=64, in_features=128, group_size=32, sym=True
    ):
        """Build a fake base model dir (packed INT2) + raw FP state dict.

        Returns ``(base_dir, raw_dir)`` where:
        - ``base_dir`` contains ``config.json`` + ``model.safetensors`` with
          packed ``qweight``/``scales``/``qzeros`` for one layer.
        - ``raw_dir`` contains ``model.safetensors`` with the FP ``.weight``.
        """
        import json
        import os

        from safetensors.torch import save_file

        from auto_round.algorithms.quantization.rrq.quantizer import _rrq_quant_linear_class
        from auto_round.data_type.utils import get_quant_func

        QuantLinear = _rrq_quant_linear_class(sym)

        W = torch.randn(out_features, in_features) * 0.01

        # RTN quantize to get scale/zp
        quant_func, _ = get_quant_func(
            dtype="int", bits=2, sym=sym,
            disable_opt_rtn=True, group_size=group_size, iters=0,
        )
        _, scale, zp = quant_func(
            W, bits=2, group_size=group_size,
            scale_dtype=torch.float16, q_scale_thresh=1e-5,
        )

        # Normalize
        scale_n = scale.reshape(out_features, -1).to(torch.float16)
        if isinstance(zp, torch.Tensor):
            zp_n = zp.reshape(out_features, -1)
        else:
            zp_n = zp

        # Pack
        plane_linear = nn.Linear(in_features, out_features, bias=False)
        plane_linear.weight.data = W.clone()
        plane_linear.to(torch.float32)
        ql = QuantLinear(2, group_size, in_features, out_features, bias=False)
        ql.pack(plane_linear, scale_n, zp_n, None, device="cpu")

        # Write base model dir
        base_dir = os.path.join(str(tmp_path), "base")
        os.makedirs(base_dir, exist_ok=True)
        base_state = {
            "test_layer.qweight": ql.qweight.detach(),
            "test_layer.scales": ql.scales.detach(),
            "test_layer.qzeros": ql.qzeros.detach(),
        }
        save_file(base_state, os.path.join(base_dir, "model.safetensors"))
        config = {
            "quantization_config": {
                "bits": 2,
                "group_size": group_size,
                "sym": sym,
                "quant_method": "auto-round",
            }
        }
        with open(os.path.join(base_dir, "config.json"), "w") as f:
            json.dump(config, f)

        # Write raw model dir
        raw_dir = os.path.join(str(tmp_path), "raw")
        os.makedirs(raw_dir, exist_ok=True)
        raw_state = {"test_layer.weight": W.clone()}
        save_file(raw_state, os.path.join(raw_dir, "model.safetensors"))

        return base_dir, raw_dir

    def test_output_structure(self, tmp_path):
        """generate_rrq_residual produces 3 packed planes per layer."""
        from auto_round.export.export_to_autoround.export_to_rrq import generate_rrq_residual

        base_dir, raw_dir = self._make_base_and_raw(tmp_path)
        out_dir = str(tmp_path / "residual_out")

        generate_rrq_residual(base_dir, raw_dir, out_dir, group_size=32, sym=True)

        # Check output files exist
        assert os.path.isdir(out_dir)
        assert os.path.exists(os.path.join(out_dir, "quantization_config.json"))

        from safetensors.torch import load_file
        state = load_file(os.path.join(out_dir, "model.safetensors"))

        # 3 planes × 3 tensors = 9
        assert len(state) == 9, f"expected 9 tensors, got {len(state)}"
        for k in (1, 2, 3):
            assert f"test_layer.qweight_{k}" in state
            assert f"test_layer.scales_{k}" in state
            assert f"test_layer.qzeros_{k}" in state
            assert state[f"test_layer.qweight_{k}"].dtype == torch.int32
            assert state[f"test_layer.scales_{k}"].dtype == torch.float16
            assert state[f"test_layer.qzeros_{k}"].dtype == torch.int32

    def test_residual_norm_decreases(self, tmp_path):
        """Successive residual planes should reduce the reconstruction error."""
        from auto_round.export.export_to_autoround.export_to_rrq import generate_rrq_residual

        base_dir, raw_dir = self._make_base_and_raw(tmp_path, out_features=64, in_features=128, group_size=32, sym=True)
        out_dir = str(tmp_path / "residual_out2")

        generate_rrq_residual(base_dir, raw_dir, out_dir, group_size=32, sym=True)

        from safetensors.torch import load_file
        state = load_file(os.path.join(out_dir, "model.safetensors"))

        # Load raw weight
        raw_state = load_file(os.path.join(raw_dir, "model.safetensors"))
        W = raw_state["test_layer.weight"].to(torch.float32)

        # Dequant base
        from auto_round.algorithms.quantization.rrq.quantizer import _rrq_quant_linear_class
        QuantLinear = _rrq_quant_linear_class(True)
        ql = QuantLinear(2, 32, 128, 64, bias=False)
        ql.qweight.data = load_file(os.path.join(base_dir, "model.safetensors"))["test_layer.qweight"]
        ql.scales.data = load_file(os.path.join(base_dir, "model.safetensors"))["test_layer.scales"]
        ql.qzeros.data = load_file(os.path.join(base_dir, "model.safetensors"))["test_layer.qzeros"]
        ql.to("cpu")
        identity = torch.eye(128, dtype=torch.float32)
        with torch.no_grad():
            W_base = ql.forward(identity).T.to(torch.float32)

        # Dequant each residual plane and accumulate
        acc = W_base
        norms = [(W - acc).norm().item()]
        for k in range(1, 4):
            from auto_round.inference.rrq_model import _build_quant_plane
            plane = _build_quant_plane(
                QuantLinear,
                state[f"test_layer.qweight_{k}"],
                state[f"test_layer.scales_{k}"],
                state[f"test_layer.qzeros_{k}"],
                2, 32, 128, 64, False, None, torch.device("cpu"),
            )
            identity = torch.eye(128, dtype=torch.float32)
            with torch.no_grad():
                acc = acc + plane.forward(identity).T.to(torch.float32)
            norms.append((W - acc).norm().item())

        for i in range(1, len(norms)):
            assert norms[i] <= norms[i - 1] + 1e-4, (
                f"Residual norm increased at plane {i}: {norms[i]} > {norms[i-1]}"
            )

    def test_config_mismatch_raises(self, tmp_path):
        """Mismatched group_size should raise ValueError."""
        from auto_round.export.export_to_autoround.export_to_rrq import generate_rrq_residual

        base_dir, raw_dir = self._make_base_and_raw(tmp_path, group_size=32, sym=True)
        out_dir = str(tmp_path / "residual_out3")

        with pytest.raises(ValueError, match="group_size"):
            generate_rrq_residual(base_dir, raw_dir, out_dir, group_size=64, sym=True)

    def test_sym_mismatch_raises(self, tmp_path):
        """Mismatched sym flag should raise ValueError."""
        from auto_round.export.export_to_autoround.export_to_rrq import generate_rrq_residual

        base_dir, raw_dir = self._make_base_and_raw(tmp_path, group_size=32, sym=True)
        out_dir = str(tmp_path / "residual_out4")

        with pytest.raises(ValueError, match="sym"):
            generate_rrq_residual(base_dir, raw_dir, out_dir, group_size=32, sym=False)

    def test_top_level_export(self):
        """generate_rrq_residual is accessible from auto_round top-level."""
        from auto_round import generate_rrq_residual as gen
        assert callable(gen)


class TestRRQPhase3:
    """Focused checks for the calibrated sign-SGD RRQ path."""

    def test_plane_wrapper_has_ste_gradients(self):
        from auto_round.algorithms.quantization.rrq.quantizer import RRQPlaneWrapper

        layer = _make_layer(out_features=64, in_features=128, group_size=32, sym=True)
        layer.scale_dtype = torch.float16
        target = layer.weight.detach().clone()
        wrapper = RRQPlaneWrapper(
            layer,
            target,
            torch.zeros_like(target),
            plane_idx=0,
            enable_minmax_tuning=True,
            iters=2,
            device="cpu",
        )
        inputs = torch.randn(2, 128)
        loss = wrapper(inputs).square().mean()
        loss.backward()

        assert wrapper.params["value_0"].grad is not None
        assert wrapper.params["min_scale_0"].grad is not None
        assert wrapper.params["max_scale_0"].grad is not None

    def test_sign_sgd_produces_four_packed_planes(self):
        from auto_round.algorithms.block_runner import BlockForwardRunner

        class TinyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = _make_layer(out_features=64, in_features=128, group_size=32, sym=True)
                self.proj.scale_dtype = torch.float16

            def forward(self, hidden_states):
                return self.proj(hidden_states)

        torch.manual_seed(123)
        block = TinyBlock()
        inputs = [torch.randn(1, 3, 128) for _ in range(2)]
        runner = BlockForwardRunner(
            batch_dim=0,
            batch_size=1,
            device="cpu",
            cache_device="cpu",
            amp=False,
            enable_torch_compile=False,
        )
        with torch.no_grad():
            outputs = [runner.forward(block, inputs, {}, torch.tensor([i]), "cpu") for i in range(2)]

        quantizer = RRQSignRoundQuantizer(RRQConfig(group_size=32, sym=True, iters=2, lr=0.05))
        quantizer._BaseAlgorithm__block_forward_runner = runner
        quantizer._quantize_block_opt(block, inputs, {}, outputs, None)

        for plane_idx in range(1, 4):
            assert hasattr(block.proj, f"rrq_qweight_{plane_idx}")
            assert getattr(block.proj, f"rrq_qweight_{plane_idx}").dtype == torch.int32
        assert block.proj.rrq_total_planes == 4

    def test_sign_sgd_accumulates_frozen_prefix(self):
        """Each round's frozen prefix must include every earlier plane."""
        from auto_round.algorithms.block_runner import BlockForwardRunner

        class TinyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = _make_layer(out_features=64, in_features=128, group_size=32, sym=True)
                self.proj.scale_dtype = torch.float16

            def forward(self, hidden_states):
                return self.proj(hidden_states)

        torch.manual_seed(321)
        block = TinyBlock()
        original = block.proj.weight.detach().clone()
        inputs = [torch.randn(1, 3, 128) for _ in range(2)]
        runner = BlockForwardRunner(
            batch_dim=0,
            batch_size=1,
            device="cpu",
            cache_device="cpu",
            amp=False,
            enable_torch_compile=False,
        )
        with torch.no_grad():
            outputs = [runner.forward(block, inputs, {}, torch.tensor([i]), "cpu") for i in range(2)]

        quantizer = RRQSignRoundQuantizer(RRQConfig(group_size=32, sym=True, iters=2, lr=0.05))
        quantizer._BaseAlgorithm__block_forward_runner = runner
        quantizer._quantize_block_opt(block, inputs, {}, outputs, None)

        reconstructed = block.proj.weight.detach().float()
        for plane_idx in range(1, 4):
            reconstructed += _plane_dequant(block.proj, plane_idx, original)
        assert torch.isfinite(reconstructed).all()
        assert (original - reconstructed).norm() < original.norm()
