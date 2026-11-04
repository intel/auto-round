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
"""Fuzzing of the GGUF export path against gguf-py's independent block decoders.

Tuning (and RTN) fake-quantizes weights with a GGUF scheme and keeps the scale/zero-point on the
layer; export packs the QDQ weights with those params. Decoding the packed blocks must give back
the QDQ weights the model was evaluated with, otherwise the exported model silently differs.
"""

from test.fuzz_helpers import (
    FLOAT_DTYPES,
    FUZZ_DEVICE,
    Q_SCALE_THRESH,
    TENSOR_PATTERNS,
    assert_where,
    dtype_slack,
    make_tensor,
    scaled_settings,
    wrapper_quant_kwargs,
)

import numpy as np
import pytest
import torch
from hypothesis import assume, given, note
from hypothesis import strategies as st

from auto_round.data_type.utils import get_quant_func
from auto_round.export.export_to_gguf.convert import _quant_data_with_args
from auto_round.export.export_to_gguf.packing import ggml_quant
from auto_round.schemes import GGUF_SCHEME_FACTS

gguf = pytest.importorskip("gguf")

SCHEMES = sorted(s for s in GGUF_SCHEME_FACTS if GGUF_SCHEME_FACTS[s]["bits"] < 16)
LEGACY = ("q4_0", "q4_1", "q5_0", "q5_1", "q8_0")
K_QUANT_BITS = {"q2_k": 2, "q3_k": 3, "q4_k": 4, "q5_k": 5, "q6_k": 6}
# How the layer's quant function is resolved: tuning (iters > 0), optimized RTN and plain RTN (iters == 0)
MODES = ("tuned", "opt_rtn", "rtn")
FP16_SMALLEST_NORMAL = 2.0**-14
FP16_MAX = 65504.0
NOT_TINY = tuple(p for p in TENSOR_PATTERNS if p != "tiny")

_FLOAT_ZP_EXPORT_REASON = (
    "fuzz-found: int_asym_float_zp QDQ computes s * clamp(round(x / s + v) + zp, 0, maxq) - s * zp, so "
    "interior values sit on the s * k grid while q4_1/q5_1 store d * q + m with m = -zp * d; packing "
    "re-rounds every value to round(k + zp), so the exported weights differ from the tuned ones by up to "
    "half a step whenever zp is not an integer (it should be s * (clamp(round(x / s + zp + v), 0, maxq) - zp))"
)


def _qtype(scheme):
    return scheme.split(":")[1]


def _ggml_type(qtype):
    return gguf.GGMLQuantizationType[qtype.upper()]


def _dequantize(packed, qtype):
    return torch.from_numpy(gguf.quants.dequantize(np.asarray(packed), _ggml_type(qtype))).double()


def _layer_attrs(qdq, scale, zp):
    """The scale/zp attributes ``WrapperLinear._apply_qdq`` leaves on the layer for GGUF schemes."""
    rows = qdq.shape[0]
    attrs = {}
    for value, main in ((scale, "scale"), (zp, "zp")):
        if isinstance(value, dict):
            attrs.update({k: v.reshape(rows, -1) if k == main else v for k, v in value.items()})
        elif isinstance(value, torch.Tensor):
            attrs[main] = value.reshape(rows, -1) if value.numel() > 1 else value.view(-1)
        else:
            attrs[main] = value
    return attrs


def _block_min(scheme, attrs):
    """Per-group ``wmin`` (``m`` in q4_1/q5_1, ``dmin * m`` in k-quants); None for symmetric types."""
    if GGUF_SCHEME_FACTS[scheme]["data_type"] == "int_asym_float_zp":
        return attrs["zp"] * attrs["scale"]
    return attrs.get("wmin")


def _fp16_stored_params(scheme, attrs):
    """The params GGUF stores as fp16 in every block: ``d``, plus ``dmin``/``m`` for asymmetric types."""
    if "d_scale" in attrs:
        return [v for v in (attrs["d_scale"], attrs.get("d_wmin")) if v is not None]
    return [v for v in (attrs["scale"], _block_min(scheme, attrs)) if v is not None]


@st.composite
def _exports(draw, scheme, mode):
    facts = GGUF_SCHEME_FACTS[scheme]
    func, key = get_quant_func(
        facts["data_type"],
        facts["bits"],
        facts["sym"],
        disable_opt_rtn=mode == "rtn",
        iters=200 if mode == "tuned" else 0,
    )
    block = 32 if _qtype(scheme) in LEGACY else 256
    shape = (draw(st.integers(1, 4)), block * draw(st.integers(1, 512 // block)))
    dtype = draw(st.sampled_from(FLOAT_DTYPES))
    pattern = draw(st.sampled_from(NOT_TINY))  # tiny weights need fp16-subnormal block scales
    seed = draw(st.integers(0, 2**16))
    magnitude = 10.0 ** draw(st.integers(-1, 2))
    scale_dtype = draw(st.sampled_from((torch.float32, torch.float16)))
    note(
        f"{scheme} {mode} ({key}): shape={shape} dtype={dtype} pattern={pattern} seed={seed} "
        f"magnitude={magnitude:g} scale_dtype={scale_dtype}"
    )
    x = make_tensor(shape, dtype, pattern, seed, magnitude, device=FUZZ_DEVICE)
    config = {k: facts[k] for k in ("bits", "group_size", "super_bits", "super_group_size")}
    kwargs = wrapper_quant_kwargs(key, x, config, scale_dtype, "tune" if mode == "tuned" else "rtn", seed)
    qdq, scale, zp = func(x.clone(), **kwargs)
    return x, qdq, _layer_attrs(qdq, scale, zp)


def _export_params():
    params = []
    for scheme in SCHEMES:
        marks = ()
        if GGUF_SCHEME_FACTS[scheme]["data_type"] == "int_asym_float_zp":
            marks = pytest.mark.xfail(strict=True, reason=_FLOAT_ZP_EXPORT_REASON)
        params += [pytest.param(scheme, mode, marks=marks, id=f"{scheme}-{mode}") for mode in MODES]
    return params


def _export_tolerance(scheme, x, ref, attrs):
    super_block = gguf.GGML_QUANT_SIZES[_ggml_type(_qtype(scheme))][0]
    # GGUF rounds d/dmin to fp16, moving d * sc * q - dmin * m by up to 2^-11 * (|qdq| + 2 |wmin|);
    # bound it per (super-)block with a 2x margin, on top of the QDQ output's own rounding.
    magnitude = ref.reshape(-1, super_block).abs().amax(-1, keepdim=True)
    wmin = _block_min(scheme, attrs)
    if isinstance(wmin, torch.Tensor):
        wmin = wmin.detach().cpu().double().abs().reshape(magnitude.shape[0], -1)
        magnitude = magnitude + 2 * wmin.amax(-1, keepdim=True)
    tol = (2.0**-10 * magnitude).expand(-1, super_block).reshape(ref.shape) + dtype_slack(x.dtype, ref)
    if "d_scale" in attrs:
        # A sub-scale clamped up to q_scale_thresh after double quantization is off the d_scale grid;
        # export stores the nearest multiple instead, moving that group by at most 2^bits * thresh.
        sub_scale = attrs["scale"].detach().cpu().double().reshape(-1)
        clamped = sub_scale <= Q_SCALE_THRESH * (1 + 1e-3)
        clamped = clamped.repeat_interleave(ref.numel() // sub_scale.numel()).reshape(ref.shape)
        tol = tol + clamped * 2 ** GGUF_SCHEME_FACTS[scheme]["bits"] * Q_SCALE_THRESH
    return tol


@pytest.mark.parametrize("scheme, mode", _export_params())
def test_gguf_export_reproduces_qdq_weights(scheme, mode):
    qtype = _qtype(scheme)

    @scaled_settings(0.4)
    @given(export=_exports(scheme, mode))
    def check(export):
        x, qdq, attrs = export
        for stored in _fp16_stored_params(scheme, attrs):
            nonzero = stored.abs()[stored != 0]
            assume(bool(((nonzero >= FP16_SMALLEST_NORMAL) & (nonzero <= FP16_MAX)).all()))

        packed = _quant_data_with_args(
            qdq.clone(),  # the packers work in place on fp32 input
            _ggml_type(qtype),
            attrs.get("scale"),
            attrs.get("zp"),
            d_scale=attrs.get("d_scale"),
            wmin=attrs.get("wmin"),
            d_wmin=attrs.get("d_wmin"),
            device=qdq.device,
        )
        decoded = _dequantize(packed, qtype)
        ref = qdq.detach().cpu().double()
        assert decoded.shape == ref.shape
        tol = _export_tolerance(scheme, x, ref, attrs)
        assert_where(
            (decoded - ref).abs() <= tol, "exported GGUF weights differ from the QDQ weights", qdq=ref, decoded=decoded
        )

    check()


@st.composite
def _raw_weights(draw, block, log10_magnitude, patterns):
    shape = (draw(st.integers(1, 4)), block * draw(st.integers(1, 512 // block)))
    dtype = draw(st.sampled_from(FLOAT_DTYPES))
    pattern = draw(st.sampled_from(patterns))
    seed = draw(st.integers(0, 2**16))
    magnitude = 10.0 ** draw(st.integers(*log10_magnitude))
    note(f"weights: shape={shape} dtype={dtype} pattern={pattern} seed={seed} magnitude={magnitude:g}")
    # export always hands fp32 to the packers
    return make_tensor(shape, dtype, pattern, seed, magnitude, device=FUZZ_DEVICE).float()


def _legacy_step(x, qtype):
    blocks = x.reshape(-1, 32)
    if qtype in ("q4_1", "q5_1"):
        step = (blocks.amax(-1) - blocks.amin(-1)) / {"q4_1": 15, "q5_1": 31}[qtype]
    else:
        step = blocks.abs().amax(-1) / {"q4_0": 8, "q5_0": 16, "q8_0": 127}[qtype]
    return step.unsqueeze(-1).expand(-1, 32).reshape(x.shape)


@pytest.mark.parametrize("qtype", LEGACY)
def test_legacy_ggml_quant_matches_gguf_reference(qtype):
    """Without stored params ``ggml_quant`` must agree with gguf-py's reference quantizer.

    The kernels evaluate ``x * id + 8.5`` in float64 where the reference (and llama.cpp) uses
    float32, so codes may differ by one at rounding boundaries; the bytes are not compared.
    """

    @scaled_settings(0.6)
    @given(x=_raw_weights(32, (-3, 2), TENSOR_PATTERNS))
    def check(x):
        ours = _dequantize(ggml_quant(x.clone(), qtype, device=x.device), qtype)
        ref = _dequantize(gguf.quants.quantize(x.cpu().numpy(), _ggml_type(qtype)), qtype)
        step = _legacy_step(x.cpu().double(), qtype)
        tol = step * (1 + 2.0**-10) + 2.0**-24  # one code, with d rounded to fp16 (possibly subnormal)
        assert_where(
            (ours - ref).abs() <= tol,
            f"{qtype} packing deviates from gguf-py by more than one step",
            ours=ours,
            ref=ref,
            step=step,
        )

    check()


@pytest.mark.parametrize("qtype", sorted(K_QUANT_BITS))
def test_kquant_ggml_quant_without_params_stays_within_two_steps(qtype):
    """Tensors exported without stored params (e.g. re-searched after a weight transform) are
    quantized by ``ggml_quant``'s own search; every value must stay within two nominal steps."""

    @scaled_settings(0.4)
    @given(x=_raw_weights(256, (-1, 2), NOT_TINY))
    def check(x):
        xd = x.cpu().double()
        blocks = xd.reshape(-1, 256)
        span = blocks.amax(-1, keepdim=True).clamp(min=0) - blocks.amin(-1, keepdim=True).clamp(max=0)
        # smaller super-blocks would need fp16-subnormal scales, which GGUF cannot store exactly
        assume(bool(((span == 0) | (span >= 1e-2)).all()))
        decoded = _dequantize(ggml_quant(x.clone(), qtype, device=x.device), qtype)
        step = (span / (2 ** K_QUANT_BITS[qtype] - 1)).expand(-1, 256).reshape(xd.shape)
        assert_where(
            (decoded - xd).abs() <= 2 * step,
            f"{qtype} packing error exceeds two steps",
            x=xd,
            decoded=decoded,
            step=step,
        )

    check()
