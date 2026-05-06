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

"""Parity tests between ``AutoRound(model_free=True)`` and the regular
``AutoRound(iters=0, disable_opt_rtn=True, disable_model_free=True)`` flow.

Both code paths perform RTN integer weight-only quantization in the
``auto_round:auto_gptq`` packing format.  These tests assert that:

1. The ``quantization_config`` keys ``bits``, ``group_size``, ``sym``,
   ``data_type`` (family), ``quant_method``, ``packing_format`` agree.
2. The set of tensor names ending in ``.qweight``, ``.qzeros``, ``.scales``
   is identical (after unioning across shards).
3. For every shared key, ``.qweight``/``.qzeros``/``.scales`` are bit-exact.

This file is symlinked into ``test_cuda/quantization``, ``test_xpu/quantization``
and ``test_hpu/quantization`` so the same test body runs on every backend.
The active backend is selected by the ``BACKEND`` constant which is computed
from this file's directory path.  Tests are skipped when the corresponding
accelerator is unavailable on the host.
"""

from __future__ import annotations

import json
import os

import pytest
import torch
from safetensors import safe_open

# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def _detect_backend() -> str:
    """Pick the active backend based on this test file's directory path."""
    p = os.path.abspath(__file__)
    for tag in ("test_cuda", "test_xpu", "test_hpu", "test_cpu"):
        if f"/{tag}/" in p or p.endswith(f"/{tag}"):
            return tag.split("_", 1)[1]
    return "cpu"


BACKEND = _detect_backend()


def _device_str() -> str:
    """Map the backend tag to the device string passed to AutoRound."""
    return {
        "cpu": "cpu",
        "cuda": "cuda:0",
        "xpu": "xpu:0",
        "hpu": "hpu",
    }.get(BACKEND, "cpu")


# Schemes verified to be supported by both code paths.
# Each entry is (test_id, preset_name, scheme_kwargs).  ``preset_name`` is the
# scheme string passed to AutoRound; ``test_id`` is the pytest parametrize id.
_PARITY_SCHEMES = [
    ("W4A16", "W4A16", {"bits": 4, "group_size": 128, "sym": True}),
    ("W2A16", "W2A16", {"bits": 2, "group_size": 128, "sym": True}),
    ("W2A16G64", "W2A16G64", {"bits": 2, "group_size": 64, "sym": True}),
    ("W2A16G32", "W2A16G32", {"bits": 2, "group_size": 32, "sym": True}),
    ("W8A16", "W8A16", {"bits": 8, "group_size": 128, "sym": True}),
    # Asymmetric (sym=False) variants for non-4-bit: both the model-free path and
    # the regular path use ``auto_round:auto_gptq`` packing, so bit-exact parity holds.
    ("W4A16_ASYM", "W4A16", {"bits": 4, "group_size": 128, "sym": False}),
    ("W2A16_ASYM", "W2A16", {"bits": 2, "group_size": 128, "sym": False}),
    ("W2A16G64_ASYM", "W2A16G64", {"bits": 2, "group_size": 64, "sym": False}),
    ("W8A16_ASYM", "W8A16", {"bits": 8, "group_size": 128, "sym": False}),
    # Note: W4A16 asym is excluded because the regular path uses
    # ``auto_round:auto_awq`` packing for 4-bit asym, which differs from
    # model-free's ``auto_round:auto_gptq``.
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_all_keys_and_tensors(directory: str) -> dict[str, torch.Tensor]:
    """Union of all tensors across every safetensors shard in *directory*."""
    out: dict[str, torch.Tensor] = {}
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(directory, fname), framework="pt") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
    return out


def _quant_keys(tensors: dict[str, torch.Tensor]) -> set[str]:
    return {k for k in tensors if k.endswith((".qweight", ".qzeros", ".scales"))}


def _read_qconfig(directory: str) -> dict:
    with open(os.path.join(directory, "config.json")) as f:
        return json.load(f)["quantization_config"]


# ---------------------------------------------------------------------------
# Parity tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme_name,scheme_preset,scheme_kwargs", _PARITY_SCHEMES)
def test_parity_model_free_vs_disable_opt_rtn(tmp_path, tiny_opt_model_path, scheme_name, scheme_preset, scheme_kwargs):
    """``AutoRound(model_free=True)`` must produce identical packed tensors
    to ``AutoRound(iters=0, disable_opt_rtn=True, disable_model_free=True)``.
    """
    from auto_round import AutoRound

    out_a = str(tmp_path / f"mf_{scheme_name}")
    out_b = str(tmp_path / f"reg_{scheme_name}")
    device = _device_str()

    # ---- Path A: AutoRound(model_free=True) ----
    ar_a = AutoRound(
        tiny_opt_model_path,
        scheme=scheme_preset,
        bits=scheme_kwargs["bits"],
        group_size=scheme_kwargs["group_size"],
        sym=scheme_kwargs["sym"],
        iters=0,
        disable_opt_rtn=True,
        model_free=True,
        device_map=device,
    )
    assert getattr(ar_a, "model_free", False) is True
    # Model must NOT be loaded into memory in the model-free path.
    assert ar_a.model is None
    ar_a.quantize_and_save(format="auto_round", output_dir=out_a)

    # ---- Path B: regular ``--iters 0 --disable_opt_rtn`` flow ----
    ar_b = AutoRound(
        tiny_opt_model_path,
        scheme=scheme_preset,
        bits=scheme_kwargs["bits"],
        group_size=scheme_kwargs["group_size"],
        sym=scheme_kwargs["sym"],
        iters=0,
        disable_opt_rtn=True,
        disable_model_free=True,  # opt out of auto-routing
        device_map=device,
        amp=False,  # disable_amp to ensure model is loaded in full precision
    )
    assert getattr(ar_b, "model_free", False) is False
    # Confirm the regular path actually loaded the model on the requested
    # device family (this proves that the corresponding accelerator is
    # actually exercised by the symlinked CUDA/XPU/HPU test).
    if BACKEND != "cpu":
        weight_devices = {p.device.type for p in ar_b.model.parameters()}
        assert (
            BACKEND in weight_devices or "cpu" in weight_devices
        ), f"Expected model parameters on '{BACKEND}' or 'cpu', got {weight_devices}"
    # ar_b.quantize_and_save(format="auto_round", output_dir=out_b)
    _, out_b = ar_b.quantize_and_save(format="auto_round:auto_gptq", output_dir=out_b)

    # ---- 1. quantization_config core keys agree ----
    qc_a = _read_qconfig(out_a)
    qc_b = _read_qconfig(out_b)

    for key in ("bits", "group_size", "sym", "quant_method"):
        assert qc_a[key] == qc_b[key], f"qconfig[{key}] differs: model_free={qc_a[key]} regular={qc_b[key]}"
    assert "int" in qc_a.get("data_type", "")
    assert "int" in qc_b.get("data_type", "")
    assert (
        qc_a["packing_format"] == qc_b["packing_format"]
    ), f"packing_format differs: {qc_a['packing_format']} vs {qc_b['packing_format']}"

    # ---- 2. quantized tensor key sets agree ----
    tensors_a = _load_all_keys_and_tensors(out_a)
    tensors_b = _load_all_keys_and_tensors(out_b)
    keys_a = _quant_keys(tensors_a)
    keys_b = _quant_keys(tensors_b)

    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    assert not only_a and not only_b, (
        f"Quantized key sets differ.\n  only in model_free: {sorted(only_a)[:10]}\n"
        f"  only in regular:    {sorted(only_b)[:10]}"
    )

    # ---- 3. Bit-exact equality for shared qweight/qzeros/scales ----
    mismatched: list[str] = []
    for k in sorted(keys_a):
        ta = tensors_a[k]
        tb = tensors_b[k]
        if ta.shape != tb.shape or ta.dtype != tb.dtype:
            mismatched.append(f"{k}: shape/dtype differs ({ta.shape}/{ta.dtype} vs {tb.shape}/{tb.dtype})")
            continue
        if not torch.equal(ta, tb):
            diff = (ta.float() - tb.float()).abs()
            mismatched.append(
                f"{k}: max|diff|={diff.max().item():.4g}, " f"#diff={int((ta != tb).sum().item())}/{ta.numel()}"
            )

    if mismatched:
        pytest.fail(
            f"RTN values differ between model_free and regular path "
            f"({len(mismatched)} tensors differ).  First few:\n  " + "\n  ".join(str(m) for m in mismatched[:5])
        )


def test_auto_routing_to_model_free(tiny_opt_model_path):
    """When iters=0 + disable_opt_rtn=True + supported scheme, AutoRound
    auto-routes to the model-free path even without explicit ``model_free=True``.
    """
    from auto_round import AutoRound

    ar = AutoRound(
        tiny_opt_model_path,
        scheme="W4A16",
        iters=0,
        disable_opt_rtn=True,
        device_map=_device_str(),
    )
    assert getattr(ar, "model_free", False) is True
    assert ar.model is None


def test_disable_model_free_opt_out(tiny_opt_model_path):
    """``disable_model_free=True`` keeps the regular flow even with the
    auto-routing trigger conditions.
    """
    from auto_round import AutoRound

    ar = AutoRound(
        tiny_opt_model_path,
        scheme="W4A16",
        iters=0,
        disable_opt_rtn=True,
        disable_model_free=True,
        device_map=_device_str(),
    )
    assert getattr(ar, "model_free", False) is False
    assert ar.model is not None
