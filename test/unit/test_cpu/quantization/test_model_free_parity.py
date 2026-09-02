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

"""Parity tests for ``AutoRound(model_free=True)`` routing and exports.

The integer tests cover two behaviors. Plain RTN
(``disable_opt_rtn=True``) is expected to be bit-exact between the model-free
path and the regular RTN flow.  Integer WOQ in model-free mode always uses
plain RTN regardless of ``disable_opt_rtn``; opt_rtn is disabled for INT WOQ
because it does not improve accuracy.  A dedicated test verifies that passing
``disable_opt_rtn=False`` for INT WOQ still produces the same tensors as
``disable_opt_rtn=True``.  The MXFP tests compare the AutoRound-format
quantization metadata produced for a mixed MXFP4/MXFP8 AutoScheme.
These tests assert that:

1. The ``quantization_config`` keys ``bits``, ``group_size``, ``sym``,
   ``data_type`` (family), ``quant_method``, ``packing_format`` agree.
2. The set of tensor names ending in ``.qweight``, ``.qzeros``, ``.scales``
   is identical (after unioning across shards).
3. For symmetric plain RTN, every shared ``.qweight``/``.qzeros``/``.scales``
    tensor is bit-exact.
4. For asymmetric plain RTN, the serialized export contract still matches.
5. For optimized model-free RTN, the exported quantized tensor key set matches
    plain model-free RTN and at least one quantized tensor differs.

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
_CALIBRATION_TEXT = "AutoRound parity calibration sample. " * 128


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
_PLAIN_RTN_PARITY_SCHEMES = [
    ("W4A16", "W4A16", {"bits": 4, "group_size": 128, "sym": True}),
    # Keep one grouped-2bit variant to cover non-default group_size behavior.
    ("W2A16G32", "W2A16G32", {"bits": 2, "group_size": 32, "sym": True}),
    ("W8A16", "W8A16", {"bits": 8, "group_size": 128, "sym": True}),
]

_PLAIN_RTN_CONTRACT_SCHEMES = [
    # Asymmetric plain RTN keeps export-contract parity across the two paths,
    # but the packed tensors are not bit-exact. 8-bit asym is excluded: it is
    # refused at construction on both routes (see the refusal test below).
    ("W2A16_ASYM", "W2A16", {"bits": 2, "group_size": 128, "sym": False}),
]

# Note: W4A16 asym is excluded because the regular path uses
# ``auto_round:auto_awq`` packing for 4-bit asym, which differs from
# model-free's ``auto_round``.

_OPT_RTN_PARITY_SCHEMES = [
    ("W4A16", "W4A16", {"bits": 4, "group_size": 128, "sym": True}),
    ("W2A16G32", "W2A16G32", {"bits": 2, "group_size": 32, "sym": True}),
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


def _write_local_calibration_dataset(tmp_path) -> str:
    dataset_path = tmp_path / "local_calibration.json"
    with open(dataset_path, "w") as f:
        json.dump([_CALIBRATION_TEXT], f)
    return str(dataset_path)


def _int_export_format(sym: bool) -> str:
    return "auto_round:auto_gptq" if sym else "auto_round"


def _assert_int_export_contract_parity(out_model_free: str, out_regular: str) -> None:
    qc_a = _read_qconfig(out_model_free)
    qc_b = _read_qconfig(out_regular)

    for key in ("bits", "group_size", "sym", "quant_method", "packing_format"):
        assert qc_a[key] == qc_b[key], f"qconfig[{key}] differs: model_free={qc_a[key]} regular={qc_b[key]}"
    assert "int" in qc_a.get("data_type", "")
    assert "int" in qc_b.get("data_type", "")

    tensors_a = _load_all_keys_and_tensors(out_model_free)
    tensors_b = _load_all_keys_and_tensors(out_regular)
    keys_a = _quant_keys(tensors_a)
    keys_b = _quant_keys(tensors_b)

    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    assert not only_a and not only_b, (
        f"Quantized key sets differ.\n  only in model_free: {sorted(only_a)[:10]}\n"
        f"  only in regular:    {sorted(only_b)[:10]}"
    )


def _assert_int_tensor_parity(out_model_free: str, out_regular: str) -> None:
    tensors_a = _load_all_keys_and_tensors(out_model_free)
    tensors_b = _load_all_keys_and_tensors(out_regular)
    keys_a = _quant_keys(tensors_a)

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


def _assert_int_tensor_difference(out_opt_rtn: str, out_plain_rtn: str) -> None:
    tensors_opt = _load_all_keys_and_tensors(out_opt_rtn)
    tensors_plain = _load_all_keys_and_tensors(out_plain_rtn)
    keys_opt = _quant_keys(tensors_opt)
    keys_plain = _quant_keys(tensors_plain)

    assert keys_opt == keys_plain, (
        "Quantized key sets differ between optimized and plain model_free RTN.\n"
        f"  only in opt_rtn:   {sorted(keys_opt - keys_plain)[:10]}\n"
        f"  only in plain_rtn: {sorted(keys_plain - keys_opt)[:10]}"
    )

    differing = []
    for key in sorted(keys_opt):
        tensor_opt = tensors_opt[key]
        tensor_plain = tensors_plain[key]
        if tensor_opt.shape != tensor_plain.shape or tensor_opt.dtype != tensor_plain.dtype:
            differing.append(
                f"{key}: shape/dtype differs ({tensor_opt.shape}/{tensor_opt.dtype} vs {tensor_plain.shape}/{tensor_plain.dtype})"
            )
            continue
        if not torch.equal(tensor_opt, tensor_plain):
            differing.append(key)

    assert (
        differing
    ), "Expected model_free optimized RTN to differ from plain model_free RTN, but all tensors matched exactly."


def _assert_mxfp_auto_scheme_config_parity(out_model_free: str, out_regular: str) -> None:
    model_free_config = _read_qconfig(out_model_free)
    regular_config = _read_qconfig(out_regular)

    scheme_keys = (
        "bits",
        "group_size",
        "sym",
        "data_type",
        "act_bits",
        "act_group_size",
        "act_sym",
        "act_data_type",
        "act_dynamic",
    )
    for key in scheme_keys:
        assert model_free_config.get(key) == regular_config.get(key), (
            f"qconfig[{key}] differs: model_free={model_free_config.get(key)} " f"regular={regular_config.get(key)}"
        )

    assert model_free_config["bits"] == 4
    assert model_free_config["act_bits"] == 4
    assert model_free_config["packing_format"] == regular_config["packing_format"]
    assert model_free_config.get("block_name_to_quantize") == regular_config.get("block_name_to_quantize")
    assert model_free_config.get("extra_config", {}) == regular_config.get("extra_config", {})
    assert any(cfg.get("bits") == 8 for cfg in model_free_config.get("extra_config", {}).values())


# ---------------------------------------------------------------------------
# Parity tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scheme_name,scheme_preset,scheme_kwargs", _PLAIN_RTN_PARITY_SCHEMES)
def test_parity_model_free_vs_plain_rtn_regular_flow(
    tmp_path, tiny_opt_model_path, scheme_name, scheme_preset, scheme_kwargs
):
    """``AutoRound(model_free=True, disable_opt_rtn=True)`` must match the
    regular plain-RTN flow.
    """
    from auto_round import AutoRound

    out_a = str(tmp_path / f"mf_{scheme_name}")
    out_b = str(tmp_path / f"reg_{scheme_name}")
    device = _device_str()
    export_format = _int_export_format(scheme_kwargs["sym"])

    # ---- Path A: AutoRound(model_free=True, disable_opt_rtn=True) ----
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
        enable_torch_compile=False,  # disable torch.compile to ensure model is loaded in full precision
    )
    assert getattr(ar_a, "model_free", False) is True
    # Model must NOT be loaded into memory in the model-free path.
    assert ar_a.model is None
    _, out_a = ar_a.quantize_and_save(format=export_format, output_dir=out_a)

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
        enable_torch_compile=False,  # disable torch.compile to ensure model is loaded in full precision
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
    _, out_b = ar_b.quantize_and_save(format=export_format, output_dir=out_b)

    _assert_int_export_contract_parity(out_a, out_b)
    _assert_int_tensor_parity(out_a, out_b)


@pytest.mark.parametrize("scheme_name,scheme_preset,scheme_kwargs", _PLAIN_RTN_CONTRACT_SCHEMES)
def test_plain_rtn_asymmetric_export_contract_parity(
    tmp_path, tiny_opt_model_path, scheme_name, scheme_preset, scheme_kwargs
):
    """Asymmetric plain RTN keeps export-contract parity across model-free and
    regular flow, even though tensor packing is not bit-exact.
    """
    from auto_round import AutoRound

    out_model_free = str(tmp_path / f"mf_{scheme_name}")
    out_regular = str(tmp_path / f"reg_{scheme_name}")
    device = _device_str()
    export_format = _int_export_format(sym=False)

    model_free = AutoRound(
        tiny_opt_model_path,
        scheme=scheme_preset,
        bits=scheme_kwargs["bits"],
        group_size=scheme_kwargs["group_size"],
        sym=False,
        iters=0,
        disable_opt_rtn=True,
        model_free=True,
        device_map=device,
        enable_torch_compile=False,
    )
    assert getattr(model_free, "model_free", False) is True
    assert model_free.model is None
    _, out_model_free = model_free.quantize_and_save(format=export_format, output_dir=out_model_free)

    regular = AutoRound(
        tiny_opt_model_path,
        scheme=scheme_preset,
        bits=scheme_kwargs["bits"],
        group_size=scheme_kwargs["group_size"],
        sym=False,
        iters=0,
        disable_opt_rtn=True,
        disable_model_free=True,
        device_map=device,
        amp=False,
        enable_torch_compile=False,
    )
    assert getattr(regular, "model_free", False) is False
    _, out_regular = regular.quantize_and_save(format=export_format, output_dir=out_regular)

    _assert_int_export_contract_parity(out_model_free, out_regular)


def test_w8_asymmetric_refused_on_both_routes(tiny_opt_model_path):
    """8-bit asym is refused at construction on both the model-free and the
    regular route: both default to native int8-packed export formats, which
    cannot represent the 8-bit zero point (vLLM serves W8 asym only via
    compressed-tensors; see allow_w8_asym).
    """
    from auto_round import AutoRound

    device = _device_str()
    common = dict(
        scheme="W8A16",
        bits=8,
        group_size=128,
        sym=False,
        iters=0,
        disable_opt_rtn=True,
        device_map=device,
        enable_torch_compile=False,
    )
    with pytest.raises(ValueError, match="8-bit asymmetric"):
        AutoRound(tiny_opt_model_path, model_free=True, **common)
    with pytest.raises(ValueError, match="8-bit asymmetric"):
        AutoRound(tiny_opt_model_path, disable_model_free=True, amp=False, **common)


@pytest.mark.parametrize("scheme_name,scheme_preset,scheme_kwargs", _OPT_RTN_PARITY_SCHEMES)
def test_model_free_int_opt_rtn_same_as_plain_rtn(
    tmp_path, tiny_opt_model_path, scheme_name, scheme_preset, scheme_kwargs
):
    """``AutoRound(model_free=True, disable_opt_rtn=False)`` for integer WOQ must
    produce identical tensors to plain RTN (``disable_opt_rtn=True``), because
    opt_rtn is always disabled for INT WOQ in model-free mode.
    """
    from auto_round import AutoRound

    out_opt_rtn = str(tmp_path / f"mf_opt_rtn_{scheme_name}")
    out_plain_rtn = str(tmp_path / f"mf_plain_rtn_{scheme_name}")
    device = _device_str()
    export_format = _int_export_format(sym=True)

    opt_rtn = AutoRound(
        tiny_opt_model_path,
        scheme=scheme_preset,
        bits=scheme_kwargs["bits"],
        group_size=scheme_kwargs["group_size"],
        sym=True,
        iters=0,
        disable_opt_rtn=False,
        model_free=True,
        device_map=device,
        enable_torch_compile=False,
    )
    assert getattr(opt_rtn, "model_free", False) is True
    assert opt_rtn.model is None
    _, out_opt_rtn = opt_rtn.quantize_and_save(format=export_format, output_dir=out_opt_rtn)

    plain_rtn = AutoRound(
        tiny_opt_model_path,
        scheme=scheme_preset,
        bits=scheme_kwargs["bits"],
        group_size=scheme_kwargs["group_size"],
        sym=True,
        iters=0,
        disable_opt_rtn=True,
        model_free=True,
        device_map=device,
        enable_torch_compile=False,
    )
    assert getattr(plain_rtn, "model_free", False) is True
    assert plain_rtn.model is None
    _, out_plain_rtn = plain_rtn.quantize_and_save(format=export_format, output_dir=out_plain_rtn)

    _assert_int_export_contract_parity(out_opt_rtn, out_plain_rtn)
    _assert_int_tensor_parity(out_opt_rtn, out_plain_rtn)


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


@pytest.mark.parametrize(
    ("test_name", "model_free_disable_opt_rtn", "regular_kwargs"),
    [
        ("plain_rtn", True, {"disable_opt_rtn": True}),
    ],
)
@pytest.mark.timeout(60)
def test_mxfp_auto_scheme_quantization_config_parity(
    tmp_path, tiny_opt_model_path, test_name, model_free_disable_opt_rtn, regular_kwargs
):
    """Mixed MXFP AutoScheme must serialize the same scheme metadata in both model-free and regular RTN flows."""
    pytest.importorskip("compressed_tensors")

    from auto_round import AutoRound, AutoScheme

    out_model_free = str(tmp_path / f"mxfp_model_free_{test_name}")
    out_regular = str(tmp_path / f"mxfp_regular_{test_name}")
    calibration_dataset = _write_local_calibration_dataset(tmp_path)
    scheme_kwargs = {
        "avg_bits": 6.0,
        "options": ("MXFP4", "MXFP8"),
        "nsamples": 1,
        "ignore_scale_zp_bits": True,
    }

    torch.manual_seed(42)
    model_free = AutoRound(
        tiny_opt_model_path,
        scheme=AutoScheme(**scheme_kwargs),
        iters=0,
        disable_opt_rtn=model_free_disable_opt_rtn,
        model_free=True,
        dataset=calibration_dataset,
        nsamples=1,
        seqlen=8,
        batch_size=1,
        device_map=_device_str(),
    )
    _, out_model_free = model_free.quantize_and_save(format="auto_round", output_dir=out_model_free)

    torch.manual_seed(42)
    regular = AutoRound(
        tiny_opt_model_path,
        scheme=AutoScheme(**scheme_kwargs),
        iters=0,
        disable_model_free=True,
        dataset=calibration_dataset,
        nsamples=1,
        seqlen=8,
        batch_size=1,
        device_map=_device_str(),
        **regular_kwargs,
    )
    _, out_regular = regular.quantize_and_save(format="auto_round", output_dir=out_regular)

    _assert_mxfp_auto_scheme_config_parity(out_model_free, out_regular)
