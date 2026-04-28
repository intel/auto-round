"""Regression tests for the ``scale_dtype`` plumbing in QuantLinear classes.

These tests pin two contracts:

1. **Backwards compatibility.** Constructing any of the supported QuantLinear
   classes without passing ``scale_dtype`` keeps the historical behaviour:
   ``scales`` buffer dtype is unchanged from before the refactor.

2. **BF16 round-trip.** When ``scale_dtype=torch.bfloat16`` is supplied, the
   ``scales`` buffer is BF16 *and* a BF16 calibrated scale tensor packed
   through ``QuantLinear.pack(...)`` round-trips without being silently
   down-cast to FP16.

The second contract is the one that motivated the refactor: small SSM
``out_proj`` scales (Mamba2, Nemotron-H) collapse to FP16 sub-normals when
hardcoded ``.half()`` casts run during pack, which destroys quantized
quality regardless of any user-visible ``scale_dtype`` configuration.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from auto_round.export.export_to_autoround.qlinear_triton_act import QuantLinear as TritonActQuantLinear
from auto_round_extension.torch.qlinear_torch import QuantLinear as TorchQuantLinear
from auto_round_extension.torch.qlinear_torch_zp import QuantLinear as TorchZpQuantLinear


def _build_linear(in_features: int, out_features: int) -> nn.Linear:
    layer = nn.Linear(in_features, out_features, bias=False)
    with torch.no_grad():
        # Small but non-zero weights so quantization round-trip is meaningful.
        layer.weight.copy_(torch.randn_like(layer.weight) * 1e-2)
    return layer


def _build_scales_zeros(
    in_features: int,
    out_features: int,
    group_size: int,
    dtype: torch.dtype,
):
    n_groups = math.ceil(in_features / group_size)
    # Mix of normal-range and small-but-finite scales — the small ones are
    # the FP16 sub-normal trap that justifies BF16 support.
    scales = torch.full((out_features, n_groups), 1e-3, dtype=dtype)
    scales[:, 0] = 5e-6  # below FP16 sub-normal threshold (~6.1e-5)
    zeros = torch.full((out_features, n_groups), 8, dtype=torch.int32)
    return scales, zeros


@pytest.mark.parametrize(
    "qlinear_cls,default_dtype",
    [
        (TorchQuantLinear, torch.float16),
        (TorchZpQuantLinear, torch.float16),
        (TritonActQuantLinear, torch.bfloat16),
    ],
)
def test_default_scale_dtype_unchanged(qlinear_cls, default_dtype):
    """Without an explicit ``scale_dtype`` kwarg, the scales buffer dtype must
    match the historical default for that class. This guards every existing
    quantized checkpoint against silent dtype churn."""

    layer = qlinear_cls(bits=4, group_size=32, infeatures=64, outfeatures=64, bias=False)
    assert layer.scales.dtype == default_dtype


@pytest.mark.parametrize("qlinear_cls", [TorchQuantLinear, TorchZpQuantLinear, TritonActQuantLinear])
def test_explicit_bf16_scale_dtype(qlinear_cls):
    """When BF16 is requested, the scales buffer is BF16."""

    layer = qlinear_cls(
        bits=4,
        group_size=32,
        infeatures=64,
        outfeatures=64,
        bias=False,
        scale_dtype=torch.bfloat16,
    )
    assert layer.scales.dtype == torch.bfloat16


@pytest.mark.parametrize("qlinear_cls", [TorchQuantLinear, TorchZpQuantLinear])
def test_pack_does_not_downcast_bf16_scales(qlinear_cls):
    """Pack a BF16-scale layer and verify the scales buffer stays BF16 *and*
    preserves the small-magnitude entry that would underflow in FP16."""

    in_features, out_features, group_size = 64, 32, 32
    linear = _build_linear(in_features, out_features)
    scales, zeros = _build_scales_zeros(in_features, out_features, group_size, dtype=torch.bfloat16)

    qlayer = qlinear_cls(
        bits=4,
        group_size=group_size,
        infeatures=in_features,
        outfeatures=out_features,
        bias=False,
        scale_dtype=torch.bfloat16,
    )
    qlayer.device = torch.device("cpu")  # pack_layer sets this externally
    qlayer.pack(linear, scales, zeros)

    assert qlayer.scales.dtype == torch.bfloat16, (
        f"{qlinear_cls.__name__}.pack down-cast BF16 scales to "
        f"{qlayer.scales.dtype} — the bug we are guarding against."
    )
    # The 5e-6 scale would round to 0 in FP16 (sub-normal limit ~6.1e-5).
    # In BF16 it stays representable. Assert it survives the round-trip.
    smallest = qlayer.scales.float().abs().min().item()
    assert smallest > 0.0, "BF16 scale collapsed to zero during pack"
    assert smallest < 1e-4, (
        f"Smallest scale unexpectedly large ({smallest:.3e}); the test "
        "fixture no longer exercises the sub-normal regime."
    )


@pytest.mark.parametrize("qlinear_cls", [TorchQuantLinear, TorchZpQuantLinear])
def test_pack_default_fp16_unchanged(qlinear_cls):
    """The historical FP16 path must keep producing FP16 scales after pack."""

    in_features, out_features, group_size = 64, 32, 32
    linear = _build_linear(in_features, out_features)
    scales, zeros = _build_scales_zeros(in_features, out_features, group_size, dtype=torch.float16)

    qlayer = qlinear_cls(
        bits=4,
        group_size=group_size,
        infeatures=in_features,
        outfeatures=out_features,
        bias=False,
    )
    qlayer.device = torch.device("cpu")  # pack_layer sets this externally
    qlayer.pack(linear, scales, zeros)

    assert qlayer.scales.dtype == torch.float16


@pytest.mark.slow
def test_end_to_end_mixed_scale_dtype(tmp_path):
    """End-to-end: quantize opt-125m with one layer overridden to BF16 scales.

    Asserts both contracts hold after a real ``quantize_and_save`` round-trip:

    * Layers without override → FP16 scales on disk (no regression).
    * The single overridden layer → BF16 scales on disk.

    This is the smallest model in the test suite that exercises the full
    ``compressor → wrapper → pack_layer → QuantLinear → safetensors`` chain.
    """

    pytest.importorskip("safetensors")
    from safetensors import safe_open

    from auto_round import AutoRound

    from ...helpers import opt_name_or_path

    target_layer = "model.decoder.layers.0.self_attn.q_proj"
    layer_config = {target_layer: {"scale_dtype": torch.bfloat16}}

    save_dir = str(tmp_path / "saved_mixed")
    autoround = AutoRound(
        model=opt_name_or_path,
        bits=4,
        group_size=32,
        sym=True,
        iters=1,
        seqlen=2,
        nsamples=1,
        layer_config=layer_config,
    )
    autoround.quantize_and_save(output_dir=save_dir, format="auto_round")

    # Find the safetensors shard(s) and pull every *.scales tensor's dtype.
    import glob

    shards = glob.glob(f"{save_dir}/*.safetensors")
    assert shards, "no safetensors shard found"

    scale_dtypes: dict[str, torch.dtype] = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                if k.endswith(".scales"):
                    scale_dtypes[k] = f.get_tensor(k).dtype

    target_key = f"{target_layer}.scales"
    assert target_key in scale_dtypes, (
        f"override target {target_key} not present in saved scales — keys: " f"{list(scale_dtypes)[:5]}..."
    )
    assert (
        scale_dtypes[target_key] == torch.bfloat16
    ), f"override layer scales saved as {scale_dtypes[target_key]}, expected bfloat16"

    other_dtypes = {k: dt for k, dt in scale_dtypes.items() if k != target_key}
    fp16_count = sum(1 for dt in other_dtypes.values() if dt == torch.float16)
    assert fp16_count == len(other_dtypes), (
        "Default-path layers must remain FP16 (no regression). Got mixed dtypes: " f"{set(other_dtypes.values())}"
    )
