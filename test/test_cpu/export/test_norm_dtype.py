"""Regression tests for the ``norm_dtype`` export option.

These tests pin three contracts:

1. **Norm classifier correctness.** ``_is_norm_module`` recognises torch
   built-ins (LayerNorm, RMSNorm, GroupNorm, BatchNorm2d) *and* HF-style custom
   subclasses that live outside ``torch.nn.modules.normalization`` (e.g.
   ``LlamaRMSNorm``). Non-norm lookalikes (``Normalize``) are rejected.

2. **In-memory cast isolation.** ``_cast_norm_modules(model, torch.float32)``
   changes *only* the parameters/buffers of norm modules; ``nn.Linear`` and
   other weight-carrying modules retain their original dtype.

3. **End-to-end dtype on disk.** An opt-125m quantization run with
   ``norm_dtype=torch.float32`` serialises norm weights as ``float32`` in
   safetensors, while non-norm unquantized weights (embeddings, lm_head) and
   quantized ``.scales`` buffers retain their defaults. The default path
   (``norm_dtype`` not supplied) leaves norm dtypes unchanged.

Motivation: residual-stream accumulation across deep transformer / SSM hybrid
stacks loses precision when norm outputs are BF16. Exporting norms at higher
precision lets the inference engine carry residuals in F32 without a runtime
upcast. Norm parameters are a negligible fraction of the checkpoint, so the
disk cost of this option is effectively zero.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from auto_round.export.export_to_autoround.export import (
    _cast_norm_modules,
    _is_norm_module,
    _resolve_dtype_spec,
)

# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class _LlamaRMSNorm(nn.Module):
    """Mimics HF's custom norm naming — class lives outside torch.nn."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))


class _NemotronHRMSNorm(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))


class _Normalize(nn.Module):
    """Vision-style transform — must NOT be classified as a residual norm."""

    def __init__(self):
        super().__init__()


@pytest.mark.parametrize(
    "module,expected",
    [
        (nn.LayerNorm(8), True),
        (nn.GroupNorm(2, 8), True),
        (nn.BatchNorm1d(8), True),
        (nn.BatchNorm2d(8), True),
        (nn.InstanceNorm3d(8), True),
        (_LlamaRMSNorm(), True),
        (_NemotronHRMSNorm(), True),
        (nn.Linear(8, 8), False),
        (nn.Conv2d(1, 1, 3), False),
        (nn.Embedding(16, 8), False),
        (_Normalize(), False),
    ],
)
def test_is_norm_module_classifier(module, expected):
    assert _is_norm_module(module) is expected


# ---------------------------------------------------------------------------
# Dtype spec parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec,expected",
    [
        (None, None),
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        ("float32", torch.float32),
        ("fp32", torch.float32),
        ("F32", torch.float32),
        ("  bf16 ", torch.bfloat16),
        ("bfloat16", torch.bfloat16),
        ("fp16", torch.float16),
        ("half", torch.float16),
    ],
)
def test_resolve_dtype_spec_valid(spec, expected):
    assert _resolve_dtype_spec(spec) is expected


def test_resolve_dtype_spec_bad_string():
    with pytest.raises(ValueError, match="Unsupported dtype string"):
        _resolve_dtype_spec("quadruple")


def test_resolve_dtype_spec_bad_type():
    with pytest.raises(TypeError, match="Expected torch.dtype or str"):
        _resolve_dtype_spec(32)


# ---------------------------------------------------------------------------
# In-memory cast
# ---------------------------------------------------------------------------


def _toy_model(base_dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """A mini-transformer-shaped module: linear + norm + custom RMSNorm."""
    m = nn.Sequential(
        nn.Linear(8, 8, bias=False),
        nn.LayerNorm(8),
        _LlamaRMSNorm(8),
        nn.Linear(8, 8, bias=True),
    )
    return m.to(base_dtype)


def test_cast_norm_modules_casts_only_norms():
    model = _toy_model(torch.bfloat16)
    count = _cast_norm_modules(model, torch.float32)

    # Two norm-shaped modules in the toy: LayerNorm, _LlamaRMSNorm.
    assert count == 2

    # Norms upcast to F32.
    assert model[1].weight.dtype == torch.float32
    assert model[1].bias.dtype == torch.float32
    assert model[2].weight.dtype == torch.float32

    # Linears left alone.
    assert model[0].weight.dtype == torch.bfloat16
    assert model[3].weight.dtype == torch.bfloat16
    assert model[3].bias.dtype == torch.bfloat16


def test_cast_norm_modules_none_is_noop():
    model = _toy_model(torch.bfloat16)
    count = _cast_norm_modules(model, None)
    assert count == 0
    # Nothing changed.
    assert model[1].weight.dtype == torch.bfloat16
    assert model[2].weight.dtype == torch.bfloat16


def test_cast_norm_modules_idempotent():
    model = _toy_model(torch.bfloat16)
    first = _cast_norm_modules(model, torch.float32)
    second = _cast_norm_modules(model, torch.float32)
    assert first == second == 2
    assert model[1].weight.dtype == torch.float32


# ---------------------------------------------------------------------------
# End-to-end: quantize + serialise with norm_dtype=float32
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_end_to_end_norm_dtype_float32(tmp_path):
    """Smallest real model that exercises compressor → export → safetensors.

    opt-125m has ``LayerNorm`` modules; after ``quantize_and_save`` with
    ``norm_dtype=torch.float32`` their weights must be ``float32`` on disk,
    while the quantized Linear layers still pack ``.scales`` in the backend
    default (FP16) and the rest of the unquantized tensors keep their dtypes.
    """
    pytest.importorskip("safetensors")
    from safetensors import safe_open

    from auto_round import AutoRound

    from ...helpers import opt_name_or_path

    save_dir = str(tmp_path / "saved_norm_f32")
    autoround = AutoRound(
        model=opt_name_or_path,
        bits=4,
        group_size=32,
        sym=True,
        iters=1,
        seqlen=2,
        nsamples=1,
    )
    autoround.quantize_and_save(
        output_dir=save_dir,
        format="auto_round",
        norm_dtype=torch.float32,
    )

    import glob

    shards = glob.glob(f"{save_dir}/*.safetensors")
    assert shards, "no safetensors shard emitted"

    norm_dtypes: dict[str, torch.dtype] = {}
    scale_dtypes: dict[str, torch.dtype] = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                if (
                    "layer_norm" in k
                    or k.endswith("final_layer_norm.weight")
                    or k.endswith("final_layer_norm.bias")
                    or k.endswith("self_attn_layer_norm.weight")
                    or k.endswith("self_attn_layer_norm.bias")
                ):
                    norm_dtypes[k] = f.get_tensor(k).dtype
                elif k.endswith(".scales"):
                    scale_dtypes[k] = f.get_tensor(k).dtype

    assert norm_dtypes, "No norm tensors found in shards — test assumption about opt-125m " "parameter names is broken."
    bad = {k: dt for k, dt in norm_dtypes.items() if dt != torch.float32}
    assert not bad, f"Expected every norm tensor to be float32; got non-F32: {bad}"

    # Quantized-path dtype unchanged (F16 is the default for the torch QuantLinear).
    non_f16_scales = {k: dt for k, dt in scale_dtypes.items() if dt != torch.float16}
    assert not non_f16_scales, f"norm_dtype must not leak into scale buffers; got: {non_f16_scales}"


@pytest.mark.slow
@pytest.mark.parametrize(
    "spec,expected",
    [
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
        (torch.bfloat16, torch.bfloat16),
    ],
    ids=["str-fp16", "str-bf16", "dtype-bf16"],
)
def test_end_to_end_norm_dtype_opt_in_variants(tmp_path, spec, expected):
    """E2E coverage for the opt-in ``norm_dtype`` beyond fp32.

    Complements ``test_end_to_end_norm_dtype_float32`` by exercising the
    low-precision overrides and both the string alias and raw ``torch.dtype``
    input forms. Scales must stay fp16 regardless of norm_dtype.
    """
    pytest.importorskip("safetensors")
    from safetensors import safe_open

    from auto_round import AutoRound

    from ...helpers import opt_name_or_path

    save_dir = str(tmp_path / f"saved_norm_{expected}".replace(".", "_"))
    autoround = AutoRound(
        model=opt_name_or_path,
        bits=4,
        group_size=32,
        sym=True,
        iters=1,
        seqlen=2,
        nsamples=1,
    )
    autoround.quantize_and_save(output_dir=save_dir, format="auto_round", norm_dtype=spec)

    import glob

    norm_dtypes: dict[str, torch.dtype] = {}
    scale_dtypes: dict[str, torch.dtype] = {}
    for shard in glob.glob(f"{save_dir}/*.safetensors"):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                if "layer_norm" in k:
                    norm_dtypes[k] = f.get_tensor(k).dtype
                elif k.endswith(".scales"):
                    scale_dtypes[k] = f.get_tensor(k).dtype

    assert norm_dtypes, "no norm tensors seen on disk"
    bad = {k: dt for k, dt in norm_dtypes.items() if dt != expected}
    assert not bad, f"Expected every norm tensor to be {expected}; got: {bad}"

    non_f16_scales = {k: dt for k, dt in scale_dtypes.items() if dt != torch.float16}
    assert not non_f16_scales, f"norm_dtype must not leak into scale buffers; got: {non_f16_scales}"


@pytest.mark.slow
def test_end_to_end_default_path_unchanged(tmp_path):
    """Without ``norm_dtype``, norm weight dtypes match the model default.

    This guards existing checkpoints from silent dtype churn once the new
    option is wired in.
    """
    pytest.importorskip("safetensors")
    from safetensors import safe_open
    from transformers import AutoConfig

    from auto_round import AutoRound
    from auto_round.utils.device import CpuInfo, detect_device

    from ...helpers import opt_name_or_path

    # Expectation (computed before execution): AutoRound's ``_set_amp_dtype``
    # casts the model to bf16 on CPU/HPU (or fp32 if CPU lacks bf16 support)
    # and to the checkpoint native dtype on GPU/XPU. The on-disk norm dtype
    # follows that cast.
    device = str(detect_device(None))
    if device.startswith("cpu"):
        expected_norm_dtype = torch.bfloat16 if CpuInfo().bf16 else torch.float32
    elif "hpu" in device:
        expected_norm_dtype = torch.bfloat16
    else:
        ckpt_dtype = getattr(AutoConfig.from_pretrained(opt_name_or_path), "torch_dtype", None)
        expected_norm_dtype = ckpt_dtype if isinstance(ckpt_dtype, torch.dtype) else torch.float16

    save_dir = str(tmp_path / "saved_default")
    autoround = AutoRound(
        model=opt_name_or_path,
        bits=4,
        group_size=32,
        sym=True,
        iters=1,
        seqlen=2,
        nsamples=1,
    )
    autoround.quantize_and_save(output_dir=save_dir, format="auto_round")

    import glob

    shards = glob.glob(f"{save_dir}/*.safetensors")
    seen_norm_dtypes: set[torch.dtype] = set()
    for shard in shards:
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                if "layer_norm" in k:
                    seen_norm_dtypes.add(f.get_tensor(k).dtype)

    assert seen_norm_dtypes, "default path: no norm tensors seen on disk"
    assert seen_norm_dtypes == {expected_norm_dtype}, (
        f"Default path changed norm dtype. Expected {expected_norm_dtype}, " f"saw {seen_norm_dtypes}."
    )
