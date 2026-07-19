# SVDQuant RTN Residual Iteration and Nunchaku Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add RTN-backed SVDQuant residual iteration and export its MXFP4 residual plus BF16 low-rank branch in a Nunchaku-loadable format.

**Architecture:** Keep SVDQuant model-independent: the transform owns alternating SVD updates, an adapter calls AutoRound's existing RTN QDQ function, and an AutoRound-owned packer serializes E2M1/UE8M0 tensors. Runtime-specific key fusion and metadata live behind model adapters; FLUX is the first integration adapter, not a dependency of the core algorithm.

**Tech Stack:** Python 3.12, PyTorch, AutoRound quantization registry, safetensors, pytest, optional CUDA/Nunchaku integration tests.

---

### Task 1: Add residual iteration configuration

**Files:**
- Modify: `auto_round/algorithms/transforms/svdquant/config.py`
- Modify: `test/test_cpu/algorithms/test_svdquant.py`

**Step 1: Write failing configuration tests**

Add tests that assert:

```python
config = SVDQuantConfig()
assert config.residual_iters == 1
assert config.residual_early_stop is False
assert config.residual_quant_method == "rtn"

with pytest.raises(ValueError, match="residual_iters"):
    SVDQuantConfig(residual_iters=0)

with pytest.raises(ValueError, match="residual_quant_method"):
    SVDQuantConfig(residual_iters=2, residual_quant_method="signround")
```

Also verify `residual_iters=1, residual_quant_method="signround"` is rejected. The
compatibility parameter is fixed to `"rtn"` for every iteration count; downstream
SignRound support is selected independently by the pipeline quantization algorithm.

**Step 2: Run the focused test and verify failure**

Run:

```bash
/home/user2/data/xixi/.venv/bin/python -m pytest test/test_cpu/algorithms/test_svdquant.py -q
```

Expected: failure because the new attributes and validation do not exist.

**Step 3: Implement configuration and validation**

Extend `SVDQuantConfig.__init__` with:

```python
residual_iters: int = (1,)
residual_early_stop: bool = (False,)
residual_quant_method: str = ("rtn",)
```

Normalize the method to lowercase, reject iteration counts below one, and reject non-RTN methods only when `residual_iters > 1`. Include all three values in `__repr__`.

**Step 4: Run tests and verify pass**

Run the command from Step 2.

Expected: all `test_svdquant.py` tests pass.

**Step 5: Commit**

```bash
git add auto_round/algorithms/transforms/svdquant/config.py test/test_cpu/algorithms/test_svdquant.py
git commit -m "feat: configure SVDQuant residual iteration"
```

### Task 2: Add an RTN QDQ adapter using AutoRound's data-type registry

**Files:**
- Create: `auto_round/algorithms/transforms/svdquant/residual.py`
- Create: `test/test_cpu/algorithms/test_svdquant_residual.py`

**Step 1: Write failing QDQ adapter tests**

Test a deterministic `(3, 64)` weight with scheme:

```python
scheme = ResidualQuantScheme(
    data_type="mx_fp4e2m1",
    bits=4,
    group_size=32,
    sym=True,
)
qdq = rtn_qdq_residual(weight, scheme)
assert qdq.shape == weight.shape
assert qdq.dtype == weight.dtype
assert torch.isfinite(qdq).all()
```

Compare it directly with the result returned by `get_quant_func(..., iters=0)` using the same arguments. Add validation tests for missing scheme attributes and MXFP4 group size other than 32.

**Step 2: Run the new test and verify failure**

```bash
/home/user2/data/xixi/.venv/bin/python -m pytest test/test_cpu/algorithms/test_svdquant_residual.py -q
```

Expected: import failure because `residual.py` does not exist.

**Step 3: Implement the adapter**

Implement an immutable `ResidualQuantScheme` plus:

```python
def rtn_qdq_residual(weight: torch.Tensor, scheme: ResidualQuantScheme) -> torch.Tensor:
    quant_func, resolved_dtype = get_quant_func(
        dtype=scheme.data_type,
        bits=scheme.bits,
        sym=scheme.sym,
        disable_opt_rtn=True,
        group_size=scheme.group_size,
        iters=0,
    )
    qdq, _scale_or_exp, _zero = quant_func(
        tensor=weight,
        bits=scheme.bits,
        group_size=scheme.group_size,
        data_type=resolved_dtype.removeprefix("rtn_"),
    )
    return qdq
```

Resolve the exact registered dtype carefully: add a test proving that `mx_fp4e2m1` reaches `quant_mx` with `data_type="mx_fp4e2m1"`, rather than falling back accidentally to another FP4 variant. Do not duplicate E2M1 numerical quantization in this module.

**Step 4: Run focused tests**

Run the command from Step 2.

Expected: all tests pass on CPU.

**Step 5: Commit**

```bash
git add auto_round/algorithms/transforms/svdquant/residual.py test/test_cpu/algorithms/test_svdquant_residual.py
git commit -m "feat: reuse RTN QDQ for SVDQuant residuals"
```

### Task 3: Implement alternating residual iteration

**Files:**
- Modify: `auto_round/algorithms/transforms/svdquant/apply.py`
- Modify: `test/test_cpu/algorithms/test_svdquant.py`
- Modify: `test/test_cpu/algorithms/test_svdquant_residual.py`

**Step 1: Write failing one-round compatibility test**

Use a fixed float32 Linear and compare `residual_iters=1` against an explicit truncated SVD reference. Assert the low-rank factors and residual reconstruct the current one-shot result within tolerance.

**Step 2: Write failing multi-round best-candidate test**

Use a matrix where MXFP4 creates measurable error. Record reconstruction errors for the first and selected candidates:

```python
decomposed = transform._decompose_linear(layer)
q = rtn_qdq_residual(decomposed.residual_linear.weight, scheme)
error = torch.sum((weight_hat - (q + low_rank(decomposed))) ** 2)
assert error <= first_iteration_error
```

Monkeypatch `rtn_qdq_residual` for an early-stop test so iteration two is worse, then assert the first candidate is retained.

**Step 3: Run focused tests and verify failure**

```bash
/home/user2/data/xixi/.venv/bin/python -m pytest \
  test/test_cpu/algorithms/test_svdquant.py \
  test/test_cpu/algorithms/test_svdquant_residual.py -q
```

Expected: multi-round tests fail because `_decompose_linear` only performs one SVD.

**Step 4: Extract scheme resolution from the source Linear**

Build `ResidualQuantScheme` from the copied AutoRound attributes `data_type`, `bits`, `group_size`, and `sym`. Raise an error containing the module/global name when `residual_iters > 1` and the scheme is incomplete.

**Step 5: Implement the alternating loop**

Refactor SVD factorization into a helper returning `(low_rank, down_weight, up_weight)`. For multiple iterations:

```python
quantized_residual = torch.zeros_like(weight_hat)
best = None
best_error = torch.tensor(float("inf"), device=weight_hat.device)

for iteration in range(config.residual_iters):
    low_rank, down, up = truncated_svd(weight_hat - quantized_residual, rank)
    residual = weight_hat - low_rank
    quantized_residual = rtn_qdq_residual(residual, scheme)
    error = torch.sum((weight_hat - quantized_residual - low_rank).square())
    if torch.isfinite(error) and error < best_error:
        best = residual.clone(), down.clone(), up.clone()
        best_error = error
    elif config.residual_early_stop:
        break
```

Keep the existing no-QDQ fast path for `residual_iters=1`. Materialize only the retained high-precision residual and BF16 low-rank factors.

**Step 6: Run all SVDQuant CPU tests**

Run the command from Step 3.

Expected: pass, with one-round compatibility unchanged and multi-round best selection working.

**Step 7: Commit**

```bash
git add auto_round/algorithms/transforms/svdquant/apply.py test/test_cpu/algorithms/test_svdquant.py test/test_cpu/algorithms/test_svdquant_residual.py
git commit -m "feat: iterate RTN residuals in SVDQuant"
```

### Task 4: Implement AutoRound-owned MXFP4 and UE8M0 reference encoding

**Files:**
- Create: `auto_round/export/svdquant_mxfp4.py`
- Create: `test/test_cpu/export/test_svdquant_mxfp4.py`

**Step 1: Write failing E2M1 codebook tests**

Cover all positive representable E2M1 magnitudes and signs:

```text
0, 0.5, 1, 1.5, 2, 3, 4, 6
```

Assert two four-bit codes pack into one byte and unpack exactly to their original codes. Include saturation and tie-rounding cases matching `auto_round.data_type.mxfp.quant_element`.

**Step 2: Write failing UE8M0 tests**

Test exponent-code round trips, zero groups, minimum/maximum exponent saturation, and group size 32. Compare decoded scales with scales used by AutoRound's MXFP4 QDQ.

**Step 3: Run tests and verify failure**

```bash
/home/user2/data/xixi/.venv/bin/python -m pytest test/test_cpu/export/test_svdquant_mxfp4.py -q
```

Expected: import failure because the codec module does not exist.

**Step 4: Implement logical encoding and decoding**

Implement pure PyTorch helpers:

```python
encode_e2m1(values, scales) -> uint8_codes
decode_e2m1(codes, scales, dtype) -> tensor
encode_ue8m0(scales) -> uint8_codes
decode_ue8m0(codes) -> float32_scales
pack_nibbles(codes) -> uint8
unpack_nibbles(packed) -> uint8_codes
```

Keep logical encoding separate from Nunchaku tile reordering. Add source attribution comments where behavior is derived from published/runtime format conventions, without importing external projects.

**Step 5: Verify codec tests**

Run the command from Step 3.

Expected: all logical codec tests pass.

**Step 6: Commit**

```bash
git add auto_round/export/svdquant_mxfp4.py test/test_cpu/export/test_svdquant_mxfp4.py
git commit -m "feat: encode SVDQuant MXFP4 residual tensors"
```

### Task 5: Implement Nunchaku tile packing and reversible QDQ

**Files:**
- Modify: `auto_round/export/svdquant_mxfp4.py`
- Modify: `test/test_cpu/export/test_svdquant_mxfp4.py`
- Reference only: `/home/user2/data/xixi/deepcompressor/deepcompressor/backend/nunchaku/utils.py`
- Fixture source: `/home/user2/data/xixi/flux.1-dev-mxfp4-lowrank-smoke.safetensors`

**Step 1: Create a small known-good fixture in the test**

Do not commit the 6.4 GB model. Extract deterministic slices and expected packed bytes/shapes from one known-good MXFP4 Linear, and encode those expected values directly as small tensors in the test with a comment documenting the source key and commit.

**Step 2: Write failing layout tests**

Test:

- weight padding and two-values-per-byte packing;
- `wscales` `torch.uint8` output;
- Nunchaku tile reorder followed by inverse reorder;
- packed result equality with the known-good fixture;
- unpack/dequant result equality with RTN QDQ within exact representable-value tolerance.

**Step 3: Run tests and verify failure**

Run the Task 4 test command.

Expected: layout tests fail because only logical encoding exists.

**Step 4: Implement packer and inverse reference path**

Add:

```python
class NunchakuMXFP4Packer:
    def pack_residual(self, weight, group_size=32) -> PackedMXFP4: ...
    def unpack_residual(self, qweight, wscales, logical_shape) -> torch.Tensor: ...
```

`PackedMXFP4` records `qweight`, `wscales`, logical shape, and padded shape. Keep inverse unpacking as a test/reference implementation; runtime export writes only Nunchaku-required tensors.

**Step 5: Run codec/layout tests**

Expected: all pass, including known-good fixture comparison.

**Step 6: Commit**

```bash
git add auto_round/export/svdquant_mxfp4.py test/test_cpu/export/test_svdquant_mxfp4.py
git commit -m "feat: pack MXFP4 tensors for Nunchaku"
```

### Task 6: Replace the exporter prototype with runtime-ready packed export

**Files:**
- Modify: `auto_round/export/svdquant_nunchaku.py`
- Modify: `test/test_cpu/export/test_svdquant_nunchaku_export.py`

**Step 1: Write failing packed-provider tests**

Assert default runtime export contains:

```text
0.qweight
0.wscales
0.smooth
0.smooth_orig
0.lora_down
0.lora_up
0.bias
```

Assert it does not contain `0.residual.weight`, `qweight` is byte-packed, `wscales` is `torch.uint8`, and low-rank tensors are BF16 with Nunchaku layouts.

**Step 2: Write failing metadata tests**

Require `quantization_config` to deserialize to:

```python
{
    "method": "svdquant",
    "weight": {"dtype": "fp4_e2m1_all", "scale_dtype": "ue8m0", "group_size": 32},
    "activation": {"dtype": "fp4_e2m1_all", "scale_dtype": "ue8m0", "group_size": 32},
    "rank": 2,
}
```

Keep an explicit `debug_unpacked=True` path for intermediate artifacts only.

**Step 3: Run tests and verify failure**

```bash
/home/user2/data/xixi/.venv/bin/python -m pytest \
  test/test_cpu/export/test_svdquant_mxfp4.py \
  test/test_cpu/export/test_svdquant_nunchaku_export.py -q
```

Expected: prototype exporter tests fail because the default provider writes `residual.weight` and prototype metadata.

**Step 4: Implement `MXFP4ResidualTensorProvider` and strict config**

Use `NunchakuMXFP4Packer` to emit `qweight/wscales`. Validate MXFP4 E2M1, UE8M0, group size 32, consistent rank, finite tensors, and required runtime metadata before writing.

**Step 5: Add the model-adapter protocol and identity adapter**

Add:

```python
class SVDQuantModelAdapter(Protocol):
    def map_module(self, name, module): ...
    def metadata(self, model): ...
    def validate(self, tensors, metadata): ...
```

The identity adapter supports codec tests but marks output as a generic intermediate. Runtime mode must require an adapter that supplies `model_class` and serialized model `config`.

**Step 6: Run exporter tests**

Run the command from Step 3.

Expected: all codec and exporter tests pass; source inspection still confirms there are no DeepCompressor or Nunchaku imports.

**Step 7: Commit**

```bash
git add auto_round/export/svdquant_nunchaku.py test/test_cpu/export/test_svdquant_nunchaku_export.py
git commit -m "feat: export packed SVDQuant MXFP4 artifacts"
```

### Task 7: Register the AutoRound output format

**Files:**
- Modify: `auto_round/utils/common.py`
- Modify: `auto_round/formats.py`
- Modify: `auto_round/export/__init__.py`
- Create: `test/test_cpu/export/test_svdquant_nunchaku_format.py`

**Step 1: Write a failing format-resolution test**

Construct the minimum compressor stub needed by `get_formats` and assert:

```python
formats = get_formats("svdquant_nunchaku", compressor)
assert len(formats) == 1
assert formats[0].format_name == "svdquant_nunchaku"
```

Test that unsupported schemes fail clearly and that output kwargs reach the direct exporter.

**Step 2: Run test and verify failure**

```bash
/home/user2/data/xixi/.venv/bin/python -m pytest test/test_cpu/export/test_svdquant_nunchaku_format.py -q
```

Expected: format is absent from `SUPPORTED_FORMATS` and `OutputFormat._format_list`.

**Step 3: Register and implement the format class**

Add `svdquant_nunchaku` to supported formats and define an `OutputFormat` subclass whose `save_quantized` delegates to `save_svdquant_nunchaku_safetensors`. Keep architecture adapters passed explicitly through exporter kwargs.

**Step 4: Run focused format and exporter tests**

```bash
/home/user2/data/xixi/.venv/bin/python -m pytest \
  test/test_cpu/export/test_svdquant_nunchaku_format.py \
  test/test_cpu/export/test_svdquant_nunchaku_export.py -q
```

Expected: pass.

**Step 5: Commit**

```bash
git add auto_round/utils/common.py auto_round/formats.py auto_round/export/__init__.py test/test_cpu/export/test_svdquant_nunchaku_format.py
git commit -m "feat: register SVDQuant Nunchaku export format"
```

### Task 8: Add the FLUX adapter and comparison gates

**Files:**
- Create: `auto_round/export/svdquant_adapters/__init__.py`
- Create: `auto_round/export/svdquant_adapters/flux.py`
- Create: `test/test_cpu/export/test_svdquant_flux_adapter.py`
- Create: `test/test_cuda/export/test_svdquant_nunchaku_integration.py`
- Modify: `docs/svdquant_nunchaku_mxfp4_review.md`

**Step 1: Write failing FLUX mapping tests**

Use small synthetic tensors with FLUX-like module names. Cover dual transformer blocks, single transformer blocks, QKV/MLP fusion, low-rank concatenation, norm exceptions, padding, and metadata:

```text
model_class = NunchakuFluxTransformer2dModel
weight.dtype = fp4_e2m1_all
activation.dtype = fp4_e2m1_all
rank = configured rank
```

**Step 2: Implement the adapter only against generic export records**

The adapter may import Diffusers model/config types when available, but must not import DeepCompressor or Nunchaku. Keep all FLUX key maps and fusion rules in `flux.py`.

**Step 3: Run CPU adapter tests**

```bash
/home/user2/data/xixi/.venv/bin/python -m pytest test/test_cpu/export/test_svdquant_flux_adapter.py -q
```

Expected: pass without a GPU or Nunchaku installation.

**Step 4: Export a small real artifact and validate safetensors**

Run AutoRound with `residual_iters=1`, RTN, MXFP4 group size 32, and the FLUX adapter. Inspect every key, dtype, shape, metadata field, and file size before runtime loading.

**Step 5: Run QDQ versus Nunchaku single-Linear kernel comparison**

On an SM120/SM121 GPU with the existing xixi Nunchaku environment, compare unpacked QDQ and the Nunchaku kernel using identical BF16 inputs. Reuse the established tolerance from `/home/user2/data/xixi/check_mxfp4_kernel_vs_qdq.py`.

Expected: no layout-scale mismatch and output error within the established BF16 tolerance.

**Step 6: Run end-to-end FLUX comparison**

Use the same BF16 base model, prompt, seed, scheduler, guidance, image dimensions, and 20 steps for:

- the known-good DeepCompressor MXFP4 artifact;
- the AutoRound RTN SVDQuant MXFP4 artifact.

Do not proceed to subjective image comparison if QDQ or the single-kernel gate fails. Save generated images outside the repository.

**Step 7: Update documentation with exact commands and limitations**

Document the AutoRound invocation, export command, QDQ validation, Nunchaku load command, `residual_iters=1` compatibility, RTN-only outer-loop limitation, GPU architecture requirement, and artifact locations.

**Step 8: Run the full focused suite**

```bash
/home/user2/data/xixi/.venv/bin/python -m pytest \
  test/test_cpu/algorithms/test_svdquant.py \
  test/test_cpu/algorithms/test_svdquant_residual.py \
  test/test_cpu/export/test_svdquant_mxfp4.py \
  test/test_cpu/export/test_svdquant_nunchaku_export.py \
  test/test_cpu/export/test_svdquant_nunchaku_format.py \
  test/test_cpu/export/test_svdquant_flux_adapter.py -q
```

Expected: all pass. Run the CUDA integration separately on supported hardware and record pass/skip explicitly.

**Step 9: Commit**

```bash
git add auto_round/export/svdquant_adapters test/test_cpu/export/test_svdquant_flux_adapter.py test/test_cuda/export/test_svdquant_nunchaku_integration.py docs/svdquant_nunchaku_mxfp4_review.md
git commit -m "feat: validate SVDQuant MXFP4 with Nunchaku FLUX"
```

### Task 9: Final regression and dependency audit

**Files:**
- Modify only if a regression test exposes a defect.

**Step 1: Run formatting and lint checks on changed files**

Use the repository's configured formatter/linter commands. Do not apply repository-wide formatting.

**Step 2: Run the focused CPU suite from Task 8**

Expected: all pass.

**Step 3: Run import-isolation audit**

```bash
rg -n "(^| )import (deepcompressor|nunchaku)|from (deepcompressor|nunchaku)" \
  auto_round/algorithms/transforms/svdquant auto_round/export/svdquant_*
```

Expected: no matches.

**Step 4: Inspect the exported file**

Verify no `residual.weight`, no float residual copy, correct metadata, expected rank, `uint8` MXFP4 scales, BF16 low-rank tensors, and no duplicate keys.

**Step 5: Record final verification**

Update the review document with exact CPU test counts, CUDA test result or skip reason, artifact hash, and end-to-end image paths.

**Step 6: Commit verification-only documentation changes**

```bash
git add docs/svdquant_nunchaku_mxfp4_review.md
git commit -m "docs: record SVDQuant MXFP4 verification"
```
