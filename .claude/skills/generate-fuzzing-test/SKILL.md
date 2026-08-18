---
name: generate-fuzzing-test
description: "Generate fuzzing / property-based tests for a new or changed function in AutoRound. Use when a function needs randomized-input testing, robustness checks against NaN/Inf/extreme values, edge-case coverage for tensor shapes and dtypes, invariant/oracle verification, or hardening of quantization math (scale, group_size, cast/round functions). Keywords: fuzz, fuzzing, property-based, hypothesis, random input, robustness, edge case, invariant, oracle."
argument-hint: "Path or name of the function to fuzz (e.g. auto_round/data_type/nvfp.py::cast_to_fp4)"
---

# Generate Fuzzing Tests for a Function

Produce a randomized-input (fuzzing / property-based) test that hardens a target
function against malformed inputs, numerical edge cases, and shape/dtype variety —
without asserting an exact golden output. The goal is to catch crashes, NaN/Inf
leaks, silent shape bugs, and broken invariants.

## When to Use
- A new function was added (data type cast, scale computation, quant/dequant, reshape helper) and needs robustness coverage.
- An existing function changed and you want to verify it still holds its invariants across random inputs.
- A bug was found on an unusual input and you want a regression fuzzer around it.
- The user says "fuzz this", "add property tests", "test random inputs", or "make this robust".

## Procedure

### 1. Analyze the target
Read the function. Record:
- **Signature** — parameter names, types, defaults, and which are tensors vs scalars vs config flags.
- **Input domain** — expected dtype(s) (`fp32`, `bf16`, `float8_*`), rank/shape assumptions, whether `group_size` divisibility or padding is required, valid ranges.
- **Output contract** — return shape/dtype, and what "correct" means structurally (not the exact numbers).
- **Danger points** — division (scale = 1/x), `log2`/`exp`/`frexp`, clamping, `.reshape`, `.to(dtype)`, in-place ops, indexing.

### 2. Choose a fuzzing style
**Prefer `hypothesis`** (property-based) for its shrinking + broad coverage. Fall
back to a manual randomized loop only when `hypothesis` cannot be used.

| Situation | Style |
|-----------|-------|
| Default — any pure numeric/tensor function | **Property-based with `hypothesis`** (`hypothesis.extra.numpy` for arrays → convert to torch). |
| `hypothesis` genuinely unusable (heavy fixtures, unsupported input) | **Manual randomized loop** — `for _ in range(N)` with `torch.rand`/`randn` and seeded RNG. |
| Reproducing a specific crash | **Parametrized edge-case table** plus one of the above. |

`hypothesis` is not yet a repo dependency. Before writing the test, check
`test/unit/test_cpu/requirements.txt`; if `hypothesis` is absent, add it there and
tell the user. If they decline the new dependency, switch to the manual loop.
Use `@settings(deadline=None, max_examples=200)` to keep runs CI-friendly and
avoid flaky deadline failures on slow tensor ops.

### 3. Generate the input space
Cover BOTH random and adversarial inputs. Always include these edge cases:
- **Shapes**: 1-D, 2-D, non-contiguous, empty (`shape[dim]==0`), size-1 dims, and shapes where `group_size` does / does not divide evenly.
- **Values**: all-zeros, all-equal, very large (`1e30`), very small (`1e-30`), negative, mixed sign, `+/-inf`, `NaN`, denormals.
- **Dtypes**: `float32`, `bfloat16`, `float16` (and the function's target quant dtype).
- **Devices**: CPU always; add CUDA branch guarded by `torch.cuda.is_available()` only if relevant.
Always seed RNG (`torch.manual_seed`) so failures are reproducible.

### 4. Assert invariants, not golden values
A fuzzer checks *properties* (oracles), because the input is random. Pick the ones that apply:
- **No crash** — the call returns instead of raising on valid-domain inputs.
- **Finiteness** — `torch.isfinite(out).all()` for finite inputs (a common quantization bug: NaN from divide-by-zero scale).
- **Shape/dtype** — output shape and dtype match the contract.
- **Bounds** — quantized/cast values fall within the format's representable range (e.g. `|out| <= FLOAT4_E2M1_MAX`).
- **Idempotence / round-trip** — `dequant(quant(x))` is close to `x` within tolerance, or `f(f(x)) == f(x)`.
- **Equivariance** — scaling the input by a positive constant scales the output predictably.
- **Reference oracle** — compare against a slow NumPy/pure-python reference (`ref_*` helpers already exist in `data_type` modules) with `torch.testing.assert_close`.

### 5. Place and format the test
- Location follows the tiered layout: fast fuzzers go in `test/unit/test_cpu/<area>/` mirroring the source path (e.g. source `auto_round/data_type/nvfp.py` → test `test/unit/test_cpu/data_type/test_nvfp.py`). See [AGENTS.md](../../../AGENTS.md) for the tier/hardware layout.
- Add to an existing test file for that module if one exists; otherwise create a new `test_<module>.py`.
- Include the Apache 2.0 header (copy from a sibling test file).
- Group cases in a `class Test<Function>Fuzz:` with descriptive method names.
- Line length 120; imports isort/black style; use `pytest`, `torch`, and `torch.testing.assert_close`.

### 6. Run and iterate
- Run `pytest test/unit/test_cpu/<area>/test_<module>.py -x -q`.
- If a random case fails, **do not** just loosen the assertion — decide whether it is a real bug in the target function or an out-of-domain input, then fix the function or tighten the input generator accordingly.
- On a genuine bug, keep the minimal failing input as an explicit `@pytest.mark.parametrize` regression case alongside the fuzzer.

## Template

```python
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ... (full header — copy from a sibling test file)

"""Fuzzing tests for ``auto_round.<module>.<func>``."""

import pytest
import torch

from auto_round.<module> import <func>

SEED = 0
DTYPES = [torch.float32, torch.bfloat16, torch.float16]
SHAPES = [(1,), (8,), (3, 32), (4, 33), (0, 16), (1, 1)]


def _rand(shape, dtype):
    return torch.randn(shape, dtype=torch.float32).to(dtype)


class TestFuncFuzz:
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("shape", SHAPES)
    def test_no_crash_and_finite(self, shape, dtype):
        torch.manual_seed(SEED)
        x = _rand(shape, dtype)
        out = <func>(x)
        assert out.shape == x.shape           # adjust to real contract
        assert torch.isfinite(out.float()).all()

    @pytest.mark.parametrize(
        "x",
        [
            torch.zeros(16),
            torch.full((16,), 1e30),
            torch.full((16,), 1e-30),
            torch.tensor([float("inf"), float("-inf"), float("nan"), 0.0]),
        ],
    )
    def test_edge_values(self, x):
        out = <func>(x)                        # must not raise
        finite_in = torch.isfinite(x)
        assert torch.isfinite(out[finite_in]).all()

    def test_random_loop_invariant(self):
        torch.manual_seed(SEED)
        for _ in range(200):
            x = torch.randn(torch.randint(1, 64, (1,)).item())
            out = <func>(x)
            # e.g. bound / round-trip / reference-oracle invariant here
            assert out.abs().max() <= EXPECTED_MAX
```

### Preferred: `hypothesis` template

```python
# Apache 2.0 header ...
"""Property-based fuzzing for ``auto_round.<module>.<func>``."""

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from auto_round.<module> import <func>

finite = st.floats(min_value=-1e30, max_value=1e30, allow_nan=False, allow_infinity=False, width=32)


class TestFuncFuzz:
    @settings(deadline=None, max_examples=200)
    @given(arr=arrays(dtype=np.float32, shape=array_shapes(min_dims=1, max_dims=2, max_side=64), elements=finite))
    def test_finite_and_bounded(self, arr):
        x = torch.from_numpy(arr)
        out = <func>(x)
        assert torch.isfinite(out.float()).all()   # no NaN/Inf leak
        assert out.abs().max() <= EXPECTED_MAX      # format bound; adjust to contract
```

## AutoRound-specific reminders
- Quantization bugs most often surface as **NaN/Inf from a zero or overflowing scale** — always include an all-zeros and an all-equal input, and assert finiteness.
- When `group_size` matters, generate shapes where the last dim is **not** a multiple of `group_size` to exercise padding paths.
- Many `data_type` modules ship a `ref_*` pure reference — use it as the oracle instead of inventing expected numbers.
- Keep iteration counts small enough for PR CI (a few hundred, not tens of thousands); heavier fuzzing belongs in `test/integration/` or `test/e2e/`.
- Respect `FORCE_BF16` / device availability guards already used in sibling tests.
