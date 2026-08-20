# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Activation fused HMT + MXFP4 quantization (XPU).

Pipeline::

    FP16/BF16 activation -> 32-point normalized Hadamard -> MXFP4 quantization
    -> packed FP4 codes + E8M0 scales

Frozen MVP contract (Phase 0):

* ``hadamard_dim == group_size == 32`` and ``K % 32 == 0``;
* ``H`` is a normalized 32x32 Hadamard matrix (already contains ``1/sqrt(32)``),
  so ``y = reshape(x, [-1, 32]) @ H`` needs no extra scaling;
* per 32-element group ``amax = max(|y_g|)``,
  ``e8m0 = clamp(floor(log2(amax)) - 2 + 127, 0, 254)`` and the effective scale
  is ``2 ** (e8m0 - 127)`` (standard E8M0, always a power of two);
* ``q = y * 2 ** -(e8m0 - 127)`` is encoded as ``signbit(q) << 3 | magnitude``
  with FP4 (E2M1) magnitude levels ``0, 0.5, 1, 1.5, 2, 3, 4, 6`` and the
  nearest-even thresholds of ``vllm_ext/fp4_utils.py::cast_to_fp4``;
* **zero is canonicalised**: whenever the magnitude index is 0 the sign bit is
  dropped, so the code is ``0x0`` and never ``0x8`` (negative zero). See
  "Canonical zero" below for why this rule is required rather than optional;
* two codes share one byte, the even element occupying the low nibble
  (identical to ``vllm_ext/fp4_utils.py::pack_fp4_to_uint8``);
* an all-zero group produces ``e8m0 = 0`` and all-zero codes;
* NaN/Inf are outside the supported input domain. The reference always rejects
  them; the XPU entry point only does so under ``check_finite=True``, because
  the scan reads the whole activation and syncs, costing several times the
  fused kernel itself.

FP32 transform contract (Phase 2, revised in Phase 3)
-----------------------------------------------------

Bit-exactness between the SYCL kernel and this reference requires a *defined*
summation order for ``y = x_g @ H``. There are two paths, and each is bit-exact
against its own reference; they are deliberately *not* bit-exact against each
other, because a butterfly network and a 32-term dot product round differently.
:func:`transform_reference` mirrors the choice the wrapper makes.

**FWHT (default, used for the normalized Sylvester matrix).** Five butterfly
stages; stage ``s`` pairs each lane with ``lane ^ (1 << s)`` and the lane
holding the high half of the pair computes the difference. A single final
multiply by ``H[0][0] == 1/sqrt(32)`` applies the normalization. Only adds and
subtracts occur, so there is no multiply-add for the compiler to contract, and
the order is fully determined by the stage index.

This path exists for performance and is not merely an optimization detail. The
kernel is intended to be memory bound, but the O(D^2) path below costs 32
multiplies plus 32 adds per lane with FMA disabled, which caps effective
bandwidth at roughly 60% of the measured streaming-copy baseline on Arc Pro B60
*before* accounting for shuffles and matrix loads. The butterfly costs 5 adds
plus one scale, moving the bottleneck back to memory.

**Path A (only for a caller-supplied non-Sylvester matrix).** ``acc`` starts at
``+0.0`` and is updated as ``acc = fp32_add(acc, fp32_mul(x[j], H[j][i]))`` for
``j = 0 .. 31`` in increasing ``j``, with a separate FP32 rounding after the
multiply and after the add (no fused multiply-add, no reassociation). The
kernel enforces this with ``#pragma clang fp contract(off)``.
``torch.matmul`` is deliberately not used for either path because its blocking,
FMA usage and reassociation are unspecified and would make the bit-exact
acceptance criterion untestable.

Canonical zero (Phase 2)
------------------------

There is one quantity the accumulation contract above cannot pin down: the
*sign* of a result that is mathematically zero. When a group of 32 inputs is
constant, every output column except the first cancels exactly, and the residue
left by FP32 rounding is on the order of ``1e-8`` with an order-dependent sign.
Device-side flush-to-zero of FP32 subnormals produces the same ambiguity for
very small inputs, where the CPU reference keeps the subnormal but the GPU
returns a signed zero.

Such a value always quantizes to FP4 magnitude index 0, so the ambiguity can
only ever affect the sign bit, turning ``0x0`` into ``0x8`` (negative zero).
Because ``0x8`` and ``0x0`` dequantize to the same number, no information is
lost by forbidding ``0x8``, and doing so makes the encoding a total function of
the mathematical value rather than of the rounding residue. Both this reference
and the kernel therefore drop the sign bit whenever the magnitude index is 0.

This is a deliberate, documented deviation from
``vllm_ext/fp4_utils.py::pack_fp4_to_uint8``, which applies ``signbit``
unconditionally: that helper encodes already-clean dequantized values, where a
negative zero can only appear if the caller supplied one.
"""

from __future__ import annotations

import torch

HADAMARD_DIM = 32
GROUP_SIZE = 32

# FP4 (E2M1) magnitude levels.
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# (threshold, is_closed_interval, magnitude_index), evaluated in order. Mirrors
# ``cast_to_fp4`` including the alternating ``<=`` / ``<`` boundary operators.
_E2M1_THRESHOLDS = (
    (0.25, True, 0),
    (0.75, False, 1),
    (1.25, True, 2),
    (1.75, False, 3),
    (2.5, True, 4),
    (3.5, False, 5),
    (5.0, True, 6),
)

_HADAMARD_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def _sylvester_hadamard(dim: int) -> torch.Tensor:
    if dim < 1 or (dim & (dim - 1)) != 0:
        raise ValueError(f"Hadamard dimension must be a power of two, got {dim}")
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < dim:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h


def get_hadamard_matrix(dim: int = HADAMARD_DIM, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return the normalized ``dim x dim`` Hadamard matrix (FP32, contiguous)."""
    key = (dim, str(torch.device(device)))
    cached = _HADAMARD_CACHE.get(key)
    if cached is None:
        cached = (_sylvester_hadamard(dim) / (dim**0.5)).to(device=device).contiguous()
        _HADAMARD_CACHE[key] = cached
    return cached


def _validate_hadamard(hadamard_matrix: torch.Tensor, *, check_finite: bool = True) -> None:
    if not isinstance(hadamard_matrix, torch.Tensor):
        raise TypeError(f"hadamard_matrix must be a torch.Tensor, got {type(hadamard_matrix)}")
    if hadamard_matrix.shape != (HADAMARD_DIM, HADAMARD_DIM):
        raise ValueError(
            f"hadamard_matrix must have shape ({HADAMARD_DIM}, {HADAMARD_DIM}), got {tuple(hadamard_matrix.shape)}"
        )
    if hadamard_matrix.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"hadamard_matrix must be float32 or float64, got {hadamard_matrix.dtype}")
    # Unlike the checks above, this one reads a device tensor and forces a
    # device->host sync, which costs about as much as the fused kernel itself.
    # Callers that have already established the matrix is the known-good default
    # pass check_finite=False.
    if check_finite and not torch.isfinite(hadamard_matrix).all():
        raise ValueError("hadamard_matrix must contain only finite values")


def _validate_activation(x: torch.Tensor, *, require_xpu: bool, check_finite: bool = True) -> tuple[int, int]:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"x must be float16 or bfloat16, got {x.dtype}")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    if require_xpu and x.device.type != "xpu":
        raise ValueError(f"mxfp4_hadamard_quant is only supported on XPU, got device {x.device}")
    k = x.shape[-1]
    if k == 0 or x.numel() == 0:
        raise ValueError("x must not be empty")
    if k % GROUP_SIZE != 0:
        raise ValueError(f"the last dimension of x must be a multiple of {GROUP_SIZE}, got {k}")
    # This scan reads all of x and then forces a device->host sync on the
    # result, which on XPU costs roughly 4x the fused kernel itself. It is a
    # debugging aid, not part of the numerical contract, so the device entry
    # point leaves it off by default (see ``check_finite`` there).
    if check_finite and not torch.isfinite(x).all():
        raise ValueError("x must contain only finite values (NaN/Inf are not supported)")
    return x.numel() // k, k


def hadamard_transform_reference(x_groups: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """``x_groups [G, 32] @ h [32, 32]`` under the Path A FP32 accumulation contract.

    Sums over ``j`` in increasing order with a separate FP32 rounding after each
    multiply and each add, matching the kernel loop exactly. ``torch.matmul`` is
    intentionally avoided (unspecified blocking / FMA / reassociation).

    Used only for a caller-supplied non-Sylvester matrix; the default matrix
    goes through :func:`fwht_transform_reference`.
    """
    x_groups = x_groups.to(torch.float32)
    h = h.to(torch.float32)
    acc = torch.zeros_like(x_groups)
    for j in range(HADAMARD_DIM):
        acc = acc + x_groups[:, j : j + 1] * h[j]
    return acc


def fwht_transform_reference(x_groups: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
    """32-point fast Walsh-Hadamard transform under the frozen butterfly contract.

    Computes the same mathematical result as ``x_groups @ H`` for the normalized
    Sylvester matrix ``H``, but in ``log2(32) = 5`` butterfly stages instead of
    32 multiply-accumulates. ``norm = H[0][0] = 1/sqrt(32)`` is applied *first*,
    then stage ``s`` pairs each lane with ``lane ^ (1 << s)``:

        ``acc = (lane & h) ? (partner - acc) : (acc + partner)``

    Normalizing up front rather than at the end costs the same single multiply
    but bounds the intermediates by ``sqrt(32) * max|x|`` instead of
    ``32 * max|x|``, keeping the safe input range identical to Path A. With the
    scale applied last, inputs above ``FP32_MAX / 32`` overflow to infinity even
    though the mathematical result is perfectly representable.

    Taking ``norm`` from the matrix itself (rather than recomputing
    ``1/sqrt(32)``) guarantees the kernel and this reference scale by the
    identical FP32 value.

    Every intermediate is a plain FP32 add or subtract, so there is nothing for
    the compiler to contract into an FMA and the order is fully determined by
    the stage index -- which is what keeps this bit-exact against the kernel.
    """
    acc = x_groups.to(torch.float32) * norm.to(device=x_groups.device, dtype=torch.float32)
    lanes = torch.arange(HADAMARD_DIM, device=acc.device)
    for stage in range(HADAMARD_DIM.bit_length() - 1):
        h = 1 << stage
        partner = acc[:, lanes ^ h]
        acc = torch.where((lanes & h) != 0, partner - acc, acc + partner)
    return acc


def is_default_hadamard(hadamard_matrix: torch.Tensor) -> bool:
    """True if ``hadamard_matrix`` is exactly the normalized Sylvester matrix.

    Only that matrix may take the FWHT path, because the butterfly network
    implements the Sylvester ordering specifically.
    """
    # Callers normally pass the tensor returned by get_hadamard_matrix, which is
    # cached per device; recognising it by identity avoids a device->host copy
    # and the sync that torch.equal would impose on every quantization call.
    if hadamard_matrix is _HADAMARD_CACHE.get((HADAMARD_DIM, str(hadamard_matrix.device))):
        return True
    h = hadamard_matrix.to(torch.float32)
    return bool(torch.equal(h.cpu(), get_hadamard_matrix(HADAMARD_DIM, "cpu")))


def transform_reference(x_groups: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Dispatch to the FWHT or Path A reference, mirroring the kernel's choice."""
    if is_default_hadamard(h):
        return fwht_transform_reference(x_groups, h.reshape(-1)[0])
    return hadamard_transform_reference(x_groups, h)


def _e8m0_and_quantized(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(e8m0 [G], q [G, 32])`` for FP32 transformed groups ``y [G, 32]``."""
    amax = y.abs().amax(dim=-1)
    # frexp: amax = mantissa * 2 ** exp with mantissa in [0.5, 1)
    # => floor(log2(amax)) == exp - 1 (exact, also for exact powers of two).
    _, exp = torch.frexp(amax)
    scale_exp = exp.to(torch.int32) - 1 - 2
    e8m0 = torch.clamp(scale_exp + 127, 0, 254)
    # torch.frexp(inf) returns exponent 0, which would silently yield e8m0 = 124.
    # The kernel uses ilogb, which saturates, so an overflowed group must clamp
    # to 254 here too or the two would disagree. y can only be non-finite when
    # the transform of a finite input overflowed FP32, i.e. for inputs beyond
    # the documented safe range; the group is garbage either way, but the two
    # implementations must still agree on it.
    e8m0 = torch.where(torch.isfinite(amax), e8m0, torch.full_like(e8m0, 254))
    zero_group = amax == 0
    e8m0 = torch.where(zero_group, torch.zeros_like(e8m0), e8m0)
    q = torch.ldexp(y, -(e8m0 - 127).unsqueeze(-1))
    q = torch.where(zero_group.unsqueeze(-1), torch.zeros_like(q), q)
    return e8m0.to(torch.uint8), q


def _encode_fp4(q: torch.Tensor) -> torch.Tensor:
    """Encode FP32 values into 4-bit ``sign << 3 | magnitude_index`` codes."""
    a = q.abs()
    idx = torch.full_like(a, len(E2M1_VALUES) - 1, dtype=torch.int32)
    for threshold, closed, value in reversed(_E2M1_THRESHOLDS):
        hit = a <= threshold if closed else a < threshold
        idx = torch.where(hit, torch.full_like(idx, value), idx)
    sign = torch.signbit(q).to(torch.int32) << 3
    # Canonical zero: magnitude 0 always encodes as 0x0, never 0x8 (negative
    # zero). The sign of a value that rounds to zero is not reproducible across
    # implementations, so it must not reach the output. See module docstring.
    return torch.where(idx == 0, idx, idx | sign).to(torch.uint8)


def pack_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit codes ``[M, K]`` into bytes ``[M, K // 2]`` (even element = low nibble)."""
    low = codes[..., 0::2].to(torch.uint8) & 0x0F
    high = codes[..., 1::2].to(torch.uint8) & 0x0F
    return low | (high << 4)


def mxfp4_hadamard_quant_reference(
    x: torch.Tensor, hadamard_matrix: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch FP32 reference for :func:`mxfp4_hadamard_quant`.

    Runs on any device (including CPU) and defines the frozen numerical contract.
    """
    # x is validated first so that a non-tensor argument raises TypeError rather
    # than an attribute error while resolving the default Hadamard matrix.
    num_rows, k = _validate_activation(x, require_xpu=False)
    if hadamard_matrix is None:
        hadamard_matrix = get_hadamard_matrix(HADAMARD_DIM, x.device)
    _validate_hadamard(hadamard_matrix)

    h = hadamard_matrix.to(device=x.device, dtype=torch.float32).contiguous()
    y = transform_reference(x.contiguous().reshape(-1, HADAMARD_DIM), h)
    e8m0, q = _e8m0_and_quantized(y)
    codes = _encode_fp4(q).reshape(num_rows, k)
    return pack_codes(codes), e8m0.reshape(num_rows, k // GROUP_SIZE)


_XMX_SUPPORTED: bool | None = None


def _xmx_supported() -> bool:
    """True when the current XPU build exposes the XMX fast path.

    Probes once by forcing the XMX path on a tiny tensor; the C++ binding raises
    ``RuntimeError`` when ARK_SYCL_TLA is not compiled in. The result is cached.
    """
    global _XMX_SUPPORTED
    if _XMX_SUPPORTED is None:
        try:
            x = torch.zeros(1, GROUP_SIZE, dtype=torch.float16, device="xpu")
            mxfp4_hadamard_quant(x, _force_xmx=True)
            _XMX_SUPPORTED = True
        except (RuntimeError, ValueError, NotImplementedError):
            _XMX_SUPPORTED = False
    return _XMX_SUPPORTED


def mxfp4_hadamard_quant(
    x: torch.Tensor,
    hadamard_matrix: torch.Tensor | None = None,
    *,
    check_finite: bool = False,
    _force_xmx: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused 32-point Hadamard transform + MXFP4 quantization on XPU.

    Args:
        x: FP16/BF16 XPU activation with ``x.shape[-1] % 32 == 0``.
        hadamard_matrix: normalized ``32 x 32`` Hadamard matrix. Defaults to the
            Sylvester matrix returned by :func:`get_hadamard_matrix`.
        check_finite: reject NaN/Inf in ``x`` before launching. Off by default:
            the check reads all of ``x`` and syncs on the result, which costs
            several times the fused kernel itself. NaN/Inf are still outside the
            supported input domain -- the kernel simply does not police it on the
            hot path. :func:`mxfp4_hadamard_quant_reference` always checks.
        _force_xmx: private override used by tests/benchmarks (None = auto).

        Routing is automatic: the normalized Sylvester matrix always takes the
        bit-exact FWHT path (first priority); any other Hadamard matrix falls
        back to the XMX fast path when the build supports it (relaxed contract:
        H stored in the activation dtype, DPAS accumulation, tolerance-based
        acceptance -- see ``xpu_mxfp4_hadamard_design_revised.md``
        §11.4/§11.10), otherwise to the bit-exact Path A.

    Returns:
        ``(out_codes, out_scale)`` where ``out_codes`` is ``uint8 [M, K // 2]``
        (two packed FP4 codes per byte) and ``out_scale`` is
        ``uint8 [M, K // 32]`` (one E8M0 exponent per group), with
        ``M = x.numel() // K``.
    """
    from . import cvt_dtype, get_lib, get_stream

    num_rows, k = _validate_activation(x, require_xpu=True, check_finite=check_finite)
    if hadamard_matrix is None:
        # The default matrix is known to be the Sylvester one, so the FWHT path
        # is taken without paying for a comparison on the hot path.
        hadamard_matrix = get_hadamard_matrix(HADAMARD_DIM, x.device)
        use_fwht = True
    else:
        # Structural checks are cheap. The finiteness check is not: it syncs on
        # the device every call. The default matrix is known finite, so only a
        # caller-supplied one pays for it.
        _validate_hadamard(hadamard_matrix, check_finite=False)
        use_fwht = is_default_hadamard(hadamard_matrix)
        if not use_fwht:
            _validate_hadamard(hadamard_matrix)

    # Path resolution (auto-router): FWHT has first priority for the Sylvester
    # matrix; any other (custom) matrix falls back to the XMX fast path when the
    # build supports it (relaxed contract), otherwise to Path A. ``_force_xmx``
    # is a private override used by tests/benchmarks.
    if _force_xmx is not None:
        use_xmx = bool(_force_xmx)
    elif use_fwht:
        use_xmx = False
    else:
        use_xmx = _xmx_supported()

    lib = get_lib(x)
    if lib is None or not hasattr(lib, "mxfp4_hadamard_quant"):
        raise NotImplementedError("Current XPU build does not expose mxfp4_hadamard_quant")

    x_arg = x.contiguous().reshape(num_rows, k)
    h_arg = hadamard_matrix.to(device=x.device, dtype=torch.float32).contiguous()
    out_codes = torch.empty((num_rows, k // 2), dtype=torch.uint8, device=x.device)
    out_scale = torch.empty((num_rows, k // GROUP_SIZE), dtype=torch.uint8, device=x.device)

    lib.mxfp4_hadamard_quant(
        get_stream(x_arg),
        x_arg.data_ptr(),
        h_arg.data_ptr(),
        out_codes.data_ptr(),
        out_scale.data_ptr(),
        num_rows,
        k,
        cvt_dtype(x_arg.dtype),
        use_fwht,
        use_xmx,
    )
    return out_codes, out_scale
