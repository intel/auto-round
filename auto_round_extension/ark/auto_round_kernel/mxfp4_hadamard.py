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

# Attention head dim. The quantization group stays at ``GROUP_SIZE`` for every
# transform size, so a 128-element row produces 4 independent MXFP4 groups and
# the output layout is identical to the 32-point case.
HADAMARD_DIM_128 = 128

# Transform sizes the XPU kernel implements: ``GROUP_SIZE * L`` for a power-of-
# two lane count ``L``. Every size above 32 is FWHT-only (Sylvester matrix) and
# fans a row out over ``L`` cooperating sub-group lanes, so ``L`` must divide the
# sub-group size of 32 -- see ``fwht_quant_cooperative`` in
# xpu_mxfp4_hadamard.hpp. 512 is the largest instantiated size; nothing in the
# design stops 1024, it simply has no consumer.
MAX_LANES_PER_ROW = 16
SUPPORTED_HADAMARD_DIMS = tuple(GROUP_SIZE * (1 << i) for i in range(MAX_LANES_PER_ROW.bit_length()))

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


def _validate_hadamard(hadamard_matrix: torch.Tensor, *, check_finite: bool = True) -> int:
    """Validate the matrix and return its dimension ``D``.

    ``D`` is read from the matrix rather than passed in, so a caller selects the
    transform size simply by handing over a ``get_hadamard_matrix(D)`` tensor.
    """
    if not isinstance(hadamard_matrix, torch.Tensor):
        raise TypeError(f"hadamard_matrix must be a torch.Tensor, got {type(hadamard_matrix)}")
    shape = tuple(hadamard_matrix.shape)
    if len(shape) != 2 or shape[0] != shape[1] or shape[0] not in SUPPORTED_HADAMARD_DIMS:
        allowed = " or ".join(f"({d}, {d})" for d in SUPPORTED_HADAMARD_DIMS)
        raise ValueError(f"hadamard_matrix must have shape {allowed}, got {shape}")
    if hadamard_matrix.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"hadamard_matrix must be float32 or float64, got {hadamard_matrix.dtype}")
    # Unlike the checks above, this one reads a device tensor and forces a
    # device->host sync, which costs about as much as the fused kernel itself.
    # Callers that have already established the matrix is the known-good default
    # pass check_finite=False.
    if check_finite and not torch.isfinite(hadamard_matrix).all():
        raise ValueError("hadamard_matrix must contain only finite values")
    return shape[0]


def _require_activation_tensor(x: torch.Tensor) -> None:
    """Reject a non-tensor ``x`` before anything dereferences it.

    Both entry points resolve the Hadamard matrix (and hence ``x.device``)
    before the full activation validation runs, so this cheap check has to
    happen first for a bad ``x`` to raise TypeError rather than AttributeError.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x)}")


def _validate_activation(
    x: torch.Tensor, *, require_xpu: bool, check_finite: bool = True, hadamard_dim: int = HADAMARD_DIM
) -> tuple[int, int]:
    _require_activation_tensor(x)
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"x must be float16 or bfloat16, got {x.dtype}")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    if require_xpu and x.device.type != "xpu":
        raise ValueError(f"mxfp4_hadamard_quant is only supported on XPU, got device {x.device}")
    k = x.shape[-1]
    if k == 0 or x.numel() == 0:
        raise ValueError("x must not be empty")
    # The transform dimension is the stricter of the two constraints (it is a
    # multiple of GROUP_SIZE), so checking it alone is sufficient.
    if k % hadamard_dim != 0:
        raise ValueError(f"the last dimension of x must be a multiple of {hadamard_dim}, got {k}")
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
    for j in range(h.shape[0]):
        acc = acc + x_groups[:, j : j + 1] * h[j]
    return acc


def fwht_transform_reference(x_groups: torch.Tensor, norm: torch.Tensor, *, norm_last: bool = False) -> torch.Tensor:
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

    ``norm_last`` moves the scaling after the butterflies, mirroring the
    cooperative ``D > 32`` kernel. That kernel cannot scale on load: doing so
    leaves a multiply feeding the first stage's add, which the GPU backend fuses
    into an FMA that no compiler flag or pragma was able to suppress, breaking
    bit-exactness. See the normalization note above ``fwht_quant_cooperative``
    in xpu_mxfp4_hadamard.hpp. The two orders round differently and are
    deliberately not bit-exact against each other.
    """
    dim = x_groups.shape[-1]
    scale = norm.to(device=x_groups.device, dtype=torch.float32)
    acc = x_groups.to(torch.float32)
    if not norm_last:
        acc = acc * scale
    lanes = torch.arange(dim, device=acc.device)
    for stage in range(dim.bit_length() - 1):
        h = 1 << stage
        partner = acc[:, lanes ^ h]
        acc = torch.where((lanes & h) != 0, partner - acc, acc + partner)
    if norm_last:
        acc = acc * scale
    return acc


def is_default_hadamard(hadamard_matrix: torch.Tensor) -> bool:
    """True if ``hadamard_matrix`` is exactly the normalized Sylvester matrix.

    Only that matrix may take the FWHT path, because the butterfly network
    implements the Sylvester ordering specifically.
    """
    dim = hadamard_matrix.shape[0]
    # Callers normally pass the tensor returned by get_hadamard_matrix, which is
    # cached per device; recognising it by identity avoids a device->host copy
    # and the sync that torch.equal would impose on every quantization call.
    if hadamard_matrix is _HADAMARD_CACHE.get((dim, str(hadamard_matrix.device))):
        return True
    h = hadamard_matrix.to(torch.float32)
    return bool(torch.equal(h.cpu(), get_hadamard_matrix(dim, "cpu")))


def transform_reference(x_groups: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Dispatch to the FWHT or Path A reference, mirroring the kernel's choice."""
    if is_default_hadamard(h):
        # D > 32 runs the cooperative kernel, which normalizes after the
        # butterflies rather than on load; the reference has to match.
        return fwht_transform_reference(x_groups, h.reshape(-1)[0], norm_last=h.shape[0] > GROUP_SIZE)
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
    The transform dimension is taken from ``hadamard_matrix`` (any of
    :data:`SUPPORTED_HADAMARD_DIMS`); the quantization group is always 32, so a
    128-point row yields 4 groups.
    """
    _require_activation_tensor(x)
    if hadamard_matrix is None:
        hadamard_matrix = get_hadamard_matrix(HADAMARD_DIM, x.device)
    # The matrix is validated next because it determines the divisibility
    # constraint the activation is then checked against.
    dim = _validate_hadamard(hadamard_matrix)
    num_rows, k = _validate_activation(x, require_xpu=False, hadamard_dim=dim)

    h = hadamard_matrix.to(device=x.device, dtype=torch.float32).contiguous()
    y = transform_reference(x.contiguous().reshape(-1, dim), h)
    # The transform works on D-element rows, the quantizer on 32-element groups.
    e8m0, q = _e8m0_and_quantized(y.reshape(-1, GROUP_SIZE))
    codes = _encode_fp4(q).reshape(num_rows, k)
    return pack_codes(codes), e8m0.reshape(num_rows, k // GROUP_SIZE)


def mxfp4_quant_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch FP32 reference for the **quant-only** baseline.

    Quantizes the *raw* activation (no Hadamard transform) with the same frozen
    MXFP4 contract as :func:`mxfp4_hadamard_quant_reference`: per-32-element
    E8M0 scale + packed FP4 codes. It is the reference for the quant-only path
    of :func:`mxfp4_hadamard_quant` (``_quant_only=True``), the Hadamard
    ablation of the bandwidth benchmark: that mode moves byte-identical traffic
    and only drops the transform, so the measured ratio isolates its cost.

    Mathematically it equals ``mxfp4_hadamard_quant_reference(x, I)``: running
    the reference transform with the identity matrix is bit-exact with applying
    no transform at all (it only ever adds exact zeros), so the two agree byte
    for byte.
    """
    num_rows, k = _validate_activation(x, require_xpu=False)
    x_groups = x.contiguous().reshape(-1, GROUP_SIZE).to(torch.float32)
    e8m0, q = _e8m0_and_quantized(x_groups)
    codes = _encode_fp4(q).reshape(num_rows, k)
    return pack_codes(codes), e8m0.reshape(num_rows, k // GROUP_SIZE)


def mxfp4_stream_reference(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference for the **stream-only** roofline baseline.

    The stream-only mode exists to answer "how fast could *any* kernel with this
    access pattern be?". It keeps the fused kernel's loads, packing shape and
    stores but replaces the transform and the quantization math with the
    cheapest data-dependent function available: the low nibble of each element's
    raw bit pattern, packed under the usual even-element-low-nibble rule, plus
    an XOR fold for the scale byte. The output is not a quantization of anything
    and must never be consumed as one.

    This reference is not part of the numerical contract -- it exists so a test
    can prove the baseline kernel actually reads every input element. Without
    it, dead-code elimination of the loads would turn the baseline into an empty
    kernel reporting an unreachable bandwidth, silently inflating every ratio
    measured against it.

    The scale byte folds the four code bytes that hold elements
    ``{0, 1}, {8, 9}, {16, 17}, {24, 25}`` of the group, which is exactly the
    low byte of the kernel's XOR over its four packed 32-bit words.
    """
    num_rows, k = _validate_activation(x, require_xpu=False, check_finite=False)
    # view() reinterprets the 16-bit float as raw bits without converting.
    bits = x.contiguous().view(torch.int16).reshape(-1, GROUP_SIZE).to(torch.int32) & 0x0F
    codes = pack_codes(bits.to(torch.uint8)).reshape(num_rows, k // 2)
    groups = codes.reshape(-1, GROUP_SIZE // 2)
    fold = groups[:, 0] ^ groups[:, 4] ^ groups[:, 8] ^ groups[:, 12]
    return codes, fold.reshape(num_rows, k // GROUP_SIZE)


_XMX_SUPPORTED: bool | None = None


def _xmx_supported() -> bool:
    """True when the current XPU build exposes the XMX path.

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
    _quant_only: bool = False,
    _stream_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Hadamard transform + MXFP4 quantization on XPU.

    Args:
        x: FP16/BF16 XPU activation with ``x.shape[-1] % D == 0``, where ``D``
            is the dimension of ``hadamard_matrix``.
        hadamard_matrix: normalized ``D x D`` Hadamard matrix with ``D`` in
            :data:`SUPPORTED_HADAMARD_DIMS`, i.e. ``32 * 2**n`` for ``n = 0..4``.
            Defaults to the ``32 x 32`` Sylvester matrix returned by
            :func:`get_hadamard_matrix`. ``D = 32`` runs one work-item per
            quantization group; every larger ``D`` is implemented by a
            cooperative ``D / 32``-lane FWHT and supports the Sylvester matrix
            only. Pass ``get_hadamard_matrix(128, dev)`` for the attention
            head-dim transform.
        check_finite: reject NaN/Inf in ``x`` before launching. Off by default:
            the check reads all of ``x`` and syncs on the result, which costs
            several times the fused kernel itself. NaN/Inf are still outside the
            supported input domain -- the kernel simply does not police it on the
            hot path. :func:`mxfp4_hadamard_quant_reference` always checks.
        _force_xmx: private override used by tests/benchmarks (None = auto).
        _quant_only: private quant-only baseline override (default False). When
            True, the Hadamard transform is stripped and the *raw* activation is
            quantized (see :func:`mxfp4_quant_reference`), with identical memory
            traffic to the fused path -- the C++ dispatcher ignores
            use_fwht/use_xmx in this mode and always uses the per-item FWHT
            layout. This is the quant-only baseline: identical traffic, no
            transform, so the benchmark's ``f/q`` ratio isolates its cost.
        _stream_only: private stream-only baseline override (default False).
            When True, both the transform *and* the quantization math are
            stripped, leaving the loads, the packing shape and the stores. The
            byte traffic and the item mapping are identical to the fused path,
            so ``BW_fused / BW_stream_only`` is the kernel's utilization of the
            traffic-matched roofline -- the denominator the benchmark gates on.
            The output is *not* a quantization; see
            :func:`mxfp4_stream_reference`. Mutually exclusive with
            ``_quant_only``.

        Routing is automatic: the normalized Sylvester matrix always takes the
        bit-exact FWHT path (first priority); any other Hadamard matrix falls
        back to the XMX path when the build supports it (relaxed contract:
        H stored in the activation dtype, DPAS accumulation, tolerance-based
        acceptance), otherwise to the bit-exact Path A.

    Returns:
        ``(out_codes, out_scale)`` where ``out_codes`` is ``uint8 [M, K // 2]``
        (two packed FP4 codes per byte) and ``out_scale`` is
        ``uint8 [M, K // 32]`` (one E8M0 exponent per group), with
        ``M = x.numel() // K``.
    """
    from . import cvt_dtype, get_lib, get_stream

    if _quant_only and _stream_only:
        raise ValueError("_quant_only and _stream_only are mutually exclusive")
    _require_activation_tensor(x)

    hadamard_dim = HADAMARD_DIM
    if _quant_only or _stream_only:
        # Baselines run on the flat [total_groups, 32] view, so no Hadamard
        # matrix is involved: no validation and no device comparison. A
        # (default) matrix is still materialised so the pointer argument passed
        # to the C++ binding stays valid; the dispatcher never dereferences it
        # in either baseline mode.
        hadamard_matrix = get_hadamard_matrix(HADAMARD_DIM, x.device)
        use_fwht = True
        use_xmx = False
    elif hadamard_matrix is None:
        # The default matrix is known to be the Sylvester one, so the FWHT path
        # is taken without paying for a comparison on the hot path.
        hadamard_matrix = get_hadamard_matrix(HADAMARD_DIM, x.device)
        use_fwht = True
    else:
        # Structural checks are cheap. The finiteness check is not: it syncs on
        # the device every call. The default matrix is known finite, so only a
        # caller-supplied one pays for it.
        hadamard_dim = _validate_hadamard(hadamard_matrix, check_finite=False)
        use_fwht = is_default_hadamard(hadamard_matrix)
        if not use_fwht:
            if hadamard_dim != HADAMARD_DIM:
                # Only the butterfly network is implemented above D = 32; there
                # is no O(D^2) fallback and no XMX path for it.
                raise NotImplementedError(f"hadamard_dim {hadamard_dim} supports the normalized Sylvester matrix only")
            _validate_hadamard(hadamard_matrix)

    num_rows, k = _validate_activation(x, require_xpu=True, check_finite=check_finite, hadamard_dim=hadamard_dim)

    # Path resolution (auto-router): FWHT has first priority for the Sylvester
    # matrix; any other (custom) matrix falls back to the XMX path when the
    # build supports it (relaxed contract), otherwise to Path A. ``_force_xmx``
    # is a private override used by tests/benchmarks. ``_quant_only`` always
    # disables XMX: it is implemented on the shared per-item FWHT layout, which
    # is exactly the layout the quant-only baseline must match.
    if not (_quant_only or _stream_only):
        if _force_xmx is not None:
            use_xmx = bool(_force_xmx)
        elif use_fwht or hadamard_dim != HADAMARD_DIM:
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
        _quant_only,
        hadamard_dim,
        _stream_only,
    )
    return out_codes, out_scale
