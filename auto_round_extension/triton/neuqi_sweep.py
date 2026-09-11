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
"""Triton kernel for the NeUQI all-candidates zero-point sweep.

One launch evaluates every per-group scale candidate (``scales``, [C, K])
against every integer zero point for a chunk of groups. Unlike the
``torch.compile``-fused expression -- whose kernel parallelizes over the
(candidate, zero-point) lanes and therefore re-reads each weight element from
L2 once per lane -- this kernel loads each element of a [BC, g] data tile into
registers exactly once and computes all K x Z losses from registers
(amplification 1x). Measured on an RTX 3090 (2M groups, g=64, bits=4, 64+32
candidate grid, fp32): 0.216 s vs 0.488 s (compiled batched expression) and
12.3 s (eager chunked sweep). Selections match the torch paths' tie profile
(same ~0.07% flipped groups, worst fp64 relative loss difference ~5e-5): the
kernel rounds with round-half-to-even (matching ``torch.round``; libdevice
``round`` is half-away-from-zero and would change selections) and keeps the
first-minimum zero-point rule via a strict ``<`` with ascending zero points.

Contract (``neuqi_sweep_triton``):
    data:   [C, g] float32, contiguous, CUDA
    qw:     [C, g] float32, contiguous, CUDA, or ``None``
    scales: [C, K] float32, contiguous, CUDA
    maxq:   maximum integer value (``2**bits - 1``)
    -> (loss [C, K] float32, argmin zero point [C, K] int32)

Contract (``sym_search_triton``): the symmetric scale search with the same
registers-resident structure, one data load per element and all K candidates
under BOTH clamp conventions computed from registers. Rounding and tie rules
match ``_sym_loss_chunk`` (round-half-to-even; the standard convention wins a
loss tie against the mirrored one; ascending-candidate first-minimum).

    data:   [C, g] float32, contiguous, CUDA
    qw:     [C, g] float32, contiguous, CUDA, or ``None``
    scales: [C, K] float32 POSITIVE per-group candidates, contiguous, CUDA
    nmax:   symmetric integer bound (``2**(bits - 1)``)
    -> (best_loss [C] float32, best candidate index [C] int64,
        best_mirror [C] bool: the mirrored convention won that group)

Contract (``sym_search_shared_triton``): the coarse stage of the symmetric
search in NORMALIZED space. The driver precomputes dn = w / s0 once (s0 =
per-group no-clip scale); every group then shares the same candidate fracs
f_k, so the kernel multiplies dn by the SHARED 1/f_k scalar instead of a
per-(group, candidate) scale -- no [C, K] scales tensor at all, no division
inside the candidate loop. The loss omits the s0^2 factor (argmin within a
group is invariant); the driver unnormalizes the winner for cross-stage
comparisons. Measured RTX 3090, 2M groups, g=128, 256+64 candidates: 0.24 s
vs 0.57-0.79 s for the compiled core (parity: same ~0.5% tie-flip profile,
max fp64 rel diff 1.7e-6 vs eager).

    dn:     [C, g] float32, contiguous, CUDA (w / s0)
    qw:     [C, g] float32, contiguous, CUDA, or ``None``
    fracs:  [K] float32 POSITIVE shared multipliers, contiguous, CUDA
    invf:   [K] float32, 1 / fracs, contiguous, CUDA, same device
    nmax:   symmetric integer bound (``2**(bits - 1)``)
    -> (best_loss [C] float32 NORMALIZED, best candidate index [C] int64,
        best_mirror [C] bool: the mirrored convention won that group)
"""

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - depends on host toolchain
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = e


if triton is not None:

    @triton.jit
    def _rint_f32(x):
        """Native round-half-to-even: cvt.rni.f32.f32 (exact torch.round)."""
        return tl.inline_asm_elementwise("cvt.rni.f32.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _neuqi_sweep_kernel(
        data_ptr,
        qw_ptr,
        scales_ptr,
        loss_ptr,
        zp_ptr,
        C,
        G,
        K,
        NZ,
        MAXQ: tl.constexpr,
        HAS_QW: tl.constexpr,
        BC: tl.constexpr,
        GP: tl.constexpr,
        NOMASK: tl.constexpr,
        ZU: tl.constexpr,
    ):
        """Registers-resident (candidate, zero-point) sweep.

        Rounding is native cvt.rni.f32.f32 (round-half-to-even, exactly
        torch.round; measured selection-identical to the previous arithmetic
        emulation). NOMASK skips the tail-group mask selects when GP == G
        (pow2 group size). ZU statically unrolls the zero-point loop when
        MAXQ + 1 <= 32 (bits <= 5) for ILP -- wider bit widths keep the
        runtime loop to bound compile time and register pressure.
        """
        pid = tl.program_id(0)
        c_off = pid * BC + tl.arange(0, BC)
        g_off = tl.arange(0, GP)
        cm = c_off < C
        gm = g_off < G
        d = tl.load(data_ptr + c_off[:, None] * G + g_off[None, :], mask=cm[:, None] & gm[None, :], other=0.0)
        if HAS_QW:
            qwt = tl.load(qw_ptr + c_off[:, None] * G + g_off[None, :], mask=cm[:, None] & gm[None, :], other=0.0)
        else:
            qwt = tl.zeros((BC, GP), dtype=tl.float32) + 1.0
        for k in range(0, K):
            sc = tl.load(scales_ptr + c_off * K + k, mask=cm, other=1.0)  # [BC]
            x = d / sc[:, None]
            r = _rint_f32(x)
            r = tl.minimum(tl.maximum(r, -MAXQ - 1.0), 2.0 * MAXQ + 1.0)
            best = tl.zeros((BC,), dtype=tl.float32) + float("inf")
            bz = tl.zeros((BC,), dtype=tl.int32)
            if ZU:
                for z in tl.static_range(0, MAXQ + 1):
                    q = tl.minimum(tl.maximum(r + z, 0.0), MAXQ)
                    err = sc[:, None] * (q - z) - d
                    loss2 = err * err * qwt
                    if NOMASK:
                        acc = tl.sum(loss2, axis=1)
                    else:
                        acc = tl.sum(tl.where(gm[None, :], loss2, 0.0), axis=1)
                    better = acc < best  # strict: ascending z keeps the first minimum
                    best = tl.where(better, acc, best)
                    bz = tl.where(better, z, bz)
            else:
                for z in range(0, NZ):
                    q = tl.minimum(tl.maximum(r + z, 0.0), MAXQ)
                    err = sc[:, None] * (q - z) - d
                    loss2 = err * err * qwt
                    if NOMASK:
                        acc = tl.sum(loss2, axis=1)
                    else:
                        acc = tl.sum(tl.where(gm[None, :], loss2, 0.0), axis=1)
                    better = acc < best
                    best = tl.where(better, acc, best)
                    bz = tl.where(better, z, bz)
            tl.store(loss_ptr + c_off * K + k, best, mask=cm)
            tl.store(zp_ptr + c_off * K + k, bz, mask=cm)


if triton is not None:

    @triton.jit
    def _sym_search_kernel(
        data_ptr,
        qw_ptr,
        scales_ptr,
        loss_ptr,
        idx_ptr,
        mirror_ptr,
        C,
        G,
        K,
        NMAX: tl.constexpr,
        HAS_QW: tl.constexpr,
        BC: tl.constexpr,
        GP: tl.constexpr,
    ):
        pid = tl.program_id(0)
        c_off = pid * BC + tl.arange(0, BC)
        g_off = tl.arange(0, GP)
        cm = c_off < C
        gm = g_off < G
        d = tl.load(data_ptr + c_off[:, None] * G + g_off[None, :], mask=cm[:, None] & gm[None, :], other=0.0)
        if HAS_QW:
            qwt = tl.load(qw_ptr + c_off[:, None] * G + g_off[None, :], mask=cm[:, None] & gm[None, :], other=0.0)
        else:
            qwt = tl.zeros((BC, GP), dtype=tl.float32) + 1.0
        best = tl.zeros((BC,), dtype=tl.float32) + float("inf")
        bk = tl.zeros((BC,), dtype=tl.int32)
        bmir = tl.zeros((BC,), dtype=tl.int32)
        for k in range(0, K):
            sc = tl.load(scales_ptr + c_off * K + k, mask=cm, other=1.0)  # [BC]
            x = d / sc[:, None]
            fl = tl.floor(x)
            fr = x - fl
            # round-half-to-even (matches torch.round)
            fl_odd = fl - 2.0 * tl.floor(fl * 0.5) == 1.0
            take_up = (fr > 0.5) | ((fr == 0.5) & fl_odd)
            r = fl + take_up.to(tl.float32)
            # standard convention: clamp(-nmax, nmax - 1), dequant s * q
            q_std = tl.minimum(tl.maximum(r, -NMAX), NMAX - 1.0)
            e_std = sc[:, None] * q_std - d
            l_std = tl.sum(tl.where(gm[None, :], e_std * e_std * qwt, 0.0), axis=1)
            # mirrored convention (negative-scale family): clamp(-(nmax - 1), nmax)
            q_mir = tl.minimum(tl.maximum(r, -(NMAX - 1.0)), NMAX)
            e_mir = sc[:, None] * q_mir - d
            l_mir = tl.sum(tl.where(gm[None, :], e_mir * e_mir * qwt, 0.0), axis=1)
            mir = l_mir < l_std  # strict: the standard convention wins a tie
            loss = tl.where(mir, l_mir, l_std)
            better = loss < best  # strict: ascending k keeps the first minimum
            best = tl.where(better, loss, best)
            bk = tl.where(better, k, bk)
            bmir = tl.where(better, mir.to(tl.int32), bmir)
        tl.store(loss_ptr + c_off, best, mask=cm)
        tl.store(idx_ptr + c_off, bk, mask=cm)
        tl.store(mirror_ptr + c_off, bmir, mask=cm)


if triton is not None:

    @triton.jit
    def _neuqi_shared_kernel(
        dn_ptr,
        qw_ptr,
        fracs_ptr,
        invf_ptr,
        loss_ptr,
        kp_ptr,
        zp_ptr,
        C,
        G,
        K,
        MAXQ: tl.constexpr,
        HAS_QW: tl.constexpr,
        BC: tl.constexpr,
        GP: tl.constexpr,
        NOMASK: tl.constexpr,
        ZU: tl.constexpr,
    ):
        """Coarse stage of the joint (scale, zero-point) search in NORMALIZED space.

        The driver precomputes dn = w / s0 once (s0 = per-group min-max
        scale); every group then shares the same candidate fracs f_k, so the
        kernel multiplies dn by the SHARED 1/f_k scalar -- no [C, K] scales
        tensor, no division inside the candidate loop. The loss omits the
        s0^2 factor (the argmin over (k, z) within a group is invariant); the
        driver unnormalizes the winner. In-kernel argmin over k keeps the
        ascending-k first-minimum rule of torch's min(dim=1). Rounding is
        native cvt.rni.f32.f32; tie rules match the reference sweep (ascending
        z keeps the first minimum, strict <).
        """
        pid = tl.program_id(0)
        c_off = pid * BC + tl.arange(0, BC)
        g_off = tl.arange(0, GP)
        cm = c_off < C
        gm = g_off < G
        d = tl.load(dn_ptr + c_off[:, None] * G + g_off[None, :], mask=cm[:, None] & gm[None, :], other=0.0)
        if HAS_QW:
            qwt = tl.load(qw_ptr + c_off[:, None] * G + g_off[None, :], mask=cm[:, None] & gm[None, :], other=0.0)
        else:
            qwt = tl.zeros((BC, GP), dtype=tl.float32) + 1.0
        best = tl.zeros((BC,), dtype=tl.float32) + float("inf")
        bk = tl.zeros((BC,), dtype=tl.int32)
        bz = tl.zeros((BC,), dtype=tl.int32)
        for k in range(0, K):
            f = tl.load(fracs_ptr + k)
            invf = tl.load(invf_ptr + k)
            x = d * invf
            r = _rint_f32(x)
            r = tl.minimum(tl.maximum(r, -MAXQ - 1.0), 2.0 * MAXQ + 1.0)
            l_best = tl.zeros((BC,), dtype=tl.float32) + float("inf")
            l_bz = tl.zeros((BC,), dtype=tl.int32)
            if ZU:
                for z in tl.static_range(0, MAXQ + 1):
                    q = tl.minimum(tl.maximum(r + z, 0.0), MAXQ)
                    err = f * (q - z) - d
                    loss2 = err * err * qwt
                    if NOMASK:
                        acc = tl.sum(loss2, axis=1)
                    else:
                        acc = tl.sum(tl.where(gm[None, :], loss2, 0.0), axis=1)
                    better = acc < l_best
                    l_best = tl.where(better, acc, l_best)
                    l_bz = tl.where(better, z, l_bz)
            else:
                for z in range(0, MAXQ + 1):
                    q = tl.minimum(tl.maximum(r + z, 0.0), MAXQ)
                    err = f * (q - z) - d
                    loss2 = err * err * qwt
                    if NOMASK:
                        acc = tl.sum(loss2, axis=1)
                    else:
                        acc = tl.sum(tl.where(gm[None, :], loss2, 0.0), axis=1)
                    better = acc < l_best
                    l_best = tl.where(better, acc, l_best)
                    l_bz = tl.where(better, z, l_bz)
            better_k = l_best < best
            best = tl.where(better_k, l_best, best)
            bk = tl.where(better_k, k, bk)
            bz = tl.where(better_k, l_bz, bz)
        tl.store(loss_ptr + c_off, best, mask=cm)
        tl.store(kp_ptr + c_off, bk, mask=cm)
        tl.store(zp_ptr + c_off, bz, mask=cm)


def neuqi_shared_sweep_triton(dn, qw, fracs, invf, maxq):
    """Coarse-stage joint (scale, zero-point) search in normalized space on CUDA.

    Returns (best NORMALIZED loss [C] float32, best candidate index [C] int64,
    best zero point [C] int32); the driver unnormalizes the loss by s0^2 and
    maps the index through the shared frac grid.
    """
    assert triton is not None, f"triton is not importable: {_TRITON_IMPORT_ERROR}"
    assert dn.is_cuda and dn.dtype == torch.float32 and dn.is_contiguous()
    assert fracs.is_contiguous() and fracs.dtype == torch.float32 and fracs.ndim == 1
    assert invf.is_contiguous() and invf.dtype == torch.float32 and invf.shape == fracs.shape
    if qw is not None:
        assert qw.dtype == torch.float32 and qw.is_contiguous() and qw.shape == dn.shape
    _dev = dn.device
    assert fracs.device == _dev, f"neuqi_shared: fracs on {fracs.device} but dn on {_dev}"
    assert invf.device == _dev, f"neuqi_shared: invf on {invf.device} but dn on {_dev}"
    if qw is not None:
        assert qw.device == _dev, f"neuqi_shared: qw on {qw.device} but dn on {_dev}"

    c, g = dn.shape
    k = fracs.numel()
    gp = _next_pow2(g)
    bc = max(1, min(64, 4096 // gp))
    loss = torch.empty((c,), device=_dev, dtype=torch.float32)
    kp = torch.empty((c,), device=_dev, dtype=torch.int32)
    zp = torch.empty((c,), device=_dev, dtype=torch.int32)
    with torch.cuda.device(_dev):  # Triton launches on the CURRENT device
        _neuqi_shared_kernel[(triton.cdiv(c, bc),)](
            dn,
            qw if qw is not None else dn,  # dummy pointer when unweighted (dead lanes)
            fracs,
            invf,
            loss,
            kp,
            zp,
            c,
            g,
            k,
            MAXQ=maxq,
            HAS_QW=qw is not None,
            BC=bc,
            GP=gp,
            NOMASK=gp == g,
            ZU=maxq + 1 <= 32,
            num_warps=4,
        )
    return loss, kp.long(), zp


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def neuqi_sweep_triton(data, qw, scales, maxq):
    """Run the all-candidates zero-point sweep on CUDA (see module docstring)."""
    assert triton is not None, f"triton is not importable: {_TRITON_IMPORT_ERROR}"
    assert data.is_cuda and data.dtype == torch.float32 and data.is_contiguous()
    assert scales.is_contiguous() and scales.dtype == torch.float32
    if qw is not None:
        assert qw.dtype == torch.float32 and qw.is_contiguous() and qw.shape == data.shape
    # Triton dereferences raw pointers with no device consistency check -- a
    # cross-device input surfaces as a delayed "device-side assert" far from
    # the launch. Fail here, naming the offending tensor, instead.
    _dev = data.device
    assert scales.device == _dev, f"neuqi_sweep: scales on {scales.device} but data on {_dev}"
    if qw is not None:
        assert qw.device == _dev, f"neuqi_sweep: qw on {qw.device} but data on {_dev}"

    c, g = data.shape
    k = scales.shape[1]
    nz = maxq + 1
    gp = _next_pow2(g)
    bc = max(1, min(64, 4096 // gp))
    loss = torch.empty((c, k), device=data.device, dtype=torch.float32)
    zp = torch.empty((c, k), device=data.device, dtype=torch.int32)
    # Triton launches on the CURRENT cuda device; with tensors on another GPU
    # (block homes under streaming rotation) the launcher rejects the pointer
    # ("cannot be accessed from Triton (cpu tensor?)") or misbehaves. Pin the
    # context to the data's device for the launch.
    with torch.cuda.device(_dev):
        return _launch_sweep(data, qw, scales, loss, zp, c, g, k, nz, maxq, bc, gp)


def sym_search_triton(data, qw, scales, nmax):
    """Evaluate all per-group scale candidates under both symmetric clamp
    conventions on CUDA (see module docstring)."""
    assert triton is not None, f"triton is not importable: {_TRITON_IMPORT_ERROR}"
    assert data.is_cuda and data.dtype == torch.float32 and data.is_contiguous()
    assert scales.is_contiguous() and scales.dtype == torch.float32
    if qw is not None:
        assert qw.dtype == torch.float32 and qw.is_contiguous() and qw.shape == data.shape
    _dev = data.device
    assert scales.device == _dev, f"sym_search: scales on {scales.device} but data on {_dev}"
    if qw is not None:
        assert qw.device == _dev, f"sym_search: qw on {qw.device} but data on {_dev}"

    c, g = data.shape
    k = scales.shape[1]
    gp = _next_pow2(g)
    bc = max(1, min(64, 4096 // gp))
    loss = torch.empty((c,), device=_dev, dtype=torch.float32)
    idx = torch.empty((c,), device=_dev, dtype=torch.int32)
    mirror = torch.empty((c,), device=_dev, dtype=torch.int32)
    with torch.cuda.device(_dev):  # Triton launches on the CURRENT device
        _sym_search_kernel[(triton.cdiv(c, bc),)](
            data,
            qw if qw is not None else data,  # dummy pointer when unweighted
            scales,
            loss,
            idx,
            mirror,
            c,
            g,
            k,
            NMAX=nmax,
            HAS_QW=qw is not None,
            BC=bc,
            GP=gp,
            num_warps=4,
        )
    return loss, idx.long(), mirror.bool()


if triton is not None:

    @triton.jit
    def _sym_shared_kernel(
        dn_ptr,
        qw_ptr,
        fracs_ptr,
        invf_ptr,
        loss_ptr,
        idx_ptr,
        mirror_ptr,
        C,
        G,
        K,
        NMAX: tl.constexpr,
        HAS_QW: tl.constexpr,
        BC: tl.constexpr,
        GP: tl.constexpr,
        NOMASK: tl.constexpr,
    ):
        """Coarse-stage symmetric search in normalized space (see module docstring).

        Rounding is native cvt.rni.f32.f32 (round-half-to-even, matching
        torch.round). Tie rules match _sym_loss_chunk: the standard convention
        wins a loss tie (strict <), ascending-k first-minimum (strict <).
        NOMASK skips the tail-group mask selects when GP == G (pow2 group size).
        """
        pid = tl.program_id(0)
        c_off = pid * BC + tl.arange(0, BC)
        g_off = tl.arange(0, GP)
        cm = c_off < C
        gm = g_off < G
        d = tl.load(dn_ptr + c_off[:, None] * G + g_off[None, :], mask=cm[:, None] & gm[None, :], other=0.0)
        if HAS_QW:
            qwt = tl.load(qw_ptr + c_off[:, None] * G + g_off[None, :], mask=cm[:, None] & gm[None, :], other=0.0)
        best = tl.zeros((BC,), dtype=tl.float32) + float("inf")
        bk = tl.zeros((BC,), dtype=tl.int32)
        bmir = tl.zeros((BC,), dtype=tl.int32)
        for k in range(0, K):
            f = tl.load(fracs_ptr + k)
            invf = tl.load(invf_ptr + k)
            x = d * invf
            r = tl.inline_asm_elementwise(
                "cvt.rni.f32.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1
            )
            q_std = tl.minimum(tl.maximum(r, -NMAX), NMAX - 1.0)
            e_std = f * q_std - d
            q_mir = tl.minimum(tl.maximum(r, -(NMAX - 1.0)), NMAX)
            e_mir = f * q_mir - d
            if HAS_QW:
                se = e_std * e_std * qwt
                sm = e_mir * e_mir * qwt
            else:
                se = e_std * e_std
                sm = e_mir * e_mir
            if NOMASK:
                l_std = tl.sum(se, axis=1)
                l_mir = tl.sum(sm, axis=1)
            else:
                l_std = tl.sum(tl.where(gm[None, :], se, 0.0), axis=1)
                l_mir = tl.sum(tl.where(gm[None, :], sm, 0.0), axis=1)
            mir = l_mir < l_std
            loss = tl.where(mir, l_mir, l_std)
            better = loss < best
            best = tl.where(better, loss, best)
            bk = tl.where(better, k, bk)
            bmir = tl.where(better, mir.to(tl.int32), bmir)
        tl.store(loss_ptr + c_off, best, mask=cm)
        tl.store(idx_ptr + c_off, bk, mask=cm)
        tl.store(mirror_ptr + c_off, bmir, mask=cm)


def sym_search_shared_triton(dn, qw, fracs, invf, nmax):
    """Coarse-stage symmetric search in normalized space on CUDA (see module docstring)."""
    assert triton is not None, f"triton is not importable: {_TRITON_IMPORT_ERROR}"
    assert dn.is_cuda and dn.dtype == torch.float32 and dn.is_contiguous()
    assert fracs.is_contiguous() and fracs.dtype == torch.float32 and fracs.ndim == 1
    assert invf.is_contiguous() and invf.dtype == torch.float32 and invf.shape == fracs.shape
    if qw is not None:
        assert qw.dtype == torch.float32 and qw.is_contiguous() and qw.shape == dn.shape
    _dev = dn.device
    assert fracs.device == _dev, f"sym_shared: fracs on {fracs.device} but dn on {_dev}"
    assert invf.device == _dev, f"sym_shared: invf on {invf.device} but dn on {_dev}"
    if qw is not None:
        assert qw.device == _dev, f"sym_shared: qw on {qw.device} but dn on {_dev}"

    c, g = dn.shape
    k = fracs.numel()
    gp = _next_pow2(g)
    bc = max(1, min(64, 4096 // gp))
    loss = torch.empty((c,), device=_dev, dtype=torch.float32)
    idx = torch.empty((c,), device=_dev, dtype=torch.int32)
    mirror = torch.empty((c,), device=_dev, dtype=torch.int32)
    with torch.cuda.device(_dev):  # Triton launches on the CURRENT device
        _sym_shared_kernel[(triton.cdiv(c, bc),)](
            dn,
            qw if qw is not None else dn,  # dummy pointer when unweighted
            fracs,
            invf,
            loss,
            idx,
            mirror,
            c,
            g,
            k,
            NMAX=nmax,
            HAS_QW=qw is not None,
            BC=bc,
            GP=gp,
            NOMASK=gp == g,
            num_warps=4,
        )
    return loss, idx.long(), mirror.bool()


def _launch_sweep(data, qw, scales, loss, zp, c, g, k, nz, maxq, bc, gp):
    _neuqi_sweep_kernel[(triton.cdiv(c, bc),)](
        data,
        qw if qw is not None else data,  # dummy pointer when unweighted (dead lanes)
        scales,
        loss,
        zp,
        c,
        g,
        k,
        nz,
        MAXQ=maxq,
        HAS_QW=qw is not None,
        BC=bc,
        GP=gp,
        NOMASK=gp == g,
        ZU=maxq + 1 <= 32,
        num_warps=4,
    )
    return loss, zp
