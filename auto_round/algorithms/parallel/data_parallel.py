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
"""Single-process data parallelism for the SignRound tuning loop.

Runs one block's iteration loop on the home GPU plus W-1 mirror replicas
(persistent deep copies of the wrapped block). Per iteration the global
calibration batch is sharded across replicas; each replica computes its
forward/loss/backward on its own device; gradients are exchanged with a
halving-doubling all-reduce over flat fp32 buffers so every replica ends
with the same averaged gradient; the deterministic SignSGD step is then
applied locally on every replica (identical inputs -> identical update),
which keeps mirrors in sync without any parameter broadcast.

World size 1 (default) executes none of this code.
"""

from __future__ import annotations

import copy
import queue
import threading
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch

from auto_round.logger import logger


@dataclass
class DDPPlan:
    """Resolved data-parallel plan for one block."""

    world: int
    devices: List[torch.device]
    shard_size: int
    notes: List[str] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return self.world > 1


def block_has_tuning_entries(block) -> bool:
    """True if any module in the block tree carries tuning parameters at all
    (round or min-max). All-float pinned blocks (a ``'bits': 16,
    'data_type': 'float'`` layer_config pin) return False; callers use this to
    decline the parallel lane before mirror setup because the serial path's
    empty-params guard (quantizer) would no-op the block anyway.
    """
    for _, mod in block.named_modules():
        params = getattr(mod, "params", None)
        if isinstance(params, dict) and params:
            return True
    return False


def _parse_device_token(tok: str, dev_type: str = "cuda") -> torch.device:
    """Parse an explicit-device token: ``3`` -> ``<dev_type>:3``, ``cuda:3``/``xpu:3`` as-is."""
    tok = tok.strip()
    if tok.isdigit():
        return torch.device(dev_type, int(tok))
    return torch.device(tok)


_SUPPORTED_ACCEL_TYPES = ("cuda", "xpu", "hpu")


def _accel_device_count(dev_type: str) -> int:
    """Device count for an accelerator backend, 0 when the backend is absent.

    ``getattr(torch, ...)`` can raise during backend module init on builds
    compiled without the backend, so the whole probe is guarded.
    """
    try:
        mod = getattr(torch, dev_type, None)
        fn = getattr(mod, "device_count", None)
        if callable(fn):
            return int(fn())
    except Exception:  # pragma: no cover - backend-specific probe failure
        pass
    return 0


def _accel_free_bytes_map(dev_type: str):
    """Per-device free-memory map for the backend, or None when unavailable.

    cuda/xpu expose ``mem_get_info``; other backends (and missing APIs) return
    None so the plan resolves without VRAM filtering (explicit devices then
    decide the fleet).
    """
    try:
        mod = getattr(torch, dev_type, None)
        if mod is not None and getattr(mod, "is_available", lambda: False)():
            mem = getattr(mod, "mem_get_info", None)
            if callable(mem):
                return {torch.device(dev_type, i): mem(i)[0] for i in range(_accel_device_count(dev_type))}
    except Exception as e:  # pragma: no cover - backend-specific probe failure
        logger.warning(
            "[tune-ddp] free-memory probe failed for %s (%s); resolving the plan without VRAM filtering",
            dev_type,
            e,
        )
    return None


@dataclass(frozen=True)
class ParallelPolicy:
    """Resolved ``--parallel_quantization`` policy, set once at the entry.

    ``world`` is the requested DDP world (1 = off/serial); ``source`` records
    where it came from ("off" | "auto" | "explicit"); ``collect_forward_cap``
    mirrors AR_TUNE_DDP_MAX_COLLECT_FORWARD_DEVICES (0 = uncapped). The policy
    is set on the CompressContext once at the entry; the engine reads it from
    there.
    """

    world: int = 1
    source: str = "off"
    collect_forward_cap: int = 0

    @property
    def enabled(self) -> bool:
        return self.world > 1


def _quantizer_policy(quantizer) -> ParallelPolicy:
    """The run's ParallelPolicy as visible to a quantizer (default: off)."""
    ctx = getattr(quantizer, "compress_context", None)
    policy = getattr(ctx, "parallel_policy", None)
    return policy if isinstance(policy, ParallelPolicy) else ParallelPolicy()


class ParallelTuneState:
    """Engine-owned parallel state for one quantizer run.

    The quantizer hosts it as ``parallel_state`` (a mount point only -- every
    field is engine state): the cached engagement plan, the block-scoped
    collection mirror pool handed over by the composer, and the flag that a
    lane consumed that pool.
    """

    __slots__ = ("plan", "pool", "pool_used")

    def __init__(self) -> None:
        self.plan: Optional["DDPPlan"] = None
        self.pool: Optional["MirrorPool"] = None
        self.pool_used: bool = False


def parallel_state(quantizer, create: bool = False) -> Optional[ParallelTuneState]:
    """The quantizer's engine state holder (created on demand)."""
    st = getattr(quantizer, "parallel_state", None)
    if st is None and create:
        st = ParallelTuneState()
        quantizer.parallel_state = st
    return st


def resolve_parallel_world(parallel: str, n_visible: int, iters: int = 0) -> int:
    """Resolve the ``--parallel_quantization`` argument into a DDP world size.

    Policy entry point (called by the CLI): ``off`` -> 1, ``auto`` -> all
    visible devices, ``N`` -> N. The power-of-two rule applies only when the
    tune loop runs (``iters > 0``): the gradient exchange is a recursive
    halving-doubling reduction over the replicas and needs a power-of-two
    rank count. At ``iters == 0`` the loop is empty, every phase that runs
    (sharded collection, searches) is world-count-agnostic, so any device
    count works and ``auto`` takes all visible devices. A requested world is
    a requirement: an invalid value, or more devices requested than are
    visible, raises instead of silently shrinking the request.
    """
    parallel = str(parallel or "off").strip().lower()
    if parallel == "off":
        return 1
    if n_visible < 2:
        raise RuntimeError("--parallel_quantization auto needs at least 2 visible CUDA devices")
    if parallel == "auto":
        if iters > 0:
            return 1 << (max(n_visible, 1).bit_length() - 1)  # largest power of two <= visible
        return n_visible  # no exchange at iters=0: every visible device works
    if parallel.isdigit():
        world = int(parallel)
        if world < 2:
            raise RuntimeError(f"--parallel_quantization world must be at least 2, got {world}")
        if world > n_visible:
            raise RuntimeError(
                f"--parallel_quantization {world} requires at least {world} visible CUDA devices, got {n_visible}"
            )
        if world & (world - 1) and iters > 0:
            raise RuntimeError(
                f"--parallel_quantization world must be a power of two when iters > 0 (the gradient "
                f"exchange is a halving-doubling reduction), got {world}"
            )
        return world
    raise RuntimeError(f"--parallel_quantization accepts off|auto|N, got {parallel!r}")


def resolve_ddp_plan(
    world: int,
    home: torch.device,
    batch_size: int,
    visible_devices: Optional[Sequence[int]] = None,
    vram_free_bytes: Optional[int] = None,
    mirror_footprint_bytes: Optional[int] = None,
    margin_bytes: int = 2 * 1024**3,
    pow2_floor: bool = True,
) -> DDPPlan:
    """Pick mirror devices for ``world``-way data parallelism of one block.

    Rules (each adjustment is recorded in ``notes``):
    - world <= 1 or unsupported home -> disabled
    - batch smaller than the world -> demote to the largest world that fits
      the batch: with ``pow2_floor`` (the iters>0 contract) the largest power
      of two that fits; with it off (iters=0, no gradient exchange) the exact
      batch size; a batch of 1 runs serial (logged; a tiny batch is a
      legitimate configuration)
    - home + next devices in ascending visible order
    - visible devices fewer than the requested world -> reported (the
      engagement layer raises)
    - per-mirror VRAM check: advisory only -- a WARNING names devices whose
      free memory is below the estimated mirror footprint; the requested
      world stays intact (a wrong estimate failing the run loudly is
      preferable to silently shrinking it)
    """
    notes: List[str] = []
    if world is None or world <= 1:
        return DDPPlan(1, [home], batch_size, notes)
    if home.type not in _SUPPORTED_ACCEL_TYPES:
        notes.append(f"home device {home} is not a supported accelerator")
        return DDPPlan(1, [home], batch_size, notes)
    world = int(world)
    if batch_size < world:
        # sum-reduced losses make uneven shards exact, so the batch may
        # split unevenly -- with a live tune loop the exchange world itself
        # must stay a power of two (pow2_floor); at iters=0 there is no
        # exchange and the demotion is exact. A replica needs at least one
        # sample; below two usable replicas the lane runs serial (logged; a
        # tiny batch is a legitimate configuration)
        if pow2_floor:
            _w = world
            while _w > 1 and batch_size < _w:
                _w //= 2
        else:
            _w = min(world, max(batch_size, 1))
        if _w < 2:
            logger.info("[tune-ddp] batch %d smaller than any usable world; running serial", batch_size)
            return DDPPlan(1, [home], batch_size, notes)
        if _w < world:
            notes.append(f"batch {batch_size} smaller than world {world}; world set to {_w}")
            logger.info("[tune-ddp] %s", notes[-1])
        world = _w

    if visible_devices:
        order = [torch.device(home.type, i) for i in sorted(visible_devices)]
    else:
        order = [torch.device(home.type, i) for i in range(_accel_device_count(home.type))]

    rotated = [home] + [d for d in order if d != home]
    if len(rotated) < world:
        # the requested world is a requirement: fewer visible devices than
        # requested is reported to the caller (the engagement layer raises);
        # the world stays as asked
        notes.append(f"visible {home.type} devices ({len(rotated)}) fewer than the requested world {world}")
        return DDPPlan(1, [home], batch_size, notes)
    devices = rotated[:world]
    if vram_free_bytes is not None and mirror_footprint_bytes is not None:
        for dev in devices:
            if dev == home:
                continue  # the home already holds the block; this check prices mirrors
            free = vram_free_bytes.get(dev, 0) if hasattr(vram_free_bytes, "get") else vram_free_bytes
            if free < mirror_footprint_bytes + margin_bytes:
                notes.append(
                    f"low free VRAM on {dev}: free {free / 2**30:.1f}GiB < estimated mirror footprint "
                    f"{(mirror_footprint_bytes + margin_bytes) / 2**30:.1f}GiB (estimate only; device kept)"
                )
                logger.warning("[tune-ddp] %s", notes[-1])
    return DDPPlan(world, devices, batch_size // world, notes)


def _move_tensor(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a tensor to ``device`` (seam for tests on single-device hosts)."""
    return t.to(device)


def resolve_tune_ddp_plan_(quantizer, block, fp_inputs, fp_outputs, home, world=None, log: bool = True):
    """Resolve the engaged DDP plan for one block -- shared by quantizer + composer.

    Single source of truth for engagement so the collection pass (composer)
    and the tune (quantizer) resolve identically: both check the same eligibility
    (policy world, accelerator home, no grad scaler, no LFQ, list pools, not
    on the torchrun lane), price the same mirror
    footprint, apply the same device selection and the same
    power-of-two gate. The resolved plan is cached on the quantizer's parallel
    state (engine-owned);
    the tune-side caller re-resolves post-wrap (exact wrapper pricing, fresh
    VRAM), so pass ``log=False`` from the pre-wrap collection call to keep the
    engagement/decline lines one-per-block. ``fp_outputs`` may be None when the
    reference outputs are not collected yet (composer entry) -- the tune-side
    caller re-checks the outputs are a list before engaging.

    Returns the DDPPlan (world=1 => not engaged).
    """
    from auto_round.utils.distributed import is_distributed

    _st = parallel_state(quantizer)
    cached = _st.plan if _st is not None else None
    if cached is not None:
        return cached
    if world is None:
        world = _quantizer_policy(quantizer).world
    # the pow2 rule is iters-conditional: the exchange collectives need a
    # power-of-two rank count, and they run only when the tune loop does
    _iters = int(getattr(quantizer, "iters", 0) or 0)
    home = torch.device(home) if not isinstance(home, torch.device) else home
    if home.type in _SUPPORTED_ACCEL_TYPES and home.index is None:
        try:
            _mod = getattr(torch, home.type, None)
            _cur = getattr(_mod, "current_device", None)
            home = torch.device(home.type, int(_cur()) if callable(_cur) else 0)
        except Exception:  # backend module init can assert on builds without it
            home = torch.device(home.type, 0)
    decline = []
    # NOTE: engagement itself is not iters-gated -- the DDP world shards
    # the sharded no-grad collection passes at iters=0 exactly as at
    # iters>0 (the tune loop has zero iterations there); only the pow2
    # rule is iters-conditional (see _iters below)
    if world > 1 and home.type not in _SUPPORTED_ACCEL_TYPES:
        decline.append(f"home device {home} is not a supported accelerator ({'/'.join(_SUPPORTED_ACCEL_TYPES)})")
    # eligibility constraints are DECLARED by the quantizer
    # (BaseQuantizer.parallel_constraints); the resolver consumes the
    # declaration
    _constraints_fn = getattr(quantizer, "parallel_constraints", None)
    _constraints = _constraints_fn() if callable(_constraints_fn) else {}
    _scaler_active = bool(_constraints.get("scaler_active"))
    _lfq_active = bool(_constraints.get("enable_lfq"))
    _home_span_devs = {p.device for p in block.parameters() if p.device.type == home.type}
    eligible = (
        world > 1
        and home.type in _SUPPORTED_ACCEL_TYPES
        and not _scaler_active
        and not _lfq_active
        and isinstance(fp_inputs, list)
        and (fp_outputs is None or isinstance(fp_outputs, list))
        and not is_distributed()
        and len(_home_span_devs) <= 1
    )
    if world > 1 and _scaler_active:
        decline.append("a grad scaler is active")
    if world > 1 and _lfq_active:
        decline.append("enable_lfq")
    if world > 1 and not isinstance(fp_inputs, list):
        decline.append("non-list calibration inputs (diffusion-style pools)")
    if world > 1 and isinstance(fp_inputs, list) and fp_outputs is not None and not isinstance(fp_outputs, list):
        decline.append("non-list reference outputs (diffusion-style pools)")
    # Parallel tuning tunes whole-block mirrors: every replica, including the
    # home copy, must sit on ONE device for the tune. A block whose weights
    # span several devices of the home's accelerator type (multi-device auto
    # placement) is a different topology (pipeline-style stages), so fail
    # visibly: the placement itself must be single-device when parallel
    # tuning runs. CPU-resident tensors are legal alongside the
    # home device (pinned subtrees such as ngram embedding tables stay on CPU
    # by design).
    if world > 1 and len(_home_span_devs) > 1:
        decline.append(
            f"block spans {len(_home_span_devs)} {home.type.upper()} devices; parallel tuning requires "
            "single-device block placement (run with no --device_map or a single-device "
            "--device_map; multi-device/auto maps may shard a block across GPUs -- or run "
            "with --parallel_quantization off)"
        )
    if world > 1 and is_distributed():
        decline.append("multi-process torchrun lane is active (--parallel_quantization is single-process)")

    plan = DDPPlan(1, [home], len(fp_inputs) if isinstance(fp_inputs, list) else 0)
    if eligible:
        nsamples = len(fp_inputs)
        batch_size = getattr(getattr(quantizer, "calibration_context", None), "batch_size", nsamples)
        global_batch_size = min(nsamples, batch_size * getattr(quantizer, "gradient_accumulate_steps", 1))
        mirror_bytes = sum(pp.numel() * pp.element_size() for pp in block.parameters()) + sum(
            pp.numel() * pp.element_size()
            for _m in block.modules()
            if hasattr(_m, "orig_layer")
            for pp in _m.params.values()
        )
        # charge the per-forward activation working set on top of the mirror:
        # the existing estimator walks the block's real shapes at the
        # per-forward batch (the serial micro-batch size -- replicas chunk
        # their shards to it), so an activation-heavy block cannot silently
        # pass a weights-only budget
        try:
            from auto_round.utils.device import estimate_tuning_block_mem

            _per_fwd = min(int(batch_size), max(1, global_batch_size // max(1, world)))
            _layer_mem, _act_mem, _io_mem, _add_mem = estimate_tuning_block_mem(block, fp_inputs, _per_fwd)
            # ``_act_mem`` already sums the per-layer ``output_memory``
            # (grad-doubled) with the MoE ratio applied -- the dict totals
            # are already included
            _act_bytes = int((_act_mem + _io_mem + _add_mem) * 2**30)
            mirror_bytes += _act_bytes
        except Exception as e:  # pragma: no cover - estimator is best-effort
            logger.info("[tune-ddp] activation pricing skipped (estimator failed: %s)", e)
        free = _accel_free_bytes_map(home.type)
        _n_devs = _accel_device_count(home.type)
        plan = resolve_ddp_plan(
            world,
            home,
            global_batch_size,
            visible_devices=list(range(_n_devs)) if _n_devs else None,
            vram_free_bytes=free,
            mirror_footprint_bytes=mirror_bytes,
            pow2_floor=_iters > 0,
        )
        if log:
            global _ENGAGED_LOGGED_SIG
            sig = (plan.world, tuple(str(d) for d in plan.devices), plan.shard_size)
            _changed = sig != _ENGAGED_LOGGED_SIG
        else:
            _changed = False
        if plan.enabled and _iters > 0 and plan.world & (plan.world - 1) != 0:
            decline.append(
                f"resolved world {plan.world} is not a power of two (the gradient exchange needs "
                "one when the tune loop runs)"
            )
            plan = DDPPlan(1, [home], nsamples)
        elif plan.enabled and log and _changed:
            logger.info(
                "[tune-ddp] engaged: world=%d, batch_size per device=%d, devices=%s",
                plan.world,
                plan.shard_size,
                [str(d) for d in plan.devices],
            )
            _ENGAGED_LOGGED_SIG = sig
        if plan.enabled and plan.world > 1:
            # parallel lanes multiply the static shape specializations of the
            # shared compiled quant functions (per-layer v shapes x shard /
            # micro-batch chunk shapes across the replicas' searches), so the
            # dynamo cache limit must scale with the engaged world or the lane
            # silently degrades to eager mid-block once the limit is hit.
            # Monotone raise-only, so the
            # per-block resolve is idempotent and a serial run keeps the
            # plain env default.
            try:
                from auto_round import envs as _cache_envs
                from auto_round.utils.device import _bump_dynamo_cache_limit

                _bump_dynamo_cache_limit(int(_cache_envs.AR_DYNAMO_CACHE_SIZE_LIMIT) * plan.world)
            except Exception:  # pragma: no cover - best effort; the tune continues
                pass
    if world > 1 and not plan.enabled:
        reasons = decline + [n for n in plan.notes if n]
        if reasons:
            # A requested parallel world is a hard requirement: continuing
            # on the serial path would silently invalidate the requested
            # configuration and any measurement against it. `auto`
            # resolves to the largest power of two fitting the visible devices
            # at entry and the VRAM check is advisory, so reaching this raise
            # means parallel tuning is genuinely ineligible for this run
            # (an engagement gate or fewer visible devices than the world).
            raise RuntimeError(
                f"parallel tuning with world={world} is ineligible: "
                f"{'; '.join(reasons)}. Fix the blocking condition listed above, or run "
                "with --parallel_quantization off to tune serial deliberately."
            )
    parallel_state(quantizer, create=True).plan = plan
    return plan


def _relocate_params(module: torch.nn.Module, device: torch.device) -> None:
    """Move ALL state that ``nn.Module.to()`` cannot see onto ``device``.

    Wrapper modules keep three kinds of non-registered state that a mirrored
    block would otherwise leave on the home device:
    - tunable tensors in the plain ``params`` dict (``v``/``min_scale``/...)
      -- recreated as fresh leaf Parameters (grad state resets, correct for a
      fresh mirror);
    - plain tensor attributes (``weight_min``/``weight_max`` anchors, cached
      imatrix slices, ...) -- moved in place;
    - device-typed attributes (``device``/``output_device``/``tuning_device``)
      -- repointed so wrapper forward staging targets the mirror.
    """
    for _n, m in module.named_modules():
        params = getattr(m, "params", None)
        if isinstance(params, dict):
            for key, val in params.items():
                if isinstance(val, torch.nn.Parameter):
                    # move IN PLACE: the dict entry and the registered
                    # _parameters entry are the SAME object by construction
                    # (wrapper _init_params). Replacing the object here broke
                    # that aliasing -- optimizers collected the dict's dead
                    # clone while the forward/backward used the registered
                    # Parameter, so mirror grads stayed None, sync_grads
                    # silently skipped the allreduce, and the home stepped on
                    # its own shard's gradient only (quality regressed,
                    # monotonically in world size).
                    val.data = _move_tensor(val.detach(), device)
        for key, val in list(m.__dict__.items()):
            if isinstance(val, torch.Tensor) and val.device != device and val.device.type != "meta":
                m.__dict__[key] = _move_tensor(val, device)
            elif isinstance(val, torch.device):
                m.__dict__[key] = device
            elif (
                key in ("tuning_device", "device", "output_device")
                and isinstance(val, str)
                and _parse_device_token(val).type == device.type
            ):
                # wrapper staging targets (plain strings -- invisible to the
                # torch.device branch above); same accelerator family only
                m.__dict__[key] = str(device)


def _param_grad_buffers(params_by_device: List[List[torch.nn.Parameter]]) -> List[Optional[torch.Tensor]]:
    """Flatten each replica's gradients into one contiguous fp32 buffer."""
    bufs: List[Optional[torch.Tensor]] = []
    for params in params_by_device:
        parts = [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
        if not parts:
            bufs.append(None)
            continue
        buf = torch.cat(parts) if len(parts) > 1 else parts[0].clone()
        bufs.append(buf.to(torch.float32) if buf.dtype != torch.float32 else buf)
    return bufs


def _write_back_grads(buf: Optional[torch.Tensor], params: List[torch.nn.Parameter]) -> None:
    """Scatter an averaged flat buffer back into ``param.grad`` tensors."""
    if buf is None:
        return
    offset = 0
    for p in params:
        if p.grad is None:
            continue
        n = p.grad.numel()
        p.grad.copy_(buf[offset : offset + n].view_as(p.grad))
        offset += n


def _transport_segment(seg: torch.Tensor, dev: torch.device) -> torch.Tensor:
    """Move a peer's segment onto ``dev`` (fp32 wire, lossless).

    The wire hop uses non_blocking=True: the allreduce cost is
    payload-independent across dtypes (host-blocking-bound), and cross-device
    copy_ is stream-ordered on both endpoints, so dropping the host block
    lets the exchange chain execute back-to-back on the GPUs without races.
    """
    return seg.to(dev, non_blocking=True)


def halving_doubling_allreduce(buffers: List[torch.Tensor], scale: float = 1.0) -> None:
    """In-place all-reduce across device-resident buffers (single process).

    Chunked recursive halving-doubling: the flat space is split into W
    chunks; the reduce-scatter phase leaves rank r owning reduced chunk r
    (each step exchanges half of the working block with the partner rank);
    the all-gather phase re-exchanges owned chunks until every rank holds
    the fully reduced buffer. Per-rank traffic is 2*(W-1)/W*bytes, same as a
    ring. Cross-device ``to()`` copies use P2P when available. ``scale`` is
    applied at the end (pass ``1/world`` to average). Requires a
    power-of-two world (the resolver guarantees it).
    """
    world = len(buffers)
    if world < 2:
        if buffers:
            buffers[0].mul_(scale)
        return
    if world & (world - 1):
        raise ValueError(f"halving_doubling_allreduce needs a power-of-two world, got {world}")

    numel = buffers[0].numel()
    chunk = (numel + world - 1) // world

    _reduce_scatter_halving(buffers)
    _allgather_doubling(buffers)

    for buf in buffers:
        buf.mul_(scale)


def _reduce_scatter_halving(buffers: List[torch.Tensor]) -> None:
    """Recursive-halving reduce-scatter across device-resident buffers.

    The flat space is split into W chunks; the working block per rank halves
    each step (each rank keeps one half, adding the partner's copy of
    it). Rank r ends up owning reduced chunk r; the other
    chunks hold partial garbage. Accumulation stays fp32 on the receiver.
    Requires a power-of-two world.
    """
    world = len(buffers)
    numel = buffers[0].numel()
    chunk = (numel + world - 1) // world
    length = world  # chunks in each rank's working block
    while length > 1:
        half = length // 2
        for rank in range(world):
            base = rank - (rank % length)  # aligned working-block start
            mid, hi = base + half, base + length
            if rank % length < half:
                partner = rank + half  # keep [base, mid)
                seg = buffers[partner][chunk * base : chunk * mid]
                seg = _transport_segment(seg, buffers[rank].device)
                buffers[rank][chunk * base : chunk * mid].add_(seg)
            else:
                partner = rank - half  # keep [mid, hi)
                seg = buffers[partner][chunk * mid : chunk * hi]
                seg = _transport_segment(seg, buffers[rank].device)
                buffers[rank][chunk * mid : chunk * hi].add_(seg)
        length = half


def _allgather_doubling(buffers: List[torch.Tensor]) -> None:
    """Recursive-doubling all-gather of per-rank owned chunks.

    Mirrors the reduce-scatter block structure: the working block per rank
    doubles each step; ranks exchange the chunks the partner owns (copies,
    no adds). Every rank ends holding every chunk -- bitwise identical
    (same-device fp32 copies for gradient buffers, plain int8 copies for
    the sign buffers).
    """
    world = len(buffers)
    numel = buffers[0].numel()
    chunk = (numel + world - 1) // world
    length = 2
    while length <= world:
        half = length // 2
        for rank in range(world):
            base = rank - (rank % length)
            mid, hi = base + half, base + length
            if rank % length < half:
                partner = rank + half
                src = buffers[partner][chunk * mid : chunk * hi]
                dst = buffers[rank][chunk * mid : chunk * hi]
                dst.copy_(_transport_segment(src, dst.device))
            else:
                partner = rank - half
                src = buffers[partner][chunk * base : chunk * mid]
                dst = buffers[rank][chunk * base : chunk * mid]
                dst.copy_(_transport_segment(src, dst.device))
        length *= 2


_SIGN_LOGGED = False
_FULLVALUE_LOGGED = False
_ENGAGED_LOGGED_SIG = None  # module-level init for the global read in resolve_tune_ddp_plan_


def sign_exchange_allreduce(buffers: List[torch.Tensor]) -> None:
    """In-place sign all-reduce for SignSGD gradients (single process).

    The SignRound optimizer consumes ONLY torch.sign(grad) (weight_decay is
    always 0), so the averaged gradient's magnitude never reaches the
    update. This exchanges exactly what the step needs: a recursive-halving
    reduce-scatter leaves rank r owning the reduced chunk r; each rank
    then computes torch.sign() ONCE on its exact fp32 chunk and the signs
    are all-gathered as int8 -- a lossless wire format 4x smaller than
    fp32. The signs every rank applies are bitwise identical and exactly
    faithful to the true fp32 mean.

    Valid only when the optimizer is pure sign-SGD: a momentum buffer or
    weight decay would mix magnitudes back into the update, so callers must
    gate on momentum == 0 (weight_decay is hard-wired to 0 in the tuner).
    """
    world = len(buffers)
    if world < 2:
        for buf in buffers:
            buf.sign_()
        return
    if world & (world - 1):
        raise ValueError(f"sign_exchange_allreduce needs a power-of-two world, got {world}")

    _reduce_scatter_halving(buffers)

    numel = buffers[0].numel()
    chunk = (numel + world - 1) // world
    sign_bufs: List[torch.Tensor] = []
    for rank, buf in enumerate(buffers):
        signs = torch.empty(numel, dtype=torch.int8, device=buf.device)
        lo, hi = chunk * rank, min(numel, chunk * (rank + 1))
        signs[lo:hi] = torch.sign(buf[lo:hi]).to(torch.int8)
        sign_bufs.append(signs)

    # int8 payload over plain device copies, lossless
    _allgather_doubling(sign_bufs)

    for buf, signs in zip(buffers, sign_bufs):
        buf.copy_(signs)  # int8 -> fp32: -1.0 / 0.0 / 1.0


# hook-written statistics that are safe to merge across mirror copies:
# associative reductions reproduce the serial totals up to fp32 summation order
_MERGEABLE_STATS = {
    "imatrix": "sum",  # module.imatrix += sum(x^2) per shard
    "imatrix_cnt": "sum",  # scalar row count; MUST co-travel with imatrix or the
    # normalize step (imatrix /= imatrix_cnt) hits a module that received the
    # merged tensor while its own shard collected zero routed rows (cold
    # experts under skewed routing) -- AttributeError on imatrix_cnt
    "act_max": "max",  # element-wise running max
}


def _merge_mirror_stats(home: torch.nn.Module, mirrors: List[torch.nn.Module]) -> None:
    """Fold hook-written statistics from mirror copies back into the home.

    Each mirror forwarded a disjoint sample shard; its hooks accumulated
    into mirror-local module attrs which die with the mirror. The stats we
    know how to merge (see ``_MERGEABLE_STATS``) are associative, so the
    merged home totals equal the serial pass's totals up to fp32 summation
    order (~1e-7 relative) -- far below any quantization-relevant scale.
    """
    home_mods = dict(home.named_modules())
    for mirror in mirrors:
        if mirror is None:
            continue
        for name, m_mod in mirror.named_modules():
            h_mod = home_mods.get(name)
            if h_mod is None:
                continue
            for attr, how in _MERGEABLE_STATS.items():
                m_val = getattr(m_mod, attr, None)
                if m_val is None:
                    continue
                h_val = getattr(h_mod, attr, None)
                if not torch.is_tensor(m_val):
                    # scalar stats (imatrix_cnt is a python int): fold with the
                    # same copy-or-sum semantics so they always co-travel with
                    # their tensor sibling
                    if not isinstance(m_val, (int, float)) or isinstance(m_val, bool):
                        continue
                    if h_val is None:
                        setattr(h_mod, attr, m_val)
                    else:
                        setattr(h_mod, attr, h_val + m_val)
                    continue
                if h_val is None:
                    setattr(h_mod, attr, m_val.to(h_mod.weight.device if hasattr(h_mod, "weight") else m_val.device))
                    continue
                m_val = m_val.to(h_val.device, h_val.dtype)
                if how == "sum":
                    setattr(h_mod, attr, h_val + m_val)
                else:  # max
                    setattr(h_mod, attr, torch.max(h_val, m_val))


_pool_move_warned: set = set()


def expect_pool_local(pieces, device, site: str) -> None:
    """Warn (once per site) when pool pieces are NOT on ``device``.

    The distributed-pool contract says DDP shard reads are device-local; the
    defensive ``.to(device)`` at those sites would silently paper over a
    placement bug (or a demoted block paying per-batch copies). This makes
    any actual cross-device engagement visible: count, devices found, site.
    """
    device = torch.device(device)
    stray = [t for t in pieces if t.device != device]
    if stray and site not in _pool_move_warned:
        _pool_move_warned.add(site)
        found = sorted({str(t.device) for t in stray})
        logger.warning(
            "[tune-ddp] %s: %d/%d pool pieces are not on %s (found on %s) -- cross-device copies engaged; "
            "expected zero under the distributed pool",
            site,
            len(stray),
            len(pieces),
            device,
            found,
        )


def distribute_pool(pool: List[torch.Tensor], devices: List[torch.device]) -> None:
    """Scatter a per-sample calibration pool across ``devices`` (in place).

    Device r owns its contiguous ceil/floor slice -- the same boundaries the
    DDP tune shards and the sharded collection use (contiguous_shard_bounds;
    a remainder lands on an earlier replica) -- so shard-local reads
    (ref_r build, _select_batch for shard r, per-replica collections) never
    cross devices. Pieces already on their target device are untouched
    (idempotent; a pool produced by the previous block's sharded collection
    on the same group costs nothing). Pools smaller than the world are left
    alone (serial consumers handle them via move-on-demand cats).
    """
    n = len(pool)
    world = len(devices)
    if world < 2 or n < world:
        return
    bounds = contiguous_shard_bounds(n, world)
    for r, dev in enumerate(devices):
        dev = torch.device(dev)
        for i in range(bounds[r], bounds[r + 1]):
            if pool[i].device != dev:
                pool[i] = pool[i].to(dev)


def contiguous_shard_bounds(n: int, world: int) -> List[int]:
    """Contiguous ceil/floor split bounds over ``n`` items for ``world`` shards.

    The first ``n % world`` shards take one extra item (remainder on the
    EARLIER shards). This exact idiom is a cross-component invariant: pool
    placement, collection shards, tune shards and warm-up windows must all
    slice identically or pools and shards silently misalign -- every site
    must call this helper for the bounds.
    """
    sizes = [n // world + (1 if r < n % world else 0) for r in range(world)]
    bounds = [0]
    for sz in sizes:
        bounds.append(bounds[-1] + sz)
    return bounds


class MirrorPool:
    """Persistent-per-block ephemeral mirrors, reused across collection phases.

    Builds the same per-device replicas ``sharded_nograd_forward`` builds
    per call (home device keeps the original block; every other device gets a
    relocated deepcopy), but ONCE per block: the collection forwards reuse
    the pool, the RTN/OptRTN/NeUQI searches resolve their jobs onto the
    resident mirror weights (zero search traffic) with results written back
    home and synced to every mirror (so the quantized-output cascade may
    reuse them), and the SignRound tune ADOPTS the mirrors (wrap-in-place,
    ownership transfer to the ReplicaGroup -- the group's teardown then owns
    their lifetime). ``release()`` drops the ctx-side references; the home
    block stays canonical throughout.
    """

    def __init__(self, block, devices):
        import copy as _copy

        home_dev = _block_device(block)
        self.block = block
        self.devices = list(devices)
        self.reps: List[torch.nn.Module] = []
        self._owned: List[torch.nn.Module] = []  # deepcopies this pool must free
        for dev in self.devices:
            if dev == home_dev:
                self.reps.append(block)
            else:
                m = _copy.deepcopy(block).to(dev)
                _relocate_params(m, dev)
                self.reps.append(m)
                self._owned.append(m)

    @property
    def world(self) -> int:
        return len(self.reps)

    def mirror_layer(self, rep_idx: int, name: str) -> torch.nn.Module:
        for n, m in self.reps[rep_idx].named_modules():
            if n == name:
                return m
        raise KeyError(name)

    def sync_hooks_from(self, block) -> None:
        """Mirror the home block's CURRENT forward hooks onto every replica.

        A fresh-deepcopy pass inherits the hook registry at copy time; a
        persistent pool must re-sync instead, or it would carry hooks removed
        from the home block (stale stats writes) and miss hooks registered
        after the pool was built (missing shard statistics). Hook callables
        are shared by identity -- the same behavior deepcopy provides.
        """
        home_hooks = {}
        for n, m in block.named_modules():
            fwd = getattr(m, "_forward_hooks", None)
            pre = getattr(m, "_forward_pre_hooks", None)
            if fwd or pre:
                fwd_wk = getattr(m, "_forward_hooks_with_kwargs", None)
                pre_wk = getattr(m, "_forward_pre_hooks_with_kwargs", None)
                home_hooks[n] = (dict(fwd or {}), dict(pre or {}), dict(fwd_wk or {}), dict(pre_wk or {}))
        for rep in self.reps:
            if rep is block:
                continue
            for _n, m in rep.named_modules():
                m._forward_hooks.clear()
                m._forward_pre_hooks.clear()
                if hasattr(m, "_forward_hooks_with_kwargs"):
                    m._forward_hooks_with_kwargs.clear()
                if hasattr(m, "_forward_pre_hooks_with_kwargs"):
                    m._forward_pre_hooks_with_kwargs.clear()
            for n, (fwd, pre, fwd_wk, pre_wk) in home_hooks.items():
                try:
                    m = rep.get_submodule(n)
                except AttributeError:
                    continue
                for k, h in fwd.items():
                    m.register_forward_hook(h, with_kwargs=bool(fwd_wk.get(k, False)))
                for k, h in pre.items():
                    m.register_forward_pre_hook(h, with_kwargs=bool(pre_wk.get(k, False)))

    def release(self) -> None:
        self._owned.clear()
        self.reps = []


def sharded_nograd_forward(
    runner,
    block,
    inputs,
    input_others: dict,
    out_device: torch.device,
    devices: List[torch.device],
    sample_count: Optional[int] = None,
    merge_stats: bool = False,
    max_devices: int = 0,
    pool: Optional[MirrorPool] = None,
):
    """Parallelize a no-grad collection forward across ``devices``.

    ``max_devices > 0`` truncates the shard set to the first K devices
    (shards grow accordingly): forward hooks force dynamo graph breaks,
    leaving the compiled runner as python-bound eager sections that
    GIL-convoy beyond a handful of threads -- hook-carrying passes are
    capped by the caller while hookless passes shard wide.

    The collection passes (reference outputs, quantized-output cascade) are
    plain forwards over the whole sample pool on one GPU while the mirrors
    idle. Here the pool is split into equal disjoint shards; an ephemeral
    copy of the block on each device forwards its shard in a parallel thread;
    per-shard outputs are parked on ``out_device`` and concatenated in order.
    Bit-identical to the serial pass: rows are sample-independent and the
    module copies carry identical weights. Falls back to the serial runner
    call when the pool is not divisible or fewer than two devices are given.
    Mirrors are dropped (freed) afterwards. Returned pieces stay on the
    device that computed them (distributed pool); consumers either read
    shard-locally or use device-safe cats.
    """
    world = len(devices)
    n = sample_count if sample_count is not None else (len(inputs) if isinstance(inputs, list) else 0)
    if max_devices and 0 < max_devices < world:
        logger.info(
            "[tune-ddp] sharded collect: capping concurrent shards at %d of %d device(s) (hook pass)",
            max_devices,
            world,
        )
        devices = list(devices)[:max_devices]
        world = max_devices
    # ceil/floor contiguous split (same tolerance the search seam applies):
    # any sample count shards -- a remainder lands on an earlier replica and
    # the wall is the largest slice; nothing is dropped or run serially
    world = max(1, min(world, n))
    if len(devices) < 2 or world < 2:
        return runner(block, inputs, input_others, cache_device=out_device)
    devices = list(devices)[:world]
    bounds = contiguous_shard_bounds(n, world)
    shards = [list(range(bounds[r], bounds[r + 1])) for r in range(world)]
    # the input pool typically arrives pooled on the primary device; without
    # re-placement every shard pulls its pieces cross-device from that ONE
    # source, which dominates the shard-forward time. Distribute first
    # (idempotent, same layout the tune engagement
    # enforces later); serial consumers afterwards re-gather to home.
    # Tensor pools only: the helper's contract also admits opaque pools.
    if (
        isinstance(inputs, list)
        and inputs
        and all(isinstance(t, torch.Tensor) for t in inputs)
        and (sample_count is None or sample_count == len(inputs))
    ):
        distribute_pool(inputs, devices)
    import time as _time

    _t_mirrors = _time.perf_counter()
    mirrors: List[torch.nn.Module] = []  # the actual deepcopies used this pass
    if pool is not None and pool.world >= len(devices):
        # capped passes REUSE the full-width pool: take the first K replicas
        # and leave the rest idle (a fresh per-pass deepcopy of the block,
        # plus a second full-width rebuild once the pool was withheld, would
        # cost far more). The unused replicas idle;
        # their stats stay stale for this pass and the home fold below only
        # merges the mirrors that actually ran.
        pool.sync_hooks_from(block)
        reps = pool.reps[: len(devices)]
        mirrors = [m for m in reps if m is not block]
        pool_owned = True
    else:
        home_dev = _block_device(block)
        reps: List[torch.nn.Module] = []
        for dev in devices:
            if dev == home_dev:
                reps.append(block)
            else:
                m = copy.deepcopy(block).to(dev)
                _relocate_params(m, dev)
                reps.append(m)
                mirrors.append(m)
        pool_owned = False
    _t_setup_ms = (_time.perf_counter() - _t_mirrors) * 1000
    parts: List = [None] * world
    fwd_walls = [0.0] * world  # per-thread stores at distinct indices: race-free
    evs: List = [None] * world

    def _run(r):
        rep = reps[r]
        dev_r = _block_device(rep)
        t_r = _time.perf_counter()
        ev = None
        if dev_r.type == "cuda":
            ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            ev[0].record()
        # outputs stay ON the replica that computed them: the per-sample list
        # returned below is a distributed pool (device r owns shard r's rows)
        if dev_r.type == "cuda":
            with torch.cuda.device(dev_r):
                parts[r] = runner(rep, inputs, input_others, shards[r], cache_device=dev_r)
        else:
            parts[r] = runner(rep, inputs, input_others, shards[r], cache_device=dev_r)
        if ev is not None:
            ev[1].record()
            evs[r] = ev
        fwd_walls[r] = (_time.perf_counter() - t_r) * 1000

    # spawn path (the ReplicaGroup pool stays out: this bare helper instance
    # has no lifecycle and would leak pool workers -- collection passes are
    # twice per block, so the spawn cost is irrelevant here)
    run_threaded_spawn([lambda r=r: _run(r) for r in range(world)])
    # per-shard GPU times: sync the participating devices (their outputs are
    # consumed right after anyway -- the first downstream read syncs too)
    fwd_gpu = [0.0] * world
    for r, dev in enumerate(devices):
        if evs[r] is not None:
            try:
                torch.cuda.synchronize(dev)
                fwd_gpu[r] = evs[r][0].elapsed_time(evs[r][1])
            except RuntimeError as e:  # a broken pair stays non-fatal to the pass
                logger.warning_once("[perf] sharded-collect GPU wall unavailable for a pair (%s)", e)
                fwd_gpu[r] = float("nan")
    _t_split = _time.perf_counter()
    # Return the SERIAL structure: a per-sample list ([1, S, H] pieces from
    # split_outputs) -- downstream consumers cat per-sample refs along dim 0
    # (tune loss) and iterate the list for the
    # cascade inputs, so a tensor here changes every per-sample shape.
    pieces: List[torch.Tensor] = []
    split_outputs = getattr(runner, "split_outputs", None)
    for part in parts:
        for piece in split_outputs(part) if split_outputs else torch.split(part, 1, dim=0):
            # clone: torch.split/split_outputs return VIEWS into the batched
            # shard base -- a view-holding pool pins the whole [n, S, H] base
            # on its device for as long as any piece lives, and that base is
            # large at tuning batch sizes. Standalone pieces free the base at
            # pass end.
            pieces.append(piece.clone())
    # threaded shard calls each stamped a PARTIAL last_output_dict (their own
    # shard's rows); the serial text path leaves it unset so callers fall back
    # to the returned list -- clear the residue to keep that contract.
    runner.last_output_dict = None
    _t_merge = _time.perf_counter()
    if merge_stats:
        _merge_mirror_stats(block, mirrors)
    if not pool_owned:
        mirrors.clear()  # drop mirror refs; the caching allocator reclaims them
        reps.clear()  # pool-owned reps stay with the pool (later passes reuse them)
    # steady-state breakdown for the collection passes: setup = ephemeral
    # mirror deepcopy/relocate (per pass!), fwd = threaded shard forwards
    # (wall includes enqueue, gpu is the device-side chain incl. the
    # per-thread output parking), split/merge = host assembly. Logged for
    # EVERY pass so first-pass warm-up is separable from steady state.
    _walls = sorted(fwd_walls)
    _gpus = sorted(fwd_gpu)
    logger.debug(
        "[tune-ddp] sharded collect breakdown: world=%d n=%d setup=%.0fms "
        "fwd_wall[min/med/max]=%.0f/%.0f/%.0fms fwd_gpu[min/med/max]=%.0f/%.0f/%.0fms "
        "merge=%.0fms",
        world,
        n,
        _t_setup_ms,
        _walls[0],
        _walls[len(_walls) // 2],
        _walls[-1],
        _gpus[0],
        _gpus[len(_gpus) // 2],
        _gpus[-1],
        (_time.perf_counter() - _t_merge) * 1000,
    )
    return pieces


def pre_wrap_shard_candidate(quantizer) -> bool:
    """Cheap pre-wrap probe: might the data-parallel lane engage for this block?

    Only the policy world and CUDA availability -- the full resolver runs
    post-wrap (it needs wrapper params for mirror pricing). A false positive
    is safe: the serial fallback fills the deferred searches on the home
    device. A false negative only loses sharding.
    """
    try:
        world = _quantizer_policy(quantizer).world
    except (TypeError, ValueError):
        world = 1
    return world >= 2 and torch.cuda.is_available()


def run_deferred_wrap_searches(block, replica_group) -> None:
    """Run deferred wrap-time init-scale searches, sharded across the replicas.

    Mirrors-first pattern: each deferred wrapper's search runs ONCE on exactly
    one replica (round-robin over the plan devices, on that replica's local
    mirror copy), the deterministic result is broadcast to the same-named
    wrappers on every replica, and each replica then finalizes (compiles its
    own quant func on its own device) so tuning starts from identical state.
    ``replica_group=None`` (or a non-parallel plan) runs everything serially on
    the home device -- identical semantics to the plain serial path (which runs
    the searches immediately).
    """
    wrappers = [(n, m) for n, m in block.named_modules() if getattr(m, "_init_search_deferred", False)]
    if not wrappers:
        return
    names = [n for n, _ in wrappers]

    plan = getattr(replica_group, "plan", None)
    world = getattr(plan, "world", 1) if plan is not None else 1
    from auto_round.algorithms.quantization import search_dispatch

    if replica_group is None or world < 2 or len(wrappers) < 2:
        # serial lane: the engine batches same-key staged searches on the home
        # device (singletons and the kill switch take the identical per-module
        # call), then every wrapper finalizes
        staged = [w for _n, w in wrappers if getattr(w, "_deferred_search_inputs", None) is not None]
        unstaged = [w for _n, w in wrappers if getattr(w, "_deferred_search_inputs", None) is None]
        if not search_dispatch.run_batched_wrap_search(staged):
            for w in staged:
                w._run_deferred_search_now()
        for w in unstaged:
            w._run_deferred_search_now()
        for _n, w in wrappers:
            w._finalize_deferred_init()
        return

    devices = list(plan.devices)
    home = devices[0]
    rep_wrappers = []
    for rep in replica_group.replicas:
        rep_wrappers.append({n: m for n, m in rep.named_modules() if hasattr(m, "_run_deferred_search_now")})
    # round-robin the searches: consecutive layers spread across the replicas
    owner = {n: i % world for i, n in enumerate(names)}

    _round_stats: dict = {}

    def _search_on(r):
        def _go():
            import threading as _the
            import time as _stime

            from auto_round.algorithms.quantization.search_dispatch import _mark_worker_eager

            if _the.current_thread() is not _the.main_thread():
                _mark_worker_eager()
            _t_round = _stime.perf_counter()
            dev = devices[r]
            subset = [rep_wrappers[r][n] for n in names if owner[n] == r]
            # the owner's mirror copies already live on the owner's device, so
            # the engine forms exactly one device group and runs it inline:
            # same-key staged searches in one stacked call, singletons per
            # module -- identical results to the per-module searches
            staged = [w for w in subset if getattr(w, "_deferred_search_inputs", None) is not None]
            unstaged = [w for w in subset if getattr(w, "_deferred_search_inputs", None) is None]
            # staging is metadata-only: the engine derives each search's inputs
            # from the owner's own mirror layer at consume time, so the stacked
            # calls run on the owner's device by construction (nothing to move)
            # worker threads skip lazy compilation: swap compiled callables
            # to their eager originals for this round, restore afterwards
            _restore_eager = search_dispatch.swap_wrapper_callables_to_eager(subset)
            try:
                if staged and not search_dispatch.run_batched_wrap_search(staged, log_line=False):
                    unstaged.extend(staged)
                    staged = []
                for w in unstaged:
                    w._run_deferred_search_now()
            finally:
                if _restore_eager is not None:
                    _restore_eager()
            _round_stats[r] = (len(subset), _stime.perf_counter() - _t_round)

        return _go

    # serial compile warm-up: the first round runs on the main thread so the
    # search kernels materialize before the threaded fan-out (dynamo
    # compilation from several worker threads races -- same hazard the tune
    # warm-up avoids; deadlocked the threaded-only variant on the MoE)
    _search_on(0)()
    if world > 1:
        run_threaded_spawn([_search_on(r) for r in range(1, world)])
    # phase boundary: drain the search rounds before pulling results home
    from auto_round.algorithms.parallel.rtn_sharding import _sync_pool_devices as _spb

    _bar = _spb(devices)
    # home collection: pull every searched init_scale to the home device
    results = {}
    for n in names:
        src = rep_wrappers[owner[n]][n].init_scale
        results[n] = src.to(home) if torch.is_tensor(src) else src
    # phase boundary: results are home before the finalize fan-out re-broadcasts
    _bar += _spb(devices)

    if _round_stats:
        # ONE line for the whole adoption: the rounds run in parallel (one
        # thread per replica), so the true wall is the LARGEST round, and
        # per-device detail lives in the engine's per-call line when enabled
        _n_mod = sum(c for c, _t in _round_stats.values())
        _wall = max(_t for _c, _t in _round_stats.values())
        logger.debug(
            "[batched-search] wrap searches: %d modules, %d parallel rounds, %.2fs wall (bar=%.2fs)",
            _n_mod,
            len(_round_stats),
            _wall,
            _bar,
        )

    def _finalize_on(r):
        def _go():
            dev = devices[r]
            if dev.type == "cuda":
                with torch.cuda.device(dev):
                    for n in names:
                        val = results[n].to(dev) if torch.is_tensor(results[n]) else results[n]
                        rep_wrappers[r][n]._finalize_deferred_init(val)
            else:
                for n in names:
                    rep_wrappers[r][n]._finalize_deferred_init(results[n])

        return _go

    run_threaded_spawn([_finalize_on(r) for r in range(world)])


def run_threaded_spawn(fns: Sequence) -> None:
    """Run one callable per item in parallel SPAWNED threads (legacy path).

    Exceptions propagate: the first failure (by thread index) is re-raised
    in the joining thread -- a swallowed worker failure would otherwise
    leave missing shard losses/grads and corrupt the step silently.
    """
    errors: dict = {}

    def _guarded(idx, fn):
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors[idx] = exc

    threads = [threading.Thread(target=_guarded, args=(i, fn)) for i, fn in enumerate(fns)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[min(errors)]


class ReplicaThreadPool:
    """Persistent per-replica workers replacing per-iteration thread spawns.

    The tune loop calls ``run_threaded`` twice per iteration (forward shards
    + mirror optimizer steps); spawning and joining ``world`` fresh threads
    each call costs ~1-2 ms per thread of setup/teardown plus GIL churn --
    a measurable slice of the per-iteration host gap. The pool keeps one
    daemon worker per replica alive for the block's lifetime, fed through
    per-worker queues; each round's completion is signalled by per-worker
    events, preserving the spawn path's semantics exactly: every callable
    runs to completion, and the first failure BY WORKER INDEX is re-raised
    in the calling thread. Workers are daemon threads so a pool abandoned
    by a crash can never hang process exit; ``shutdown()`` joins them in
    the normal flow (ReplicaGroup.teardown).
    """

    def __init__(self, n_workers: int) -> None:
        self.n = n_workers
        self._queues: List["queue.Queue"] = [queue.Queue() for _ in range(n_workers)]
        self._errors: List[Optional[BaseException]] = [None] * n_workers
        self._events: List[threading.Event] = []
        self._threads: List[threading.Thread] = []
        for i in range(n_workers):
            t = threading.Thread(target=self._worker, args=(i,), name=f"tune-ddp-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def _worker(self, idx: int) -> None:
        for fn in iter(self._queues[idx].get, None):  # None = poison pill
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001 - surfaced by run()
                self._errors[idx] = exc
            finally:
                self._events[idx].set()

    def run(self, fns: Sequence) -> None:
        """Execute one round: fns[i] runs on worker i; raise first-by-index error."""
        if len(fns) != self.n:
            raise ValueError(f"wrong count: pool has {self.n} worker(s), got {len(fns)} callable(s)")
        # fresh round state before anything is enqueued (workers are idle
        # between rounds -- run() only returns after every event of the
        # previous round was set)
        self._errors = [None] * self.n
        self._events = [threading.Event() for _ in range(self.n)]
        for i, fn in enumerate(fns):
            self._queues[i].put(fn)
        for ev in self._events:
            ev.wait()
        failed = [i for i, exc in enumerate(self._errors) if exc is not None]
        if failed:
            raise self._errors[min(failed)]

    def shutdown(self) -> None:
        for q in self._queues:
            q.put(None)
        for t in self._threads:
            t.join(timeout=30)


def _enforce_mirror_device_(mirror: torch.nn.Module, dev: torch.device) -> List[str]:
    """Relocate accelerator tensors inside ``mirror`` that sit on another GPU.

    Belt-and-braces after replicate+repair + ``_relocate_params``: tuning
    correctness requires every CUDA/XPU tensor of a mirror to live on the
    mirror's own device -- a leftover broadcast source or a still-shared
    home tensor would make the mirror forward read cross-device (observed
    as a layernorm device-mismatch on replicate-built mirrors). CPU tensors
    are deliberately left alone (pinned tables / self-managed subtrees).
    Returns the names of the tensors that were moved.
    """
    moved: List[str] = []

    def _off_device(t) -> bool:
        return torch.is_tensor(t) and t.device.type in ("cuda", "xpu", "hpu") and t.device != dev

    for name, m in mirror.named_modules():
        for key, p in list(getattr(m, "_parameters", {}).items()):
            if p is not None and _off_device(p):
                m._parameters[key] = torch.nn.Parameter(p.detach().to(dev), requires_grad=p.requires_grad)
                m.__dict__.pop(key, None)
                moved.append(f"{name}.{key}(param:{p.device})")
        for key, b in list(getattr(m, "_buffers", {}).items()):
            if b is not None and _off_device(b):
                m._buffers[key] = b.detach().to(dev)
                m.__dict__.pop(key, None)
                moved.append(f"{name}.{key}(buffer:{b.device})")
        for key, val in list(vars(m).items()):
            if key.startswith("_") or key in ("training", "T_destination"):
                continue  # torch-internal bookkeeping; underscore attrs keep their device
            if _off_device(val):
                setattr(m, key, val.detach().clone().to(dev))
                moved.append(f"{name}.{key}(attr:{val.device})")
        if hasattr(m, "orig_layer") and isinstance(getattr(m, "params", None), dict):
            fixed = {}
            for k, v in m.params.items():
                if _off_device(v):
                    fixed[k] = v.detach().clone().to(dev)
                    moved.append(f"{name}.params[{k}](:{v.device})")
                else:
                    fixed[k] = v
            m.params = fixed
    return moved


class ReplicaGroup:
    """Persistent mirrors of a wrapped block for the iteration loop."""

    def __init__(self, block, plan: DDPPlan) -> None:
        self.plan = plan
        self.home = block
        self.mirrors: List[torch.nn.Module] = []
        for dev in plan.devices[1:]:  # plan.devices[0] is the home by construction
            mirror = self._make_mirror(block, dev)
            _relocate_params(mirror, dev)
            _strays = _enforce_mirror_device_(mirror, dev)
            if _strays:
                logger.warning(
                    "[tune-ddp] mirror device sweep moved %d straggler tensor(s) onto %s: %s%s",
                    len(_strays),
                    dev,
                    ", ".join(_strays[:8]),
                    " ..." if len(_strays) > 8 else "",
                )
            self.mirrors.append(mirror)
        self.replicas = [block] + self.mirrors
        self.world = len(self.replicas)
        # persistent replica worker pool (built lazily on first run_threaded;
        # None until then)
        self._pool = None

    @classmethod
    def adopt(cls, block, plan: "DDPPlan", mirrors: List[torch.nn.Module]) -> "ReplicaGroup":
        """Adopt ALREADY-RESIDENT mirrors (the collection MirrorPool's reps).

        The mirrors were deepcopied + relocated at pool build and wrapped in
        place by the engagement (same wrapper machinery as the home block, so
        replica wrapper state is identical by deterministic construction).
        No second weight copy happens: the pool transfers ownership, its
        ``release()`` only drops list references afterwards.
        """
        group = cls.__new__(cls)
        group.plan = plan
        group.home = block
        group.mirrors = []
        for dev, mirror in zip(plan.devices[1:], mirrors):
            _strays = _enforce_mirror_device_(mirror, dev)
            if _strays:
                logger.warning(
                    "[tune-ddp] adopted mirror device sweep moved %d straggler tensor(s) onto %s: %s%s",
                    len(_strays),
                    dev,
                    ", ".join(_strays[:8]),
                    " ..." if len(_strays) > 8 else "",
                )
            group.mirrors.append(mirror)
        group.replicas = [block] + group.mirrors
        group.world = len(group.replicas)
        group._pool = None
        group._torn_down = False
        return group

    def _make_mirror(self, block, dev):
        """Mirror the block onto ``dev`` with a plain deepcopy.

        deepcopy mirrors are the same construction the sharded collection
        mirrors use; every tensor is re-leafed as an independent Parameter by
        :func:`_relocate_params` and verified device-local by the
        :func:`_enforce_mirror_device_` sweep afterwards.
        """
        return copy.deepcopy(block).to(dev)

    def round_params(self) -> List[List[torch.nn.Parameter]]:
        out = []
        for rep in self.replicas:
            ps = []
            for _n, m in rep.named_modules():
                if hasattr(m, "orig_layer") and "v" in getattr(m, "params", {}):
                    ps.append(m.params["v"])
            out.append(ps)
        return out

    def sync_grads(
        self, params_per_replica: List[List[torch.nn.Parameter]], prof=None, sign_exchange: bool = False
    ) -> None:
        """All-reduce v-gradients so every replica holds the identical average.

        ``sign_exchange=True`` (caller-gated to pure sign-SGD, i.e. no
        momentum) exchanges int8 signs instead of averaged values -- bitwise
        identical across ranks, at least as faithful to the fp32 mean, and
        the all-gather wire shrinks 4x (see sign_exchange_allreduce).
        ``prof`` (optional tune profiler) splits the work into bufprep /
        exchange / writeback stages so the profile line shows where the
        allreduce time actually goes.
        """
        from auto_round.utils.tune_profile import stage as _stage

        with _stage(prof, "bufprep"):
            bufs = _param_grad_buffers(params_per_replica)
        if any(b is None for b in bufs):
            # a replica without gradients means the collected params differ
            # from the ones the forward/backward touched -- the tune would
            # silently degrade to single-shard updates
            logger.warning(
                "[tune-ddp] sync_grads skipped: %d/%d replica(s) have no gradients on their collected params",
                sum(1 for b in bufs if b is None),
                len(bufs),
            )
            return
        with _stage(prof, "exchange"):
            if sign_exchange:
                global _SIGN_LOGGED
                if not _SIGN_LOGGED:
                    _SIGN_LOGGED = True
                    logger.debug(
                        "[tune-ddp] sign-cast exchange engaged: world=%d (int8 sign allgather)",
                        self.world,
                    )
                sign_exchange_allreduce(bufs)
            else:
                global _FULLVALUE_LOGGED
                if not _FULLVALUE_LOGGED:
                    _FULLVALUE_LOGGED = True
                    logger.info(
                        "[tune-ddp] full-value gradient exchange: world=%d (fp32, momentum enabled)", self.world
                    )
                halving_doubling_allreduce(bufs, scale=1.0 / self.world)
        with _stage(prof, "writeback"):
            for buf, params in zip(bufs, params_per_replica):
                _write_back_grads(buf, params)

    def run_threaded(self, fns: Sequence) -> None:
        """Run one callable per replica in parallel threads.

        Uses the persistent worker pool (the tune loop calls this twice per
        iteration, and pool reuse removes the per-call thread spawn/join
        overhead), falling back to ``run_threaded_spawn`` when the pool is
        not yet built or the callable count does not match the pool width.

        Exceptions propagate: the first failure (by thread index) is re-raised
        in the joining thread -- a swallowed worker failure would otherwise
        leave missing shard losses/grads and corrupt the step silently.
        """
        pool = getattr(self, "_pool", None)
        if pool is None:
            self._pool = pool = ReplicaThreadPool(len(fns))
        if len(fns) == pool.n:
            pool.run(fns)
        else:
            run_threaded_spawn(fns)

    def teardown(self) -> None:
        if getattr(self, "_torn_down", False):
            return
        self._torn_down = True
        pool, self._pool = getattr(self, "_pool", None), None
        if pool is not None:
            pool.shutdown()
        self.mirrors = []
        self.replicas = [self.home]


def _block_device(block) -> torch.device:
    p = next(block.parameters(), None)
    return p.device if p is not None else torch.device("cpu")
