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


def _parse_device_token(tok: str) -> torch.device:
    """Parse an explicit-device token: ``3`` -> ``cuda:3``, ``cuda:3``/``xpu:3`` as-is."""
    tok = tok.strip()
    if tok.isdigit():
        return torch.device("cuda", int(tok))
    return torch.device(tok)


def resolve_ddp_plan(
    world: int,
    home: torch.device,
    batch_size: int,
    visible_cuda_devices: Optional[Sequence[int]] = None,
    explicit_devices: Optional[Sequence] = None,
    vram_free_bytes: Optional[int] = None,
    mirror_footprint_bytes: Optional[int] = None,
    margin_bytes: int = 2 * 1024**3,
) -> DDPPlan:
    """Pick mirror devices for ``world``-way data parallelism of one block.

    Rules (each demotion is recorded in ``notes``):
    - world <= 1 or non-CUDA home -> disabled
    - batch_size % world != 0 -> disabled (shard means must reproduce the
      global mean exactly)
    - explicit device list -> use as-is after the home (deduplicated)
    - otherwise home + next devices in ascending visible order
    - per-mirror VRAM guard: skip any device that cannot hold the mirror
      footprint with margin; the world shrinks accordingly
    """
    notes: List[str] = []
    if world is None or world <= 1:
        return DDPPlan(1, [home], batch_size, notes)
    if home.type != "cuda":
        notes.append("home device is not CUDA")
        return DDPPlan(1, [home], batch_size, notes)
    world = int(world)
    if batch_size % world != 0:
        notes.append(f"batch_size {batch_size} not divisible by world {world}")
        return DDPPlan(1, [home], batch_size, notes)

    if explicit_devices:
        order = [_parse_device_token(str(d)) for d in explicit_devices]
    elif visible_cuda_devices:
        order = [torch.device("cuda", i) for i in sorted(visible_cuda_devices)]
    else:
        order = [torch.device("cuda", i) for i in range(torch.cuda.device_count())]

    rotated = [home] + [d for d in order if d != home]
    devices: List[torch.device] = []
    for dev in rotated[:world]:
        if vram_free_bytes is not None and mirror_footprint_bytes is not None:
            free = vram_free_bytes.get(dev, 0) if hasattr(vram_free_bytes, "get") else vram_free_bytes
            if dev != home and free < mirror_footprint_bytes + margin_bytes:
                notes.append(f"skip {dev}: free {free / 2**30:.1f}GiB < footprint+margin")
                continue
        devices.append(dev)

    if len(devices) < 2:
        notes.append("no mirror device passed the VRAM guard")
        return DDPPlan(1, [home], batch_size, notes)
    if len(devices) < world:
        notes.append(f"world reduced {world} -> {len(devices)} by VRAM guard")
    return DDPPlan(len(devices), devices, batch_size // len(devices), notes)


def _move_tensor(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a tensor to ``device`` (seam for tests on single-device hosts)."""
    return t.to(device)


def resolve_tune_ddp_plan_(quantizer, block, fp_inputs, fp_outputs, home, world=None, log: bool = True):
    """Resolve the engaged DDP plan for one block -- shared by quantizer + composer.

    Single source of truth for engagement so the collection pass (composer) and
    the tune (quantizer) can never diverge: both check the same eligibility
    (world env, CUDA home, no grad scaler, gradient_accumulate_steps=1,
    no LFQ, list pools, not on the torchrun lane), price the same mirror
    footprint, apply the same VRAM/explicit-device selection and the same
    power-of-two gate. The resolved plan is cached on the quantizer instance;
    the tune-side caller re-resolves post-wrap (exact wrapper pricing, fresh
    VRAM), so pass ``log=False`` from the pre-wrap collection call to keep the
    engagement/decline lines one-per-block. ``fp_outputs`` may be None when the
    reference outputs are not collected yet (composer entry) -- the tune-side
    caller re-checks the outputs are a list before engaging.

    Returns the DDPPlan (world=1 => not engaged).
    """
    from auto_round import envs as _envs
    from auto_round.utils.distributed import is_distributed

    cached = getattr(quantizer, "_resolved_ddp_plan", None)
    if cached is not None:
        return cached
    if world is None:
        world = int(getattr(_envs, "AR_TUNE_DDP_WORLD", 1) or 1)
    home = torch.device(home) if not isinstance(home, torch.device) else home
    if home.type == "cuda" and home.index is None:
        home = torch.device("cuda", torch.cuda.current_device())
    decline = []
    # NOTE: no iters gate -- the DDP world shards the sharded no-grad
    # collection passes at iters=0 exactly as at iters>0 (the 0-length tune
    # loop simply never runs); this restores the campaign semantics where the
    # env was never iters-gated
    if world > 1 and home.type != "cuda":
        decline.append(f"home device {home} is not CUDA")
    # not every block quantizer family exposes _get_scaler (e.g. RTN /
    # OptimizedRTN); a missing method means "no scaler" for eligibility
    _scaler_fn = getattr(quantizer, "_get_scaler", None)
    _scaler = _scaler_fn() if callable(_scaler_fn) else None
    eligible = (
        world > 1
        and home.type == "cuda"
        and _scaler is None
        and getattr(quantizer, "gradient_accumulate_steps", 1) == 1
        and not getattr(quantizer, "enable_lfq", False)
        and isinstance(fp_inputs, list)
        and (fp_outputs is None or isinstance(fp_outputs, list))
        and not is_distributed()
    )
    if world > 1 and _scaler is not None:
        decline.append("a grad scaler is active")
    if world > 1 and getattr(quantizer, "gradient_accumulate_steps", 1) != 1:
        decline.append("gradient_accumulate_steps != 1")
    if world > 1 and getattr(quantizer, "enable_lfq", False):
        decline.append("enable_lfq")
    if world > 1 and isinstance(fp_inputs, list) and fp_outputs is not None and not isinstance(fp_outputs, list):
        decline.append("non-list reference outputs (diffusion-style pools)")
    if world > 1 and is_distributed():
        decline.append("multi-process torchrun lane is active (AR_TUNE_DDP_WORLD is single-process)")

    plan = DDPPlan(1, [home], len(fp_inputs) if isinstance(fp_inputs, list) else 0)
    if eligible:
        nsamples = len(fp_inputs)
        batch_size = getattr(getattr(quantizer, "calibration_context", None), "batch_size", nsamples)
        global_batch_size = min(nsamples, batch_size * getattr(quantizer, "gradient_accumulate_steps", 1))
        explicit = [d.strip() for d in str(_envs.AR_TUNE_DDP_DEVICES or "").split(",") if d.strip()]
        mirror_bytes = sum(pp.numel() * pp.element_size() for pp in block.parameters()) + sum(
            pp.numel() * pp.element_size()
            for _m in block.modules()
            if hasattr(_m, "orig_layer")
            for pp in _m.params.values()
        )
        free = None
        try:
            free = {
                torch.device("cuda", idx): torch.cuda.mem_get_info(idx)[0] for idx in range(torch.cuda.device_count())
            }
        except Exception as e:  # pragma: no cover - non-CUDA reachability
            logger.warning("[tune-ddp] free-VRAM probe failed (%s); resolving the plan without VRAM filtering", e)
            free = None
        plan = resolve_ddp_plan(
            world,
            home,
            global_batch_size,
            visible_cuda_devices=list(range(torch.cuda.device_count())) if free else None,
            explicit_devices=explicit or None,
            vram_free_bytes=free,
            mirror_footprint_bytes=mirror_bytes,
        )
        if log:
            global _ENGAGED_LOGGED_SIG
            sig = (plan.world, tuple(str(d) for d in plan.devices), plan.shard_size)
            _changed = sig != _ENGAGED_LOGGED_SIG
        else:
            _changed = False
        if plan.enabled and plan.world & (plan.world - 1) != 0:
            decline.append(f"resolved world {plan.world} is not a power of two")
            plan = DDPPlan(1, [home], nsamples)
        elif plan.enabled and log and _changed:
            logger.info(
                "[tune-ddp] engaged: world=%d shard=%d devices=%s",
                plan.world,
                plan.shard_size,
                [str(d) for d in plan.devices],
            )
            _ENGAGED_LOGGED_SIG = sig
    if world > 1 and not plan.enabled:
        reasons = decline + [n for n in plan.notes if n]
        if reasons:
            # A requested parallel world is a requirement, not a preference:
            # continuing on the serial path would silently invalidate the
            # requested configuration and any measurement against it. `auto`
            # still auto-detects -- the VRAM guard reduces the world to the
            # devices that fit (engaging at >=2), so reaching here means
            # parallel tuning is genuinely infeasible for this run.
            raise RuntimeError(
                f"parallel tuning with world={world} is ineligible: "
                f"{'; '.join(reasons)}. Fix the blocking condition (free mirror VRAM / "
                "AR_TUNE_DDP_DEVICES / config), or run with --parallel_quantization off "
                "to tune serial deliberately."
            )
    quantizer._resolved_ddp_plan = plan
    return plan


def gather_block_for_mirroring_(block, home: torch.device) -> bool:
    """Gather a (possibly module-sharded) block whole onto ``home`` for DDP mirroring.

    The data-driven multi-GPU lane shards a block's leaves across the device
    list (``set_auto_device_map_for_block_with_tuning``), so the source block
    may span several devices when the DDP plan resolves. Every replica --
    including the home -- must sit whole on ONE device for the tune forward,
    so the gather runs on the source block BEFORE the mirrors are built;
    mirror fit has already been priced into the plan, so a block that fits a
    mirror fits its home.

    Preserves CPU-pinned / self-managed subtrees (e.g. ngram embedding
    tables) and repoints the per-leaf ``tuning_device`` markers (plain
    strings -- ``_relocate_params`` only sees ``torch.device`` attrs) so
    wrapper input staging targets the gathered device instead of a stale
    shard. Returns True when any state actually moved.
    """
    moved = False
    devs = {p.device for p in block.parameters()}
    if devs and devs != {home}:
        logger.info(
            "[tune-ddp] gathering block onto %s before mirroring (source spanned %d device(s): %s)",
            home,
            len(devs),
            sorted(str(d) for d in devs),
        )
        from auto_round.utils.model import move_to_device_preserving_cpu_pinned

        move_to_device_preserving_cpu_pinned(block, home)
        moved = True
    home_s = str(home)
    for _n, m in block.named_modules():
        td = getattr(m, "tuning_device", None)
        if td is not None and str(td) != home_s:
            m.tuning_device = home
            moved = True
    _relocate_params(block, home)
    return moved


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
      -- repointed so wrapper forward staging targets the mirror, not home.
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


def _xchg(seg: torch.Tensor, dev: torch.device, dtype: torch.dtype, transport: str) -> torch.Tensor:
    """Move a peer's segment onto ``dev`` in the requested transport dtype.

    Transport ORDER matters: a combined ``.to(device, dtype)`` cross-device
    copy casts on the SOURCE first and then memcpys -- with an fp32
    destination the wire carried full fp32 bytes and the bf16 transport was
    a no-op on payload (measured: bf16 allreduce time == fp32 on a
    half-duplex-per-link fabric). Cast down on the source, move the reduced
    bytes, cast back up on the receiver instead.

    int8 uses symmetric per-segment scaling: one amax per exchanged segment
    rides along as an fp32 scalar. The step size stays relative to the
    segment max, and the averaged-gradient signs SignSGD consumes are only
    perturbed inside a band far below typical |grad| magnitudes. All int8
    arithmetic stays fp32: the first implementation routed the quantize /
    dequantize through fp64, whose temporaries carry 2x fp32 traffic and
    made the exchange SLOWER than plain fp32 wire (measured 360-390 ms vs
    ~243 fp32 per tune iteration at world=4), on top of ~2.5 GB of extra
    peak VRAM.

    The wire hop itself uses non_blocking=True: the measured allreduce is
    payload-independent across fp32/bf16 (~243/~231 ms), i.e. dominated by
    the per-exchange HOST blocking of a synchronous copy rather than wire
    bytes. Cross-device copy_ is stream-ordered on both endpoints (it
    records an event on the source stream and makes the destination stream
    wait), and every producer/consumer of a region runs on that device's
    current stream in issue order, so dropping the host block lets the
    exchange chain execute back-to-back on the GPUs without races.
    """
    if transport == "fp32":
        return seg.to(dev, non_blocking=True)
    if transport == "bf16":
        return seg.to(torch.bfloat16).to(dev, non_blocking=True).to(dtype)
    if transport == "int8":
        with torch.no_grad():
            src = seg.detach()
            amax = src.abs().amax()
            inv = 127.0 / amax.clamp_min(torch.finfo(src.dtype).tiny)
            q = torch.round(src * inv).clamp_(-127.0, 127.0).to(torch.int8)
            q = q.to(dev, non_blocking=True)
            scale = amax.to(dev, non_blocking=True) / 127.0
            return q.to(dtype).mul_(scale)
    raise ValueError(f"unknown gradient transport {transport!r} (fp32|bf16|int8)")


def _encode_transport(t: torch.Tensor, transport: str):
    """Encode a gradient tensor for wire transport.

    Returns ``(payload, meta)``: the wire tensor (int8 / bfloat16 / fp32) and
    the int8 per-bucket amax scalar (``None`` for fp32 / bf16). fp32 returns
    the tensor itself (read-only alias -- callers must not mutate it).
    """
    if transport == "int8":
        amax = t.abs().amax()
        inv = 127.0 / amax.clamp_min(torch.finfo(t.dtype).tiny)
        return torch.round(t * inv).clamp_(-127.0, 127.0).to(torch.int8), amax
    if transport == "bf16":
        return t.to(torch.bfloat16), None
    return t, None


def halving_doubling_allreduce(buffers: List[torch.Tensor], scale: float = 1.0, transport: str = "fp32") -> None:
    """In-place all-reduce across device-resident buffers (single process).

    Chunked recursive halving-doubling: the flat space is split into W
    chunks; the reduce-scatter phase leaves rank r owning reduced chunk r
    (each step exchanges half of the working block with the partner rank);
    the all-gather phase re-exchanges owned chunks until every rank holds
    the fully reduced buffer. Per-rank traffic is 2*(W-1)/W*bytes, same as a
    ring. Cross-device ``to()`` copies use P2P when available. ``scale`` is
    applied at the end (pass ``1/world`` to average). ``transport`` selects
    the exchange dtype: fp32 (exact), bf16 (half wire bytes) or int8
    (quarter wire bytes, symmetric per-segment amax scaling); accumulation
    stays fp32. Requires a power-of-two world (the resolver guarantees it).
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

    _reduce_scatter_halving(buffers, transport)
    _allgather_doubling(buffers, transport)

    for buf in buffers:
        buf.mul_(scale)


def _reduce_scatter_halving(buffers: List[torch.Tensor], transport: str) -> None:
    """Recursive-halving reduce-scatter across device-resident buffers.

    The flat space is split into W chunks; the working block per rank halves
    each step (each rank keeps one half, adding the partner's transport-
    rounded copy of it). Rank r ends up owning reduced chunk r; the other
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
                seg = _xchg(seg, buffers[rank].device, buffers[rank].dtype, transport)
                buffers[rank][chunk * base : chunk * mid].add_(seg)
            else:
                partner = rank - half  # keep [mid, hi)
                seg = buffers[partner][chunk * mid : chunk * hi]
                seg = _xchg(seg, buffers[rank].device, buffers[rank].dtype, transport)
                buffers[rank][chunk * mid : chunk * hi].add_(seg)
        length = half


def _allgather_doubling(buffers: List[torch.Tensor], transport: str) -> None:
    """Recursive-doubling all-gather of per-rank owned chunks.

    Mirrors the reduce-scatter block structure: the working block per rank
    doubles each step; ranks exchange the chunks the partner owns (copies,
    no adds). Every rank ends holding every chunk -- bitwise identical when
    the transport is lossless for the buffer dtype (fp32 buffers with fp32
    transport, int8 buffers with fp32 transport).
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
                dst.copy_(_xchg(src, dst.device, dst.dtype, transport))
            else:
                partner = rank - half
                src = buffers[partner][chunk * base : chunk * mid]
                dst = buffers[rank][chunk * base : chunk * mid]
                dst.copy_(_xchg(src, dst.device, dst.dtype, transport))
        length *= 2


_SIGN_LOGGED = False
_ENGAGED_LOGGED_SIG = None  # module-level init for the global read in resolve_tune_ddp_plan_


def sign_exchange_allreduce(buffers: List[torch.Tensor], transport: str = "fp32") -> None:
    """In-place sign all-reduce for SignSGD gradients (single process).

    The SignRound optimizer consumes ONLY torch.sign(grad) (weight_decay is
    always 0), so the averaged gradient's magnitude never reaches the
    update. This exchanges exactly what the step needs: a recursive-halving
    reduce-scatter (identical transport rounding of the partials as
    halving-doubling) leaves rank r owning the reduced chunk r; each rank
    then computes torch.sign() ONCE on its exact fp32 chunk and the signs
    are all-gathered as int8 -- a lossless wire format 4x smaller than
    fp32. Compared to a full halving-doubling allreduce this REMOVES the
    all-gather's transport rounding (bf16 rounding can zero out tiny
    averaged gradients, losing their sign), so the signs every rank applies
    are bitwise identical and at least as faithful to the true fp32 mean.

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

    _reduce_scatter_halving(buffers, transport)

    numel = buffers[0].numel()
    chunk = (numel + world - 1) // world
    sign_bufs: List[torch.Tensor] = []
    for rank, buf in enumerate(buffers):
        signs = torch.empty(numel, dtype=torch.int8, device=buf.device)
        lo, hi = chunk * rank, min(numel, chunk * (rank + 1))
        signs[lo:hi] = torch.sign(buf[lo:hi]).to(torch.int8)
        sign_bufs.append(signs)

    # fp32 transport on int8 buffers = plain device copies, lossless
    _allgather_doubling(sign_bufs, "fp32")

    for buf, signs in zip(buffers, sign_bufs):
        buf.copy_(signs)  # int8 -> fp32: -1.0 / 0.0 / 1.0


# hook-written statistics that are safe to merge across mirror copies:
# associative reductions reproduce the serial totals up to fp32 summation order
_MERGEABLE_STATS = {
    "imatrix": "sum",  # module.imatrix += sum(x^2) per shard
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
                if not torch.is_tensor(m_val):
                    continue
                h_val = getattr(h_mod, attr, None)
                if h_val is None:
                    setattr(h_mod, attr, m_val.to(h_mod.weight.device if hasattr(h_mod, "weight") else m_val.device))
                    continue
                m_val = m_val.to(h_val.device, h_val.dtype)
                if how == "sum":
                    setattr(h_mod, attr, h_val + m_val)
                else:  # max
                    setattr(h_mod, attr, torch.max(h_val, m_val))


_pool_move_warned: set = set()
_coll_mirror_setup_logged: set = set()


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

    Device r owns samples [r*shard, (r+1)*shard) -- the same boundaries the
    DDP tune shards and the sharded collection use -- so shard-local reads
    (ref_r build, _select_batch for shard r, per-replica collections) never
    cross devices. Pieces already on their target device are untouched
    (idempotent; a pool produced by the previous block's sharded collection
    on the same group costs nothing). Pools smaller than / indivisible by
    the world are left alone (serial consumers handle them via move-on-demand
    cats).
    """
    n = len(pool)
    world = len(devices)
    if world < 2 or n < world or n % world != 0:
        return
    shard = n // world
    for r, dev in enumerate(devices):
        dev = torch.device(dev)
        for i in range(r * shard, (r + 1) * shard):
            if pool[i].device != dev:
                pool[i] = pool[i].to(dev)


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
    if world < 2 or n < world or n % world != 0:
        return runner(block, inputs, input_others, cache_device=out_device)
    shard = n // world
    shards = [list(range(r * shard, (r + 1) * shard)) for r in range(world)]
    # the input pool typically arrives pooled on the primary device; without
    # re-placement every shard pulls its pieces cross-device from that ONE
    # source -- measured at world=8: uniformly 2.3-2.8 s of shard-forward
    # time versus 0.6-0.8 s for the same pass reading an already-distributed
    # pool. Distribute first (idempotent, same layout the tune engagement
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
    home_dev = _block_device(block)
    mirrors: List[torch.nn.Module] = []
    reps: List[torch.nn.Module] = []
    for dev in devices:
        if dev == home_dev:
            reps.append(block)
            mirrors.append(None)
        else:
            m = copy.deepcopy(block).to(dev)
            _relocate_params(m, dev)
            reps.append(m)
            mirrors.append(m)
    global _coll_mirror_setup_logged
    if world > 1 and "_coll" not in _coll_mirror_setup_logged:
        _coll_mirror_setup_logged.add("_coll")
        from auto_round import envs as _penvs

        _pl = logger.info if getattr(_penvs, "AR_PERF_COUNTERS", False) else logger.debug
        _pl(
            "[tune-ddp] collection mirror setup: %.0f ms per pass (world=%d) -- included in ref_collect/post_collect",
            (_time.perf_counter() - _t_mirrors) * 1000,
            world,
        )
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

    # spawn path, not the ReplicaGroup pool: this bare helper instance has no
    # lifecycle and would leak pool workers (collection passes are twice per
    # block -- the spawn cost is irrelevant here)
    run_threaded_spawn([lambda r=r: _run(r) for r in range(world)])
    # per-shard GPU times: sync the participating devices (their outputs are
    # consumed right after anyway -- the first downstream read syncs too)
    fwd_gpu = [0.0] * world
    for r, dev in enumerate(devices):
        if evs[r] is not None:
            try:
                torch.cuda.synchronize(dev)
                fwd_gpu[r] = evs[r][0].elapsed_time(evs[r][1])
            except RuntimeError as e:  # a broken pair must never kill the pass
                logger.warning_once("[perf] sharded-collect GPU wall unavailable for a pair (%s)", e)
                fwd_gpu[r] = float("nan")
    _t_split = _time.perf_counter()
    # Return the SERIAL structure: a per-sample list ([1, S, H] pieces from
    # split_outputs), NOT one flat/batched tensor -- downstream consumers cat
    # per-sample refs along dim 0 (tune loss) and iterate the list for the
    # cascade inputs, so a tensor here changes every per-sample shape.
    pieces: List[torch.Tensor] = []
    split_outputs = getattr(runner, "split_outputs", None)
    for part in parts:
        pieces.extend(split_outputs(part) if split_outputs else torch.split(part, 1, dim=0))
    # threaded shard calls each stamped a PARTIAL last_output_dict (their own
    # shard's rows); the serial text path leaves it unset so callers fall back
    # to the returned list -- clear the residue to keep that contract.
    runner.last_output_dict = None
    _t_merge = _time.perf_counter()
    if merge_stats:
        _merge_mirror_stats(block, mirrors)
    mirrors.clear()  # drop mirror refs; the caching allocator reclaims them
    reps.clear()
    # steady-state breakdown for the collection passes: setup = ephemeral
    # mirror deepcopy/relocate (per pass!), fwd = threaded shard forwards
    # (wall includes enqueue, gpu is the device-measured chain incl. the
    # per-thread output parking), split/merge = host assembly. Logged for
    # EVERY pass: the first-call-per-block compile signature (uniformly
    # ~2.3-2.8 s fwd at world=8 vs 0.6-0.8 s on the second pass) needs the
    # per-block view to separate warmup from steady state.
    _walls = sorted(fwd_walls)
    _gpus = sorted(fwd_gpu)
    logger.debug(
        "[tune-ddp] sharded collect breakdown: world=%d n=%d setup=%.0fms "
        "fwd_wall[min/med/max]=%.0f/%.0f/%.0fms fwd_gpu[min/med/max]=%.0f/%.0f/%.0fms "
        "split=%.0fms merge=%.0fms",
        world,
        n,
        _t_setup_ms,
        _walls[0],
        _walls[len(_walls) // 2],
        _walls[-1],
        _gpus[0],
        _gpus[len(_gpus) // 2],
        _gpus[-1],
        (_t_merge - _t_split) * 1000,
        (_time.perf_counter() - _t_merge) * 1000,
    )
    return pieces


def pre_wrap_shard_candidate() -> bool:
    """Cheap pre-wrap probe: might the data-parallel lane engage for this block?

    Only the env world and CUDA availability -- the full resolver runs
    post-wrap (it needs wrapper params for mirror pricing). A false positive
    is safe: the serial fallback fills the deferred searches on the home
    device. A false negative only loses sharding.
    """
    from auto_round import envs as _envs

    try:
        world = int(getattr(_envs, "AR_TUNE_DDP_WORLD", 1) or 1)
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
    the home device -- identical semantics to the never-deferred serial path.
    """
    wrappers = [(n, m) for n, m in block.named_modules() if getattr(m, "_init_search_deferred", False)]
    if not wrappers:
        return
    names = [n for n, _ in wrappers]

    plan = getattr(replica_group, "plan", None)
    world = getattr(plan, "world", 1) if plan is not None else 1
    if replica_group is None or world < 2 or len(wrappers) < 2:
        for _n, w in wrappers:
            w.run_deferred_init_search()
        for _n, w in wrappers:
            w._finalize_deferred_init()
        return

    devices = list(plan.devices)
    home = devices[0]
    rep_wrappers = []
    for rep in replica_group.replicas:
        rep_wrappers.append({n: m for n, m in rep.named_modules() if hasattr(m, "run_deferred_init_search")})
    # round-robin the searches: consecutive layers spread across the replicas
    owner = {n: i % world for i, n in enumerate(names)}

    def _search_on(r):
        def _go():
            dev = devices[r]
            if dev.type == "cuda":
                with torch.cuda.device(dev):
                    for n in names:
                        if owner[n] == r:
                            rep_wrappers[r][n].run_deferred_init_search()
            else:
                for n in names:
                    if owner[n] == r:
                        rep_wrappers[r][n].run_deferred_init_search()

        return _go

    run_threaded_spawn([_search_on(r) for r in range(world)])

    # home collection: pull every searched init_scale to the home device
    results = {}
    for n in names:
        src = rep_wrappers[owner[n]][n].init_scale
        results[n] = src.to(home) if torch.is_tensor(src) else src

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

    def __init__(self, block, plan: DDPPlan, grad_transport: str = "bf16") -> None:
        self.plan = plan
        self.grad_transport = grad_transport
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
            # a replica without gradients means the collected params are not
            # the ones the forward/backward touched -- the tune would silently
            # degrade to single-shard updates
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
                    logger.info(
                        "[tune-ddp] sign-cast exchange engaged: world=%d transport=%s (int8 sign allgather)",
                        self.world,
                        self.grad_transport,
                    )
                sign_exchange_allreduce(bufs, transport=self.grad_transport)
            else:
                halving_doubling_allreduce(bufs, scale=1.0 / self.world, transport=self.grad_transport)
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
