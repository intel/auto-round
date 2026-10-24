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
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import torch

from auto_round.logger import logger


@dataclass
class DDPPlan:
    """Resolved data-parallel plan for one block."""

    world: int
    devices: list[torch.device]
    shard_size: int
    notes: list[str] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return self.world > 1


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
    mirrors AR_TUNE_DDP_MAX_COLLECT_FORWARD_DEVICES (0 = uncapped);
    ``batch_mode`` carries ``--parallel_batch`` ("split" = the global batch
    shards across replicas, serial-equivalent updates; "per_replica" = every
    replica draws the full --batch_size from its own pool shard, torchrun-like
    effective batch = batch_size x world). The policy is set on the
    CompressContext once at the entry; the engine reads it from there.
    """

    world: int = 1
    source: str = "off"
    collect_forward_cap: int = 0
    batch_mode: str = "split"

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
        self.plan: DDPPlan | None = None
        self.pool: MirrorPool | None = None
        self.pool_used: bool = False


def parallel_state(quantizer, create: bool = False) -> ParallelTuneState | None:
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


def log_zero_shot_parallel_noop_(policy) -> None:
    """ONE INFO line when an explicit ``--parallel_quantization N`` lands on
    the zero-shot RTN path.

    Plain RTN without calibration (``need_calib == False``) runs
    ``Orchestrator._quantize_zero_shot`` -- no calibration forward, so the
    parallel engine never engages and the request is silently ignored. Emit
    exactly one INFO line for an EXPLICIT request (``auto``/``off`` add no
    user expectation and stay silent). No refusal: weights-only RTN has
    nothing to shard.
    """
    if isinstance(policy, ParallelPolicy) and getattr(policy, "source", "") == "explicit" and policy.world > 1:
        logger.info(
            "[tune-ddp] --parallel_quantization %d has no effect on this run: "
            "zero-shot RTN needs no calibration forward (weights only); quantizing serially",
            policy.world,
        )


def check_parallel_device_map_compat_(world: int, device_map, iters: int) -> None:
    """Stop at policy resolution (before model load) when a run cannot parallel-tune.

    A multi-device ``--device_map`` with a live tune loop (``iters > 0``)
    places blocks across several accelerator devices (pipeline-style
    stages); the data-parallel lane mirrors whole blocks on single devices,
    so without this check the run loads the model and fails only at block 0
    in ``resolve_tune_ddp_plan_`` ("block spans N devices; parallel tuning
    requires single-device block placement"). Detect the same condition
    from the device map alone and stop with the same message, before the
    model loads. The runtime per-block check stays as the second line of
    defense (API users, and any map the resolver observes differently).
    Device-map semantics are untouched: a later PP+DP design keeps
    device_map as the per-group pipeline map.
    """
    if world < 2 or iters <= 0 or device_map is None:
        return
    from auto_round.utils.device import parse_available_devices

    try:
        devs = [d for d in parse_available_devices(device_map) if not str(d).startswith("cpu")]
    except Exception:  # pragma: no cover - unparsable map: the runtime check decides
        return
    if len(devs) > 1:
        dev_type = str(devs[0]).split(":")[0]
        dev_type = dev_type.upper() if ":" in str(devs[0]) else "CUDA"
        raise RuntimeError(
            f"block spans {len(devs)} {dev_type} devices; parallel tuning requires "
            "single-device block placement (run with no --device_map or a single-device "
            "--device_map; multi-device/auto maps may shard a block across GPUs -- or run "
            "with --parallel_quantization off)"
        )


def resolve_ddp_plan(
    world: int,
    home: torch.device,
    batch_size: int,
    visible_devices: Sequence[int] | None = None,
    vram_free_bytes: int | None = None,
    mirror_footprint_bytes: int | None = None,
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
    notes: list[str] = []
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
    (policy world, accelerator home, no LFQ, list pools, not
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
    _lfq_active = bool(_constraints.get("enable_lfq"))
    _home_span_devs = {p.device for p in block.parameters() if p.device.type == home.type}
    eligible = (
        world > 1
        and home.type in _SUPPORTED_ACCEL_TYPES
        and not _lfq_active
        and isinstance(fp_inputs, list)
        and (fp_outputs is None or isinstance(fp_outputs, list))
        and not is_distributed()
        and len(_home_span_devs) <= 1
    )
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
            (
                _layer_mem,
                _act_mem,
                _io_mem,
                _add_mem,
                _pd_act,
                _pd_exp,
            ) = estimate_tuning_block_mem(block, fp_inputs, _per_fwd)
            # ``_act_mem`` already sums the per-layer ``output_memory``
            # (grad-doubled) with the MoE ratio applied -- the dict totals
            # are already included. The two per-device maps
            # (``_pd_act``/``_pd_exp``) split the same activation charge by
            # weight home for multi-GPU placement planning; a mirror is
            # single-device by contract, so the aggregate charge is the
            # right price here.
            _act_bytes = int((_act_mem + _io_mem + _add_mem) * 2**30)
            mirror_bytes += _act_bytes
        except Exception as e:  # pragma: no cover - estimator is best-effort
            # a failed estimate is a degraded state: the budget silently
            # drops back to weights-only pricing (an activation-heavy block
            # can then pass a plan it does not fit)
            logger.warning("[tune-ddp] activation pricing skipped (estimator failed: %s)", e)
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
        if plan.world > 1:
            # F5: the final peak-VRAM summary only reports devices the
            # memory monitor samples, and its callers pass the device-map
            # list -- replicas engaged outside the map would stay invisible.
            # Register the plan devices once the lane engages (idempotent,
            # per block; best-effort: the monitor must never gate tuning)
            try:
                from auto_round.utils.device import memory_monitor

                memory_monitor.register_extra_devices(plan.devices)
            except Exception:  # pragma: no cover - diagnostics must not gate tuning
                pass
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
            _mode = _quantizer_policy(quantizer).batch_mode
            if _mode == "per_replica":
                # each replica draws the full requested batch (clamped to its
                # pool shard): the effective global batch scales with the world
                _per_draw = max(1, min(global_batch_size, nsamples // max(plan.world, 1)))
                _eff_batch = _per_draw * plan.world
            else:
                _eff_batch = global_batch_size
            logger.info(
                "[tune-ddp] engaged: world=%d, batch_size per device=%d, batch=%s, "
                "effective global batch=%d, devices=%s",
                plan.world,
                plan.shard_size,
                _mode,
                _eff_batch,
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


def _param_grad_buffers(params_by_device: list[list[torch.nn.Parameter]]) -> list[torch.Tensor | None]:
    """Flatten each replica's gradients into one contiguous fp32 buffer."""
    bufs: list[torch.Tensor | None] = []
    for params in params_by_device:
        parts = [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
        if not parts:
            bufs.append(None)
            continue
        buf = torch.cat(parts) if len(parts) > 1 else parts[0].clone()
        bufs.append(buf.to(torch.float32) if buf.dtype != torch.float32 else buf)
    return bufs


def _write_back_grads(buf: torch.Tensor | None, params: list[torch.nn.Parameter]) -> None:
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


def halving_doubling_allreduce(buffers: list[torch.Tensor], scale: float = 1.0) -> None:
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


def _reduce_scatter_halving(buffers: list[torch.Tensor]) -> None:
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


def _allgather_doubling(buffers: list[torch.Tensor]) -> None:
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


_ENGAGED_LOGGED_SIG = None  # module-level init for the global read in resolve_tune_ddp_plan_


# Calibration statistics folded from the mirror copies into the home after a
# sharded collection pass. The engine owns act_max (the composer's act_max
# hooks); a block quantizer declares what its own hooks write in
# ``BaseQuantizer.calibration_stats``, and a pass merges the union (see
# ``calibration_merge_stats``). Associative reductions reproduce the serial
# totals up to fp32 summation order.
ENGINE_CALIBRATION_STATS = MappingProxyType({"act_max": "max"})  # element-wise running max
STAT_REDUCTIONS = ("sum", "max")


def calibration_merge_stats(quantizer, quantizer_hooks) -> dict[str, str] | None:
    """Shard-merge map for a calibration pass carrying ``quantizer_hooks``.

    The engine's act_max plus the quantizer's declared ``calibration_stats``.
    ``None`` when the quantizer registered hooks without a declaration: the
    engine cannot fold statistics it does not know, so the pass must stay
    serial (a sharded pass would keep only the home shard's samples).
    """
    declared = getattr(quantizer, "calibration_stats", None)
    if quantizer_hooks and declared is None:
        return None
    stats = dict(ENGINE_CALIBRATION_STATS)
    for attr, how in (declared or {}).items():
        if how not in STAT_REDUCTIONS:
            raise ValueError(
                f"{type(quantizer).__name__}.calibration_stats[{attr!r}] = {how!r}: "
                f"supported shard reductions are {STAT_REDUCTIONS}"
            )
        stats[attr] = how
    return stats


def _merge_mirror_stats(home: torch.nn.Module, mirrors: list[torch.nn.Module], stats) -> None:
    """Fold hook-written statistics from mirror copies back into the home.

    Each mirror forwarded a disjoint sample shard; its hooks accumulated
    into mirror-local module attrs which die with the mirror. ``stats`` maps
    each attribute to its reduction ("sum" | "max"); both are associative, so
    the merged home totals equal the serial pass's totals up to fp32
    summation order (~1e-7 relative) -- far below any quantization-relevant
    scale. Python-scalar statistics (a row count) fold with the same
    semantics, so they always co-travel with their tensor sibling.

    Each mirror value is folded exactly once and then removed from the
    mirror: pool mirrors live across all collection passes of a block, so a
    later pass's merge would otherwise add the same shard values to the home
    again (a hook pass followed by any other sharded pass doubled the
    mirrors' share).
    """
    home_mods = dict(home.named_modules())
    for mirror in mirrors:
        if mirror is None:
            continue
        for name, m_mod in mirror.named_modules():
            h_mod = home_mods.get(name)
            if h_mod is None:
                continue
            for attr, how in stats.items():
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
                elif h_val is None:
                    setattr(h_mod, attr, m_val.to(h_mod.weight.device if hasattr(h_mod, "weight") else m_val.device))
                else:
                    m_val = m_val.to(h_val.device, h_val.dtype)
                    if how == "sum":
                        setattr(h_mod, attr, h_val + m_val)
                    else:  # max
                        setattr(h_mod, attr, torch.max(h_val, m_val))
                delattr(m_mod, attr)  # folded: a later pass must not add it again


_MODULE_INTERNALS = frozenset(vars(torch.nn.Module()))


def _stat_signature(val):
    """Change signature of a module attribute a hook may accumulate into:
    tensors by identity + in-place version, containers holding tensors by
    identity + length; other kinds (config scalars, strings, bookkeeping
    such as hook-handle lists) are not tracked."""
    if torch.is_tensor(val):
        return (id(val), val._version)
    if isinstance(val, (list, dict)):
        items = val.values() if isinstance(val, dict) else val
        if any(torch.is_tensor(v) for v in items):
            return (id(val), len(val))
    return None


def _mirror_attr_snapshot(mirrors) -> list[dict]:
    return [
        {
            (name, attr): _stat_signature(val)
            for name, mod in mirror.named_modules()
            for attr, val in vars(mod).items()
            if attr not in _MODULE_INTERNALS
        }
        for mirror in mirrors
    ]


def _warn_undeclared_mirror_writes(mirrors, snapshots, merge_stats) -> None:
    """Safety net for the ``calibration_stats`` contract.

    A tensor/container attribute that changed on a mirror module during a
    sharded pass but is not in the merge map dies with the mirror, so the
    home keeps only its own shard's values. Warn once per attribute name.
    """
    for mirror, snap in zip(mirrors, snapshots):
        for name, mod in mirror.named_modules():
            for attr, val in vars(mod).items():
                if attr in _MODULE_INTERNALS or attr in merge_stats:
                    continue
                sig = _stat_signature(val)
                if sig is not None and snap.get((name, attr)) != sig:
                    logger.warning_once(
                        "[tune-ddp] '%s' changed on mirror modules during a sharded collection pass but is not a "
                        "declared calibration statistic: if a calibration hook accumulates it, the mirrors' share "
                        "is dropped (the home keeps only its own shard). Declare its reduction in the block "
                        "quantizer's calibration_stats, or run with --parallel_quantization off.",
                        attr,
                    )


def distribute_pool(pool: list[torch.Tensor], devices: list[torch.device]) -> None:
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


def contiguous_shard_bounds(n: int, world: int) -> list[int]:
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
    the pool; after a zero-shot RTN/OptRTN quantize the mirrors are synced to
    the quantized home block so the cascade reuses them; the tune ranks tune
    the mirrors (spmd). ``release()`` drops the references; the home block
    stays canonical throughout.
    """

    def __init__(self, block, devices):
        import copy as _copy

        home_dev = _block_device(block)
        self.block = block
        self.devices = list(devices)
        self.reps: list[torch.nn.Module] = []
        self._owned: list[torch.nn.Module] = []  # deepcopies this pool must free
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
    devices: list[torch.device],
    sample_count: int | None = None,
    merge_stats: Mapping[str, str] | None = None,
    max_devices: int = 0,
    pool: MirrorPool | None = None,
):
    """Parallelize a no-grad collection forward across ``devices``.

    ``merge_stats`` (attr -> "sum" | "max") folds the hook-written statistics
    from the mirrors into the home after the pass; ``None`` merges nothing.

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
    mirrors: list[torch.nn.Module] = []  # the actual deepcopies used this pass
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
        reps: list[torch.nn.Module] = []
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
    parts: list = [None] * world
    fwd_walls = [0.0] * world  # per-thread stores at distinct indices: race-free
    evs: list = [None] * world

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

    # spawn path: collection passes run twice per block, so the per-call
    # thread spawn cost is irrelevant here
    _attr_snaps = _mirror_attr_snapshot(mirrors) if merge_stats is not None else None
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
    pieces: list[torch.Tensor] = []
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
    if merge_stats is not None:
        _merge_mirror_stats(block, mirrors, merge_stats)
        _warn_undeclared_mirror_writes(mirrors, _attr_snaps, merge_stats)
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


def _block_device(block) -> torch.device:
    p = next(block.parameters(), None)
    return p.device if p is not None else torch.device("cpu")
