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

"""Per-chunk device placement for block calibration data (--calibration_data_device).

Policy (user rulings, Sep-13):

* single-device-first: when the consumer device (per-block loss device on
  iters>0 lanes, else the cache primary) holds the pool beside its full charge
  (computed working allowance + flat 0.25 GiB allocator reserve + measured
  collection-window term + occupied pool bytes + per-device activation
  charge), every chunk stays there -- zero peer traffic.
* otherwise water-fill: chunks are placed so every receiving device ends at
  the SAME predicted headroom level (min-max optimal); when the pool
  over-commits the fleet the over-commitment is spread evenly across the
  charged headrooms. Charges: per-device tuning state (14 B/param logical,
  deduped), per-device activation budget (estimator + routed/stacks split by
  expert homes), the consumer's need model, and resident pool bytes.
* ``cpu`` mode parks the pools on host RAM explicitly (forwards still run on
  the GPUs) -- the pool-scoped equivalent of ``low_gpu_mem_usage`` without its
  other side effects. ``auto`` never falls back to CPU silently: over-
  commitment is spread, never parked on the host unasked.

``calibration_data_device`` parameter (CLI ``--calibration_data_device`` / API
keyword): ``auto`` (default) | ``off`` | ``cpu`` | explicit csv (``cuda:1,cuda:2``).
There is deliberately no environment variable for this knob.

CUDA-only today: the free-memory probe underneath is a cuda API, so on other
accelerator families the resolver returns ``None`` (policy inactive, pools keep
their today placement) rather than guessing at unprobed capacities.
"""

from typing import Callable, List, Optional, Sequence

import torch

from auto_round.logger import logger


class PoolPlacement:
    """Deterministic per-chunk device plan for one calibration output pool."""

    def __init__(self, devices: Sequence[str], capacities: Sequence[int], n_chunks: int):
        self.devices = [str(d) for d in devices]
        self.capacities = [int(c) for c in capacities]
        self.plan = _spread_plan(self.devices, self.capacities, n_chunks)

    def device_for_index(self, i: int) -> str:
        """Device for output chunk ``i`` (deterministic, wraps for over-long pools)."""
        return self.plan[i % len(self.plan)]

    def counts(self) -> dict:
        """Chunks per device (diagnostics)."""
        out: dict = {}
        for d in self.plan:
            out[d] = out.get(d, 0) + 1
        return out

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"PoolPlacement({self.counts()})"


def _spread_plan(devices: Sequence[str], capacities: Sequence[int], n_chunks: int) -> List[str]:
    """Largest-remainder proportional split, interleaved across devices.

    Interleaving (round-robin order of the assigned slots) keeps consecutive
    chunks on different devices so transient per-chunk peaks never stack on a
    single peer.
    """
    total = sum(capacities)
    if total <= 0 or n_chunks <= 0 or not devices:
        return list(devices)
    quotas = [c * n_chunks / total for c in capacities]
    counts = [int(q) for q in quotas]
    remainders = sorted(range(len(devices)), key=lambda i: quotas[i] - counts[i], reverse=True)
    for i in range(n_chunks - sum(counts)):
        counts[remainders[i % len(remainders)]] += 1
    # interleave: one slot per device in turn, skipping exhausted devices
    plan: List[str] = []
    cursors = [0] * len(devices)
    while any(cursors[i] < counts[i] for i in range(len(devices))):
        for i in range(len(devices)):
            if cursors[i] < counts[i]:
                plan.append(devices[i])
                cursors[i] += 1
    return plan


def _bytes_by_device(obj) -> tuple:
    """(device_str -> bytes, total bytes) for tensor leaves of a nested pool object."""
    per_device: dict = {}

    def _walk(o):
        if isinstance(o, torch.Tensor):
            key = str(o.device)
            per_device[key] = per_device.get(key, 0) + int(o.numel()) * o.element_size()
        elif isinstance(o, dict):
            for v in o.values():
                _walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                _walk(v)

    _walk(obj)
    return per_device, sum(per_device.values())


def _short_device_key(dev: str) -> str:
    """'cuda:0' -> '0', 'cpu' -> 'cpu' (memory-monitor grammar)."""
    d = str(dev)
    if d.startswith("cuda:"):
        return d.split(":", 1)[1]
    return d


def calib_data_line(inputs, aux, plan, outputs_bytes: int, n_chunks: int, primary: str) -> str:
    """One-line calibration-data summary in the memory-monitor format:

    ``'input': 8.12GB, 'output': 8.12GB, 'aux': 0.12GB, 'per_device': {'0': 2.03GB, '1': 2.03GB, 'cpu': 0.12GB}``

    ``inputs`` is the list of live input pool objects (fp and q chains merged --
    the dual chain is upstream's qon architecture; during collection both pools
    coexist, which is the peak the placement accounts for). Input/aux devices
    are ground truth (walked from the tensors); output bytes are this block's
    planned placement (the plan, or all-primary under today's behavior).
    Per-device totals combine parked + planned.
    """
    per_device: dict = {}
    input_total = 0
    for obj in inputs:
        by_dev, total = _bytes_by_device(obj)
        input_total += total
        for d, b in by_dev.items():
            per_device[d] = per_device.get(d, 0) + b
    by_dev, aux_total = _bytes_by_device(aux)
    for d, b in by_dev.items():
        per_device[d] = per_device.get(d, 0) + b
    if outputs_bytes > 0:
        if plan is not None:
            per_chunk = outputs_bytes / max(n_chunks, 1)
            for dev, cnt in plan.counts().items():
                per_device[dev] = per_device.get(dev, 0) + int(cnt * per_chunk)
        else:
            per_device[str(primary)] = per_device.get(str(primary), 0) + outputs_bytes
    devs = ", ".join(
        f"'{_short_device_key(d)}': {b / 2**30:.2f}GB" for d, b in sorted(per_device.items(), key=lambda kv: -kv[1])
    )
    if plan is None:
        place = "none->primary"
    elif len(plan.devices) == 1:
        place = "single"
    else:
        place = "spread"
    return (
        f"'input': {input_total / 2**30:.2f}GB, 'output': {outputs_bytes / 2**30:.2f}GB, "
        f"'aux': {aux_total / 2**30:.2f}GB, 'plan': '{place}', 'per_device': {{{devs}}}"
    )


def _tensor_bytes(obj) -> int:
    """Total bytes of tensor leaves in a nested list/tuple/dict pool object."""
    if isinstance(obj, torch.Tensor):
        return int(obj.numel()) * obj.element_size()
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(v) for v in obj)
    return 0


def _pool_chunk_count(obj) -> int:
    """Number of per-sample chunks in a pool object (length of first list found)."""
    if isinstance(obj, (list, tuple)):
        tensor_like = [v for v in obj if isinstance(v, torch.Tensor)]
        if tensor_like:
            return len(obj)
        for v in obj:
            n = _pool_chunk_count(v)
            if n:
                return n
        return 0
    if isinstance(obj, dict):
        for v in obj.values():
            n = _pool_chunk_count(v)
            if n:
                return n
    return 0


def _move_pool_to(obj, target: str):
    """Recursively move a pool's tensors onto ``target`` in place (skip locals)."""
    if isinstance(obj, torch.Tensor):
        return obj.to(target) if str(obj.device) != target else obj
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = _move_pool_to(obj[i], target)
        return obj
    if isinstance(obj, tuple):
        return tuple(_move_pool_to(v, target) for v in obj)
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            obj[k] = _move_pool_to(obj[k], target)
        return obj
    return obj


def consolidate_pool_onto(objs, target: str, block, batch_size: int, reserved_bytes: int = 0, iters: int = 0) -> str:
    """Consolidate the incoming calibration pools onto the compute device.

    Returns ``'local'`` (already on target), ``'consolidated'`` (moved in one
    bulk pass), or ``'spread'`` (the pools do not fit next to the block's
    working set -- consumers keep fetching batch-by-batch instead). This is the
    fits-home rung of the ladder: fit-primary / fit-home -> single device,
    oversized pool -> spread + per-batch gather.
    """
    if not str(target).startswith("cuda"):
        return "spread"
    if any(isinstance(o, (tuple, torch.Tensor)) for o in objs if o is not None):
        # a top-level tuple or bare-tensor pool cannot be moved in place (the
        # move helper returns a NEW object for those; reassigning the caller's
        # list would not rebind the caller's own references) -- leave it and
        # say so rather than reporting a phantom consolidation
        logger.warning(
            "[calib-data-device] top-level %s pool cannot be consolidated in place; leaving it",
            "tuple" if any(isinstance(o, tuple) for o in objs if o is not None) else "tensor",
        )
        return "spread"
    from auto_round.utils.device import probe_usable_bytes

    total = sum(_tensor_bytes(o) for o in objs if o is not None)
    if total <= 0:
        return "local"
    free = probe_usable_bytes(target)
    if free is None:
        return "spread"
    need = placement_need_bytes(block, objs[0], batch_size, iters=iters, primary=target)
    # ``reserved_bytes``: output pools that will share the target this block.
    # The first GPU validation showed the inputs-only budget was optimistic --
    # consolidating 4GiB of inputs onto the busiest device while the outputs
    # were also headed there produced exactly the OOM the policy exists to
    # prevent. Consolidate only when BOTH fit beside the working set.
    if free - need - reserved_bytes < total:
        return "spread"
    devices = set()

    def _collect(o):
        if isinstance(o, torch.Tensor):
            devices.add(str(o.device))
        elif isinstance(o, dict):
            for v in o.values():
                _collect(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                _collect(v)

    for o in objs:
        if o is not None:
            _collect(o)
    if devices <= {target}:
        return "local"
    for o in objs:
        if o is not None:
            _move_pool_to(o, target)
    return "consolidated"


def resolve_pool_placement(
    pool_bytes: int,
    n_chunks: int,
    primary: str,
    need_bytes: int,
    candidate_devices: Sequence[str],
    free_probe: Callable[[str], Optional[int]],
    mode: str = "auto",
    consumer: Optional[str] = None,
    occupied_bytes: int = 0,
    peer_state_bytes: Optional[dict] = None,
    activation_bytes: Optional[dict] = None,
) -> Optional[PoolPlacement]:
    """Decide per-chunk output placement for a calibration pool.

    ``need_bytes`` is the candidate's own forward working-set demand (estimated
    per block via :func:`placement_need_bytes`); the pool must fit in
    ``free(candidate) - need_bytes`` to stay single-device-resident.

    ``consumer`` (iters>0 lane) retargets the single-device preference from the
    cache primary to the block's tune consumer (loss/compute device): the fp
    reference pool is bulk-pulled there anyway, so outputs born on the consumer
    make that pull a no-op and keep the primary free of pool traffic. The cache
    primary then participates as a plain peer (no working-set charge).

    ``occupied_bytes`` counts pool bytes that ALREADY sit on the candidate
    (this block's incoming input pool) against its single-device budget; without
    it the gate alternates -- one block's single plan loads the device, the next
    block's resolve sees less free and spreads, the next one singles again.

    Returns ``None`` when the caller should keep today's behavior (policy off,
    CPU-parked lane, or no usable probed device). Capacity shortfalls do NOT
    return None: the water-fill spreads the pool and, on over-commit, spreads
    the over-commitment evenly across charged headrooms.
    """
    mode = (mode or "auto").strip().lower()
    if mode == "off":
        return None
    if mode == "cpu":
        return PoolPlacement(["cpu"], [1], n_chunks)  # explicit host-RAM parking
    if str(primary).startswith("cpu"):
        logger.debug("[calib-data-device] resolve: none (cpu primary)")
        return None  # low_gpu_mem_usage (or a CPU lane) owns placement here
    forced = None
    if mode not in ("", "auto"):
        # plain csv list: "cuda:1,cuda:2" or bare indices "1,2" (accelerate
        # device_map style) -- digit tokens get the cuda: prefix
        forced = ["cuda:" + t if t.isdigit() else t for t in (d.strip() for d in mode.split(",")) if t]

    consumer = str(consumer) if consumer is not None else None
    if consumer is not None and (consumer == str(primary) or not consumer.startswith("cuda")):
        consumer = None  # degenerate target: fall back to primary-first
    candidates: List[str] = []
    if forced is not None:
        candidates = forced
    elif consumer is not None:
        # consumer-first: the cache primary demotes to a plain peer
        seen = {consumer}
        candidates = [consumer]
        for d in [str(primary)] + [str(x) for x in candidate_devices]:
            if d.startswith("cpu") or d in seen:
                continue
            seen.add(d)
            candidates.append(d)
    else:
        seen = {str(primary)}
        candidates = [str(primary)]
        for d in candidate_devices:
            key = str(d)
            if key.startswith("cpu") or key in seen:
                continue
            seen.add(key)
            candidates.append(key)

    probed = [(d, free_probe(d)) for d in candidates]
    rejected = [d for d, f in probed if f is None]
    if rejected:
        # a forced csv with a typo (e.g. cuda1) must not vanish silently
        logger.warning(
            "[calib-data-device] ignoring unusable device entr%s %s (not parseable as a cuda device with free memory)",
            "y" if len(rejected) == 1 else "ies",
            ", ".join(rejected),
        )
    usable = [(d, f) for d, f in probed if f is not None and f > 0]
    if not usable:
        logger.debug("[calib-data-device] resolve: none (no usable device among %s)", candidates)
        return None

    eff = consumer if consumer is not None else str(primary)
    primary_free = dict(usable).get(eff, 0)
    _act = activation_bytes or {}
    _eff_charge = need_bytes + occupied_bytes + _act.get(eff, 0)
    if primary_free - _eff_charge >= pool_bytes:
        # single-device-first: identical to today's behavior, zero peer
        # traffic; the activation charge matches the water-fill rung below
        # (without it the rung takes a single plan the loop cannot hold)
        return PoolPlacement([eff], [max(primary_free - _eff_charge, 1)], n_chunks)

    # Water-fill placement with full charges: place the pool so every
    # receiving device ends at the SAME predicted headroom level t --
    # i.e. minimize the maximum predicted VRAM across the fleet,
    # calculated upfront. Per-device headroom h_i = free_i - charge_i,
    # where the charge is the device's own in-loop load that materializes
    # after this probe (tuning state x 14 B/param on every device, the
    # entry's forward-graph activation budget, the consumer's working set
    # and resident pool bytes). Allocation a_i = max(h_i - t, 0) with t
    # solving sum(a_i) = pool_bytes:
    #   t >= 0  : everything fits; every receiver ends with the common
    #             margin t (min-max optimal -- nobody is the thin card);
    #   t <  0  : the pool over-commits the fleet; the over-commitment is
    #             spread EVENLY (each device exceeds its charged headroom
    #             by the same -t) instead of piling onto whoever had raw
    #             free. There is no ignore-charges branch: the charges
    #             shape the split in both regimes.
    peer_state = peer_state_bytes or {}
    act = activation_bytes or {}
    headroom = [
        f - (need_bytes + occupied_bytes if d == eff else peer_state.get(d, 0)) - act.get(d, 0) for d, f in usable
    ]
    # solve the level t by binary search on monotone g(t) = sum(max(h-t,0)) - pool
    lo, hi = min(headroom) - pool_bytes, max(headroom)
    for _ in range(64):
        mid = (lo + hi) / 2
        if sum(max(h - mid, 0) for h in headroom) > pool_bytes:
            lo = mid
        else:
            hi = mid
    level = (lo + hi) / 2
    alloc = [max(h - level, 0) for h in headroom]
    if sum(alloc) <= 0:  # degenerate: keep the planless default (loud OOM path)
        logger.debug(
            "[calib-data-device] resolve: none (all candidate headrooms exhausted; pool %.2fGiB)",
            pool_bytes / 2**30,
        )
        return None
    devs = [d for (d, _f), a in zip(usable, alloc) if a > 0]
    caps = [int(a) for (d, _f), a in zip(usable, alloc) if a > 0]
    plan_obj = PoolPlacement(devs, caps, n_chunks)
    plan_obj.level_bytes = int(level)  # common predicted headroom after fill (negative = even over-commit)
    return plan_obj


# Minimal fragmentation pad for probe->peak drift. NOT card-proportional:
# fragmentation does not scale with device size, and the free-memory probe is
# the primary actual signal this pad only backs up. Observed on 24 GiB cards:
# the linear_loop tune loop ran at 23.09-23.14 GiB reserved of 23.58 (~98%)
# without OOM, so the pad needs only cover allocator slack in the last
# allocations. Proportional ceilings, where wanted, belong to budget ratios
# (cf. max_mem_ratio/card_0_threshold=0.9 in the mapped-placement path).
_RESERVE_BYTES = int(0.25 * 2**30)
# Floor for the computed working allowance (tiny blocks / degenerate reads)
_WORKING_FLOOR_BYTES = int(0.125 * 2**30)


def _activation_bytes_for(block, pool, batch_size, iters, config) -> dict:
    """Per-device activation budget charged into every device's headroom.

    The tune loop's forward/backward graph executes on EVERY device the
    block spans: each module's saved output + grad lands on the module's
    weight home (the entry's batch cats AND the MoE stages' routed-row
    caches -- the 3.2 GiB/peer that state-only charges left unpriced on the
    4-GPU hy3 lane). {device: bytes}; {} when unknown (charges nothing).
    """
    if iters <= 0:
        return {}
    try:
        from auto_round.algorithms.quantization.sign_round.quantizer import _activation_bytes_by_device

        got = _activation_bytes_by_device(block, pool, batch_size, config)
        return {str(d): int(b) for d, b in got.items()} if got else {}
    except Exception as e:  # pragma: no cover - placement must never break
        logger.debug("[calib-data-device] activation budget unavailable (%s)", e)
        return {}


def _state_bytes_by_device(block) -> dict:
    """In-loop tuning state per device: params homed there x 14 B (actual walk).

    Same layout constant as :func:`placement_need_bytes` (fp32 value + grad +
    best-params snapshot + bf16 copy), computed for EVERY device so pool
    placement can charge peers too. At attach time (when pools are placed)
    none of it exists yet: values materialize at wrap, grads and snapshots
    during the loop -- which is exactly why peer devices left at their full
    probed free over-fill by the first snapshot write (observed: 24 MiB
    snapshot ask failing at 5.5 MiB free on a card the resolve gate had
    treated as empty).
    """
    out: dict = {}
    try:
        for p in block.parameters():
            out[str(p.device)] = out.get(str(p.device), 0) + p.numel() * 14
    except Exception as e:  # pragma: no cover - placement must never break
        logger.debug("[calib-data-device] per-device state walk failed (%s)", e)
        return {}
    return out


def _dominant_param_esize(block) -> int:
    """Element size (bytes) of the block's dominant parameter dtype."""
    counts: dict = {}
    try:
        for p in block.parameters():
            counts[p.element_size()] = counts.get(p.element_size(), 0) + p.numel()
    except Exception as e:  # pragma: no cover - exotic modules: fall back to bf16-ish
        logger.debug("[calib-data-device] dominant-esize walk failed on %r (%s); assuming 2", block, e)
        return 2
    return max(counts, key=lambda k: counts[k]) if counts else 2


def _widest_out_and_hidden(block):
    """(widest out_features, modal in_features) over the block's Linear/Conv1D."""
    """
    The modal in_features approximates the hidden size (every attention/mlp
    projection consumes hidden-wide inputs); the widest out_features bounds
    the largest single-module activation buffer (e.g. a wide shared-expert
    up/gate projection), which is the dominant transient that lives next to
    the pools during a collection forward.
    """
    import transformers

    in_counts: dict = {}
    widest = 0
    for m in block.modules():
        w = getattr(m, "weight", None)
        if not isinstance(w, torch.Tensor) or w.dim() != 2:
            continue
        if type(m) == transformers.pytorch_utils.Conv1D:
            in_f, out_f = int(w.shape[0]), int(w.shape[1])
        elif isinstance(m, torch.nn.Linear):
            in_f, out_f = int(w.shape[1]), int(w.shape[0])
        else:
            continue
        in_counts[in_f] = in_counts.get(in_f, 0) + 1
        widest = max(widest, out_f)
    hidden = max(in_counts, key=lambda k: in_counts[k]) if in_counts else None
    return widest, hidden


def _working_allowance_bytes(block, pool, batch_size: int) -> int:
    """Simultaneous-transient allowance for one collection/tune batch."""
    """
    First principles: at any instant during a block forward, the transients
    sitting next to the pools are (a) the batch IO on the compute device --
    the staged input batch plus the produced output batch (2 generations of
    ``batch_size`` pool samples), and (b) ONE module's activation buffer, the
    widest projection's ``tokens x out_features`` in the block's dominant
    dtype (in+out buffers -> x2). The earlier flat 3 GiB charge was
    census-derived for one config; this derives the same quantity from the
    block and pool actually being placed. Over-charging is the safe
    direction (less consolidation); the census rounds measured the real
    simultaneous set at ~4.6 GiB worst case on hy3, which this formula
    reproduces within ~20%.
    """
    try:
        per_chain = _tensor_bytes(pool)
        n_chunks = max(_pool_chunk_count(pool), 1)
        batch_bytes = int(per_chain * min(int(batch_size), n_chunks) / n_chunks)
        need = 2 * batch_bytes
        widest, hidden = _widest_out_and_hidden(block)
        sample_numel = 0
        if isinstance(pool, (list, tuple)) and pool:
            for v in pool:
                if isinstance(v, torch.Tensor):
                    sample_numel = int(v.numel())
                    break
        if widest and hidden and sample_numel:
            tokens = sample_numel * min(int(batch_size), n_chunks) / hidden
            esize = _dominant_param_esize(block)
            need += int(tokens * widest * esize * 2)
        return max(need, _WORKING_FLOOR_BYTES)
    except Exception as e:  # pragma: no cover - placement must never break quantization
        logger.warning(
            "[calib-data-device] working-set estimate failed for %s (%s); using %.2fGiB floor",
            type(block).__name__,
            e,
            _WORKING_FLOOR_BYTES / 2**30,
        )
        return _WORKING_FLOOR_BYTES


def placement_need_bytes(block, pool, batch_size: int, iters: int = 0, primary: str = None) -> int:
    """Candidate-device working-set need for the placement gates, first principles.

    The earlier port reused ``estimate_tuning_block_mem`` (mapped-placement
    card-0 accounting), whose MoE term prices a hy3 block at ~343GiB: every
    module's batch activation counted with a x2 grad multiplier stacked with
    the x6 routing fudge, all assumed simultaneously live. That vetoed every
    primary residency for every MoE block. An intermediate model additionally
    charged ``2 * pool bytes`` as a "window-2 retention" term -- but the census
    rounds later showed that retention IS the in+out pool pair itself, which
    the gates already count explicitly (``total`` + ``reserved_bytes`` in the
    consolidation gate, ``pool_bytes`` + ``occupied_bytes`` in the resolve
    gate). A first revision took that reasoning too far and dropped the term
    from the RESOLVE gate as well -- but there it encodes the MEASURED
    collection window (retained batch outputs + route caches ~= 2x pool
    bytes on a single-compute-device MoE block), and dropping it let a
    block-2 single plan through that OOMed on the first server run. The
    term is restored, making the need strictly >= the validated formula;
    the computed allowance below is the additional, tune-loop-accurate
    part. Total charge:

    - the computed batch working allowance (``_working_allowance_bytes``:
      2 batch IO generations + the widest projection's activation buffer,
      derived from this block and pool -- NOT a flat constant);
    - at iters>0 only, the per-parameter tuning state that materializes on
      each weight's home device at wrap time (fp32 value + fp32 grad +
      best-params snapshot + bf16 copy = 14 B/param), charged for parameters
      homed on the candidate device alone -- peers host their own state and
      pay nothing. Wrap-phase coexistence (pools + state) is what makes this
      term real even though state materializes after collection;
    - the flat 0.25 GiB allocator reserve (a fragmentation pad does not scale
      with card size; the free-memory probe is the primary capacity signal).
    """
    try:
        need = _working_allowance_bytes(block, pool, batch_size) + _RESERVE_BYTES
        # MEASURED collection-window term (window~=2 chain retention + the
        # linear_loop route caches observed in the block censuses: on a
        # single-compute-device iters=0 MoE block the live transients ran
        # to ~3x pool bytes -- 8.25 GiB of retained batch-shaped outputs
        # plus 4 GiB of selected-hidden next to a 4 GiB pool). An earlier
        # revision dropped this term as a "double count" and the first
        # server run OOMed exactly there (block-2 single plan on the
        # primary); the computed allowance models the TUNE loop's
        # transients, not the collection forward's. Keeping the term makes
        # this need strictly >= the previously validated formula, so every
        # placement decision can only be equal or more conservative.
        need += 2 * _tensor_bytes(pool)
        if iters > 0 and block is not None and primary is not None:
            state_bytes = sum(p.numel() for p in block.parameters() if str(p.device) == str(primary)) * 14
            need += state_bytes
        return int(need)
    except Exception as e:  # pragma: no cover - placement must never break quantization
        logger.warning(
            "[calib-data-device] working-set estimate failed for %s (%s); using %.2fGiB floor",
            type(block).__name__,
            e,
            _WORKING_FLOOR_BYTES / 2**30,
        )
        return _WORKING_FLOOR_BYTES + _RESERVE_BYTES


def resolve_placement_for_pool(
    pool,
    chains: int,
    primary: str,
    candidate_devices: Sequence[str],
    block=None,
    batch_size: int = 8,
    mode: str = "auto",
    iters: int = 0,
    consumer: str = None,
    config=None,
) -> Optional[PoolPlacement]:
    """Resolve placement from a live pool object (orchestrator entry point).

    ``primary`` must be the lane's actual cache device: a CPU primary
    (``low_gpu_mem_usage``) deactivates the policy so the two mechanisms never
    fight. ``chains`` doubles the byte demand when a second (quantized-input)
    pool of the same size will also be produced for the block. ``block`` feeds
    the per-block working-set estimate. ``consumer`` (iters>0 lane) retargets
    the single-device preference to the tune consumer device; see
    :func:`resolve_pool_placement`.
    """
    from auto_round.utils.device import probe_usable_bytes

    pool_bytes = _tensor_bytes(pool) * max(int(chains), 1)
    n_chunks = _pool_chunk_count(pool)
    if n_chunks <= 0 or pool_bytes <= 0:
        return None
    consumer = str(consumer) if consumer is not None else None
    by_dev, _ = _bytes_by_device(pool)
    # occupied = pool bytes already resident on the SINGLE-DEVICE candidate
    # (consumer when retargeted, else the cache primary). Charging it for
    # the primary too is what stops the block-N single plan from ignoring
    # the input pool the block-N-1 single plan left sitting there (the
    # block-2 OOM: 4 GiB uncharged). The q-input pool mirrors the fp
    # pool's placement (same chunk layout), so resident bytes scale with
    # the chain multiplier.
    occupied = int(by_dev.get(str(consumer if consumer is not None else primary), 0) * max(int(chains), 1))
    try:
        plan = resolve_pool_placement(
            pool_bytes,
            n_chunks,
            primary,
            placement_need_bytes(block, pool, batch_size, iters=iters, primary=consumer or primary),
            candidate_devices,
            probe_usable_bytes,
            mode=mode,
            consumer=consumer,
            occupied_bytes=occupied,
            peer_state_bytes=_state_bytes_by_device(block) if iters > 0 else None,
            activation_bytes=_activation_bytes_for(block, pool, batch_size, iters, config) if iters > 0 else None,
        )
    except Exception as e:  # pragma: no cover - placement must never break quantization
        logger.warning("[calib-data-device] placement resolve failed (%s); keeping single-device behavior", e)
        return None
    return plan
