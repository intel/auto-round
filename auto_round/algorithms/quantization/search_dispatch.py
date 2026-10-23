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

"""Run weight-local quantization searches in parallel, one worker per weight device.

Some quantization searches only read the module's own weight (plus small
per-module statistics such as the imatrix) and never touch calibration
activations. When the block's weights are sharded across several GPUs such
searches can run directly on the device that already hosts each weight: no
copies, no mirrors, and the devices naturally share the work. This module
provides the generic grouping/execution helpers used by the wrapper and RTN
search loops.

Searches that depend on activations (e.g. the AWQ clip search) are not
weight-local and must stay on their serial paths.
"""

import contextlib
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import torch

import auto_round.envs as envs
from auto_round.logger import logger
from auto_round.utils.device import is_hpu_lazy_mode, probe_usable_bytes
from auto_round.utils.oom import dump_oom_tensor_census_  # re-exported for existing call sites
from auto_round.utils.perf_counters import perf_counters_enabled
from auto_round.wrapper import WrapperLinear


def group_items_by_device(
    items: Iterable[Any],
    device_of: Callable[[Any], str | None],
    none_key: str = "uncategorized",
) -> "OrderedDict[str, list[tuple[int, Any]]]":
    """Group ``(index, item)`` pairs by ``device_of(item)`` preserving input order.

    Args:
        items: Iterable of items to group.
        device_of: Callable mapping an item to a device key (str or torch.device).
        none_key: Bucket label for items whose device is ``None``.

    Returns:
        ``OrderedDict[device_key, list[(index, item)]]`` with devices in first-seen
        order and items in original order within each device.
    """
    groups = OrderedDict()
    for idx, item in enumerate(items):
        key = device_of(item)
        if key is None:
            key = none_key
        groups.setdefault(key, []).append((idx, item))
    return groups


@contextlib.contextmanager
def _null_ctx() -> Iterator[None]:
    yield


def _device_worker_ctx(key: str) -> contextlib.AbstractContextManager:
    """Launch-context for a worker thread pinned to one accelerator family.

    cuda/xpu workers get the family's device context so ops land on the weight
    device even when the ambient current device differs; anything else (cpu,
    unknown families, or a torch build without that context manager) runs
    bare -- explicit-device ops (``.to(dev)`` / stacked inputs) still land
    correctly, the context only pins implicit device selection.
    """
    try:
        dev = torch.device(key)
    except (ValueError, RuntimeError):  # unparsable key: run bare
        return _null_ctx()
    if dev.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.device(dev)
    if dev.type == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "device"):
        try:
            return torch.xpu.device(dev)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("[search] xpu device context unavailable (%s); running unpinned", e)
            return _null_ctx()
    return _null_ctx()


def run_items_by_device(
    groups: "OrderedDict[str, list[tuple[int, Any]]]",
    fn: Callable[[int, Any], Any],
    use_cuda_ctx: bool = True,
) -> None:
    """Run ``fn(item)`` for every grouped item, one worker thread per device group.

    A single group is executed inline on the calling thread. Any exception raised
    by ``fn`` is re-raised on the calling thread after all workers joined
    (fail-visible: search failures are never silently swallowed).

    Args:
        groups: Output of :func:`group_items_by_device`.
        fn: Callable executed as ``fn(index, item)``; must be safe to run
            concurrently across groups (items within one group run serially on
            their worker).
        use_cuda_ctx: Wrap each accelerator worker in its family's device
            context (cuda/xpu) so ops land on the weight device even when the
            ambient current device differs; other families run bare.
    """
    if len(groups) <= 1:
        for _idx, item in next(iter(groups.values()), []):
            fn(_idx, item)
        return

    first_error = []
    error_lock = threading.Lock()
    threads = []

    def _worker(device_key: str, indexed_items: list[tuple[int, Any]]) -> None:
        try:
            key = str(device_key)
            with _device_worker_ctx(key) if use_cuda_ctx else _null_ctx():
                for _idx, item in indexed_items:
                    fn(_idx, item)
        except BaseException as exc:  # noqa: B036 - re-raised below, never swallowed
            with error_lock:
                if first_error:
                    logger.error(
                        "[batched-search] worker %s also failed (%r); raising the first failure",
                        device_key,
                        exc,
                    )
                else:
                    first_error.append(exc)

    for device_key, indexed_items in groups.items():
        if not indexed_items:
            continue
        t = threading.Thread(target=_worker, args=(device_key, indexed_items), name=f"search-dispatch-{device_key}")
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    if first_error:
        raise first_error[0]


def cross_device_search_disabled() -> bool:
    """Kill switch for running batched searches on idle devices."""
    return bool(envs.AR_DISABLE_CROSS_DEVICE_SEARCH)


def pick_search_worker_devices(
    working_set_bytes: int, home_device: str | None = None, margin_bytes: int = 512 * 2**20
) -> list[str]:
    """Viable worker devices for a batched search chunk, in device-index order.

    The searches are weight-local (weight + imatrix only), so the home device
    is preferred whenever it has headroom for the chunk's working set;
    devices that hold no model weights at all serve as overflow for chunks
    whose home device is too full (e.g. --device_map 0 with more cards
    visible). The home device participates like any other candidate; when
    nothing fits, the home device is returned so the caller keeps today's
    behavior and relies on the per-chunk OOM fallback.

    Device families enumerate through their own APIs (cuda, xpu, hpu, mps),
    so offload engages wherever the free-memory probe works. HPU lazy mode
    is the exception: lazy tensors execute on per-thread device contexts, so
    a stacked search spread onto worker threads reaches the write-back's
    host copy with lazy tensors that have no storage attached
    (``data_ptr()`` fails). HPU lazy searches therefore stay on the home
    device -- stacked batching still runs there, inline on the calling
    thread; compile mode (the eager pipeline) spreads like any other family.
    """
    if cross_device_search_disabled():
        return [home_device] if home_device is not None else []
    home_type = str(home_device).split(":")[0] if home_device else ""
    if home_type == "hpu" and is_hpu_lazy_mode():
        return [home_device] if home_device is not None else []
    try:
        from auto_round.utils.device_manager import get_ar_device

        count = get_ar_device(home_type).device_count() if home_type else 0
    except Exception as e:  # pragma: no cover - exotic backends
        logger.warning("[search] device enumeration failed (%s); keeping searches on the home device", e)
        count = 0
    if count == 0:
        return [home_device] if home_device is not None else []
    viable = []
    for idx in range(count):
        key = f"{home_type}:{idx}"
        free = probe_usable_bytes(key)
        if free is None or free - margin_bytes < working_set_bytes:
            continue
        viable.append(key)
    if not viable:
        return [home_device] if home_device is not None else []
    return viable


_ENGAGED_LOGGED = set()


def log_engaged_once(label: str) -> None:
    """Log the batching engagement once per process (INFO, no counters).

    Detailed per-block counters are intentionally not emitted here; they belong
    to the perf-counter infrastructure of the parallel-tuning work.
    """
    if label in _ENGAGED_LOGGED:
        return
    _ENGAGED_LOGGED.add(label)
    logger.info("[batched-search] %s: running weight-local searches with one worker per device", label)


def batched_search_disabled() -> bool:
    """Kill switch for the batched search machinery (stacking + per-device workers)."""
    return bool(envs.AR_DISABLE_BATCHED_SEARCH)


def _wrap_batch_device_of(inputs: tuple[Any, ...]) -> str:
    return str(inputs[0].device)


def _fn_key(fn: Callable[..., Any]) -> str | tuple[Any, ...]:
    """Stable, safe key term for a resolved search callable.

    Plain functions key by identity location (same function = same behavior);
    ``functools.partial`` objects key by their visible captures; anything else
    (e.g. a per-call closure, whose captures are not introspectable) keys by
    ``repr`` so it never merges with a lookalike. Merging two different searches
    into one stacked batch would silently apply the wrong math to one side, so
    unknown callables always stay separate.
    """
    import functools

    if isinstance(fn, functools.partial):
        kwargs = tuple(sorted(fn.keywords.items(), key=lambda kv: str(kv[0])))
        try:
            qualname = fn.func.__qualname__
            module = fn.func.__module__
        except AttributeError as e:  # pragma: no cover - exotic callables
            logger.debug("[batched-search] partial target %r has no qualname (%s); keying by repr", fn.func, e)
            return repr(fn)
        return ("partial", module, qualname, fn.args, kwargs)
    if callable(fn):
        if getattr(fn, "__closure__", None):
            # a per-call closure shares (module, qualname) with its siblings
            # from the same factory; its captures are not introspectable, so
            # key by repr and never merge lookalikes into one batch
            return ("closure", repr(fn))
        try:
            return ("fn", fn.__module__, fn.__qualname__)
        except AttributeError as e:  # pragma: no cover - exotic callables
            logger.debug("[batched-search] callable %r has no qualname (%s); keying by repr", fn, e)
            return repr(fn)
    return repr(fn)


_NON_STACKABLE_SEARCH_NAMES = {"search_nvfp4_scale", "opt_rtn_fast_nvfp4"}


def _search_fn_stackable(fn: Callable[..., Any]) -> bool:
    """Whether a resolved search/quant callable is safe to call on a stacked batch.

    The nv-fp4 init-scale search internally flattens its scale buffer
    (``nv_fp4`` returns ``[numel/group, 1]``) while its loss/mask keep the
    batch dims, so a stacked call faults with a mask/tensor shape mismatch;
    the OptRTN nv quant func routes through the same search. Those callables
    must stay per-module until the search itself is made batch-aware.
    """
    fn = getattr(fn, "_torchdynamo_orig_callable", None) or fn
    fn = getattr(fn, "func", None) or fn  # unwrap functools.partial
    return getattr(fn, "__name__", "") not in _NON_STACKABLE_SEARCH_NAMES


def _wrap_batch_key(inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    weight, _data_type, bits, imatrix_raw, thresh, search_fn = inputs
    return (
        str(weight.device),
        tuple(weight.shape),
        str(weight.dtype),
        bits,
        float(thresh),
        imatrix_raw is not None,
        _fn_key(search_fn),
    )


def _materialize_wrap_imatrix(chunk: list[WrapperLinear], stacked_w: torch.Tensor) -> torch.Tensor:
    """Chunk-time imatrix stack from the staged RAW column vectors.

    None means uniform importance: an all-None chunk (the default tuning
    lane) becomes one ``ones_like`` allocation instead of N expanded copies.
    """
    raws = [w._deferred_search_inputs[3] for w in chunk]
    if raws[0] is None:
        return torch.ones_like(stacked_w)
    from auto_round.data_type.utils import reshape_imatrix_for_weight

    return torch.stack(
        [
            reshape_imatrix_for_weight(r, w._deferred_search_inputs[0], w.orig_layer.group_size)
            for r, w in zip(raws, chunk)
        ]
    )


_WRAP_BATCH_MAX_ELEMS = 2**28  # ~1 GiB fp32 stacked weights per batched call (matches the NeUQI expert batching)


def _wrap_batch_max_elems() -> "int | None":
    """Element budget per stacked batch; AR_SEARCH_BATCH_GB overrides in GiB of fp32 weights.

    Override-only: unset means the batch size is capacity-driven (the probed
    free VRAM against the search's transient footprint), so roomy devices use
    proportionally larger batches.  Invalid values raise (the env parser's
    ValueError) instead of silently falling back to a default budget.
    """
    gb = envs.AR_SEARCH_BATCH_GB
    if gb is not None:
        return max(int(gb * 2**30 // 4), 1)
    return None


def _is_neuqi_search(fn) -> bool:
    """Whether a staged search/quant callable routes through the NeUQI grids."""
    fn = getattr(fn, "_torchdynamo_orig_callable", None) or fn
    fn = getattr(fn, "func", None) or fn  # unwrap functools.partial
    return getattr(fn, "__module__", "").endswith("neuqi")


def search_transient_bytes(search_fn, total_elements: int) -> int:
    """Peak VRAM one stacked search call needs, beyond the stacked input.

    The grid searches materialize an fp32 copy chain over the batch (inputs,
    loss temporaries): ~16 B per element, the expanded imatrix included via
    the caller's element count.  NeUQI-class searches add their internal
    sweep budget ONCE PER CALL: the sweeps cap every [chunk, candidates,
    group] temporary at ``_MAX_TMP_ELEMS`` elements regardless of the grid
    sizes (a wider grid only processes more row-chunks), and a handful of
    those temporaries are live at once -- the budget does not scale with the
    number of stacked modules.
    """
    total = total_elements * 4 * 4
    if _is_neuqi_search(search_fn):
        from auto_round.data_type.neuqi import _MAX_TMP_ELEMS

        total += _MAX_TMP_ELEMS * 4 * 8
    return total


def search_sweep_budget(search_fn) -> int:
    """The per-call (chunk-size-independent) part of a search's transient."""
    if _is_neuqi_search(search_fn):
        from auto_round.data_type.neuqi import _MAX_TMP_ELEMS

        return _MAX_TMP_ELEMS * 4 * 8
    return 0


def _plan_capacity_chunks(entries: list) -> list:
    """Per-device batch caps with count-balanced work spread.

    ``entries`` items are ``(home_key, per_module_bytes, sweep_bytes, max_modules,
    modules)``; the returned items are ``(worker_key, home_key, chunk_modules)``.

    Free VRAM is probed on every candidate device (all visible cuda devices
    through ``pick_search_worker_devices``; kill switch honored).  A device's
    batches are sized so ONE chunk's transient -- the per-module copy chain
    times the module count plus the per-call sweep budget -- fits in half its
    probed free VRAM: chunks on a device run serially (one worker thread per
    executing device), so only each single chunk must fit, and roomy devices
    take proportionally larger batches than tight ones.  Each module
    stays on its home while the home is a candidate and holds less than its
    fair share of the work; the overflow goes to the least-assigned candidate.
    A placement that already spans every device therefore moves nothing, a
    tight home keeps chewing through its share in small batches, and
    concentrated homes spill onto idle devices.  An unprobeable home falls
    back to fixed-width chunks at home.

    Estimates guide this planner only: the execute-time headroom guards and
    the OOM ladder own real overflow.
    """
    usable_cache: OrderedDict[str, int | None] = OrderedDict()

    def _usable(dev: str):
        if dev not in usable_cache:
            usable_cache[dev] = probe_usable_bytes(dev)
        return usable_cache[dev]

    grand = sum(len(modules) for _home, _per, _sweep, _max_n, modules in entries)
    counts: OrderedDict[str, int] = OrderedDict()

    plans: list = []
    for home, per, sweep, max_n, modules in entries:
        workers = pick_search_worker_devices(per + sweep, home_device=home)
        if home in workers:
            cands = [home] + [w for w in workers if w != home]
        else:
            cands = list(workers) if workers else [home]
        n_max = {}
        for d in cands:
            u = _usable(d)
            if u is None:
                n_max[d] = max_n
            else:
                n_max[d] = max(1, min(max_n, (u // 2 - sweep) // max(per, 1)))
            counts.setdefault(d, 0)
        share = max(1, -(-grand // len(cands)))  # a device's fair share of the work
        alloc: OrderedDict[str, list] = OrderedDict((d, []) for d in cands)
        for m in modules:
            if home in cands and counts.get(home, 0) < share:
                d = home  # resident modules stay put while home holds its share
            else:
                d = min(cands, key=lambda k: (counts.setdefault(k, 0), cands.index(k)))
            alloc[d].append(m)
            counts[d] = counts.get(d, 0) + 1
        for d in cands:
            cap = n_max[d]
            for start in range(0, len(alloc[d]), cap):
                plans.append((d, home, alloc[d][start : start + cap]))
    return plans


def run_batched_wrap_search(deferred_wrappers: list[WrapperLinear], max_batch: int | None = None) -> bool:
    """Run deferred weight-local wrap searches on stacked same-shape batches.

    Wrappers stage ``(weight_reshape, data_type, bits, imatrix, q_scale_thresh,
    search_fn)`` tuples in ``_deferred_search_inputs``, where ``search_fn`` is the
    exact callable the per-module path would have invoked (resolved at wrap time,
    so future dispatch changes -- e.g. alternative optimized searches -- travel
    with the module automatically). Modules whose staged key
    ``(device, shape, weight dtype, bits, threshold, search_fn)`` matches are
    stacked along a leading dim and searched with ONE ``search_fn`` call: the
    per-row math is unchanged (row-independent reductions over the last dim), so
    results are bit-identical to the per-module path while python/launch overhead
    drops by the batch factor. Batches are capped by ``max_batch`` and by a VRAM
    budget on the home device. Chunks are then planned centrally: every chunk
    starts on the device that already holds its weights, and chunks rebalance
    onto under-loaded viable devices so the search uses every visible device
    with headroom. Execution uses one worker thread per executing device --
    each device is owned by exactly one thread, keeping the searches'
    per-grid-step host syncs private. Singleton groups take the identical
    per-module call on the calling thread.

    Returns True when the inputs were consumed; False when batching is disabled
    by AR_DISABLE_BATCHED_SEARCH (the caller then runs the searches per module).
    """
    if batched_search_disabled():
        return False
    if not deferred_wrappers:
        return False

    device_groups = OrderedDict()
    for w in deferred_wrappers:
        inputs = w._deferred_search_inputs
        if inputs is None:
            raise RuntimeError(
                f"{type(w).__name__} was queued for batched wrap search without staged inputs; "
                "the defer_search flag never reached its search init"
            )
        dev = _wrap_batch_device_of(inputs)
        device_groups.setdefault(dev, []).append(w)

    singletons: list[tuple[str, WrapperLinear]] = []
    plan_entries: list[tuple[str, int, int, int, list[WrapperLinear]]] = []
    for dev, wrappers in device_groups.items():
        by_key = OrderedDict()
        for w in wrappers:
            by_key.setdefault(_wrap_batch_key(w._deferred_search_inputs), []).append(w)
        for group in by_key.values():
            if len(group) < 2 or not _search_fn_stackable(group[0]._deferred_search_inputs[5]):
                singletons.extend((dev, w) for w in group)
                continue
            inputs0 = group[0]._deferred_search_inputs
            elements = inputs0[0].numel() * (2 if inputs0[3] is not None else 1)
            per_module = elements * 4 * 4  # the fp32 copy chain per module
            sweep = search_sweep_budget(inputs0[5])
            max_n = 1024 if max_batch is None else max(1, max_batch)
            elem_cap = _wrap_batch_max_elems()
            if elem_cap is not None:
                max_n = min(max_n, max(1, elem_cap // max(elements, 1)))
            plan_entries.append((dev, per_module, sweep, max_n, group))

    # capacity-weighted planning: chunks form directly on the device that
    # runs them, sized to its tracked headroom (see _plan_capacity_chunks)
    planned = _plan_capacity_chunks(plan_entries)
    chunks = [(home, chunk) for _worker, home, chunk in planned]

    # Phase 3: one worker thread per executing device, and each device is
    # owned by exactly one thread. The grid searches end every step with a
    # host sync, so two threads sharing one device's default stream convoy
    # each other; exclusive buckets keep every sync private.
    buckets: OrderedDict[str, list[tuple[str, list[WrapperLinear]]]] = OrderedDict()
    for worker, home, chunk in planned:
        buckets.setdefault(worker, []).append((home, chunk))

    stats = {wk: {"modules": 0, "batches": 0, "singletons": 0} for wk in buckets}

    @torch.no_grad()
    def _run_one(wrapper: WrapperLinear) -> None:
        inputs = wrapper._deferred_search_inputs
        if inputs is None:
            return
        weight, _data_type, bits, imatrix_raw, _thresh, search_fn = inputs
        from auto_round.data_type.utils import reshape_imatrix_for_weight

        imatrix = reshape_imatrix_for_weight(imatrix_raw, weight, wrapper.orig_layer.group_size)
        wrapper.finalize_batched_search(search_fn(weight, bits, imatrix))

    @torch.no_grad()
    def _chunk_transient(chunk: list[WrapperLinear]) -> int:
        inputs0 = chunk[0]._deferred_search_inputs
        elements = inputs0[0].numel() * (2 if inputs0[3] is not None else 1)
        return search_transient_bytes(inputs0[5], elements * len(chunk))

    @torch.no_grad()
    def _run_chunk_stacked(home_key: str, chunk: list[WrapperLinear], worker_key: str) -> None:
        """One stacked search call, halved when the executing device lacks headroom.

        Plan-time estimates can drift; this check runs at the moment of truth.
        """
        if len(chunk) > 1:
            usable = probe_usable_bytes(worker_key)
            if usable is not None and usable - 512 * 2**20 < _chunk_transient(chunk):
                mid = len(chunk) // 2
                _run_chunk_stacked(home_key, chunk[:mid], worker_key)
                _run_chunk_stacked(home_key, chunk[mid:], worker_key)
                return
        inputs0 = chunk[0]._deferred_search_inputs
        _bits = inputs0[2]
        search_fn = inputs0[5]
        if worker_key != home_key:
            # never carry a home-compiled closure onto another device:
            # dynamo guards are device-pinned (recompile zoo) -- the
            # eager original (when exposed) is device-agnostic
            _eager_fn = getattr(search_fn, "_torchdynamo_orig_callable", None)
            if _eager_fn is not None:
                search_fn = _eager_fn
            # stack directly on the worker so the home device never holds the
            # assembled batch (its modules' weights stay resident there)
            stacked_w = torch.stack([w._deferred_search_inputs[0].to(worker_key, non_blocking=True) for w in chunk])
        else:
            stacked_w = torch.stack([w._deferred_search_inputs[0] for w in chunk])
        stacked_im = _materialize_wrap_imatrix(chunk, stacked_w)
        if worker_key != home_key and torch.is_tensor(stacked_im) and stacked_im.device != stacked_w.device:
            # the expanded imatrix stack materializes from home-resident staged
            # columns; move it beside the stacked weights before the search
            stacked_im = stacked_im.to(worker_key, non_blocking=True)
        try:
            results = search_fn(stacked_w, _bits, stacked_im)
        except torch.OutOfMemoryError:
            logger.warning(
                "[batched-search] stacked wrap search OOM (%d modules); finishing this chunk "
                "per-module (shrink batches with AR_SEARCH_BATCH_GB or disable with "
                "AR_DISABLE_BATCHED_SEARCH=1)",
                len(chunk),
            )
            dump_oom_tensor_census_("wrap search")
            for w in chunk:
                w._run_deferred_search_now()
            stats[worker_key]["singletons"] += len(chunk)
            return
        if worker_key != home_key:
            if torch.is_tensor(results):
                results = results.to(home_key)
            else:
                results = [r.to(home_key) if torch.is_tensor(r) else r for r in results]
        for w, res in zip(chunk, results):
            w.finalize_batched_search(res)
        stats[worker_key]["batches"] += 1
        stats[worker_key]["modules"] += len(chunk)

    @torch.no_grad()
    def _execute(worker_key: str, items: list[tuple[str, list[WrapperLinear]]]) -> None:
        _t0 = time.perf_counter()
        for home_key, chunk in items:
            _run_chunk_stacked(home_key, chunk, worker_key)
        stats[worker_key]["wall"] = time.perf_counter() - _t0

    # singleton groups and non-stackable searches run per module on the
    # calling thread, each on its own weight's device
    for _dev, w in singletons:
        _run_one(w)
    _n_single = len(singletons)

    if len(buckets) > 1:
        pairs = list(buckets.items())
        keyed = group_items_by_device(pairs, device_of=lambda p: p[0])
        run_items_by_device(keyed, lambda _idx, p: _execute(p[0], p[1]))
    elif buckets:
        wk, items = next(iter(buckets.items()))
        _execute(wk, items)
    _t_total = sum(st.get("wall", 0.0) for st in stats.values())
    _n_mod = sum(st["modules"] for st in stats.values()) + _n_single
    _n_batch = sum(st["batches"] for st in stats.values())
    from auto_round.utils.pool_placement import _short_device_key

    per_device = (
        "{" + ", ".join(f"'{_short_device_key(dev)}': {st.get('wall', 0.0):.2f}" for dev, st in stats.items()) + "}"
    )
    _n_offloaded = sum(1 for worker, home, _chunk in planned if worker != home)
    if perf_counters_enabled():
        logger.info(
            "[batched-search] wrap search: %d modules, %d batch + %d singleton search calls"
            + (f" ({_n_offloaded} chunk(s) offloaded to idle devices)" if _n_offloaded else "")
            + ", %.2fs device-wall (%.2fs summed) walls %s",
            _n_mod,
            _n_batch,
            _n_single,
            max(st.get("wall", 0.0) for st in stats.values()) if stats else 0.0,
            _t_total,
            per_device,
        )
    return True


def run_batched_neuqi_anchors(deferred_wrappers: list) -> bool:
    """Run deferred NeUQI grid anchors on stacked same-shape batches.

    The tuning-path anchor search (``neuqi_anchor_margins``) is weight-local
    and per-group, so same-shape modules stack into one call with bit-identical
    margins.  Chunking follows the AR_SEARCH_BATCH_GB element budget; worker
    assignment uses the same home-first balanced planner and exclusive
    one-thread-per-device execution as ``run_batched_wrap_search`` (the anchor
    grids contain no host syncs, but exclusivity keeps the dispatch uniform);
    a stacked call that OOMs finishes its chunk per module.

    Returns True when the anchors were consumed; False when batching is
    disabled by AR_DISABLE_BATCHED_SEARCH (the caller runs each anchor per
    module).
    """
    if batched_search_disabled():
        return False
    if not deferred_wrappers:
        return True

    from auto_round.data_type.neuqi import _prepare_neuqi_qw, neuqi_anchor_margins

    def _staged(w):
        staged = getattr(w, "_deferred_neuqi_anchor", None)
        if staged is None:
            raise RuntimeError(
                f"{type(w).__name__} was queued for the batched NeUQI anchor without staged inputs; "
                "the defer_search flag never reached the anchor init"
            )
        return staged

    def _anchor_key(staged):
        weight, bits, group_size, imatrix, sym = staged
        return (str(weight.device), tuple(weight.shape), bits, group_size, sym, imatrix is not None)

    def _anchor_chunk_transient(staged_list) -> int:
        from auto_round.data_type.neuqi import _MAX_TMP_ELEMS

        # the anchor runs the NeUQI sweeps on the stacked batch: an fp32 copy
        # chain over the batch (stacked and flattened inputs, importance
        # weights, and the sweeps' full-size intermediates -- about sixteen
        # weight-sized fp32 buffers are live at peak) that scales with the
        # module count, plus the sweeps' internal temporary budget, which is
        # capped at _MAX_TMP_ELEMS elements regardless of the grid sizes and
        # does not scale with the number of stacked modules
        staged0 = staged_list[0]
        per = staged0[0].numel() * 4 * 16
        if staged0[3] is not None:
            per += staged0[0].numel() * 4
        return per * len(staged_list) + _MAX_TMP_ELEMS * 4 * 8

    singletons: list = []
    plan_entries: list[tuple[str, int, int, int, list]] = []
    by_key: OrderedDict[tuple, list] = OrderedDict()
    for w in deferred_wrappers:
        by_key.setdefault(_anchor_key(_staged(w)), []).append(w)
    from auto_round.data_type.neuqi import _MAX_TMP_ELEMS

    for group in by_key.values():
        if len(group) < 2:
            singletons.extend(group)
            continue
        staged0 = _staged(group[0])
        # same element accounting as the wrap lane: the expanded imatrix
        # reaches the weight's size, so it counts against the override budget
        elements_per_module = staged0[0].numel() * (2 if staged0[3] is not None else 1)
        per_module = staged0[0].numel() * 4 * 16 + (staged0[0].numel() * 4 if staged0[3] is not None else 0)
        sweep = _MAX_TMP_ELEMS * 4 * 8
        max_n = 1024
        elem_cap = _wrap_batch_max_elems()
        if elem_cap is not None:
            max_n = min(max_n, max(1, elem_cap // max(elements_per_module, 1)))
        plan_entries.append((str(staged0[0].device), per_module, sweep, max_n, group))

    # capacity-weighted planning: anchor chunks form directly on the device
    # that runs them, sized to its tracked headroom (see _plan_capacity_chunks)
    planned = _plan_capacity_chunks(plan_entries)
    staged_all = [[_staged(w) for w in chunk] for _worker, _home, chunk in planned]

    buckets: OrderedDict[str, list] = OrderedDict()
    for worker, _home, chunk in planned:
        buckets.setdefault(worker, []).append(chunk)

    stats = {wk: {"modules": 0, "batches": 0, "singletons": 0} for wk in buckets}

    @torch.no_grad()
    def _run_anchor_stack(chunk: list, worker_key: str):
        """One stacked anchor call, split in half when the executing device
        lacks headroom for the chunk's real footprint at call time (estimates
        guide the planner; this check runs at the moment of truth)."""
        staged = [_staged(w) for w in chunk]
        need = _anchor_chunk_transient(staged)
        usable = probe_usable_bytes(worker_key)
        if len(chunk) > 1 and usable is not None and usable - 512 * 2**20 < need:
            mid = len(chunk) // 2
            first = _run_anchor_stack(chunk[:mid], worker_key)
            second = _run_anchor_stack(chunk[mid:], worker_key)
            return torch.cat([first[0], second[0]]), torch.cat([first[1], second[1]])
        _bits, _gs, _sym = staged[0][1], staged[0][2], staged[0][4]
        stacked_w = torch.stack([s[0] for s in staged]).reshape(-1, _gs)
        stacked_im = None
        if staged[0][3] is not None:
            # fully-expanded per-module importance weights stack into the
            # anchor's own normalization unchanged (its reshape/expand is
            # a pass-through for an already full-size layout)
            stacked_im = torch.stack([_prepare_neuqi_qw(s[3], s[0], _bits, _gs) for s in staged]).reshape(-1, _gs)
        return neuqi_anchor_margins(stacked_w, _bits, _gs, stacked_im, worker_key, sym=_sym)

    @torch.no_grad()
    def _execute(worker_key: str, chunk_list: list) -> None:
        _t0 = time.perf_counter()
        for chunk in chunk_list:
            staged = [_staged(w) for w in chunk]
            homes = [str(s[0].device) for s in staged]
            _bits, _gs, _sym = staged[0][1], staged[0][2], staged[0][4]
            # the anchor grid works on a flat [groups, group_size] layout;
            # stacking along a leading module axis then flattening preserves
            # each group's rows (row-major), so per-module margins come back
            # as consecutive row blocks
            stacked_w = torch.stack([s[0] for s in staged]).reshape(-1, _gs)
            stacked_im = None
            if staged[0][3] is not None:
                # fully-expanded per-module importance weights stack into the
                # anchor's own normalization unchanged (its reshape/expand is
                # a pass-through for an already full-size layout)
                stacked_im = torch.stack([_prepare_neuqi_qw(s[3], s[0], _bits, _gs) for s in staged]).reshape(-1, _gs)
            try:
                wmin, wmax = _run_anchor_stack(chunk, worker_key)
            except torch.OutOfMemoryError:
                logger.warning(
                    "[batched-search] stacked NeUQI anchor OOM (%d modules); finishing this chunk "
                    "per-module (shrink batches with AR_SEARCH_BATCH_GB or disable with "
                    "AR_DISABLE_BATCHED_SEARCH=1)",
                    len(chunk),
                )
                dump_oom_tensor_census_("neuqi anchor")
                for w in chunk:
                    w._run_deferred_neuqi_anchor_now()
                stats[worker_key]["singletons"] += len(chunk)  # counted like the wrap lane's fallbacks
                continue
            wmin = wmin.reshape(len(chunk), -1)
            wmax = wmax.reshape(len(chunk), -1)
            for w, s, mn, mx in zip(chunk, staged, wmin, wmax):
                w._apply_neuqi_anchor(mn.to(s[0].device), mx.to(s[0].device))
            stats[worker_key]["batches"] += 1
            stats[worker_key]["modules"] += len(chunk)
        stats[worker_key]["wall"] = time.perf_counter() - _t0

    for w in singletons:
        w._run_deferred_neuqi_anchor_now()

    if len(buckets) > 1:
        pairs = list(buckets.items())
        keyed = group_items_by_device(pairs, device_of=lambda p: p[0])
        run_items_by_device(keyed, lambda _idx, p: _execute(p[0], p[1]))
    elif buckets:
        wk, items = next(iter(buckets.items()))
        _execute(wk, items)

    _t_total = sum(st.get("wall", 0.0) for st in stats.values())
    _n_single = len(singletons) + sum(st["singletons"] for st in stats.values())
    _n_mod = sum(st["modules"] for st in stats.values()) + _n_single
    _n_batch = sum(st["batches"] for st in stats.values())
    _n_offloaded = sum(1 for worker, home, _chunk in planned if worker != home)
    from auto_round.utils.pool_placement import _short_device_key

    per_device = (
        "{" + ", ".join(f"'{_short_device_key(dev)}': {st.get('wall', 0.0):.2f}" for dev, st in stats.items()) + "}"
    )
    if perf_counters_enabled():
        logger.info(
            "[batched-search] neuqi anchors: %d modules, %d batch + %d singleton search calls"
            + (f" ({_n_offloaded} chunk(s) offloaded to idle devices)" if _n_offloaded else "")
            + ", %.2fs device-wall (%.2fs summed) walls %s",
            _n_mod,
            _n_batch,
            _n_single,
            max(st.get("wall", 0.0) for st in stats.values()) if stats else 0.0,
            _t_total,
            per_device,
        )
    return True
