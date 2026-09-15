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

import torch

import auto_round.envs as envs
from auto_round.logger import logger
from auto_round.utils.device import probe_usable_bytes
from auto_round.utils.oom import dump_oom_tensor_census_  # re-exported for existing call sites


def group_items_by_device(items, device_of, none_key="uncategorized"):
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
def _null_ctx():
    yield


def _device_worker_ctx(key: str):
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
        except Exception:  # pragma: no cover - defensive
            return _null_ctx()
    return _null_ctx()


def run_items_by_device(groups, fn, use_cuda_ctx=True):
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

    def _worker(device_key, indexed_items):
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


def multigpu_search_disabled():
    """Kill switch for running batched searches on idle devices."""
    return bool(envs.AR_DISABLE_MULTIGPU_SEARCH)


def pick_search_worker_devices(working_set_bytes, home_device=None, margin_bytes=512 * 2**20):
    """Viable worker devices for a batched search chunk, in device-index order.

    The searches are weight-local (weight + imatrix only), so a chunk may run on
    ANY device with headroom for its transient working set -- including devices
    that hold no model weights at all (the zero-shot lane's idle GPUs). The home
    device participates only when it fits like any other candidate; when nothing
    fits, the home device is returned so the caller keeps today's behavior and
    relies on the per-chunk OOM fallback.

    CUDA-only today: the free-memory probe underneath is a cuda API, so
    non-cuda fleets simply get ``[home_device]`` (no offload, no crash).
    """
    if multigpu_search_disabled():
        return [home_device] if home_device is not None else []
    try:
        count = torch.cuda.device_count()
    except Exception:  # pragma: no cover - non-cuda builds
        count = 0
    if count == 0:
        return [home_device] if home_device is not None else []
    viable = []
    for idx in range(count):
        key = f"cuda:{idx}"
        free = probe_usable_bytes(key)
        if free is None or free - margin_bytes < working_set_bytes:
            continue
        viable.append(key)
    if not viable:
        return [home_device] if home_device is not None else []
    return viable


_ENGAGED_LOGGED = set()


def log_engaged_once(label):
    """Log the batching engagement once per process (INFO, no counters).

    Detailed per-block counters are intentionally not emitted here; they belong
    to the perf-counter infrastructure of the parallel-tuning work.
    """
    if label in _ENGAGED_LOGGED:
        return
    _ENGAGED_LOGGED.add(label)
    logger.info("[batched-search] %s: running weight-local searches with one worker per device", label)


def batched_search_disabled():
    """Kill switch for the batched search machinery (stacking + per-device workers)."""
    return bool(envs.AR_DISABLE_BATCHED_SEARCH)


def _wrap_batch_device_of(inputs):
    return str(inputs[0].device)


def _fn_key(fn):
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


def _search_fn_stackable(fn):
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


def _wrap_batch_key(inputs):
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


def _materialize_wrap_imatrix(chunk, stacked_w):
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


def _wrap_batch_max_elems():
    """Element budget per stacked batch; AR_SEARCH_BATCH_GB overrides in GiB of fp32 weights.

    Invalid values raise (the env parser's ValueError) instead of silently
    falling back to the default budget.
    """
    gb = envs.AR_SEARCH_BATCH_GB
    if gb is not None:
        return max(int(gb * 2**30 // 4), 1)
    return _WRAP_BATCH_MAX_ELEMS


def _batch_cap(group, device_key, max_batch):
    """Modules per stacked batch: explicit cap > free-VRAM probe > 64, capped by the element budget."""
    if max_batch is not None:
        return max(1, max_batch)
    inputs0 = group[0]._deferred_search_inputs
    # staged imatrix is the raw column; the chunk-time expansion reaches the
    # weight's full size, so the budget counts it at its expanded size
    elements_per_module = inputs0[0].numel() * (2 if inputs0[3] is not None else 1)
    # the search is bandwidth-bound: batches beyond ~1 GiB of stacked weights move
    # the same total bytes, so the fixed element budget only lowers transient VRAM
    elem_cap = max(1, _wrap_batch_max_elems() // max(elements_per_module, 1))
    probe_cap = 64
    usable = probe_usable_bytes(device_key)
    if usable is not None:
        per_module_bytes = elements_per_module * 4 * 4  # fp32 working set incl. temporaries
        probe_cap = max(1, min(1024, usable // 2 // max(per_module_bytes, 1)))
    return max(1, min(probe_cap, elem_cap))


def run_batched_wrap_search(deferred_wrappers, max_batch=None):
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
    budget on the group's device; groups on different devices run on one worker
    thread per device; singleton groups take the identical per-module call.

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

    stats = {dev: {"modules": 0, "batches": 0, "singletons": 0} for dev in device_groups}

    @torch.no_grad()
    def _run_one(wrapper):
        inputs = wrapper._deferred_search_inputs
        if inputs is None:
            return
        weight, _data_type, bits, imatrix_raw, _thresh, search_fn = inputs
        from auto_round.data_type.utils import reshape_imatrix_for_weight

        imatrix = reshape_imatrix_for_weight(imatrix_raw, weight, wrapper.orig_layer.group_size)
        wrapper.finalize_batched_search(search_fn(weight, bits, imatrix))

    @torch.no_grad()
    def _run_device(device_key, wrappers):
        _t0 = time.perf_counter()
        by_key = OrderedDict()
        for w in wrappers:
            by_key.setdefault(_wrap_batch_key(w._deferred_search_inputs), []).append(w)
        for _key, group in by_key.items():
            if len(group) < 2 or not _search_fn_stackable(group[0]._deferred_search_inputs[5]):
                for w in group:
                    _run_one(w)
                stats[device_key]["singletons"] += len(group)
                continue
            cap = _batch_cap(group, device_key, max_batch)
            for start in range(0, len(group), cap):
                chunk = group[start : start + cap]
                inputs0 = chunk[0]._deferred_search_inputs
                _bits = inputs0[2]
                search_fn = inputs0[5]
                stacked_w = torch.stack([w._deferred_search_inputs[0] for w in chunk])
                stacked_im = _materialize_wrap_imatrix(chunk, stacked_w)
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
                    stats[device_key]["singletons"] += len(chunk)
                    continue
                for w, res in zip(chunk, results):
                    w.finalize_batched_search(res)
                stats[device_key]["batches"] += 1
        stats[device_key]["modules"] = len(wrappers)
        stats[device_key]["wall"] = time.perf_counter() - _t0

    if len(device_groups) > 1:
        keyed = group_items_by_device(
            list(device_groups.values()), device_of=lambda ws: _wrap_batch_device_of(ws[0]._deferred_search_inputs)
        )
        run_items_by_device(
            keyed, lambda _idx, ws: _run_device(_wrap_batch_device_of(ws[0]._deferred_search_inputs), ws)
        )
    else:
        for dev, ws in device_groups.items():
            _run_device(dev, ws)
    _t_total = sum(st.get("wall", 0.0) for st in stats.values())
    _n_mod = sum(st["modules"] for st in stats.values())
    _n_batch = sum(st["batches"] for st in stats.values())
    _n_single = sum(st["singletons"] for st in stats.values())
    from auto_round.utils.pool_placement import _short_device_key

    per_device = (
        "{" + ", ".join(f"'{_short_device_key(dev)}': {st.get('wall', 0.0):.2f}" for dev, st in stats.items()) + "}"
    )
    logger.debug(
        "[batched-search] wrap search: %d modules, %d batch + %d singleton search calls, "
        "%.2fs device-wall (%.2fs summed) walls %s",
        _n_mod,
        _n_batch,
        _n_single,
        max(st.get("wall", 0.0) for st in stats.values()),
        _t_total,
        per_device,
    )
    return True
