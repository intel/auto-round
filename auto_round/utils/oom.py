# Copyright (c) 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

"""OOM diagnostics: tensor census, pluggable anywhere.

Three entry points, cheapest first:

* ``oom_census("context")`` -- a few-line context manager around any
  suspicious frame (including code this repo does not own); prints the
  census and re-raises.
* ``dump_oom_tensor_census_("context")`` -- the bare one-liner for
  existing ``except torch.OutOfMemoryError`` blocks.
* ``install_oom_census_hook()`` -- install once (the CLI does); fires
  for ANY uncaught CUDA OOM on any thread, wherever it escapes.

All are best effort: the census swallows its own failures and never
masks the original error.
"""

from contextlib import contextmanager

import torch

from auto_round.logger import logger


def _shape_key(shape) -> tuple:
    """Shape as a hashable key; symbolic dims (SymInt) fall back to a string form."""
    try:
        return tuple(int(d) for d in shape)
    except Exception as e:  # symbolic dims (SymInt) or exotic shapes: stringify
        logger.debug("[oom] census shape key fell back to str for %s (%s)", type(shape).__name__, e)
        return (str(tuple(shape)),)


def _group_tensors_by_shape(objs) -> tuple:
    """(device, dtype, shape) -> [count, bytes] over accelerator tensors; sorted by bytes.

    Covers every non-cpu accelerator (cuda, hpu, xpu, mps, ...): the census
    must name the residents on whichever device ran out.
    """

    skipped = 0

    groups: dict = {}
    for obj in objs:
        try:
            if not isinstance(obj, torch.Tensor) or obj.device.type in ("cpu", "meta"):
                continue
            key = (str(obj.device), str(obj.dtype), _shape_key(obj.shape))
            nbytes = int(obj.numel()) * obj.element_size()
        except Exception as e:  # one unreadable tensor must never kill the census
            logger.debug("[oom] census skipping unreadable tensor (%s)", e)
            skipped += 1
            continue
        g = groups.get(key)
        if g is None:
            groups[key] = [1, nbytes]
        else:
            g[0] += 1
            g[1] += nbytes
    return sorted(groups.items(), key=lambda kv: -kv[1][1]), skipped


def _representatives(groups, objs):
    """One representative tensor per group, drawn from a single scan."""
    wanted = {g[0] for g in groups}  # group keys are (device, dtype, shape)
    seen = {}
    for obj in objs:
        try:
            if not isinstance(obj, torch.Tensor) or obj.device.type in ("cpu", "meta"):
                continue
            key = (str(obj.device), str(obj.dtype), _shape_key(obj.shape))
        except Exception as e:
            logger.debug("[oom] census representative scan skipped a tensor (%s)", e)
            continue
        if key in wanted and key not in seen:
            seen[key] = obj
    for gk, meta in groups:
        if gk in seen:
            yield seen[gk], (gk, meta)


def _is_census_noise(ref, skip_ids):
    """Referrers that are the census's own structures or whole-heap scans."""
    if id(ref) in skip_ids:
        return True
    if isinstance(ref, (list, tuple, set)) and len(ref) > 4096:
        return True  # heap snapshots and other scans, never real holders
    if isinstance(ref, dict) and ref:
        for k in list(ref.keys())[:3]:
            if isinstance(k, tuple) and len(k) == 3 and all(isinstance(x, str) for x in k):
                return True  # the census's own representative dicts
    return False


def _describe_frame(frame, target):
    """Frame as '<func> (<file>:<line>)' plus the local names holding target."""
    try:
        code = frame.f_code
        where = f"{code.co_name} ({code.co_filename.split('/')[-1]}:{frame.f_lineno})"
        names = []
        for k, v in list(frame.f_locals.items())[:80]:
            try:
                if v is target or (isinstance(v, list) and len(v) <= 4096 and target in v):
                    names.append(str(k))
            except Exception as e:
                logger.debug("[oom] census frame-local scan skipped a local (%s)", e)
                continue
        return where + (f" locals:{','.join(names[:4])}" if names else "")
    except Exception as e:
        logger.debug("[oom] census frame description fell back to bare 'frame' (%s)", e)
        return "frame"


def _attr_name_of(owner, target):
    """The attribute of ``owner`` (if any) that holds ``target`` by identity."""
    try:
        d = getattr(owner, "__dict__", None)
        if isinstance(d, dict):
            for k, v in d.items():
                if v is target:
                    return str(k)
    except Exception as e:
        logger.debug("[oom] census attr-name scan failed for owner %s (%s)", type(owner).__name__, e)
    return None


def _describe(obj, depth=0):
    """One-line description of a holder object."""
    import types

    t = type(obj)
    if t is dict:
        keys = [str(k) for k in list(obj.keys())[:3]]
        return f"dict[{','.join(keys)}]"
    if t is types.FrameType:
        return "frame"
    if t is types.GeneratorType:
        return "generator"
    if t in (list, tuple, set):
        return f"{t.__name__}(len={len(obj)})"
    name = getattr(obj, "__class__", t).__name__
    mod = getattr(getattr(obj, "__class__", t), "__module__", "")
    return f"{mod}.{name}" if mod and mod != "builtins" else name


def _describe_referrers(tensor, skip_ids, limit=6, depth=0):
    """Holders of ``tensor``; small containers are unwrapped one more level."""
    import gc as _gc

    out = []
    try:
        for ref in _gc.get_referrers(tensor):
            if _is_census_noise(ref, skip_ids):
                continue
            import types as _types

            if isinstance(ref, _types.FrameType):
                out.append(_describe_frame(ref, tensor))
                if len(out) >= limit:
                    break
                continue
            desc = _describe(ref)
            attr = _attr_name_of(ref, tensor) if depth == 0 else None
            if attr:
                desc += f".{attr}"
            out.append(desc)
            # unwrap small containers one level: name who holds the holder
            if depth == 0 and isinstance(ref, (list, tuple, set, dict)) and len(ref) <= 4096:
                for owner in _gc.get_referrers(ref):
                    if _is_census_noise(owner, skip_ids) or owner is tensor:
                        continue
                    od = _describe(owner, 1)
                    oattr = _attr_name_of(owner, ref)
                    out.append(f"  <- {od}" + (f".{oattr}" if oattr else ""))
                    if len(out) >= limit:
                        break
            if len(out) >= limit:
                break
    except Exception as e:  # pragma: no cover - diagnostics must not mask the OOM
        out.append(f"<referrer scan failed: {e}>")
    return out


def dump_oom_tensor_census_(context: str = "") -> None:
    """Tensor census at OOM time: per-device allocator state + top tensor groups.

    Names the accumulating residents (a leak shows as one (shape, dtype)
    group growing across blocks). Best effort: never masks the OOM itself.
    """
    import gc
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # isinstance over heap objects
            _dump_census(gc, context)
    except Exception as e:  # pragma: no cover - diagnostics must not mask the OOM
        logger.error("[oom] tensor census failed (%s)", e)


def _dump_census(gc, context=""):
    idx = "?"  # bound before the loop so the handler below can always name it
    if context:
        logger.error("[oom] tensor census (context: %s)", context)
    try:
        try:
            for idx in range(torch.cuda.device_count()):
                logger.error(
                    "[oom] cuda:%s allocated=%.2fGiB reserved=%.2fGiB",
                    idx,
                    torch.cuda.memory_allocated(idx) / 2**30,
                    torch.cuda.memory_reserved(idx) / 2**30,
                )
        except Exception as e:  # pragma: no cover - allocator stats are cuda-only
            logger.debug("[oom] allocator stats unavailable for cuda:%s (%s)", idx, e)
        objs = gc.get_objects()
        per_device: dict = {}
        top, _skipped = _group_tensors_by_shape(objs)
        for (_dev, _dt, _shape), (_cnt, _nb) in top:
            per_device[_dev] = per_device.get(_dev, 0) + _nb
        for _dev, _nb in sorted(per_device.items(), key=lambda kv: -kv[1]):
            logger.error("[oom] %s live tensors ≈ %.2fGiB", _dev, _nb / 2**30)
        if _skipped:
            logger.error("[oom] census skipped %d unreadable tensors", _skipped)
        for (_dev, _dt, _shape), (_cnt, _nb) in top[:8]:
            logger.error("[oom] %s %s %s x%d = %.2fGiB", _dev, _dt, list(_shape), _cnt, _nb / 2**30)
        # holder attribution runs LAST and fully guarded: it must never cost the
        # inventory above (a symbolic-shape failure here previously ate the totals)
        try:
            skip = {id(objs), id(top)}
            for rep, ((_dev, _dt, _shape), (_cnt, _nb)) in _representatives(top[:6], objs):
                for desc in _describe_referrers(rep, skip):
                    logger.error("[oom]   %s %s held by: %s", _dev, list(_shape), desc)
        except Exception as e:  # pragma: no cover - must not cost the inventory
            logger.error("[oom] holder attribution failed (%s)", e)
    except Exception as e:  # pragma: no cover - must not mask the OOM
        logger.error("[oom] tensor census failed (%s)", e)


def _is_oom(exc: BaseException) -> bool:
    """torch.OutOfMemoryError (cuda, modern xpu) or message-based OOM (hpu et al.)."""
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


@contextmanager
def oom_census(context: str = ""):
    """Census-on-OOM context manager: plug around any frame, upstream code included.

    .. code-block:: python

        with oom_census("block collection forward"):
            out = block_forward_fn(block, fp_inputs, input_others)
    """
    try:
        yield
    except Exception as exc:
        if _is_oom(exc):
            dump_oom_tensor_census_(context)
        raise


_OOM_HOOK_INSTALLED = False


def install_oom_census_hook() -> bool:
    """Install the last-resort census for uncaught CUDA OOMs (any thread).

    Fires wherever a ``torch.OutOfMemoryError`` escapes to the top of the
    process -- including frames this repo does not own -- printing the tensor
    census before the default exception reporting takes over. Idempotent;
    returns True when it installed the hooks.
    """
    global _OOM_HOOK_INSTALLED
    if _OOM_HOOK_INSTALLED:
        return False
    import sys
    import threading

    prior_sys = sys.excepthook
    prior_the = threading.excepthook

    def _sys_hook(tp, val, tb):
        try:
            if _is_oom(val):
                dump_oom_tensor_census_("uncaught")
        except Exception as e:  # pragma: no cover - diagnostics must not mask the error
            logger.error("[oom] census hook failed while reporting (%s)", e)
        prior_sys(tp, val, tb)

    def _the_hook(args):
        try:
            if _is_oom(args.exc_value):
                name = args.thread.name if args.thread is not None else "?"
                dump_oom_tensor_census_(f"uncaught (thread {name})")
        except Exception as e:  # pragma: no cover - diagnostics must not mask the error
            logger.error("[oom] census hook failed while reporting (%s)", e)
        prior_the(args)

    sys.excepthook = _sys_hook
    threading.excepthook = _the_hook
    _OOM_HOOK_INSTALLED = True
    return True
