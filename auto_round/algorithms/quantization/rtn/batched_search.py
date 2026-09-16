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

"""Batched zero-shot (iters=0) search driver.

Layers whose staged key -- (device, weight shape/dtype, bits, group size,
symmetry, threshold, resolved quant callable) -- matches are stacked along a
leading dim and quantized with ONE ``weight_quant_func`` call. The quant math
is row-independent, and the per-column imatrix is pre-expanded to each
module's full weight shape before stacking so the quant function's internal
flatten-expand becomes an identity: results are bit-identical to the serial
``unwrapper({})`` path. Write-back goes through the wrapper's ``_apply_qdq``
(the unwrapper's own conventions), so serial and batched outputs can never
drift apart.
"""

from collections import OrderedDict

import torch

from auto_round.algorithms.quantization.search_dispatch import (
    _batch_cap,
    _fn_key,
    dump_oom_tensor_census_,
    group_items_by_device,
    pick_search_worker_devices,
    run_items_by_device,
)
from auto_round.logger import logger
from auto_round.utils import set_module


def _staged_weight(wrapper):
    """The weight exactly as _qdq_weight would pass it to the quant func."""
    weight = wrapper.orig_layer.weight
    import transformers

    if type(wrapper.orig_layer) == transformers.pytorch_utils.Conv1D:
        weight = weight.t()
    return weight


def _staged_imatrix(wrapper, weight):
    """The RAW per-column imatrix (or None) -- expansion happens chunk-time.

    Staging the expanded full-size copy would pin N x weight bytes across the
    whole staging phase; the chunk-time expansion below is transient and
    bounded by the batch cap. Groups are homogeneous on imatrix presence (it
    is part of the staged key), so chunks are all-None or all-tensor.
    """
    del weight
    return getattr(wrapper.orig_layer, "imatrix", None)


def _expand_imatrix(im, weight, group_size):
    """Expand one module's raw imatrix to its full weight shape (chunk-time).

    Mirrors the quant funcs' own imatrix handling exactly (data_type/int.py):
    the raw per-column vector is group-padded (fill 1e-5) BEFORE being row
    expanded, so the stacked flat layout byte-matches what the serial
    per-module call produces after its internal flatten-pad-expand. Skipping
    the pad misplaces the group alignment for every module after the first
    when ``in % group_size != 0``; casting to the weight dtype (bf16/fp16
    weights with an fp32 hook-collected imatrix) silently diverges the
    search loss weighting from the serial path.
    """
    if im is None:
        return None
    im = im.to(weight.device)  # device only: keep the collected precision
    if im.dim() == 1 and weight.dim() == 2:
        out_f, in_f = weight.shape
        pad_len = 0
        if isinstance(group_size, int) and 0 < group_size <= in_f and in_f % group_size != 0:
            pad_len = ((in_f + group_size - 1) // group_size) * group_size - in_f
        if pad_len:
            im = torch.nn.functional.pad(im, (0, pad_len), value=1e-5)
        return im.unsqueeze(0).expand(out_f, -1).contiguous()
    if im.dim() == 2 and weight.dim() == 2 and im.shape[0] == 1:
        # row vector form: same treatment
        out_f, in_f = weight.shape
        pad_len = 0
        if isinstance(group_size, int) and 0 < group_size <= in_f and in_f % group_size != 0:
            pad_len = ((in_f + group_size - 1) // group_size) * group_size - in_f
        if pad_len:
            im = torch.nn.functional.pad(im, (0, pad_len), value=1e-5)
        return im.expand(out_f, -1).contiguous()
    if im.shape == weight.shape:
        return im.contiguous()
    return im


def _extra_quant_kwargs_key(wrapper):
    """Per-module _extra_quant_kwargs hook values as a hashable key term.

    Any wrapper subclass defining the hook injects arbitrary per-module kwargs
    into the quant call; two modules with differing extras must never share a
    stacked chunk (same hazard class as differing global_scale).
    """
    hook = getattr(wrapper, "_extra_quant_kwargs", None)
    if not callable(hook):
        return None
    return tuple(sorted((str(k), repr(v)) for k, v in hook().items()))


def _staged_key(wrapper, weight, imatrix):
    layer = wrapper.orig_layer
    # The stacked call assembles quant kwargs once from chunk[0], so every
    # per-layer kwarg _quant_call_kwargs injects must be part of the key:
    # modules differing in any of them never share a batch (nv-fp4 computes
    # weight_global_scale per layer; scale_dtype/super_* come from
    # per-layer config pins). imatrix is passed as a stacked override.
    global_scale = getattr(wrapper, "weight_global_scale", None)
    return (
        str(weight.device),
        tuple(weight.shape),
        str(weight.dtype),
        getattr(layer, "bits", None),
        getattr(layer, "group_size", None),
        getattr(layer, "sym", None),
        float(getattr(wrapper, "q_scale_thresh", 1e-5)),
        imatrix is not None,
        # normalize away per-wrapper torch.compile wrappers: the underlying
        # eager original is shared per config, while each compiled wrapper is
        # unique (repr-keyed) and would silently collapse every group to size 1
        _fn_key(getattr(wrapper.weight_quant_func, "_torchdynamo_orig_callable", None) or wrapper.weight_quant_func),
        str(getattr(layer, "scale_dtype", None)),
        getattr(layer, "super_bits", None),
        getattr(layer, "super_group_size", None),
        None if global_scale is None else float(global_scale),
        _extra_quant_kwargs_key(wrapper),
    )


def swap_wrapper_callables_to_eager(wrapper):
    """Point compiled wrapper callables at their eager originals.

    Returns a restore fn or None. ``unwrapper({})`` may call BOTH the
    (possibly torch.compile-wrapped) weight_quant_func and -- for
    act-quantized layers, act_bits <= 8 -- the compiled act_quant_func (the
    act tail's first-ever call in the iters=0 lane); on worker threads
    either first call would race dynamo's trace lock, exactly like the
    stacked path.
    """
    restores = []

    for attr in ("weight_quant_func", "act_quant_func"):
        fn = getattr(wrapper, attr, None)
        if fn is None:
            continue
        orig = getattr(fn, "_torchdynamo_orig_callable", None)
        if orig is None:
            continue
        setattr(wrapper, attr, orig)
        restores.append((attr, fn))

    if not restores:
        return None

    def _restore():
        for attr, fn in restores:
            setattr(wrapper, attr, fn)

    return _restore


@torch.no_grad()
def run_batched_rtn_search(model, staged, max_batch=None):
    """Finish deferred zero-shot wrappers on stacked same-shape batches.

    Args:
        model: The model tree (for set_module re-attachment).
        staged: list of (layer_name, wrapper) with the search deferred.
        max_batch: optional explicit chunk size.

    Every staged module is finished by this function itself -- grouped and
    stacked, singleton-per-module, or via the per-module OOM fallback -- so
    the caller never has leftover wrappers to finish.
    """
    entries = []
    for layer_name, wrapper in staged:
        weight = _staged_weight(wrapper)
        imatrix = _staged_imatrix(wrapper, weight)
        entries.append({"name": layer_name, "w": wrapper, "weight": weight, "im": imatrix})

    device_groups = OrderedDict()
    for e in entries:
        device_groups.setdefault(str(e["weight"].device), []).append(e)

    # Offload pass: the searches are weight-local, so chunks may run on any
    # device with headroom (the zero-shot lane leaves every non-home GPU idle).
    # Chunks are assigned round-robin over viable worker devices; the weights
    # and imatrices move to the worker for the stacked call and results write
    # back cross-device through _apply_qdq.
    chunks = []  # (home_device, entries)
    for dev, batch in device_groups.items():
        by_key = OrderedDict()
        for e in batch:
            by_key.setdefault(_staged_key(e["w"], e["weight"], e["im"]), []).append(e)
        for _key, group in by_key.items():
            from auto_round.algorithms.quantization.search_dispatch import _search_fn_stackable

            if len(group) < 2 or not _search_fn_stackable(group[0]["w"].weight_quant_func):
                chunks.extend((dev, [e]) for e in group)
                continue
            cap = _batch_cap(
                [
                    type("S", (), {"_deferred_search_inputs": (e["weight"], None, None, e["im"], None, None)})()
                    for e in group
                ],
                dev,
                max_batch,
            )
            for start in range(0, len(group), cap):
                chunks.append((dev, group[start : start + cap]))

    def _chunk_working_set(chunk):
        e0 = chunk[0]
        per = (e0["weight"].numel() + (e0["im"].numel() if e0["im"] is not None else 0)) * 4 * 4
        return per * len(chunk)

    buckets = OrderedDict()
    rr = 0
    for dev, chunk in chunks:
        workers = pick_search_worker_devices(_chunk_working_set(chunk), home_device=dev)
        worker = workers[rr % len(workers)] if workers else dev
        rr += 1
        buckets.setdefault(str(worker), []).append(chunk)
    if buckets:
        # len(c) counts MODULES per chunk, len(cs) chunks per worker: print
        # both so the line cannot read as "one chunk per module" when
        # batching is engaged; per-worker chunk counts use the memory-monitor
        # short-device grammar ({"0": 5, ...})
        from auto_round.utils.pool_placement import _short_device_key

        _n_mods = sum(len(c) for cs in buckets.values() for c in cs)
        _n_chunks = sum(len(cs) for cs in buckets.values())
        _workers = ", ".join(f"'{_short_device_key(w)}': {len(cs)}" for w, cs in buckets.items())
        logger.debug("[rtn-batch] %d modules in %d chunks over workers {%s}", _n_mods, _n_chunks, _workers)

    def _worker_of(chunk):
        for wk, cs in buckets.items():
            if any(c is chunk for c in cs):
                return str(wk)
        return str(chunk[0]["weight"].device)  # not found (should not happen): stay home

    def _unwrap_with_cpu_fallback(w, name):
        """unwrapper({}) under the serial lane's OOM->CPU contract.

        The per-module search can OOM on the device exactly like the serial
        path's own search (base.py); there the run falls back to CPU instead
        of crashing. The batched lane previously called unwrapper({})
        unguarded -- a module that OOMed took the whole run down, violating
        the progressive batched -> per-module -> CPU ladder.
        """
        try:
            return w.unwrapper({})
        except torch.OutOfMemoryError:
            from auto_round.algorithms.quantization.search_dispatch import dump_oom_tensor_census_
            from auto_round.wrapper import WrapperLinear

            dump_oom_tensor_census_("rtn batched unwrapper")
            logger.warning("[rtn-batch] per-module search OOM for %s; falling back to CPU", name)
            layer = w.orig_layer if hasattr(w, "orig_layer") else w
            layer = layer.to("cpu")
            layer = WrapperLinear(
                layer,
                enable_minmax_tuning=False,
                enable_norm_bias_tuning=False,
                enable_round_tuning=False,
                enable_torch_compile=False,
                iters=0,
            )
            return layer.unwrapper({})

    @torch.no_grad()
    def _run_chunk(chunk, worker, threaded=False):
        w0 = chunk[0]["w"]
        dev = str(chunk[0]["weight"].device)
        worker = str(worker)
        if len(chunk) == 1:
            _restore = swap_wrapper_callables_to_eager(chunk[0]["w"]) if (threaded or worker != dev) else None
            try:
                layer = _unwrap_with_cpu_fallback(chunk[0]["w"], chunk[0]["name"])
            finally:
                if _restore is not None:
                    _restore()
            set_module(model, chunk[0]["name"], layer)
            return
        weights = [e["weight"] for e in chunk]
        ims = [e["im"] for e in chunk]
        if worker != dev:
            weights = [w.to(worker) for w in weights]
            ims = [im.to(worker) if im is not None else None for im in ims]
        stacked_w = torch.stack(weights)
        stacked_im = None
        if ims and ims[0] is not None:
            _gs = w0.orig_layer.group_size
            stacked_im = torch.stack([_expand_imatrix(im, w, _gs) for im, w in zip(ims, weights)])
        kwargs = w0._quant_call_kwargs(
            torch.tensor(0.0), torch.tensor(1.0), torch.tensor(1.0), imatrix_override=stacked_im
        )
        kwargs = _relocate_tensor_kwargs(kwargs, str(stacked_w.device))
        fn = w0.weight_quant_func
        if worker != dev or threaded:
            # measured: the compiled path on workers (pre-warmed) ran 3x slower than
            # eager for this bandwidth-bound grid search; eager also sidesteps the
            # cross-thread dynamo trace race entirely -- including for home-staying
            # chunks, which the multi-device path also executes on worker threads
            fn = getattr(fn, "_torchdynamo_orig_callable", None) or fn
        try:
            qdq, scale, zp = fn(stacked_w, **kwargs)
        except torch.OutOfMemoryError:
            logger.warning(
                "[rtn-batch] stacked search OOM (%d modules); finishing this chunk per-module "
                "(shrink batches with AR_SEARCH_BATCH_GB or disable with AR_DISABLE_BATCHED_SEARCH=1)",
                len(chunk),
            )
            dump_oom_tensor_census_("rtn batched search")
            for e in chunk:
                _restore = swap_wrapper_callables_to_eager(e["w"]) if (threaded or worker != dev) else None
                try:
                    layer = _unwrap_with_cpu_fallback(e["w"], e["name"])
                finally:
                    if _restore is not None:
                        _restore()
                set_module(model, e["name"], layer)
            return
        n = len(chunk)
        scale_parts = _split_leading(scale, n)
        zp_parts = _split_leading(zp, n)
        import transformers

        for i, e in enumerate(chunk):
            res = qdq[i]
            w = e["w"]
            # mirror the serial _qdq_weight output contract: cast back to the
            # stored dtype and restore the HF Conv1D [in, out] layout (staging
            # transposed it)
            res = res.to(w.orig_layer.weight.dtype)
            if type(w.orig_layer) == transformers.pytorch_utils.Conv1D:
                res = res.t()
            # route the write-back through unwrapper({}) with the precomputed
            # search result injected: _apply_qdq alone would skip the
            # unwrapper tail (bias/meta update, static-act rescale, act
            # metadata, WrapperWALayer attachment) and leave act-quantized
            # layers (act_bits <= 8) silently inconsistent with the serial
            # path. The injection short-circuits _qdq_weight's recompute so
            # no compiled weight_quant_func runs here; the act tail may still
            # call its compiled act_quant_func, so swap BOTH callables to
            # eager on worker threads before unwrapping.
            _restore = swap_wrapper_callables_to_eager(w) if (threaded or worker != dev) else None
            try:
                w._presolved_qdq = (res, scale_parts[i], zp_parts[i])
                layer = w.unwrapper({})
            finally:
                if _restore is not None:
                    _restore()
            set_module(model, e["name"], layer)

    if len(buckets) > 1:
        keyed = OrderedDict()
        for wk, cs in buckets.items():
            keyed.setdefault(str(wk), []).append((len(keyed), cs))
        run_items_by_device(keyed, lambda _idx, cs: [_run_chunk(c, _worker_of(c), threaded=True) for c in cs])
    else:
        for wcs in buckets.values():
            for c in wcs:
                _run_chunk(c, _worker_of(c))


def _relocate_tensor_kwargs(kwargs: dict, device: str) -> dict:
    """Move accelerator-valued kwargs onto the compute device.

    The weight-local searches normally run on the weight's own device, so
    tensor kwargs (e.g. an nv-fp4 ``global_scale``) live wherever the layer
    lives; under worker offload the stacked call runs elsewhere and a
    left-behind operand would fault at the first cross-device op (the mapped-
    lane init_scale crash class). CPU scalars broadcast and are left alone.
    """
    for key, val in kwargs.items():
        if isinstance(val, torch.Tensor) and val.device.type != "cpu" and str(val.device) != device:
            kwargs[key] = val.to(device)
    return kwargs


def _split_leading(result, n):
    """Split a quant result along the (possibly flattened) leading batch dim.

    Stacked calls may return per-module rows flattened into the leading dim
    (e.g. scale ``[N*out, 1]``) or keep a batch dim (``[N, out, groups]``);
    both reshape to ``[N, -1, ...]`` identically. Scalars pass through.
    Dicts are split recursively: GGUF K-quant funcs return scale/zp as dicts
    (``{"scale": ..., "d_scale": ...}``) whose tensor values carry the
    flattened batch dim -- sharing the whole dict per module would give every
    layer N-times the expected metadata (caught at GGUF export validation).
    Non-tensor dict values (e.g. ``None``) are shared as-is.
    """
    if isinstance(result, dict):
        split_values = {key: _split_leading(value, n) for key, value in result.items()}
        return [{key: values[i] for key, values in split_values.items()} for i in range(n)]
    if not isinstance(result, torch.Tensor):
        return [result] * n
    if result.dim() == 0 or result.numel() == 1:
        return [result] * n
    return result.reshape(n, -1, *result.shape[1:])
