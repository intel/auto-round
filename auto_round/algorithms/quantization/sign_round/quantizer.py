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
import copy
import time as _ptime
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch
from torch import autocast

from auto_round.algorithms.block_runner import BlockForwardRunner
from auto_round.algorithms.quantization.base import BaseQuantizer
from auto_round.algorithms.quantization.sign_round.config import SignRoundConfig
from auto_round.algorithms.quantization.sign_round.sign_sgd import SignSGD
from auto_round.algorithms.registry import register_pipeline_member
from auto_round.compressors.utils import (
    IndexSampler,
    collect_best_params,
    snapshot_best_params,
)
from auto_round.logger import logger


def _tuning_state_bytes(block, target_dev):
    """In-loop tuning state for parameters homed on ``target_dev`` (logical count).

    14 B per logical parameter: fp32 value + fp32 grad (first backward) +
    best-params snapshot + bf16 copy. At tune time ``block.parameters()``
    yields BOTH the original weights and the wrappers' registered value
    params -- counting both would price every wrapped parameter twice
    (measured: a 34.63 GiB reserve on a lane whose honest state is ~17).
    Wrapper tuning tensors are identified by identity (any ``.params``
    dict on a module) and excluded; the original weights carry the charge.
    """
    return _logical_state_by_device(block).get(str(target_dev), 0)


def _logical_state_by_device(block):
    """Deduped per-device form of :func:`_tuning_state_bytes` ({device: bytes}).

    ``_state_bytes_by_device`` counts ``block.parameters()`` raw, which at
    tune time includes BOTH the original weights and the wrappers' fp32
    value params -- every wrapped parameter priced twice (the 4x3090 auto
    check logged state 23.3 GiB where the honest logical state is ~12).
    """
    tuning = set()
    for m in block.modules():
        _p = getattr(m, "params", None)
        if isinstance(_p, dict):
            for _v in _p.values():
                if torch.is_tensor(_v):
                    tuning.add(id(_v))
    out = {}
    for p in block.parameters():
        if id(p) in tuning:
            continue
        out[str(p.device)] = out.get(str(p.device), 0) + p.numel() * 14
    return out


def _ensure_routed_shape_recorders_(block):
    """Attach fire-once recorders on MoE containers for the activation budget.

    Records ``top_k = top_k_index.shape[-1]`` and ``routed_rows = index.numel()``
    from the first NATURAL dispatch (forced all-experts routing -- tracing /
    warmup -- is skipped: it would record top_k = num_experts and over-reserve).
    Layout-agnostic: no attribute spellings, only arg shapes. The experts
    interface passes ``(hidden_states, top_k_index, top_k_weights)``; the
    recorder reads the two leading tensor args. The hook removes itself after
    the first record.
    """
    from auto_round.modeling.fused_moe.utils import force_all_experts_routing_enabled

    if force_all_experts_routing_enabled():
        return
    from auto_round.utils.model import is_moe_layer

    def _record(module, args):
        recorded = False
        try:
            tensors = [a for a in args if torch.is_tensor(a)]
            if len(tensors) < 2:
                return
            hidden, index = tensors[0], tensors[1]
            if hidden.ndim < 2 or index.ndim < 1:
                return
            module._routed_shape_rec_ = (int(index.shape[-1]), int(index.numel()))
            recorded = True
            logger.debug(
                "[routed-shape] recorded top_k=%d routed_rows=%d",
                module._routed_shape_rec_[0],
                module._routed_shape_rec_[1],
            )
        except Exception as e:  # pragma: no cover - diagnostics only, never break the forward
            logger.debug("[routed-shape] recorder skipped (%s)", e)
        finally:
            if recorded:
                # self-remove ONLY on a successful record: removing on a
                # miss (non-dispatch arg shapes) silently disarms the
                # recorder before it ever sees a natural dispatch
                for handle in list(getattr(module, "_routed_rec_handles_", [])):
                    handle.remove()
                module._routed_rec_handles_ = []

    def _is_experts_container(name, module):
        # container may carry the marker itself, expose num_experts, be a
        # ModuleList, or carry the MoE marker only on an ANCESTOR
        # (HYV3Experts: none of the former)
        if is_moe_layer(module) or isinstance(getattr(module, "num_experts", None), int):
            return True
        stem = name
        while "." in stem:
            stem = stem.rsplit(".", 1)[0]
            mod = mods.get(stem)
            if mod is not None and (is_moe_layer(mod) or isinstance(getattr(mod, "num_experts", None), int)):
                return True
        return False

    mods = dict(block.named_modules())
    for _name, module in mods.items():
        if (
            "expert" in _name.lower()
            and _is_experts_container(_name, module)
            and not hasattr(module, "_routed_shape_rec_")
        ):
            module._routed_rec_handles_ = [module.register_forward_pre_hook(_record)]


def _grouped_stack_bytes(block, device, config=None):
    """Qdq-stack bytes the grouped experts modes materialize on ``device``."""
    d = _grouped_stack_bytes_detail(block, device, config=config)
    return d["retention"] + d["transient"]


def _grouped_stack_bytes_detail(block, device, config=None):
    """Per-term split of :func:`_grouped_stack_bytes` (retention + transient).

    Two chunk-aware terms (retention is chunk-independent; transient is
    chunk-sized) for ``auto``/``linear_grouped``/
    ``linear_grouped_sliced`` (linear_loop stacks nothing):

    - retention (chunk-independent): autograd keeps EVERY chunk's stacked
      fp32 values (straight-through mask) and qdq output (GEMM backward) --
      6 B per weight element of the experts HOMED on the device, observed as
      31 simultaneous 16-expert stacks on the OOMed lane;
    - transient (chunk-sized): the quant func streams the stacked chunk
      through several passes (~2x chunk bytes). Chunk size comes from the
      same source the grouped path uses (_qdq_chunk_size: AR_MOE_CHUNK or
      the auto working-set budget; 0 = fuse everything = all-experts chunk).

    Mode resolution matches the dispatch: an EXPLICIT env value governs,
    otherwise ``config._experts_implementation`` (so gates stay consistent
    with a mid-run auto switch to linear_loop instead of charging stacks
    for a lane that no longer stacks any). Mode-set-but-loop-ran fallbacks
    can only overcharge -- safe direction.
    """
    import auto_round.envs as _envs
    from auto_round.modeling.fused_moe.moe_experts_interface import GROUPED_LINEAR_IMPL

    mode = str(getattr(_envs, "AR_MOE_EXPERTS_IMPL", "auto") or "auto").lower()
    if mode in ("", "auto"):
        mode = str(getattr(config, "_experts_implementation", GROUPED_LINEAR_IMPL) or GROUPED_LINEAR_IMPL)
    if mode not in ("", "auto", "linear_grouped", "linear_grouped_sliced"):
        return {"retention": 0, "transient": 0}
    retention = 0
    transient = 0
    try:
        from auto_round.modeling.fused_moe.grouped_experts import _qdq_chunk_size
        from auto_round.utils.model import is_moe_layer

        def _is_expert_leaf(name, leaf):
            # expert leaves are named "experts.N.proj" in every unfused layout;
            # the container may carry num_experts, be a ModuleList, or carry
            # the MoE marker only on an ANCESTOR (HYV3Experts: none of the
            # former -- a name-only walk returned 0 stacks on the real lane
            # and silently disabled the activation charge and the auto
            # linear_loop pick)
            if "expert" not in name.lower() or "shared" in name.lower():
                # shared experts are plain modules outside the dispatch (the
                # repo's own is_moe_expert predicate excludes them); stacking
                # or routing charges on them are phantom bytes
                return False
            stem = name
            while "." in stem:
                stem = stem.rsplit(".", 1)[0]
                mod = mods.get(stem)
                if mod is None:
                    continue
                if (
                    isinstance(mod, torch.nn.ModuleList)
                    or isinstance(getattr(mod, "num_experts", None), int)
                    or is_moe_layer(mod)
                ):
                    return True
            return False

        # group leaves by slot name so per-slot chunking matches the real
        # grouped path (gate/up/down are chunked independently)
        mods = dict(block.named_modules())
        by_slot = {}
        seen_weight_ids = set()
        for name, leaf in mods.items():
            # at tune time named_modules yields BOTH the wrapper
            # (...gate_proj) and its orig_layer child (...gate_proj.orig_
            # layer), each exposing a same-shape weight -- counting both
            # doubled retention (measured: 10.1-10.3 GiB charged where the
            # logical 6 B/elem charge is ~5.2)
            if name.endswith(".orig_layer"):
                continue
            weight = getattr(leaf, "weight", None)
            if weight is None or not hasattr(weight, "numel"):
                continue
            if id(weight) in seen_weight_ids:
                continue
            if not _is_expert_leaf(name, leaf):
                continue
            seen_weight_ids.add(id(weight))
            by_slot.setdefault(type(leaf).__name__ + "." + getattr(leaf, "slot_tag", ""), []).append((leaf, weight))
        for _slot, pairs in by_slot.items():
            local_pairs = [(l, w) for l, w in pairs if str(w.device) == str(device)]
            if not local_pairs:
                continue
            elems = sum(int(w.numel()) for _l, w in local_pairs)
            retention += elems * 6  # fp32 value stack + bf16 qdq output
            try:
                chunk = _qdq_chunk_size([l for l, _w in local_pairs])  # env/budget-aware; 0 = all
            except Exception as e:
                logger.debug("[tune] chunk sizing failed for slot %r on %s (%s); using 16", _slot, device, e)
                chunk = 16
            per_leaf = elems // max(1, len(local_pairs))
            chunk_elems = elems if chunk <= 0 else min(elems, chunk * per_leaf)
            transient += chunk_elems * 6  # ~2 streaming passes over the chunk
    except Exception as e:  # pragma: no cover - never break the gate
        logger.debug("[tune] grouped-stack accounting unavailable (%s)", e)
        return {"retention": 0, "transient": 0}
    return {"retention": retention, "transient": transient}


def _routed_budget_bytes(block, tensors, batch_size, config=None):
    """tokens x top_k routed working set x6 (in/out accumulators + bwd grad).

    Top_k resolution order: config values (maintained spelling list) ->
    recorded NATURAL-dispatch arg shapes (layout/spelling agnostic) ->
    container attrs. None when nothing resolves (callers decline).
    """
    from auto_round.utils.device import get_first_available_attr
    from auto_round.utils.model import is_moe_layer

    top_k = None
    if config is not None:
        top_k = get_first_available_attr(config, ["num_experts_per_tok", "moe_num_active_primary_experts"])
        if top_k is None:
            moe_topk = getattr(config, "moe_topk", None)  # HunYuan MoE V1
            if isinstance(moe_topk, (list, tuple)) and moe_topk:
                top_k = moe_topk[0]
    if top_k is None:
        for m in block.modules():
            rec = getattr(m, "_routed_shape_rec_", None)
            if rec is not None:
                top_k, _rows = rec
                break
    if top_k is None:
        for _name, module in block.named_modules():
            if is_moe_layer(module) or isinstance(getattr(module, "num_experts", None), int):
                top_k = getattr(module, "top_k", None) or getattr(module, "num_experts_per_tok", None)
                if top_k is not None:
                    break
    if top_k is None:
        return None
    ref = (
        tensors[0]
        if isinstance(tensors, list) and tensors
        else next(
            (t for t in (tensors.values() if isinstance(tensors, dict) else []) if isinstance(t, torch.Tensor)), None
        )
    )
    if ref is None or ref.ndim < 2:
        return None
    row_len = int(ref.shape[-2]) if ref.ndim >= 3 else int(ref.shape[0])
    hidden_bytes = int(ref.shape[-1]) * ref.element_size()
    tune_tokens = max(1, int(batch_size)) * max(1, row_len)
    routed_rows = None
    for m in block.modules():
        rec = getattr(m, "_routed_shape_rec_", None)
        if rec is not None:
            top_k_rec, routed_rows = rec
            break
    if routed_rows is not None:
        # recorded rows scale per-token to this batch's tokens (collection may
        # have seen a larger batch; a floor-div here silently kept the
        # oversized count)
        seen_tokens = max(1, routed_rows // max(1, int(top_k)))
        routed_rows = int(routed_rows * tune_tokens / seen_tokens)
    else:
        routed_rows = tune_tokens * int(top_k)
    return int(routed_rows * hidden_bytes * 6)


def _activation_bytes_by_device(block, tensors, batch_size, config=None):
    """Per-DEVICE activation charge for the tune loop: {device: bytes}.

    The loop's forward/backward executes on every device the block spans --
    each module's saved output + grad lands on the module's WEIGHT HOME, and
    each expert's routed-row caches land on the expert's home. Charging the
    whole activation budget to the entry alone (the previous form) left the
    MoE stages' transients unpriced: measured 3.2 GiB/peer unaccounted on a
    4-GPU hy3 lane while the entry's charge matched its realized peak to
    0.2 GiB.

    Terms per device:
    - est split: the estimator's per-module output bytes (x2 grads, expert
      outputs ratio-scaled) bucketed by weight home -- counts shared experts
      wherever they are PLAIN modules outside the dispatch.
    - routed split: tokens x top_k x hidden x 6 distributed across devices
      by homed-expert count (first-order static split; per-expert token
      counts vary with routing draws).
    - grouped stacks: chunk-INDEPENDENT qdq retention, additive (a different
      tensor class entirely).

    Per-device composition is max(est_d, routed_d) -- the terms overlap on
      the routed rows exactly as in the scalar form, per device. None only
      when neither term resolves anywhere (callers decline).
    """
    try:
        from auto_round.utils.device import estimate_tuning_block_mem
        from auto_round.utils.model import is_moe_layer

        est_by_dev = {}
        expert_counts = {}
        est_failed = False
        has_moe = any(is_moe_layer(m) or isinstance(getattr(m, "num_experts", None), int) for m in block.modules())
        try:
            _ld, _la, _io, _ad, est_by_dev, expert_counts = estimate_tuning_block_mem(
                block, tensors, batch_size, config
            )
        except Exception as e:  # per-module accounting is best-effort for dense
            logger.warning("[tune] per-module activation estimate unavailable (%s)", e)
            est_failed = True
        if est_failed and has_moe:
            # an MoE block without the estimator loses the expert homes that
            # attribute routed bytes and stacks to devices; the composed
            # charge would collapse toward state-only and keep an
            # over-budget grouped lane -- decline instead
            return None
        if not has_moe:
            expert_counts = {}

        routed_by_dev = {}
        routed_total = None
        # routed x6 was calibrated on a LINEAR_LOOP lane (per-expert route
        # caches + batch cats, 12.7 GiB measured). Grouped mode builds no
        # route caches -- its routed working set (sorted flat buffers) is
        # covered by the estimator's ratio-scaled expert outputs, and the
        # weight-side retention arrives via the stacks term. Charging both
        # on a grouped lane double-charged ~3 GiB and falsely switched the
        # measured-working 5x3090 lane to linear_loop.
        stacks_any = any(
            _grouped_stack_bytes(block, d, config=config) > 0 for d in set(est_by_dev) | set(expert_counts)
        )
        if has_moe and not stacks_any:
            routed_total = _routed_budget_bytes(block, tensors, batch_size, config)
        total_experts = sum(expert_counts.values())
        if routed_total is not None and total_experts > 0:
            routed_by_dev = {d: int(routed_total * c / total_experts) for d, c in expert_counts.items()}
        elif routed_total is not None:
            # no expert homes resolved (detection is name-based): never drop
            # the budget silently -- split evenly across the graph's devices,
            # or onto the pool reference's device when no module info at all
            if est_by_dev:
                n = len(est_by_dev)
                routed_by_dev = {d: int(routed_total / n) for d in est_by_dev}
                logger.debug(
                    "[tune] routed budget %.2fGiB split evenly over %d devices (no expert homes)",
                    routed_total / 2**30,
                    n,
                )
            else:
                ref_dev = None
                if isinstance(tensors, list) and tensors and isinstance(tensors[0], torch.Tensor):
                    ref_dev = str(tensors[0].device)
                elif isinstance(tensors, dict):
                    for t in tensors.values():
                        if isinstance(t, torch.Tensor):
                            ref_dev = str(t.device)
                            break
                if ref_dev is not None:
                    routed_by_dev = {ref_dev: routed_total}

        out = {}
        for d in set(est_by_dev) | set(routed_by_dev):
            # est split is GiB floats (estimator), routed split is bytes
            m = max(int(est_by_dev.get(d, 0.0) * 2**30), int(routed_by_dev.get(d, 0)))
            out[d] = m + _grouped_stack_bytes(block, d, config=config)
        return out or None
    except Exception as e:  # pragma: no cover - placement must never break tuning
        logger.warning("[tune] activation estimate failed (%s); treating as unknown", e)
        return None


def _block_activation_bytes(block, tensors, batch_size, config=None, device=None):
    """Back-compat scalar view: this DEVICE's slice of the per-device charge."""
    by_dev = _activation_bytes_by_device(block, tensors, batch_size, config)
    if by_dev is None:
        return None
    if device is None:
        return max(by_dev.values())
    return by_dev.get(str(device))


_MOE_IMPL_AUTO_LINEAR_LOOP_DONE = False
_MOE_IMPL_AUTO_DONE_REF = None  # weakref to the deciding run's config


def _maybe_auto_linear_loop_for_tuning(block, tensors, batch_size, iters, config, model):
    """One-shot auto pick: ``linear_loop`` when grouped tuning cannot fit.

    Grouped tuning stacks per-expert fp32 value/grad tensors on every
    expert-homing device (chunk-INDEPENDENT autograd retention) ON TOP of the
    in-loop tuning state; when a device cannot hold both beside the reserve,
    the first backward OOMs only after a wasted grouped_mm attempt plus a
    sliced-fallback re-staging (the 4x3090 hy3 failure mode). The charges are
    the same ones the placement gates use: per-device tuning state charged
    at 6 of the 14 B/param layout (grads + bf16 are guaranteed in-loop; the
    values are already inside the probed free and the best-params snapshot
    parks to host under pressure) plus the per-device activation budget
    (routed transients on loop lanes, grouped stacks on grouped lanes).
    Any expert-homing device over budget -> switch the WHOLE run to
    linear_loop once, loudly. Explicit AR_MOE_EXPERTS_IMPL choices are never
    overridden.
    """
    global _MOE_IMPL_AUTO_LINEAR_LOOP_DONE, _MOE_IMPL_AUTO_DONE_REF
    import weakref

    _key_obj = config if config is not None else model
    try:
        _key = ("w", weakref.ref(_key_obj)) if _key_obj is not None else ("n", None)
    except TypeError:  # non-weakrefable object: fall back to identity
        _key = ("i", id(_key_obj))
    _prev = _MOE_IMPL_AUTO_DONE_REF
    _same = False
    if _prev is not None and _prev[0] == _key[0]:
        if _prev[0] == "w":
            _a, _b = _prev[1](), _key[1]()
            _same = _a is not None and _a is _b  # dead ref -> re-decide
        else:
            _same = _prev[1] == _key[1]
    if not _same:
        # a new (or garbage-collected) run's config re-decides: no silent
        # skip for a second model quantized in the same process
        _MOE_IMPL_AUTO_DONE_REF = _key
        _MOE_IMPL_AUTO_LINEAR_LOOP_DONE = False
    if _MOE_IMPL_AUTO_LINEAR_LOOP_DONE or iters is None or int(iters) <= 0:
        return
    from auto_round import envs as _envs_mod
    from auto_round.modeling.fused_moe.moe_experts_interface import GROUPED_LINEAR_IMPL, LINEAR_LOOP_IMPL
    from auto_round.utils.device import probe_usable_bytes
    from auto_round.utils.pool_placement import _RESERVE_BYTES

    requested = str(getattr(_envs_mod, "AR_MOE_EXPERTS_IMPL", "auto") or "auto").lower()
    if requested not in ("", "auto"):
        _MOE_IMPL_AUTO_LINEAR_LOOP_DONE = True  # explicit choice governs
        logger.debug("[moe-impl] auto check skipped: explicit AR_MOE_EXPERTS_IMPL=%s", requested)
        return
    # The dispatch reads the MODEL's config; model_context.config may be a
    # separate object that never received _experts_implementation (the exact
    # silent no-op the 4x3090 lane exposed). Prefer whichever candidate
    # actually carries the attr; switch ALL of them when we switch.
    candidates = [c for c in (config, getattr(model, "config", None)) if c is not None]
    cfg = next((c for c in candidates if getattr(c, "_experts_implementation", None)), None)
    impl = str(getattr(cfg, "_experts_implementation", GROUPED_LINEAR_IMPL)) if cfg is not None else GROUPED_LINEAR_IMPL
    if impl != GROUPED_LINEAR_IMPL:
        logger.debug("[moe-impl] auto check skipped: current impl is %s (not grouped)", impl)
        return
    try:
        state = _logical_state_by_device(block) or {}
        if not state:
            logger.debug("[moe-impl] auto check skipped: no per-device tuning state resolved")
            return
        act = _activation_bytes_by_device(block, tensors, batch_size, cfg) or {}
        if not any(_grouped_stack_bytes(block, d) > 0 for d in state):
            # dense block, loop impl already, or no experts homed: not the
            # decision point yet -- retry on the next block
            logger.debug("[moe-impl] auto check deferred: no grouped stacks homed on any state device")
            return
        over = []
        max_ratio_dev, max_ratio = None, -1.0
        for d, state_bytes in state.items():
            # values (4 B/param) are already inside the probed free here; of
            # the remainder, grads (4) + bf16 copy (2) are guaranteed in-loop,
            # while the best-params snapshot (4) PARKS TO HOST under pressure
            # (measured on the 5/6-GPU lanes: host RAM spikes as snapshots
            # stop fitting) -- charging it as guaranteed over-declined the
            # knife-edge-but-working 5x3090 grouped lane by ~3.5 GiB
            remaining_state = state_bytes * 6 // 14
            demand = remaining_state + act.get(d, 0) + _RESERVE_BYTES
            free = probe_usable_bytes(d)
            _st = _grouped_stack_bytes_detail(block, d)
            _est_routed = max(act.get(d, 0) - _st["retention"] - _st["transient"], 0)
            logger.debug(
                "[moe-impl] auto check %s: demand %.2f GiB (state %.2f + act %.2f "
                "[stacks retention %.2f + transient %.2f, est/routed %.2f] + reserve) vs free %.2f GiB",
                d,
                demand / 2**30,
                remaining_state / 2**30,
                act.get(d, 0) / 2**30,
                _st["retention"] / 2**30,
                _st["transient"] / 2**30,
                _est_routed / 2**30,
                -1.0 if free is None else free / 2**30,
            )
            if free is not None:
                ratio = demand / max(int(free), 1)
                if ratio > max_ratio:
                    max_ratio, max_ratio_dev = ratio, d
                if demand > int(free):
                    over.append((d, demand, int(free), remaining_state, act.get(d, 0)))
        _MOE_IMPL_AUTO_LINEAR_LOOP_DONE = True
        if over:
            d, demand, free, st, ac = over[0]
            logger.info(
                "[moe-impl] auto: grouped tuning needs %.1f GiB on %s (free %.1f: state %.1f + "
                "stacks/transients %.1f + reserve) -- switching this run to %s",
                demand / 2**30,
                d,
                free / 2**30,
                st / 2**30,
                ac / 2**30,
                LINEAR_LOOP_IMPL,
            )
            seen_cfg = {id(c) for c in candidates}
            _nm = getattr(block, "named_modules", None)
            for _n, m in (_nm() if callable(_nm) else []):
                # some archs deepcopy the config per layer; the dispatch
                # reads each module's own config, so switch those too
                c = getattr(m, "config", None)
                if c is not None and id(c) not in seen_cfg:
                    candidates.append(c)
                    seen_cfg.add(id(c))
            for c in candidates:
                c._experts_implementation = LINEAR_LOOP_IMPL
        elif max_ratio_dev is not None:
            logger.debug(
                "[moe-impl] auto: grouped fits (max demand/free ratio %.2f on %s)",
                max_ratio,
                max_ratio_dev,
            )
    except Exception as e:  # never break tuning over an advisory auto-pick
        logger.debug("[moe-impl] auto feasibility check failed (%s); keeping grouped", e)


def _pull_pool_if_fits(pool, target_dev, block, batch_size, iters, label, charge_activation=False, config=None):
    """Bulk-move a whole tune-loop pool onto the device that reads it every iteration.

    iters>0 strategy only (loop-amortized): ``active_inputs`` -> the entry
    device the forward gathers onto (BlockForwardRunner.device, home of the
    block's starting modules), ``fp_outputs`` -> the loss device. The iters=0
    lane never calls this -- its pools are single-pass streamed, so a bulk
    move buys nothing. Declines (pool stays sharded behind the per-batch
    gather) unless free(target) covers the pool beside the loop transients
    and the target's own in-loop tuning state (14 B/param), which
    materializes after this probe -- the term that separates the 8-GPU lane
    (pulls) from the 4-GPU lane (declines; peers completed the loop at the
    same state density without the pool). The loss side has no per-iteration
    graph retention (one cat per iteration, freed after the loss -- validated
    by the full 80-block pull run).

    ``charge_activation`` adds the block's ACTUAL activation accounting
    (per-module output shapes, MoE-ratio scaled -- see
    ``_block_activation_bytes``) and is set only for the input pull: the
    entry device carries the loop's forward graph (batch cats + routed-row
    caches materialize along the hidden flow regardless of weight homes).
    """
    if pool is None or target_dev is None:
        return pool
    _act_bytes = _block_activation_bytes(block, pool, batch_size, config, str(target_dev)) if charge_activation else 0
    if charge_activation and _act_bytes is None:
        return pool  # unknown activation cost: never pull blind
    _tgt = torch.device(target_dev)
    try:
        tensors = pool if isinstance(pool, list) else None
        if tensors is None or not tensors or not any(hasattr(t, "device") for t in tensors):
            return pool
        if all(t.device == _tgt for t in tensors):
            return pool
        pool_b = sum(t.numel() * t.element_size() for t in tensors)
        if charge_activation:
            pool_b += _act_bytes
        try:
            from auto_round.utils.device import probe_usable_bytes

            free = probe_usable_bytes(str(_tgt))
        except Exception as e:  # pragma: no cover - diagnostics only
            logger.warning("[%s] free-memory probe failed for %s (%s); keeping pool sharded", label, _tgt, e)
            free = None
        try:
            from auto_round.utils.pool_placement import _RESERVE_BYTES, _working_allowance_bytes

            reserve = _working_allowance_bytes(block, tensors, batch_size) + _RESERVE_BYTES
            reserve += _tuning_state_bytes(block, _tgt)
        except Exception as e:  # pragma: no cover - gate must never break tuning
            logger.warning("[%s] working-set estimate failed (%s); using flat 4GiB reserve", label, e)
            reserve = 4 << 30
        if free is not None and free - reserve >= pool_b:
            moved = [t.to(_tgt) for t in tensors]
            logger.debug("[%s bulk pull -> %s (%.2f GiB, free %.2f GiB)", label, _tgt, pool_b / 2**30, free / 2**30)
            return moved
        if free is not None:
            logger.debug(
                "[%s stays sharded, per-batch gather (data %.2f GiB + reserve %.2f GiB vs free %.2f GiB on %s)",
                label,
                pool_b / 2**30,
                reserve / 2**30,
                free / 2**30,
                _tgt,
            )
    except Exception as e:  # pragma: no cover - placement must never break tuning
        logger.warning("[%s bulk pull failed (%s); keeping data sharded", label, e)
    return pool


def _tune_phase_line(phases: dict, iters: int) -> str:
    """Format the per-block tuning phase breakdown for AR_PERF_COUNTERS.

    ``wrap`` = wrapper_block (params init, quant-func resolve + optional
    per-wrapper torch.compile, SignRoundV2 init-scale search -- batched
    same-shape when eligible); ``prepare`` = tuning-param collection +
    optimizer/scheduler build + sampler setup; ``loop`` = the iteration
    try/finally; ``tail`` = best-params restore, clear_memory, unwrapping.
    The optional ``loop`` split appears when counters ran: ``sampler`` =
    index draw, ``snap`` = best-params snapshots, ``step`` = gradient
    sync + optimizer step, ``serial: fwd+loss+bwd`` = the inline forward,
    loss (incl. the .item() drain) and backward of every serial batch.
    """
    line = "[perf] tune phases (iters=%d): wrap=%.2fs prepare=%.2fs loop=%.2fs tail=%.2fs" % (
        iters,
        phases.get("wrap", 0.0),
        phases.get("prepare", 0.0),
        phases.get("loop", 0.0),
        phases.get("tail", 0.0),
    )
    if "lp_sampler" in phases:
        line += " (loop: sampler=%.2fs snap=%.2fs step=%.2fs rest=%.2fs" % (
            phases["lp_sampler"],
            phases["lp_snap"],
            phases["lp_step"],
            phases["lp_rest"],
        )
        if phases.get("lp_serial", 0.0):
            line += " serial: fwd+loss+bwd=%.2fs" % phases["lp_serial"]
        line += ")"
    return line


from auto_round.utils import (
    htcore,
    is_hpex_available,
    mv_module_from_gpu,
    set_amax_for_all_moe_layers,
)
from auto_round.utils.device import clear_memory_if_reached_threshold
from auto_round.utils.device_manager import device_manager
from auto_round.utils.distributed import setup_ddp_if_needed_
from auto_round.wrapper import WrapperLinear, unwrapper_block, unwrapper_layer, wrapper_block

if TYPE_CHECKING:
    from auto_round.algorithms.composer import BlockContext


@register_pipeline_member(SignRoundConfig)
class SignRoundQuantizer(BaseQuantizer):

    def __init__(self, config: SignRoundConfig) -> None:
        super().__init__(config)
        self.iters = config.iters
        self.lr = config.lr
        self.minmax_lr = config.minmax_lr
        self.lr_scheduler = config.lr_scheduler
        self.momentum = config.momentum
        self.enable_minmax_tuning = config.enable_minmax_tuning
        self.enable_norm_bias_tuning = config.enable_norm_bias_tuning
        self.gradient_accumulate_steps = config.gradient_accumulate_steps

        self.enable_alg_ext = config.enable_alg_ext
        self.not_use_best_mse = config.not_use_best_mse
        self.enable_quanted_input = config.enable_quanted_input
        self.dynamic_max_gap = config.dynamic_max_gap
        self.enable_lfq = config.enable_lfq

        self.optimizer = self._get_optimizer(optimizer=config.optimizer)
        self.wrapper_block = wrapper_block
        # Kept for per-layer (mixed-bit) lr resolution during tuning.
        self._config = config
        self.lr_is_auto = getattr(config, "lr_is_auto", False)
        self.minmax_lr_is_auto = getattr(config, "minmax_lr_is_auto", False)
        # Emit the low-bit lr notice at most once across all blocks/layers.
        self._logged_low_bit_lr = False

    def _maybe_log_low_bit_lr(self, bits) -> None:
        """Log once when low-bit (<=3) layers get the higher 2.0/iters lr."""
        if self._logged_low_bit_lr or not self.lr_is_auto:
            return
        if self.iters >= 1000 and bits is not None and bits <= 3:
            logger.info("using higher lr (2.0/iters) for <=3 bit layers to improve accuracy")
            self._logged_low_bit_lr = True

    def dispatch_block(self, block, input_ids, input_others):
        """Multi-GPU aware block dispatch for SignRound tuning.

        Stores _card_0_in_high_risk and _loss_device on self for use in quantize_block.
        Returns the block after device placement.
        """
        from auto_round.utils import is_auto_device_mapping

        if (
            is_auto_device_mapping(device_manager.device_map)
            and len(device_manager.device_list) > 1
            and not self.model_context.is_diffusion
        ):
            from auto_round.utils.device import set_auto_device_map_for_block_with_tuning

            card_0_in_high_risk, loss_device = set_auto_device_map_for_block_with_tuning(
                block,
                device_manager.device_list,
                input_ids,
                self.compress_context.low_gpu_mem_usage,
                self.calibration_context.batch_size,
                device_manager.device,
            )
            if len(device_manager.device_list) > 1:
                from accelerate.hooks import AlignDevicesHook, add_hook_to_module

                for _n, _mod in block.named_modules():
                    if len(list(_mod.children())) != 0 or not hasattr(_mod, "tuning_device"):
                        continue
                    add_hook_to_module(_mod, AlignDevicesHook(_mod.tuning_device, io_same_device=True), True)
        else:
            from auto_round.utils.model import move_to_device_preserving_cpu_pinned, place_ngram_embeddings_for_tuning_

            # Honor AR_NGRAM_DEVICE even on a single GPU (default keeps the table on CPU and
            # logs a hint); this also surfaces the ngram size / placement info to the user.
            place_ngram_embeddings_for_tuning_(block)
            block = move_to_device_preserving_cpu_pinned(block, device_manager.device)
            card_0_in_high_risk, loss_device = False, device_manager.device

        self._card_0_in_high_risk = card_0_in_high_risk
        self._loss_device = loss_device
        return block

    def _get_non_zero_cnt(self, tensor: list[torch.Tensor], indices: list[int]) -> int:
        current_tensors = [tensor[i] for i in indices]
        non_zero_cnt = 0
        for t in current_tensors:
            non_zero_cnt += torch.count_nonzero(t).item()
        return non_zero_cnt

    def _get_loss(
        self,
        pred_output: torch.Tensor,
        ref_output: torch.Tensor,
        indices: torch.Tensor,
        loss_func: Callable,
        device: Union[str, torch.device] = "cpu",
        valid_token_mask: Optional[torch.Tensor] = None,
        input_ids=None,
    ):
        autocast_ctx = (
            nullcontext()
            if self.model_context.amp
            else autocast(device_type=str(device).split(":")[0], dtype=self.model_context.amp_dtype)
        )
        if valid_token_mask:
            tmp_attention_mask = [valid_token_mask[i] for i in indices]
            tmp_attention_mask = torch.cat(tmp_attention_mask, dim=0).to(device)
            tmp_attention_mask.unsqueeze_(-1)

            with autocast_ctx:
                loss = loss_func(  # pylint: disable=not-callable
                    (pred_output * tmp_attention_mask).to(torch.float32),
                    (ref_output * tmp_attention_mask).to(torch.float32),
                )
        else:
            with autocast_ctx:
                loss = loss_func(  # pylint: disable=not-callable
                    pred_output.to(torch.float32), ref_output.to(torch.float32)
                )

        return loss

    def _count_samples(self, inputs: Any) -> int:
        if isinstance(inputs, dict):
            hs = inputs.get("hidden_states")
            return len(hs) if isinstance(hs, list) else hs.shape[self.calibration_context.batch_dim]
        elif isinstance(inputs, list):
            return len(inputs)
        else:
            return inputs.shape[self.calibration_context.batch_dim]

    def _init_lm_components(self) -> None:
        """Lazily locate and cache ``_post_block_modules`` and ``_lm_head`` for LFQ loss.

        ``_post_block_modules`` is an ordered list of modules that must be applied
        between the last transformer-block output and the lm_head projection.
        Most architectures have a single final norm; OPT additionally has an
        optional ``project_out`` projection.

        Supported without manual configuration:
            LLaMA / Qwen / Gemma / Mistral / InternLM / Phi-3 — ``model.model.norm``
            OPT — ``model.model.decoder.{final_layer_norm, project_out}`` (both optional)
            GPT-2 / Falcon / Bloom — ``model.transformer.ln_f``
            GPT-NeoX / Pythia — ``model.gpt_neox.final_layer_norm``
            Phi / Phi-2 — ``model.model.final_layernorm``
            MPT — ``model.transformer.norm_f``
            ChatGLM — ``model.transformer.encoder.final_layernorm``
            RWKV — ``model.rwkv.ln_out``

        Raises ``AttributeError`` if no lm_head equivalent can be found.
        """
        if hasattr(self, "_lm_head"):
            return

        model = self.model

        # ── lm_head ──────────────────────────────────────────────────────────
        for name in ("lm_head", "embed_out", "output", "head"):
            if hasattr(model, name):
                self._lm_head = getattr(model, name)
                break
        else:
            raise AttributeError(
                f"Cannot locate lm_head in {type(model).__name__}. " "Checked: lm_head, embed_out, output, head."
            )

        # ── post-block processing (ordered list applied before lm_head) ───────
        # OPT: decoder has both an optional final_layer_norm *and* an optional
        # project_out that maps ffn_dim → word_embed_proj_dim.  Detect by probing
        # for the characteristic project_out attribute (may be None).
        try:
            decoder = model.model.decoder
            _ = decoder.project_out  # raises AttributeError if not an OPT decoder
            self._post_block_modules = [m for m in (decoder.final_layer_norm, decoder.project_out) if m is not None]
            return
        except AttributeError:
            pass

        # All other architectures: single optional final norm.
        norm_getters = [
            lambda: model.model.norm,  # LLaMA / Qwen / Gemma / Mistral / InternLM / Phi-3
            lambda: model.transformer.ln_f,  # GPT-2 / Falcon / Bloom
            lambda: model.gpt_neox.final_layer_norm,  # GPT-NeoX / Pythia
            lambda: model.model.final_layernorm,  # Phi / Phi-2
            lambda: model.transformer.norm_f,  # MPT
            lambda: model.transformer.encoder.final_layernorm,  # ChatGLM
            lambda: model.rwkv.ln_out,  # RWKV
        ]
        self._post_block_modules = []
        for getter in norm_getters:
            try:
                norm = getter()
                if norm is not None:
                    self._post_block_modules = [norm]
                    break
            except AttributeError:
                continue

    # Keywords that identify non-text (visual / audio / multimodal) blocks.
    # LFQ loss is only meaningful for pure language-model decoder blocks.
    _NON_TEXT_BLOCK_KEYWORDS = frozenset(
        {
            "vis",
            "vision",
            "visual",
            "image",
            "img",
            "audio",
            "video",
            "patch",
            "pixel",
            "clip",
            "vit",
            "perceiver",
            "resampler",
            "connector",
            "projector",
        }
    )

    def _is_text_decoder_block(self, block_name: str) -> bool:
        """Return ``True`` if *block_name* refers to a text-decoder block.

        Blocks whose names contain any of the non-text keywords (vision, audio,
        image, …) are considered multimodal and excluded from LFQ loss.
        """
        name_lower = block_name.lower()
        return not any(kw in name_lower for kw in self._NON_TEXT_BLOCK_KEYWORDS)

    def lfq_loss(self, hidden_state: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute LM cross-entropy loss from the last block's hidden states.

        Applies every post-block module (final norm, optional projection, …) in
        order, then runs lm_head and computes next-token prediction loss.
        Positions marked with ``-100`` in *input_ids* are excluded from the loss.

        Args:
            hidden_state: Last block output, shape ``[batch, seq_len, hidden]``.
            input_ids:    Token-ID labels with ``-100`` for ignored positions,
                          shape ``[batch, seq_len]``.

        Returns:
            Scalar cross-entropy loss tensor.
        """
        self._init_lm_components()
        device = hidden_state.device

        for module in self._post_block_modules:
            module.to(device)
            hidden_state = module(hidden_state)

        self._lm_head.to(device)
        logits = self._lm_head(hidden_state)

        if hasattr(self.model, "loss_function"):
            loss = self.model.loss_function(
                logits=logits,
                labels=input_ids.to(device),
                vocab_size=self.model.config.vocab_size,
            )
        else:
            import torch.nn.functional as F

            # Standard causal-LM shift: predict token t+1 from hidden state t.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous().to(device)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return loss

    def quantize_block(
        self,
        block,
        fp_inputs,
        input_others,
        fp_outputs,
        q_inputs,
        block_ctx,
        input_ids=None,
        **kwargs,
    ) -> dict:
        """Apply the AutoRound optimization algorithm to a block.

        This is the pure-algorithm entry point.  All infrastructure concerns
        (device placement, act-max hook collection, DDP setup, memory cleanup,
        logging) are handled by the Compressor before and after this call.

        Args:
            block: The transformer block module to quantize.
            fp_inputs: FP calibration inputs for this block (list[Tensor] or dict
                for diffusion models).
            input_others: Auxiliary kwargs passed to the block forward
                (e.g. attention_mask, position_ids).
            fp_outputs: FP reference outputs of the block used as the optimization
                target for the sign-gradient descent loss (list[Tensor]).
            q_inputs: Quantized inputs from the previous block, or ``None`` when
                cascaded quantized-input is disabled.
            block_ctx: Per-block pipeline context (BlockContext).
            input_ids: Raw token IDs from the tokenizer (``[1, seq_len]`` per
                sample). Used to derive the valid-token loss mask once (result
                cached on ``self._cached_valid_token_mask`` for reuse across
                all blocks). ``None`` disables loss masking.
            **kwargs: Reserved for forward-compatibility with future parameters.

        Returns:
            dict: Best quantization parameters found during optimization, or an
                empty dict if no trainable parameters were found.
        """
        device = device_manager.device
        loss_device = getattr(self, "_loss_device", device)
        card_0_in_high_risk = getattr(self, "_card_0_in_high_risk", False)
        mid_iter_mem_check = self.compress_context.low_gpu_mem_usage and card_0_in_high_risk

        valid_token_mask = None
        # Derive valid_token_mask from raw token IDs when not supplied by caller.
        # Result is cached on self so it is computed only once across all blocks.
        if input_ids is not None:
            if not hasattr(self, "_cached_valid_token_mask"):
                self._cached_valid_token_mask = self._compute_valid_token_mask(input_ids)
            valid_token_mask = self._cached_valid_token_mask

        # Use quantized inputs if available and enabled
        active_inputs = q_inputs if (q_inputs is not None and self.enable_quanted_input) else fp_inputs
        nsamples = len(active_inputs) if isinstance(active_inputs, list) else self._count_samples(active_inputs)

        import auto_round.envs as _envs

        _tune_perf = (
            {"wrap": 0.0, "prepare": 0.0, "loop": 0.0, "tail": 0.0}
            if getattr(_envs, "AR_PERF_COUNTERS", False)
            else None
        )
        _loop_perf = (
            {"sampler": 0.0, "snap": 0.0, "step": 0.0, "s_fwd": 0.0, "s_loss": 0.0, "s_bwd": 0.0}
            if _tune_perf is not None
            else None
        )
        _tp0 = _ptime.perf_counter()
        quantized_layer_names, unquantized_layer_names = self.wrapper_block(
            block,
            self.enable_minmax_tuning,
            self.enable_norm_bias_tuning,
            enable_torch_compile=self.compress_context.enable_torch_compile,
            device=device,
        )
        if _tune_perf is not None:
            _tune_perf["wrap"] = _ptime.perf_counter() - _tp0
            _tp1 = _ptime.perf_counter()

        round_params = []
        minmax_params = []
        # Group parameters by their effective lr so that mixed-bit configs
        # (e.g. a 4-bit model with a few 2-bit layers) use a per-layer lr
        # derived from each layer's own bit-width.
        round_lr_groups: dict[float, list] = {}
        minmax_lr_groups: dict[float, list] = {}
        for n, m in block.named_modules():
            if hasattr(m, "orig_layer"):
                layer_bits = getattr(m.orig_layer, "bits", None)
                layer_lr = self._config.compute_lr(layer_bits)
                if layer_lr is None:
                    layer_lr = self.lr
                self._maybe_log_low_bit_lr(layer_bits)
                layer_minmax_lr = self._config.compute_minmax_lr(layer_bits)
                if layer_minmax_lr is None:
                    layer_minmax_lr = self.minmax_lr
                for key in m.params.keys():
                    if "min" in key or "max" in key:
                        minmax_params.append(m.params[key])
                        minmax_lr_groups.setdefault(float(layer_minmax_lr), []).append(m.params[key])
                    else:
                        round_params.append(m.params[key])
                        round_lr_groups.setdefault(float(layer_lr), []).append(m.params[key])

        lr = torch.tensor(self.lr)
        minmax_lr = torch.tensor(self.minmax_lr)

        extra_kwargs = {} if self.momentum is None else {"momentum": self.momentum}

        if len(round_params) + len(minmax_params) <= 0:
            dump_info = (
                f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
                f"layers in the block"
            )
            logger.info(dump_info)
            unwrapper_block(block, {})
            return {}

        # Build optimizer param groups with a per-layer lr for the rounding
        # parameters (and min-max parameters when enabled).
        params = [{"params": ps, "lr": torch.tensor(group_lr)} for group_lr, ps in round_lr_groups.items()]
        if self.enable_minmax_tuning:
            params += [{"params": ps, "lr": torch.tensor(group_lr)} for group_lr, ps in minmax_lr_groups.items()]

        optimizer = self.optimizer(
            params,
            lr=lr,
            weight_decay=0,
            **extra_kwargs,
        )

        if self.lr_scheduler is None:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.iters
            )
        else:
            lr_schedule = copy.deepcopy(self.lr_scheduler)

        last_best_iter = 0
        best_loss = torch.finfo(torch.float).max
        num_elm = 1
        mse_reduction = "mean"
        if self.gradient_accumulate_steps != 1:
            mse_reduction = "sum"
        mse_loss = torch.nn.MSELoss(reduction=mse_reduction).to(device)
        scaler = self._get_scaler()  # pylint: disable=assignment-from-none
        init_loss = None
        best_params = {}
        total_loss = 0
        batch_size = self.calibration_context.batch_size
        global_batch_size = batch_size * self.gradient_accumulate_steps
        global_batch_size = min(nsamples, global_batch_size)
        # Compute num_elm once before the loop (used to normalise the accumulated loss).
        # We assume the block input and output shape is same
        if self.gradient_accumulate_steps != 1 and not valid_token_mask:
            whole_indices = torch.arange(global_batch_size)
            if isinstance(active_inputs, list):  # dict for diffusion, tricky setting, not sure whether it's correct
                num_elm = sum(active_inputs[i.item()].numel() for i in whole_indices)

        block, sync_gradients = setup_ddp_if_needed_(self, block, device_manager.device_list)
        index_sampler = IndexSampler(nsamples, global_batch_size)
        block_fwd = self.block_forward

        # When low_gpu_mem_usage is enabled, active_inputs / fp_outputs are intentionally
        # kept on CPU to limit GPU memory.  However block_fwd normally routes pred_output
        # through CPU (cache_device="cpu") and the very next line moves it back to
        # loss_device — a wasteful GPU→CPU→GPU roundtrip on every batch × iteration.
        # pred_output is a transient single-batch tensor consumed immediately for the
        # loss and then freed, so keeping it on the compute device costs no persistent
        # extra memory.  Pass it as a per-call override so self.cache_device is unchanged.
        _fwd_cache_device = (
            device
            if getattr(self.compress_context, "low_gpu_mem_usage", False) and not str(device).startswith("cpu")
            else None
        )

        tuning_cache = None
        # Only opt-in diffusion tuning can enter the CUDA staging path.
        cache_budget = getattr(self.model_context, "diffusion_tuning_cache_size", 0)
        use_tuning_cache = (
            getattr(self.model_context, "is_diffusion", False)
            and (cache_budget == "auto" or cache_budget > 0)
            and self.compress_context.low_gpu_mem_usage
            and str(device).startswith("cuda")
            and len(device_manager.device_list) == 1
            and (loss_device is None or torch.device(loss_device) == torch.device(device))
        )

        if _tune_perf is not None:
            _tune_perf["prepare"] = _ptime.perf_counter() - _tp1
            _tp2 = _ptime.perf_counter()
        # iters>0 hot-pool strategy (loop-amortized; the iters=0 lane streams
        # its pools once and never reaches these pulls): each pool moves in
        # bulk onto the device that reads it every iteration -- the input
        # pool onto the entry device (BlockForwardRunner.device, where the
        # forward gathers each batch), the fp reference pool onto the loss
        # device. Both gates charge ACTUAL data: exact tuning state from the
        # wrapper walker, and (input pull) the routed-buffer budget proven on
        # the mapped-streaming lane -- tokens x top_k hidden rows as in/out
        # accumulators plus the backward grad (x6), which reproduces the
        # measured 12.7 GiB loop retention on hy3 from shapes alone. Declined
        # pulls keep the per-batch gather (measured ~1-2 s/block).
        if (getattr(self, "iters", 0) or 0) > 0:
            _maybe_auto_linear_loop_for_tuning(
                block,
                active_inputs,
                batch_size,
                self.iters,
                getattr(getattr(self, "model_context", None), "config", None),
                getattr(self, "model", None),
            )
            _entry_dev = str(getattr(block_fwd, "device", device)) if block_fwd is not None else str(device)
            if isinstance(active_inputs, list):
                active_inputs = _pull_pool_if_fits(
                    active_inputs,
                    _entry_dev,
                    block,
                    batch_size,
                    self.iters,
                    "tune] block input activations",
                    charge_activation=True,
                    config=getattr(getattr(self, "model_context", None), "config", None),
                )
            # the diffusion tuning cache requires its pools on the host (it
            # stages batches to the GPU itself via pinned slots); a bulk pull
            # onto the loss device would put the cached reference outputs on
            # cuda and the cache would decline every batch signature
            if fp_outputs and loss_device is not None and not use_tuning_cache:
                fp_outputs = _pull_pool_if_fits(
                    fp_outputs,
                    str(loss_device),
                    block,
                    batch_size,
                    self.iters,
                    "tune] fp reference outputs",
                )

        try:
            for i in range(self.iters):
                # Auto observes a complete forward/backward/optimizer iteration
                # on the legacy path before allocating any extra GPU buffers.
                if use_tuning_cache and i == (1 if cache_budget == "auto" else 0):
                    from auto_round.compressors.diffusion.tuning_cache import DiffusionTuningCache

                    tuning_cache = DiffusionTuningCache.create(
                        block,
                        block_fwd,
                        active_inputs,
                        input_others,
                        fp_outputs,
                        index_sampler,
                        self.iters - i,
                        cache_budget,
                        device,
                    )
                if self.enable_alg_ext and self.scheme.data_type.endswith("dq"):
                    for n, m in block.named_modules():
                        m.cur_iter = i
                total_loss = 0
                if _loop_perf is not None:
                    _smp_t0 = _ptime.perf_counter()
                global_indices = index_sampler.next_batch()
                if valid_token_mask:
                    num_elm = self._get_non_zero_cnt(valid_token_mask, global_indices)
                if _loop_perf is not None:
                    _loop_perf["sampler"] += _ptime.perf_counter() - _smp_t0

                for batch_start in range(0, len(global_indices), batch_size):
                    indices = global_indices[batch_start : batch_start + batch_size]
                    if _loop_perf is not None:
                        _sf_t0 = _ptime.perf_counter()
                    staged = tuning_cache.get(indices) if tuning_cache is not None else None
                    if staged is None:
                        # fp_outputs may be sharded across park devices
                        # (--calibration_data_device): gather onto one device
                        # before cat. Uniform-device selections are returned
                        # untouched (byte-identical to the single-device path);
                        # tensors are immutable, so concurrent gather is safe.
                        _sel = [fp_outputs[i] for i in indices]
                        _gather = BlockForwardRunner._gather_same_device(
                            _sel, str(loss_device) if loss_device is not None else str(_sel[0].device)
                        )
                        ref_output = torch.cat(_gather, dim=0).to(loss_device)
                        pred_output = block_fwd.forward(block, active_inputs, input_others, indices, _fwd_cache_device)
                    else:
                        ref_output = staged[2]
                        pred_output = tuning_cache.forward(block, staged, _fwd_cache_device)
                    if loss_device is not None:
                        pred_output = pred_output.to(loss_device)
                    if _loop_perf is not None:
                        _loop_perf["s_fwd"] += _ptime.perf_counter() - _sf_t0
                        _sl_t0 = _ptime.perf_counter()
                    if (
                        block_ctx.block_index == block_ctx.block_cnt - 1
                        and self.enable_lfq
                        and input_ids is not None
                        and self._is_text_decoder_block(block_ctx.block_name)
                    ):
                        loss = self.lfq_loss(pred_output, torch.cat([input_ids[i] for i in indices], dim=0))
                    else:
                        loss = self._get_loss(pred_output, ref_output, indices, mse_loss, device, valid_token_mask)
                    num_elm = 1 if num_elm <= 0 else num_elm
                    total_loss += loss.item() / num_elm
                    if _loop_perf is not None:
                        _loop_perf["s_loss"] += _ptime.perf_counter() - _sl_t0
                        _sb_t0 = _ptime.perf_counter()

                    if mid_iter_mem_check:
                        # clear memory to avoid OOM due to memory fragmentation
                        clear_memory_if_reached_threshold(threshold=0.5, device_list=device_manager.device_list)

                    self._scale_loss_and_backward(scaler, loss)
                    if _loop_perf is not None:
                        _loop_perf["s_bwd"] += _ptime.perf_counter() - _sb_t0

                    if mid_iter_mem_check:
                        # clear memory to avoid OOM due to memory fragmentation
                        clear_memory_if_reached_threshold(threshold=0.8, device_list=device_manager.device_list)

                if i == 0:
                    init_loss = total_loss
                current_lr = optimizer.param_groups[0]["lr"]
                logger.debug("iter %d loss: %.3e lr: %s", i, total_loss, current_lr)

                if _loop_perf is not None:
                    _snap_t0 = _ptime.perf_counter()
                if total_loss < best_loss:
                    best_loss = total_loss
                    if not self.not_use_best_mse:
                        best_params = (
                            tuning_cache.collect_best_params()
                            if tuning_cache is not None and tuning_cache.best is not None
                            else snapshot_best_params(block, self.compress_context.cache_device)
                        )
                        last_best_iter = i
                if self.not_use_best_mse and i == self.iters - 1:
                    best_params = (
                        tuning_cache.collect_best_params()
                        if tuning_cache is not None and tuning_cache.best is not None
                        else snapshot_best_params(block, self.compress_context.cache_device)
                    )

                if _loop_perf is not None:
                    _loop_perf["snap"] += _ptime.perf_counter() - _snap_t0
                    _stp_t0 = _ptime.perf_counter()
                if not self.not_use_best_mse:
                    if 0 < self.dynamic_max_gap <= i - last_best_iter:
                        break
                sync_gradients()
                self._step(scaler, optimizer, lr_schedule)
                if _loop_perf is not None:
                    _loop_perf["step"] += _ptime.perf_counter() - _stp_t0

        finally:
            if tuning_cache is not None:
                tuning_cache.close()
        if _tune_perf is not None:
            _tune_perf["loop"] = _ptime.perf_counter() - _tp2
            _tp3 = _ptime.perf_counter()

        last_loss = total_loss
        best_iter = self.iters
        if not self.not_use_best_mse:
            last_loss = best_loss
            best_iter = last_best_iter
        if self.iters > 0:
            dump_info = (
                f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
                f"layers in the block, loss iter 0: {init_loss:.3e} -> iter {best_iter}: {last_loss:.3e}"
            )
        else:
            dump_info = (
                f"quantized {len(quantized_layer_names)}/{(len(quantized_layer_names) + len(unquantized_layer_names))} "
                "layers in the block"
            )

        self.compress_context.clear_memory()  # clear cached memory during training
        if len(unquantized_layer_names) != 0:
            logger.info(f"Unquantized layers: {unquantized_layer_names}")
        with torch.no_grad():
            unwrapper_block(block, best_params)

        if _tune_perf is not None:
            _tune_perf["tail"] = _ptime.perf_counter() - _tp3
            _serial = _loop_perf["s_fwd"] + _loop_perf["s_loss"] + _loop_perf["s_bwd"]
            _rest = max(
                _tune_perf["loop"] - _loop_perf["sampler"] - _loop_perf["snap"] - _loop_perf["step"] - _serial,
                0.0,
            )
            _tune_perf.update(
                {
                    "lp_sampler": _loop_perf["sampler"],
                    "lp_snap": _loop_perf["snap"],
                    "lp_step": _loop_perf["step"],
                    "lp_rest": _rest,
                    "lp_serial": _serial,
                }
            )
            logger.info("%s", _tune_phase_line(_tune_perf, self.iters))

        if self.config.is_act_nv_fp:
            # enable moe experts act_max automatic generation for WrapperWALayer
            set_amax_for_all_moe_layers(block, attr_name="orig_layer.act_max")

        logger.infoclean(dump_info)
        return best_params

    def quantize_layer_outside_block(
        self,
        layer: "torch.nn.Module",
        fp_inputs: Optional[list[torch.Tensor]] = None,
        q_inputs: Optional[list[torch.Tensor]] = None,
        disable_opt_rtn: Optional[bool] = None,
        input_ids: Optional[list[torch.Tensor]] = None,
    ):
        """Quantize a single layer that lives outside a transformer block.

        When ``fp_inputs`` is provided the layer is tuned with the sign-gradient
        descent optimizer (same loss loop as block-level quantization).  When
        ``fp_inputs`` is ``None`` the method falls back to zero-shot RTN.

        Args:
            layer: The layer module to quantize.  Must have a ``global_name``
                attribute for model re-insertion and logging.
            fp_inputs: Per-sample FP activations fed into this layer, used as
                calibration inputs during optimization. ``None`` triggers RTN
                fallback.
            q_inputs: Per-sample quantized activations from the previous stage,
                used instead of ``fp_inputs`` during the forward pass when
                cascaded quantized-input is enabled. ``None`` means use
                ``fp_inputs`` for both reference and tuning forward.
            disable_opt_rtn: Override optimized-RTN; ``None`` defers to quantizer config.
            input_ids: Raw token IDs from the tokenizer (``[1, seq_len]`` per
                sample); used to derive the valid-token loss mask via
                ``_compute_valid_token_mask``. ``None`` disables loss masking.
        """

        layer_name = layer.global_name
        if fp_inputs is None:
            logger.info(f"using rtn to quantize {layer_name}")
            self._quantize_layer_via_rtn(
                layer,
                disable_opt_rtn=(
                    disable_opt_rtn if disable_opt_rtn is not None else getattr(self.config, "disable_opt_rtn", True)
                ),
            )
            return

        # Derive valid_token_mask from raw token IDs when not supplied by caller.
        # Reuse the cached mask if already computed by a previous block.
        valid_token_mask = None
        if input_ids is not None:
            if not hasattr(self, "_cached_valid_token_mask"):
                self._cached_valid_token_mask = self._compute_valid_token_mask(input_ids)
            valid_token_mask = self._cached_valid_token_mask

        logger.info(f"quantizing layer {layer_name}")
        # Layer is already on the correct device (placed by the caller / AlgorithmComposer).
        device = layer.weight.device if hasattr(layer, "weight") else device_manager.device
        for i in range(len(fp_inputs)):
            fp_inputs[i] = fp_inputs[i].to(layer.weight.dtype)
            if q_inputs is not None:
                q_inputs[i] = q_inputs[i].to(layer.weight.dtype)

        wrapper_linear = WrapperLinear(
            layer,
            enable_minmax_tuning=self.enable_minmax_tuning,
            enable_torch_compile=self.compress_context.enable_torch_compile,
            device=device,
        ).to(device)
        round_params = []
        minmax_params = []
        for key in wrapper_linear.params.keys():
            if "min" in key or "max" in key:
                minmax_params.append(wrapper_linear.params[key])
            else:
                round_params.append(wrapper_linear.value)
        if len(round_params) + len(minmax_params) <= 0:
            dump_info = f"quantized {layer_name}"
            logger.info(dump_info)
            with torch.no_grad():
                unwrapper_layer(self.model, wrapper_linear, layer_name, {})
            mv_module_from_gpu(layer)

        lr = torch.tensor(self.lr)
        minmax_lr = torch.tensor(self.minmax_lr)
        # Use a lr derived from this layer's own bit-width so mixed-bit configs
        # (e.g. a 4-bit model with a few 2-bit layers) tune each layer correctly.
        layer_bits = getattr(layer, "bits", None)
        layer_lr = self._config.compute_lr(layer_bits)
        if layer_lr is not None:
            lr = torch.tensor(layer_lr)
        self._maybe_log_low_bit_lr(layer_bits)
        layer_minmax_lr = self._config.compute_minmax_lr(layer_bits)
        if layer_minmax_lr is not None:
            minmax_lr = torch.tensor(layer_minmax_lr)
        if self.enable_minmax_tuning:
            optimizer = self.optimizer(
                [{"params": round_params}, {"params": minmax_params, "lr": minmax_lr}], lr=lr, weight_decay=0
            )
        else:
            optimizer = self.optimizer(round_params, lr=lr, weight_decay=0)

        if self.lr_scheduler is None:
            lr_schedule = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.iters
            )
        else:
            lr_schedule = copy.deepcopy(self.lr_scheduler)
        nsamples = len(fp_inputs)
        last_best_iter = 0
        best_loss = torch.finfo(torch.float).max
        best_params = None
        scaler = self._get_scaler()  # pylint: disable=assignment-from-none
        init_loss = None

        gradient_accumulate_steps = (
            self.calibration_context.batch_size * self.gradient_accumulate_steps
        )  # Force to low gpu

        total_loss = 0
        num_elm = 1
        mse_reduction = "mean"
        if gradient_accumulate_steps != 1:
            mse_reduction = "sum"
        mse_loss = torch.nn.MSELoss(reduction=mse_reduction).to(device)
        batch_size = 1  # Force to low gpu
        global_batch_size = gradient_accumulate_steps
        global_batch_size = min(nsamples, global_batch_size)
        # Compute num_elm once before the loop.
        if gradient_accumulate_steps != 1:
            whole_indices = list(range(global_batch_size))
            if valid_token_mask:
                num_elm = self._get_non_zero_cnt(valid_token_mask, whole_indices)
            elif q_inputs is not None:
                num_elm = self._count_layer_input_elements(q_inputs, whole_indices)
            else:
                num_elm = self._count_layer_input_elements(fp_inputs, whole_indices)

        index_sampler = IndexSampler(nsamples, global_batch_size)

        for i in range(self.iters):
            total_loss = 0
            global_indices = index_sampler.next_batch()

            for batch_start in range(0, len(global_indices), batch_size):
                indices = global_indices[batch_start : batch_start + batch_size]
                if q_inputs is not None:
                    current_input = [q_inputs[i] for i in indices]
                    current_input = torch.cat(current_input, dim=0).to(device)
                    org_input = [fp_inputs[i] for i in indices]
                    org_input = torch.cat(org_input, dim=0).to(device)
                else:
                    current_input = [fp_inputs[i] for i in indices]
                    current_input = torch.cat(current_input, dim=0).to(device)
                    org_input = current_input
                with torch.no_grad():
                    current_output = layer(org_input)
                autocast_ctx = (
                    nullcontext()
                    if not self.model_context.amp
                    else autocast(device_type=str(device).split(":")[0], dtype=self.model_context.amp_dtype)
                )
                if valid_token_mask:
                    tmp_valid_mask = [valid_token_mask[i] for i in indices]
                    tmp_valid_mask = torch.cat(tmp_valid_mask, dim=0).to(device)
                    tmp_valid_mask.unsqueeze_(-1)

                    with autocast_ctx:
                        output_q = wrapper_linear(current_input)  # pylint: disable=not-callable
                        loss = mse_loss(  # pylint: disable=not-callable
                            (output_q * tmp_valid_mask).to(torch.float32),
                            (current_output * tmp_valid_mask).to(torch.float32),
                        )

                else:
                    with autocast_ctx:
                        output_q = wrapper_linear(current_input)  # pylint: disable=not-callable
                        loss = mse_loss(  # pylint: disable=not-callable
                            output_q.to(torch.float32),
                            current_output.to(torch.float32),  # mul 1.0 will copy the output
                        )

                num_elm = 1 if num_elm <= 0 else num_elm
                total_loss += loss.item() / num_elm

                self._scale_loss_and_backward(scaler, loss)
            if i == 0:
                init_loss = total_loss
            current_lr = optimizer.param_groups[0]["lr"]
            logger.debug("iter %d loss: %.3e lr: %s", i, total_loss, current_lr)

            if total_loss < best_loss:
                best_loss = total_loss
                if not self.not_use_best_mse:
                    best_params = collect_best_params(wrapper_linear, self.compress_context.cache_device)
                    last_best_iter = i
            if self.not_use_best_mse and i == self.iters - 1:
                best_params = collect_best_params(wrapper_linear, self.compress_context.cache_device)

            if not self.not_use_best_mse:
                if 0 < self.dynamic_max_gap <= i - last_best_iter:
                    break
            self._step(scaler, optimizer, lr_schedule)

        last_loss = total_loss
        best_iter = self.iters
        if not self.not_use_best_mse:
            last_loss = best_loss
            best_iter = last_best_iter
        with torch.no_grad():
            unwrapper_layer(self.model, wrapper_linear, layer_name, best_params)
        mv_module_from_gpu(layer)
        dump_info = f"quantized {layer_name},  loss iter 0: {init_loss:.3e} -> iter {best_iter}: {last_loss:.3e}"
        logger.info(dump_info)

    def finalize_run(self) -> None:
        """Clear per-run caches (``_cached_valid_token_mask``, LFQ components)."""
        for attr in ("_cached_valid_token_mask", "_lm_head", "_post_block_modules"):
            if hasattr(self, attr):
                delattr(self, attr)

    def _get_optimizer(self, optimizer: Any):
        """Returns the specified optimizer. In SignRound, we fix the optimizer.

        Args:
        optimizer: The optimizer to be used.

        Returns:
        The specified optimizer.
        """
        if optimizer is not None:
            logger.warning_once(
                "The optimizer setting in config will be ignored in AutoRound, using SignSGD as default."
            )
        return SignSGD

    def _count_layer_input_elements(self, input_ids, indices: list) -> int:
        return sum(input_ids[i].numel() for i in indices)

    def _get_scaler(self):
        """Returns scaler, in SignRound, no need to use scaler."""
        return None

    def _scale_loss_and_backward(self, scaler: Any, loss: torch.Tensor) -> torch.Tensor:
        """Scales the loss and performs backward pass.

        Args:
        scaler: The scaler to be used.
        loss: The loss to be scaled.

        Returns:
        The scaled loss.
        """
        scale_loss = loss * 1000
        try:
            scale_loss.backward()
        except torch.OutOfMemoryError:
            from auto_round.algorithms.quantization.search_dispatch import dump_oom_tensor_census_

            dump_oom_tensor_census_("tune backward")
            raise
        if is_hpex_available():
            htcore.mark_step()
        return scale_loss

    def _step(self, scaler: Any, optimizer: Any, lr_schedule: Any):
        """Performs a step in the optimization process.

        Args:
        scaler: The scaler to be used.
        optimizer: The optimizer for the step.
        lr_schedule: The learning rate schedule.

        Returns:
        None
        """
        optimizer.step()
        # for hpu
        if is_hpex_available():
            htcore.mark_step()
        optimizer.zero_grad()
        lr_schedule.step()
