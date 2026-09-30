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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
"""Tune-loop memory model, MoE-implementation policy, and pool pulls.

Shared by the quantizer base class (the ``maybe_adapt_moe_implementation``
and ``pull_tuning_pool`` hooks) so individual algorithm implementations stay
free of fleet-memory policy: everything here reads only the block's wrapper
layout, the batch size, the run's config, and the devices' free memory.
"""

from typing import Any

import torch

from auto_round.logger import logger


def _tuning_state_bytes(block: torch.nn.Module, target_dev: str) -> int:
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


def _logical_state_by_device(block: torch.nn.Module) -> dict[str, int]:
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


def _ensure_routed_shape_recorders_(block: torch.nn.Module) -> None:
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

    def _record(module: torch.nn.Module, args: Any) -> None:
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

    def _is_experts_container(name: str, module: torch.nn.Module) -> bool:
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


def _grouped_stack_bytes(block: torch.nn.Module, device: str, config: Any = None) -> int:
    """Qdq-stack bytes the grouped experts modes materialize on ``device``."""
    d = _grouped_stack_bytes_detail(block, device, config=config)
    return d["retention"] + d["transient"]


def _grouped_stack_bytes_detail(block: torch.nn.Module, device: str, config: Any = None) -> dict[str, int]:
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


def _routed_budget_bytes(block: torch.nn.Module, tensors: Any, batch_size: int, config: Any = None) -> int | None:
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


def _activation_bytes_by_device(
    block: torch.nn.Module, tensors: Any, batch_size: int, config: Any = None
) -> dict[str, int] | None:
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


def _block_activation_bytes(
    block: torch.nn.Module, tensors: Any, batch_size: int, config: Any = None, device: str | None = None
) -> int | None:
    """Back-compat scalar view: this DEVICE's slice of the per-device charge."""
    by_dev = _activation_bytes_by_device(block, tensors, batch_size, config)
    if by_dev is None:
        return None
    if device is None:
        return max(by_dev.values())
    return by_dev.get(str(device))


_MOE_IMPL_AUTO_LINEAR_LOOP_DONE = False
_MOE_IMPL_AUTO_DONE_REF = None  # weakref to the deciding run's config


def _maybe_auto_linear_loop_for_tuning(
    block: torch.nn.Module, tensors: Any, batch_size: int, iters: int, config: Any, model: torch.nn.Module
) -> None:
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


def _pull_pool_if_fits(
    pool: Any,
    target_dev: str,
    block: torch.nn.Module,
    batch_size: int,
    iters: int,
    label: str,
    charge_activation: bool = False,
    config: Any = None,
) -> Any:
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
