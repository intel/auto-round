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
"""Composer-owned parallel tuning context.

Wraps the single-process data-parallel engine (:mod:`data_parallel`) behind the
interface the algorithms see: ``None`` means fully serial (zero overhead), a
live context fans work out to replica mirrors and exchanges gradients.

Two phase-scoped flavors share this class:

* **collection** (``TuneParallelContext.for_collection``): devices only; the
  no-grad collection forwards are sharded across ephemeral mirrors
  (:func:`data_parallel.sharded_nograd_forward`). Always returns a context --
  ``devices is None`` keeps the serial single-GPU collection.
* **tune** (``TuneParallelContext.create``): owns the plan, the persistent
  :class:`data_parallel.ReplicaGroup`, the distributed calibration pool, the
  mirror optimizers and the per-iteration gradient consensus. Returns ``None``
  when the parallel lane declines (caller falls back to serial).

The algorithm (SignRound-family tune loop) stays parallel-unaware: it builds
one ``step_fn(rep, shard, dev, record)`` closure -- forward, loss, backward --
and the context runs it serially for the warm-up and threaded for the loop.
"""

import time as _ptime
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch

from auto_round.algorithms.parallel.data_parallel import (
    ReplicaGroup,
    _quantizer_policy,
    block_has_tuning_entries,
    distribute_pool,
    expect_pool_local,
    parallel_state,
    pre_wrap_shard_candidate,
    resolve_tune_ddp_plan_,
    run_deferred_wrap_searches,
    sharded_nograd_forward,
)
from auto_round.compressors.utils import shard_samplers
from auto_round.logger import logger


class _null_scope:
    """``with``-scope that is a no-op (CPU devices cannot enter torch.cuda.device)."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _device_scope(dev: torch.device):
    """``torch.cuda.device(dev)`` for cuda devices, no-op otherwise
    (same guard style as the sharded RTN searches)."""
    if dev.type == "cuda":
        return torch.cuda.device(dev)
    return _null_scope()


class _StepRecord:
    """Per-replica wall splits recorded inside an algorithm ``step_fn``."""

    __slots__ = ("fwd", "bwd")

    def __init__(self) -> None:
        self.fwd: float = 0.0
        self.bwd: float = 0.0


class TuneParallelContext:
    """Engine-owned parallel context; ``None`` (tune) / ``devices is None``
    (collection) means the serial path."""

    # ── construction ─────────────────────────────────────────────────────────

    def __init__(self) -> None:
        # collection flavor
        self.devices: Optional[List[torch.device]] = None
        # tune flavor
        self.group: Optional[ReplicaGroup] = None
        self.plan: Any = None
        self.quantizer: Any = None
        self.block: Any = None
        self.active_inputs: Any = None
        self.input_others: Any = None
        self.fp_outputs: Any = None
        self.mirror_optimizers: List[Any] = []
        self.mirror_schedules: List[Any] = []
        self.params_per_replica: List[List[torch.nn.Parameter]] = []
        self._samplers: Optional[List[Any]] = None
        self._pending_sync = False
        self.perf: dict = {}
        # collect-forward concurrency cap (from the run's ParallelPolicy;
        # 0 = uncapped) -- see collect_forward(hook_pass=True)
        self.collect_forward_cap: int = 0
        # per-block mirror pool (None until the first sharded collection pass;
        # the iters=0 lane reuses it for the searches and the cascade -- see
        # the tune ADOPTS the pool's mirrors when aligned, or the composer
        # releases it before the post-quantize steps)
        self.pool = None

    # ── P3: collection phase (composer-owned) ────────────────────────────────

    @classmethod
    def for_collection(cls, composer, block, fp_inputs) -> "TuneParallelContext":
        """Mirror devices for sharding the no-grad collection forwards.

        Uses the SAME engagement resolver as the tune
        (:func:`data_parallel.resolve_tune_ddp_plan_`, cached on the quantizer)
        so the collection gates stay identical to what the tune later engages
        -- plus two collection-specific gates: the runner must use the
        plain tensor output layout, and the block class must not be a
        multi-output registry entry (sharding clears ``last_output_dict``,
        which those blocks need to feed the next block). ``devices is None``
        keeps the serial single-GPU collection.
        """
        ctx = cls()
        if not isinstance(fp_inputs, list) or not fp_inputs:
            return ctx
        runner_cfg = getattr(composer.block_forward, "output_config", None)
        if runner_cfg and list(runner_cfg) != ["hidden_states"]:
            return ctx  # multi-output layouts need the serial last_output_dict path
        from auto_round.algorithms.block_runner import _BLOCK_OUTPUT_REGISTRY

        if len(_BLOCK_OUTPUT_REGISTRY.get(type(block).__name__, ["hidden_states"])) != 1:
            return ctx  # registry multi-output blocks (e.g. GlmMoeDsa) need serial collection
        try:
            home = next(block.parameters(), torch.empty(0)).device
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("[tune-ddp] collection sharding declined: cannot resolve the block home device (%s)", e)
            return ctx
        try:
            plan = resolve_tune_ddp_plan_(composer.block_quantizer, block, fp_inputs, None, home, log=False)
        except RuntimeError:
            # the resolver raises when a requested world is ineligible
            # (requirement semantics) -- let it propagate: silently continuing
            # with serial collection would invalidate the request
            raise
        except Exception as e:  # pragma: no cover - resolver reachability on odd hosts
            logger.warning("[tune-ddp] collection sharding declined: resolver unreachable (%s)", e)
            return ctx
        ctx.devices = plan.devices if plan.world > 1 else None
        ctx.collect_forward_cap = _quantizer_policy(composer.block_quantizer).collect_forward_cap
        return ctx

    def distribute_pools(self, fp_inputs, q_inputs) -> None:
        """Distribute the calibration pool so each DDP device owns its sample
        shard (matching the tune shards); shard-local reads stay device-local;
        serial consumers use device-safe cats."""
        if not self.devices:
            return
        if isinstance(fp_inputs, list) and fp_inputs:
            distribute_pool(fp_inputs, self.devices)
        if isinstance(q_inputs, list) and q_inputs:
            distribute_pool(q_inputs, self.devices)

    def collect_forward(
        self,
        block_forward,
        block,
        inputs,
        input_others,
        out_dev=None,
        allow_shard: bool = True,
        hook_pass: bool = False,
    ):
        """Collection forward: sharded across DDP mirrors when eligible.

        ``out_dev`` optionally overrides the runner's output cache device.
        ``allow_shard=False`` forces the serial path: passes that carry
        forward hooks (fp-input / q-input stats that cannot be merged)
        must not shard -- hook writes land on the ephemeral mirror copies
        and are freed with them, silently dropping that shard's statistics.
        Mergeable stats (imatrix, act_max) are folded from the mirrors back
        into the home, so those hook passes may shard.

        ``hook_pass=True`` optionally caps the concurrent shards via the
        policy's ``collect_forward_cap`` (from AR_TUNE_DDP_MAX_COLLECT_FORWARD_DEVICES,
        default 0 = no cap): forward
        hooks force dynamo graph breaks, leaving the compiled runner as
        python-bound eager sections that can GIL-convoy under many threads.
        """
        if self.devices is None or not allow_shard:
            return block_forward(block, inputs, input_others, cache_device=out_dev)
        # The pool is ALWAYS built at the full device width and reused by
        # every sharding-eligible pass; a hook pass capped below the world
        # only limits WHICH pool replicas run -- sharded_nograd_forward slices
        # the replica subset in place.
        world = len(self.devices)
        cap = int(self.collect_forward_cap or 0)
        n = len(inputs) if isinstance(inputs, (list, tuple)) else 0
        pool = None
        if world >= 2 and isinstance(block, torch.nn.Module):
            # ceil/floor split: any n >= world shards (a remainder lands on
            # an earlier replica); n < world runs below the pool's width and
            # stays serial
            if n >= world:
                if self.pool is None:
                    from auto_round.algorithms.parallel.data_parallel import MirrorPool

                    _t0 = _ptime.perf_counter()
                    self.pool = MirrorPool(block, self.devices)
                    logger.debug(
                        "[tune-ddp] pool build (first sharded pass): world=%d build=%.2fs "
                        "(per-device deepcopy + relocate)",
                        world,
                        _ptime.perf_counter() - _t0,
                    )
                pool = self.pool
        return sharded_nograd_forward(
            block_forward,
            block,
            inputs,
            input_others,
            out_dev,
            self.devices,
            merge_stats=True,
            max_devices=cap if hook_pass else 0,
            pool=pool,
        )

    def ensure_pool(self, block) -> None:
        """Build the per-block mirror pool if absent (block-wide, not tied to
        any single pass's forward eligibility): the searches and the tune
        consume it even when a collection pass fell back serial."""
        if self.pool is None and self.devices and len(self.devices) >= 2:
            from auto_round.algorithms.parallel.data_parallel import MirrorPool

            _t0 = _ptime.perf_counter()
            self.pool = MirrorPool(block, self.devices)
            logger.debug(
                "[tune-ddp] pool build: world=%d build=%.2fs (per-device deepcopy + relocate)",
                len(self.devices),
                _ptime.perf_counter() - _t0,
            )

    def release_pool(self) -> None:
        """Drop the per-block mirror pool (adoption transfers ownership to the
        ReplicaGroup; this clears the ctx-side references only)."""
        if self.pool is not None:
            self.pool.release()
            self.pool = None

    # ── P1: tune phase (quantizer-owned) ─────────────────────────────────────

    @staticmethod
    def defer_wrap_searches(quantizer) -> bool:
        """Engine policy: should the wrap-time searches defer until the
        mirrors exist (so they can run round-robin on the replicas)?"""
        return pre_wrap_shard_candidate(quantizer)

    @classmethod
    def create(
        cls,
        quantizer,
        block,
        active_inputs,
        fp_outputs,
        device,
        input_others=None,
        nsamples=None,
    ) -> Optional["TuneParallelContext"]:
        """Engage the parallel tune lane for this block, or return ``None``.

        Resolves the plan fresh (this call runs POST-wrap, so the mirror
        pricing sees the wrapper's fp32 value params and the free-VRAM
        snapshot is current for THIS block), distributes the calibration
        pool, builds the persistent
        replica group and runs the deferred wrap-time searches on the mirrors
        (before any forward -- the warm-up and the tune loop both need
        init_scale present).
        """
        # drop the composer's pre-wrap plan: the post-wrap resolution below is
        # authoritative for the tune
        _st = parallel_state(quantizer)
        if _st is not None:
            _st.plan = None
        plan = resolve_tune_ddp_plan_(quantizer, block, active_inputs, fp_outputs, device)
        if plan.enabled and not isinstance(fp_outputs, list):
            # the composer may have resolved before reference outputs existed;
            # non-list outputs (diffusion-style) cannot be pool-distributed
            logger.info("[tune-ddp] declining: reference outputs are not a list")
            plan = type(plan)(1, plan.devices[:1], plan.shard_size)
            parallel_state(quantizer, create=True).plan = plan
        if not plan.enabled:
            return None
        # All-float pinned blocks (a 'bits':16 'data_type':'float' layer_config
        # pin) carry no tuning parameters: the serial path's empty-params
        # guard no-ops them, so the parallel lane would only pay mirror
        # setup + pool distribution for nothing (the serial early-return also
        # leaves the group intact). Decline before mirror setup.
        if not block_has_tuning_entries(block):
            logger.info("[tune-ddp] declining: no tuning parameters in this block (all-float pinned); serial path")
            return None

        # Placement contract: the resolver declined (or raised) when the
        # block's weights span several CUDA devices, so the source block
        # already sits whole on the home device and every mirror will sit
        # whole on exactly one device.
        # distributed calibration pool: shard-local tune reads; each
        # device owns a contiguous 1/world slice of the samples
        distribute_pool(active_inputs, plan.devices)
        distribute_pool(fp_outputs, plan.devices)

        _t0 = _ptime.perf_counter()
        group = cls._adopt_or_build_group_(quantizer, block, plan)
        # phase boundary: mirror construction (adopt or deepcopy) moves
        # weights/stats cross-device; drain before the deferred searches
        from auto_round.algorithms.parallel.rtn_sharding import _sync_pool_devices as _spb

        _spb(plan.devices)
        ctx = cls()
        ctx.quantizer = quantizer
        ctx.block = block
        ctx.plan = plan
        ctx.group = group
        ctx.collect_forward_cap = _quantizer_policy(quantizer).collect_forward_cap
        ctx.active_inputs = active_inputs
        ctx.input_others = input_others
        ctx.fp_outputs = fp_outputs
        ctx.nsamples = nsamples
        ctx.perf = {
            "build": _ptime.perf_counter() - _t0,
            "warm": 0.0,
            "fwd": [],
            "bwd": [],
            "exch": [],
            "step": [],
        }
        for note in plan.notes:
            logger.info("[tune-ddp] %s", note)

        # mirrors-first: the deferred wrap-time searches run round-robin on
        # the replicas now that the mirrors exist (before any forward --
        # the warm-up and the tune loop both need init_scale present)
        run_deferred_wrap_searches(block, group)
        return ctx

    @staticmethod
    def _adoption_stat_targets_(block):
        """(name, stat-holder) pairs for the pre-adoption stats distribution.

        Names must be WRAPPER-level (the pool's pre-wrap layer map has no
        ``.orig_layer`` children): wrappers that register ``orig_layer`` as a
        child module also yield it from ``named_modules`` carrying the very
        same stats object -- skip those duplicates or the mirror lookup
        KeyErrors on real models (wrappers register orig_layer as a module,
        e.g. Qwen3.8's linear_attn.out_proj).
        """
        targets = []
        for n, m in block.named_modules():
            if n.endswith(".orig_layer"):
                continue  # duplicate of the parent wrapper entry (same object)
            if hasattr(m, "orig_layer") or hasattr(m, "imatrix"):
                targets.append((n, getattr(m, "orig_layer", m)))
        return targets

    @staticmethod
    def _adopt_or_build_group_(quantizer, block, plan):
        """Adopt the collection MirrorPool's resident mirrors when aligned
        (wrap-in-place, zero second weight copy); otherwise deepcopy.

        Adoption path: the pool mirrors carry the block's weights already;
        the engagement wraps each mirror with the SAME wrapper call the home
        block received (deterministic init from identical weights + the
        distributed home stats), so the replicas match the deepcopy
        construction exactly. The pool then transfers ownership; its
        ``release()`` only drops references.
        """
        _st_adopt = parallel_state(quantizer)
        pool = _st_adopt.pool if _st_adopt is not None else None
        if pool is None or pool.block is not block or list(pool.devices) != list(plan.devices):
            return ReplicaGroup(block, plan)

        # home stats (imatrix etc.) read through the wrappers must reach the
        # mirror layers before their wrap consumes them
        from auto_round.algorithms.parallel.data_parallel import ReplicaGroup as _RG
        from auto_round.algorithms.parallel.rtn_sharding import _distribute_search_stats

        targets = TuneParallelContext._adoption_stat_targets_(block)
        if targets:
            _distribute_search_stats(pool, block, targets)
            # phase boundary: the wraps below read the distributed stats on
            # their own devices; drain the copies first
            from auto_round.algorithms.parallel.rtn_sharding import _sync_pool_devices

            _sync_pool_devices(pool.devices)

        # stale collection hooks on mirror layers: clear everything, then
        # re-register exactly the hooks the home ORIG layers carry now
        # (rotation included; removed stats hooks stay removed)
        home_hooks = {}
        for n, m in block.named_modules():
            ol = getattr(m, "orig_layer", None)
            if ol is not None:
                fwd = dict(getattr(ol, "_forward_hooks", None) or {})
                pre = dict(getattr(ol, "_forward_pre_hooks", None) or {})
                if fwd or pre:
                    fwd_wk = dict(getattr(ol, "_forward_hooks_with_kwargs", None) or {})
                    pre_wk = dict(getattr(ol, "_forward_pre_hooks_with_kwargs", None) or {})
                    home_hooks[n] = (fwd, pre, fwd_wk, pre_wk)
        for r in range(pool.world):
            rep = pool.reps[r]
            if rep is block:
                continue
            for name, _m in targets:
                lm = pool.mirror_layer(r, name)
                lm._forward_hooks.clear()
                lm._forward_pre_hooks.clear()
                if hasattr(lm, "_forward_hooks_with_kwargs"):
                    lm._forward_hooks_with_kwargs.clear()
                if hasattr(lm, "_forward_pre_hooks_with_kwargs"):
                    lm._forward_pre_hooks_with_kwargs.clear()
            # non-target modules carry no collection/rotation hooks by
            # construction (both register on quantization targets only)
            for n, (fwd, pre, fwd_wk, pre_wk) in home_hooks.items():
                try:
                    lm = pool.mirror_layer(r, n)
                except KeyError:
                    continue
                for k, h in fwd.items():
                    lm.register_forward_hook(h, with_kwargs=bool(fwd_wk.get(k, False)))
                for k, h in pre.items():
                    lm.register_forward_pre_hook(h, with_kwargs=bool(pre_wk.get(k, False)))

        # wrap each mirror in place with the same wrapper call home received
        # (pool.devices == plan.devices, so reps[r] lives on plan.devices[r])
        _defer = TuneParallelContext.defer_wrap_searches(quantizer)
        for r in range(1, pool.world):
            quantizer.wrapper_block(
                pool.reps[r],
                quantizer.enable_minmax_tuning,
                quantizer.enable_norm_bias_tuning,
                enable_torch_compile=getattr(quantizer.compress_context, "enable_torch_compile", False),
                device=pool.devices[r],
                enable_neuqi=getattr(quantizer.config, "enable_neuqi", False),
                defer_search=_defer,
            )
        parallel_state(quantizer, create=True).pool_used = True
        # OWNERSHIP TRANSFER: the group now owns the mirrors. Clearing the
        # parallel-state pool ref tells the composer to drop the ctx pool too -- the
        # adopted mirrors carry the tune's LAST optimizer state, while home
        # is unwrapped with the BEST params, so the Step-6 cascade runs on
        # the canonical home block.
        _st_clear = parallel_state(quantizer)
        if _st_clear is not None:
            _st_clear.pool = None
        return _RG.adopt(block, plan, pool.reps[1:])

    @property
    def world(self) -> int:
        return self.group.world if self.group is not None else 1

    @property
    def shard_size(self) -> int:
        return getattr(self.plan, "shard_size", 0) if self.plan is not None else 0

    def build_mirror_optimizers(self, make_optimizer, make_schedule, collect_params_fn) -> None:
        """Replicate the home optimizer/schedule structure on every mirror.

        ``make_optimizer(param_groups)`` and ``make_schedule(optimizer)`` are
        algorithm-owned closures (identical construction to the home
        optimizer); ``collect_params_fn(module)`` returns the algorithm's
        ``(round, minmax, round_lr_groups, minmax_lr_groups)`` tuple.
        """
        enable_minmax_tuning = getattr(self.quantizer, "enable_minmax_tuning", False)
        for mirror in self.group.mirrors:
            r_ps, m_ps, r_gr, m_gr = collect_params_fn(mirror)
            m_params = [{"params": ps, "lr": torch.tensor(g_lr)} for g_lr, ps in r_gr.items()]
            if enable_minmax_tuning:
                m_params += [{"params": ps, "lr": torch.tensor(g_lr)} for g_lr, ps in m_gr.items()]
            m_opt = make_optimizer(m_params)
            m_sched = make_schedule(m_opt)
            self.mirror_optimizers.append(m_opt)
            self.mirror_schedules.append(m_sched)
        self.params_per_replica = [
            r_ps + m_ps if enable_minmax_tuning else r_ps
            for r_ps, m_ps, _r, _m in (collect_params_fn(rep) for rep in self.group.replicas)
        ]

    def warmup(self, step_fn: Callable, home_optimizer) -> None:
        """Warm every replica SERIALLY in the main thread: torch.compile
        materializes its per-device kernels lazily at first call, and
        compiling from several worker threads at once races in dynamo.
        The warm-up also validates each mirror end-to-end; a failure
        ABORTS the run (DDP was explicitly requested -- continuing
        serially would silently invalidate the configuration and any
        measurement against it). Grads are discarded.

        ``step_fn(rep, shard, dev, record)`` is the algorithm's tune step
        (forward + loss + backward); it is the SAME closure the loop uses.
        """
        quantizer = self.quantizer
        try:
            _t0 = _ptime.perf_counter()
            _pool_shard = max(1, self.nsamples // self.group.world)
            for r, rep in enumerate(self.group.replicas):
                # warm only one batch's worth (shard_size samples) at the
                # head of this replica's pool shard -- warming the whole
                # shard would materialize its full activations on the mirror
                _warm = list(range(r * _pool_shard, r * _pool_shard + self.plan.shard_size))
                _dev_r = next(rep.parameters()).device
                with _device_scope(_dev_r):
                    step_fn(rep, _warm, _dev_r, _StepRecord())
            self.perf["warm"] = _ptime.perf_counter() - _t0
            for _opt in [home_optimizer] + self.mirror_optimizers:
                _opt.zero_grad()
        except Exception as _warm_err:  # noqa: BLE001 - re-raised with context
            self._log_warmup_failure(_warm_err)
            self.group.teardown()
            raise RuntimeError(
                "parallel tuning was requested but the replica warm-up failed -- "
                "refusing to continue on the serial path (that would silently "
                "invalidate the requested configuration and any measurement "
                "against it). Fix the underlying failure or run with "
                "--parallel_quantization off."
            ) from _warm_err

    def _log_warmup_failure(self, err: Exception) -> None:
        """Fail-visible warm-up diagnostics: tensor-device census + hook
        inventory per replica, kwargs fingerprint once. Never masks ``err``."""
        try:
            for _ri, _rep in enumerate(self.group.replicas):
                _census: dict = {}
                for _on, _pp in _rep.named_parameters():
                    _census[str(_pp.device)] = _census.get(str(_pp.device), 0) + 1
                # hooks survive deepcopy/replicate and are invisible to
                # parameter walks; a hook with a baked-in device (e.g. an
                # AlignDevicesHook pointing at the primary) moves inputs or
                # reloads weights onto the WRONG GPU mid-forward
                _hooks: dict = {}
                for _mn, _mod in _rep.named_modules():
                    for _fn in list(getattr(_mod, "_forward_pre_hooks", {}).values()) + list(
                        getattr(_mod, "_forward_hooks", {}).values()
                    ):
                        _t = type(_fn).__name__
                        _hooks[_t] = _hooks.get(_t, 0) + 1
                logger.error(
                    "[tune-ddp] warm-up failed; replica[%d] param device census: %s; hooks: %s",
                    _ri,
                    _census,
                    _hooks or "none",
                )
                if _ri == 0:  # fingerprint once: kwargs types/shapes + pool geometry
                    try:

                        def _fp(v):
                            if torch.is_tensor(v):
                                return tuple(v.shape)
                            if isinstance(v, tuple):
                                return "(" + ", ".join(_fp(t) for t in v) + ")"
                            if isinstance(v, (list,)):
                                return f"list[{len(v)}]"
                            return type(v).__name__

                        _pool = len(self.active_inputs) if isinstance(self.active_inputs, list) else "dict"
                        logger.error(
                            "[tune-ddp] kwargs fingerprint (pool n=%s, batch=%s): %s",
                            _pool,
                            getattr(getattr(self.quantizer, "calibration_context", None), "batch_size", "?"),
                            {k: _fp(v) for k, v in self.input_others.items()},
                        )
                    except Exception:  # pragma: no cover - diagnostics must not mask
                        pass
        except Exception:  # pragma: no cover - diagnostics must not mask
            pass

    def shards(self, nsamples: int, global_batch_size: int) -> None:
        """Build per-replica shard samplers when the global batch splits
        evenly across the world (else the loop falls back to index slicing)."""
        if self.group is not None and global_batch_size % self.group.world == 0:
            self._samplers = shard_samplers(nsamples, self.group.world, global_batch_size // self.group.world)

    def next_shards(self, index_sampler) -> Tuple[List[List[int]], List[int]]:
        """Draw the next global batch and split it into per-replica shards.

        Returns ``(shards, global_indices)``; the sampler path rebuilds
        ``global_indices`` from the shards so the two stay consistent."""
        if self._samplers is not None:
            shards = [s_.next_batch() for s_ in self._samplers]
            global_indices = [j for sh in shards for j in sh]
            return shards, global_indices
        global_indices = index_sampler.next_batch()
        n = len(global_indices)
        world = self.group.world
        # ceil/floor split: sum-reduced losses make uneven shards exact, so
        # any global batch size shards -- a remainder sample lands on an
        # earlier replica and every sample is kept
        sizes = [n // world + (1 if r < n % world else 0) for r in range(world)]
        bounds = [0]
        for _sz in sizes:
            bounds.append(bounds[-1] + _sz)
        shards = [global_indices[bounds[r] : bounds[r + 1]] for r in range(world)]
        return shards, global_indices

    def run_step(self, step_fn: Callable, shards: Sequence[Sequence[int]]) -> List[Optional[torch.Tensor]]:
        """Fan the tune step out to every replica in parallel threads.

        ``step_fn(rep, shard, dev, record)`` (algorithm-owned) runs forward +
        loss + backward on the replica's shard; it returns the loss tensor
        (detached here) and fills ``record.fwd/.bwd`` for the perf split.
        """
        if self._pending_sync:
            # a previous run_step's gradients went unexchanged: the
            # replicas have already stepped apart (each optimizer consumed
            # its own shard's gradient) -- fail visibly before the replicas
            # diverge
            logger.error(
                "[tune-ddp] %s: run_step called twice without an intervening sync_grads -- "
                "replica gradients were never exchanged; tuning is diverging",
                type(self.block).__name__ if self.block is not None else "block",
            )
        world = self.group.world
        losses: List[Optional[torch.Tensor]] = [None] * world
        records = [_StepRecord() for _ in range(world)]

        def _one(r: int) -> Callable[[], None]:
            rep = self.group.replicas[r]
            dev_r = next(rep.parameters()).device
            rec = records[r]

            def _inner() -> None:
                with _device_scope(dev_r):
                    losses[r] = step_fn(rep, shards[r], dev_r, rec).detach()

            return _inner

        self.group.run_threaded([_one(r) for r in range(world)])
        self.perf["fwd"].append(max(rec.fwd for rec in records))
        self.perf["bwd"].append(max(rec.bwd for rec in records))
        self._pending_sync = True
        return losses

    def sync_grads(self, sign_exchange: bool) -> None:
        """Cross-replica gradient exchange (sign-cast consensus when
        ``sign_exchange`` -- the algorithm states the property; the engine
        does not infer it)."""
        _t0 = _ptime.perf_counter()
        self.group.sync_grads(self.params_per_replica, sign_exchange=sign_exchange)
        self.perf["exch"].append(_ptime.perf_counter() - _t0)
        self._pending_sync = False

    def mean_loss(self, losses: Sequence[Optional[torch.Tensor]], num_elm, divide_world: bool = True) -> float:
        """Report the global-batch mean, normalized by the valid-element count
        exactly like the serial path.

        ``divide_world=True`` divides mean-reduced shard losses by the world
        (equal-size shard means average to the global mean). ``False`` fits
        sum-reduced shard losses (the DDP lane's micro-batch chunks): the
        shard sums already add up to the serial global sum, so only the
        element count divides."""
        _ne = 1 if num_elm <= 0 else num_elm
        _total = sum(l.item() for l in losses if l is not None)
        return _total / self.group.world / _ne if divide_world else _total / _ne

    def step(self, home_step_fn: Callable[[], None]) -> None:
        """Run the home step and every mirror step in parallel threads (home
        first so the persistent pool serves the step too instead of falling
        back to per-iteration thread spawn)."""

        def _mirror_step(opt, sched):
            opt.step()
            opt.zero_grad()
            sched.step()

        _t0 = _ptime.perf_counter()
        self.group.run_threaded(
            [home_step_fn]
            + [
                lambda oo=opt, ss=sch: _mirror_step(oo, ss)
                for opt, sch in zip(self.mirror_optimizers, self.mirror_schedules)
            ]
        )
        self.perf["step"].append(_ptime.perf_counter() - _t0)

    def teardown(self) -> None:
        _t0 = _ptime.perf_counter()
        self.group.teardown()
        self.perf["teardown"] = _ptime.perf_counter() - _t0

    def log_perf(self, block) -> None:
        """AR_PERF_COUNTERS summary (mean/max per iteration)."""
        from auto_round import envs as _envs

        if not getattr(_envs, "AR_PERF_COUNTERS", False):
            return
        _fwd, _exch, _step = self.perf["fwd"], self.perf["exch"], self.perf["step"]

        def _ms(x):
            return f"{1000 * x:.0f}ms"

        _bwd = self.perf["bwd"]
        logger.info(
            "[perf] tune-ddp block (%s): mirrors=%s warmup=%s "
            "fwd=%s/%s bwd=%s/%s exch=%s/%s step=%s/%s teardown=%s (mean/max per iter)",
            type(block).__name__,
            _ms(self.perf["build"]),
            _ms(self.perf["warm"]),
            _ms(sum(_fwd) / max(len(_fwd), 1)),
            _ms(max(_fwd)) if _fwd else "n/a",
            _ms(sum(_bwd) / max(len(_bwd), 1)),
            _ms(max(_bwd)) if _bwd else "n/a",
            _ms(sum(_exch) / max(len(_exch), 1)),
            _ms(max(_exch)) if _exch else "n/a",
            _ms(sum(_step) / max(len(_step), 1)),
            _ms(max(_step)) if _step else "n/a",
            _ms(self.perf.get("teardown", 0.0)),
        )

    # small helpers re-exported for the algorithm's step_fn --------------

    @staticmethod
    def expect_pool_local(pieces, device, site: str) -> None:
        expect_pool_local(pieces, device, site)
