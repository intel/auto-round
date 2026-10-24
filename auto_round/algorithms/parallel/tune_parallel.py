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
"""Composer-owned context for the sharded calibration (collection) forwards.

``TuneParallelContext.for_collection`` resolves the block's devices; the
no-grad collection forwards then shard across the block's mirror pool
(:func:`data_parallel.sharded_nograd_forward`). ``devices is None`` keeps the
serial single-GPU collection. The tune itself runs on thread ranks
(:mod:`spmd`).
"""

import time as _ptime

import torch

from auto_round.algorithms.parallel.data_parallel import (
    ENGINE_CALIBRATION_STATS,
    _quantizer_policy,
    distribute_pool,
    resolve_tune_ddp_plan_,
    sharded_nograd_forward,
)
from auto_round.logger import logger


class _null_scope:
    """``with``-scope that is a no-op (CPU devices cannot enter torch.cuda.device)."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _device_scope(dev: torch.device):
    """``torch.cuda.device(dev)`` for cuda devices, no-op otherwise."""
    if dev.type == "cuda":
        return torch.cuda.device(dev)
    return _null_scope()


class TuneParallelContext:
    """Collection-phase parallel context; ``devices is None`` means serial."""

    def __init__(self) -> None:
        self.devices: list[torch.device] | None = None
        # collect-forward concurrency cap (from the run's ParallelPolicy;
        # 0 = uncapped) -- see collect_forward(hook_pass=True)
        self.collect_forward_cap: int = 0
        # per-block mirror pool (None until the first sharded collection
        # pass; reused by the collection forwards, the iters=0 cascade and
        # the tune ranks)
        self.pool = None

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
        merge_stats=ENGINE_CALIBRATION_STATS,
    ):
        """Collection forward: sharded across DDP mirrors when eligible.

        ``out_dev`` optionally overrides the runner's output cache device.
        ``allow_shard=False`` forces the serial path.

        ``merge_stats`` maps the statistics the pass's hooks write to their
        shard reduction (``calibration_merge_stats``): hook writes land on the
        mirror copies and are folded back into the home after the pass.
        ``None`` means the block quantizer's hooks write statistics it did
        not declare: the pass stays serial, since sharding would silently
        keep only the home shard's samples.

        ``hook_pass=True`` optionally caps the concurrent shards via the
        policy's ``collect_forward_cap`` (from AR_TUNE_DDP_MAX_COLLECT_FORWARD_DEVICES,
        default 0 = no cap): forward
        hooks force dynamo graph breaks, leaving the compiled runner as
        python-bound eager sections that can GIL-convoy under many threads.
        """
        if self.devices is None or not allow_shard:
            return block_forward(block, inputs, input_others, cache_device=out_dev)
        if merge_stats is None:
            logger.warning_once(
                "[tune-ddp] the block quantizer registers calibration hooks without declaring calibration_stats; "
                "those calibration passes run serially (declare each statistic's shard reduction to shard them)"
            )
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
            merge_stats=merge_stats,
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
        """Drop the per-block mirror pool (clears the ctx-side references only)."""
        if self.pool is not None:
            self.pool.release()
            self.pool = None
