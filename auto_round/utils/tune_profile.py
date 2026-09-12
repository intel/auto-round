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
"""Env-gated per-stage profiler for the block tuning loop (``AR_TUNE_PROFILE``).

Measures host wall time per loop stage and, on CUDA devices, records event
pairs whose elapsed times are read only at summary time -- one synchronize
for the whole block, so the instrumentation itself adds no per-iteration
sync points and does not perturb the sync-bubble it is meant to observe.

Callers hold a profiler instance (or None) and wrap regions::

    with stage(prof, "fwd"):
        ...
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from typing import DefaultDict, List, Optional, Tuple, Union

import torch

from auto_round.logger import logger

_STAGE_ORDER = ("ref_h2d", "fwd", "loss", "sync", "bwd", "snapshot", "step")


class TuneProfiler:
    """Accumulates per-stage host timings (and deferred CUDA-event timings)."""

    def __init__(self, device: Optional[Union[str, torch.device]] = None, debug: bool = False) -> None:
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.debug = debug
        self._use_events = self.device.type == "cuda" and torch.cuda.is_available()
        self.host: DefaultDict[str, float] = defaultdict(float)  # seconds
        self.counts: DefaultDict[str, int] = defaultdict(int)
        self._events: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    @contextmanager
    def stage(self, name: str):
        """Time a region: host wall always; CUDA event pair when available."""
        t0 = time.perf_counter()
        ev0 = None
        if self._use_events:
            ev0 = torch.cuda.Event(enable_timing=True)
            ev0.record()
        try:
            yield
        finally:
            if ev0 is not None:
                ev1 = torch.cuda.Event(enable_timing=True)
                ev1.record()
                self._events.append((name, ev0, ev1))
            self.host[name] += time.perf_counter() - t0
            self.counts[name] += 1

    def gpu_totals(self) -> DefaultDict[str, float]:
        """Resolve deferred CUDA events; syncs once. Seconds per stage.

        Syncs EVERY CUDA device, not just the profiler's home: stage events
        can be recorded on (or wait behind work enqueued to) mirror streams
        -- e.g. the source-side casts of a reduced-precision gradient
        transport -- and a home-only synchronize leaves those events
        un-resolved, which elapsed_time treats as a hard error. A pair that
        is somehow still pending is skipped rather than crashing the run.
        """
        totals: DefaultDict[str, float] = defaultdict(float)
        if self._use_events and self._events:
            for _idx in range(torch.cuda.device_count()):
                torch.cuda.synchronize(_idx)
            for name, ev0, ev1 in self._events:
                if not ev0.query() or not ev1.query():  # pragma: no cover - defensive
                    continue
                try:
                    totals[name] += ev0.elapsed_time(ev1) / 1000.0  # ms -> s
                except RuntimeError:  # pragma: no cover - cross-device/broken pair
                    continue
        return totals

    def log_placement(
        self,
        device,
        loss_device,
        cache_device,
        inputs,
        fp_outputs,
        nsamples,
    ) -> None:
        """One-line dump of tensor residency for the block (AR_TUNE_PROFILE=2).

        ``inputs`` / ``fp_outputs`` are lists of (device, shape) summaries.
        """
        parts = []
        for label, entries in (("inputs", inputs), ("fp_outputs", fp_outputs)):
            if entries:
                uniq = sorted({f"{d}{tuple(s)}" for d, s in entries}, key=str)
                parts.append(f"{label}={uniq}")
        logger.info(
            f"[tune-profile-placement] device={device} loss_device={loss_device} "
            f"cache_device={cache_device} nsamples={nsamples} " + " ".join(parts)
        )

    def log_summary(self, block_name: str, iters_done: int, wall: float, prefix: str = "tune-profile") -> None:
        """Emit a single greppable INFO line with the per-block stage table."""
        per_iter = 1000.0 / iters_done if iters_done > 0 else 0.0  # s -> ms/iter
        host_ms = {name: self.host[name] * per_iter for name in sorted(self.host)}
        # stable ordering: known tune-loop stages first, then any extra stages
        # recorded by other callers (e.g. compress_block phases) alphabetically
        ordered = [n for n in _STAGE_ORDER if n in host_ms] + sorted(set(host_ms) - set(_STAGE_ORDER))
        known = sum(host_ms.get(name, 0.0) for name in ordered)
        other_ms = max(0.0, wall * 1000.0 / max(iters_done, 1) - known)
        host_parts = " ".join(f"{name}={host_ms[name]:.1f}" for name in ordered)
        gpu = self.gpu_totals()
        gpu_parts = ""
        bubble = ""
        if gpu:
            gpu_ms = {name: gpu[name] * per_iter for name in sorted(gpu)}
            gpu_parts = " | gpu ms/iter: " + " ".join(f"{name}={gpu_ms[name]:.1f}" for name in gpu_ms)
            bubble = f" | bubble={wall * 1000.0 / max(iters_done, 1) - sum(gpu_ms.values()):.1f}"
        logger.info(
            f"[{prefix}] block={block_name} iters={iters_done} wall={wall:.1f}s "
            f"({wall * 1000.0 / max(iters_done, 1):.1f} ms/iter) | host ms/iter: {host_parts} "
            f"other={other_ms:.1f}{gpu_parts}{bubble}"
        )


def stage(prof: Optional[TuneProfiler], name: str):
    """Timed context when ``prof`` is active, nullcontext otherwise."""
    return prof.stage(name) if prof is not None else nullcontext()
