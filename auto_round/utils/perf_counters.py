# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time as _ptime


class _PerfCounters:
    """Named-span time accumulator behind AR_PERF_COUNTERS.

    Keeps no measurement semantics of its own: callers either bracket code
    with ``start``/``stop`` per key, chain phase boundaries with ``tick``
    (each tick closes the previous span and opens the named one), or feed
    individual durations with ``observe`` (count + total + max per key).
    Times accumulate across brackets. When the env var is off every method
    is a no-op and ``get`` returns 0.0, so call sites need no guards.
    """

    def __init__(self):
        self.enabled = perf_counters_enabled()
        self.totals: "dict[str, float] | None" = {} if self.enabled else None
        self.counts: "dict[str, int]" = {}
        self.maxima: "dict[str, float]" = {}
        self._t0: "dict[str, float]" = {}
        self._current: "str | None" = None

    def start(self, key: str) -> None:
        if self.enabled:
            self._t0[key] = _ptime.perf_counter()

    def stop(self, key: str) -> None:
        if self.enabled:
            self.totals[key] = self.totals.get(key, 0.0) + (_ptime.perf_counter() - self._t0.pop(key))

    def tick(self, key: str) -> None:
        """Close the current chain span (if any) and open ``key``."""
        if not self.enabled:
            return
        if self._current is not None:
            self.stop(self._current)
        self.start(key)
        self._current = key

    def observe(self, key: str, duration: float) -> None:
        if self.enabled:
            self.counts[key] = self.counts.get(key, 0) + 1
            self.totals[key] = self.totals.get(key, 0.0) + duration
            self.maxima[key] = max(self.maxima.get(key, 0.0), duration)

    def get(self, key: str) -> float:
        return self.totals.get(key, 0.0) if self.enabled else 0.0

    def set(self, key: str, value: float) -> None:
        if self.enabled:
            self.totals[key] = value


def perf_counters_enabled() -> bool:
    """Whether AR_PERF_COUNTERS timing output is on.

    Single source of truth for the switch: the recorder and any gating that
    wants the same condition ask here, so the env name cannot drift apart.
    """
    import auto_round.envs as _envs

    return bool(getattr(_envs, "AR_PERF_COUNTERS", False))
