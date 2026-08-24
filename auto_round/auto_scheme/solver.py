# Copyright (c) 2025 Intel Corporation
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

"""Bit-allocation solvers for AutoScheme.

Two solvers are available, selected through ``AutoScheme.solver``:

``"dp"``
    The historical knapsack dynamic program (:func:`choose_bits_per_layer_with_path`).
    Exact on a discretised bit grid, but the state space grows with the budget, which
    dominates the runtime on large models.

``"lagrangian"``
    Solves the same knapsack through its Lagrangian dual. For a price ``lam`` (units:
    loss per bit) every layer independently picks ``argmin_s loss_s + lam * bits_s``.
    Total bits decrease monotonically in ``lam``, so a bisection on ``lam`` drives the
    solution onto the budget. This hits a *fractional* avg_bits target exactly and needs
    no discretised state space, so it is typically an order of magnitude faster than the
    DP while producing the same allocation.

The dual only ever lands on the *convex hull* of each layer's (bits, loss) curve, so two
primal repair passes close the integrality gap: :func:`_greedy_repair` spends leftover
budget, and :func:`_swap_local_search` fixes cases where one downgrade funds one upgrade.

This module is deliberately free of model/torch dependencies so it can be unit-tested in
isolation.
"""

from __future__ import annotations

from typing import Optional

from auto_round.logger import logger

__all__ = [
    "solve_lagrangian",
    "solve_allocation",
]

_EPS = 1e-12


# ----------------------------------------------------------------------------------- #
# Score-table helpers
#
# ``total_scores`` maps a DP key (the first layer name of a shared-layer group) to a list
# of candidate options, each option being ``[scheme_index, bits_cost, loss_cost, names]``.
# ----------------------------------------------------------------------------------- #
def _option_cost(opt, lam: float) -> float:
    """Lagrangian cost ``loss + lam * bits`` of a single option."""
    return opt[2] + lam * opt[1]


def _pick_option(opts, lam: float):
    """Pick the option minimising the Lagrangian cost; ties broken toward fewer bits."""
    return min(opts, key=lambda o: (_option_cost(o, lam), o[1]))


def _total_bits(assign: dict) -> int:
    """Total bit cost of an assignment (``key -> option``)."""
    return sum(opt[1] for opt in assign.values())


def _total_loss(assign: dict) -> float:
    """Total predicted loss of an assignment (``key -> option``)."""
    return sum(opt[2] for opt in assign.values())


# ----------------------------------------------------------------------------------- #
# Lagrangian (shadow-price) solver
# ----------------------------------------------------------------------------------- #
def solve_lagrangian(total_scores: dict, budget: int, max_iter: int = 80) -> Optional[dict]:
    """Solve the bit-allocation knapsack through its Lagrangian dual.

    Args:
        total_scores: Score table.
        budget: Upper bound on total bits.
        max_iter: Bisection iterations.

    Returns:
        The assignment mapping key -> option, or ``None`` when even the cheapest
        configuration exceeds ``budget`` (i.e. the target is infeasible).
    """
    if not total_scores:
        return {}

    cheapest = {key: min(opts, key=lambda o: o[1]) for key, opts in total_scores.items()}
    if _total_bits(cheapest) > budget:
        return None

    # lam = 0 -> pure loss minimisation. If it already fits, nothing to trade off.
    assign = {key: _pick_option(opts, 0.0) for key, opts in total_scores.items()}
    if _total_bits(assign) <= budget:
        return assign

    # Bracket the price: grow the upper bound until the budget is satisfied.
    lo, hi = 0.0, 1e-9
    for _ in range(200):
        probe = {key: _pick_option(opts, hi) for key, opts in total_scores.items()}
        if _total_bits(probe) <= budget:
            break
        lo, hi = hi, hi * 4.0
    else:  # pragma: no cover - defensive
        logger.warning("AutoScheme: Lagrangian upper price search did not converge.")

    best = None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        probe = {key: _pick_option(opts, mid) for key, opts in total_scores.items()}
        if _total_bits(probe) <= budget:
            best, hi = probe, mid
        else:
            lo = mid

    if best is None:  # pragma: no cover - defensive
        best = {key: _pick_option(opts, hi) for key, opts in total_scores.items()}
    best = _greedy_repair(total_scores, best, budget)
    best = _swap_local_search(total_scores, best, budget)
    return best


def _greedy_repair(total_scores: dict, assign: dict, budget: int) -> dict:
    """Spend the budget slack left by the dual's integrality gap.

    The dual solution is generally not budget-tight. Repeatedly apply the upgrade with the
    best loss-reduction-per-extra-bit that still fits, which is the standard primal repair
    for a Lagrangian-relaxed knapsack.
    """
    used = _total_bits(assign)
    slack = budget - used
    if slack <= 0:
        return assign

    while True:
        best_key, best_opt, best_gain = None, None, 0.0
        for key, opts in total_scores.items():
            cur = assign[key]
            for opt in opts:
                extra_bits = opt[1] - cur[1]
                if extra_bits <= 0 or extra_bits > slack:
                    continue
                gain = (cur[2] - opt[2]) / extra_bits
                if gain > best_gain:
                    best_key, best_opt, best_gain = key, opt, gain
        if best_key is None:
            break
        slack -= best_opt[1] - assign[best_key][1]
        assign[best_key] = best_opt
    return assign


def _swap_local_search(total_scores: dict, assign: dict, budget: int, max_rounds: int = 200) -> dict:
    """Close part of the duality gap with pairwise exchanges.

    A price-based solution can only ever land on the *convex hull* of each layer's
    (bits, loss) curve, so options that are dominated in the hull -- but optimal in the
    true (non-convex) problem -- are unreachable for every ``lam``. One downgrade funding
    one upgrade repairs the common cases.
    """
    for _ in range(max_rounds):
        used = _total_bits(assign)
        best_move, best_delta = None, -_EPS
        for up_key, up_opts in total_scores.items():
            up_cur = assign[up_key]
            for up_opt in up_opts:
                if up_opt[1] <= up_cur[1]:
                    continue
                need = up_opt[1] - up_cur[1]
                gain = up_cur[2] - up_opt[2]
                if gain <= 0:
                    continue
                if used + need <= budget:  # pure upgrade, handled by _greedy_repair
                    continue
                for down_key, down_opts in total_scores.items():
                    if down_key == up_key:
                        continue
                    down_cur = assign[down_key]
                    for down_opt in down_opts:
                        freed = down_cur[1] - down_opt[1]
                        if freed <= 0 or used + need - freed > budget:
                            continue
                        delta = gain - (down_opt[2] - down_cur[2])
                        if delta > best_delta:
                            best_delta = delta
                            best_move = (up_key, up_opt, down_key, down_opt)
        if best_move is None:
            break
        up_key, up_opt, down_key, down_opt = best_move
        assign[up_key], assign[down_key] = up_opt, down_opt
    return assign


def solve_allocation(total_scores: dict, budget: int, solver: str = "dp", max_states: Optional[int] = None):
    """Dispatch to the requested allocation solver.

    Args:
        total_scores: Score table.
        budget: Total bit budget.
        solver: ``"dp"`` (knapsack DP, the historical default) or ``"lagrangian"``
            (shadow-price bisection).
        max_states: DP beam width; ignored by the Lagrangian solver.

    Returns:
        The assignment (``key -> option``), or ``None`` when the target is infeasible.
    """
    if solver == "lagrangian":
        return solve_lagrangian(total_scores, budget)

    from auto_round.auto_scheme.delta_loss import choose_bits_per_layer_with_path

    _, path = choose_bits_per_layer_with_path(total_scores, budget, max_states=max_states)
    if path is None:
        return None

    chosen_index = {tuple(names): scheme_index for names, scheme_index in path}
    assign = {}
    for key, opts in total_scores.items():
        for opt in opts:
            if tuple(opt[3]) in chosen_index and chosen_index[tuple(opt[3])] == opt[0]:
                assign[key] = opt
                break
        else:  # pragma: no cover - defensive
            assign[key] = min(opts, key=lambda o: o[1])
    return assign

