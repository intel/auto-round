"""Unit tests for the AutoScheme bit-allocation DP.

These exercise ``choose_bits_per_layer_with_path`` directly (it previously had no
coverage) and pin down that the back-pointer reconstruction returns exactly the
same optimum as an exhaustive search, that the returned path is feasible and
visits every layer once, and that the ``max_states`` beam cap still yields a
valid full-length path.
"""

import itertools

import pytest

from auto_round.auto_scheme.delta_loss import choose_bits_per_layer_with_path


def _brute_force(layers, P):
    """Exhaustive reference: min total loss over all feasible option choices.

    Returns (best_loss, best_choice) where best_choice is the list of selected
    option indices per layer, or (None, None) if nothing fits the budget.
    """
    names = list(layers.keys())
    best_loss = None
    best_choice = None
    for combo in itertools.product(*(range(len(layers[n])) for n in names)):
        total_bits = sum(layers[names[i]][combo[i]][1] for i in range(len(names)))
        if total_bits > P:
            continue
        total_loss = sum(layers[names[i]][combo[i]][2] for i in range(len(names)))
        # Tie-break is irrelevant for the loss value we assert on.
        if best_loss is None or total_loss < best_loss:
            best_loss = total_loss
            best_choice = combo
    return best_loss, best_choice


def _make_layers(specs):
    """Build a ``layers`` dict from ``{name: [(bits, loss), ...]}``.

    Each option is stored as the (scheme, bits_cost, loss_cost, layer_names)
    tuple the DP expects; ``scheme`` is the option index so callers can map it
    back to a concrete scheme, exactly as ``_gen_layer_config`` does.
    """
    layers = {}
    for name, opts in specs.items():
        layers[name] = [(idx, bits, loss, (name,)) for idx, (bits, loss) in enumerate(opts)]
    return layers


def _assert_valid_path(layers, P, best_loss, best_path):
    names = list(layers.keys())
    # One entry per layer, in layer order.
    assert len(best_path) == len(names)
    for (layer_names, scheme), name in zip(best_path, names):
        assert layer_names == (name,)
    # The path is feasible and its loss matches the reported optimum.
    total_bits = 0
    total_loss = 0.0
    for (layer_names, scheme), name in zip(best_path, names):
        scheme_idx, bits_cost, loss_cost, opt_layer_names = layers[name][scheme]
        assert opt_layer_names == (name,)
        total_bits += bits_cost
        total_loss += loss_cost
    assert total_bits <= P
    assert total_loss == pytest.approx(best_loss)


def test_matches_brute_force_optimum():
    layers = _make_layers(
        {
            "l0": [(4, 1.0), (2, 4.0), (8, 0.2)],
            "l1": [(4, 0.9), (2, 3.5), (8, 0.1)],
            "l2": [(4, 1.1), (2, 5.0), (8, 0.15)],
            "l3": [(4, 0.8), (2, 2.0), (8, 0.05)],
            "l4": [(4, 1.3), (2, 4.5), (8, 0.25)],
        }
    )
    P = 22  # forces a non-trivial mix; uniform 8-bit (40) and 4-bit (20) bracket it
    best_loss, best_path = choose_bits_per_layer_with_path(layers, P)
    ref_loss, ref_choice = _brute_force(layers, P)

    assert ref_choice is not None
    assert best_loss == pytest.approx(ref_loss)
    _assert_valid_path(layers, P, best_loss, best_path)


def test_grouped_layer_names_preserved():
    # An option may cover several layers at once (e.g. tied weights); the grouped
    # tuple must survive reconstruction intact.
    layers = {
        "g0": [(0, 6, 1.0, ("a", "b")), (1, 3, 4.0, ("a", "b"))],
        "g1": [(0, 4, 0.5, ("c",)), (1, 2, 2.0, ("c",))],
    }
    P = 12
    best_loss, best_path = choose_bits_per_layer_with_path(layers, P)
    assert best_loss == pytest.approx(1.5)
    assert best_path == [(("a", "b"), 0), (("c",), 0)]


def test_infeasible_returns_none():
    layers = _make_layers({"l0": [(8, 1.0)], "l1": [(8, 1.0)]})
    best_loss, best_path = choose_bits_per_layer_with_path(layers, P=4)
    assert best_loss is None
    assert best_path is None


def test_max_states_beam_still_valid():
    # With a tight beam the result may be suboptimal, but it must remain a
    # feasible, full-length path.
    specs = {f"l{i}": [(4, 1.0 + 0.01 * i), (2, 3.0), (6, 0.5)] for i in range(8)}
    layers = _make_layers(specs)
    P = 30
    best_loss, best_path = choose_bits_per_layer_with_path(layers, P, max_states=2)
    assert best_path is not None
    _assert_valid_path(layers, P, best_loss, best_path)

    # max_states=1 collapses the beam to the single lowest-loss state each step,
    # which can greedily strand the budget and report infeasible; that is the
    # existing beam behaviour. We only require that *if* a path comes back it is
    # well-formed (the back-pointer reconstruction stays correct under pruning).
    best_loss1, best_path1 = choose_bits_per_layer_with_path(layers, P, max_states=1)
    if best_path1 is not None:
        _assert_valid_path(layers, P, best_loss1, best_path1)
