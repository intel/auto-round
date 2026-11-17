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
"""Fuzzing of layer-name pattern handling: ``to_standard_regex``, ``matches_any_regex`` and the
model-free ``_PatternMatcher``.

All callers match with ``re.search``, so the oracles below are stated in search semantics: a
meta-free pattern is a plain substring, a bare ``.`` is a literal dot, and a trailing ``.`` on an
ignore pattern means "followed by a component boundary".
"""

import re

import pytest
from hypothesis import assume, given, note
from hypothesis import strategies as st

from auto_round.utils.common import matches_any_regex, to_standard_regex
from auto_round.utils.model_free_utils import _PatternMatcher

_META = set("*{}^$|()[]?+\\")
_VOCAB = ["model", "layers", "0", "1", "10", "mlp", "gate", "gate_proj", "up_proj", "q_proj", "self_attn", "experts"]

layer_names = st.lists(st.sampled_from(_VOCAB), min_size=1, max_size=6).map(".".join)
literal_patterns = st.text(st.sampled_from(list("abgmelp01._-/ ")), min_size=1, max_size=12)
_regexish = st.text(st.sampled_from(list("ab.*+?{}^$|()[]\\0123,-")), max_size=16)
# Repeat counts >= 2**32 hit an OverflowError path that is covered by its own xfail below.
regexish_patterns = _regexish.filter(lambda p: not re.search(r"\d{10,}", p))

# (user pattern, equivalent python regex) tokens; a bare "." is a literal dot, quantified dots stay regex.
_TOKENS = [
    ("model", "model"),
    ("layers", "layers"),
    ("mlp", "mlp"),
    ("gate", "gate"),
    ("proj", "proj"),
    ("0", "0"),
    ("1", "1"),
    (".", r"\."),
    ("_", "_"),
    (r"\.", r"\."),
    (".*", ".*"),
    (".+", ".+"),
    (r"\d+", r"\d+"),
    ("[0-9]+", "[0-9]+"),
    ("(q|k|v)_proj", "(q|k|v)_proj"),
    ("(gate|up)", "(gate|up)"),
]


@st.composite
def regex_patterns(draw):
    tokens = draw(st.lists(st.sampled_from(_TOKENS), min_size=1, max_size=6))
    head = draw(st.sampled_from(["", "^"]))
    tail = draw(st.sampled_from(["", "$"]))
    user = head + "".join(t for t, _ in tokens) + tail
    reference = head + "".join(r for _, r in tokens) + tail
    assume(any(c in _META for c in user))
    return user, reference


@given(pattern=literal_patterns, name=st.one_of(layer_names, literal_patterns), data=st.data())
def test_meta_free_patterns_match_as_substrings(pattern, name, data):
    if data.draw(st.booleans()):
        name = name[: len(name) // 2] + pattern + name[len(name) // 2 :]
    regex = to_standard_regex(pattern)
    assert bool(re.search(regex, name)) == (pattern in name)


@given(pattern_and_reference=regex_patterns(), name=layer_names)
def test_regex_patterns_keep_search_semantics_with_literal_bare_dots(pattern_and_reference, name):
    pattern, reference = pattern_and_reference
    regex = to_standard_regex(pattern)
    note(f"regex={regex!r}")
    assert bool(re.search(regex, name)) == bool(re.search(reference, name))


@given(pattern=st.one_of(regexish_patterns, literal_patterns))
def test_standard_regex_is_idempotent_and_compilable(pattern):
    try:
        regex = to_standard_regex(pattern)
    except ValueError:
        return
    re.compile(regex)
    assert to_standard_regex(regex) == regex


def _reference_matches_any(name: str, patterns) -> bool:
    for pattern in patterns:
        raw = pattern[2:] if pattern.startswith(("+:", "-:")) else pattern
        try:
            if re.search(raw, name):
                return True
        except re.error:
            continue
    return False


_prefixed = st.builds(lambda pre, p: pre + p, st.sampled_from(["", "+:", "-:"]), regexish_patterns)


@given(patterns=st.lists(st.one_of(_prefixed, literal_patterns), max_size=4), name=st.text("ab.01", max_size=10))
def test_matches_any_regex_follows_reference_and_never_raises(patterns, name):
    config = {p: {"bits": 8} for p in patterns}
    assert matches_any_regex(name, config) == _reference_matches_any(name, config)


@pytest.mark.xfail(
    strict=True,
    reason="fuzz-found: repeat counts >= 2**32 make re.compile raise OverflowError, which escapes both "
    "to_standard_regex (documented to raise ValueError) and matches_any_regex (documented to skip invalid regexes)",
)
@given(count=st.integers(2**32, 10**15), func=st.sampled_from(["to_standard_regex", "matches_any_regex"]))
def test_huge_repeat_counts_are_reported_as_invalid_patterns(count, func):
    pattern = f"layers.{{{count}}}"
    if func == "to_standard_regex":
        with pytest.raises(ValueError):
            to_standard_regex(pattern)
    else:
        assert matches_any_regex("layers.0", {pattern: {}}) is False


_DEFAULT_SCHEMES = st.sampled_from([{"bits": 4, "group_size": 128}, {"bits": 8}, {"group_size": 32}])
_LAYER_SCHEMES = st.fixed_dictionaries({"bits": st.sampled_from([2, 4, 8, 16])}) | st.fixed_dictionaries(
    {"group_size": st.sampled_from([32, 64])}
)


def _expected_scheme(layer, layer_config, default):
    def apply(cfg):
        return None if cfg.get("bits", default.get("bits", 4)) >= 16 else {**default, **cfg}

    if layer in layer_config:
        return apply(layer_config[layer])
    for key, cfg in layer_config.items():
        if key in layer:
            return apply(cfg)
    return default


@st.composite
def _substring_keys(draw, layers):
    layer = draw(st.sampled_from(layers))
    start = draw(st.integers(0, len(layer) - 1))
    stop = draw(st.integers(start + 1, len(layer)))
    return layer[start:stop]


@given(layers=st.lists(layer_names, min_size=1, max_size=5, unique=True), default=_DEFAULT_SCHEMES, data=st.data())
def test_pattern_matcher_resolves_literal_keys_by_substring_with_exact_names_first(layers, default, data):
    keys = data.draw(st.lists(st.one_of(_substring_keys(layers), st.sampled_from(layers)), max_size=4, unique=True))
    layer_config = {key: data.draw(_LAYER_SCHEMES) for key in keys}
    note(f"layer_config={layer_config!r}")
    matcher = _PatternMatcher([], layer_config, default)
    for layer in layers:
        assert matcher.resolve_scheme(f"{layer}.weight") == _expected_scheme(layer, layer_config, default)


def _expected_ignored(layer, ignore_patterns, layer_config):
    if layer in layer_config:
        return False
    for pattern in ignore_patterns:
        if not pattern.endswith("."):
            if pattern in layer:
                return True
            continue
        core = pattern.rstrip(".")
        for start in range(len(layer) + 1):
            end = start + len(core)
            if layer.startswith(core, start) and (end == len(layer) or layer[end] == "."):
                return True
    return False


@given(layers=st.lists(layer_names, min_size=1, max_size=5, unique=True), data=st.data())
def test_pattern_matcher_ignore_patterns_honour_component_boundaries(layers, data):
    ignore = data.draw(
        st.lists(
            st.one_of(_substring_keys(layers), _substring_keys(layers).map(lambda k: k + "."), literal_patterns),
            max_size=4,
        )
    )
    exact = data.draw(st.lists(st.sampled_from(layers), max_size=2, unique=True))
    layer_config = {name: {"bits": 8} for name in exact}
    note(f"ignore={ignore!r} layer_config={layer_config!r}")
    matcher = _PatternMatcher(ignore, layer_config, {"bits": 4})
    for layer in layers:
        assert matcher.should_ignore(f"{layer}.weight") == _expected_ignored(layer, ignore, layer_config)


@pytest.mark.xfail(
    strict=True,
    reason="fuzz-found: _PatternMatcher falls back to plain substring matching on 're.error', but "
    "to_standard_regex raises ValueError for invalid regexes, so the fallback is dead and construction fails",
)
@given(layer=layer_names, broken=st.sampled_from(["[", "(", "**", "{2,1}", ")"]))
def test_pattern_matcher_falls_back_to_substring_for_invalid_regex_keys(layer, broken):
    key = layer + broken
    matcher = _PatternMatcher([], {key: {"bits": 8}}, {"bits": 4})
    assert matcher.resolve_scheme(f"model.{key}.x.weight") == {"bits": 8}
    assert matcher.resolve_scheme(f"{layer}.weight") == {"bits": 4}
