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
"""Fuzzing of the free-form CLI value parsers (``str2bool``, ``--group_size``, ``--layer_config``)."""

import argparse
import json

import pytest
from hypothesis import given, note
from hypothesis import strategies as st

from auto_round.cli.parser import _parse_group_size
from auto_round.utils.common import parse_layer_config_arg, str2bool

_TRUE_WORDS = ("yes", "true", "t", "y", "1")
_FALSE_WORDS = ("no", "false", "f", "n", "0")
_SCALAR_WORDS = ("true", "false", "null", "none", "nan", "inf", "infinity")
_PUNCT = "{}:,\"'“” \\-.+"

any_text = st.text(
    st.one_of(st.sampled_from(list(_PUNCT + "0123456789abc")), st.characters(exclude_categories=("Cs",))),
    max_size=40,
)


@st.composite
def random_case(draw, words):
    word = draw(st.sampled_from(words))
    flips = draw(st.lists(st.booleans(), min_size=len(word), max_size=len(word)))
    return "".join(c.upper() if flip else c for c, flip in zip(word, flips))


@given(value=st.one_of(random_case(_TRUE_WORDS + _FALSE_WORDS), any_text))
def test_str2bool_matches_word_model(value):
    lowered = value.lower()
    if lowered in _TRUE_WORDS:
        assert str2bool(value) is True
    elif lowered in _FALSE_WORDS:
        assert str2bool(value) is False
    else:
        with pytest.raises(argparse.ArgumentTypeError):
            str2bool(value)


@given(value=st.booleans())
def test_str2bool_passes_bools_through(value):
    assert str2bool(value) is value


@given(value=st.integers(-(2**40), 2**40))
def test_group_size_int_round_trips(value):
    parsed = _parse_group_size(str(value))
    assert type(parsed) is int and parsed == value


_pad = st.text(st.sampled_from(" \t"), max_size=2)


@given(values=st.lists(st.integers(-1, 4096), min_size=2, max_size=4), data=st.data())
def test_group_size_tuple_round_trips_with_padding(values, data):
    text = ",".join(data.draw(_pad) + str(v) + data.draw(_pad) for v in values)
    assert _parse_group_size(text) == tuple(values)


@given(text=any_text)
def test_group_size_returns_int_or_int_tuple_or_value_error(text):
    try:
        parsed = _parse_group_size(text)
    except ValueError:
        return
    if isinstance(parsed, tuple):
        assert parsed and all(type(v) is int for v in parsed)
    else:
        assert type(parsed) is int


@pytest.mark.xfail(
    strict=True,
    reason="fuzz-found: a comma-free --group_size such as ' 64' or '+64' parses to the 1-tuple (64,) instead of 64",
)
@given(value=st.integers(1, 4096), wrap=st.sampled_from([" {}", "{} ", "+{}", "\t{}"]))
def test_group_size_without_commas_is_an_int(value, wrap):
    parsed = _parse_group_size(wrap.format(value))
    assert type(parsed) is int and parsed == value


def _is_plain_token(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return token.lower() not in _SCALAR_WORDS
    return False


_identifier = st.from_regex(r"[A-Za-z_][A-Za-z0-9_]{0,10}", fullmatch=True).filter(_is_plain_token)
_layer_piece = st.one_of(_identifier, st.integers(0, 99).map(str))
_regex_piece = st.sampled_from(["\\.", "\\d+", "\\w*", ".*", "[0-9]+", "^", "$"])
_plain_layer_name = st.lists(_layer_piece, min_size=1, max_size=5).map(".".join).filter(_is_plain_token)
_regex_layer_name = st.lists(st.one_of(_layer_piece, _regex_piece), min_size=1, max_size=6).map("".join)
_regex_layer_name = _regex_layer_name.filter(lambda k: _is_plain_token(k) and k.strip() == k)

_scalars = st.one_of(
    st.integers(-(10**9), 10**9),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.none(),
    _identifier,
)
_attr_dicts = st.dictionaries(_identifier, _scalars, max_size=4)


def layer_configs(keys):
    return st.dictionaries(keys, st.one_of(_attr_dicts, _scalars), max_size=4)


def _typed(value):
    """Make ``1``, ``1.0`` and ``True`` compare unequal so type coercions are caught."""
    if isinstance(value, dict):
        return {k: _typed(v) for k, v in value.items()}
    return (type(value).__name__, value)


def _cli_dump(value, sep: str) -> str:
    if isinstance(value, dict):
        body = f",{sep}".join(f"{k}{sep}:{sep}{_cli_dump(v, sep)}" for k, v in value.items())
        return f"{{{sep}{body}{sep}}}"
    return repr(value) if isinstance(value, float) else str(value)


_quote_styles = st.sampled_from(["plain", "single", "double", "curly"])


def _quote(text: str, style: str) -> str:
    if style == "single":
        return f"'{text}'"
    if style == "double":
        return f'"{text}"'
    if style == "curly":
        parts = text.split('"')
        return "".join(p + ("“" if i % 2 == 0 else "”") for i, p in enumerate(parts[:-1])) + parts[-1]
    return text


@given(config=layer_configs(_plain_layer_name), indent=st.sampled_from([None, 0, 2]), style=_quote_styles)
def test_layer_config_json_round_trips(config, indent, style):
    text = _quote(json.dumps(config, indent=indent), style)
    note(f"text={text!r}")
    assert _typed(parse_layer_config_arg(text)) == _typed(config)


@given(config=layer_configs(_regex_layer_name), style=st.sampled_from(["plain", "single"]))
def test_layer_config_json_with_shell_style_regex_backslashes_round_trips(config, style):
    text = _quote(json.dumps(config).replace("\\\\", "\\"), style)
    note(f"text={text!r}")
    assert _typed(parse_layer_config_arg(text)) == _typed(config)


@given(config=layer_configs(st.one_of(_plain_layer_name, _regex_layer_name)), sep=st.sampled_from(["", " "]))
def test_layer_config_cli_syntax_round_trips(config, sep):
    text = _cli_dump(config, sep)
    note(f"text={text!r}")
    assert _typed(parse_layer_config_arg(text)) == _typed(config)


_malformed = st.one_of(any_text, any_text.map(lambda s: "{" + s), any_text.map(lambda s: "{" + s + "}"))


@given(text=_malformed)
def test_layer_config_parser_only_raises_value_error(text):
    try:
        parse_layer_config_arg(text)
    except ValueError:
        pass


@pytest.mark.xfail(
    strict=True,
    reason="fuzz-found: parse_layer_config_arg returns non-dict JSON values (e.g. '5', '[1]', 'null') although it is "
    "annotated '-> dict' and the CLI fallback path raises '--layer_config must parse to a dictionary'",
)
@given(value=st.one_of(st.integers(), st.booleans(), st.none(), st.lists(st.integers(), max_size=3)))
def test_layer_config_rejects_json_that_is_not_an_object(value):
    with pytest.raises(ValueError):
        parse_layer_config_arg(json.dumps(value))
