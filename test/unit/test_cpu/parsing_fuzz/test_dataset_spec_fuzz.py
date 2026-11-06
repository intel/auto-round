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
"""Fuzzing of the calibration dataset spec mini-language (``name:key=value,...``).

Round-trip properties hold on the documented domain of :func:`build_dataset_spec`:
dataset names are plain HF IDs / paths (they are not escaped), ``split``/``fields``
items cannot contain ``+`` (the list separator is not escaped), and every option value
except ``separator`` loses leading/trailing whitespace that is not ``\\n``/``\\t``.
"""

import dataclasses
import re

from hypothesis import given, note
from hypothesis import strategies as st

from auto_round.utils.dataset_utils import (
    CalibDataset,
    build_dataset_spec,
    escape_spec_value,
    parse_dataset_spec,
    split_unescaped,
    strip_dataset_spec_whitespace,
    unescape_spec_value,
)

_STRUCTURAL = ":,\\\n\t =+{}'\""
_spec_chars = st.one_of(st.sampled_from(list(_STRUCTURAL)), st.characters(exclude_categories=("Cs",)))
spec_text = st.text(_spec_chars, max_size=24)

_NAMES = st.from_regex(r"[A-Za-z0-9_][A-Za-z0-9_./-]{0,20}", fullmatch=True)


def _strip_safe(value: str) -> bool:
    escaped = escape_spec_value(value)
    return escaped == escaped.strip()


option_text = spec_text.filter(_strip_safe)
list_item = st.text(_spec_chars.filter(lambda c: c != "+"), min_size=1, max_size=12).filter(_strip_safe)


def _str_or_list(items):
    return st.one_of(items, st.lists(items, min_size=1, max_size=4))


@st.composite
def calib_datasets(draw):
    text_source = draw(st.sampled_from(["none", "fields", "template"]))
    return CalibDataset(
        name=draw(_NAMES),
        split=draw(st.none() | _str_or_list(list_item)),
        num=draw(st.none() | st.integers(-(10**6), 10**9)),
        concat=draw(st.booleans()),
        fields=draw(_str_or_list(list_item)) if text_source == "fields" else None,
        template=draw(option_text) if text_source == "template" else None,
        separator=draw(st.just("\n\n") | spec_text),
        apply_chat_template=draw(st.booleans()),
        system_prompt=draw(st.none() | option_text),
        timeout=draw(st.just(300) | st.integers(0, 10**6)),
    )


def _as_list(value):
    return [value] if isinstance(value, str) else value


def _expected_parse(ds: CalibDataset) -> dict:
    has_text_source = ds.template is not None or ds.fields is not None
    return {
        "name": ds.name,
        "split": _as_list(ds.split),
        "num": ds.num,
        "concat": ds.concat,
        "fields": _as_list(ds.fields),
        "template": ds.template,
        "separator": ds.separator if has_text_source else "\n\n",
        "apply_chat_template": ds.apply_chat_template,
        "system_prompt": ds.system_prompt,
        "timeout": ds.timeout,
    }


def _normalized(parsed: dict) -> dict:
    return {**parsed, "fields": _as_list(parsed["fields"])}


def _reference_split(spec: str, delim: str, maxsplit: int) -> list[str]:
    parts, current = [], ""
    for token in re.findall(r"\\.|.", spec, flags=re.DOTALL):
        if token == delim and (maxsplit < 0 or len(parts) < maxsplit):
            parts.append(current)
            current = ""
        else:
            current += token
    return parts + [current]


@given(value=spec_text)
def test_escape_round_trips_and_hides_structural_delimiters(value):
    escaped = escape_spec_value(value)
    assert unescape_spec_value(escaped) == value
    assert "\n" not in escaped and "\t" not in escaped
    for delim in ":,":
        assert split_unescaped(escaped, delim) == [escaped]


@given(spec=spec_text, delim=st.sampled_from([":", ",", "+", "="]), maxsplit=st.sampled_from([-1, 0, 1, 2, 5]))
def test_split_unescaped_matches_reference_and_is_lossless(spec, delim, maxsplit):
    parts = split_unescaped(spec, delim, maxsplit)
    assert parts == _reference_split(spec, delim, maxsplit)
    assert delim.join(parts) == spec
    if maxsplit >= 0:
        assert len(parts) <= maxsplit + 1
    else:
        assert all(split_unescaped(part, delim) == [part] for part in parts)


@given(ds=calib_datasets())
def test_spec_string_round_trips_through_parser(ds):
    spec = ds.to_spec_string()
    note(f"spec={spec!r}")
    assert spec == build_dataset_spec(**dataclasses.asdict(ds))
    assert _normalized(parse_dataset_spec(spec)) == _expected_parse(ds)


@given(datasets=st.lists(calib_datasets(), min_size=1, max_size=4))
def test_comma_joined_specs_split_back_into_datasets(datasets):
    specs = [ds.to_spec_string() for ds in datasets]
    assert split_unescaped(",".join(specs), ",") == specs


_cosmetic_ws = st.text(st.sampled_from(" \t"), max_size=3)


@given(datasets=st.lists(calib_datasets(), min_size=1, max_size=3), data=st.data())
def test_cosmetic_whitespace_is_stripped_without_touching_values(datasets, data):
    specs = [ds.to_spec_string() for ds in datasets]
    decorated_entries = []
    for spec in specs:
        name, *options = split_unescaped(spec, ":")
        entry = data.draw(_cosmetic_ws) + name + data.draw(_cosmetic_ws)
        for option in options:
            key, value = option.split("=", 1)
            entry += ":" + data.draw(_cosmetic_ws) + key + data.draw(_cosmetic_ws) + "=" + value
        decorated_entries.append(entry)
    decorated = ",".join(decorated_entries)
    note(f"decorated={decorated!r}")

    original = ",".join(specs)
    assert strip_dataset_spec_whitespace(original) == original
    assert strip_dataset_spec_whitespace(decorated) == original
    # parse_dataset_spec trims the name only when options follow it; option-less entries
    # rely on strip_dataset_spec_whitespace (always applied first by the calibration loaders).
    for entry, ds in zip(decorated_entries, datasets):
        if ":" in entry:
            assert _normalized(parse_dataset_spec(entry)) == _expected_parse(ds)


def _is_subsequence(short: str, long: str) -> bool:
    it = iter(long)
    return all(ch in it for ch in short)


@given(spec=spec_text)
def test_strip_whitespace_only_removes_whitespace(spec):
    stripped = strip_dataset_spec_whitespace(spec)
    assert _is_subsequence(stripped, spec)
    assert [c for c in stripped if not c.isspace()] == [c for c in spec if not c.isspace()]


_known_keys = st.sampled_from(["split", "num", "concat", "fields", "template", "separator", "timeout", "x"])
_option = st.one_of(spec_text, st.builds(lambda k, v: f"{k}={v}", _known_keys, spec_text))


@given(name=spec_text, options=st.lists(_option, max_size=4))
def test_parse_arbitrary_spec_returns_dict_or_value_error(name, options):
    spec = ":".join([name, *options])
    try:
        parsed = parse_dataset_spec(spec)
    except ValueError:
        return
    assert isinstance(parsed["name"], str)
    assert parsed["split"] is None or all(isinstance(s, str) for s in parsed["split"])
    assert parsed["num"] is None or isinstance(parsed["num"], int)
    assert isinstance(parsed["timeout"], int)
    assert isinstance(parsed["concat"], bool) and isinstance(parsed["apply_chat_template"], bool)
