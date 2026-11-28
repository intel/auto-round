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
"""Fuzzing of the checkpoint path-containment guards in ``auto_round/utils/path_safety.py``.

Shard names in ``*.index.json`` are attacker-controlled. The oracle normalises the declared
name with an independent component stack (``..`` pops, ``.``/empty are dropped) and requires
the result to stay inside the checkpoint directory, compared component-wise so a sibling such
as ``base2/`` is never mistaken for ``base/``. Symlinks are deliberately *not* followed for
containment (Hugging Face cache layout), only for the regular-file check.
"""

import os
import re
from pathlib import Path
from test.fuzz_helpers import scaled_settings

import pytest
from hypothesis import example, given, note
from hypothesis import strategies as st

from auto_round.utils.path_safety import (
    UnsafeCheckpointPathError,
    resolve_within_directory,
    sanitize_shard_name,
    validate_weight_map,
)

pytestmark = pytest.mark.skipif(os.name != "posix" or not hasattr(os, "mkfifo"), reason="POSIX path semantics")

_ENTRIES = ["model.safetensors", "sub", "shard.safetensors", "fifo", "link_in", "link_out", "link_dir", "dangling"]
_COMPONENTS = st.sampled_from(_ENTRIES + ["missing.bin", "..", ".", "", "base", "base2", "x.safetensors"])

relative_names = st.lists(_COMPONENTS, min_size=1, max_size=6).map("/".join)
windows_absolute_names = st.builds(
    lambda template, tail: template.format(tail),
    st.sampled_from(["C:/{}", "c:\\{}", "\\\\server\\share\\{}", "//server/share/{}", "\\\\?\\C:\\{}"]),
    st.text(st.sampled_from("ab./\\"), max_size=8),
)
_WINDOWS_ABSOLUTE = re.compile(r"^([A-Za-z]:[\\/]|[\\/]{2})")
_any_text = st.text(st.characters(exclude_categories=("Cs",)), max_size=40)

# Known attack shapes, always replayed so the cheap CI profile cannot miss them.
_ATTACK_EXAMPLES = (
    "../base2/x.safetensors",
    "../../outside.safetensors",
    "sub/../../base2/x.safetensors",
    "../base/model.safetensors",
    "model.safetensors/..",
    ".",
    "fifo",
    "link_dir",
    "link_out",
    "dangling",
)


def _with_attack_examples(test):
    for name in _ATTACK_EXAMPLES:
        test = example(name=name)(test)
    return test


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    root = tmp_path_factory.mktemp("path_safety_fuzz")
    base = root / "base"
    (base / "sub").mkdir(parents=True)
    (root / "base2").mkdir()
    (root / "base2" / "x.safetensors").write_bytes(b"x")
    (root / "outside.safetensors").write_bytes(b"o")
    (base / "model.safetensors").write_bytes(b"m")
    (base / "sub" / "shard.safetensors").write_bytes(b"s")
    os.mkfifo(base / "fifo")
    os.symlink("model.safetensors", base / "link_in")
    os.symlink("../outside.safetensors", base / "link_out")
    os.symlink("sub", base / "link_dir")
    os.symlink("does_not_exist", base / "dangling")
    return base


def _lexical_parts(base: Path, name: str) -> list[str]:
    stack = list(base.parts)
    for component in name.split("/"):
        if component in ("", "."):
            continue
        if component == "..":
            if len(stack) > 1:
                stack.pop()
        else:
            stack.append(component)
    return stack


def _expected_outcome(base: Path, name):
    """Return ``("ok", path)`` or ``("reject", reason)`` for an artifact-declared name."""
    if not isinstance(name, str) or not name or "\x00" in name or name.startswith("/"):
        return "reject", "sanitize"
    if _WINDOWS_ABSOLUTE.match(name):
        return "reject", "windows absolute"
    parts = _lexical_parts(base, name)
    if parts[: len(base.parts)] != list(base.parts):
        return "reject", "escape"
    target = Path(*parts)
    if target == base or (os.path.exists(target) and not os.path.isfile(target)):
        return "reject", "not a regular file"
    return "ok", target


@scaled_settings(8)
@given(name=relative_names)
@_with_attack_examples
def test_resolve_matches_independent_containment_oracle(sandbox, name):
    kind, expected = _expected_outcome(sandbox, name)
    note(f"name={name!r} expected={kind}:{expected}")
    if kind == "reject":
        with pytest.raises(UnsafeCheckpointPathError):
            resolve_within_directory(sandbox, name)
        return
    resolved = resolve_within_directory(sandbox, name)
    assert resolved == expected
    assert resolved.is_absolute() and ".." not in resolved.parts
    assert resolved.is_relative_to(sandbox) and resolved != sandbox
    assert not resolved.exists() or resolved.is_file()


@scaled_settings(4)
@given(name=relative_names, as_path_like=st.booleans())
def test_resolve_accepts_relative_and_path_like_base_dirs(sandbox, name, as_path_like):
    relative_base = os.path.relpath(sandbox)
    base_arg = Path(relative_base) if as_path_like else relative_base
    kind, expected = _expected_outcome(sandbox, name)
    if kind == "reject":
        with pytest.raises(UnsafeCheckpointPathError):
            resolve_within_directory(base_arg, name)
    else:
        assert resolve_within_directory(base_arg, name) == expected


@given(name=st.one_of(windows_absolute_names, _any_text.map(lambda s: "/" + s)))
def test_absolute_names_are_rejected_for_both_path_flavours(sandbox, name):
    with pytest.raises(UnsafeCheckpointPathError):
        sanitize_shard_name(name)
    with pytest.raises(UnsafeCheckpointPathError):
        resolve_within_directory(sandbox, name)


@given(value=st.one_of(st.none(), st.integers(), st.binary(max_size=8), st.lists(st.text(max_size=3), max_size=2)))
def test_non_string_names_are_rejected(sandbox, value):
    with pytest.raises(UnsafeCheckpointPathError):
        sanitize_shard_name(value)
    with pytest.raises(UnsafeCheckpointPathError):
        resolve_within_directory(sandbox, value)


@given(prefix=_any_text, suffix=_any_text)
def test_nul_bytes_are_rejected(prefix, suffix):
    with pytest.raises(UnsafeCheckpointPathError):
        sanitize_shard_name(prefix + "\x00" + suffix)


@given(name=_any_text)
def test_arbitrary_names_either_resolve_inside_or_raise_the_dedicated_error(sandbox, name):
    try:
        resolved = resolve_within_directory(sandbox, name)
    except UnsafeCheckpointPathError as exc:
        assert isinstance(exc, ValueError)
        return
    assert sanitize_shard_name(name) == name
    assert resolved.is_relative_to(sandbox) and resolved != sandbox
    assert resolved == Path(*_lexical_parts(sandbox, name))


_tensor_names = st.text(st.sampled_from("abc.0123_"), min_size=1, max_size=12)


@given(
    weight_map=st.dictionaries(_tensor_names, st.one_of(relative_names, windows_absolute_names), max_size=6),
    index_name=st.sampled_from([None, "model.safetensors.index.json"]),
)
def test_validate_weight_map_reports_first_offending_tensor(sandbox, weight_map, index_name):
    index_path = None if index_name is None else sandbox / index_name
    offending = [tensor for tensor, shard in weight_map.items() if _expected_outcome(sandbox, shard)[0] == "reject"]
    note(f"offending={offending!r}")
    if not offending:
        validated = validate_weight_map(weight_map, sandbox, index_path=index_path)
        assert validated == weight_map and validated is not weight_map
        return
    with pytest.raises(UnsafeCheckpointPathError) as excinfo:
        validate_weight_map(weight_map, sandbox, index_path=index_path)
    label = index_name or "weight_map"
    assert f"{label}[{offending[0]}]" in str(excinfo.value)


@given(value=st.one_of(st.none(), st.integers(), st.lists(st.integers(), max_size=2), st.text(max_size=4)))
def test_validate_weight_map_rejects_non_mappings(sandbox, value):
    with pytest.raises(UnsafeCheckpointPathError):
        validate_weight_map(value, sandbox)
