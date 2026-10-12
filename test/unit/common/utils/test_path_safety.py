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

"""Tests for checkpoint shard-path containment (CWE-22).

A sharded checkpoint declares its own shard file names in
``model.safetensors.index.json``.  Those names are attacker-controlled input, so
they must never be joined onto the checkpoint directory and opened verbatim: a
``../`` value climbs out of the directory and an absolute value replaces it,
which lets the model read (and inject) any file on the victim's filesystem.
"""

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from auto_round.utils.disk_stream_util import SafetensorsIndex
from auto_round.utils.offload import load_block_from_model_files
from auto_round.utils.path_safety import (
    UnsafeCheckpointPathError,
    resolve_within_directory,
    sanitize_shard_name,
    validate_weight_map,
)


def _write_index(model_dir: Path, weight_map: dict) -> None:
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))


def _make_outside_secrets(root: Path) -> Path:
    """Create weight files *outside* any model directory (the attack target)."""
    secret_dir = root / "SECRET_outside_model"
    secret_dir.mkdir(parents=True, exist_ok=True)
    save_file({"stolen.weight": torch.full((4,), 1337.0)}, str(secret_dir / "secret.safetensors"))
    torch.save({"blk.layer.weight": torch.full((1, 4), 999.0)}, str(secret_dir / "secret.bin"))
    return secret_dir


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 1)


# ---------------------------------------------------------------------------
# sanitize_shard_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "model-00001-of-00002.safetensors",
        "model.safetensors",
        "./model.safetensors",
        "nested/shard.safetensors",
    ],
)
def test_sanitize_shard_name_accepts_relative_names(name):
    assert sanitize_shard_name(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        None,
        42,
        "/etc/passwd",
        "/abs/model.safetensors",
        "C:\\weights\\model.safetensors",
        "\\\\server\\share\\model.safetensors",
        "model\x00.safetensors",
    ],
)
def test_sanitize_shard_name_rejects_unsafe_values(name):
    with pytest.raises(UnsafeCheckpointPathError):
        sanitize_shard_name(name)


# ---------------------------------------------------------------------------
# resolve_within_directory
# ---------------------------------------------------------------------------


def test_resolve_allows_file_inside_directory(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    shard = model_dir / "model-00001-of-00002.safetensors"
    shard.write_bytes(b"")

    assert resolve_within_directory(model_dir, "model-00001-of-00002.safetensors") == shard


def test_resolve_allows_subdirectory_inside_directory(tmp_path):
    model_dir = tmp_path / "model"
    (model_dir / "nested").mkdir(parents=True)
    shard = model_dir / "nested" / "shard.safetensors"
    shard.write_bytes(b"")

    assert resolve_within_directory(model_dir, "nested/shard.safetensors") == shard


def test_resolve_allows_missing_file_for_caller_side_handling(tmp_path):
    """A shard that is simply absent must not raise: callers keep their own
    missing-file handling (warnings, partial downloads, shard scheduling)."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    resolved = resolve_within_directory(model_dir, "model-00002-of-00002.safetensors")

    assert not resolved.exists()
    assert resolved.is_relative_to(model_dir)


@pytest.mark.parametrize(
    "shard_name",
    ["..", "../secret.safetensors", "../../secret.safetensors", "nested/../../secret.safetensors"],
)
def test_resolve_rejects_relative_traversal(tmp_path, shard_name):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (tmp_path / "secret.safetensors").write_bytes(b"")

    with pytest.raises(UnsafeCheckpointPathError):
        resolve_within_directory(model_dir, shard_name)


def test_resolve_rejects_absolute_path(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    outside = tmp_path / "secret.safetensors"
    outside.write_bytes(b"")

    with pytest.raises(UnsafeCheckpointPathError):
        resolve_within_directory(model_dir, str(outside))


def test_resolve_allows_hf_cache_snapshot_symlinks(tmp_path):
    """HF cache snapshots link every file to ``../../blobs/<sha>``; those links
    point outside the snapshot directory and must still be accepted."""
    repo = tmp_path / "models--org--name"
    snapshot = repo / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (repo / "blobs").mkdir()
    (repo / "blobs" / "deadbeef").write_bytes(b"")
    try:
        (snapshot / "model-00001-of-00001.safetensors").symlink_to("../../blobs/deadbeef")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available on this platform")

    resolved = resolve_within_directory(snapshot, "model-00001-of-00001.safetensors")

    assert resolved == snapshot / "model-00001-of-00001.safetensors"
    assert validate_weight_map({"w": "model-00001-of-00001.safetensors"}, snapshot)


def test_resolve_allows_symlink_pointing_inside(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    target = model_dir / "real.safetensors"
    target.write_bytes(b"")
    try:
        (model_dir / "link.safetensors").symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available on this platform")

    assert resolve_within_directory(model_dir, "link.safetensors") == model_dir / "link.safetensors"


def test_resolve_rejects_non_regular_file(tmp_path):
    """A FIFO declared as a shard would block the reader indefinitely, so it is
    rejected before anything tries to open it."""
    if not hasattr(os, "mkfifo"):
        pytest.skip("mkfifo is POSIX-only")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    os.mkfifo(model_dir / "shard.safetensors")

    with pytest.raises(UnsafeCheckpointPathError, match="not a regular file"):
        resolve_within_directory(model_dir, "shard.safetensors")


def test_resolve_rejects_directory(tmp_path):
    model_dir = tmp_path / "model"
    (model_dir / "shard.safetensors").mkdir(parents=True)

    with pytest.raises(UnsafeCheckpointPathError, match="not a regular file"):
        resolve_within_directory(model_dir, "shard.safetensors")


# ---------------------------------------------------------------------------
# validate_weight_map
# ---------------------------------------------------------------------------


def test_validate_weight_map_rejects_non_dict(tmp_path):
    with pytest.raises(UnsafeCheckpointPathError):
        validate_weight_map(["not", "a", "mapping"], tmp_path)


def test_validate_weight_map_rejects_traversal_and_names_tensor(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (tmp_path / "secret.safetensors").write_bytes(b"")

    with pytest.raises(UnsafeCheckpointPathError, match="model.layers.0.self_attn.q_proj.weight"):
        validate_weight_map(
            {"model.layers.0.self_attn.q_proj.weight": "../secret.safetensors"},
            model_dir,
            index_path=model_dir / "model.safetensors.index.json",
        )


def test_validate_weight_map_is_a_drop_in_replacement(tmp_path):
    """Callers join the returned names themselves, so the relative names have to
    survive validation unchanged."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    weight_map = {"a.weight": "model-00001-of-00002.safetensors", "b.weight": "nested/b.safetensors"}

    assert validate_weight_map(weight_map, model_dir) == weight_map


# ---------------------------------------------------------------------------
# SafetensorsIndex -- the disk-streaming shard reader
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shard_value", ["relative", "absolute"])
def test_safetensors_index_rejects_traversal(tmp_path, shard_value):
    secret_dir = _make_outside_secrets(tmp_path)
    model_dir = tmp_path / "malicious_model"
    model_dir.mkdir()
    if shard_value == "relative":
        shard_name = "../SECRET_outside_model/secret.safetensors"
    else:
        shard_name = str(secret_dir / "secret.safetensors")
    _write_index(model_dir, {"stolen.weight": shard_name})

    with pytest.raises(UnsafeCheckpointPathError):
        SafetensorsIndex(str(model_dir))


def test_safetensors_index_reads_legit_shard(tmp_path):
    """Positive control: an ordinary sharded checkpoint must still load."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    value = torch.full((4,), 7.0)
    save_file({"model.embed_tokens.weight": value}, str(model_dir / "model-00001-of-00001.safetensors"))
    _write_index(model_dir, {"model.embed_tokens.weight": "model-00001-of-00001.safetensors"})

    index = SafetensorsIndex(str(model_dir))
    tensor = index.read_tensor("model.embed_tokens.weight")

    assert torch.equal(tensor, value)


def test_safetensors_index_reads_unsharded_checkpoint(tmp_path):
    """Positive control for the single-file path, whose weight map is built from
    the file's own contents rather than from an index."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    value = torch.full((4,), 3.0)
    save_file({"model.embed_tokens.weight": value}, str(model_dir / "model.safetensors"))

    assert torch.equal(SafetensorsIndex(str(model_dir)).read_tensor("model.embed_tokens.weight"), value)


def test_safetensors_index_tensor_shape_is_contained(tmp_path):
    """``tensor_shape`` must go through the same containment as ``read_tensors``."""
    _make_outside_secrets(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    save_file({"a.weight": torch.zeros(2, 3)}, str(model_dir / "shard.safetensors"))
    _write_index(model_dir, {"a.weight": "shard.safetensors"})
    index = SafetensorsIndex(str(model_dir))

    assert index.tensor_shape("a.weight") == (2, 3)

    index.weight_map["stolen.weight"] = "../SECRET_outside_model/secret.safetensors"
    with pytest.raises(UnsafeCheckpointPathError):
        index.tensor_shape("stolen.weight")


# ---------------------------------------------------------------------------
# load_block_from_model_files -- the offloaded-reload reader
# ---------------------------------------------------------------------------


def test_load_block_from_model_files_rejects_traversal(tmp_path):
    _make_outside_secrets(tmp_path)
    model_dir = tmp_path / "malicious_model"
    model_dir.mkdir()
    _write_index(model_dir, {"blk.layer.weight": "../SECRET_outside_model/secret.bin"})

    block = _Block()
    before = block.layer.weight.detach().clone()

    with pytest.raises(UnsafeCheckpointPathError):
        load_block_from_model_files(str(model_dir), "blk.layer", block.layer)

    assert torch.equal(block.layer.weight.detach(), before), "weights must not be replaced from outside the model dir"


def test_load_block_from_model_files_loads_legit_shard(tmp_path):
    """Positive control: reloading a block from its own shard still works."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    value = torch.full((1, 4), 5.0)
    save_file({"blk.layer.weight": value}, str(model_dir / "shard.safetensors"))
    _write_index(model_dir, {"blk.layer.weight": "shard.safetensors"})

    block = _Block()
    load_block_from_model_files(str(model_dir), "blk.layer", block.layer)

    assert torch.equal(block.layer.weight.detach(), value)


# ---------------------------------------------------------------------------
# GGUF export and pipeline artifact copy
# ---------------------------------------------------------------------------


def test_gguf_get_tensor_from_file_rejects_traversal(tmp_path):
    from auto_round.export.export_to_gguf.special_handle import get_tensor_from_file

    _make_outside_secrets(tmp_path)
    model_dir = tmp_path / "malicious_model"
    model_dir.mkdir()
    _write_index(model_dir, {"stolen.weight": "../SECRET_outside_model/secret.safetensors"})

    with pytest.raises(UnsafeCheckpointPathError):
        get_tensor_from_file(str(model_dir), "stolen.weight")


@pytest.mark.parametrize("component_name", ["../../outside", "/abs/outside", "a/b", ".."])
def test_copy_pipeline_artifacts_rejects_unsafe_component_name(tmp_path, component_name):
    from auto_round.export.utils import _copy_pipeline_artifacts

    source_dir = tmp_path / "pipe"
    source_dir.mkdir()
    (source_dir / "model_index.json").write_text(json.dumps({component_name: ["diffusers", "X"]}))

    with pytest.raises(UnsafeCheckpointPathError):
        _copy_pipeline_artifacts(str(source_dir), str(tmp_path / "out"))


def test_find_pipeline_model_subfolder_rejects_traversal(tmp_path):
    from auto_round.utils.model import _find_pipeline_model_subfolder

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "config.json").write_text(json.dumps({"architectures": ["EvilForCausalLM"]}))
    source_dir = tmp_path / "pipe"
    source_dir.mkdir()
    (source_dir / "model_index.json").write_text(json.dumps({"../outside": ["transformers", "X"]}))

    with pytest.raises(UnsafeCheckpointPathError):
        _find_pipeline_model_subfolder(str(source_dir))
