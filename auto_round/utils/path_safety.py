# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Containment checks for file paths declared by model artifacts.

A sharded checkpoint ships its own ``*.index.json`` mapping tensor names to shard
file names.  That mapping is *attacker-controlled input*: a model directory or a
Hugging Face repo is a downloaded artifact, not trusted local state.  Joining such
a shard name onto a directory and opening it verbatim lets the artifact escape its
own directory (CWE-22): ``dir / "../../secret.safetensors"`` climbs out of ``dir``,
and ``dir / "/abs/path.safetensors"`` *replaces* ``dir`` entirely -- for
``pathlib``'s ``/`` and ``os.path.join`` alike.  The same artifact can also point a
shard at a FIFO or device node and block the reader forever.

Every read of an artifact-declared checkpoint file therefore goes through
:func:`resolve_within_directory` (or :func:`validate_weight_map` for a whole
``weight_map``), which enforces:

* the declared name may not be absolute,
* after lexical normalisation it must stay inside the base directory, and
* if it exists it must be a regular file (symlinks followed), so a FIFO / device
  node declared by the artifact cannot stall or divert the reader.

Containment is checked on the declared *name*, not on symlink targets: the
Hugging Face cache links every snapshot file to ``../../blobs/<sha>``, so
rejecting out-of-directory link targets would reject every Hub download.  A
symlink the artifact ships is trusted exactly like any other file it ships,
which is also what ``transformers`` / ``safetensors`` do when loading it.

Names whose target does not exist are returned as-is, so callers keep their
existing missing-file handling.  Validation is deliberately paired with
resolution *at the point of use*: validating the map fails fast with a clear
error, and resolving again at open time means a call site cannot accidentally
bypass the check by joining the raw name itself.
"""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath
from typing import Any, Dict

__all__ = [
    "UnsafeCheckpointPathError",
    "sanitize_shard_name",
    "resolve_within_directory",
    "validate_weight_map",
]


class UnsafeCheckpointPathError(ValueError):
    """Raised when a checkpoint artifact declares a path outside its own directory."""


def _looks_absolute(relative_path: str) -> bool:
    """True for POSIX-absolute paths and for Windows drive/UNC prefixes.

    ``C:/weights/x.safetensors`` is not absolute on POSIX, but joining it would
    still be interpreted as a drive path by a Windows consumer of the same
    checkpoint, so both spellings are rejected.
    """
    return os.path.isabs(relative_path) or PureWindowsPath(relative_path).is_absolute()


def sanitize_shard_name(shard_name: Any, *, origin: str = "weight_map") -> str:
    """Validate one artifact-declared shard name without touching the filesystem.

    Rejects the cases where a join stops being a lookup *inside* the directory:
    non-strings, empty names, embedded NUL bytes, and absolute paths.  ``..``
    escapes are caught by :func:`resolve_within_directory`, which is the
    only place that knows the directory to compare against.
    """
    if not isinstance(shard_name, str) or not shard_name:
        raise UnsafeCheckpointPathError(f"{origin}: shard name must be a non-empty string, got {shard_name!r}")
    if "\x00" in shard_name:
        raise UnsafeCheckpointPathError(f"{origin}: shard name contains a NUL byte: {shard_name!r}")
    if _looks_absolute(shard_name):
        raise UnsafeCheckpointPathError(
            f"{origin}: shard name {shard_name!r} is an absolute path; shard references must be "
            "relative to the checkpoint directory"
        )
    return shard_name


def resolve_within_directory(
    base_dir: str | os.PathLike,
    relative_path: Any,
    *,
    origin: str = "weight_map",
) -> Path:
    """Resolve an artifact-declared path to a file contained in *base_dir*.

    Args:
        base_dir: Directory the artifact is allowed to read from (the checkpoint
            or output directory the reference is relative to).
        relative_path: The path as declared by the artifact.
        origin: Human-readable label used in error messages, e.g.
            ``"model.safetensors.index.json[model.layers.0.weight]"``.

    Returns:
        The resolved absolute path.  Sub-directories inside *base_dir* are
        allowed; anything that escapes it is not.

    Raises:
        UnsafeCheckpointPathError: The path is absolute, escapes *base_dir*, or
            points at something that is not a regular file.
    """
    name = sanitize_shard_name(relative_path, origin=origin)
    base = Path(os.path.abspath(base_dir))
    # Callers open the returned normalised path, so ``link/..`` cannot be re-expanded by the OS.
    resolved = Path(os.path.normpath(base / name))
    if resolved != base and not resolved.is_relative_to(base):
        raise UnsafeCheckpointPathError(
            f"{origin}: path {name!r} resolves to {str(resolved)!r}, outside {str(base)!r}; " "refusing to open it"
        )
    # A FIFO (or device node) is not a weight file, and opening one blocks the
    # reader until a writer shows up -- a trivially reachable denial of service
    # from a crafted index.  Names that do not exist are left for the caller.
    if resolved.exists() and not resolved.is_file():
        raise UnsafeCheckpointPathError(
            f"{origin}: path {name!r} resolves to {str(resolved)!r}, which is not a regular file; "
            "refusing to open it"
        )
    return resolved


def validate_weight_map(
    weight_map: Any,
    base_dir: str | os.PathLike,
    *,
    index_path: str | os.PathLike | None = None,
) -> Dict[str, str]:
    """Validate every shard reference in an artifact-provided ``weight_map``.

    Returns a new dict with the same keys and the same relative shard names, so it
    is a drop-in replacement for ``json.load(...)["weight_map"]``.  Raises
    :class:`UnsafeCheckpointPathError` naming the offending tensor, which turns a
    malicious artifact into one clear error at load time instead of a confusing
    ``FileNotFoundError`` -- or, worse, a silent success -- at open time.
    """
    label = Path(index_path).name if index_path is not None else "weight_map"
    if not isinstance(weight_map, dict):
        raise UnsafeCheckpointPathError(
            f"{label}: 'weight_map' must be a JSON object mapping tensor names to shards, "
            f"got {type(weight_map).__name__}"
        )
    validated: Dict[str, str] = {}
    for tensor_name, shard_name in weight_map.items():
        resolve_within_directory(base_dir, shard_name, origin=f"{label}[{tensor_name}]")
        validated[tensor_name] = shard_name
    return validated
