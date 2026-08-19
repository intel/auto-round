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
"""Build the native MiniMax-H3 FL2VA layout around an AutoRound export."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

_SHARED_COMPONENTS = ("transformer", "tokenizer", "processor", "text_encoder", "scheduler", "audio_scheduler")


def _resolve_native_component(official_fl2va: Path, name: str) -> Path:
    candidates = [official_fl2va / name]
    if name == "video_vae":
        candidates.append(official_fl2va / "vae")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    expected = " or ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Native MiniMax-H3 component {name!r} not found at {expected}")


def repack_native_vae(checkpoint: str | Path, official_fl2va: str | Path) -> Path:
    """Add a self-contained native-VAE ``FL2VA`` partition to *checkpoint*.

    AutoRound exports the transformer and shared components in a modular
    Diffusers repository.  The VAE is not quantized, so the exported Diffusers
    VAE is replaced only in the Omni-facing partition by the original release
    remote-code components.  Shared non-VAE components remain linked to the
    AutoRound export, avoiding a second copy of the large transformer.
    """

    checkpoint = Path(checkpoint).resolve()
    official_fl2va = Path(official_fl2va).resolve()
    # Accept either the release's model root or its FL2VA subdirectory.  The
    # latter is convenient for a downloaded task-specific snapshot, while the
    # former is what users commonly have after downloading the full model.
    if (official_fl2va / "FL2VA").is_dir():
        official_fl2va = official_fl2va / "FL2VA"
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"AutoRound checkpoint directory not found: {checkpoint}")
    if not official_fl2va.is_dir():
        raise FileNotFoundError(f"Official MiniMax-H3 FL2VA directory not found: {official_fl2va}")

    partition = checkpoint / "FL2VA"
    if partition.exists():
        raise FileExistsError(f"Refusing to overwrite existing Omni partition: {partition}")

    for name in _SHARED_COMPONENTS:
        source = checkpoint / name
        if not source.exists():
            raise FileNotFoundError(f"AutoRound export is missing shared component {source}")

    partition.mkdir()
    manifest = {
        "_class_name": "MiniMaxH3Pipeline",
        "_minimax_h3": {
            "partition": "fl2va",
            "tasks": ["t2va", "fl2va"],
            "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
        },
    }
    (partition / "model_index.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    for name in _SHARED_COMPONENTS:
        (partition / name).symlink_to(Path("..") / name)
    for name in ("video_vae", "audio_vae"):
        shutil.copytree(_resolve_native_component(official_fl2va, name), partition / name)
    return partition
