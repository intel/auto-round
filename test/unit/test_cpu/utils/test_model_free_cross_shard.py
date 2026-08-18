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

"""Unit tests for cross-shard FP8 scale handling in model_free_utils."""

import json
import os
import tempfile

import pytest
import torch

from auto_round.utils.model_free_utils import (
    _build_cross_shard_pairs_from_weight_map,
    _hydrate_missing_fp8_scales_from_index,
)

# ---------------------------------------------------------------------------
# _build_cross_shard_pairs_from_weight_map
# ---------------------------------------------------------------------------


class TestBuildCrossShardPairs:
    """Tests for _build_cross_shard_pairs_from_weight_map."""

    def _make_weight_map(self, entries: dict[str, str]) -> dict[str, str]:
        return dict(entries)

    def test_no_fp8_entries_returns_empty(self):
        weight_map = {
            "model.layer.weight": "shard-00001.safetensors",
            "model.layer.bias": "shard-00001.safetensors",
        }
        recipient_to_donors, donor_shard_tensors = _build_cross_shard_pairs_from_weight_map(weight_map)
        assert recipient_to_donors == {}
        assert donor_shard_tensors == {}

    def test_same_shard_scale_not_cross(self):
        """weight and weight_scale_inv in the same shard → not a cross-shard pair."""
        weight_map = {
            "model.layer.weight": "shard-00001.safetensors",
            "model.layer.weight_scale_inv": "shard-00001.safetensors",
        }
        recipient_to_donors, donor_shard_tensors = _build_cross_shard_pairs_from_weight_map(weight_map)
        assert recipient_to_donors == {}
        assert donor_shard_tensors == {}

    def test_cross_shard_single_pair(self):
        """weight in shard-1, weight_scale_inv in shard-2."""
        weight_map = {
            "model.layer.weight": "shard-00001.safetensors",
            "model.layer.weight_scale_inv": "shard-00002.safetensors",
        }
        recipient_to_donors, donor_shard_tensors = _build_cross_shard_pairs_from_weight_map(weight_map)

        assert "shard-00001.safetensors" in recipient_to_donors
        donor_map = recipient_to_donors["shard-00001.safetensors"]
        assert "shard-00002.safetensors" in donor_map
        assert "model.layer.weight_scale_inv" in donor_map["shard-00002.safetensors"]

        assert "shard-00002.safetensors" in donor_shard_tensors
        assert "model.layer.weight_scale_inv" in donor_shard_tensors["shard-00002.safetensors"]

    def test_cross_shard_multiple_layers_one_donor(self):
        """Multiple layers whose scale_inv all live in the same donor shard."""
        weight_map = {
            "model.layer0.weight": "shard-00001.safetensors",
            "model.layer0.weight_scale_inv": "shard-00002.safetensors",
            "model.layer1.weight": "shard-00001.safetensors",
            "model.layer1.weight_scale_inv": "shard-00002.safetensors",
        }
        recipient_to_donors, donor_shard_tensors = _build_cross_shard_pairs_from_weight_map(weight_map)

        donor_map = recipient_to_donors["shard-00001.safetensors"]
        scales = donor_map["shard-00002.safetensors"]
        assert "model.layer0.weight_scale_inv" in scales
        assert "model.layer1.weight_scale_inv" in scales
        assert len(donor_shard_tensors["shard-00002.safetensors"]) == 2

    def test_cross_shard_multiple_donors(self):
        """Recipient shard needs scales from two different donor shards."""
        weight_map = {
            "model.layerA.weight": "shard-00001.safetensors",
            "model.layerA.weight_scale_inv": "shard-00002.safetensors",
            "model.layerB.weight": "shard-00001.safetensors",
            "model.layerB.weight_scale_inv": "shard-00003.safetensors",
        }
        recipient_to_donors, donor_shard_tensors = _build_cross_shard_pairs_from_weight_map(weight_map)

        donor_map = recipient_to_donors["shard-00001.safetensors"]
        assert "shard-00002.safetensors" in donor_map
        assert "shard-00003.safetensors" in donor_map
        assert len(donor_shard_tensors) == 2

    def test_scale_inv_without_matching_weight_ignored(self):
        """weight_scale_inv present in weight_map but no corresponding .weight → ignored."""
        weight_map = {
            "model.layer.weight_scale_inv": "shard-00002.safetensors",
            # no "model.layer.weight" key at all
        }
        recipient_to_donors, donor_shard_tensors = _build_cross_shard_pairs_from_weight_map(weight_map)
        assert recipient_to_donors == {}
        assert donor_shard_tensors == {}

    def test_empty_weight_map(self):
        recipient_to_donors, donor_shard_tensors = _build_cross_shard_pairs_from_weight_map({})
        assert recipient_to_donors == {}
        assert donor_shard_tensors == {}


# ---------------------------------------------------------------------------
# _hydrate_missing_fp8_scales_from_index
# ---------------------------------------------------------------------------


def _write_fake_fp8_shard(path: str, tensors: dict[str, torch.Tensor]) -> None:
    """Save a dict of tensors as a safetensors file."""
    from safetensors.torch import save_file

    save_file(tensors, path)


def _write_index_json(directory: str, weight_map: dict[str, str]) -> str:
    index_path = os.path.join(directory, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump({"weight_map": weight_map}, f)
    return index_path


class TestHydrateMissingFp8Scales:
    """Tests for _hydrate_missing_fp8_scales_from_index."""

    def test_non_safetensors_shard_returns_unchanged(self, tmp_path):
        raw = {"w": torch.zeros(4)}
        result = _hydrate_missing_fp8_scales_from_index(raw, str(tmp_path / "model.bin"))
        assert result is raw

    def test_no_fp8_weights_returns_unchanged(self, tmp_path):
        shard_path = str(tmp_path / "shard-00001.safetensors")
        # BF16 weight — not FP8, so nothing to hydrate
        raw = {"model.layer.weight": torch.zeros(4, dtype=torch.bfloat16)}
        result = _hydrate_missing_fp8_scales_from_index(raw, shard_path)
        assert result is raw

    def test_all_scales_present_returns_unchanged(self, tmp_path):
        shard_path = str(tmp_path / "shard-00001.safetensors")
        raw = {
            "model.layer.weight": torch.zeros(4, dtype=torch.float8_e4m3fn),
            "model.layer.weight_scale_inv": torch.ones(1),
        }
        result = _hydrate_missing_fp8_scales_from_index(raw, shard_path)
        assert "model.layer.weight_scale_inv" in result

    def test_cross_shard_hydration_local_mode(self, tmp_path):
        """Recipient shard gets scale_inv from donor shard in local (non-streaming) mode."""
        donor_name = "shard-00002.safetensors"
        recipient_name = "shard-00001.safetensors"
        scale_name = "model.layer.weight_scale_inv"

        # Write donor shard with scale_inv
        donor_path = tmp_path / donor_name
        _write_fake_fp8_shard(str(donor_path), {scale_name: torch.ones(1)})

        # Write index.json in the same directory
        weight_map = {
            "model.layer.weight": recipient_name,
            scale_name: donor_name,
        }
        _write_index_json(str(tmp_path), weight_map)

        # Recipient shard has FP8 weight but no scale_inv
        recipient_path = tmp_path / recipient_name
        raw = {"model.layer.weight": torch.zeros(4, dtype=torch.float8_e4m3fn)}

        result = _hydrate_missing_fp8_scales_from_index(raw, str(recipient_path))
        assert scale_name in result, "scale_inv should be hydrated from donor shard"
        assert result[scale_name].dtype == torch.float32 or result[scale_name].numel() == 1

    def test_cross_shard_hydration_streaming_mode(self, tmp_path):
        """In streaming mode, index.json lives in work_dir, shards in cache subdir."""
        work_dir = tmp_path / "work_dir"
        cache_dir = work_dir / ".cache" / "model_free_source_shards"
        work_dir.mkdir()
        cache_dir.mkdir(parents=True)

        donor_name = "shard-00002.safetensors"
        recipient_name = "shard-00001.safetensors"
        scale_name = "model.layer.weight_scale_inv"

        # Donor shard lives in cache subdir
        _write_fake_fp8_shard(str(cache_dir / donor_name), {scale_name: torch.ones(1)})

        # Index.json lives in work_dir (not in cache subdir)
        weight_map = {
            "model.layer.weight": recipient_name,
            scale_name: donor_name,
        }
        _write_index_json(str(work_dir), weight_map)

        # Recipient shard path is in cache dir
        recipient_path = cache_dir / recipient_name
        raw = {"model.layer.weight": torch.zeros(4, dtype=torch.float8_e4m3fn)}

        result = _hydrate_missing_fp8_scales_from_index(
            raw,
            str(recipient_path),
            index_dir=str(work_dir),
            donor_shard_dir=str(cache_dir),
        )
        assert scale_name in result, "streaming mode: scale_inv should be hydrated via index_dir/donor_shard_dir params"

    def test_missing_index_json_returns_unchanged(self, tmp_path):
        """If no index.json exists, raw_tensors is returned as-is (no crash)."""
        shard_path = str(tmp_path / "shard-00001.safetensors")
        raw = {"model.layer.weight": torch.zeros(4, dtype=torch.float8_e4m3fn)}
        result = _hydrate_missing_fp8_scales_from_index(raw, shard_path)
        assert result is raw

    def test_donor_shard_missing_on_disk_returns_unchanged(self, tmp_path):
        """Index references a donor shard that doesn't exist on disk → graceful skip."""
        scale_name = "model.layer.weight_scale_inv"
        recipient_name = "shard-00001.safetensors"
        donor_name = "shard-00002.safetensors"  # not written

        weight_map = {
            "model.layer.weight": recipient_name,
            scale_name: donor_name,
        }
        _write_index_json(str(tmp_path), weight_map)

        raw = {"model.layer.weight": torch.zeros(4, dtype=torch.float8_e4m3fn)}
        result = _hydrate_missing_fp8_scales_from_index(raw, str(tmp_path / recipient_name))
        assert scale_name not in result  # hydration skipped silently

    def test_multiple_layers_hydrated_from_single_donor(self, tmp_path):
        """Multiple missing scale_inv tensors are hydrated in a single donor open."""
        donor_name = "shard-00002.safetensors"
        recipient_name = "shard-00001.safetensors"
        scales = {
            "model.layerA.weight_scale_inv": torch.ones(1),
            "model.layerB.weight_scale_inv": torch.ones(1),
        }
        _write_fake_fp8_shard(str(tmp_path / donor_name), scales)

        weight_map = {
            "model.layerA.weight": recipient_name,
            "model.layerA.weight_scale_inv": donor_name,
            "model.layerB.weight": recipient_name,
            "model.layerB.weight_scale_inv": donor_name,
        }
        _write_index_json(str(tmp_path), weight_map)

        raw = {
            "model.layerA.weight": torch.zeros(4, dtype=torch.float8_e4m3fn),
            "model.layerB.weight": torch.zeros(4, dtype=torch.float8_e4m3fn),
        }
        result = _hydrate_missing_fp8_scales_from_index(raw, str(tmp_path / recipient_name))
        assert "model.layerA.weight_scale_inv" in result
        assert "model.layerB.weight_scale_inv" in result
