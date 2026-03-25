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

import os
from types import SimpleNamespace

import torch

from auto_round.compressors.shard_writer import ShardWriter


class _ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)


class _DiffusionStyleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([_ToyBlock()])
        self.proj_out = torch.nn.Linear(4, 2)
        self.config = SimpleNamespace(model_type="toy-diffusion")


class _RounderStub:
    def __init__(self, model, output_dir):
        self.model = model
        self.bits = 4
        self.formats = [object()]
        self.max_shard_size = "1MB"
        self.safe_serialization = False
        self._output_dir = output_dir

    def _get_save_folder_name(self, _format):
        return self._output_dir


def test_finalize_saves_tail_layer_when_tie_word_embeddings_missing(tmp_path):
    model = _DiffusionStyleModel()
    rounder = _RounderStub(model, str(tmp_path))
    writer = ShardWriter(rounder)

    assert writer.lm_head_name == "proj_out"
    assert not hasattr(model.config, "tie_word_embeddings")

    writer.save_module(model.transformer_blocks[0], "transformer_blocks.0")
    writer.finalize()

    shard_path = os.path.join(tmp_path, "model.bin")
    saved_tensors = torch.load(shard_path, map_location="cpu")

    assert "transformer_blocks.0.linear.weight" in saved_tensors
    assert "proj_out.weight" in saved_tensors, "proj_out must be saved when tie_word_embeddings is absent"
    assert "proj_out.bias" in saved_tensors


class _LMStyleModel(torch.nn.Module):
    """Model whose config explicitly sets tie_word_embeddings=True."""

    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([_ToyBlock()])
        self.lm_head = torch.nn.Linear(4, 2, bias=False)
        self.config = SimpleNamespace(model_type="toy-lm", tie_word_embeddings=True)


def test_finalize_skips_lm_head_when_tie_word_embeddings_true(tmp_path):
    """Complementary test: when tie_word_embeddings=True the lm_head should be
    skipped (not written to disk) and offloaded to meta."""
    model = _LMStyleModel()
    rounder = _RounderStub(model, str(tmp_path))
    writer = ShardWriter(rounder)

    assert writer.lm_head_name == "lm_head"

    writer.save_module(model.transformer_blocks[0], "transformer_blocks.0")
    writer.finalize()

    shard_path = os.path.join(tmp_path, "model.bin")
    saved_tensors = torch.load(shard_path, map_location="cpu")

    assert "transformer_blocks.0.linear.weight" in saved_tensors
    assert "lm_head.weight" not in saved_tensors, "lm_head must be skipped when tied"
    assert model.lm_head.weight.device.type == "meta"


class _BlockWithPlainTensor(torch.nn.Module):
    """Simulates third-party modules that store plain Tensors in _parameters."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        # Inject a plain Tensor (not nn.Parameter) into _parameters,
        # mimicking what some fla/CUDA extension modules do.
        self._parameters["plain_weight"] = torch.zeros(4)


def test_finalize_handles_module_with_plain_tensor_in_parameters(tmp_path):
    """_offload_to_meta must not crash when a module has a plain torch.Tensor
    in _parameters (regression test for issue #1499)."""
    model = _DiffusionStyleModel()
    # Replace one block with one that has a plain tensor
    model.transformer_blocks[0] = _BlockWithPlainTensor()
    rounder = _RounderStub(model, str(tmp_path))
    writer = ShardWriter(rounder)

    writer.save_module(model.transformer_blocks[0], "transformer_blocks.0")
    writer.finalize()  # Must not raise AssertionError

    shard_path = os.path.join(tmp_path, "model.bin")
    saved_tensors = torch.load(shard_path, map_location="cpu")
    assert "transformer_blocks.0.linear.weight" in saved_tensors
    assert "transformer_blocks.0.plain_weight" in saved_tensors
    # The offloaded module should be on meta device
    assert model.transformer_blocks[0].linear.weight.device.type == "meta"
    assert model.transformer_blocks[0]._parameters["plain_weight"].device.type == "meta"
