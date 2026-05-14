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
from auto_round.context.compress import CompressContext
from auto_round.context.model import ModelContext


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


class _FormatStub:

    def get_backend_name(self):
        return "auto_round"


def _make_writer(model, output_dir, monkeypatch):
    ShardWriter.reset()
    compress_context = SimpleNamespace(formats=[_FormatStub()], output_dir=output_dir)
    model_context = SimpleNamespace(is_diffusion=False)
    monkeypatch.setattr(CompressContext, "get_context", classmethod(lambda cls: compress_context))
    monkeypatch.setattr(ModelContext, "get_context", classmethod(lambda cls: model_context))
    return ShardWriter(model, bits=4, max_shard_size="1MB", safe_serialization=False)


def test_finalize_saves_tail_layer_when_tie_word_embeddings_missing(tmp_path, monkeypatch):
    model = _DiffusionStyleModel()
    writer = _make_writer(model, str(tmp_path), monkeypatch)

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


def test_finalize_skips_lm_head_when_tie_word_embeddings_true(tmp_path, monkeypatch):
    """Complementary test: when tie_word_embeddings=True the lm_head should be
    skipped (not written to disk) and offloaded to meta."""
    model = _LMStyleModel()
    writer = _make_writer(model, str(tmp_path), monkeypatch)

    assert writer.lm_head_name == "lm_head"

    writer.save_module(model.transformer_blocks[0], "transformer_blocks.0")
    writer.finalize()

    shard_path = os.path.join(tmp_path, "model.bin")
    saved_tensors = torch.load(shard_path, map_location="cpu")

    assert "transformer_blocks.0.linear.weight" in saved_tensors
    assert "lm_head.weight" not in saved_tensors, "lm_head must be skipped when tied"
    assert model.lm_head.weight.device.type == "meta"


def test_finalize_offloads_module_with_tensor_in_parameters(tmp_path, monkeypatch):
    model = _DiffusionStyleModel()
    model.transformer_blocks[0].linear._parameters["weight"] = model.transformer_blocks[0].linear.weight.to("cpu")
    writer = _make_writer(model, str(tmp_path), monkeypatch)

    writer.save_module(model.transformer_blocks[0], "transformer_blocks.0")
    writer.finalize()

    offloaded_weight = model.transformer_blocks[0].linear._parameters["weight"]
    assert isinstance(offloaded_weight, torch.nn.Parameter)
    assert offloaded_weight.device.type == "meta"
