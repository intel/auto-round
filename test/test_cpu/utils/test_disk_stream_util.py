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

"""Unit tests for auto_round.utils.disk_stream_util (AR_DISK_STREAM_MODEL primitives)."""

import torch
import torch.nn as nn
from accelerate import init_empty_weights

from auto_round.utils.disk_stream_util import (
    build_meta_model,
    free_module,
    materialize_module,
    materialize_non_block_params,
    total_resident_bytes,
)


class TestMaterializeModuleRoundTrip:
    def test_materialize_then_free_round_trip(self, tiny_opt_model_path):
        model, _tokenizer, index = build_meta_model(tiny_opt_model_path)
        block = model.model.decoder.layers[0]

        for _, tensor in list(block.named_parameters()):
            assert str(tensor.device) == "meta"

        materialize_module(block, "model.decoder.layers.0", index, device="cpu")
        for _, tensor in list(block.named_parameters()):
            assert str(tensor.device) != "meta"
        assert total_resident_bytes(block) > 0

        free_module(block)
        for _, tensor in list(block.named_parameters()):
            assert str(tensor.device) == "meta"
        assert total_resident_bytes(block) == 0

    def test_already_materialized_params_are_left_alone(self, tiny_opt_model_path):
        """Shared/tied weights may already be real by the time materialize_module
        runs on a second block referencing them; it must not touch (or crash on)
        a parameter that's already off meta."""
        model, _tokenizer, index = build_meta_model(tiny_opt_model_path)
        block = model.model.decoder.layers[0]
        materialize_module(block, "model.decoder.layers.0", index, device="cpu")
        weight_before = block.self_attn.k_proj.weight.data.clone()

        # Re-materializing an already-real block must be a no-op, not an error.
        materialize_module(block, "model.decoder.layers.0", index, device="cpu")
        assert torch.equal(block.self_attn.k_proj.weight.data, weight_before)


class TestMaterializeModuleDtype:
    """Regression coverage for the dtype-promotion bug: materialize_module used
    to always force the checkpoint's raw on-disk dtype, fighting whatever compute
    dtype the caller had already promoted the (still-meta) model to."""

    def test_prefers_declared_meta_dtype_over_checkpoint_dtype(self, tiny_opt_model_path):
        model, _tokenizer, index = build_meta_model(tiny_opt_model_path)
        block = model.model.decoder.layers[0]

        # Simulate ModelContext._set_amp_dtype()'s `model.to(amp_dtype)` promoting
        # the still-meta block to bf16, even though the checkpoint itself is fp16
        # (facebook/opt-125m's native dtype).
        block.to(torch.bfloat16)
        for _, tensor in block.named_parameters():
            assert tensor.dtype == torch.bfloat16

        materialize_module(block, "model.decoder.layers.0", index, device="cpu")
        for name, tensor in block.named_parameters():
            assert tensor.dtype == torch.bfloat16, f"{name} was materialized as {tensor.dtype}, expected bfloat16"

    def test_falls_back_to_checkpoint_dtype_when_meta_declared_float32(self, tiny_opt_model_path):
        """The one case the checkpoint's dtype must still win: a meta skeleton
        built without an enclosing dtype context (e.g. a module built under
        `torch.device("meta")` alone) defaults to float32 regardless of the
        checkpoint's real dtype."""
        with init_empty_weights():
            linear = nn.Linear(64, 64, bias=False)
        assert linear.weight.dtype == torch.float32

        from auto_round.utils.disk_stream_util import SafetensorsIndex

        with torch.no_grad():
            real_weight = torch.randn(64, 64, dtype=torch.float16)

        class _FakeIndex(SafetensorsIndex):
            def __init__(self):
                self.weight_map = {"fc.weight": "dummy"}

            def has_tensor(self, name):
                return name in self.weight_map

            def read_tensors(self, names, device="cpu"):
                return {n: real_weight.clone() for n in names}

        materialize_module(linear, "fc", _FakeIndex(), device="cpu")
        assert linear.weight.dtype == torch.float16


class TestBuildMetaModel:
    def test_non_block_params_are_real_blocks_stay_meta(self, tiny_opt_model_path):
        model, tokenizer, index = build_meta_model(tiny_opt_model_path)
        assert tokenizer is not None

        block_names = ["model.decoder.layers.0", "model.decoder.layers.1"]
        materialize_non_block_params(model, block_names, index, device="cpu")

        # Embeddings (a non-block module) must be real now.
        assert str(model.model.decoder.embed_tokens.weight.device) != "meta"
        # Decoder blocks must still be untouched (meta).
        for name, tensor in model.model.decoder.layers[0].named_parameters():
            assert str(tensor.device) == "meta", f"{name} was unexpectedly materialized"
