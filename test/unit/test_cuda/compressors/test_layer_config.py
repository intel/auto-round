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
"""Real-GPU mixed-precision (layer_config) tests.

Exercises ``auto_round.compressors`` and the per-layer config resolution path on
the GPU by exporting a model with a mixed-precision ``layer_config`` and
verifying the reloaded model runs on CUDA.
"""

import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM

from auto_round import AutoRound

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")


class TestLayerConfigGpu:
    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    @pytest.mark.timeout(180)
    def test_mixed_precision_quantize_reload(self, tiny_opt_model_path):
        layer_config = {
            "model.decoder.layers.0.self_attn.q_proj": {"bits": 8},
            "model.decoder.layers.1.fc1": {"bits": 4, "group_size": 32},
        }
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            group_size=128,
            sym=True,
            layer_config=layer_config,
            iters=0,
            disable_opt_rtn=True,
            nsamples=1,
            seqlen=16,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cuda:0", trust_remote_code=True)
        assert isinstance(model, torch.nn.Module)
        input_ids = torch.randint(0, 1000, (1, 16), device="cuda:0")
        with torch.no_grad():
            out = model(input_ids)
        assert out.logits.shape[0] == 1

    @pytest.mark.timeout(180)
    def test_layer_config_string_scheme(self, tiny_opt_model_path):
        """layer_config values given as preset scheme strings."""
        layer_config = {
            "model.decoder.layers.0.self_attn.q_proj": "W8A16",
            "model.decoder.layers.1.fc1": "W4A16",
        }
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            group_size=128,
            sym=True,
            layer_config=layer_config,
            iters=0,
            disable_opt_rtn=True,
            nsamples=1,
            seqlen=16,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cuda:0", trust_remote_code=True)
        assert isinstance(model, torch.nn.Module)

    @pytest.mark.timeout(180)
    def test_ignore_layers_not_quantized(self, tiny_opt_model_path):
        """A layer listed in ignore_layers must not be quantized."""
        ignored = "model.decoder.layers.0.self_attn.q_proj"
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            group_size=128,
            sym=True,
            ignore_layers=ignored,
            iters=0,
            disable_opt_rtn=True,
            nsamples=1,
            seqlen=16,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cuda:0", trust_remote_code=True)
        q_proj = model
        for part in ignored.split("."):
            q_proj = getattr(q_proj, part)
        assert not hasattr(q_proj, "qweight"), "ignored layer was unexpectedly quantized"

    @pytest.mark.timeout(180)
    def test_layer_config_bits2_8(self, tiny_opt_model_path):
        """Mixed 2-bit and 8-bit per-layer config."""
        layer_config = {
            "model.decoder.layers.0.self_attn.q_proj": {"bits": 8},
            "model.decoder.layers.1.fc1": {"bits": 2, "group_size": 32},
        }
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            group_size=128,
            sym=True,
            layer_config=layer_config,
            iters=0,
            disable_opt_rtn=True,
            nsamples=1,
            seqlen=16,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cuda:0", trust_remote_code=True)
        assert isinstance(model, torch.nn.Module)
