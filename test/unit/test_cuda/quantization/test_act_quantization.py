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
import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM

from auto_round import AutoRound
from auto_round.experimental.qmodules.fake import FakeActQuantLinear
from auto_round.wrapper import WrapperWALayer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires CUDA")


class TestActQuantizationGpu:
    @pytest.fixture(autouse=True)
    def _save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")
        yield
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def _reload_and_forward(self, quantized_model_path):
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cuda:0", trust_remote_code=True)
        assert isinstance(model, torch.nn.Module)
        input_ids = torch.randint(0, 1000, (1, 16), device="cuda:0")
        with torch.no_grad():
            out = model(input_ids)
        assert out.logits.shape[0] == 1

    @pytest.mark.timeout(180)
    def test_w4a8_act_dynamic(self, tiny_opt_model_path):
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            act_bits=8,
            group_size=128,
            act_group_size=32,
            act_dynamic=True,
            sym=True,
            iters=1,
            nsamples=2,
            seqlen=16,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")
        self._reload_and_forward(quantized_model_path)

    @pytest.mark.timeout(180)
    def test_w8a8_act_dynamic(self, tiny_opt_model_path):
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=8,
            act_bits=8,
            group_size=128,
            act_group_size=32,
            act_dynamic=True,
            sym=True,
            iters=0,
            disable_opt_rtn=True,
            nsamples=2,
            seqlen=16,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")
        self._reload_and_forward(quantized_model_path)

    @pytest.mark.timeout(180)
    def test_w4a8_act_static(self, tiny_opt_model_path):
        """Static (non-dynamic) activation quantization path."""
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            act_bits=8,
            group_size=128,
            act_group_size=128,
            act_dynamic=False,
            sym=True,
            iters=0,
            disable_opt_rtn=True,
            nsamples=2,
            seqlen=16,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")
        self._reload_and_forward(quantized_model_path)

    @pytest.mark.timeout(180)
    def test_w4a8_act_group128(self, tiny_opt_model_path):
        """Wider act_group_size exercises a different act-hook grouping path."""
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=4,
            act_bits=8,
            group_size=128,
            act_group_size=128,
            act_dynamic=True,
            sym=True,
            iters=1,
            nsamples=2,
            seqlen=16,
        )
        _, quantized_model_path = autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")
        self._reload_and_forward(quantized_model_path)

    @pytest.mark.timeout(180)
    def test_w8a8_asym(self, tiny_opt_model_path):
        """Asymmetric W8A8 cannot be exported to the auto_round format.

        ``auto_round`` supports W8A8 export only for the symmetric path; an
        asymmetric W8A8 config is rejected during format resolution.
        """
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=8,
            act_bits=8,
            group_size=128,
            act_group_size=32,
            act_dynamic=True,
            sym=False,
            iters=0,
            disable_opt_rtn=True,
            nsamples=2,
            seqlen=16,
        )
        with pytest.raises(ValueError, match="does not support exporting"):
            autoround.quantize_and_save(output_dir=self.save_dir, format="auto_round")

    @pytest.mark.timeout(180)
    def test_w4a8_reload_layer_type_diff_but_output_equal(self, tiny_opt_model_path):
        """Save path keeps wrappers; reload path materializes FakeActQuantLinear, outputs should match."""
        old_offload = os.environ.get("AR_DISABLE_OFFLOAD")
        os.environ["AR_DISABLE_OFFLOAD"] = "1"
        try:
            autoround = AutoRound(
                tiny_opt_model_path,
                bits=4,
                act_bits=8,
                group_size=128,
                act_group_size=32,
                act_dynamic=True,
                sym=True,
                iters=0,
                disable_opt_rtn=True,
                nsamples=2,
                seqlen=16,
            )
            quantized_model, quantized_model_path = autoround.quantize_and_save(
                output_dir=self.save_dir, format="auto_round"
            )

            quantized_model.eval()
            reloaded_model = AutoModelForCausalLM.from_pretrained(
                quantized_model_path, device_map="cuda:0", trust_remote_code=True
            )
            reloaded_model.eval()

            src_layer = quantized_model.model.decoder.layers[0].fc2
            dst_layer = reloaded_model.model.decoder.layers[0].fc2
            assert isinstance(src_layer, WrapperWALayer)
            assert isinstance(dst_layer, FakeActQuantLinear)
            assert type(src_layer) is not type(dst_layer)

            state_path = os.path.join(quantized_model_path, "model.safetensors")
            if os.path.exists(state_path):
                from safetensors.torch import load_file as load_state_dict

                state_dict = load_state_dict(state_path)
            else:
                state_dict = torch.load(os.path.join(quantized_model_path, "pytorch_model.bin"), weights_only=True)

            weight_key = "model.decoder.layers.0.fc2.weight"
            bias_key = "model.decoder.layers.0.fc2.bias"
            assert weight_key in state_dict
            assert bias_key in state_dict
            assert "model.decoder.layers.0.fc2.orig_layer.weight" not in state_dict

            torch.manual_seed(7)
            probe = torch.randn(2, 3, dst_layer.in_features, device="cuda:0", dtype=dst_layer.weight.dtype)
            with torch.no_grad():
                qdq_probe = dst_layer.qdq_input(probe)
                src_out = (
                    torch.nn.functional.linear(
                        qdq_probe,
                        state_dict[weight_key].to(qdq_probe.device, qdq_probe.dtype),
                        state_dict[bias_key].to(qdq_probe.device, qdq_probe.dtype),
                    )
                    .float()
                    .cpu()
                )
                dst_out = dst_layer(probe).float().cpu()

            torch.testing.assert_close(src_out, dst_out, rtol=1e-3, atol=1e-3)
        finally:
            if old_offload is None:
                os.environ.pop("AR_DISABLE_OFFLOAD", None)
            else:
                os.environ["AR_DISABLE_OFFLOAD"] = old_offload
