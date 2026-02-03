# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import os
import tempfile

import pytest
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer
from transformers.testing_utils import (
    require_accelerate,
    require_intel_extension_for_pytorch,
    require_torch_gpu,
    require_torch_multi_gpu,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available

from ...helpers import get_model_path

if is_torch_available():
    import torch


# @slow
@require_torch_gpu
@require_accelerate
class TestAutoRound:
    model_name = "OPEA/Qwen2.5-1.5B-Instruct-int4-sym-inc"
    input_text = "There is a girl who likes adventure,"
    EXPECTED_OUTPUTS = set()
    ## Different backends may produce slight variations in output
    EXPECTED_OUTPUTS.add(
        "There is a girl who likes adventure, and she has been exploring the world "
        "for many years. She travels to different countries and cultures, trying new "
        "things every day. One of her favorite places to visit is a small village in "
        "the mountains where"
    )
    EXPECTED_OUTPUTS.add(
        "There is a girl who likes adventure, and she has been exploring the world for many years. She has visited every country in Europe and has even traveled to some of the most remote parts of Africa. She enjoys hiking through the mountains and discovering"
    )
    # transformers 5.0.0
    EXPECTED_OUTPUTS.add(
        "There is a girl who likes adventure, and she has been exploring the world for many years. She has visited every country in Europe and has even traveled to some of the most remote parts of Africa. She has also climbed mountains, swum"
    )

    device_map = "cuda"

    # called only once for all test in this class
    @classmethod
    def setup_class(cls):
        """
        Setup quantized model
        """
        torch.cuda.synchronize()
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.quantized_model = AutoModelForCausalLM.from_pretrained(
            cls.model_name, device_map=cls.device_map, torch_dtype=torch.float16
        )

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def test_quantized_model(self):
        """
        Simple test that checks if the quantized model is working properly
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)
        output = self.quantized_model.generate(**input_ids, max_new_tokens=40, do_sample=False)
        assert self.tokenizer.decode(output[0], skip_special_tokens=True) in self.EXPECTED_OUTPUTS

    def test_raise_if_non_quantized(self, tiny_opt_model_path):
        model_id = tiny_opt_model_path
        quantization_config = AutoRoundConfig(bits=4)
        with pytest.raises(ValueError):
            _ = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)

    def test_quantized_model_bf16(self):
        """
        Simple test that checks if the quantized model is working properly with bf16
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)
        quantization_config = AutoRoundConfig(backend="triton")
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device_map,
            quantization_config=quantization_config,
        )

        output = quantized_model.generate(**input_ids, max_new_tokens=40, do_sample=False)
        output_sentence = self.tokenizer.decode(output[0], skip_special_tokens=True)
        assert output_sentence in self.EXPECTED_OUTPUTS

    @require_intel_extension_for_pytorch
    def test_quantized_model_on_cpu(self):
        """
        Simple test that checks if the quantized model is working properly
        """
        input_ids = self.tokenizer(self.input_text, return_tensors="pt")

        quantized_model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto")
        output = quantized_model.generate(**input_ids, max_new_tokens=40, do_sample=False)

        assert self.tokenizer.decode(output[0], skip_special_tokens=True) in self.EXPECTED_OUTPUTS

    def test_save_pretrained(self):
        """
        Simple test that checks if the quantized model is working properly after being saved and loaded
        """

        ## some backends like marlin/ipex will repack the weight that caused the weight shape changed
        with tempfile.TemporaryDirectory() as tmpdirname:
            quantization_config = AutoRoundConfig(backend="triton")
            quantized_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device_map,
                torch_dtype=torch.float16,
                quantization_config=quantization_config,
            )

            quantized_model.save_pretrained(tmpdirname)
            model = AutoModelForCausalLM.from_pretrained(tmpdirname, device_map="cuda")

            input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(torch_device)

            output = model.generate(**input_ids, max_new_tokens=40, do_sample=False)
            assert self.tokenizer.decode(output[0], skip_special_tokens=True) in self.EXPECTED_OUTPUTS

    @require_torch_multi_gpu
    def test_quantized_model_multi_gpu(self):
        """
        Simple test that checks if the quantized model is working properly with multiple GPUs
        """
        quantization_config = AutoRoundConfig(backend="triton")
        quantized_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", quantization_config=quantization_config, torch_dtype="auto"
        )
        input_ids = self.tokenizer(self.input_text, return_tensors="pt").to(quantized_model.device)
        output = quantized_model.generate(**input_ids, max_new_tokens=40, do_sample=False)
        assert self.tokenizer.decode(output[0], skip_special_tokens=True) in self.EXPECTED_OUTPUTS

    def test_convert_from_gptq(self):
        """
        Simple test that checks if auto-round work properly with gptq format
        """
        model_name = "ybelkada/opt-125m-gptq-4bit"

        quantization_config = AutoRoundConfig()

        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="cuda", quantization_config=quantization_config, torch_dtype="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        tokenizer.decode(model.generate(**inputs, max_new_tokens=5)[0])

    @require_intel_extension_for_pytorch
    def test_convert_from_awq_cpu(self):
        """
        Simple test that checks if auto-round work properly with awq format
        """
        model_name = "casperhansen/opt-125m-awq"

        quantization_config = AutoRoundConfig()

        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="cpu", quantization_config=quantization_config, torch_dtype="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        tokenizer.decode(model.generate(**inputs, max_new_tokens=5)[0])

    def test_mixed_bits(self):
        """
        Simple test that checks if auto-round work properly with mixed bits
        """
        model_name = get_model_path("facebook/opt-125m")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        layer_config = {
            "model.decoder.layers.0.self_attn.k_proj": {"bits": 8},
            "model.decoder.layers.6.self_attn.out_proj": {"bits": 2, "group_size": 32},
        }

        bits, group_size, sym = 4, 128, True
        from auto_round import AutoRound

        autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym, layer_config=layer_config)
        with tempfile.TemporaryDirectory() as tmpdirname:
            _, quantized_model_path = autoround.quantize_and_save(output_dir=tmpdirname)

            model = AutoModelForCausalLM.from_pretrained(
                quantized_model_path, torch_dtype=torch.float16, device_map="cuda"
            )
            text = "There is a girl who likes adventure,"
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            tokenizer.decode(model.generate(**inputs, max_new_tokens=5)[0])


#     FAILED export/test_gguf.py::TestAutoRound::test_gguf_format - AssertionError: qwen2 tuning fail
# FAILED export/test_gguf.py::TestAutoRound::test_all_format - SystemExit: 1
# FAILED export/test_gguf.py::TestAutoRound::test_vlm_gguf - AttributeError: 'Qwen2VLForConditionalGeneration' object has no attribute 'last_layer_name_to_block_name'
