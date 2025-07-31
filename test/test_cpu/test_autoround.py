import copy
import shutil
import sys
import unittest

from auto_round.eval.evaluation import simple_evaluate_user_model

sys.path.insert(0, "../..")
import torch
from _test_helpers import model_infer
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(3):
            yield torch.ones([1, 10], dtype=torch.long)


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        model_name = "facebook/opt-125m"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.llm_dataloader = LLMDataLoader()
        self.save_folder = "./saved"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.save_folder, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_remove_whole_block(self):
        layer_config = {
            "model.decoder.layers.0.self_attn.k_proj": {"bits": 32},
            "model.decoder.layers.0.self_attn.v_proj": {"bits": 32},
            "model.decoder.layers.0.self_attn.q_proj": {"bits": 32},
            "model.decoder.layers.0.self_attn.out_proj": {"bits": 32},
            "model.decoder.layers.0.fc1": {"bits": 32},
            "model.decoder.layers.0.fc2": {"bits": 32},
        }
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            layer_config=layer_config,
        )
        autoround.quantize()

    def test_consective_quant(self):
        bits, group_size, sym = 4, -1, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()

        model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()

    def test_mx_fp4(self):
        bits, group_size, sym = 4, 32, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            data_type="mx_fp4",
        )
        autoround.quantize()

    def test_nsample(self):
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=4,
            group_size=128,
            seqlen=2,
            nsamples=3,
            batch_size=3,
            iters=2,
            dataset=self.llm_dataloader,
            gradient_accumulate_steps=4,
        )
        autoround.quantize()

    def test_default(self):
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()

        autoround.save_quantized(output_dir="./saved", inplace=False, format="itrex")
        try:
            import auto_gptq
        except:
            return
        if torch.cuda.is_available():
            autoround.save_quantized(output_dir="./saved", inplace=False)

    def test_sym(self):
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()

    def test_w4g1(self):
        bits, group_size, sym = 4, -1, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()

    def test_w3g128(self):
        bits, group_size, sym = 3, 128, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()

    def test_w2g128(self):
        bits, group_size, sym = 2, 128, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()

    def test_disable_quanted_input(self):
        bits, group_size, sym = 4, -1, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            enable_quanted_input=False,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()

    def test_enable_norm_bias_tuning(self):
        bits, group_size, sym = 4, -1, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            enable_quanted_input=False,
            enable_norm_bias_tuning=True,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()

    def test_enable_norm_bias_tuning_qwen3(self):
        bits, group_size, sym = 4, 128, True
        model_name = "Qwen/Qwen3-0.6B"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            enable_norm_bias_tuning=True,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()

    def test_disable_minmax_tuning(self):
        bits, group_size, sym = 4, -1, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            enable_minmax_tuning=False,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()

    #
    def test_signround(self):
        bits, group_size, sym = 4, -1, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            enable_minmax_tuning=False,
            enable_quanted_input=False,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()

    def test_lm_head(self):
        bits, group_size, sym = 4, -1, False
        layer_config = {"lm_head": {"data_type": "int"}}
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            enable_minmax_tuning=False,
            enable_quanted_input=False,
            dataset=self.llm_dataloader,
            layer_config=layer_config,
        )
        autoround.quantize()

    def test_wa_quant(self):
        bits, group_size, sym, act_bits = 4, 128, False, 4
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            act_bits=4,
        )
        autoround.quantize()

    def test_auto_device_map(self):
        bits, group_size, sym = 4, 128, False
        model_name = "facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", trust_remote_code=True, device_map="auto"
        )
        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()

    def test_fp32(self):
        bits, group_size, sym = 4, 128, False
        model_name = "facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True, device_map="auto"
        )
        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            amp=False,
        )
        autoround.quantize()

    def test_tensor_reshape(self):
        model_name = "facebook/opt-125m"
        bits, group_size, sym = 4, 100, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()

    def test_rtn(self):
        model_name = "facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym, iters=1, nsamples=1)
        quantized_model_path = self.save_folder
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
        model = AutoModelForCausalLM.from_pretrained(
            self.save_folder,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
        model_infer(model, tokenizer)
        shutil.rmtree(self.save_folder)

    def test_fallback_layers(self):
        bits, group_size, sym = 4, 128, True
        model_name = "facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True, device_map="auto"
        )
        layer_config = {
            "model.decoder.layers.0.self_attn.q_proj": {"bits": 16},
            "model.decoder.layers.1.self_attn.k_proj": {"bits": 16},
        }
        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            layer_config=layer_config,
        )
        autoround.quantize()
        quantized_model_path = self.save_folder

        autoround.save_quantized(output_dir=quantized_model_path, format="auto_round", inplace=True)
        quantization_config = AutoRoundConfig(backend="ipex")

        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map="cpu", quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=1)[0])
        shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_not_convert_modules(self):
        import requests
        from PIL import Image
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        from auto_round_extension.ipex.qlinear_ipex_awq import QuantLinear

        model_name = "Qwen/Qwen2-VL-2B-Instruct-AWQ"
        quantization_config = AutoRoundConfig()
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, quantization_config=quantization_config, device_map="cpu", torch_dtype=torch.float16
        )
        self.assertTrue(isinstance(model.visual.blocks[0].attn.qkv, torch.nn.Linear))
        self.assertFalse(isinstance(model.visual.merger.mlp[0], QuantLinear))
        if hasattr(model.model, "language_model"):
            self.assertTrue(isinstance(model.model.language_model.layers[0].self_attn.v_proj, QuantLinear))
        else:
            self.assertTrue(isinstance(model.model.layers[0].self_attn.v_proj, QuantLinear))

        processor = AutoProcessor.from_pretrained(model_name, size=None)
        image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_url,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = Image.open(requests.get(image_url, stream=True).raw)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=1)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)

    def test_fallback_layers_regex_awq(self):
        model_name = "facebook/opt-125m"
        bits, group_size, sym = 4, 128, True
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        layer_config = {
            "model\.decoder\.layers\.(?:[0-9]|1[0-1])\.self_attn\.q_proj": {"bits": 16},
            "model.decoder.layers.1.self_attn.k_proj": {"bits": 16},
        }
        autoround = AutoRound(
            model,
            tokenizer=tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            layer_config=layer_config,
        )
        autoround.quantize()
        quantized_model_path = self.save_folder

        autoround.save_quantized(output_dir=quantized_model_path, format="auto_awq", inplace=True)
        quantization_config = AutoRoundConfig()

        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map="auto", quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=5)[0])
        print(res)
        shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_fallback_layers_regex_gptq(self):
        model_name = "facebook/opt-125m"
        bits, group_size, sym = 4, 128, True
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        layer_config = {
            "model\.decoder\.layers\.(?:[0-9]|1[0-1])\.self_attn\.q_proj": {"bits": 16},
            ##"model.decoder.layers.1.self_attn.k_proj": {"bits": 16}
        }
        autoround = AutoRound(
            model,
            tokenizer=tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            layer_config=layer_config,
        )
        autoround.quantize()
        quantized_model_path = self.save_folder

        autoround.save_quantized(output_dir=quantized_model_path, format="auto_gptq", inplace=True)
        quantization_config = AutoRoundConfig()

        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map="auto", quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=5)[0])
        print(res)
        shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_fallback_layers_regex_round(self):
        model_name = "facebook/opt-125m"
        bits, group_size, sym = 4, 128, True
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        layer_config = {
            "model\.decoder\.layers\.(?:[0-9]|1[0-1])\.self_attn\.q_proj": {"bits": 16},
            "model.decoder.layers.1.self_attn.k_proj": {"bits": 16},
        }
        autoround = AutoRound(
            model,
            tokenizer=tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            layer_config=layer_config,
        )
        autoround.quantize()
        quantized_model_path = self.save_folder

        autoround.save_quantized(output_dir=quantized_model_path, format="auto_round", inplace=True)
        quantization_config = AutoRoundConfig()

        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, device_map="auto", quantization_config=quantization_config
        )
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=5)[0])
        print(res)
        shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_fallback_layers_regex_exception(self):
        model_name = "facebook/opt-125m"
        bits, group_size, sym = 4, 128, True
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        layer_config = {"model.decoder.layers.12.self_attn.k_proj": {"bits": 16}}
        with self.assertRaises(ValueError):
            autoround = AutoRound(
                model,
                tokenizer=tokenizer,
                bits=bits,
                group_size=group_size,
                sym=sym,
                iters=2,
                seqlen=2,
                dataset=self.llm_dataloader,
                layer_config=layer_config,
            )
            autoround.quantize()


if __name__ == "__main__":
    unittest.main()
