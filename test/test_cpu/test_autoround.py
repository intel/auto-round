import copy
import shutil
import sys
import unittest

from parameterized import parameterized

sys.path.insert(0, "../..")

import torch
from _test_helpers import model_infer
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound
from auto_round.eval.evaluation import simple_evaluate_user_model
from auto_round.utils import get_module


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(3):
            yield torch.ones([1, 10], dtype=torch.long)


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.llm_dataloader = LLMDataLoader()
        self.save_folder = "./saved"

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.save_folder, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_bits_setting(self):
        layer_config = {"model.decoder.layers.0.self_attn.k_proj": {"data_type": "mx_fp8", "group_size": 32}}
        autoround = AutoRound(
            "/tf_dataset/auto_round/models/facebook/opt-125m", iters=2, seqlen=2, nsamples=1, layer_config=layer_config
        )
        autoround.quantize()
        module = get_module(autoround.model, "model.decoder.layers.0.self_attn.k_proj")
        if module.bits != 8:
            raise ValueError(f"Expected bits to be 8, but got {module.bits}")

    def test_layer_config(self):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        layer_config = {"self_attn": {"bits": 4, "data_type": "nv_fp", "act_bits": 16, "group_size": 16}}
        autoround = AutoRound(
            model_name,
            self.tokenizer,
            scheme="NVFP4",
            iters=0,
            seqlen=2,
            dataset=self.llm_dataloader,
            layer_config=layer_config,
            amp=False,
        )
        autoround.quantize_and_save(self.save_folder, inplace=False, format="fake")
        shutil.rmtree(self.save_folder)

    def test_remove_whole_block(self):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
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
            model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            layer_config=layer_config,
        )
        autoround.quantize()

    def test_consecutive_quant(self):
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

        model = AutoModelForCausalLM.from_pretrained(
            "/tf_dataset/auto_round/models/microsoft/phi-2", torch_dtype="auto", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "/tf_dataset/auto_round/models/microsoft/phi-2", trust_remote_code=True
        )
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
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        bits, group_size, sym = 4, 32, False
        autoround = AutoRound(
            model_name,
            bits=bits,
            act_bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            nsamples=2,
            seqlen=128,
            data_type="mx_fp4",
            act_data_type="mx_fp_rceil",
        )
        model, _ = autoround.quantize()
        result = simple_evaluate_user_model(
            model, self.tokenizer, batch_size="auto:8", tasks="lambada_openai", limit=32
        )
        print(result["results"]["lambada_openai"]["acc,none"])
        self.assertGreater(result["results"]["lambada_openai"]["acc,none"], 0.3)  # 0.375

    def test_nv_fp4(self):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        bits, group_size, sym = 4, 16, False
        autoround = AutoRound(
            model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            data_type="nv_fp4",
        )
        model, _ = autoround.quantize()
        result = simple_evaluate_user_model(
            model, self.tokenizer, batch_size="auto:8", tasks="lambada_openai", limit=32
        )
        print(result["results"]["lambada_openai"]["acc,none"])
        self.assertGreater(result["results"]["lambada_openai"]["acc,none"], 0.35)

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

    def test_w4g1(self):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        bits, group_size, sym = 4, -1, True
        autoround = AutoRound(
            model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()

    @parameterized.expand([(2,), (3,), (4,)])
    def test_g128(self, bits):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        group_size, sym = 128, True
        autoround = AutoRound(
            model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            dataset=self.llm_dataloader,
        )
        model, _ = autoround.quantize()
        if bits > 2:
            result = simple_evaluate_user_model(
                model, self.tokenizer, batch_size="auto:8", tasks="lambada_openai", limit=32
            )
            print(result["results"]["lambada_openai"]["acc,none"])
            self.assertGreater(result["results"]["lambada_openai"]["acc,none"], 0.3)

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

    def test_enable_norm_bias_tuning_qwen3(self):
        bits, group_size, sym = 4, 128, True
        model_name = "/tf_dataset/auto_round/models/Qwen/Qwen3-0.6B"
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
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        bits, group_size, sym = 4, -1, False
        autoround = AutoRound(
            model_name,
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

    def test_lm_head_layer_config_way(self):
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
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        bits, group_size, sym, act_bits = 4, 128, False, 4
        autoround = AutoRound(
            model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            act_bits=act_bits,
        )
        autoround.quantize()

    def test_auto_device_map(self):
        bits, group_size, sym = 4, 128, False
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
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

    def test_device_map_dict(self):
        bits, group_size, sym = 4, 128, False
        device_map = {".*": "cpu"}
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            device_map=device_map,
        )
        autoround.quantize()

        # test model_name
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        autoround = AutoRound(
            model_name,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            device_map=device_map,
        )
        autoround.quantize()

    def test_fp32(self):
        bits, group_size, sym = 4, 128, False
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
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
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym, iters=0, nsamples=1)
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

    def test_embed_quant(self):
        bits, group_size, sym = 4, 128, True
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        layer_config = {
            "model.decoder.embed_tokens": {"bits": 4},
        }
        autoround = AutoRound(
            model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            nsamples=3,
            dataset=self.llm_dataloader,
            layer_config=layer_config,
        )
        autoround.quantize()

    def test_fallback_layers(self):
        bits, group_size, sym = 4, 128, True
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True, device_map="auto"
        )
        layer_config = {
            "model.decoder.layers.0.self_attn.q_proj": {"bits": 16},
            "model.decoder.layers.1.self_attn.k_proj": {"bits": 16},
            "model.decoder.embed_tokens": {"bits": 16},
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

        model_name = "/tf_dataset/auto_round/models/Qwen/Qwen2-VL-2B-Instruct-AWQ"
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
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        bits, group_size, sym = 4, 128, True
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        layer_config = {
            r"model\.decoder\.layers\.(?:[0-9]|1[0-1])\.self_attn\.q_proj": {"bits": 16},
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
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        bits, group_size, sym = 4, 128, True
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        layer_config = {
            r"model\.decoder\.layers\.(?:[0-9]|1[0-1])\.self_attn\.q_proj": {"bits": 16},
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
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        bits, group_size, sym = 4, 128, True
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        layer_config = {
            r"model\.decoder\.layers\.(?:[0-9]|1[0-1])\.self_attn\.q_proj": {"bits": 16},
            r"model.decoder.layers.1.self_attn.k_proj": {"bits": 16},
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
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
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

    # def test_fp8_model_input_rtn_generation(self):
    #     model_name = "Qwen/Qwen3-0.6B-FP8"
    #     ar = AutoRound(model=model_name, iters=0)
    #     ar.quantize_and_save(output_dir=self.save_folder)
    #     model = AutoModelForCausalLM.from_pretrained(self.save_folder, torch_dtype="auto", trust_remote_code=True)
    #     tokenizer = AutoTokenizer.from_pretrained(self.save_folder)
    #     text = "There is a girl who likes adventure,"
    #     inputs = tokenizer(text, return_tensors="pt").to(model.device)
    #     print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))

    def test_dequant_fp8_weight(self):
        from auto_round.utils import dequant_block_fp8_weight

        # test pad and unpad
        weight = torch.randn(587, 7168)
        weight_scale = torch.randn(5, 56)
        block_size = [128, 128]
        dequant_weight = dequant_block_fp8_weight(weight, weight_scale, block_size)
        self.assertEqual(dequant_weight.shape.numel(), 4207616)

        # test experts are stacked.
        weight = torch.randn([32, 5760, 1440])
        weight_scale = torch.randn([32, 5760, 90])
        block_size = [1, 16]
        dequant_weight = dequant_block_fp8_weight(weight, weight_scale, block_size)
        self.assertEqual(len(dequant_weight.shape), 3)
        self.assertEqual(dequant_weight.shape[0], 32)
        self.assertEqual(dequant_weight.shape.numel(), 32 * 5760 * 1440)

    def test_mixed_bit_setting(self):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        layer_config = {"model.decoder.layers.7.fc1": {"bits": 8, "act_bits": 8}}
        ar = AutoRound(model_name, data_type="mx_fp4", act_bits=4, iters=0, layer_config=layer_config)
        ar.quantize()
        layer_config = ar.layer_config
        if (
            layer_config["model.decoder.layers.7.fc1"]["bits"] != 8
            or layer_config["model.decoder.layers.7.fc1"]["act_bits"] != 8
        ):
            raise ValueError("mixed bits is not correct")

    def test_alg_ext(self):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        ar = AutoRound(model_name, scheme="W2A16", iters=1, nsamples=1, enable_alg_ext=True)
        ar.quantize()

    def test_alg_ext_import(self):
        from auto_round.alg_ext import quantize_block_ext

    def test_invalid_layer_config(self):
        with self.assertRaises(ValueError):
            layer_config = {"model.decoder.layers.2.self_attnx": {"bits": 2}}
            ar = AutoRound(
                "/tf_dataset/auto_round/models/facebook/opt-125m",
                scheme="W3A16",
                nsamples=1,
                iters=1,
                layer_config=layer_config,
            )
            ar.quantize()
        with self.assertRaises(ValueError):
            layer_config = {"model.decoder.layers.2.self_attn": {"bit": 2}}  # should be bits
            ar = AutoRound(
                "/tf_dataset/auto_round/models/facebook/opt-125m",
                scheme="W3A16",
                nsamples=1,
                iters=1,
                layer_config=layer_config,
            )
            ar.quantize()

    def test_quant_lm_head(self):
        model_name = "/tf_dataset/auto_round/models/Qwen/Qwen3-8B"
        ar = AutoRound(model_name, quant_lm_head=True, iters=0, disable_opt_rtn=True)
        ar.quantize_and_save(output_dir=self.save_folder, format="auto_round")
        model = AutoModelForCausalLM.from_pretrained(self.save_folder, device_map="cpu")
        assert "lm_head" in model.config.quantization_config.extra_config
        assert model.config.quantization_config.extra_config["lm_head"]["bits"] == 4

    def test_quant_lm_head_layer_config(self):
        model_name = "/tf_dataset/auto_round/models/Qwen/Qwen3-8B"
        layer_config = {"lm_head": {"bits": 4}}
        ar = AutoRound(model_name, quant_lm_head=True, iters=0, disable_opt_rtn=True, layer_config=layer_config)
        ar.quantize_and_save(output_dir=self.save_folder, format="auto_round")
        model = AutoModelForCausalLM.from_pretrained(self.save_folder, device_map="cpu")
        assert "lm_head" in model.config.quantization_config.extra_config
        assert model.config.quantization_config.extra_config["lm_head"]["bits"] == 4

    def test_compressor(self):
        model_name = "Qwen/Qwen2-VL-2B-Instruct"
        ar = AutoRound(model_name, enable_adam=True)
        self.assertEqual(ar.optimizer, torch.optim.AdamW)
        self.assertTrue(ar.mllm)

        # test old api
        from auto_round import AutoRoundMLLM

        ar = AutoRoundMLLM(model_name)
        self.assertTrue(ar.mllm)

    def test_attention_mask_in_dataset(self):
        from transformers import AutoTokenizer

        model_name = "/tf_dataset/auto_round/models/Qwen/Qwen3-0.6B"
        # model_name = "/models/Qwen3-0.6B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        text = ["haha", "hello world"]
        res = tokenizer(text, return_tensors="pt", max_length=8, padding="max_length", truncation=True)
        data = [res.data]

        text = ["qudd", "hfd"]
        res = tokenizer(text, return_tensors="pt", max_length=8, padding="max_length", truncation=True)
        data.append(res.data)
        from auto_round import AutoRound

        ar = AutoRound(model_name, iters=1, dataset=data, seqlen=8)
        ar.quantize()

    def test_attention_mask_via_tokenize_in_dataset(self):
        from transformers import AutoTokenizer

        model_name = "/tf_dataset/auto_round/models/Qwen/Qwen3-0.6B"
        # model_name = "/models/Qwen3-0.6B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        text = ["haha", "hello world"]
        res = tokenizer(text, return_tensors="pt", max_length=8, padding="max_length", truncation=True)
        res.data.pop("attention_mask")
        data = [res.data]

        text = ["qudd", "hfd"]
        res = tokenizer(text, return_tensors="pt", max_length=8, padding="max_length", truncation=True)
        res.data.pop("attention_mask")
        data.append(res.data)
        from auto_round import AutoRound

        ar = AutoRound(model_name, iters=1, dataset=data, seqlen=8)
        ar.quantize()

    def test_create_adam(self):
        model_name = "/tf_dataset/auto_round/models/Qwen/Qwen3-0.6B"
        from auto_round import AutoRound

        ar = AutoRound(model=model_name, enable_adam=True)


if __name__ == "__main__":
    unittest.main()
