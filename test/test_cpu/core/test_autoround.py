import copy
import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound
from auto_round.eval.evaluation import simple_evaluate_user_model
from auto_round.utils import get_module

from ...helpers import get_model_path, model_infer, opt_name_or_path, qwen_name_or_path


class TestAutoRound:
    @classmethod
    def setup_class(self):
        model_name = opt_name_or_path
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.save_folder = "./saved"

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.save_folder, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_bits_setting(self, tiny_opt_model_path):
        layer_config = {"model.decoder.layers.0.self_attn.k_proj": {"data_type": "mx_fp8", "group_size": 32}}
        autoround = AutoRound(tiny_opt_model_path, iters=2, seqlen=2, nsamples=1, layer_config=layer_config)
        autoround.quantize()
        module = get_module(autoround.model, "model.decoder.layers.0.self_attn.k_proj")
        if module.bits != 8:
            raise ValueError(f"Expected bits to be 8, but got {module.bits}")

    def test_layer_config(self, tiny_opt_model_path, dataloader):
        model_name = tiny_opt_model_path
        layer_config = {"self_attn": {"bits": 4, "data_type": "nv_fp", "act_bits": 16, "group_size": 16}}
        autoround = AutoRound(
            model_name,
            self.tokenizer,
            scheme="NVFP4",
            iters=0,
            seqlen=2,
            dataset=dataloader,
            layer_config=layer_config,
            amp=False,
        )
        autoround.quantize_and_save(self.save_folder, inplace=False, format="fake")
        shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_remove_whole_block(self, tiny_opt_model_path, dataloader):
        model_name = tiny_opt_model_path
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
            dataset=dataloader,
            layer_config=layer_config,
        )
        autoround.quantize()

    def test_consecutive_quant(self, tiny_opt_model_path, tiny_phi2_model_path, dataloader):
        bits, group_size, sym = 4, -1, False
        autoround = AutoRound(
            tiny_opt_model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
        )
        autoround.quantize()

        autoround = AutoRound(
            tiny_phi2_model_path,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
        )
        autoround.quantize()

    def test_mx_fp4(self, dataloader):
        model_name = opt_name_or_path
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
        assert result["results"]["lambada_openai"]["acc,none"] > 0.3  # 0.375

    def test_nv_fp4(self, dataloader):
        model_name = opt_name_or_path
        bits, group_size, sym = 4, 16, False
        autoround = AutoRound(
            model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
            data_type="nv_fp4",
        )
        model, _ = autoround.quantize()
        result = simple_evaluate_user_model(
            model, self.tokenizer, batch_size="auto:8", tasks="lambada_openai", limit=32
        )
        print(result["results"]["lambada_openai"]["acc,none"])
        assert result["results"]["lambada_openai"]["acc,none"] > 0.35

    def test_w4g1(self, tiny_opt_model_path, dataloader):
        model_name = tiny_opt_model_path
        bits, group_size, sym = 4, -1, True
        autoround = AutoRound(
            model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            dataset=dataloader,
        )
        autoround.quantize()

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_g128(self, bits, dataloader):
        model_name = opt_name_or_path
        group_size, sym = 128, True
        autoround = AutoRound(
            model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=10,
            dataset=dataloader,
        )
        model, _ = autoround.quantize()
        if bits > 2:
            result = simple_evaluate_user_model(
                model, self.tokenizer, batch_size="auto:8", tasks="lambada_openai", limit=32
            )
            print(result["results"]["lambada_openai"]["acc,none"])
            assert result["results"]["lambada_openai"]["acc,none"] > 0.3

    def test_disable_quanted_input(self, dataloader):
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
            dataset=dataloader,
        )
        autoround.quantize()

    def test_enable_norm_bias_tuning_qwen3(self, tiny_qwen_model_path, dataloader):
        bits, group_size, sym = 4, 128, True
        model_name = tiny_qwen_model_path
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
            dataset=dataloader,
        )
        autoround.quantize()

    def test_enable_norm_bias_tuning(self, dataloader):
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
            dataset=dataloader,
        )
        autoround.quantize()

    def test_disable_minmax_tuning(self, dataloader):
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
            dataset=dataloader,
        )
        autoround.quantize()

    #
    def test_signround(self, tiny_opt_model_path, dataloader):
        model_name = tiny_opt_model_path
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
            dataset=dataloader,
        )
        autoround.quantize()

    def test_lm_head_layer_config_way(self, dataloader):
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
            dataset=dataloader,
            layer_config=layer_config,
        )
        autoround.quantize()

    def test_wa_quant(self, tiny_opt_model_path, dataloader):
        model_name = tiny_opt_model_path
        bits, group_size, sym, act_bits = 4, 128, False, 4
        autoround = AutoRound(
            model_name,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
            act_bits=act_bits,
        )
        autoround.quantize()

    def test_auto_device_map(self, tiny_opt_model_path, dataloader):
        bits, group_size, sym = 4, 128, False
        model_name = tiny_opt_model_path
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
            dataset=dataloader,
        )
        autoround.quantize()

    def test_device_map_dict(self, tiny_opt_model_path, dataloader):
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
            dataset=dataloader,
            device_map=device_map,
        )
        autoround.quantize()

        # test model_name
        model_name = tiny_opt_model_path
        autoround = AutoRound(
            model_name,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
            device_map=device_map,
        )
        autoround.quantize()

    def test_fp32(self, tiny_opt_model_path, dataloader):
        bits, group_size, sym = 4, 128, False
        model_name = tiny_opt_model_path
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
            dataset=dataloader,
            amp=False,
        )
        autoround.quantize()

    def test_tensor_reshape(self, dataloader):
        bits, group_size, sym = 4, 100, False
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=dataloader,
        )
        autoround.quantize()

    def test_rtn(self, tiny_opt_model_path):
        model_name = tiny_opt_model_path
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
        shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_embed_quant(self, tiny_opt_model_path, dataloader):
        bits, group_size, sym = 4, 128, True
        model_name = tiny_opt_model_path
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
            dataset=dataloader,
            layer_config=layer_config,
        )
        autoround.quantize()

    def test_fallback_layers(self, tiny_opt_model_path, dataloader):
        bits, group_size, sym = 4, 128, True
        model_name = tiny_opt_model_path
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
            dataset=dataloader,
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

        model_name = get_model_path("Qwen/Qwen2-VL-2B-Instruct-AWQ")
        quantization_config = AutoRoundConfig()
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, quantization_config=quantization_config, device_map="cpu", torch_dtype=torch.float16
        )
        assert isinstance(model.visual.blocks[0].attn.qkv, torch.nn.Linear)
        assert not isinstance(model.visual.merger.mlp[0], QuantLinear)
        if hasattr(model.model, "language_model"):
            assert isinstance(model.model.language_model.layers[0].self_attn.v_proj, QuantLinear)
        else:
            assert isinstance(model.model.layers[0].self_attn.v_proj, QuantLinear)

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

    def test_fallback_layers_regex_awq(self, tiny_opt_model_path, dataloader):
        model_name = tiny_opt_model_path
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
            dataset=dataloader,
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

    def test_fallback_layers_regex_gptq(self, tiny_opt_model_path, dataloader):
        model_name = tiny_opt_model_path
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
            dataset=dataloader,
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

    def test_fallback_layers_regex_round(self, tiny_opt_model_path, dataloader):
        model_name = tiny_opt_model_path
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
            dataset=dataloader,
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

    def test_fallback_layers_regex_exception(self, tiny_opt_model_path, dataloader):
        model_name = tiny_opt_model_path
        bits, group_size, sym = 4, 128, True
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        layer_config = {"model.decoder.layers.12.self_attn.k_proj": {"bits": 16}}
        with pytest.raises(ValueError):
            autoround = AutoRound(
                model,
                tokenizer=tokenizer,
                bits=bits,
                group_size=group_size,
                sym=sym,
                iters=2,
                seqlen=2,
                dataset=dataloader,
                layer_config=layer_config,
            )
            autoround.quantize()

    def test_dequant_fp8_weight(self):
        from auto_round.utils import dequant_block_fp8_weight

        # test pad and unpad
        weight = torch.randn(587, 7168)
        weight_scale = torch.randn(5, 56)
        block_size = [128, 128]
        dequant_weight = dequant_block_fp8_weight(weight, weight_scale, block_size)
        assert dequant_weight.shape.numel() == 4207616

        # test experts are stacked.
        weight = torch.randn([32, 5760, 1440])
        weight_scale = torch.randn([32, 5760, 90])
        block_size = [1, 16]
        dequant_weight = dequant_block_fp8_weight(weight, weight_scale, block_size)
        assert len(dequant_weight.shape) == 3
        assert dequant_weight.shape[0] == 32
        assert dequant_weight.shape.numel() == 32 * 5760 * 1440

    def test_mixed_bit_setting(self, tiny_opt_model_path):
        model_name = tiny_opt_model_path
        layer_config = {"model.decoder.layers.1.fc1": {"bits": 8, "act_bits": 8}}
        ar = AutoRound(model_name, data_type="mx_fp4", act_bits=4, iters=0, layer_config=layer_config)
        ar.quantize()
        layer_config = ar.layer_config
        if (
            layer_config["model.decoder.layers.1.fc1"]["bits"] != 8
            or layer_config["model.decoder.layers.1.fc1"]["act_bits"] != 8
        ):
            raise ValueError("mixed bits is not correct")

    def test_invalid_layer_config(self, tiny_opt_model_path):
        with pytest.raises(ValueError):
            layer_config = {"model.decoder.layers.2.self_attnx": {"bits": 2}}
            ar = AutoRound(
                tiny_opt_model_path,
                scheme="W3A16",
                nsamples=1,
                iters=1,
                layer_config=layer_config,
            )
            ar.quantize()
        with pytest.raises(ValueError):
            layer_config = {"model.decoder.layers.2.self_attn": {"bit": 2}}  # should be bits
            ar = AutoRound(
                tiny_opt_model_path,
                scheme="W3A16",
                nsamples=1,
                iters=1,
                layer_config=layer_config,
            )
            ar.quantize()

    def test_quant_lm_head(self, tiny_untied_qwen_model_path):
        model_name = tiny_untied_qwen_model_path
        ar = AutoRound(model_name, quant_lm_head=True, iters=0, seqlen=8, nsamples=1, disable_opt_rtn=True)
        ar.quantize_and_save(output_dir=self.save_folder, format="auto_round")
        model = AutoModelForCausalLM.from_pretrained(self.save_folder, device_map="cpu")
        assert "lm_head" in model.config.quantization_config.extra_config
        assert model.config.quantization_config.extra_config["lm_head"]["bits"] == 4

        layer_config = {"lm_head": {"bits": 4}}
        ar = AutoRound(
            model_name,
            quant_lm_head=False,
            iters=0,
            seqlen=8,
            nsamples=1,
            disable_opt_rtn=True,
            layer_config=layer_config,
        )
        ar.quantize_and_save(output_dir=self.save_folder, format="auto_round")
        model = AutoModelForCausalLM.from_pretrained(self.save_folder, device_map="cpu")
        assert "lm_head" in model.config.quantization_config.extra_config
        assert model.config.quantization_config.extra_config["lm_head"]["bits"] == 4

    def test_quant_lm_head_layer_config(self, tiny_untied_qwen_model_path):
        model_name = tiny_untied_qwen_model_path
        layer_config = {"lm_head": {"bits": 4}}
        ar = AutoRound(
            model_name,
            quant_lm_head=True,
            iters=0,
            seqlen=8,
            nsamples=1,
            disable_opt_rtn=True,
            layer_config=layer_config,
        )
        ar.quantize_and_save(output_dir=self.save_folder, format="auto_round")
        model = AutoModelForCausalLM.from_pretrained(self.save_folder, device_map="cpu")
        assert "lm_head" in model.config.quantization_config.extra_config
        assert model.config.quantization_config.extra_config["lm_head"]["bits"] == 4

    def test_compressor(self, tiny_qwen_vl_model_path):
        model_name = tiny_qwen_vl_model_path
        ar = AutoRound(model_name, enable_adam=True)
        assert ar.optimizer == torch.optim.AdamW
        assert ar.mllm

        # test old api
        from auto_round import AutoRoundMLLM

        ar = AutoRoundMLLM(model_name)
        assert ar.mllm

    def test_attention_mask_in_dataset(self):
        from transformers import AutoTokenizer

        model_name = qwen_name_or_path
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

        model_name = qwen_name_or_path
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

    def test_low_cpu_mem_usage(self, tiny_opt_model_path, dataloader):
        bits, group_size = 4, 32
        model_name = tiny_opt_model_path
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        quantized_model_path = self.save_folder
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            iters=2,
            seqlen=10,
            dataset=dataloader,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_create_adam(self):
        model_name = qwen_name_or_path
        from auto_round import AutoRound

        ar = AutoRound(model=model_name, enable_adam=True)
