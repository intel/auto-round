import copy
import shutil
import sys
import unittest

sys.path.insert(0, "../..")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(3):
            yield torch.ones([1, 10], dtype=torch.long)


class TestAutoRoundAct(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_mx_fp4(self):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            act_bits=4,
            data_type="mx_fp",
        )
        autoround.quantize()

    def test_wint4fp8_dynamic(self):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        bits, group_size = 4, 128
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            act_bits=8,
            data_type="fp8",
            act_data_type="fp8",
        )
        autoround.quantize()

    def test_wint4fp8_static(self):
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            act_bits=8,
            data_type="fp8_to_int_sym",
            act_dynamic=False,
            act_data_type="fp8",
        )
        autoround.quantize()

    def test_wfp8afp8_static(self):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        from auto_round.wrapper import WrapperWALayer

        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        autoround = AutoRound(
            model,
            tokenizer,
            group_size=128,
            act_group_size=-1,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            data_type="fp8",
            act_dynamic=False,
            act_data_type="fp8",
        )
        autoround.quantize()

        self.assertTrue(isinstance(autoround.model.model.decoder.layers[2].self_attn.k_proj, WrapperWALayer))
        self.assertEqual(autoround.model.model.decoder.layers[2].self_attn.k_proj.orig_layer.act_scale.shape[0], 30)
        self.assertEqual(autoround.model.model.decoder.layers[2].self_attn.k_proj.orig_layer.act_max.shape[0], 30)

        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        autoround = AutoRound(
            model,
            tokenizer,
            group_size=128,
            act_group_size=128,
            iters=0,
            seqlen=2,
            dataset=self.llm_dataloader,
            data_type="fp8",
            act_dynamic=False,
            act_data_type="fp8",
        )
        autoround.quantize()
        self.assertTrue(isinstance(autoround.model.model.decoder.layers[2].self_attn.k_proj, WrapperWALayer))

        self.assertEqual(
            autoround.model.model.decoder.layers[2].self_attn.k_proj.orig_layer.act_scale.shape[0],
            int(3 * 10 * 768 / 128),
        )
        self.assertEqual(
            autoround.model.model.decoder.layers[2].self_attn.k_proj.orig_layer.act_max.shape[0],
            int(3 * 10 * 768 / 128),
        )

	
    def test_act_config_saving(self):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        scheme = "MXFP4"
        layer_config = {
            "lm_head": {"act_bits": 8, "bits": 8},
        }
        autoround = AutoRound(
            model=model_name,
            scheme=scheme,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            layer_config=layer_config,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="cpu")
        lmhead_config = model.config.quantization_config.extra_config["lm_head"]
        assert "act_data_type" in lmhead_config.keys() and lmhead_config["act_data_type"] == 'mx_fp_rceil'
        assert "act_bits" in lmhead_config.keys() and lmhead_config["act_bits"] == 8
        assert "act_group_size" in lmhead_config.keys() and lmhead_config["act_group_size"] == 32


if __name__ == "__main__":
    unittest.main()
