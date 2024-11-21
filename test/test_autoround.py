import copy
import shutil
import sys
import unittest

sys.path.insert(0, "..")
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

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

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_remove_whole_block(self):
        layer_config={"model.decoder.layers.0.self_attn.k_proj":{"bits":32},
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
            layer_config=layer_config
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
            data_type="mx_fp4"
        )
        autoround.quantize()

    def test_nsample(self):
       autoround= AutoRound(
           self.model,
           self.tokenizer,
           bits=4,
           group_size=128,
           seqlen=2,
           nsamples=3,
           batch_size=3,
           iters=2,
           dataset=self.llm_dataloader,
           gradient_accumulate_steps=4)
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
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True, device_map='auto')
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
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, trust_remote_code=True, device_map='auto')
        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            amp=False
        )
        autoround.quantize()


    def test_fallback_layers(self):
        bits, group_size, sym = 4, 128, True
        model_name = "facebook/opt-125m"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, trust_remote_code=True,
                                                     device_map='auto')
        layer_config = {"model.decoder.layers.0.self_attn.q_proj": {"bits": "16"},
                        "model.decoder.layers.1.self_attn.k_proj": {"bits": "16"}}
        autoround = AutoRound(
            model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
            layer_config=layer_config
        )
        autoround.quantize()
        quantized_model_path = "./saved"

        autoround.save_quantized(output_dir=quantized_model_path, format="auto_round", inplace=True)

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path,
                                                     device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        res = tokenizer.decode(model.generate(**inputs, max_new_tokens=1)[0])



if __name__ == "__main__":
    unittest.main()

