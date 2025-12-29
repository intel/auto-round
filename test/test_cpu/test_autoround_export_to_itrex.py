import copy
import shutil

import pytest
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound

from ..helpers import get_model_path, gptj_name_or_path


class SimpleDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.randn([1, 30])


class TestAutoroundExport:
    approach = "weight_only"

    @classmethod
    def setup_class(self):
        self.gptj = transformers.AutoModelForCausalLM.from_pretrained(
            gptj_name_or_path,
            torchscript=True,
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(gptj_name_or_path, trust_remote_code=True)
        self.gptj_no_jit = transformers.AutoModelForCausalLM.from_pretrained(
            gptj_name_or_path,
        )
        self.lm_input = torch.ones([1, 10], dtype=torch.long)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_autoround_int_quant(self):
        model = copy.deepcopy(self.gptj)
        out1 = model(self.lm_input)
        round = AutoRound
        optq_1 = round(model, self.tokenizer, nsamples=1, amp=False, seqlen=10, iters=10, enable_torch_compile=False)
        q_model, layer_config1 = optq_1.quantize()  ##compile model
        from auto_round.export.export_to_itrex import pack_model

        compressed_model = pack_model(model=q_model, layer_config=layer_config1)
        out2 = model(self.lm_input)
        out3 = q_model(self.lm_input)
        out4 = compressed_model(self.lm_input)
        assert torch.all(torch.isclose(out1[0], out2[0], atol=1e-1))
        assert not torch.all(out1[0] == out2[0])
        assert torch.all(out2[0] == out3[0])
        assert torch.all(torch.isclose(out3[0], out4[0], atol=1e-3))
        assert "transformer.h.0.attn.k_proj.qzeros" in compressed_model.state_dict().keys()

        model = copy.deepcopy(self.gptj)
        out6 = model(self.lm_input)
        optq_2 = round(model, self.tokenizer, device="cpu", nsamples=1, seqlen=10)
        q_model, layer_config2 = optq_2.quantize()
        compressed_model = pack_model(model=q_model, layer_config=layer_config2, inplace=False)
        compressed_model = compressed_model.to(torch.float32)
        out4 = q_model(self.lm_input)
        out5 = compressed_model(self.lm_input)
        assert torch.all(out1[0] == out6[0])
        assert torch.all(torch.isclose(out4[0], out5[0], atol=5e-3))

    def test_config(self):
        from auto_round.export.export_to_itrex import QuantConfig

        config = QuantConfig.from_pretrained(get_model_path("TheBloke/Llama-2-7B-Chat-GPTQ"))
        config.save_pretrained("quantization_config_dir")
        loaded_config = QuantConfig.from_pretrained("quantization_config_dir")
        assert config.group_size == loaded_config.group_size
        assert config.desc_act == loaded_config.desc_act
        assert config.bits == loaded_config.bits
        assert config.sym == loaded_config.sym

    def test_xpu_export(self):
        model = copy.deepcopy(self.gptj)
        out1 = model(self.lm_input)
        round = AutoRound
        optq_1 = round(model, self.tokenizer, nsamples=1, amp=False, seqlen=10, iters=10, enable_torch_compile=False)
        q_model, layer_config1 = optq_1.quantize()
        from auto_round.export.export_to_itrex import pack_model

        compressed_model_xpu = pack_model(model=q_model, layer_config=layer_config1, device="xpu", inplace=False)
        compressed_model_cpu = pack_model(model=q_model, layer_config=layer_config1, inplace=False)
        out2 = model(self.lm_input)
        out3 = q_model(self.lm_input)
        out4 = compressed_model_xpu(self.lm_input)
        out5 = compressed_model_cpu(self.lm_input)
        assert torch.all(torch.isclose(out1[0], out2[0], atol=1e-1))
        assert not torch.all(out1[0] == out2[0])
        assert torch.all(out2[0] == out3[0])
        assert torch.all(torch.isclose(out3[0], out4[0], atol=1e-3))
        assert torch.all(torch.isclose(out4[0], out5[0], atol=1e-5))
