import shutil
from test.helpers import get_model_path, lamini_name_or_path

import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound


class TestQuantizationBlocks:
    @classmethod
    def setup_class(self):
        self.model_name = lamini_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

    @classmethod
    def teardown_class(self):
        shutil.rmtree("runs", ignore_errors=True)

    @pytest.fixture(autouse=True)
    def setup_save_dir(self, tmp_path):
        self.save_dir = str(tmp_path / "saved")

    def test_block_name_quant(self, dataloader):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        from auto_round.utils import get_block_names

        llm_block_names = get_block_names(self.model)
        bits, group_size, sym, batch_size = 4, 128, True, 20
        autoround = AutoRound(
            self.model,
            self.tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=2,
            seqlen=2,
            batch_size=batch_size,
            dataset=dataloader,
            to_quant_block_names=llm_block_names,
        )
        autoround.quantize()

        quantized_model_path = self.save_dir
        autoround.save_quantized(quantized_model_path, inplace=False, safe_serialization=False, format="auto_round")

        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        text = "There is a girl who likes adventure,"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        print(tokenizer.decode(model.generate(**inputs, max_new_tokens=50)[0]))
        quant_config = model.config.quantization_config
        assert quant_config.block_name_to_quantize is not None

    def test_mm_block_name(self, tiny_qwen_vl_model_path):
        from transformers import Qwen2VLForConditionalGeneration

        from auto_round.utils import get_block_names

        model_name = tiny_qwen_vl_model_path
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True, device_map="auto")
        block_name = get_block_names(model, quant_vision=True)
        assert len(block_name) == 2
        assert all(["visual.merger.mlp" not in n for n in block_name])
        block_name = get_block_names(model, quant_vision=False)
        assert len(block_name) == 1
        assert block_name == get_block_names(model)

    def test_moe(self):
        from auto_round.utils import get_block_names

        model_name = get_model_path("Qwen/Qwen1.5-MoE-A2.7B")
        # config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        block_name = get_block_names(model)
        block_name_2 = get_block_names(model, quant_vision=True)
        assert block_name == block_name_2
        assert len(block_name_2) == 1
        assert "model.layers.23" == block_name_2[0][-1]
