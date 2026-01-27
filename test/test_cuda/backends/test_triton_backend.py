import shutil

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound
from auto_round.eval.evaluation import simple_evaluate_user_model
from auto_round.testing_utils import require_greater_than_050

from ...helpers import model_infer


class TestAutoRoundTritonBackend:
    @classmethod
    def setup_class(self):
        self.model_name = "/models/opt-125m"
        self.save_folder = "./saved"

    @classmethod
    def teardown_class(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    @require_greater_than_050
    def test_tritonv2_4bits_asym(self, dataloader):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        bits, group_size, sym = 4, 128, False
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=1,
            seqlen=2,
            dataset=dataloader,
        )
        quantized_model_path = self.save_folder
        _, quantized_model_path = autoround.quantize_and_save(
            output_dir=quantized_model_path, format="auto_round:gptqmodel"
        )
        quantized_model_path = quantized_model_path[0]

        quantization_config = AutoRoundConfig(backend="tritonv2")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
        print(result["results"]["lambada_openai"]["acc,none"])
        assert result["results"]["lambada_openai"]["acc,none"] > 0.34
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
        print(result["results"]["lambada_openai"]["acc,none"])
        assert result["results"]["lambada_openai"]["acc,none"] > 0.34
        torch.cuda.empty_cache()
        shutil.rmtree("./saved", ignore_errors=True)

    @require_greater_than_050
    def test_tritonv2_2bits_asym(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        bits, group_size, sym = 2, 32, False
        autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym)
        quantized_model_path = self.save_folder
        _, quantized_model_path = autoround.quantize_and_save(output_dir=quantized_model_path)
        quantized_model_path = quantized_model_path[0]

        quantization_config = AutoRoundConfig(backend="tritonv2")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
        print(result["results"]["lambada_openai"]["acc,none"])
        assert result["results"]["lambada_openai"]["acc,none"] > 0.19
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
        print(result["results"]["lambada_openai"]["acc,none"])
        assert result["results"]["lambada_openai"]["acc,none"] > 0.19
        torch.cuda.empty_cache()
        shutil.rmtree("./saved", ignore_errors=True)

    @require_greater_than_050
    def test_tritonv2_4bits_sym(self, dataloader):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        bits, group_size, sym = 4, 128, True
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            iters=1,
            seqlen=2,
            dataset=dataloader,
        )
        quantized_model_path = self.save_folder
        _, quantized_model_path = autoround.quantize_and_save(output_dir=quantized_model_path)
        quantized_model_path = quantized_model_path[0]

        quantization_config = AutoRoundConfig(backend="tritonv2")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
        # print(result['results']['lambada_openai']['acc,none'])
        assert result["results"]["lambada_openai"]["acc,none"] > 0.26
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
        # print(result['results']['lambada_openai']['acc,none'])
        assert result["results"]["lambada_openai"]["acc,none"] > 0.26
        torch.cuda.empty_cache()

        shutil.rmtree("./saved", ignore_errors=True)

    @require_greater_than_050
    def test_tritonv2_8bits_sym(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        bits, group_size, sym = 4, 256, True
        autoround = AutoRound(model, tokenizer, bits=bits, group_size=group_size, sym=sym, nsamples=1, iters=1)
        quantized_model_path = self.save_folder
        _, quantized_model_path = autoround.quantize_and_save(output_dir=quantized_model_path)
        quantized_model_path = quantized_model_path[0]

        quantization_config = AutoRoundConfig(backend="tritonv2")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
        print(result["results"]["lambada_openai"]["acc,none"])
        assert result["results"]["lambada_openai"]["acc,none"] > 0.27
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
        # print(result['results']['lambada_openai']['acc,none'])
        assert result["results"]["lambada_openai"]["acc,none"] > 0.27
        torch.cuda.empty_cache()
        shutil.rmtree("./saved", ignore_errors=True)

    @require_greater_than_050
    def test_tritonv2_2bits_sym(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        bits, group_size, sym = 2, 64, True
        autoround = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
        )
        quantized_model_path = self.save_folder
        _, quantized_model_path = autoround.quantize_and_save(output_dir=quantized_model_path)
        quantized_model_path = quantized_model_path[0]

        quantization_config = AutoRoundConfig(backend="tritonv2")
        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, torch_dtype=torch.float16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
        print(result["results"]["lambada_openai"]["acc,none"])
        assert result["results"]["lambada_openai"]["acc,none"] > 0.18
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            quantized_model_path, torch_dtype=torch.bfloat16, device_map="auto", quantization_config=quantization_config
        )

        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        model_infer(model, tokenizer)
        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="lambada_openai")
        # print(result['results']['lambada_openai']['acc,none'])
        assert result["results"]["lambada_openai"]["acc,none"] > 0.18
        torch.cuda.empty_cache()
        shutil.rmtree("./saved", ignore_errors=True)
