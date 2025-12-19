import copy
import shutil
import sys
import unittest

sys.path.insert(0, "../..")
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from auto_round import AutoRound
from auto_round.testing_utils import require_awq, require_optimum


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


class TestAutoRound(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.model_name = "facebook/opt-125m"
        self.save_dir = "./saved"
        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_fp8input_mxfp4_llmcompressor_format(self):
        model_name = "/models/Qwen3-0.6B-FP8"
        scheme = "mxfp4"
        ar = AutoRound(
            model=model_name,
            iters=2,
            seqlen=2,
            scheme=scheme,
            dataset=self.llm_dataloader,
        )
        compressed_model, _ = ar.quantize_and_save(output_dir=self.save_dir, format="llm_compressor")
        tmp_layer = compressed_model.model.layers[3].self_attn.q_proj
        assert (
            hasattr(tmp_layer, "weight_scale")
            and hasattr(tmp_layer, "weight_packed")
            and tmp_layer.weight_scale.dtype is torch.uint8
            and tmp_layer.weight_scale.shape[0] == 2048
        ), "Illegal MXFP4 packing name or data_type or shape"
        quantization_config = AutoConfig.from_pretrained(self.save_dir, trust_remote_code=True).quantization_config
        assert (
            quantization_config["format"] == "float-quantized"
            and quantization_config["config_groups"]["group_0"]["weights"]["is_mx"] is True
            and quantization_config["config_groups"]["group_0"]["weights"]["num_bits"] == 4
        ), f"Invalid MXFP4 quantization configuration: {quantization_config}"
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_nvfp4_llmcompressor_format(self):
        scheme = "nvfp4"
        autoround = AutoRound(
            self.model_name,
            scheme=scheme,
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        quantized_model_path = self.save_dir
        compressed_model, _ = autoround.quantize_and_save(output_dir=quantized_model_path, format="llm_compressor")
        tmp_layer = compressed_model.model.decoder.layers[3].self_attn.q_proj
        assert (
            hasattr(tmp_layer, "weight_scale")
            and hasattr(tmp_layer, "weight_global_scale")
            and hasattr(tmp_layer, "input_global_scale")
            and tmp_layer.weight_packed.dtype is torch.uint8
            and tmp_layer.weight_scale.dtype is torch.float8_e4m3fn
            and tmp_layer.weight_scale.shape[0] == 768
        ), "Illegal NVFP4 packing name or data_type or shape"
        quantization_config = AutoConfig.from_pretrained(
            quantized_model_path, trust_remote_code=True
        ).quantization_config
        assert (
            quantization_config["format"] == "nvfp4-pack-quantized"
            and quantization_config["config_groups"]["group_0"]["input_activations"]["num_bits"] == 4
        ), f"Invalid NVFP4 quantization configuration: {quantization_config}"
        shutil.rmtree("./saved", ignore_errors=True)
        # from vllm import LLM, SamplingParams
        # prompts = [
        #     "The capital of France is",
        #     "The future of AI is",
        # ]
        ## Create a sampling params object.
        # sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=20)
        # QUANTIZATION = "compressed-tensors"
        # llm = LLM(model=quantized_model_path,
        #         #   quantization=QUANTIZATION,
        #           trust_remote_code=True,
        #           tensor_parallel_size=1,
        #           enforce_eager=True,
        #           gpu_memory_utilization=0.7,
        # )
        # outputs = llm.generate(prompts, sampling_params)
        # # Print the outputs.
        # for output in outputs:
        #     prompt = output.prompt
        #     generated_text = output.outputs[0].text
        #     if "France" in prompt:
        #         assert "Paris" in generated_text

    def test_nvfp4_moe_actmax_rtn(self):
        model_name = "/data0/deepseek-ai/DeepSeek-V2-Lite"
        scheme = "nvfp4"
        autoround = AutoRound(
            model_name,
            scheme=scheme,
            iters=0,
            seqlen=2,
            nsamples=2,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()
        quantized_model_path = self.save_dir
        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_round")

    def test_nvfp4_moe_actmax_ar(self):
        model_name = "/data0/deepseek-ai/DeepSeek-V2-Lite"
        scheme = "nvfp4"
        autoround = AutoRound(
            model_name,
            scheme=scheme,
            iters=1,
            seqlen=2,
            nsamples=2,
            dataset=self.llm_dataloader,
        )
        autoround.quantize()
        quantized_model_path = self.save_dir
        autoround.save_quantized(output_dir=quantized_model_path, inplace=False, format="auto_round")

    def test_qwen_moe_quant_infer(self):
        model_name = "/models/Qwen1.5-MoE-A2.7B"
        layer_config = {
            "layers\.(?:[3-9]|1[0-9]|2[0-3])": {"bits": 16, "act_bits": 16},
        }
        scheme = "nvfp4"
        autoround = AutoRound(
            model_name,
            scheme=scheme,
            iters=1,
            seqlen=3,
            nsamples=2,
            dataset=self.llm_dataloader,
            layer_config=layer_config,
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, inplace=True, format="auto_round")
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, torch_dtype="auto", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        from auto_round.eval.evaluation import simple_evaluate_user_model

        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="piqa")
        print(result["results"]["piqa"]["acc,none"])
        self.assertGreater(result["results"]["piqa"]["acc,none"], 0.7)
        shutil.rmtree(quantized_model_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
