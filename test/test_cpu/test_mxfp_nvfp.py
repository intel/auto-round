import os
import shutil
import sys
import unittest

from parameterized import parameterized

sys.path.insert(0, "../..")
import torch
from transformers import AutoModelForCausalLM, AutoRoundConfig, AutoTokenizer

from auto_round import AutoRound


def _get_folder_size(path: str) -> float:
    """Return folder size in GB."""
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024**3)  # convert to GB


class LLMDataLoader:
    def __init__(self):
        self.batch_size = 1

    def __iter__(self):
        for i in range(2):
            yield torch.ones([1, 10], dtype=torch.long)


class TestAutoRoundFP(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        model_name = "facebook/opt-125m"  # /tf_dataset/auto_round/models/
        self.save_dir = "./saved"
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.llm_dataloader = LLMDataLoader()

    @classmethod
    def tearDownClass(self):
        shutil.rmtree("./saved", ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_nvfp4_moe_actmax_rtn(self):
        model_name = "/tf_dataset/auto_round/models/deepseek-ai/DeepSeek-V2-Lite"
        layer_config = {
            "self_attn": {"bits": 16, "act_bits": 16},
            "mlp.shared_experts": {"bits": 16, "act_bits": 16},
        }
        scheme = "nvfp4"
        autoround = AutoRound(
            model_name,
            scheme=scheme,
            iters=0,
            seqlen=2,
            nsamples=2,
            dataset=self.llm_dataloader,
            layer_config=layer_config,
        )
        compressed_model, _ = autoround.quantize()
        assert hasattr(compressed_model.model.layers[1].mlp.experts[0].gate_proj.orig_layer, "act_max")

    def test_nvfp4_moe_actmax_ar(self):
        model_name = "/tf_dataset/auto_round/models/deepseek-ai/DeepSeek-V2-Lite"
        layer_config = {
            "q_proj": {"bits": 16, "act_bits": 16},
            "mlp.shared_experts": {"bits": 16, "act_bits": 16},
            "experts.*2": {"bits": 16, "act_bits": 16},
            "experts.*5": {"bits": 16, "act_bits": 16},
        }
        scheme = "nvfp4"
        autoround = AutoRound(
            model_name,
            scheme=scheme,
            iters=1,
            seqlen=2,
            nsamples=2,
            dataset=self.llm_dataloader,
            layer_config=layer_config,
        )
        autoround.quantize_and_save(output_dir=self.save_dir, inplace=True, format="auto_round")

    def test_mxfp4_llmcompressor_format(self):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        from transformers import AutoConfig

        scheme = "MXFP4"
        layer_config = {}
        fp_layers_str = "k_proj"
        from auto_round.utils import get_fp_layer_names

        not_quantize_layer_names = get_fp_layer_names(model, fp_layers_str)
        for name in not_quantize_layer_names:
            layer_config[name] = {"bits": 16, "act_bits": 16, "data_type": "float"}
        autoround = AutoRound(
            model_name,
            scheme=scheme,
            iters=2,
            seqlen=2,
            layer_config=layer_config,
            dataset=self.llm_dataloader,
        )
        quantized_model_path = self.save_dir
        autoround.quantize()
        compressed_model = autoround.save_quantized(
            output_dir=quantized_model_path, inplace=True, format="llm_compressor"
        )
        tmp_layer = compressed_model.model.decoder.layers[3].self_attn.q_proj
        skip_layer = compressed_model.model.decoder.layers[3].self_attn.k_proj
        assert (
            hasattr(tmp_layer, "weight_scale")
            and hasattr(tmp_layer, "weight_packed")
            and tmp_layer.weight_scale.dtype is torch.uint8
            and tmp_layer.weight_scale.shape[0] == 768
        ), "Illegal MXFP4 packing name or data_type or shape"
        assert not hasattr(skip_layer, "weight_scale") and not hasattr(  ## check skipped layers
            skip_layer, "weight_packed"
        ), "Illegal MXFP4 quantization for fp_layers"
        quantization_config = AutoConfig.from_pretrained(
            quantized_model_path, trust_remote_code=True
        ).quantization_config
        assert (
            quantization_config["format"] == "float-quantized"
            and quantization_config["config_groups"]["group_0"]["weights"]["is_mx"] is True
            and quantization_config["config_groups"]["group_0"]["weights"]["num_bits"] == 4
        ), f"Invalid MXFP4 quantization configuration: {quantization_config}"

        shutil.rmtree("./saved", ignore_errors=True)

    def test_rtn_mxfp4_llmcompressor_format(self):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        from transformers import AutoConfig

        scheme = "MXFP4"
        layer_config = {}
        fp_layers_str = "k_proj"
        from auto_round.utils import get_fp_layer_names

        not_quantize_layer_names = get_fp_layer_names(model, fp_layers_str)
        for name in not_quantize_layer_names:
            layer_config[name] = {"bits": 16, "act_bits": 16, "data_type": "float"}
        autoround = AutoRound(
            model_name,
            scheme=scheme,
            iters=0,
            seqlen=2,
            layer_config=layer_config,
            dataset=self.llm_dataloader,
        )
        quantized_model_path = self.save_dir
        autoround.quantize()
        compressed_model = autoround.save_quantized(
            output_dir=quantized_model_path, inplace=True, format="llm_compressor"
        )
        tmp_layer = compressed_model.model.decoder.layers[3].self_attn.q_proj
        skip_layer = compressed_model.model.decoder.layers[3].self_attn.k_proj
        assert (
            hasattr(tmp_layer, "weight_scale")
            and hasattr(tmp_layer, "weight_packed")
            and tmp_layer.weight_scale.dtype is torch.uint8
            and tmp_layer.weight_scale.shape[0] == 768
        ), "Illegal MXFP4 packing name or data_type or shape"
        assert not hasattr(skip_layer, "weight_scale") and not hasattr(  ## check skipped layers
            skip_layer, "weight_packed"
        ), "Illegal MXFP4 quantization for fp_layers"
        quantization_config = AutoConfig.from_pretrained(
            quantized_model_path, trust_remote_code=True
        ).quantization_config
        assert (
            quantization_config["format"] == "float-quantized"
            and quantization_config["config_groups"]["group_0"]["weights"]["is_mx"] is True
            and quantization_config["config_groups"]["group_0"]["weights"]["num_bits"] == 4
        ), f"Invalid MXFP4 quantization configuration: {quantization_config}"
        shutil.rmtree("./saved", ignore_errors=True)

    def test_mxfp8_llmcompressor_format(self):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        from transformers import AutoConfig

        scheme = "MXFP8"
        autoround = AutoRound(
            model_name,
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
            and hasattr(tmp_layer, "weight")
            and tmp_layer.weight.dtype is torch.float8_e4m3fn
            and tmp_layer.weight_scale.dtype is torch.uint8
            and tmp_layer.weight_scale.shape[0] == 768
        ), "Illegal MXFP8 packing name or data_type or shape"
        quantization_config = AutoConfig.from_pretrained(
            quantized_model_path, trust_remote_code=True
        ).quantization_config
        assert (
            quantization_config["format"] == "float-quantized"
            and quantization_config["config_groups"]["group_0"]["weights"]["is_mx"] is True
            and quantization_config["config_groups"]["group_0"]["weights"]["num_bits"] == 8
        ), f"Invalid MXFP8 quantization configuration: {quantization_config}"
        folder_size_gb = _get_folder_size(quantized_model_path)
        # Original opt-125m is < 0.5GB -> quantized mxfp8 model should be smaller but not empty
        assert (
            0.15 < folder_size_gb < 0.2
        ), f"Quantized model folder size {folder_size_gb:.2f} GB is outside the expected range (0.1~0.2 GB)"
        shutil.rmtree("./saved", ignore_errors=True)

    def test_nvfp4_llmcompressor_format(self):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        from transformers import AutoConfig

        scheme = "NVFP4"
        autoround = AutoRound(
            model_name,
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
        folder_size_gb = _get_folder_size(quantized_model_path)
        # Original opt-125m is < 0.5GB -> quantized nvfp4 model should be smaller but not empty
        assert (
            0.1 < folder_size_gb < 0.15
        ), f"Quantized model folder size {folder_size_gb:.2f} GB is outside the expected range (0.1~0.15 GB)"
        shutil.rmtree("./saved", ignore_errors=True)

    def test_nvfp4_autoround_format(self):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        from transformers import AutoConfig

        scheme = "NVFP4"
        autoround = AutoRound(
            model_name,
            scheme="NVFP4",
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        quantized_model_path = self.save_dir
        compressed_model, _ = autoround.quantize_and_save(output_dir=quantized_model_path, format="auto_round")
        tmp_layer = compressed_model.model.decoder.layers[3].self_attn.q_proj
        assert (
            hasattr(tmp_layer, "weight_scale")
            and hasattr(tmp_layer, "weight_global_scale")
            and hasattr(tmp_layer, "input_global_scale")
            and tmp_layer.weight_packed.dtype is torch.uint8
            and tmp_layer.weight_scale.dtype is torch.float8_e4m3fn
            and tmp_layer.weight_scale.shape[0] == 768
        ), "Illegal NVFP4 packing name or data_type or shape"
        shutil.rmtree("./saved", ignore_errors=True)

    def test_nvfp4_autoround_save_quantized(self):
        model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        from transformers import AutoConfig

        scheme = "NVFP4"
        autoround = AutoRound(
            model_name,
            scheme="NVFP4",
            iters=2,
            seqlen=2,
            dataset=self.llm_dataloader,
        )
        quantized_model_path = self.save_dir
        autoround.quantize()
        compressed_model = autoround.save_quantized(output_dir=quantized_model_path, format="auto_round")
        tmp_layer = compressed_model.model.decoder.layers[3].self_attn.q_proj
        assert (
            hasattr(tmp_layer, "weight_scale")
            and hasattr(tmp_layer, "weight_global_scale")
            and hasattr(tmp_layer, "input_global_scale")
            and tmp_layer.weight_packed.dtype is torch.uint8
            and tmp_layer.weight_scale.dtype is torch.float8_e4m3fn
            and tmp_layer.weight_scale.shape[0] == 768
        ), "Illegal NVFP4 packing name or data_type or shape"
        shutil.rmtree("./saved", ignore_errors=True)

    def test_qwen_moe_quant_infer(self):
        model_name = "/tf_dataset/auto_round/models/Qwen/Qwen1.5-MoE-A2.7B"
        layer_config = {
            "layers\.(?:[3-9]|1[0-9]|2[0-3])": {"bits": 16, "act_bits": 16},
        }
        scheme = "nvfp4"
        autoround = AutoRound(
            model_name,
            scheme=scheme,
            iters=1,
            seqlen=2,
            nsamples=2,
            dataset=self.llm_dataloader,
            layer_config=layer_config,
        )
        quantized_model_path = self.save_dir
        autoround.quantize_and_save(output_dir=quantized_model_path, inplace=True, format="auto_round")
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, torch_dtype="auto", device_map="cpu")
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        from auto_round.eval.evaluation import simple_evaluate_user_model

        result = simple_evaluate_user_model(model, tokenizer, batch_size=16, tasks="piqa", limit=10)
        print(result["results"]["piqa"]["acc,none"])
        self.assertGreater(result["results"]["piqa"]["acc,none"], 0.60)
        shutil.rmtree(quantized_model_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
