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
        self.model_name = "/tf_dataset/auto_round/models/facebook/opt-125m"
        self.save_dir = "./saved"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
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
            "experts.*2": {"bits": 16, "act_bits": 16},
            "experts.*5": {"bits": 16, "act_bits": 16},
            "lm_head": {"bits": 4, "act_bits": 4},  ## test lm_head quantization and exporting
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
        lm_head = compressed_model.lm_head
        assert hasattr(lm_head, "orig_layer") and hasattr(
            lm_head.orig_layer, "act_max"
        ), "Illegal NVFP4 quantization for lm_head layer"
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_nvfp4_moe_actmax_ar(self):
        model_name = "/tf_dataset/auto_round/models/deepseek-ai/DeepSeek-V2-Lite"
        layer_config = {
            "q_proj": {"bits": 16, "act_bits": 16},
            "mlp.shared_experts": {"bits": 16, "act_bits": 16},
            "experts.*2": {"bits": 16, "act_bits": 16},
            "experts.*5": {"bits": 16, "act_bits": 16},
            "lm_head": {"bits": 4, "act_bits": 4},  ## test lm_head quantization and exporting
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
        compressed_model, _ = autoround.quantize_and_save(output_dir=self.save_dir, inplace=True, format="auto_round")
        lm_head = compressed_model.lm_head
        assert (
            hasattr(lm_head, "weight_scale")
            and hasattr(lm_head, "weight_global_scale")
            and hasattr(lm_head, "input_global_scale")
            and lm_head.weight_packed.dtype is torch.uint8
            and lm_head.weight_scale.dtype is torch.float8_e4m3fn
        ), "Illegal NVFP4 packing for lm_head layer"
        quantized_model_path = self.save_dir
        model = AutoModelForCausalLM.from_pretrained(quantized_model_path, torch_dtype="auto", device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(quantized_model_path)
        from auto_round.eval.evaluation import simple_evaluate_user_model

        result = simple_evaluate_user_model(model, tokenizer, batch_size=4, tasks="piqa", limit=4)
        print(result["results"]["piqa"]["acc,none"])
        self.assertGreater(result["results"]["piqa"]["acc,none"], 0.7)
        shutil.rmtree(self.save_dir, ignore_errors=True)


    def test_mxfp4_moe_ar(self):
        model_name = "/tf_dataset/auto_round/models/deepseek-ai/DeepSeek-V2-Lite"
        layer_config = {
            "q_proj": {"bits": 16, "act_bits": 16, "data_type": "float"},
            "mlp.shared_experts": {"bits": 16, "act_bits": 16, "data_type": "float"},
            "experts.*2": {"bits": 16, "act_bits": 16, "data_type": "float"},
            "experts.*5": {"bits": 16, "act_bits": 16, "data_type": "float"},
            "lm_head": {"bits": 4, "act_bits": 4},  ## test lm_head quantization and exporting
        }
        scheme = "mxfp4"
        autoround = AutoRound(
            model_name,
            scheme=scheme,
            iters=1,
            seqlen=2,
            nsamples=2,
            dataset=self.llm_dataloader,
            layer_config=layer_config,
        )
        compressed_model, _ = autoround.quantize_and_save(output_dir=self.save_dir, inplace=True, format="auto_round")
        lm_head = compressed_model.lm_head
        assert (
            hasattr(lm_head, "weight_scale")
            and hasattr(lm_head, "weight_packed")
            and lm_head.weight_scale.dtype is torch.uint8
        ), "Illegal MXFP4 packing for lm_head layer"
        shutil.rmtree(self.save_dir, ignore_errors=True)

    def test_mxfp4_llmcompressor_format(self):
        model_name = self.model_name
        from transformers import AutoConfig

        scheme = "MXFP4"
        layer_config = {"k_proj": {"bits": 16, "act_bits": 16, "data_type": "float"}}
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
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_rtn_mxfp4_llmcompressor_format(self):
        model_name = self.model_name
        from transformers import AutoConfig

        scheme = "MXFP4"
        layer_config = {"k_proj": {"bits": 16, "act_bits": 16, "data_type": "float"}}
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
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_mxfp8_llmcompressor_format(self):
        model_name = self.model_name
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
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_nvfp4_llmcompressor_format(self):
        model_name = self.model_name
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
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_nvfp4_autoround_format(self):
        model_name = self.model_name
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
        shutil.rmtree(quantized_model_path, ignore_errors=True)

    def test_nvfp4_autoround_save_quantized(self):
        model_name = self.model_name
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
        shutil.rmtree(quantized_model_path, ignore_errors=True)

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

    @parameterized.expand(
        [
            # scheme,  static_kv_dtype, static_attention_dtype
            ("MXFP4", None, "fp8"),
            ("MXFP4", "fp8", None),
            ("MXFP8", None, "fp8"),
            ("MXFP8", "fp8", None),
            ("NVFP4", None, "fp8"),
            ("NVFP4", "fp8", None),
        ]
    )
    def test_fp8_kv_attn(self, scheme, static_kv_dtype, static_attention_dtype):
        model_name = self.model_name
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from transformers.models.opt.modeling_opt import OPTForCausalLM

        config = AutoConfig.from_pretrained(model_name)
        config.num_hidden_layers = 1
        model = OPTForCausalLM(config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        autoround = AutoRound(
            model,
            tokenizer,
            scheme=scheme,
            iters=0,
            seqlen=2,
            dataset=self.llm_dataloader,
            static_kv_dtype=static_kv_dtype,
            static_attention_dtype=static_attention_dtype,
        )

        quantized_model_path = self.save_dir
        compressed_model, _ = autoround.quantize_and_save(
            output_dir=quantized_model_path,
            format="auto_round",
        )

        attn = compressed_model.model.decoder.layers[0].self_attn
        q_proj = attn.q_proj

        # weight_scale should exist for all quantized schemes
        assert hasattr(q_proj, "weight_scale"), f"Missing weight_scale in q_proj for scheme={scheme}"
        if static_kv_dtype == "fp8":
            assert (
                compressed_model.config.quantization_config["static_kv_dtype"] == "fp8"
            ), f"Invalid static_kv_dtype in config for scheme={scheme}, static_kv_dtype={static_kv_dtype}"

        # Only when static_kv_dtype / static_attention_dtype are fp8 do we expect FP8 KV scales
        if static_kv_dtype == "fp8" or static_attention_dtype == "fp8":
            assert attn.k_scale is not None and attn.v_scale is not None, (
                f"Missing k_scale/v_scale in attention for scheme={scheme}, "
                f"static_kv_dtype={static_kv_dtype}, static_attention_dtype={static_attention_dtype}"
            )

        if static_attention_dtype == "fp8":
            assert (
                compressed_model.config.quantization_config["static_attention_dtype"] == "fp8"
            ), f"Invalid static_attention_dtype in config for scheme={scheme}, static_attention_dtype={static_attention_dtype}"
            assert (
                getattr(attn, "q_scale", None) is not None
            ), f"Missing q_scale in attention for scheme={scheme}, static_attention_dtype={static_attention_dtype}"
        shutil.rmtree(quantized_model_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
