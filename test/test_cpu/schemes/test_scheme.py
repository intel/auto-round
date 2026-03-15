import os
import shutil

import transformers

from auto_round import AutoRound
from auto_round.schemes import QuantizationScheme

from ...helpers import get_model_path, get_tiny_model, opt_name_or_path, qwen_name_or_path, save_tiny_model


class TestAutoRound:
    @classmethod
    def setup_class(self):
        self.save_folder = "./saved"

    @classmethod
    def teardown_class(self):
        shutil.rmtree(self.save_folder, ignore_errors=True)
        shutil.rmtree("runs", ignore_errors=True)

    def test_gguf(self, tiny_qwen_model_path, dataloader):
        ar = AutoRound(
            tiny_qwen_model_path,
            scheme="W2A16",
            nsamples=1,
            iters=1,
            seqlen=2,
            dataset=dataloader,
        )
        ar.quantize_and_save(self.save_folder, format="gguf:q4_k_m")
        assert ar.bits == 4
        shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_w4a16(self, tiny_opt_model_path, dataloader):
        ar = AutoRound(tiny_opt_model_path, scheme="W4A16", nsamples=1, iters=1, seqlen=2, dataset=dataloader)
        assert ar.bits == 4

    def test_w2a16_rtn(self, tiny_opt_model_path, dataloader):
        ar = AutoRound(tiny_opt_model_path, scheme="W2A16", nsamples=1, iters=0, seqlen=2, dataset=dataloader)
        assert ar.bits == 2

    def test_w4a16_mixed(self, tiny_qwen_moe_model_path, dataloader):

        layer_config = {
            "model.layers.0.self_attn.k_proj": {"bits": 16},
        }
        ar = AutoRound(
            tiny_qwen_moe_model_path,
            scheme="W4A16_MIXED",
            nsamples=1,
            iters=0,
            seqlen=2,
            dataset=dataloader,
            low_cpu_mem_usage=False,
            layer_config=layer_config,
        )
        ar.quantize_and_save(self.save_folder)
        assert ar.bits == 4
        assert ar.model.model.layers[0].self_attn.q_proj.bits == 8
        assert ar.model.model.layers[0].self_attn.k_proj.bits == 16
        assert ar.model.model.layers[0].mlp.experts[0].up_proj.bits == 4
        # assert ar.model.model.layers[0].mlp.shared_expert.gate_proj.bits == 8 # gate has been added to ignore_layers
        model = transformers.AutoModelForCausalLM.from_pretrained(self.save_folder, trust_remote_code=True)
        assert model is not None, "Model loading failed after quantization with W4A16_MIXED scheme on MoE"

        shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_w4a16_mixed_mllm(self, tiny_qwen_2_5_vl_model_path, dataloader):

        ar = AutoRound(
            tiny_qwen_2_5_vl_model_path,
            scheme="W4A16_MIXED",
            nsamples=1,
            batch_size=1,
            iters=0,
            seqlen=2,
            dataset=dataloader,
            low_cpu_mem_usage=False,
        )
        ar.quantize_and_save(self.save_folder)
        model = transformers.AutoModelForCausalLM.from_pretrained(self.save_folder, trust_remote_code=True)
        assert model is not None, "Model loading failed after quantization with W4A16_MIXED scheme on MLLM"
        assert ar.bits == 4
        assert ar.model.language_model.layers[0].self_attn.q_proj.bits == 16
        assert ar.model.visual.blocks[0].attn.qkv.bits == 16
        shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_mxfp4(self, tiny_opt_model_path, dataloader):
        ar = AutoRound(tiny_opt_model_path, scheme="MXFP4", nsamples=1, iters=1, seqlen=2, dataset=dataloader)
        assert ar.bits == 4
        assert ar.act_bits == 4
        assert ar.data_type == "mx_fp"
        assert ar.act_data_type == "mx_fp"

    def test_mxfp4_rceil(self, tiny_opt_model_path):
        ar = AutoRound(tiny_opt_model_path, scheme="MXFP4_RCEIL", nsamples=1, iters=1)
        assert ar.bits == 4
        assert ar.act_bits == 4
        assert ar.data_type == "mx_fp"
        assert ar.act_data_type == "mx_fp_rceil"
        ar.quantize_and_save()
        model = transformers.AutoModelForCausalLM.from_pretrained("tmp_autoround", trust_remote_code=True)
        assert model is not None, "Model loading failed after quantization with MXFP4 scheme"
        shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_vlm(self, tiny_qwen_vl_model_path):
        from auto_round import AutoRoundMLLM

        ar = AutoRoundMLLM(tiny_qwen_vl_model_path, scheme="W2A16", nsamples=1, iters=1, seqlen=2)
        assert ar.bits == 2
        assert ar.act_bits == 16

    def test_nvfp4(self, tiny_opt_model_path, dataloader):
        ar = AutoRound(tiny_opt_model_path, scheme="NVFP4", nsamples=1, iters=1, seqlen=2, dataset=dataloader)
        assert ar.bits == 4
        assert ar.act_bits == 4
        assert ar.data_type == "nv_fp"
        assert ar.act_data_type == "nv_fp4_with_static_gs"
        ar.quantize_and_save(self.save_folder)
        model = transformers.AutoModelForCausalLM.from_pretrained(self.save_folder, trust_remote_code=True)
        assert model is not None, "Model loading failed after quantization with NVFP4 scheme"
        shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_all_scheme(self, tiny_opt_model_path, tiny_qwen_model_path, dataloader):
        import copy

        preset_schemes = ["W8A16", "MXFP8", "FPW8A16", "FP8_STATIC", "GGUF:Q2_K_S", "GGUF:Q4_K_M"]
        for scheme in preset_schemes:
            model_name = tiny_opt_model_path
            if "gguf" in scheme.lower():
                model_name = tiny_qwen_model_path
            print(f"scheme={scheme}")
            ar = AutoRound(model_name, scheme=scheme, nsamples=1, iters=1, seqlen=2, dataset=dataloader)
            ar.quantize_and_save(self.save_folder)
            shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_scheme_in_layer_config(self, dataloader):
        model = get_tiny_model(opt_name_or_path, num_layers=5)
        tokenizer = transformers.AutoTokenizer.from_pretrained(opt_name_or_path, trust_remote_code=True)
        layer_config = {
            "model.decoder.layers.2.self_attn": {"bits": 2},
            "model.decoder.layers.3.self_attn.v_proj": "W8A16",
            "model.decoder.layers.4.self_attn.k_proj": QuantizationScheme.from_dict({"group_size": 64}),
        }
        ar = AutoRound(
            model,
            tokenizer,
            scheme="W3A16",
            nsamples=1,
            iters=1,
            layer_config=layer_config,
            seqlen=2,
            dataset=dataloader,
        )

        ar.quantize_and_save(self.save_folder)
        for n, m in ar.model.named_modules():
            if n == "model.decoder.layers.2.self_attn.q_proj":
                assert m.bits == 2
            if n == "model.decoder.layers.2.self_attn.k_proj":
                assert m.bits == 2
            if n == "model.decoder.layers.3.self_attn.v_proj":
                assert m.bits == 8
            if n == "model.decoder.layers.4.self_attn.k_proj":
                assert m.group_size == 64

    def test_parse_available_devices(self):
        from auto_round.utils.device import parse_available_devices

        device_list = parse_available_devices("auto")
        assert len(device_list) == 1 and "cpu" in device_list
        device_list = parse_available_devices("a:cuda:0,b:cuda:1,c:cpu")
        assert len(device_list) == 3
        assert device_list == ["cuda:0", "cuda:1", "cpu"]
        device_list = parse_available_devices("0,1")
        assert len(device_list) == 1 and "cpu" in device_list

    def test_set_scheme(self, tiny_qwen_model_path):
        ar = AutoRound(
            tiny_qwen_model_path,
            scheme="gguf:q2_k_s",
            data_type="fp",
            nsamples=1,
            disable_opt_rtn=True,
            iters=0,
            seqlen=2,
        )
        ar.quantize_and_save(self.save_folder)

        from auto_round.schemes import QuantizationScheme

        qs = QuantizationScheme.from_dict({"bits": 4, "group_size": 64})
        ar = AutoRound(
            tiny_qwen_model_path,
            scheme=qs,
            bits=2,
            data_type="int_asym_dq",
            nsamples=1,
            iters=0,
            disable_opt_rtn=True,
            seqlen=2,
        )
        ar.quantize_and_save(self.save_folder)

    def test_fp8_static(self, tiny_opt_model_path):
        ar = AutoRound(tiny_opt_model_path, scheme="FP8_STATIC", nsamples=1, iters=1)
        assert ar.bits == 8
        assert ar.act_bits == 8
        assert ar.data_type == "fp"
        assert ar.act_data_type == "fp"
        assert ar.group_size == -1
        assert ar.act_dynamic is False
        ar.quantize_and_save()
        model = transformers.AutoModelForCausalLM.from_pretrained("tmp_autoround", trust_remote_code=True)
        assert model is not None, "Model loading failed after quantization with FP8_STATIC scheme"

    def test_fp8_static_rtn(self, tiny_opt_model_path):
        ar = AutoRound(tiny_opt_model_path, scheme="FP8_STATIC", nsamples=1, iters=0, disable_opt_rtn=True)
        assert ar.bits == 8
        assert ar.act_bits == 8
        assert ar.data_type == "fp"
        assert ar.act_data_type == "fp"
        assert ar.group_size == -1
        assert ar.act_dynamic is False
        ar.quantize_and_save(self.save_folder)
        model = transformers.AutoModelForCausalLM.from_pretrained(self.save_folder, trust_remote_code=True)
        assert model is not None, "Model loading failed after quantization with FP8_STATIC scheme"
        shutil.rmtree(self.save_folder, ignore_errors=True)

    def test_q2k_mixed(self):
        model_path = "/data0/MiroThinker-v1.5-30B"
        saved_tiny_model_path = save_tiny_model(
            model_path,
            "./tmp/tiny_qwen_model_path",
            num_layers=3,
            is_mllm=False,
        )
        autoround = AutoRound(
            saved_tiny_model_path,
            iters=0,
            nsamples=1,
            seqlen=16,
            disable_opt_rtn=True,
        )
        quantized_model_path = "./saved"
        autoround.quantize_and_save(output_dir=quantized_model_path, format="gguf:q2_k_mixed")
        gguf_file = os.listdir(quantized_model_path)[0]
        file_size = os.path.getsize(os.path.join(quantized_model_path, gguf_file)) / 1024**2
        assert abs(file_size - 1236) < 5.0
        from gguf.gguf_reader import GGUFReader

        gguf_model = GGUFReader(os.path.join(quantized_model_path, gguf_file))
        assert gguf_model.get_tensor(2).name == "blk.0.attn_v.weight"
        assert gguf_model.get_tensor(2).tensor_type.name == "Q4_K"
        assert gguf_model.get_tensor(9).name == "blk.0.ffn_up_exps.weight"
        assert gguf_model.get_tensor(9).tensor_type.name == "Q2_K"

        shutil.rmtree(saved_tiny_model_path, ignore_errors=True)
        shutil.rmtree(quantized_model_path, ignore_errors=True)
